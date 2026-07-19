"""Dynamiqs backend — JAX-native solver for differentiable, vmappable simulation.

This backend wraps `dynamiqs <https://github.com/dynamiqs/dynamiqs>`_ (which
builds on `JAX <https://github.com/google/jax>`_) to provide:

* **Full JAX traceability.** Every operator, state, and signal program
  evaluates inside JAX, so gradients flow through frame resolution, carrier
  phases, envelope parameters, crosstalk mixing, and dissipator strengths.
* **Native batched solves.** A typed :class:`~quchip.engine.ir.SolveBatch`
  is stacked along a batch axis and integrated with a single ``dq.sesolve``
  / ``dq.mesolve`` call (``vmap`` under the hood) in :meth:`solve_batch`.
  Structurally heterogeneous batches fail loudly — no silent sequential
  fallback — so the caller can regroup them via
  :func:`quchip.engine.solve_many`.

References
----------
* Guilmin et al. — *dynamiqs: an open-source Python library for GPU-accelerated
  and differentiable simulation of quantum systems* (2024)
* Bradbury et al. — *JAX: composable transformations of Python+NumPy programs*
  (2018)
* Lindblad — Commun. Math. Phys. 48, 119 (1976)
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Sequence

import dynamiqs as dq
import equinox as eqx
import numpy as np
from dynamiqs.qarrays.dense_qarray import DenseQArray
from dynamiqs.qarrays.sparsedia_qarray import SparseDIAQArray

# x64 is enabled at the package boundary in ``quchip/__init__.py``.

import jax.numpy as jnp  # noqa: E402
import jax.tree_util as jtu  # noqa: E402

from quchip.backend._dims import (  # noqa: E402
    compute_two_body_permutation,
    default_solver_steps,
    normalize_dims_from_list,
    validate_two_body_indices,
)
from quchip.backend.containers import (  # noqa: E402
    EigensystemData,
    PreparedHamiltonian,
    SolverResult,
    VmappedBatch,
)
from quchip.backend.protocol import Backend, Operator, State  # noqa: E402
from quchip.engine.ir import ScalarModulation, evaluate_signal_program  # noqa: E402


class _SignalCallable(eqx.Module):
    """Equinox wrapper: evaluates a signal-program AST under JAX tracing.

    Registering via ``eqx.Module`` makes the signal a pytree, so dynamiqs
    can vmap across batched parameters inside the carrier/envelope tree.
    """

    signal: Any

    def __call__(self, t: float) -> Any:
        return jnp.asarray(evaluate_signal_program(self.signal, t, xp=jnp))


class _BatchedSignalQArrayCallable(eqx.Module):
    """JAX-traceable ``signal(t) * operator`` with scalar/batch broadcasting.

    When ``signal`` is a scalar the product is a single qarray; when ``signal``
    carries a leading batch axis the product broadcasts against the operator's
    trailing two matrix axes, producing a batched operator ready for vmap.
    """

    signal: Any
    operator: Any

    def __call__(self, t: float) -> Any:
        prefactor = jnp.asarray(evaluate_signal_program(self.signal, t, xp=jnp), dtype=jnp.complex128)
        if prefactor.ndim == 0:
            return prefactor * self.operator
        return prefactor[..., None, None] * self.operator


class DynamiqsBackend(Backend):
    """Concrete backend backed by dynamiqs + JAX. Operators/states are ``QArray``.

    Example
    -------
    >>> from quchip.backend.dynamiqs import DynamiqsBackend
    >>> backend = DynamiqsBackend()
    >>> a = backend.destroy(3)
    >>> n = backend.matmul(backend.dag(a), a)
    >>> float(backend.to_array(n)[2, 2].real)  # doctest: +SKIP
    2.0
    """

    # ------------------------------------------------------------------
    # Scalar / array surface
    # ------------------------------------------------------------------

    @property
    def array_module(self) -> Any:
        """Return the array module for JAX-traceable numeric code (``jax.numpy``)."""
        return jnp

    def to_array(self, op: Operator) -> Any:
        """Return a dense ``jax.numpy`` array for *op*."""
        if hasattr(op, "to_jax"):
            return jnp.asarray(op.to_jax(), dtype=jnp.complex128)
        return jnp.asarray(op, dtype=jnp.complex128)

    def overlap(self, a: State, b: State) -> complex:
        """Return the complex inner product ⟨a|b⟩ for two kets, via ``dynamiqs.braket``.

        ``dynamiqs.overlap`` returns the real-valued :math:`|\\langle a|b
        \\rangle|^2` (or a density-matrix fidelity), not the protocol's
        phase-sensitive complex inner product — ``dq.braket`` is the
        matching primitive.
        """
        return dq.braket(a, b)

    def norm(self, state_or_op: State | Operator) -> Any:
        """Return the norm of a state or operator via ``dynamiqs.norm``.

        Returns the backend-native (possibly traced) scalar rather than a
        concrete ``float`` — the protocol allows a 0-d array here so a
        traced amplitude stays traceable under ``jax.jit``/``grad``.
        """
        return dq.norm(state_or_op)

    def trace(self, op: Operator) -> complex:
        """Return the scalar trace ``Tr(op)`` via ``dynamiqs.trace``."""
        return dq.trace(op)

    # ------------------------------------------------------------------
    # Operator / state factories
    # ------------------------------------------------------------------

    def destroy(self, n: int) -> Operator:
        """Return the annihilation operator for an *n*-level Fock space (``dynamiqs.destroy``)."""
        return dq.destroy(n)

    def create(self, n: int) -> Operator:
        """Return the creation operator for an *n*-level Fock space (``dynamiqs.create``)."""
        return dq.create(n)

    def number(self, n: int) -> Operator:
        """Return the number operator for an *n*-level Fock space (``dynamiqs.number``)."""
        return dq.number(n)

    def identity(self, n: int) -> Operator:
        """Return the identity operator for an *n*-level space (``dynamiqs.eye``)."""
        return dq.eye(n)

    def diag(self, values: Any, dims: list[list[int]] | None = None) -> Operator:
        """Build a backend-native sparse-DIA diagonal operator from main-diagonal *values*.

        Overrides the protocol default to keep the operator in sparse-DIA
        layout even when *values* is a JAX tracer — the offsets ``(0,)`` are
        concrete, so the SparseDIAQArray constructor accepts the traced
        values directly. Without this override, dynamiqs's
        ``from_canonical_operator`` densifies any traced DIA payload, which
        forces a dense ``H₀`` for circuit-style devices and triggers the
        sparse→dense warning during static-Hamiltonian assembly.
        """
        v = jnp.asarray(values, dtype=jnp.complex128).reshape(-1)
        n = v.shape[0]
        dim_tuple = self._coerce_dims(dims, (n, n))
        if dim_tuple is None:
            dim_tuple = (n,)
        return SparseDIAQArray(dim_tuple, False, (0,), v[None, :])

    def from_array(self, data: Any, dims: list[list[int]] | None = None) -> Operator:
        """Construct a native ``QArray`` from a dense matrix (row/col or flat *dims*)."""
        array = jnp.asarray(data, dtype=jnp.complex128)
        dims_tuple = self._coerce_dims(dims, array.shape)
        if dims_tuple is None:
            return dq.asqarray(array)
        return dq.asqarray(array, dims=dims_tuple)

    def to_canonical_operator(self, op: Operator) -> Any:
        """Serialize a ``QArray`` into the backend-agnostic canonical IR."""
        from quchip.engine.ir import CanonicalOperator

        # getattr's default must stay lazy: to_array densifies a SparseDIA
        # operator (dim² allocation) just to read a shape that is discarded
        # whenever `.dims` exists — which it does for every dynamiqs qarray.
        dims_attr = getattr(op, "dims", None)
        dims = tuple(dims_attr) if dims_attr is not None else (self.to_array(op).shape[0],)
        labels = tuple(str(i) for i in range(len(dims)))
        if isinstance(op, SparseDIAQArray):
            return CanonicalOperator.from_dia(
                jnp.asarray(op.diags, dtype=jnp.complex128),
                jnp.asarray(op.offsets, dtype=int),
                shape=op.shape, dims=dims, basis="fock", subsystem_labels=labels,
            )
        if isinstance(op, DenseQArray):
            return CanonicalOperator.from_dense(
                jnp.asarray(op.data, dtype=jnp.complex128),
                dims=dims, basis="fock", subsystem_labels=labels,
            )

        arr = jnp.asarray(op, dtype=jnp.complex128)
        return CanonicalOperator.from_dense(
            arr, dims=(arr.shape[0],), basis="fock", subsystem_labels=("0",),
        )

    def from_canonical_operator(self, canonical: Any) -> Operator:
        """Reconstruct a ``QArray`` from the canonical IR payload (dense/DIA/COO)."""
        dims = tuple(canonical.dims)
        if canonical.layout == "dense":
            return dq.asqarray(jnp.asarray(canonical.values, dtype=jnp.complex128), dims=dims)
        if canonical.layout == "dia":
            from quchip.engine.bands import _canonical_has_nonconcrete_payload, canonical_to_dense_array

            if _canonical_has_nonconcrete_payload(canonical):
                # Non-concrete (traced) offsets/values cannot be inspected by
                # SparseDIAQArray's constructor; densify to keep traceability.
                dense = canonical_to_dense_array(canonical)
                return dq.asqarray(jnp.asarray(dense, dtype=jnp.complex128), dims=dims)
            return SparseDIAQArray(
                dims,
                False,
                tuple(int(x) for x in canonical.offsets),
                jnp.asarray(canonical.values, dtype=jnp.complex128),
            )
        from quchip.engine.bands import canonical_to_coo

        rows, cols, values = canonical_to_coo(canonical)
        return self._sparse_qarray_from_coo(rows, cols, values, dims)

    def coerce_operator(self, op: Operator) -> Operator:
        """Pass *op* through unchanged (trace-safe): native operators are (JAX) arrays already."""
        return self.to_array(op)

    def dag(self, op: Operator) -> Operator:
        """Return the Hermitian conjugate ``op†`` via ``dynamiqs.dag``."""
        return dq.dag(op)

    def eigenenergies(self, op: Operator) -> Any:
        """Return the ascending eigenvalues of a Hermitian operator (``jax.numpy.linalg.eigvalsh``)."""
        return jnp.linalg.eigvalsh(self.to_array(op))

    def eigensystem_data(self, op: Operator) -> EigensystemData:
        """Return ascending eigenvalues, eigenvector matrix, and lazily built eigenstate kets."""
        dense = self.to_array(op)
        evals, evecs = jnp.linalg.eigh(dense)
        dims = getattr(op, "dims", (dense.shape[0],))

        # Defer per-column ket construction: the dressing / sweep hot path reads
        # only evals + evecs + labeling, so the D backend-ket allocations (a
        # second O(D**2) densification) are pure waste there. evecs stays a plain
        # jnp array (never routed through a Python list) so vmap/grad are intact.
        def build_states() -> list[Any]:
            return [dq.asqarray(evecs[:, idx:idx + 1], dims=dims) for idx in range(evecs.shape[1])]

        return EigensystemData(
            eigenvalues=evals,
            eigenvector_matrix=evecs,
            _states_builder=build_states,
        )

    def expect(self, op: Operator, state: State) -> complex:
        """Return the expectation value ⟨op⟩ for a ket or density matrix via ``dynamiqs.expect``."""
        return dq.expect(op, state)

    def ptrace(self, state: State, keep: int | list[int], dims: list[int]) -> State:
        """Reduce onto subsystem(s) *keep* via ``dynamiqs.ptrace`` (partial trace)."""
        keep_arg = tuple(keep) if isinstance(keep, list) else keep
        return dq.ptrace(state, keep_arg, dims=tuple(dims))

    # ------------------------------------------------------------------
    # Batched-over-time extraction (dq is batch-axis-native)
    # ------------------------------------------------------------------
    #
    # ``result.states`` is a single stacked ``QArray`` of shape ``(T, n, 1)``
    # (kets) or ``(T, n, n)`` (DMs). ``dq.expect`` / ``dq.ptrace`` /
    # ``dq.overlap`` broadcast over leading axes, so each extractor is one
    # batched call — no Python per-time loop, and the stacked ``QArray`` stays
    # a single jnp pytree (strictly more traceable than a list of tracers).

    def stack_states(self, states: Any) -> Any:
        """Pass the native stacked ``QArray`` through; restack a list if needed."""
        if hasattr(states, "shape") and not isinstance(states, (list, tuple)):
            return states
        return self._stack_state_batch(list(states), dims=getattr(states[0], "dims", None))

    def expect_over_time(self, op: Operator, stacked_states: Any) -> Any:
        """Return ⟨op⟩(t) at every save point via a single batched ``dynamiqs.expect`` call."""
        # Native dq.expect fast path; the generic protocol overlap/populations
        # extractors are already array-namespace-parameterized (jnp here), so
        # they need no dynamiqs override — and overlap stays the complex
        # ⟨target|ψ⟩ amplitude (dq.overlap would square it and drop the phase).
        return jnp.asarray(dq.expect(op, stacked_states))

    def ptrace_over_time(self, stacked_states: Any, keep: int | list[int], dims: list[int]) -> Any:
        """Reduce onto subsystem(s) *keep* at every save point (partial trace, batched ``dynamiqs.ptrace``)."""
        keep_arg = tuple(keep) if isinstance(keep, list) else keep
        reduced = dq.ptrace(stacked_states, keep_arg, dims=tuple(dims))
        return jnp.asarray(reduced.to_jax(), dtype=jnp.complex128)

    def tensor(self, *operators: Operator) -> Operator:
        """Return the tensor product of operators via ``dynamiqs.tensor`` (pass-through for one factor)."""
        if len(operators) == 1:
            return operators[0]
        return dq.tensor(*operators)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def embed_two_body(
        self,
        op_ab: Operator,
        index_a: int,
        index_b: int,
        dims: Sequence[int],
    ) -> Operator:
        """Embed a two-body operator on devices *index_a* ⊗ *index_b* into the full space.

        Keeps a sparse operator sparse; a dense operator is reordered, padded
        with identities, and permuted back to the original subsystem order.
        """
        canonical = self.to_canonical_operator(op_ab)
        if canonical.is_sparse:
            return self._embed_two_body_sparse(canonical, index_a, index_b, dims)

        op_dense, first_idx, second_idx = self._reorder_two_body_array(op_ab, index_a, index_b, dims)
        reorder, inverse_reorder = compute_two_body_permutation(first_idx, second_idx, dims)
        reordered_dims = [dims[idx] for idx in reorder]
        op_ordered = dq.asqarray(op_dense, dims=(dims[first_idx], dims[second_idx]))

        factors: list[Operator] = [op_ordered]
        factors.extend(dq.eye(dim) for dim in reordered_dims[2:])
        full_reordered = self.tensor(*factors)

        n_devices = len(dims)
        # Reshape to rank-2N, apply the inverse permutation to both the row
        # and column index groups so subsystem ordering matches the original.
        perm = tuple(inverse_reorder + [n_devices + idx for idx in inverse_reorder])
        dense = self.to_array(full_reordered).reshape(tuple(reordered_dims) + tuple(reordered_dims))
        # math.prod, not jnp: dims are static Python ints, and a reshape size
        # must stay concrete — jnp constants are tracers inside a jit trace.
        restored = jnp.transpose(dense, axes=perm).reshape(math.prod(dims), -1)
        return dq.asqarray(restored, dims=tuple(dims))

    def basis(self, n: int, k: int) -> State:
        """Return the Fock basis ket |k⟩ in an *n*-level space (``dynamiqs.basis``)."""
        return dq.basis(n, k)

    def tensor_states(self, *states: State) -> State:
        """Return the tensor product of states via ``dynamiqs.tensor`` (pass-through for one state)."""
        if len(states) == 1:
            return states[0]
        return dq.tensor(*states)

    def coherent(self, n: int, alpha: complex) -> State:
        """Return the coherent state |α⟩ truncated to *n* Fock levels (``dynamiqs.coherent``)."""
        return dq.coherent(n, alpha)

    def state_to_dm(self, state: State) -> State:
        """Return a density matrix; pass through if *state* is already one."""
        if not self.is_ket(state):
            return state
        return state @ dq.dag(state)

    def is_ket(self, state: State) -> bool:
        """Return whether *state* is a column-vector ket rather than a density matrix."""
        return len(state.shape) == 2 and state.shape[1] == 1

    # ------------------------------------------------------------------
    # Solver options / heuristics
    # ------------------------------------------------------------------

    def resolve_solver_options(
        self,
        options: dict[str, Any],
        *,
        metadata: dict[str, Any],
        tlist: Any,
    ) -> dict[str, Any]:
        """Normalize quchip aliases and fill in a ``max_steps`` budget.

        The quchip option aliases (``progress_bar`` → ``progress_meter``,
        ``nsteps`` → ``max_steps``) are mapped onto dynamiqs' canonical names
        by :meth:`_normalize_dq_options`. When ``max_steps`` is still unset,
        it is derived by :func:`~quchip.backend._dims.default_solver_steps`
        (see that function's docstring for the heuristic) — the same
        heuristic the QuTiP backend uses.

        ``max_steps`` is an abort ceiling on the integrator's total internal
        step count, not a bound on individual step size. Unlike the QuTiP
        backend, this backend does not consume ``metadata['max_step_ns']``:
        dynamiqs' deterministic adaptive methods (``Tsit5``, ``Dopri5``,
        ``Dopri8``, ``Kvaerno3``/``Kvaerno5``) expose no step-size-bound
        parameter analogous to QuTiP's ``max_step`` (verified against
        dynamiqs 0.3.3 — none of those methods accept a ``dtmax``-style
        argument). ``dq.method.Event`` does accept ``dtmax``, but it governs
        only the no-click evolution inside ``dq.jssesolve``'s jump-SSE
        integration, not the deterministic ``sesolve``/``mesolve`` path this
        backend uses; wrapping a deterministic solve in ``Event`` to borrow
        its ``dtmax`` would fail before integration even starts.

        Consequence: a finite-support pulse embedded in a long idle span can
        be silently skipped by dynamiqs' adaptive step selection, the same
        failure mode :meth:`quchip.backend.qutip.QuTiPBackend.resolve_solver_options`
        fixes via ``max_step``. The engine's ``max_step_ns`` hint is present
        in *metadata* but has no landing site on this backend today. Options
        for closing the gap — segmented integration around each pulse
        window, or an upstream request for a ``dtmax`` knob on dynamiqs'
        deterministic methods — are unimplemented future work.
        """
        resolved = self._normalize_dq_options(options)
        if "max_steps" not in resolved:
            default = default_solver_steps(metadata, tlist)
            if default is not None:
                resolved["max_steps"] = default
        return resolved

    @staticmethod
    def _normalize_dq_options(options: dict[str, Any] | None) -> dict[str, Any]:
        """Map quchip option aliases onto dynamiqs' canonical key names.

        * ``progress_bar`` → ``progress_meter``
        * ``nsteps`` → ``max_steps``

        Returns a fresh dict and is idempotent (a canonical key already present
        is left untouched), so it is safe to apply at every point those aliases
        are consumed — the option-merge boundary and the ``Options`` / method
        builders alike.
        """
        resolved = {} if options is None else dict(options)
        if "progress_meter" not in resolved and "progress_bar" in resolved:
            resolved["progress_meter"] = resolved.pop("progress_bar")
        if "max_steps" not in resolved and "nsteps" in resolved:
            resolved["max_steps"] = resolved.pop("nsteps")
        return resolved

    def coerce_state(self, state: State, dims: tuple[int, ...] | None = None) -> State:
        """Convert a QuTiP ``Qobj`` state into a dimensioned dynamiqs qarray.

        Used when a per-call ``backend="dynamiqs"`` override consumes initial
        states built under the QuTiP backend (``chip.state(...)`` before the
        call). Composite subsystem dimensions must survive the conversion:
        dynamiqs checks them when multiplying the Hamiltonian by the state.
        """
        if hasattr(state, "full"):  # qutip.Qobj duck-type; no qutip import needed
            return dq.asqarray(state, dims=dims)
        return state

    # ------------------------------------------------------------------
    # Single-problem solver dispatch
    # ------------------------------------------------------------------

    def sesolve(
        self,
        H: Any,
        psi0: State,
        tlist: Any,
        e_ops: list[Operator] | None = None,
        options: dict[str, Any] | None = None,
    ) -> SolverResult:
        """Solve the Schrödinger equation via ``dynamiqs.sesolve`` (JAX-traced)."""
        result = dq.sesolve(
            H, psi0, jnp.asarray(tlist, dtype=float),
            **self._solve_kwargs(e_ops, options),
        )
        return self._wrap_result(result, solver="sesolve")

    def mesolve(
        self,
        H: Any,
        rho0: State,
        tlist: Any,
        c_ops: list[Operator] | None = None,
        e_ops: list[Operator] | None = None,
        options: dict[str, Any] | None = None,
    ) -> SolverResult:
        """Solve the Lindblad master equation via ``dynamiqs.mesolve`` (JAX-traced)."""
        result = dq.mesolve(
            H, [] if c_ops is None else c_ops, rho0, jnp.asarray(tlist, dtype=float),
            **self._solve_kwargs(e_ops, options),
        )
        return self._wrap_result(result, solver="mesolve")

    # ------------------------------------------------------------------
    # Cached single-problem dispatch (amortize the XLA/diffrax compile floor)
    # ------------------------------------------------------------------
    #
    # A fresh ``simulate()`` re-runs ``prepare_hamiltonian`` (new closures) and
    # re-enters ``dq.sesolve`` with a structurally-fresh ``H`` pytree, so XLA
    # re-traces every call (~490 ms floor). An optimization inner loop that
    # repeatedly solves the SAME operator skeleton with different (traced)
    # pulse parameters pays that floor on every iteration.
    #
    # The fix below builds H *inside* a ``jax.jit``-compiled solve from the
    # engine's clean quchip-side pytrees (qarrays + ``ScalarModulation`` ASTs,
    # whose treedefs are stable across rebuilds, unlike dynamiqs' ephemeral
    # ``BatchedCallable`` closure ids). Every physics datum (static/dynamic
    # operator values, signal leaves, coefficients, c_ops, e_ops, psi0, tlist)
    # flows as a TRACED jit argument, so:
    #   * jax.grad / jax.vmap still flow;
    #   * a structurally-identical problem with different values reuses the
    #     compiled artifact (jax's own jit cache, keyed on argument treedefs +
    #     the static solver/options config);
    #   * NO operator/coefficient is closed over by value, so there is no
    #     stale-value reuse across e.g. a device-frequency sweep.
    #
    # The backend-private ``_jit_solve_cache`` maps a STATIC config signature
    # (solver name + options/method objects + e_ops/c_ops presence) to a
    # jitted callable. It stores ONLY pure functions + static metadata, never a
    # tracer (avoids stale-trace-context bugs). The cache lives entirely in the
    # backend; the engine emits the same HamiltonianDescription regardless.

    _jit_solve_cache: dict[Any, Any]

    def _get_jit_solve_cache(self) -> dict[Any, Any]:
        cache = getattr(self, "_jit_solve_cache", None)
        if cache is None:
            cache = {}
            self._jit_solve_cache = cache
        return cache

    def solve_problem(self, problem: Any) -> SolverResult:
        """Lower and solve a single :class:`SolveProblem` via a cached jitted solve.

        Falls back to the protocol default (one-shot ``prepare_hamiltonian`` +
        ``sesolve``/``mesolve``) whenever the description contains a dynamic
        term that is not a :class:`ScalarModulation` (the cached path can only
        rebuild ``ScalarModulation`` signals).
        """
        description = problem.hamiltonian
        if not self._description_is_cacheable(description):
            return super().solve_problem(problem)

        # The single-problem reuse routes through the shared batch-config
        # resolver: ``problem`` exposes the same ``tlist``/``c_ops``/``solver``/
        # ``options``/``e_ops`` surface, and ``description`` carries ``.metadata``.
        tlist_arr, c_ops, solver_name, opts, e_ops_arg = self._resolve_batch_config(problem, description)
        options_obj = self._options_from_dict(opts)
        method_obj = self._method_from_dict(opts)
        gradient = opts.get("gradient")

        # Decompose the description into clean, traced pytree leaves. Operator
        # *values* are rebuilt here (cheap, ~1 ms) and flow into the jit as
        # traced args (never closed over) so distinct skeletons with the same
        # structure never collide on a stale artifact.
        static_ops = [self.from_canonical_operator(t.operator) for t in description.static_terms]
        static_coeffs = [jnp.asarray(t.coefficient, dtype=jnp.complex128) for t in description.static_terms]
        dyn_ops = [self.from_canonical_operator(t.operator) for t in description.dynamic_terms]
        dyn_mods = [t.time_dependence for t in description.dynamic_terms]

        solve_fn = self._cached_jit_solve(
            solver_name=solver_name,
            options_obj=options_obj,
            method_obj=method_obj,
            gradient=gradient,
            has_e_ops=e_ops_arg is not None,
            n_static=len(static_ops),
            n_dynamic=len(dyn_ops),
            n_c_ops=len(c_ops),
        )

        result = solve_fn(
            static_ops,
            static_coeffs,
            dyn_ops,
            dyn_mods,
            c_ops,
            e_ops_arg if e_ops_arg is not None else [],
            self.coerce_state(problem.initial_state, dims=problem.chip.dims),
            tlist_arr,
        )
        return self._wrap_result(result, solver=solver_name)

    @staticmethod
    def _description_is_cacheable(description: Any) -> bool:
        """Return whether every dynamic term is a ``ScalarModulation`` that can be rebuilt inside the jit.

        A purely static description (``dynamic_terms == []``) is intentionally
        cacheable: ``all(...)`` over an empty sequence is ``True``, and the jit
        builds a static-only RHS from the (still traced) static operators. This
        is correct but an unusual use of a cache aimed at dynamic problems.
        """
        terms = getattr(description, "dynamic_terms", None)
        if terms is None:
            return False
        return all(isinstance(t.time_dependence, ScalarModulation) for t in terms)

    def _cached_jit_solve(
        self,
        *,
        solver_name: str,
        options_obj: Any,
        method_obj: Any,
        gradient: Any,
        has_e_ops: bool,
        n_static: int,
        n_dynamic: int,
        n_c_ops: int,
    ) -> Any:
        """Return a jitted solve closure for this STATIC config signature.

        The key is built ONLY from static/structural metadata: solver name,
        the (hashable, value-equal) dynamiqs ``Options``/``method``/``gradient``
        objects, observable/collapse/term *counts*, and the e_ops presence flag.
        No traced array, and no operator/coefficient *value*, touches the key:
        those all flow as jit arguments, where jax's own cache keys on their
        treedefs + shapes. Two structurally-different problems therefore land on
        different compiled artifacts (either via this dict for graph-affecting
        config, or via jax's argument-treedef cache for operator structure).
        """
        key = (
            solver_name,
            options_obj,
            method_obj,
            gradient,
            bool(has_e_ops),
            int(n_static),
            int(n_dynamic),
            int(n_c_ops),
        )
        cache = self._get_jit_solve_cache()
        fn = cache.get(key)
        if fn is not None:
            return fn

        import jax

        kwargs: dict[str, Any] = {"options": options_obj}
        if method_obj is not None:
            kwargs["method"] = method_obj
        if gradient is not None:
            kwargs["gradient"] = gradient

        def _solve(static_ops, static_coeffs, dyn_ops, dyn_mods, c_ops, e_ops, state0, tarr):
            rhs = self._assemble_modulated_rhs(
                static_ops, static_coeffs, dyn_ops, [mod.signal for mod in dyn_mods]
            )
            exp_ops = list(e_ops) if has_e_ops else None
            if solver_name == "mesolve":
                return dq.mesolve(rhs, list(c_ops), state0, tarr, exp_ops=exp_ops, **kwargs)
            return dq.sesolve(rhs, state0, tarr, exp_ops=exp_ops, **kwargs)

        jitted = jax.jit(_solve)
        cache[key] = jitted
        return jitted

    def _solve_kwargs(
        self, e_ops: list[Operator] | None, options: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Build the common dynamiqs solver kwargs from quchip's option dict."""
        kwargs: dict[str, Any] = {
            "exp_ops": e_ops,
            "options": self._options_from_dict(options),
        }
        method = self._method_from_dict(options)
        if method is not None:
            kwargs["method"] = method
        # Opt-in differentiation mode. Default (no key) leaves dynamiqs'
        # checkpointed reverse-mode adjoint untouched. ``dq.gradient.Forward()``
        # (paired with ``jax.jacfwd``/``jvp``, never ``jax.grad``) is ~2.5x
        # faster for few-input pulse optimization; the user owns the choice
        # since forward-mode cannot be a blanket default (it breaks reverse-mode
        # ``jax.grad`` on the integrator's ``lax.while_loop``).
        if options is not None and options.get("gradient") is not None:
            kwargs["gradient"] = options["gradient"]
        return kwargs

    # ------------------------------------------------------------------
    # Batched IR lowering / dispatch
    # ------------------------------------------------------------------

    def prepare_hamiltonian(
        self,
        description: Any,
        tlist: Any | None = None,
    ) -> PreparedHamiltonian:
        """Convert a :class:`HamiltonianDescription` into a dynamiqs native RHS.

        Static terms are summed as qarrays; dynamic terms with
        ``ScalarModulation`` time-dependence are wrapped via
        ``dynamiqs.modulated`` with a JAX-traceable :class:`_SignalCallable`.
        ``tlist`` is passed through in metadata but not used for sampling —
        dynamiqs evaluates callables on the integrator's adaptive grid.
        """
        static_ops = [self.from_canonical_operator(t.operator) for t in description.static_terms]
        static_coeffs = [t.coefficient for t in description.static_terms]
        dyn_ops: list[Any] = []
        dyn_signals: list[Any] = []
        for operator, signal in self._scalar_dynamic_terms(description):
            dyn_ops.append(self.from_canonical_operator(operator))
            dyn_signals.append(signal)

        rhs = self._assemble_modulated_rhs(static_ops, static_coeffs, dyn_ops, dyn_signals)
        if rhs is None:
            raise ValueError("HamiltonianDescription must contain at least one static or dynamic term.")

        return PreparedHamiltonian(rhs=rhs, metadata=dict(description.metadata))

    @staticmethod
    def _assemble_modulated_rhs(
        static_ops: list[Any],
        static_coeffs: list[Any],
        dyn_ops: list[Any],
        dyn_signals: list[Any],
    ) -> Any:
        """Sum static ``coeff·op`` terms and modulated dynamic terms into one RHS.

        Static terms accumulate as ``Σ coeff·op``; each dynamic term wraps its
        signal-program AST in a JAX-traceable :class:`_SignalCallable` via
        ``dynamiqs.modulated``. Shared by :meth:`prepare_hamiltonian` and the
        cached-jit single-solve so the static/dynamic split lives in one place
        (the cached path passes traced operators/coefficients as jit arguments,
        so this stays fully traceable).
        """
        rhs = None
        for op, coeff in zip(static_ops, static_coeffs):
            term = coeff * op
            rhs = term if rhs is None else rhs + term
        for op, signal in zip(dyn_ops, dyn_signals):
            dynamic = dq.modulated(_SignalCallable(signal), op)
            rhs = dynamic if rhs is None else rhs + dynamic
        return rhs

    def prepare_batch(self, description: Any, tlist: Any) -> VmappedBatch:
        """Lower a :class:`BatchedHamiltonianDescription` into a single vmapped RHS.

        Shared operators (static + per-slot dynamic) are converted exactly
        once via an id-keyed cache. For each dynamic slot, the per-element
        :class:`ScalarModulation` signals are stacked leaf-by-leaf along a
        leading batch axis (``jnp.stack`` on matching pytree leaves) and
        wrapped in a single :class:`_BatchedSignalQArrayCallable` so
        ``dq.timecallable`` can vmap the solve.

        Raises
        ------
        ValueError
            If a slot contains heterogeneous pytree structures (cannot be
            stacked) or a non-``ScalarModulation`` time dependence.
        """
        cached_native = self._make_op_cache()
        rhs = self._sum_terms(description.static_terms, cached_native)

        for slot, op_canonical in enumerate(description.dynamic_operators):
            op = cached_native(op_canonical)
            slot_signals = description.dynamic_signals[slot]
            ref_td = slot_signals[0]
            if not isinstance(ref_td, ScalarModulation):
                raise ValueError(f"dynamiqs prepare_batch only supports ScalarModulation (slot {slot}).")
            ref_struct = jtu.tree_structure(ref_td)
            for td in slot_signals[1:]:
                # PyTreeDef defines runtime __eq__/__ne__ (pybind11 value equality);
                # jax's stubs omit a mypy-visible signature for it.
                if jtu.tree_structure(td) != ref_struct:  # type: ignore[operator]
                    raise ValueError(f"Heterogeneous signal pytree at slot {slot}; cannot stack.")
            stacked_td = (
                ref_td if len(slot_signals) == 1
                else jtu.tree_map(lambda *xs: jnp.stack(xs), *slot_signals)
            )
            dynamic = dq.timecallable(
                _BatchedSignalQArrayCallable(signal=stacked_td.signal, operator=op)
            )
            rhs = dynamic if rhs is None else rhs + dynamic

        if rhs is None:
            raise ValueError("BatchedHamiltonianDescription must contain at least one term.")

        return VmappedBatch(
            rhs=rhs,
            batch_size=description.batch_size,
            metadata=dict(description.metadata),
            tlist=tlist,
        )

    def solve_batch(self, batch: Any, *, progress: bool = True) -> list[SolverResult]:
        """Solve a :class:`SolveBatch` via a single native dynamiqs vmap.

        Raises :class:`RuntimeError` when the batch is not structurally
        homogeneous — no silent fallback. Callers with heterogeneous inputs
        should regroup through :func:`quchip.engine.solve_many`.
        """
        if batch.batch_size == 0:
            return []

        prepared = self.prepare_batch(batch.hamiltonian, batch.tlist)
        tlist_arr, c_ops, solver_name, opts, e_ops = self._resolve_batch_config(batch, prepared)
        options = self._options_from_dict(opts, cartesian_batching=False)
        method = self._method_from_dict(opts)
        method_kw: dict[str, Any] = {"method": method} if method is not None else {}

        H = prepared.rhs
        chip_dims = getattr(batch.chip, "dims", None)
        native_states = [self.coerce_state(s, dims=chip_dims) for s in batch.initial_states]
        stacked_state = self._stack_state_batch(
            native_states,
            dims=chip_dims,
        )

        try:
            if solver_name == "mesolve":
                batched_result = dq.mesolve(
                    H, c_ops, stacked_state, tlist_arr,
                    exp_ops=e_ops, options=options, **method_kw,
                )
            else:
                batched_result = dq.sesolve(
                    H, stacked_state, tlist_arr,
                    exp_ops=e_ops, options=options, **method_kw,
                )
        except (ValueError, TypeError, AttributeError) as exc:
            raise RuntimeError(
                "Dynamiqs native batched solve_batch failed; refusing to silently fall back."
            ) from exc

        return self._split_batched_result(batched_result, solver=solver_name, count=batch.batch_size)

    # ------------------------------------------------------------------
    # Internal: dims / shape coercion
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_dims(dims: list[list[int]] | None, shape: Sequence[int]) -> tuple[int, ...] | None:
        """Normalize quchip's ``[[row], [col]]`` dims metadata to a flat dynamiqs tuple."""
        if dims is not None:
            return normalize_dims_from_list(dims)
        if len(shape) == 2:
            if shape[1] == 1 or shape[0] == shape[1]:
                return (shape[0],)
        return None

    def _reorder_two_body_array(
        self,
        op_ab: Operator,
        index_a: int,
        index_b: int,
        dims: Sequence[int],
    ) -> tuple[Any, int, int]:
        """Validate indices and reorder a two-body operator into ascending device order."""
        validate_two_body_indices(index_a, index_b, dims)

        dense = self.to_array(op_ab)
        expected_dim = dims[index_a] * dims[index_b]
        if dense.shape[0] != expected_dim or dense.shape[1] != expected_dim:
            raise ValueError(
                f"Two-body operator dimension {dense.shape[0]} does not match "
                f"dims[{index_a}]*dims[{index_b}] = {expected_dim}"
            )

        if index_a < index_b:
            return dense, index_a, index_b

        d_a, d_b = dims[index_a], dims[index_b]
        swapped = dense.reshape(d_a, d_b, d_a, d_b).transpose(1, 0, 3, 2).reshape(d_b * d_a, d_b * d_a)
        return swapped, index_b, index_a

    def _embed_two_body_sparse(
        self,
        canonical: Any,
        index_a: int,
        index_b: int,
        dims: Sequence[int],
    ) -> Operator:
        """Embed a sparse two-body operator without densifying the full matrix.

        Decomposes each nonzero's ``(row, col)`` into subsystem indices
        ``(row_A, row_B)`` × ``(col_A, col_B)`` via ``divmod``, fans each out
        across every spectator basis state using the stride array, and
        reassembles as a :class:`SparseDIAQArray`. Keeps complexity linear
        in ``nnz × n_spectators`` rather than ``nnz × total_dim``.
        """
        from quchip.engine.bands import canonical_to_coo

        n_devices = len(dims)
        validate_two_body_indices(index_a, index_b, dims)

        d_a = dims[index_a]
        d_b = dims[index_b]
        expected_dim = d_a * d_b
        if canonical.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Two-body operator dimension {canonical.shape[0]} does not match "
                f"dims[{index_a}]*dims[{index_b}] = {expected_dim}"
            )

        rows, cols, values = canonical_to_coo(canonical)
        if len(rows) == 0:
            return dq.zeros(*dims)

        other_indices = [idx for idx in range(n_devices) if idx not in (index_a, index_b)]
        strides = np.array(
            [int(np.prod(dims[idx + 1:], dtype=int)) for idx in range(n_devices)],
            dtype=int,
        )

        row_a, row_b = np.divmod(rows, d_b)
        col_a, col_b = np.divmod(cols, d_b)

        if other_indices:
            spectator_arr = np.array(
                list(itertools.product(*(range(dims[idx]) for idx in other_indices))), dtype=int,
            )
            spectator_offsets = spectator_arr @ strides[other_indices]
        else:
            spectator_offsets = np.zeros(1, dtype=int)
        n_spectators = spectator_offsets.size

        row_ab_offset = row_a * strides[index_a] + row_b * strides[index_b]
        col_ab_offset = col_a * strides[index_a] + col_b * strides[index_b]

        full_rows = (row_ab_offset[:, None] + spectator_offsets[None, :]).ravel().astype(int)
        full_cols = (col_ab_offset[:, None] + spectator_offsets[None, :]).ravel().astype(int)
        # Each source value fans out over n_spectators row/col pairs.
        full_values = jnp.repeat(jnp.asarray(values, dtype=jnp.complex128), n_spectators)

        return self._sparse_qarray_from_coo(full_rows, full_cols, full_values, tuple(dims))

    @staticmethod
    def _sparse_qarray_from_coo(
        rows: np.ndarray,
        cols: np.ndarray,
        values: Any,
        dims: tuple[int, ...],
    ) -> Operator:
        """Build a :class:`SparseDIAQArray` from COO-format indices + values."""
        total_dim = int(np.prod(dims, dtype=int))
        if values.size == 0:
            return dq.zeros(*dims)

        rows_np = np.asarray(rows, dtype=int)
        cols_np = np.asarray(cols, dtype=int)
        offsets = np.unique(cols_np - rows_np)
        offset_to_idx = {int(offset): idx for idx, offset in enumerate(offsets.tolist())}

        # Integer index structure lives on NumPy; values stay in JAX for traceability.
        diag_indices = np.array(
            [offset_to_idx[int(c - r)] for r, c in zip(rows_np, cols_np)], dtype=int,
        )
        diag_data = jnp.zeros((len(offsets), total_dim), dtype=jnp.complex128)
        diag_data = diag_data.at[diag_indices, cols_np].add(jnp.asarray(values, dtype=jnp.complex128))

        return SparseDIAQArray(
            dims,
            False,
            tuple(int(x) for x in offsets.tolist()),
            diag_data,
        )

    # ------------------------------------------------------------------
    # Internal: options / method builders
    # ------------------------------------------------------------------

    @staticmethod
    def _options_from_dict(
        options: dict[str, Any] | None,
        *,
        cartesian_batching: bool = True,
    ) -> Any:
        """Map a quchip solver-options dict onto a ``dynamiqs.Options`` object."""
        options = DynamiqsBackend._normalize_dq_options(options)
        return dq.Options(
            save_states=bool(options.get("store_states", True)),
            cartesian_batching=cartesian_batching,
            progress_meter=options.get("progress_meter", False),
        )

    @staticmethod
    def _method_from_dict(options: dict[str, Any] | None) -> Any:
        """Extract an explicit ``method`` or build ``Tsit5(max_steps=...)`` from ``max_steps``.

        ``max_steps`` is an abort ceiling (see :meth:`resolve_solver_options`
        for the step-size-bound limitation this implies for finite-support
        pulses in long idle spans).
        """
        options = DynamiqsBackend._normalize_dq_options(options)
        method = options.get("method")
        if method is not None:
            return method
        max_steps = options.get("max_steps")
        if max_steps is not None:
            return dq.method.Tsit5(max_steps=int(max_steps))
        return None

    # ------------------------------------------------------------------
    # Internal: pytree / operator stacking
    # ------------------------------------------------------------------

    def _stack_state_batch(self, states: list[State], *, dims: Any) -> Operator:
        """Stack state arrays into a single batched dynamiqs qarray."""
        return dq.asqarray(jnp.stack([self.to_array(state) for state in states]), dims=dims)

    # ------------------------------------------------------------------
    # Internal: dynamiqs Result -> SolverResult
    # ------------------------------------------------------------------

    @staticmethod
    def _split_batched_result(result: Any, *, solver: str, count: int) -> list[SolverResult]:
        """Unpack a dynamiqs flat-batched solve into per-problem :class:`SolverResult`s."""
        has_states = getattr(result, "states", None) is not None
        has_expects = getattr(result, "expects", None) is not None
        final_state = getattr(result, "final_state", None)

        return [
            SolverResult(
                times=result.tsave,
                # Keep the per-element trajectory stacked: ``result.states[i]``
                # is a ``(T, n, 1/n)`` QArray. NOT unstacked into a Python list
                # — the over-time extractors consume the stacked form directly,
                # and indexing/iterating a stacked QArray still yields per-time
                # states for viz.
                states=result.states[i] if has_states else None,
                expect=list(result.expects[i]) if has_expects else None,
                final_state=final_state[i] if final_state is not None else None,
                stats={"batched": True, "batch_index": i},
                solver=solver,
            )
            for i in range(count)
        ]

    @staticmethod
    def _wrap_result(result: Any, solver: str) -> SolverResult:
        """Convert a single dynamiqs ``Result`` into a :class:`SolverResult`."""
        # Keep the stacked ``QArray`` (shape ``(T, n, 1/n)``) intact rather than
        # exploding it into ``T`` separate states: the over-time extractors read
        # the stacked form directly and a stacked QArray still indexes/iterates
        # per-time for viz.
        states = result.states if getattr(result, "states", None) is not None else None
        expect = list(result.expects) if getattr(result, "expects", None) is not None else None

        stats: dict[str, Any] = {}
        infos = getattr(result, "infos", None)
        if infos is not None:
            stats["infos"] = str(infos)
            nsteps = DynamiqsBackend._extract_nsteps(infos)
            if nsteps is not None:
                stats["nsteps"] = nsteps

        return SolverResult(
            times=result.tsave,
            states=states,
            expect=expect,
            final_state=getattr(result, "final_state", None),
            stats=stats,
            solver=solver,
        )

    @staticmethod
    def _extract_nsteps(infos: Any) -> int | None:
        """Return the maximum step count from a dynamiqs solver ``infos`` object; ``None`` if absent.

        Returns ``None`` when ``nsteps`` is a JAX tracer (i.e., inside a JIT-traced
        loss function) — extracting a concrete integer from a traced value would break
        JAX traceability.
        """
        raw = getattr(infos, "nsteps", None)
        if raw is None:
            return None
        # Guard against traced JAX values: np.asarray on a JAX tracer raises
        # TracerArrayConversionError and breaks @jax.jit / jax.grad.
        try:
            values = np.asarray(raw)
        except Exception:
            return None
        if values.size == 0:
            return None
        return int(values.max())
