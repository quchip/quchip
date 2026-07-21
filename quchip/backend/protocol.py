"""Backend protocol and backend-agnostic solver-result containers.

quchip separates *physics description* from *solver conversion*: the engine
(``quchip.engine``) emits structured IR (``HamiltonianDescription``,
``SolveProblem``, ``BatchedHamiltonianDescription``, ``SolveBatch``) that
carries ordinary-GHz frequencies and backend-free operator payloads
(``CanonicalOperator``). Each concrete backend is free to translate that IR
into whatever native form its solver likes best — this module only fixes the
*contract* (see :class:`Backend`), not the storage.

Unit convention
---------------
All frequencies crossing the backend boundary are **ordinary** (not angular)
GHz, with time in ns. The single ``2π`` conversion lives at the engine
boundary (``stage2_assembly.py``) — backends never rescale.

Aliases
-------
``Operator`` and ``State`` are both ``typing.Any``. Concrete backends bind
them to their native type: ``qutip.Qobj`` for :class:`QuTiPBackend`,
``dynamiqs.QArray`` for :class:`DynamiqsBackend`.

References
----------
* Johansson, Nation, Nori — *QuTiP 2*, Comput. Phys. Commun. 183, 1760 (2012)
* Guilmin et al. — *dynamiqs: an open-source Python library for GPU-accelerated
  and differentiable simulation of quantum systems* (2024)
* Bradbury et al. — *JAX: composable transformations of Python+NumPy programs*
  (2018)
* Lindblad — Commun. Math. Phys. 48, 119 (1976)
* Breuer & Petruccione — *The Theory of Open Quantum Systems* (OUP, 2002)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

import numpy as np

from quchip.backend._dims import normalize_dims_from_list
from quchip.backend.containers import (
    _DEFAULT_SOLVE_OPTIONS,
    DeferredBatch,
    EagerBatch,
    EigensystemData,
    PreparedBatch,
    PreparedHamiltonian,
    SolverResult,
    VmappedBatch,
)

if TYPE_CHECKING:
    from quchip.engine.ir import (
        BatchedHamiltonianDescription,
        CanonicalOperator,
        HamiltonianDescription,
        SolveBatch,
        SolveProblem,
    )

Operator = Any
State = Any


class Backend(ABC):
    """Abstract contract implemented by every quchip operator backend.

    A backend wraps a quantum-dynamics library (QuTiP, dynamiqs, ...) and
    exposes three orthogonal surfaces:

    * **Operator algebra** — ``destroy``/``create``/``number``/``identity``,
      ``tensor``, ``embed``, ``dag``, ``matmul``, eigensystem helpers.
      Concrete operators support native Python arithmetic (``+ - * @``);
      quchip never introduces ``scale``/``add`` wrappers.
    * **State algebra** — ``basis``/``coherent``, ``overlap``, ``ptrace``,
      ``expect``, ket ↔ density-matrix conversion.
    * **IR lowering & solve dispatch** — ``prepare_hamiltonian`` /
      ``prepare_batch`` convert engine IR into native RHS form;
      ``sesolve`` / ``mesolve`` run the solver; ``solve_problem`` /
      ``parallel_solve_problems`` / ``solve_batch`` drive the full
      engine-to-result pipeline.

    Defaults provided here are NumPy-based and correct for any backend, so
    concrete subclasses only override where they can do better (e.g. QuTiP
    reuses ``Qobj.eigenstates``, dynamiqs reuses ``dq.expect``).
    """

    # ------------------------------------------------------------------
    # Array / scalar surface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def array_module(self) -> Any:
        """Return the array module used by backend-aware numeric code.

        Returns ``numpy`` for CPU-only backends and ``jax.numpy`` for
        JAX-traceable backends. Engine code that must stay differentiable
        (signal evaluation, frame assembly) routes array ops through this
        module so the dynamiqs backend preserves JAX traceability end-to-end.
        """
        ...

    @abstractmethod
    def to_array(self, op: Operator) -> Any:
        """Return a dense array (native to :attr:`array_module`) for *op*."""
        ...

    def overlap(self, a: State, b: State) -> complex:
        """Return the scalar inner product ⟨a|b⟩ for two kets."""
        a_arr = np.asarray(self.to_array(a), dtype=complex)
        b_arr = np.asarray(self.to_array(b), dtype=complex)
        return complex(np.conj(a_arr).T @ b_arr)

    def norm(self, state_or_op: State | Operator) -> Any:
        """Return the Frobenius / ℓ²-norm of a state or operator.

        A concrete ``float`` for CPU-only backends; a JAX-traceable backend
        may return a native 0-d array instead so a traced amplitude stays
        differentiable — callers that need concreteness route through
        :func:`~quchip.utils.jax_utils.maybe_concrete_scalar`.
        """
        arr = np.asarray(self.to_array(state_or_op), dtype=complex)
        return float(np.linalg.norm(arr))

    def trace(self, op: Operator) -> complex:
        """Return the scalar trace ``Tr(op)``."""
        arr = np.asarray(self.to_array(op), dtype=complex)
        return complex(np.trace(arr))

    # ------------------------------------------------------------------
    # Single-mode operator factories (Fock-basis defaults)
    # ------------------------------------------------------------------

    def _single_mode(self, data: np.ndarray) -> Operator:
        """Wrap a dense ``n × n`` Fock-basis matrix as a native operator.

        Concrete backends route through :meth:`from_canonical_operator` so
        they can choose the optimal native layout (dense, DIA, CSR).
        """
        from quchip.engine.ir import CanonicalOperator

        n = data.shape[0]
        return self.from_canonical_operator(
            CanonicalOperator.from_dense(data, dims=(n,), basis="fock", subsystem_labels=("0",))
        )

    def destroy(self, n: int) -> Operator:
        """Build the annihilation operator :math:`\\hat a` for an *n*-level Fock space."""
        data = np.zeros((n, n), dtype=complex)
        for k in range(1, n):
            data[k - 1, k] = np.sqrt(k)
        return self._single_mode(data)

    def create(self, n: int) -> Operator:
        """Build the creation operator :math:`\\hat a^\\dagger` for an *n*-level Fock space."""
        data = np.zeros((n, n), dtype=complex)
        for k in range(1, n):
            data[k, k - 1] = np.sqrt(k)
        return self._single_mode(data)

    def number(self, n: int) -> Operator:
        """Build the number operator :math:`\\hat n = \\hat a^\\dagger \\hat a`."""
        return self._single_mode(np.diag(np.arange(n, dtype=complex)))

    def identity(self, n: int) -> Operator:
        """Build the identity operator for an *n*-level space."""
        return self._single_mode(np.eye(n, dtype=complex))

    def from_array(self, data: Any, dims: list[list[int]] | None = None) -> Operator:
        """Construct a native operator from a dense matrix.

        *dims* accepts quchip's row/col layout (``[[row_dims], [col_dims]]``)
        or a flat list; ``None`` means a single subsystem of size
        ``data.shape[0]``.
        """
        from quchip.engine.ir import CanonicalOperator

        arr = np.asarray(data, dtype=complex)
        dim_tuple = normalize_dims_from_list(dims, arr.shape[0])
        labels = tuple(str(i) for i in range(len(dim_tuple)))
        return self.from_canonical_operator(
            CanonicalOperator.from_dense(arr, dims=dim_tuple, basis="fock", subsystem_labels=labels)
        )

    def diag(self, values: Any, dims: list[list[int]] | None = None) -> Operator:
        """Construct a backend-native diagonal operator from main-diagonal *values*.

        Backends that distinguish layouts (e.g. dynamiqs sparse-DIA) should
        return their sparse representation so element-wise composition with
        other diagonal operators stays sparse. Default falls through to
        :meth:`from_array` after building a dense ``np.diag``.
        """
        from quchip.engine.ir import CanonicalOperator

        arr = np.asarray(values).reshape(-1)
        dim_tuple = normalize_dims_from_list(dims, arr.shape[0])
        labels = tuple(str(i) for i in range(len(dim_tuple)))
        return self.from_canonical_operator(
            CanonicalOperator.from_dia(
                arr.astype(complex)[None, :],
                np.array([0], dtype=int),
                shape=(arr.shape[0], arr.shape[0]),
                dims=dim_tuple,
                basis="fock",
                subsystem_labels=labels,
            )
        )

    @abstractmethod
    def to_canonical_operator(self, op: Operator) -> "CanonicalOperator":
        """Serialize a native operator into the backend-agnostic canonical IR."""
        ...

    @abstractmethod
    def from_canonical_operator(self, canonical: "CanonicalOperator") -> Operator:
        """Reconstruct a native operator from the canonical IR payload."""
        ...

    # ------------------------------------------------------------------
    # Operator algebra
    # ------------------------------------------------------------------

    def coerce_operator(self, op: Operator) -> Operator:
        """Coerce an array-like local operator into backend-native form.

        The operator-side mirror of :meth:`coerce_state`. Device-side
        physical-coupling accessors (e.g.
        :meth:`~quchip.devices.protocols.ChargeCoupled.charge_coupling_operator`)
        return dense, trace-safe arrays rather than backend-native
        operators; composition entry points (:meth:`tensor`, :meth:`dag`)
        route their operands through this hook so both forms compose
        freely. Backends whose native operators are already arrays
        override with a trace-safe passthrough.
        """
        return self.from_array(self.to_array(op))

    def dag(self, op: Operator) -> Operator:
        """Return the Hermitian conjugate ``op†``."""
        arr = np.asarray(self.to_array(op), dtype=complex)
        return self.from_array(np.conj(arr).T)

    def matmul(self, a: Operator, b: Operator) -> Operator:
        """Return the matrix product ``a @ b``."""
        return a @ b

    def eigenenergies(self, op: Operator) -> Any:
        """Return the ascending eigenvalues of a Hermitian operator."""
        dense = np.asarray(self.to_array(op), dtype=complex)
        return np.linalg.eigvalsh(dense)

    def eigenstates(self, op: Operator) -> tuple[Any, Any]:
        """Return ``(eigenvalues, eigenstates)`` of a Hermitian operator, ascending."""
        data = self.eigensystem_data(op)
        return data.eigenvalues, data.eigenstates

    def eigensystem_data(self, op: Operator) -> EigensystemData:
        """Return ascending eigenvalues, stacked eigenvector matrix, and lazy eigenstates."""
        dense = np.asarray(self.to_array(op), dtype=complex)
        evals, evecs = np.linalg.eigh(dense)

        def build_states() -> list[Any]:
            return [self.from_array(evecs[:, idx:idx + 1]) for idx in range(evecs.shape[1])]

        return EigensystemData(
            eigenvalues=evals,
            eigenvector_matrix=evecs,
            _states_builder=build_states,
        )

    def expect(self, op: Operator, state: State) -> complex:
        """Return the expectation value ⟨state|op|state⟩ — accepts ket or density matrix."""
        op_arr = np.asarray(self.to_array(op), dtype=complex)
        state_arr = np.asarray(self.to_array(state), dtype=complex)
        if state_arr.ndim == 2 and state_arr.shape[1] == 1:
            return complex(np.conj(state_arr).T @ op_arr @ state_arr)
        return complex(np.trace(op_arr @ state_arr))

    def ptrace(self, state: State, keep: int | list[int], dims: list[int]) -> State:
        """Reduce a composite state onto subsystem(s) *keep* via partial trace.

        Parameters
        ----------
        state
            Composite ket or density matrix over the subsystems in *dims*.
        keep
            Subsystem index (or indices) to retain; all others are traced out.
        dims
            Subsystem dimensions of the full composite space, in order.
        """
        if isinstance(keep, int):
            keep = [keep]
        state_arr = np.asarray(self.to_array(state), dtype=complex)
        n = len(dims)

        if state_arr.ndim == 2 and state_arr.shape[1] == 1:
            state_arr = state_arr @ np.conj(state_arr).T

        rho = state_arr.reshape(list(dims) + list(dims))
        for idx in reversed(sorted(set(range(n)) - set(keep))):
            rho = np.trace(rho, axis1=idx, axis2=idx + n)
            n -= 1

        kept_dim = int(np.prod([dims[k] for k in keep]))
        return self.from_array(rho.reshape(kept_dim, kept_dim))

    def permute_state(self, state: State, dims: Sequence[int], order: Sequence[int]) -> State:
        """Reorder a composite state's subsystems.

        ``dims`` are the subsystem levels in *state*'s current tensor order.
        ``order`` follows ``numpy.transpose`` convention: ``order[i]`` is the
        *current*-order subsystem index that becomes position ``i`` after
        permuting (so ``order == list(range(len(dims)))`` is a no-op).

        Default: densify, reshape to one axis per subsystem, ``transpose``,
        reshape back — correct for any backend. A ket reshapes/transposes
        once; a density matrix applies the same subsystem permutation to
        both the row and column index groups. Concrete backends may override
        this with a native reorder (e.g. QuTiP's ``Qobj.permute``) that stays
        sparse-friendly and keeps dims metadata.

        Every array op routes through :attr:`array_module` (never bare
        ``numpy``), so a traced state stays traced end-to-end under the
        dynamiqs backend. ``dims``/``order`` are always concrete Python
        ints (subsystem structure, never a physics parameter), so their
        product is taken with :func:`math.prod` rather than
        ``array_module.prod`` — routing a *shape* value through the array
        module would turn it into a tracer under ``jit`` and make the
        subsequent ``reshape`` fail.
        """
        xp = self.array_module
        arr = xp.asarray(self.to_array(state), dtype=complex)
        dims = list(dims)
        is_ket = arr.ndim == 2 and arr.shape[1] == 1
        if is_ket:
            ket_axes = arr.reshape(dims)
            permuted_ket_axes = xp.transpose(ket_axes, order)
            return self.from_array(permuted_ket_axes.reshape(-1, 1))

        n_subsystems = len(dims)
        row_then_col_axes = arr.reshape(dims + dims)
        # The row axes take `order` directly; the column axes need the same
        # permutation, shifted by `n_subsystems` since they occupy the second
        # half of the reshaped axes.
        row_and_col_order = list(order) + [n_subsystems + i for i in order]
        permuted = xp.transpose(row_then_col_axes, row_and_col_order)
        total_dim = math.prod(dims)
        return self.from_array(permuted.reshape(total_dim, total_dim))

    # ------------------------------------------------------------------
    # Batched-over-time extraction
    # ------------------------------------------------------------------
    #
    # A trajectory is ``T`` saved states. The user-facing extractors
    # (``population_array``, ``populations``, ``overlap_array``,
    # ``amplitude_array``) need one scalar/vector per save point. Doing that
    # with a Python loop over ``T`` single-state ``expect``/``ptrace`` calls
    # makes per-point dispatch the dominant cost on long ``tlist``s and
    # clean-``O(N)`` sweeps, rather than the matrix work itself. The methods
    # below collapse that loop into a single batched op over the leading
    # time axis.
    #
    # The native stacked form is the backend's own (a stacked ``QArray`` for
    # dynamiqs, a ``np.stack`` for QuTiP) so neither layout is forced on the
    # other. dynamiqs implementations stay ``jnp`` — storing/operating on one
    # stacked array is strictly more traceable than a Python list of
    # per-time tracers.

    def stack_states(self, states: Any) -> Any:
        """Stack a trajectory's saved states into a single ``(T, …)`` array.

        Default densifies each state and ``np.stack``s along a new leading
        time axis. Backends whose solver already returns a stacked native
        state (dynamiqs) override this to a no-op pass-through.
        """
        return np.stack([np.asarray(self.to_array(s), dtype=complex) for s in states])

    @staticmethod
    def _is_ket_stack(stacked: Any) -> bool:
        """Return whether a stacked trajectory is kets ``(T, n, 1)`` vs DMs ``(T, n, n)``."""
        return stacked.ndim == 3 and stacked.shape[2] == 1

    def expect_over_time(self, op: Operator, stacked_states: Any) -> Any:
        """Return ⟨op⟩(t) for every save point, as one ``(T,)`` array in :attr:`array_module`.

        Accepts a ket stack ``(T, n, 1)`` (returns ⟨ψ|op|ψ⟩) or a
        density-matrix stack ``(T, n, n)`` (returns ``Tr(op·ρ)``). One einsum
        replaces the per-point ``expect`` loop. Every intermediate stays in
        :attr:`array_module`, so the JAX backend keeps the trace differentiable.
        """
        xp = self.array_module
        op_arr = xp.asarray(self.to_array(op), dtype=complex)
        rho = xp.asarray(self.to_array(stacked_states), dtype=complex)
        if self._is_ket_stack(rho):
            psi = rho[..., 0]
            return xp.einsum("ti,ij,tj->t", xp.conj(psi), op_arr, psi)
        return xp.einsum("tij,ji->t", rho, op_arr)

    def overlap_over_time(self, target: State, stacked_states: Any) -> Any:
        """Return ⟨target|ψ(t)⟩ for every save point, as one ``(T,)`` array.

        For a ket stack this is the **phase-sensitive complex amplitude**
        ⟨target|ψ(t)⟩ (never ``|·|²`` — phase-dependent gradients flow
        through it). For a density-matrix stack there is no single phase, so
        it returns the target-state population ⟨target|ρ(t)|target⟩. Stays in
        :attr:`array_module` for JAX traceability.
        """
        xp = self.array_module
        tgt = xp.asarray(self.to_array(target), dtype=complex).reshape(-1)
        arr = xp.asarray(self.to_array(stacked_states), dtype=complex)
        if self._is_ket_stack(arr):
            return xp.einsum("i,ti->t", xp.conj(tgt), arr[..., 0])
        return xp.einsum("i,tij,j->t", xp.conj(tgt), arr, tgt)

    def populations_over_time(self, stacked_states: Any) -> Any:
        """Return full-chip diagonal populations ``(T, ∏dims)`` (real) for every save point.

        For a ket stack this is ``|ψ(t)|²`` along the level axis (the density
        matrix is never built); for a DM stack it is the real diagonal. Stays
        in :attr:`array_module` for JAX traceability.
        """
        xp = self.array_module
        arr = xp.asarray(self.to_array(stacked_states), dtype=complex)
        if self._is_ket_stack(arr):
            return xp.abs(arr[..., 0]) ** 2
        return xp.real(xp.diagonal(arr, axis1=1, axis2=2))

    def ptrace_over_time(self, stacked_states: Any, keep: int | list[int], dims: list[int]) -> Any:
        """Reduce onto subsystem(s) *keep* at every save point, returning a ``(T, k, k)`` DM stack.

        Reduces a ket or DM trajectory onto subsystem(s) *keep* without a
        per-point Python loop.
        """
        if isinstance(keep, int):
            keep = [keep]
        rho = np.asarray(stacked_states, dtype=complex)
        n = len(dims)
        if self._is_ket_stack(rho):
            psi = rho[..., 0]
            rho = np.einsum("ti,tj->tij", psi, np.conj(psi))
        rho = rho.reshape([rho.shape[0]] + list(dims) + list(dims))
        offset = 1  # leading time axis
        live = n
        for idx in reversed(sorted(set(range(n)) - set(keep))):
            rho = np.trace(rho, axis1=offset + idx, axis2=offset + idx + live)
            live -= 1
        kept_dim = int(np.prod([dims[k] for k in keep]))
        return self.array_module.asarray(rho.reshape(rho.shape[0], kept_dim, kept_dim))

    def embed(self, op: Operator, device_index: int, dims: Sequence[int]) -> Operator:
        """Embed a single-device operator at *device_index* into the full tensor space.

        Validation is shared; the identity factors and tensor product
        dispatch through the backend's own :meth:`identity` / :meth:`tensor`,
        so each backend keeps its native-optimal layout.

        Parameters
        ----------
        op
            Single-device operator whose dimension matches ``dims[device_index]``.
        device_index
            Position of *op* within the tensor product.
        dims
            Subsystem dimensions of the full space, in order.
        """
        n_devices = len(dims)
        if device_index < 0 or device_index >= n_devices:
            raise ValueError(f"device_index {device_index} out of range for {n_devices} devices")
        if op.shape[0] != dims[device_index]:
            raise ValueError(
                f"Operator dimension {op.shape[0]} does not match dims[{device_index}] = {dims[device_index]}"
            )
        factors = [op if i == device_index else self.identity(d) for i, d in enumerate(dims)]
        return self.tensor(*factors)

    @abstractmethod
    def embed_two_body(
        self,
        op_ab: Operator,
        index_a: int,
        index_b: int,
        dims: Sequence[int],
    ) -> Operator:
        """Embed a two-body operator on devices *index_a* ⊗ *index_b* into the full space.

        Handles subsystem SWAP when ``index_a > index_b`` and identity-padding
        for non-adjacent devices, without materializing the dense full matrix
        when the backend supports sparse layouts.
        """
        ...

    # ------------------------------------------------------------------
    # State factories
    # ------------------------------------------------------------------

    def basis(self, n: int, k: int) -> State:
        """Build the Fock basis ket :math:`|k\\rangle` in an *n*-level space."""
        vec = np.zeros((n, 1), dtype=complex)
        vec[k, 0] = 1.0
        return self.from_array(vec)

    def tensor_states(self, *states: State) -> State:
        """Return the tensor product of states (defaults to :meth:`tensor`)."""
        return self.tensor(*states)

    def coherent(self, n: int, alpha: complex) -> State:
        """Build the coherent state :math:`|\\alpha\\rangle` truncated to *n* Fock levels.

        Built from the analytic series
        :math:`|\\alpha\\rangle = e^{-|\\alpha|^2/2} \\sum_k \\alpha^k/\\sqrt{\\Gamma(k+1)}\\,|k\\rangle`.
        """
        import math

        coeffs = np.array(
            [alpha**k / np.sqrt(math.factorial(k)) for k in range(n)],
            dtype=complex,
        )
        coeffs *= np.exp(-0.5 * abs(alpha) ** 2)
        return self.from_array(coeffs.reshape(n, 1))

    def state_to_dm(self, state: State) -> State:
        """Return a density matrix; pass through if *state* is already one."""
        if not self.is_ket(state):
            return state
        arr = np.asarray(self.to_array(state), dtype=complex)
        return self.from_array(arr @ np.conj(arr).T)

    def is_ket(self, state: State) -> bool:
        """Return whether *state* is a column vector (ket) rather than a density matrix."""
        arr = np.asarray(self.to_array(state), dtype=complex)
        return arr.ndim == 2 and arr.shape[1] == 1

    def as_density_matrix(self, state: State) -> State:
        """Promote a ket to its density matrix; pass a density matrix through unchanged."""
        return self.state_to_dm(state) if self.is_ket(state) else state

    @abstractmethod
    def tensor(self, *operators: Operator) -> Operator:
        """Return the tensor product of operators, preserving subsystem dims metadata."""
        ...

    # ------------------------------------------------------------------
    # Solver dispatch
    # ------------------------------------------------------------------

    @abstractmethod
    def sesolve(
        self,
        H: Any,
        psi0: State,
        tlist: Any,
        e_ops: list[Operator] | None = None,
        options: dict[str, Any] | None = None,
    ) -> SolverResult:
        """Solve the Schrödinger equation :math:`i\\hbar \\partial_t |\\psi\\rangle = H |\\psi\\rangle`.

        Parameters
        ----------
        H
            Native Hamiltonian (``Qobj``/``QobjEvo`` or dynamiqs
            ``TimeQArray``).
        psi0
            Initial ket.
        tlist
            1D array of save times in ns.
        e_ops
            Optional observables to accumulate ⟨ψ|Oᵢ|ψ⟩ along the trajectory.
        options
            Backend-specific options dict. See each backend's
            ``resolve_solver_options`` for supported keys.
        """
        ...

    @abstractmethod
    def mesolve(
        self,
        H: Any,
        rho0: State,
        tlist: Any,
        c_ops: list[Operator] | None = None,
        e_ops: list[Operator] | None = None,
        options: dict[str, Any] | None = None,
    ) -> SolverResult:
        """Solve the Lindblad master equation (Lindblad 1976; Breuer & Petruccione 2002).

        Integrates
        :math:`\\dot\\rho = -i[H,\\rho] + \\sum_k \\mathcal D[L_k]\\rho`
        with
        :math:`\\mathcal D[L]\\rho = L\\rho L^\\dagger - \\tfrac12\\{L^\\dagger L, \\rho\\}`.
        """
        ...

    def batched_sesolve(
        self,
        problems: list[dict[str, Any]],
        *,
        n_jobs: int = -1,
        progress: bool = True,
    ) -> list[SolverResult]:
        """Solve multiple sesolve problems. Default: sequential dispatch.

        Backends that can vectorize (dynamiqs ``vmap``) or parallelize (QuTiP
        via loky) override this method. Each problem dict carries the same
        kwargs as :meth:`sesolve`.
        """
        return [self.sesolve(**p) for p in problems]

    def batched_mesolve(
        self,
        problems: list[dict[str, Any]],
        *,
        n_jobs: int = -1,
        progress: bool = True,
    ) -> list[SolverResult]:
        """Solve multiple mesolve problems. Default: sequential dispatch."""
        return [self.mesolve(**p) for p in problems]

    # ------------------------------------------------------------------
    # IR lowering
    # ------------------------------------------------------------------

    def prepare_hamiltonian(
        self,
        description: "HamiltonianDescription",
        tlist: Any,
    ) -> "PreparedHamiltonian":
        """Convert a :class:`HamiltonianDescription` into a native solver RHS.

        The engine passes frequencies already converted by ``2π`` (angular
        rad/ns); backends must not rescale. Concrete backends override this
        method to choose their native time-dependence representation.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement prepare_hamiltonian()")

    def resolve_solver_options(
        self,
        options: dict[str, Any],
        *,
        metadata: dict[str, Any],
        tlist: Any,
    ) -> dict[str, Any]:
        """Merge user options with backend-side defaults and metadata heuristics.

        Default is a shallow copy; concrete backends use *metadata*
        (e.g. ``spectral_bound_ghz``) to fill in integrator step budgets when
        the user did not specify one.
        """
        return dict(options)

    def coerce_state(self, state: State, dims: tuple[int, ...] | None = None) -> State:
        """Convert a foreign-native *state* into this backend's native form.

        Called at the solve boundary so a per-call ``backend=`` override
        (see :meth:`QuantumSequence.simulate`) accepts initial states that
        were built under another backend — e.g. a QuTiP ``Qobj`` from
        ``chip.state(...)`` handed to a dynamiqs gradient solve. Default:
        pass through unchanged. ``dims`` are the chip's subsystem levels,
        for backends whose state type carries tensor structure.
        """
        _ = dims
        return state

    def solve_problem(self, problem: "SolveProblem") -> SolverResult:
        """Lower and solve a :class:`SolveProblem` — the single-element entry point.

        Picks ``mesolve`` when collapse operators are present (open system)
        or ``sesolve`` otherwise, unless ``problem.solver`` forces a choice.
        """
        prepared = self.prepare_hamiltonian(problem.hamiltonian, problem.tlist)
        tlist_arr = self.array_module.asarray(problem.tlist, dtype=float)

        c_ops = list(problem.c_ops) if problem.c_ops else []
        solver = problem.solver or ("mesolve" if c_ops else "sesolve")
        opts = self._merge_options(problem.options, metadata=prepared.metadata, tlist=tlist_arr)
        e_ops_arg = problem.e_ops if isinstance(problem.e_ops, list) else None
        psi0 = self.coerce_state(problem.initial_state, dims=problem.chip.dims)

        if solver == "sesolve":
            return self.sesolve(prepared.rhs, psi0, tlist_arr,
                                e_ops=e_ops_arg, options=opts)
        return self.mesolve(prepared.rhs, psi0, tlist_arr,
                            c_ops=c_ops, e_ops=e_ops_arg, options=opts)

    def parallel_solve_problems(
        self,
        problems: list["SolveProblem"],
        *,
        progress: bool = True,
    ) -> list[SolverResult] | None:
        """Dispatch structurally heterogeneous problems in parallel.

        Parameters
        ----------
        problems
            Problems that cannot be merged into one structural batch.
        progress
            Display solver progress when supported.

        Returns
        -------
        list[SolverResult] | None
            Backend results in input order, or ``None`` to retain the engine's
            structural-group dispatch.
        """
        _ = problems, progress
        return None

    def prepare_batch(
        self,
        description: "BatchedHamiltonianDescription",
        tlist: Any,
    ) -> "PreparedBatch":
        """Lower a :class:`BatchedHamiltonianDescription` into a prepared batch.

        The return type declares the batching strategy:
        :class:`~quchip.backend.containers.EagerBatch` (one RHS per element),
        :class:`~quchip.backend.containers.VmappedBatch` (one natively
        batched RHS — dynamiqs), or
        :class:`~quchip.backend.containers.DeferredBatch` (backend-private
        payload consumed by an overridden :meth:`solve_batch` — QuTiP).
        Default: lowers each element independently via
        :meth:`prepare_hamiltonian` into an :class:`EagerBatch`.
        """
        rhs_list: list[Any] = []
        shared_metadata: dict[str, Any] = {}
        for idx in range(description.batch_size):
            prepared = self.prepare_hamiltonian(description.element(idx), tlist)
            rhs_list.append(prepared.rhs)
            if not shared_metadata:
                shared_metadata = dict(prepared.metadata)

        return EagerBatch(
            rhs_list=rhs_list,
            batch_size=description.batch_size,
            metadata=shared_metadata,
            tlist=tlist,
        )

    def solve_batch(self, batch: "SolveBatch", *, progress: bool = True) -> list[SolverResult]:
        """Lower and solve a :class:`SolveBatch`.

        Default: prepares the batch then dispatches each element's RHS
        through :meth:`batched_sesolve` / :meth:`batched_mesolve`. Native
        backends override to avoid the per-element unpack.
        """
        if batch.batch_size == 0:
            return []

        prepared = self.prepare_batch(batch.hamiltonian, batch.tlist)
        tlist_arr, c_ops, solver_name, opts, e_ops_arg = self._resolve_batch_config(batch, prepared)

        if isinstance(prepared, DeferredBatch):
            raise RuntimeError(
                f"{type(self).__name__}.prepare_batch produced a DeferredBatch; "
                "the default solve_batch cannot dispatch it. A backend that "
                "defers RHS construction must override solve_batch to consume "
                "its own deferred payload."
            )

        dict_problems: list[dict[str, Any]] = []
        for idx in range(batch.batch_size):
            rhs = prepared.rhs if isinstance(prepared, VmappedBatch) else prepared.rhs_list[idx]
            dict_problems.append(
                self._element_solver_kwargs(
                    solver_name,
                    rhs,
                    batch.initial_states[idx],
                    tlist_arr,
                    e_ops=e_ops_arg,
                    c_ops=c_ops,
                    options=dict(opts),
                )
            )

        runner = self.batched_mesolve if solver_name == "mesolve" else self.batched_sesolve
        return runner(dict_problems, progress=progress)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_op_cache(self) -> Callable[[Any], Operator]:
        """Return an id-keyed memoizing wrapper over :meth:`from_canonical_operator`.

        A fresh cache per call: it is shared across the static and dynamic
        conversions of one batch preparation, never across calls (so JAX
        tracers cannot leak between traces).
        """
        op_cache: dict[int, Operator] = {}

        def cached(canonical: Any) -> Operator:
            key = id(canonical)
            op = op_cache.get(key)
            if op is None:
                op = self.from_canonical_operator(canonical)
                op_cache[key] = op
            return op

        return cached

    @staticmethod
    def _sum_terms(terms: Any, to_native: Callable[[Any], Operator]) -> Operator | None:
        """Sum coefficient-scaled terms into one native operator (``None`` if empty)."""
        rhs: Operator | None = None
        for term in terms:
            scaled = term.coefficient * to_native(term.operator)
            rhs = scaled if rhs is None else rhs + scaled
        return rhs

    @staticmethod
    def _scalar_dynamic_terms(description: "HamiltonianDescription") -> Iterator[tuple[Any, Any]]:
        """Yield ``(operator, signal)`` for each ``ScalarModulation`` dynamic term.

        Centralizes the time-dependence filter both backends apply when
        lowering dynamic terms. ``DynamicTerm.time_dependence`` is typed as
        ``ScalarModulation``, so the ``isinstance`` guard is defensive — it
        keeps any future non-scalar modulation from silently contributing a
        malformed RHS — while leaving observed behavior unchanged.
        """
        from quchip.engine.ir import ScalarModulation

        for term in description.dynamic_terms:
            td = term.time_dependence
            if isinstance(td, ScalarModulation):
                yield term.operator, td.signal

    def _merge_options(
        self,
        user_options: dict[str, Any],
        *,
        metadata: dict[str, Any],
        tlist: Any,
    ) -> dict[str, Any]:
        """Apply :data:`_DEFAULT_SOLVE_OPTIONS` then backend heuristics.

        The single option-merge boundary: defaults are applied here exactly
        once, and the backend's :meth:`resolve_solver_options` finalizes the
        dict (metadata-derived step budgets, portability key-stripping).
        """
        merged = dict(_DEFAULT_SOLVE_OPTIONS)
        merged.update(user_options)
        return self.resolve_solver_options(merged, metadata=metadata, tlist=tlist)

    def _resolve_batch_config(
        self,
        batch: Any,
        prepared: Any,
    ) -> tuple[Any, list[Any], str, dict[str, Any], list[Any] | None]:
        """Resolve the per-batch solve configuration shared by every backend.

        Coerces the save grid (via :attr:`array_module`), assembles collapse
        operators, selects the solver (``mesolve`` when collapse operators are
        present, unless ``batch.solver`` forces a choice), and merges options
        through the single boundary. Returns
        ``(tlist_arr, c_ops, solver_name, opts, e_ops_arg)``; each backend
        contributes only its RHS-sourcing + native-solve dispatch tail.

        *prepared* is any payload exposing ``.metadata`` — a
        :class:`PreparedBatch` on the batched paths, or the
        :class:`HamiltonianDescription` when the dynamiqs single-solve reuses
        this resolver.
        """
        tlist_arr = self.array_module.asarray(batch.tlist, dtype=float)
        c_ops = list(batch.c_ops) if batch.c_ops else []
        solver_name = batch.solver or ("mesolve" if c_ops else "sesolve")
        opts = self._merge_options(batch.options, metadata=prepared.metadata, tlist=tlist_arr)
        e_ops_arg = batch.e_ops if isinstance(batch.e_ops, list) else None
        return tlist_arr, c_ops, solver_name, opts, e_ops_arg

    @staticmethod
    def _element_solver_kwargs(
        solver_name: str,
        rhs: Any,
        state: State,
        tlist: Any,
        *,
        e_ops: list[Operator] | None,
        c_ops: list[Operator],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble one batch element's solver kwargs (pure dict construction).

        Selects the state key (``rho0`` for ``mesolve``, ``psi0`` for
        ``sesolve``) and attaches ``c_ops`` only on the open-system path. No
        backend native type is introduced here — *rhs* / *state* / *options*
        are passed through unchanged.
        """
        kwargs: dict[str, Any] = {
            "H": rhs,
            "tlist": tlist,
            "e_ops": e_ops,
            "options": options,
        }
        if solver_name == "mesolve":
            kwargs["rho0"] = state
            if c_ops:
                kwargs["c_ops"] = list(c_ops)
        else:
            kwargs["psi0"] = state
        return kwargs
