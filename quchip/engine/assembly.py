"""Assemble an :class:`EngineResult` from chip physics, controls, and frame.

Hamiltonians enter in ordinary GHz and are scaled by ``2π`` to rad/ns for
``d|ψ⟩/dt = -i H |ψ⟩``. Carrier and observable-demodulation phases convert
frequencies to rad/ns as well, and solver hints convert them back to GHz for
display.

Rotating frames subtract ``Σᵢ ω_frame,ᵢ nᵢ`` from the static Hamiltonian.
Coupling and drive operators split into excitation-change bands with weight
``w = col - row`` and carriers ``exp(-i w·ω t)``.
:class:`~quchip.approximations.Exact` keeps every band.
:class:`~quchip.approximations.RWA` keeps total-excitation-conserving static
bands and matches delivered-signal bands to operator bands. Time dependence is
represented by :class:`~quchip.engine.ir.SignalProgram` expressions, which
backends lower to native coefficients.

References: Scully & Zubairy, *Quantum Optics* (1997), §5.1; Jaynes & Cummings,
*Proc. IEEE* 51, 89 (1963); Walls & Milburn, *Quantum Optics* (2008), §10.3;
Gambetta et al., *PRA* 74, 042318 (2006); Magesan & Gambetta, *PRA* 101,
052308 (2020).
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, replace
from functools import partial
from math import prod
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from quchip.approximations import Approximation, RWA, require_approximation
from quchip.backend import _backend_context
from quchip.backend._memory import require_memory
from quchip.backend.protocol import Backend, Operator
from quchip.control.drive import BaseDrive, CouplingDrive
from quchip.chip.effective import EffectiveTerms, retained_operator
from quchip.control.signal import AnalyticSignal, SignalKey
from quchip.declarative.expr import (
    PhysicsExpr,
    _ARRAY_LOWERER,
    _bound_values,
    as_operator_expr,
    declared_excitation_changes,
    is_energy_diagonal,
    materialize_array,
    materialize_expr,
    scalar_signal_program,
    split_dynamic_hamiltonian,
)
from quchip.engine.ir import HamiltonianTemplate
from quchip.engine.ir import (
    BoundCoherentInput,
    CanonicalOperator,
    Carrier,
    CollapseTerm,
    CoherentOp,
    Conjugate,
    Constant,
    DroppedTerm,
    DriveOp,
    DynamicTerm,
    EngineResult,
    HamiltonianProgram,
    ResolvedSLH,
    _ResolvedDressingContext,
    Multiply,
    ResolvedFrame,
    ScalarModulation,
    SignalProgram,
    Scale,
    StaticTerm,
    TermOrigin,
    _as_time_coefficient,
)
from quchip.engine.ir import simplify_signal as _simplify_signal
from quchip.engine.approximations import resolve_drive_program
from quchip.engine.reference import carrier_transfer, has_filter, time_shift
from quchip.engine.basis import (
    BasisRecord,
    resolve_local_basis,
)
from quchip.engine.solver_hints import _solver_hint_metadata, _static_spectral_span
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import (
    array_namespace,
    concrete_array_module,
    contains_tracer,
    maybe_concrete_scalar,
)
from quchip.engine.bands import (
    _BAND_NORM_RTOL,
    _decompose_product_canonical_bands,
    _sandwich_canonical,
    canonical_to_coo,
    decompose_canonical_bands,
    embed_on_support,
    prune_zero_diagonals,
)

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.engine.slh import CompiledNetwork
    from quchip.engine.ir import ControlOp


# -- Static Hamiltonian --------------------------------------------------


@dataclass(frozen=True)
class _SystemResolution:
    bases: dict[str, BasisRecord]
    hamiltonians: tuple[Operator, ...]
    dims: tuple[int, ...]
    network: CompiledNetwork | None
    space: Any = None

    @property
    def operator_dims(self):
        return self.dims if self.space is None else self.space.solver_dims

    def embed(self, backend, operator, support):
        if self.space is None:
            return _embed_product(backend, operator, support, self.dims)
        return self.space.embed(backend, operator, support)


def _support_semantic_transform(labels: Sequence[str], bases: Mapping[str, BasisRecord]) -> Any | None:
    """Return the energy-to-solver map of the devices ``labels`` in tensor order, or None for the identity."""
    transforms = [bases[label].energy_to_solver() for label in labels]
    if all(transform is None for transform in transforms):
        return None
    resolved = [
        jnp.eye(bases[label].resolved_dim, dtype=jnp.complex128) if transform is None else transform
        for label, transform in zip(labels, transforms, strict=True)
    ]
    product_transform = resolved[0]
    for transform in resolved[1:]:
        product_transform = jnp.kron(product_transform, transform)
    return product_transform


def _resolve_system(chip: "Chip", backend: Backend) -> _SystemResolution:
    bases: dict[str, BasisRecord] = {}
    hamiltonians: list[Operator] = []
    dims: list[int] = []
    for device in chip.devices:
        policy = chip.resolve_basis(device)
        levels = device.resolved_dimension(chip.basis) if policy == "eigen" else None
        # A constant local model stays concrete even when another component is traced.
        with jax.ensure_compile_time_eval():
            authored = device.unresolved_hamiltonian()
            matrix = materialize_array(authored)
            record = resolve_local_basis(matrix, basis=policy, levels=levels, authored_hamiltonian=authored)
        bases[device.label] = record
        dims.append(record.resolved_dim)
        if record.kind == "native":
            hamiltonians.append(materialize_expr(authored, backend))
        else:
            hamiltonians.append(
                backend.from_array(
                    record.transform_operator(matrix),
                    dims=[[record.resolved_dim], [record.resolved_dim]],
                )
            )
    return _SystemResolution(
        bases=bases,
        hamiltonians=tuple(hamiltonians),
        dims=tuple(dims),
        network=None if chip.port_network is None else chip.port_network._compile(),
    )


def _prepare_engine_assembly(
    chip: "Chip",
    frame_spec: Any,
    approximation: Approximation | None = None,
    *,
    resolution: _SystemResolution | None = None,
) -> tuple[_SystemResolution, ResolvedFrame]:
    """Resolve local bases once, then use that static contract to resolve the frame."""
    from quchip.engine.frames import resolve_frame, resolve_reference_frequencies

    resolution = _resolve_system(chip, chip.backend) if resolution is None else resolution
    references = resolve_reference_frequencies(chip, local_resolution=resolution)
    return resolution, resolve_frame(
        chip,
        frame_spec,
        reference_frequencies=references,
        local_resolution=resolution,
        approximation=approximation,
    )


def _retained_operator(chip: "Chip", operator: Any, support: tuple[int, ...], backend: Backend,
                       owner_key: str | None = None, *,
                       bases: Mapping[str, BasisRecord] | None = None) -> tuple[Any, tuple[int, ...]]:
    """Apply a retained model's captured coordinates to a surviving physical operator."""
    labels = tuple(chip.devices[i].label for i in support)
    projected = retained_operator(chip, operator, labels, backend, owner_key, bases=bases)
    if projected is None:
        return operator, support
    return projected, tuple(chip.device_index(label) for label in projected.labels)


def _project_on_support(
    chip: "Chip",
    operator: Any,
    support: tuple[int, ...],
    bases: Mapping[str, BasisRecord],
    backend: Backend,
) -> Operator:
    """Materialize and project a local operator into the resolved support basis.

    An empty ``support`` means the operator spans the full authored chip. A
    multi-device operator that is a sum of scalar-weighted single-device
    products projects factor by factor (see :func:`_project_products`) when
    that route needs less memory than the dense projection. Any other
    multi-device operator is projected on its dense native product space.
    Each route raises :class:`MemoryError` before an estimated peak that
    cannot fit (see :func:`_projection_peaks`).
    """
    if len(support) == 1:
        local = materialize_expr(operator, backend, local_bases=bases)
        record = bases[chip.devices[support[0]].label]
        if record.kind == "native":
            return local
        return backend.from_array(
            record.transform_operator(backend.to_array(local)),
            dims=[[record.resolved_dim], [record.resolved_dim]],
        )
    labels = tuple(chip.devices[index].label for index in support or range(len(chip.devices)))
    records = [bases[label] for label in labels]
    if all(record.kind == "native" for record in records):
        return materialize_expr(operator, backend, local_bases=bases)
    if not support:
        authored_dimension = prod(chip.authored_dims)
        shape = getattr(operator, "shape", None)
        if shape != (authored_dimension, authored_dimension):
            raise ValueError(
                "A support-free operator must span the full authored chip space; "
                f"expected {(authored_dimension, authored_dimension)}, got {shape}."
            )
    names = ", ".join(labels)
    factored, dense = _projection_peaks(operator, labels, bases)
    projected = None
    if factored is not None and factored <= dense:
        require_memory(
            factored,
            task=f"The factor-by-factor projection of an operator on {names}",
            remedy="Author the operator with fewer products of sums, or reduce the device cutoffs.",
        )
        projected = _project_products(operator, labels, bases)
    if projected is None:
        require_memory(
            dense,
            task=f"The dense projection of an operator on {names} at native dimension "
                 f"N = {prod(record.native_dim for record in records)}",
            remedy="Author the operator as a short sum of single-device products, or reduce the device cutoffs.",
        )
        projected = _project_dense(operator, labels, bases, backend)
    resolved_dims = [record.resolved_dim for record in records]
    return backend.from_array(projected, dims=[resolved_dims, resolved_dims])


# Bytes of one complex128 matrix entry.
_ENTRY_BYTES = 16
# Bytes of one stored sparse entry, a complex value and an index, counted in the
# native operator, its canonical copy, and one Kronecker temporary.
_SPARSE_ENTRY_BYTES = 3 * 24
# Bookkeeping bytes of one expanded product (its tuple, coefficient, dict and
# list slot) and of one NumPy array header, measured with tracemalloc.
_PRODUCT_BYTES = 512
_ARRAY_BYTES = 128
# Python objects and small temporaries of either route, about 0.1 MB in measurements.
_BOOKKEEPING_BYTES = 2**20


def _embed_product(backend: Backend, operator: Operator, support: tuple[int, ...], dims: tuple[int, ...]) -> Operator:
    """Embed a local operator in the product space after checking its storage and one state.

    Identity factors repeat each stored local entry once per spectator basis
    state. Supports of three or more devices embed through a dense array.
    """
    if support:
        dimension = prod(dims)
        if len(support) > 2:
            operator_bytes = 3 * _ENTRY_BYTES * dimension**2
        else:
            local = prod(dims[index] for index in support)
            matrix = backend.to_array(operator)
            stored = local**2 if contains_tracer(matrix) else int(np.count_nonzero(np.asarray(matrix)))
            operator_bytes = _SPARSE_ENTRY_BYTES * stored * (dimension // local)
        require_memory(
            operator_bytes + _ENTRY_BYTES * dimension,
            task=f"Product-space operator assembly at dimension {dimension}",
            remedy="Reduce the model, or use a model and initial state with a smaller proven invariant space.",
        )
    return embed_on_support(backend, operator, support, dims)


@dataclass(frozen=True)
class _Memory:
    """Bytes that lowering an expression node keeps in its value, and needs at most."""

    held: float
    peak: float


def _in_order(children: Sequence[_Memory], new: float, held: float) -> _Memory:
    """Memory of a node whose children lower in order and stay alive until it allocates ``new`` bytes."""
    peak = live = 0.0
    for child in children:
        peak = max(peak, live + child.peak)
        live += child.held
    return _Memory(held, max(peak, live + new))


def _dense_memory(node: Any, cache: dict[int, _Memory]) -> _Memory:
    """Memory of lowering ``node`` to a dense array with ``_ARRAY_LOWERER``.

    The cache keys nodes by identity, so a shared subexpression costs one visit.
    """
    if not isinstance(node, PhysicsExpr):
        return _Memory(0.0, 0.0)
    if id(node) not in cache:
        held = _ENTRY_BYTES * float(node.shape[0]) ** 2 if node.labels else 0.0
        if node.kind in ("add", "sub", "matmul", "tensor", "scale", "mul", "pow"):
            # Arithmetic allocates its result only.
            children, new = node.args, held
        elif node.kind in ("function", "embed"):
            # A callable or an embedding can also hold one temporary copy of its result.
            children, new = node.args, 2 * held
        else:
            # A leaf builds its matrix with at most one temporary copy.
            children, new = (), 2 * held
        cache[id(node)] = _in_order([_dense_memory(arg, cache) for arg in children], new, held)
    return cache[id(node)]


def _split_memory(
    node: Any,
    labels: tuple[str, ...],
    bases: Mapping[str, BasisRecord],
    cache: dict[int, tuple[float, _Memory] | None],
    dense: dict[int, _Memory],
) -> tuple[float, _Memory] | None:
    """Count the products that :func:`_project_products` splits ``node`` into, and estimate their memory.

    Returns ``None`` when a term does not factor. Sums and scalar factors reuse
    the matrices of their terms. A product allocates a dict for each pair of
    terms, and a matrix for each device that both terms act on.
    """
    if not isinstance(node, PhysicsExpr) or not node.labels or not set(node.labels) <= set(labels):
        return None
    if id(node) in cache:
        return cache[id(node)]
    split: tuple[float, _Memory] | None = None
    if len(node.labels) == 1:
        native = bases[node.labels[0]].native_dim
        if node.shape == (native, native):
            leaf = _dense_memory(node, dense)
            split = 1.0, _in_order([leaf], _PRODUCT_BYTES + _ARRAY_BYTES, leaf.held + _PRODUCT_BYTES + _ARRAY_BYTES)
    elif node.kind == "embed":
        split = _split_memory(node.args[0], labels, bases, cache, dense)
    elif node.kind == "scale":
        operand = _split_memory(node.args[1], labels, bases, cache, dense)
        if operand is not None:
            products, memory = operand
            scalar = _dense_memory(node.args[0], dense)
            split = products, _in_order([scalar, memory], products * _PRODUCT_BYTES, memory.held)
    elif node.kind in ("add", "sub", "tensor", "matmul"):
        left = _split_memory(node.args[0], labels, bases, cache, dense)
        right = _split_memory(node.args[1], labels, bases, cache, dense)
        if left is not None and right is not None:
            children = [left[1], right[1]]
            held = left[1].held + right[1].held
            if node.kind in ("add", "sub"):
                products = left[0] + right[0]
                split = products, _in_order(children, products * _PRODUCT_BYTES, held)
            else:
                products = left[0] * right[0]
                shared = set(node.args[0].labels) & set(node.args[1].labels)
                matrices = sum(_ENTRY_BYTES * bases[label].native_dim ** 2 + _ARRAY_BYTES for label in shared)
                new = products * (_PRODUCT_BYTES + matrices)
                split = products, _in_order(children, new, held + new)
    cache[id(node)] = split
    return split


def _projection_peaks(
    operator: Any,
    labels: tuple[str, ...],
    bases: Mapping[str, BasisRecord],
) -> tuple[float | None, float]:
    """Estimate the peak bytes of the factor-by-factor and of the dense projection of ``operator``.

    Both estimates follow the depth-first lowering of the expression, in which
    a node keeps its lowered arguments until it combines them. The
    factor-by-factor estimate is ``None`` when a term does not factor. The
    estimates cover concrete values. Reverse-mode differentiation also keeps
    intermediate values for the backward pass.
    """
    records = [bases[label] for label in labels]
    native = float(prod(record.native_dim for record in records))
    resolved = float(prod(record.resolved_dim for record in records))
    dense_cache: dict[int, _Memory] = {}
    # The projection holds the dense operator, the N×R transform, its adjoint,
    # one N×R product and the R×R result.
    dense = _BOOKKEEPING_BYTES + max(
        _dense_memory(operator, dense_cache).peak,
        _ENTRY_BYTES * (native**2 + 3 * native * resolved + resolved**2),
    )
    if not isinstance(operator, PhysicsExpr) or operator.labels != labels:
        return None, dense
    split = _split_memory(operator, labels, bases, {}, dense_cache)
    if split is None:
        return None, dense
    # The sum holds the products, the running and the new total, one Kronecker
    # product and its scaled copy, and the projection temporaries of each device.
    local = sum(2 * record.native_dim * record.resolved_dim + record.resolved_dim**2 for record in records)
    memory = split[1]
    return _BOOKKEEPING_BYTES + max(memory.peak, memory.held + _ENTRY_BYTES * (4 * resolved**2 + local)), dense


def _local_products(operator: PhysicsExpr, labels: tuple[str, ...], bases: Mapping[str, BasisRecord]):
    """Factor authored algebra before choosing product or compact coordinates."""
    values = _bound_values(operator)

    def lower(node: PhysicsExpr) -> Any:
        # Concrete values stay in NumPy, so they compile no JAX program per array shape.
        value = materialize_expr(node, _ARRAY_LOWERER, bindings=values, local_bases=bases)
        return concrete_array_module(value).asarray(value)

    def split(node: Any) -> list[tuple[Any, dict[str, Any]]] | None:
        if not isinstance(node, PhysicsExpr) or not node.labels or not set(node.labels) <= set(labels):
            return None
        if len(node.labels) == 1:
            native = bases[node.labels[0]].native_dim
            matrix = lower(node)
            return [(1.0, {node.labels[0]: matrix})] if matrix.shape == (native, native) else None
        if node.kind == "embed":
            return split(node.args[0])
        if node.kind == "scale":
            scalar = lower(node.args[0])
            terms = split(node.args[1])
            if terms is None or scalar.ndim != 0:
                return None
            return [(scalar * coefficient, factors) for coefficient, factors in terms]
        if node.kind not in ("add", "sub", "tensor", "matmul"):
            return None
        left, right = split(node.args[0]), split(node.args[1])
        if left is None or right is None:
            return None
        if node.kind == "add":
            return left + right
        if node.kind == "sub":
            return left + [(-coefficient, factors) for coefficient, factors in right]
        products: list[tuple[Any, dict[str, Any]]] = []
        for left_coefficient, left_factors in left:
            for right_coefficient, right_factors in right:
                factors = dict(left_factors)
                for label, matrix in right_factors.items():
                    factors[label] = factors[label] @ matrix if label in factors else matrix
                products.append((left_coefficient * right_coefficient, factors))
        return products

    return split(operator)


def _project_products(
    operator: PhysicsExpr,
    labels: tuple[str, ...],
    bases: Mapping[str, BasisRecord],
) -> Any | None:
    """Project a sum of scalar-weighted single-device products factor by factor.

    The support transform is the Kronecker product of the local transforms, so
    ``V†(A ⊗ B)V = (V_a† A V_a) ⊗ (V_b† B V_b)``, and memory scales with each
    native dimension squared, times the number of products. Products on one
    device multiply in the native basis before projection, because
    ``V_a V_a†`` is not the identity on a truncated eigenbasis. A label absent
    from a product is the identity. Returns ``None`` when a term does not
    factor.
    """
    terms = _local_products(operator, labels, bases)
    if terms is None:
        return None
    xp = concrete_array_module(terms, [bases[label].vectors for label in labels])
    total: Any = 0.0
    for coefficient, factors in terms:
        product = xp.ones((1, 1), dtype=complex)
        for label in labels:
            record = bases[label]
            if label not in factors:
                local = xp.eye(record.resolved_dim, dtype=complex)
            elif record.kind == "native":
                local = xp.asarray(factors[label])
            else:
                vectors = xp.asarray(record.vectors)
                local = vectors.conj().T @ xp.asarray(factors[label]) @ vectors
            product = xp.kron(product, local)
        total = total + coefficient * product
    return total


def _project_dense(
    operator: Any,
    labels: tuple[str, ...],
    bases: Mapping[str, BasisRecord],
    backend: Backend,
) -> Any:
    """Project an operator through its dense native product space.

    Expressions lower to plain arrays, whose allocations follow the estimate of
    :func:`_projection_peaks`.
    """
    if isinstance(operator, PhysicsExpr):
        matrix = materialize_expr(operator, _ARRAY_LOWERER, local_bases=bases)
    else:
        matrix = backend.to_array(operator)
    records = [bases[label] for label in labels]
    xp = array_namespace(records[0].vectors)
    transform = records[0].vectors
    for record in records[1:]:
        transform = xp.kron(transform, record.vectors)
    return transform.conj().T @ matrix @ transform


def _build_static_h0(
    chip: "Chip",
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    resolution: _SystemResolution,
    static_couplings: list[Operator],
) -> Operator:
    """Build the frame-subtracted static Hamiltonian in angular units.

    .. math::
        H_0 \\;=\\; 2\\pi \\left( H_{\\text{bare}}
                 - \\sum_i \\omega^{\\text{ref}}_i \\, n_i \\right)

    where ``H_bare`` is the chip-level tensored sum of device and static
    coupling contributions (ordinary GHz), and ``ω^ref_i`` is the
    per-device frame reference. This function is one of the
    assembly paths where the 2π boundary is crossed.
    """
    bare = [resolution.embed(backend, local, (idx,)) for idx, local in enumerate(resolution.hamiltonians)]
    bare += static_couplings
    if not bare:
        raise ValueError("A chip must contain at least one device.")
    terms = [(TWO_PI, backend.linear_combination([(1, op) for op in bare]))]
    for idx, dev in enumerate(chip.devices):
        omega_ref = resolved_frame.frequencies.get(dev.label, 0.0)
        concrete_omega = maybe_concrete_scalar(omega_ref)
        if concrete_omega is not None and concrete_omega == 0.0:
            continue
        with _backend_context(backend):
            record = resolution.bases[dev.label]
            level_operator = _resolved_frame_operator(record, backend)
            n_emb = resolution.embed(backend, level_operator, (idx,))
        terms.append((-(TWO_PI * omega_ref), n_emb))
    return backend.linear_combination(terms)


def _resolved_frame_operator(record: BasisRecord, backend: Backend) -> Operator:
    """Return one device's frame generator in its resolved solver basis."""
    if record.energy_to_solver() is None:
        return backend.number(record.resolved_dim)
    return backend.from_array(
        record.level_operator(),
        dims=[[record.resolved_dim], [record.resolved_dim]],
    )


# -- Dynamic-term assembly helpers ---------------------------------------
#
# Every dynamic operator that enters the solver-facing Hamiltonian passes
# through ``_apply_2pi_canonical``: it is the *single* place the ``2π``
# (angular-frequency) boundary is crossed for time-dependent terms.
# ``bands.py`` never sees ``2π``; lab-frame
# operators arrive here in ordinary GHz and leave canonicalized in angular
# units, tagged for diagnostics.


def _apply_2pi_canonical(backend: Backend, embedded: Operator, *, dims, labels, tag: str) -> CanonicalOperator:
    """Apply the ``2π`` boundary to an embedded lab-frame operator and canonicalize."""
    return backend.to_canonical_operator(backend.linear_combination([(TWO_PI, embedded)])).with_metadata(
        dims=dims,
        subsystem_labels=labels,
        tag=tag,
    )


def _apply_2pi_scalar(value: Any) -> Any:
    """Cross the Hamiltonian ordinary-to-angular frequency boundary."""
    return TWO_PI * value


def _collect_collapse_terms(
    chip: "Chip",
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    resolution: _SystemResolution,
) -> tuple[tuple[CollapseTerm, ...], tuple[tuple[str, ...], ...]]:
    """Canonicalize every collapse channel and attach accessible-port metadata."""
    labels = tuple(device.label for device in chip.devices)
    terms: list[CollapseTerm] = []
    supports: list[tuple[str, ...]] = []
    ports_by_id = {id(port): port for port in chip.ports}
    port_counts = dict.fromkeys(ports_by_id, 0)
    with _backend_context(backend):
        for (
            operator,
            rate,
            support,
            source,
            channel,
            parameter_paths,
            owner,
        ) in chip._collapse_contributions_with_owners(resolution.bases):
            compact = (None if resolution.space is None else
                       resolution.space.authored(backend, operator, support))
            local = (None if compact is not None else
                     _project_on_support(chip, operator, support, resolution.bases, backend))
            resolved_rate = materialize_expr(rate, backend)
            retained_coordinates = isinstance(owner, EffectiveTerms) or any(
                terms.projection is not None for terms in chip.effective_terms
            )
            # A static collapse operator is only frame-invariant up to one global
            # phase; multi-device channels (collective baths, custom couplings) can
            # beat between unequal device frames, so they get the same check as
            # retained-coordinate effective channels.
            # Support-free channels (baths) already span the whole chip.
            check_support = support or tuple(range(len(labels)))
            if (retained_coordinates or len(check_support) > 1) and not _is_concrete_zero(resolved_rate):
                frequencies = tuple(resolved_frame.frequencies[labels[index]] for index in check_support)
                if not all(maybe_concrete_scalar(frequency) == 0.0 for frequency in frequencies):
                    from quchip.chip.effective import authored_excitation_changes
                    weights = authored_excitation_changes(operator, tuple(labels[i] for i in check_support),
                                                          backend, resolution.bases)
                    common = all(maybe_concrete_scalar(f) == maybe_concrete_scalar(frequencies[0])
                                 and maybe_concrete_scalar(f) is not None for f in frequencies)
                    frame_weights: tuple[tuple[int, ...], ...]
                    # One total change has one phase in a common frame. A larger structural
                    # bound needs the decomposition, which drops bands that are exactly zero.
                    if common and weights is not None and len(weights) <= 1:
                        frame_weights = ((next(iter(weights), 0),),)
                        frequencies = (frequencies[0],)
                    else:
                        if local is None:
                            local = _project_on_support(chip, operator, support, resolution.bases, backend)
                        frame_weights = tuple(_authored_bands(
                            chip, operator, local, check_support, resolution.bases, backend))
                    kind = "Effective channel" if retained_coordinates else "Collapse channel"
                    _common_band_frame_frequency(frame_weights, frequencies,
                                                 source=f"{kind} {source!r}/{channel!r}")
            if len(support) > 1 and not _is_concrete_zero(resolved_rate):
                resolved_support = tuple(labels[index] for index in support)
                if resolved_support not in supports:
                    supports.append(resolved_support)
            embedded = compact if compact is not None else resolution.embed(backend, local, support)
            canonical = backend.to_canonical_operator(embedded).with_metadata(
                dims=resolution.operator_dims,
                subsystem_labels=labels if resolution.space is None else ("excitation",),
                tag=f"collapse:{source}:{channel}",
            )
            port = ports_by_id.get(id(owner))
            term = CollapseTerm(
                operator=canonical,
                rate=resolved_rate,
                source=source,
                channel=channel,
                parameter_paths=parameter_paths,
                phase=port.phase if port is not None else None,
                frame_frequency=(
                    resolved_frame.frequencies[port.resolve_targets(chip)[0]]
                    if port is not None and port.operator is None
                    else _port_frame_frequency(chip, port, resolved_frame, backend, resolution)
                    if port is not None
                    else None
                ),
            )
            terms.append(term)
            if port is not None:
                port_counts[id(port)] += 1
    for port in chip.ports:
        if port_counts[id(port)] != 1:
            raise RuntimeError(f"Port {port.label!r} must resolve to exactly one collapse term.")
    return tuple(terms), tuple(supports)


def _compact_network_hamiltonian(chip: "Chip", resolution: _SystemResolution, backend: Backend) -> Any | None:
    """Return the Hamiltonian of cascaded port pairs on a compact space, in angular units.

    Products of port operators that are already restricted to the space drop
    each intermediate state outside it. This function forms every product on
    its ports' joint support first, so the result is ``P H_gen P``, as CSR on a
    NumPy backend. It returns ``None`` for the product space or a network
    without generated pairs, and a zero matrix when every local term is below
    the round-off threshold.
    """
    from quchip.chip.ports import Port
    from quchip.engine.sectors import generated_products

    pairs = () if resolution.network is None else resolution.network.generated_pairs
    if resolution.space is None or not pairs:
        return None
    wanted = {label for downstream, upstream, _ in pairs for label in (downstream, upstream)}
    ports: dict[str, tuple[Any, tuple[int, ...]]] = {}
    with _backend_context(backend):
        for operator, rate, support, *_, owner in chip._collapse_contributions_with_owners(resolution.bases):
            if not isinstance(owner, Port) or owner.label not in wanted:
                continue
            local = backend.to_array(_project_on_support(chip, operator, support, resolution.bases, backend))
            resolved_rate = materialize_expr(rate, backend)
            xp = concrete_array_module(local, resolved_rate, owner.phase)
            scale = xp.sqrt(xp.asarray(resolved_rate))
            if owner.phase is not None:
                scale = xp.exp(1j * xp.asarray(owner.phase)) * scale
            ports[owner.label] = (scale * xp.asarray(local), support)
    entries = []
    for term, joint in generated_products(ports, pairs, resolution.dims):
        local_dims = [resolution.dims[index] for index in joint]
        local = backend.from_array(term, dims=[local_dims, local_dims])
        entries.append(resolution.space.entries(backend, local, joint))
    size = resolution.space.size
    # Products of different pairs can share entries, which the sum adds.
    if backend.array_module is np:
        return sum((sparse.csr_matrix((values, (rows, columns)), shape=(size, size))
                    for rows, columns, values in entries), start=sparse.csr_matrix((size, size), dtype=complex))
    xp = concrete_array_module(*(values for *_, values in entries))
    total = xp.zeros((size, size), dtype=complex)
    for rows, columns, values in entries:
        if xp is np:
            np.add.at(total, (rows, columns), values)
        else:
            total = total.at[rows, columns].add(values)
    return total


def _concrete_port_resolution(
    chip: "Chip",
    port: Any,
    backend: Backend,
    resolution: _SystemResolution,
) -> CanonicalOperator:
    """Materialize one explicit port in its semantic product basis."""
    labels = port.resolve_targets(chip)
    support = tuple(chip._label_to_index[label] for label in labels)
    records = resolution.bases
    traced = contains_tracer(
        tuple(value for record in records.values() for value in (record.vectors, record.energy_vectors))
    )
    with jax.ensure_compile_time_eval(), backend.eager_operators(), _backend_context(backend):
        if traced:
            concrete_records: dict[str, BasisRecord] = {}
            for device in chip.devices:
                authored = device.unresolved_hamiltonian()
                matrix = materialize_array(authored)
                policy = chip.resolve_basis(device)
                levels = device.resolved_dimension(chip.basis) if policy == "eigen" else None
                concrete_records[device.label] = resolve_local_basis(
                    matrix,
                    basis=policy,
                    levels=levels,
                    authored_hamiltonian=authored,
                )
            records = concrete_records
        operator = _project_on_support(
            chip,
            port._authored_operator(chip),
            support,
            records,
            backend,
        )
        payload = backend.to_array(operator)
        transform = _support_semantic_transform(labels, records)
        # A port transformed inside jit or grad keeps a traced payload. Its
        # band decomposition then keeps every candidate the declared
        # excitation changes allow, so it is not pruned here.
        xp = jnp if contains_tracer(payload) else np
        values = xp.asarray(payload, dtype=complex)
        if transform is not None:
            concrete_transform = np.asarray(transform, dtype=complex)
            values = concrete_transform.conj().T @ values @ concrete_transform
    if xp is np:
        norm = float(np.linalg.norm(values))
        if norm > 0.0:
            values = np.where(np.abs(values) > 1e-10 * norm, values, 0.0)
    return CanonicalOperator.from_dense(
        values,
        dims=tuple(records[label].resolved_dim for label in labels),
        basis="semantic",
        subsystem_labels=labels,
        tag=f"port:{port.label}",
    )


def _port_bands(
    chip: "Chip", port: Any, backend: Backend, resolution: _SystemResolution
) -> dict[tuple[int, ...], CanonicalOperator]:
    """Excitation-change bands of one explicit port operator in its semantic product basis."""
    canonical = _concrete_port_resolution(chip, port, backend, resolution)
    return _decompose_product_canonical_bands(
        canonical, canonical.dims, total_changes=declared_excitation_changes(port.operator),
    )


def _frequency_groups(frequencies: tuple[Any, ...]) -> tuple[tuple[int, ...], ...]:
    """Group frame frequencies whose equality is safe to establish in Python."""
    groups: list[list[int]] = []
    for index, frequency in enumerate(frequencies):
        for group in groups:
            representative = frequencies[group[0]]
            first = maybe_concrete_scalar(frequency)
            second = maybe_concrete_scalar(representative)
            if frequency is representative or (
                first is not None and second is not None and first == second
            ):
                group.append(index)
                break
        else:
            groups.append([index])
    return tuple(tuple(group) for group in groups)


def _band_frame_frequency(weights: Sequence[int], frequencies: Sequence[Any]) -> Any:
    """Return a band's frame frequency, the sum of each level change times its device's frame frequency.

    Changes on devices that share a frame frequency are added as integers
    first, so a band whose changes cancel there is exactly static rather than
    carrying a round-off frequency.
    """
    frequencies = tuple(frequencies)
    total: Any = 0.0
    for group in _frequency_groups(frequencies):
        weight = sum(int(weights[index]) for index in group)
        if weight:
            total = total + weight * frequencies[group[0]]
    return total


def _port_frame_frequency(
    chip: "Chip",
    port: Any,
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    resolution: _SystemResolution,
) -> Any:
    """Return the common frame phase of every band in one explicit port."""
    labels = port.resolve_targets(chip)
    frequencies = tuple(resolved_frame.frequencies[label] for label in labels)
    if all(maybe_concrete_scalar(frequency) == 0.0 for frequency in frequencies):
        return 0.0
    bands = _port_bands(chip, port, backend, resolution)
    if not bands:
        raise ValueError(f"Port {port.label!r} requires a nonzero coupling operator.")

    return _common_band_frame_frequency(tuple(bands), frequencies, source=f"Port {port.label!r}")


def _common_band_frame_frequency(bands: tuple[Any, ...], frequencies: tuple[Any, ...], *, source: str) -> Any:
    """Require one removable global phase for a static collapse operator."""
    if not bands:
        return 0.0
    groups = _frequency_groups(frequencies)
    signatures: list[tuple[float, tuple[float, ...]]] = []
    traced_groups: list[tuple[tuple[int, ...], Any]] = []
    for group in groups:
        representative = frequencies[group[0]]
        concrete = maybe_concrete_scalar(representative)
        if concrete is None:
            traced_groups.append((group, representative))
    for band in bands:
        constant = 0.0
        traced_coefficients: list[float] = []
        for group in groups:
            coefficient = float(sum(band[index] for index in group))
            representative = frequencies[group[0]]
            concrete = maybe_concrete_scalar(representative)
            if concrete is None:
                traced_coefficients.append(coefficient)
            else:
                constant += coefficient * concrete
        signatures.append((constant, tuple(traced_coefficients)))

    reference = signatures[0]
    if any(
        not np.isclose(constant, reference[0], atol=1e-10, rtol=1e-10)
        or not np.allclose(coefficients, reference[1], atol=1e-10, rtol=1e-10)
        for constant, coefficients in signatures[1:]
    ):
        raise ValueError(
            f"{source} has operator bands with different phases in the selected frame. "
            "Use a common frame compatible with the operator, or the lab frame; "
            "time-dependent collapse operators are not supported."
        )
    frequency: Any = reference[0]
    has_term = frequency != 0.0
    for coefficient, (_, representative) in zip(reference[1], traced_groups, strict=True):
        if coefficient == 0.0:
            continue
        term = representative if coefficient == 1.0 else coefficient * representative
        frequency = frequency + term if has_term else term
        has_term = True
    return frequency


def _dynamic_term(
    backend: Backend,
    embedded: Operator,
    *,
    dims,
    labels,
    tag: str,
    origin: TermOrigin,
    time_dependence: ScalarModulation,
) -> DynamicTerm:
    """Wrap an embedded lab-frame operator as a ``2π``-scaled :class:`DynamicTerm`."""
    return DynamicTerm(
        operator=_apply_2pi_canonical(backend, embedded, dims=dims, labels=labels, tag=tag),
        time_dependence=time_dependence,
        origin=origin,
    )


# -- Coupling terms ------------------------------------------------------


@dataclass(frozen=True)
class BandRecord:
    """One excitation-change band of an authored operator, with its charge vector.

    ``charges[i]`` is the photon-number change the band applies to
    ``devices[i]``. ``amplitude`` is the band's largest concrete matrix element
    or ``None`` when traced.
    """
    # Frame planning and assembly read the same band records.

    devices: tuple[str, ...]
    charges: tuple[int, ...]
    amplitude: float | None
    source: str
    retained: bool = False


def _concrete_amplitude(values: Any) -> float | None:
    if contains_tracer(values):
        return None
    try:
        return float(np.max(np.abs(np.asarray(values))))
    except (TypeError, ValueError):
        return None


def _authored_bands(
    chip: "Chip",
    authored: Any,
    operator: Operator,
    support: tuple[int, ...],
    bases: Mapping[str, BasisRecord],
    backend: Backend,
    *,
    tag: str | None = None,
) -> dict[tuple[int, ...], CanonicalOperator]:
    """Resolve projected operator bands, retaining authored energy-diagonal structure."""
    labels = tuple(chip.devices[index].label for index in support)
    dims = tuple(bases[label].resolved_dim for label in labels)
    canonical = backend.to_canonical_operator(operator).with_metadata(dims=dims, subsystem_labels=labels, tag=tag)
    if is_energy_diagonal(authored, bases):
        return {(0,) * len(support): canonical}
    transform = _support_semantic_transform(labels, bases)
    changes = declared_excitation_changes(authored)
    if len(support) == 1:
        return {(weight,): band for weight, band in decompose_canonical_bands(
            canonical, dims[0], semantic_to_solver=transform, total_changes=changes,
        ).items()}
    return _decompose_product_canonical_bands(
        canonical, dims, semantic_to_solver=transform, total_changes=changes,
    )


@dataclass(frozen=True)
class _Interaction:
    label: str
    labels: tuple[str, ...]
    expression: Callable[[], Any]
    retained: bool = False

    def keeps_band(self, approximation: Approximation, weights: tuple[int, ...]) -> bool:
        """Apply the static band policy, with retained contributions exempt from filtering."""
        return self.retained or not approximation.filters_terms or approximation.keeps_operator_band(weights)


def _authored_interaction(coupling: Any, backend: Backend) -> Any:
    """Read a coupling's authored operator with the chip's backend as the default."""
    with _backend_context(backend):
        return coupling.interaction_hamiltonian()


def _interaction_contributions(chip: "Chip", effective_terms: Sequence[EffectiveTerms] | None = None):
    for coupling in chip.couplings:
        yield _Interaction(coupling.label, (coupling.device_a_label, coupling.device_b_label),
                           partial(_authored_interaction, coupling, chip.backend))
    for terms in chip.effective_terms if effective_terms is None else effective_terms:
        yield _Interaction(terms.label, terms.labels, terms.expression, retained=True)


@dataclass(frozen=True)
class _ResolvedInteraction:
    contribution: _Interaction
    support: tuple[int, ...]
    authored: Any
    operator: Any


def _resolved_interactions(chip: "Chip", resolution: _SystemResolution, backend: Backend,
                           effective_terms: Sequence[EffectiveTerms] | None = None):
    """Project local interactions once, before choosing a global or sector representation."""
    for contribution in _interaction_contributions(chip, effective_terms):
        support = tuple(chip.device_index(label) for label in contribution.labels)
        authored = contribution.expression()
        operator = _project_on_support(chip, authored, support, resolution.bases, backend)
        yield _ResolvedInteraction(contribution, support, authored, operator)


def _energy_operator(operator: Any, labels: Sequence[str], bases: Mapping[str, BasisRecord], backend: Backend) -> Any:
    """Express a projected operator on the devices ``labels``, in tensor order, in isolated energy coordinates."""
    matrix = backend.to_array(operator)
    transform = _support_semantic_transform(labels, bases)
    if transform is not None:
        xp = concrete_array_module(matrix, transform)
        transform = xp.asarray(transform)
        matrix = transform.conj().T @ xp.asarray(matrix) @ transform
    return matrix if contains_tracer(matrix) else np.asarray(matrix, dtype=complex)


def _resolved_energy_interactions(chip: "Chip", resolution: _SystemResolution, approximation: Approximation,
                                  advisories: list[str] | None = None,
                                  effective_terms: Sequence[EffectiveTerms] | None = None):
    """Yield kept static operators on their local supports in energy coordinates, in GHz.

    ``effective_terms`` selects the retained contributions. ``None`` reads all of the chip's.
    Approximation advisories go to ``advisories`` when it is given, and to ``warnings.warn`` otherwise.
    """
    for resolved in _resolved_interactions(chip, resolution, chip.backend, effective_terms):
        contribution, support = resolved.contribution, resolved.support
        operator = _energy_operator(resolved.operator, contribution.labels, resolution.bases, chip.backend)
        if approximation.filters_terms and not contribution.retained:
            dims = tuple(resolution.dims[index] for index in support)
            dimension = prod(dims)
            require_memory((16 + 24 * len(dims)) * dimension**2,
                           task=f"Static band mask on {contribution.labels}",
                           remedy="Use interactions on smaller local supports.")
            levels = np.indices(dims).reshape(len(dims), -1).T
            weights = (levels[None, :, :] - levels[:, None, :]).reshape(-1, len(dims))
            bands, inverse = np.unique(weights, axis=0, return_inverse=True)
            kept = np.array([contribution.keeps_band(approximation, tuple(int(w) for w in band)) for band in bands])
            filtered = operator * kept[inverse.reshape(-1)].reshape(dimension, dimension)
            if not contains_tracer(operator):
                norm = np.linalg.norm(operator)
                if norm > 0.0 and np.linalg.norm(filtered) <= 1e-12 * norm:
                    _report_advisory(f"Coupling {contribution.label!r} vanishes entirely under RWA().", advisories)
                elif (
                    norm > 0.0
                    and not is_energy_diagonal(resolved.authored, resolution.bases)
                    and not contains_tracer(tuple(resolution.bases[label].energies for label in contribution.labels))
                ):
                    # The diagnostic holds indices, levels, values and detunings of each nonzero entry.
                    require_memory((64 + 40 * len(dims)) * int(np.count_nonzero(operator)),
                                   task=f"Static band diagnostics on {contribution.labels}",
                                   remedy="Use interactions on smaller local supports.")
                    rows, cols = np.nonzero(operator)
                    energies = tuple(resolution.bases[label].energies for label in contribution.labels)
                    _report_advisory(_dropped_band_advisory(contribution.label, [
                        (band, contribution.keeps_band(approximation, band), score)
                        for band, (_, _, score) in _band_diagnostics(
                            rows, cols, operator[rows, cols], dims, energies).items()
                    ]), advisories)
            operator = filtered
        yield operator, support


def _may_raise_ground(chip: "Chip", approximation: Approximation, bases: Mapping[str, BasisRecord]) -> bool:
    """Return whether a retained interaction band can move the bare ground product.

    Band weights are energy-basis column minus row on each endpoint, so a band
    reaches that product only when every weight is nonpositive and at least
    one is negative. Retained contributions bypass the approximation filter.
    """
    for contribution in _interaction_contributions(chip):
        if contribution.retained:
            return True
        levels = (range(bases[label].resolved_dim) for label in contribution.labels)
        if any(
            any(raised) and approximation.keeps_operator_band(tuple(-level for level in raised))
            for raised in itertools.product(*levels)
        ):
            return True
    return False


def coupling_band_records(chip: "Chip", resolution: _SystemResolution, backend: Backend) -> list[BandRecord]:
    """Return nonzero interaction bands for frame planning."""
    records: list[BandRecord] = []
    for resolved in _resolved_interactions(chip, resolution, backend):
        contribution, support = resolved.contribution, resolved.support
        expression, h_full = resolved.authored, resolved.operator
        if _is_concrete_zero_array(backend.to_array(h_full)):
            continue
        bands = _authored_bands(
            chip, expression, h_full, support, resolution.bases, backend, tag="coupling_local",
        )
        for charges, band in bands.items():
            records.append(BandRecord(contribution.labels, charges, _concrete_amplitude(band.values),
                                      contribution.label, contribution.retained))
    return records


def port_band_records(chip: "Chip", port: Any, resolution: _SystemResolution, backend: Backend) -> list[BandRecord]:
    """Return the excitation-change bands of one quantum port's coupling ``sqrt(kappa) A``."""
    labels = tuple(port.resolve_targets(chip))
    rate = maybe_concrete_scalar(port.rate_value(chip))
    scale = None if rate is None else float(np.sqrt(abs(rate)))
    if port.operator is None:
        # The default lowering operator has weight column minus row = +1.
        return [BandRecord(labels, (1,), scale, port.label)]
    records: list[BandRecord] = []
    for charges, band in _port_bands(chip, port, backend, resolution).items():
        amplitude = _concrete_amplitude(band.values)
        scaled = None if scale is None or amplitude is None else scale * amplitude
        records.append(BandRecord(labels, tuple(int(c) for c in charges), scaled, port.label))
    return records


def _keeps_drive_band(drive: BaseDrive, approximation: Approximation, charges: tuple[int, ...]) -> bool:
    """Single-device drive bands survive unless an explicit ``keep_bands`` excludes them; pumps follow the strategy."""
    if isinstance(drive, CouplingDrive) or (isinstance(approximation, RWA) and approximation.keep_bands is not None):
        return approximation.keeps_operator_band(charges)
    return True


def _transition_diagnostic(values: Any, detunings: Any) -> tuple[Any, Any, Any]:
    """Select the entry with maximal perturbative scale, treating resonance explicitly."""
    xp = concrete_array_module(values, detunings)
    amplitudes = xp.abs(xp.asarray(values))
    scores = xp.where(
        amplitudes == 0, 0.0,
        xp.where(detunings == 0, xp.inf, amplitudes**2 / xp.where(detunings == 0, 1.0, detunings)),
    )
    largest = xp.argmax(scores)
    return amplitudes[largest], detunings[largest], scores[largest]


def _band_diagnostics(
    rows: Any,
    cols: Any,
    values: Any,
    dims: tuple[int, ...],
    energies: tuple[Any, ...],
) -> dict[tuple[int, ...], tuple[Any, Any, Any]]:
    """Return the entry with maximal |V|²/|ΔE| in each band of support-local COO entries.

    The entries use isolated energy coordinates, and band weights are column
    minus row. A nonzero resonant entry has an infinite score. A zero entry
    has score zero, including at resonance. Like band decomposition, this
    skips concrete bands at roundoff scale. Selection stays piecewise
    differentiable.
    """
    if not np.size(rows):
        return {}
    row_levels = np.stack(np.unravel_index(rows, dims), axis=-1)
    col_levels = np.stack(np.unravel_index(cols, dims), axis=-1)
    bands, inverse = np.unique(col_levels - row_levels, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)  # NumPy 2.0.0 returns a column for axis=0.
    xp = concrete_array_module(values, energies)
    values = xp.asarray(values)
    detunings = xp.abs(sum(
        xp.asarray(spectrum)[col_levels[:, axis]] - xp.asarray(spectrum)[row_levels[:, axis]]
        for axis, spectrum in enumerate(energies)
    ))
    norm = None if contains_tracer(values) else np.linalg.norm(values)
    order = np.argsort(inverse, kind="stable")
    diagnostics = {}
    for band, group in zip(bands, np.split(order, np.cumsum(np.bincount(inverse))[:-1]), strict=True):
        if norm is not None and np.linalg.norm(values[group]) <= _BAND_NORM_RTOL * norm:
            continue
        diagnostics[tuple(int(weight) for weight in band)] = _transition_diagnostic(values[group], detunings[group])
    return diagnostics


def _dropped_band_advisory(
    source: str,
    diagnostics: list[tuple[tuple[int, ...], bool, Any]],
) -> str | None:
    """Return the advisory for an interaction whose concrete discarded shift dominates, or ``None``."""
    scores = [maybe_concrete_scalar(score) for _, _, score in diagnostics]
    if not scores or any(score is None for score in scores):
        return None
    concrete = [(weights, kept, cast(float, score)) for (weights, kept, _), score in zip(diagnostics, scores)]
    dropped = [(weights, score) for weights, kept, score in concrete if not kept]
    if not dropped:
        return None
    weights, score = max(dropped, key=lambda item: item[1])
    kept_score = max((value for band, kept, value in concrete if kept and any(band)), default=0.0)
    if score > 0 and (np.isinf(score) or score > kept_score):
        pair = sorted({weights, tuple(-weight for weight in weights)})
        detail = "a nonzero resonant entry" if np.isinf(score) else "a larger entrywise |V|²/ΔE than the retained bands"
        return (
            f"RWA() drops band(s) {', '.join(map(str, pair))} on {source!r} with {detail}. "
            "This bare-energy diagnostic is a heuristic, not an error bound. "
            "Compare with Exact() or list every band to keep in RWA(keep_bands=...)."
        )
    return None


def _report_advisory(message: str | None, advisories: list[str] | None) -> None:
    """Warn with an approximation advisory, or append it to ``advisories`` for a cached analysis entry."""
    if message is None:
        return
    if advisories is None:
        warnings.warn(message, UserWarning, stacklevel=4)
    else:
        advisories.append(message)


def _resolve_coupling_terms(
    chip: "Chip",
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    resolution: _SystemResolution,
    approximation: Approximation,
    advisories: list[str] | None = None,
) -> tuple[
    list[Operator],
    list[Operator],
    list[tuple[Operator, ScalarModulation]],
    list[DroppedTerm],
    list[tuple[str, ...]],
]:
    """Project and band-resolve every coupling once.

    Returns the static interaction included in ``H₀``, frame corrections,
    dynamic carrier terms, and dropped-band records. Under ``RWA()``,
    rejected bands are omitted from the static interaction. Retained bands
    whose frame carrier
    ``Δa·ω_a + Δb·ω_b`` is *concretely* zero are already static inside
    ``H₀`` and stay there. Every other retained band is subtracted from
    ``H₀`` and re-attached with its carrier

    .. math::
        \\exp\\!\\left(-i\\,(\\Delta_a \\omega_a + \\Delta_b \\omega_b)\\,t\\right),

    the standard rotating-frame interaction picture for a bilinear
    coupling (see e.g. Gambetta et al., *PRA* **74**, 042318 (2006)).
    Traced carrier frequencies stay dynamic — concreteness is probed
    with :func:`maybe_concrete_scalar`, never by branching on a tracer.
    All emitted operators already carry the 2π factor. Approximation
    advisories go to ``advisories`` when it is given, and to
    ``warnings.warn`` otherwise.
    """
    interactions: list[Operator] = []
    frame_corrections: list[Operator] = []
    td_terms: list[tuple[Operator, ScalarModulation]] = []
    dropped: list[DroppedTerm] = []
    supports: list[tuple[str, ...]] = []
    for resolved in _resolved_interactions(chip, resolution, backend):
        contribution, support = resolved.contribution, resolved.support
        labels = contribution.labels
        frequencies = tuple(resolved_frame.frequencies.get(label, 0.0) for label in labels)
        filters_terms = approximation.filters_terms and not contribution.retained
        authored, h_full = resolved.authored, resolved.operator
        if not contains_tracer(h_full) and _is_concrete_zero_array(backend.to_array(h_full)):
            continue

        if not filters_terms and all(maybe_concrete_scalar(freq) == 0.0 for freq in frequencies):
            interactions.append(resolution.embed(backend, h_full, support))
            supports.append(labels)
            continue

        sub_bands = _authored_bands(chip, authored, h_full, support, resolution.bases, backend, tag="coupling_local")
        retained: list[Operator] = []
        diagnostics: list[tuple[tuple[int, ...], bool, Any]] = []
        band_diagnostics = {}
        if filters_terms:
            # Sparse payloads stay sparse unless a basis transform moves them to energy coordinates.
            energy_canonical = backend.to_canonical_operator(h_full)
            transform = _support_semantic_transform(labels, resolution.bases)
            if transform is not None:
                energy_canonical = _sandwich_canonical(energy_canonical, transform.conj().T, transform)
            band_diagnostics = _band_diagnostics(
                *canonical_to_coo(energy_canonical), tuple(resolution.dims[index] for index in support),
                tuple(resolution.bases[label].energies for label in labels),
            )
        for weights, band_canonical in sub_bands.items():
            osc_freq = _band_frame_frequency(weights, frequencies)
            kept = contribution.keeps_band(approximation, weights)
            if filters_terms:
                amplitude, detuning, score = band_diagnostics.get(weights, (0.0, 0.0, 0.0))
                diagnostics.append((weights, kept, score))
            if not kept:
                dropped.append(DroppedTerm(
                    source=contribution.label,
                    operator=f"coupling band ({', '.join(f'{weight:+d}' for weight in weights)}) on {'·'.join(labels)}",
                    reason="discarded by RWA band selection",
                    band_weights=weights,
                    amplitude=amplitude,
                    frequency=detuning,
                    frame_frequency=abs(osc_freq),
                ))
                continue
            band_op = backend.from_canonical_operator(band_canonical)
            retained.append(band_op)
            concrete_osc = maybe_concrete_scalar(osc_freq)
            if concrete_osc is not None and concrete_osc == 0.0:
                continue
            embedded = resolution.embed(backend, band_op, support)
            scaled = backend.linear_combination([(TWO_PI, embedded)])
            frame_corrections.append(backend.linear_combination([(-TWO_PI, embedded)]))
            td_terms.append((scaled, ScalarModulation(signal=Carrier(freq=TWO_PI * osc_freq, sign=-1))))

        if filters_terms and sub_bands and not retained:
            _report_advisory(f"Coupling {contribution.label!r} vanishes entirely under RWA().", advisories)
        elif filters_terms:
            _report_advisory(_dropped_band_advisory(contribution.label, diagnostics), advisories)
        if filters_terms:
            if retained:
                interactions.append(resolution.embed(
                    backend, backend.linear_combination([(1, op) for op in retained]), support))
        else:
            interactions.append(resolution.embed(backend, h_full, support))
        if retained or not filters_terms:
            supports.append(labels)

    return interactions, frame_corrections, td_terms, dropped, supports


def _component_authored_terms(chip, resolution, backend):
    for local_op, coefficient, support, owner, origin, tag in chip.dynamic_contributions():
        owner_labels = tuple(chip.devices[index].label for index in support)
        owner_dims = tuple(chip.devices[index].local_space().dimension for index in support)
        local_op = as_operator_expr(
            local_op,
            labels=owner_labels,
            dims=owner_dims,
            name=rf"\hat H_{{{owner.label}}}(t)",
            owner=owner,
            scope=owner.label,
        )
        local_op, support = _retained_operator(chip, local_op, support, backend, bases=resolution.bases)
        yield local_op, coefficient, support, owner, origin, tag


def _component_time_terms(
    chip: "Chip",
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    resolution: _SystemResolution,
    approximation: Approximation,
) -> tuple[list[DynamicTerm], list[DroppedTerm], list[tuple[str, ...]]]:
    """Project component time terms, then apply frame carriers and coupling RWA."""
    labels = tuple(device.label for device in chip.devices)
    dynamic: list[DynamicTerm] = []
    dropped: list[DroppedTerm] = []
    supports: list[tuple[str, ...]] = []

    for local_op, coefficient, support, owner, origin, tag in _component_authored_terms(chip, resolution, backend):
        modulation = _as_time_coefficient(coefficient, owner=type(owner).__name__)
        owner_labels = tuple(chip.devices[index].label for index in support)
        projected = _project_on_support(chip, local_op, support, resolution.bases, backend)
        bands = _authored_bands(chip, local_op, projected, support, resolution.bases, backend, tag=tag)
        frequencies = tuple(resolved_frame.frequencies.get(label, 0.0) for label in owner_labels)

        for weights, band in bands.items():
            oscillation = _band_frame_frequency(weights, frequencies)
            if len(support) == 2 and not approximation.keeps_operator_band(weights):
                dropped.append(
                    DroppedTerm(
                        source=owner.label,
                        operator=f"time-dependent coupling band {weights}",
                        reason="discarded by RWA band selection",
                        band_weights=weights,
                        frame_frequency=abs(oscillation),
                    )
                )
                continue

            band_op = backend.from_canonical_operator(band)
            embedded = resolution.embed(backend, band_op, support)
            signal = modulation.signal
            concrete = maybe_concrete_scalar(oscillation)
            if concrete is None or concrete != 0.0:
                signal = Multiply((signal, Carrier(freq=TWO_PI * oscillation, sign=-1)))
            dynamic.append(
                _dynamic_term(
                    backend,
                    embedded,
                    dims=resolution.operator_dims,
                    labels=labels if resolution.space is None else ("excitation",),
                    tag=tag,
                    origin=cast(TermOrigin, origin),
                    time_dependence=ScalarModulation(signal=signal),
                )
            )
            if len(owner_labels) > 1 and owner_labels not in supports:
                supports.append(owner_labels)
    return dynamic, dropped, supports


# -- Drive resolution ----------------------------------------------------


def _resolve_drives(
    chip: "Chip",
    drive_ops: list["DriveOp"],
) -> list[tuple[BaseDrive, "DriveOp", Any]]:
    """Map each drive op to ``(drive, drive_op, target)`` using the chip's control equipment.

    The target is the chip's canonical device (device lines) or coupling
    (edge pump lines); the two label spaces are disjoint by Chip
    construction, so a plain two-map lookup is unambiguous.

    Cross-checks the resolved drive's own wiring against the ``DriveOp``:
    an unconnected drive, a target mismatch, or a definition target that
    disagrees with the map the label resolved from all raise
    ``ValueError``. Sequence scheduling enforces the same invariant at
    schedule time (:meth:`~quchip.control.sequence.QuantumSequence._schedule_on_drive`);
    this is the matching guard for ``DriveOp`` lists built directly,
    bypassing a ``QuantumSequence``.
    """
    equipment = chip.control_equipment
    resolved: list[tuple[BaseDrive, "DriveOp", Any]] = []
    for drive_op in drive_ops:
        label = drive_op.target_label
        if label in chip.device_map:
            device: Any = chip.device_map[label]
            resolved_kind = "device"
        elif label in chip.coupling_map:
            device = chip.coupling_map[label]
            resolved_kind = "coupling"
        else:
            raise ValueError(
                f"Target label '{label}' not found on chip (neither device nor coupling). "
                f"Devices: {list(chip.device_map.keys())}; couplings: {list(chip.coupling_map.keys())}."
            )

        if equipment is None:
            raise ValueError("Chip has no control equipment configured.")
        drive = next((d for d in equipment.lines if d.label == drive_op.drive_label), None)
        if drive is None:
            raise ValueError(
                f"No drive with label '{drive_op.drive_label}' found in equipment. "
                f"Available: {[d.label for d in equipment.lines]}"
            )

        if drive.target_label is None:
            raise ValueError(
                f"Drive '{drive.label}' is not connected to a target, but a DriveOp schedules it "
                f"onto '{drive_op.target_label}'. Connect it first with drive.connect(device)."
            )
        if drive.target_label != drive_op.target_label:
            raise ValueError(
                f"Drive '{drive.label}' is wired to target '{drive.target_label}', but its DriveOp "
                f"targets '{drive_op.target_label}'."
            )
        declared_target = "coupling" if isinstance(drive, CouplingDrive) else "device"
        if declared_target != resolved_kind:
            raise ValueError(
                f"Drive '{drive.label}' declares target '{declared_target}', but target "
                f"'{drive_op.target_label}' resolved as '{resolved_kind}' on the chip."
            )
        resolved.append((drive, drive_op, device))
    return resolved


@dataclass(frozen=True, slots=True)
class _DeliveredSignal:
    """One transformed signal paired with its destination drive and target."""

    drive: BaseDrive
    target: Any
    signal: AnalyticSignal
    origin: TermOrigin


# -- Drive term compilation ----------------------------------------------


@dataclass(frozen=True)
class CompiledDriveTerm:
    """One projected operator band from an authored drive Hamiltonian term."""

    operator: CanonicalOperator
    delivered_key: SignalKey
    hamiltonian_term_index: int
    weight: int
    device_frame_freq: Any
    filter_signal_bands: bool
    origin: TermOrigin = "drive"
    tag: str | None = None


@dataclass(frozen=True)
class CompiledCoherentTerm:
    """One ``S[j, i] beta_i`` contribution to the canonical source drive."""

    operation_index: int
    exposure: str
    scattering: Any
    lowering: CanonicalOperator
    raising: CanonicalOperator
    frame_frequency: Any


@dataclass(frozen=True)
class _ResolvedDriveBand:
    operator: Operator
    hamiltonian_term_index: int
    weight: int
    frame_frequency: Any
    filter_signal_bands: bool
    origin: TermOrigin
    tag: str


def _drive_authored_terms(chip, drive, target, backend):
    probe = AnalyticSignal(program=Constant(1.0 + 0.0j))
    with _backend_context(backend):
        terms = split_dynamic_hamiltonian(drive.hamiltonian(target, probe))
    pair = ((target.device_a_label, target.device_b_label) if isinstance(drive, CouplingDrive)
            else (target.label,))
    dims = tuple(chip[label].local_space().dimension for label in pair)
    for index, (_, operator) in enumerate(terms):
        yield index, as_operator_expr(operator, labels=pair, dims=dims, owner=drive,
                                      scope=f"drive.{drive.label}", name=rf"H_{{{drive.label},{index}}}")


def drive_excitation_changes(chip, drive, target, bases, approximation):
    from quchip.chip.effective import authored_excitation_changes
    changes = set()
    for _, authored in _drive_authored_terms(chip, drive, target, chip.backend):
        support = tuple(chip.device_index(label) for label in authored.labels)
        retained = any(term.projection is not None for term in chip.effective_terms)
        if retained:
            authored, support = _retained_operator(chip, authored, support, chip.backend,
                                                   f"drive:{drive.label}", bases=bases)
        if not retained and isinstance(drive, CouplingDrive) and approximation.conserves_excitation_number():
            weights = {0}
        else:
            weights = authored_excitation_changes(authored, tuple(chip.devices[i].label for i in support),
                                                   chip.backend, bases)
            if weights is None:
                return None
            if len(support) == 1:
                weights = {n for n in weights if _keeps_drive_band(drive, approximation, (n,))}
        changes.update(weights)
    return changes


def drive_bands(
    chip: "Chip",
    drive: BaseDrive,
    target: Any,
    bases: Mapping[str, BasisRecord],
    dims: tuple[int, ...],
    backend: Backend,
    approximation: Approximation,
    *, space: Any = None, embed: bool = True,
) -> list[tuple[BandRecord, Operator, int, int]]:
    """Yield kept band records, embedded operators, term indices and original signal weights."""
    pair = ((target.device_a_label, target.device_b_label) if isinstance(drive, CouplingDrive)
            else (target.label,))
    support = tuple(chip.device_index(label) for label in pair)
    local_dims = tuple(chip[label].local_space().dimension for label in pair)
    bands: list[tuple[BandRecord, Operator, int, int]] = []
    for term_index, authored in _drive_authored_terms(chip, drive, target, backend):
        local = _project_on_support(chip, authored, support, bases, backend)
        for charges, band in _authored_bands(chip, authored, local, support, bases, backend).items():
            if not _keeps_drive_band(drive, approximation, charges):
                continue
            source_weight = charges[0] if len(pair) == 1 else 1
            local_band = backend.from_canonical_operator(band)
            if any(terms.projection is not None for terms in chip.effective_terms):
                # Select the signal's RWA partner in its authored coordinates,
                # then transform each surviving band into the retained frame.
                vectors = bases[pair[0]].vectors
                for label in pair[1:]:
                    vectors = jnp.kron(vectors, bases[label].vectors)
                matrix = vectors @ jnp.asarray(backend.to_array(local_band)) @ vectors.conj().T
                # The band changes the total energy level by the sum of its charges; declaring it keeps a
                # conserving projection's structurally empty bands out even when the drive is traced.
                band_expr = PhysicsExpr.from_matrix(matrix, labels=pair, dims=local_dims,
                                                    excitation_changes=(sum(charges),))
                projected, active_support = _retained_operator(
                    chip, band_expr, support, backend, f"drive:{drive.label}", bases=bases)
                local_band = _project_on_support(chip, projected, active_support, bases, backend)
                transformed = _authored_bands(chip, projected, local_band, active_support, bases, backend)
            else:
                active_support, transformed = support, {charges: band}
            active_labels = tuple(chip.devices[i].label for i in active_support)
            for active_charges, active_band in transformed.items():
                record = BandRecord(active_labels, active_charges, _concrete_amplitude(active_band.values), drive.label)
                local_band = backend.from_canonical_operator(active_band)
                embedded = (local_band if not embed else space.embed(backend, local_band, active_support)
                            if space is not None else _embed_product(backend, local_band, active_support, dims))
                bands.append((record, embedded, term_index, source_weight))
    return bands


def _resolved_drive_bands(
    chip: "Chip",
    drive: BaseDrive,
    target: Any,
    resolved_frame: "ResolvedFrame",
    *,
    bases: Mapping[str, BasisRecord],
    dims: tuple[int, ...],
    backend: Backend,
    approximation: Approximation,
    space: Any = None,
) -> list[_ResolvedDriveBand]:
    """Attach frame carriers to the retained bands of one drive's authored Hamiltonian."""
    single = not isinstance(drive, CouplingDrive)
    frequencies = resolved_frame.frequencies
    origin: TermOrigin = "drive" if single else "coupling"
    tag = "drive" if single else "edge_pump"
    resolved: list[_ResolvedDriveBand] = []
    for record, embedded, term_index, source_weight in drive_bands(
        chip, drive, target, bases, dims, backend, approximation, space=space
    ):
        carrier = _band_frame_frequency(record.charges, [frequencies.get(label, 0.0) for label in record.devices])
        resolved.append(
            _ResolvedDriveBand(
                operator=embedded,
                hamiltonian_term_index=term_index,
                weight=source_weight,
                frame_frequency=carrier,
                filter_signal_bands=single and approximation.filters_terms,
                origin=origin,
                tag=tag,
            )
        )
    # Equal frame frequencies share one scalar program. Merge their matrices
    # before backend lowering, especially for globally projected lab operators.
    grouped: dict[tuple[int, int, float], int] = {}
    combined: list[_ResolvedDriveBand] = []
    for band in resolved:
        frequency = maybe_concrete_scalar(band.frame_frequency)
        key = None if frequency is None else (band.hamiltonian_term_index, band.weight, float(frequency))
        if key is not None and key in grouped:
            index = grouped[key]
            combined[index] = replace(combined[index], operator=combined[index].operator + band.operator)
        else:
            if key is not None:
                grouped[key] = len(combined)
            combined.append(band)
    return combined


def _compile_drive_terms(
    chip: "Chip",
    delivered_signals: Mapping[SignalKey, _DeliveredSignal],
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    *,
    bases: Mapping[str, BasisRecord],
    dims: tuple[int, ...],
    subsystem_labels: tuple[str, ...],
    approximation: Approximation,
    space: Any = None,
) -> tuple[CompiledDriveTerm, ...]:
    """Band-decompose each drive channel into pre-embedded, 2π-scaled drive terms.

    For each authored Hamiltonian term the operator is split into
    single-subsystem excitation-change bands of weight ``w``
    (cf. :mod:`quchip.engine.bands`). The engine combines it with the complete
    delivered signal and a frame factor of the form
    ``exp(−i w ω_ref t)``, which
    is the standard rotating-wave form for a driven multi-level system
    (Jaynes & Cummings 1963; Scully & Zubairy, *Quantum Optics*, §5). A
    weight-zero band keeps the complete signal and describes longitudinal
    modulation. Nonzero weights select the corresponding carrier sideband.

    The resulting :class:`CompiledDriveTerm` is template-cached so
    homogeneous sweeps only rebuild signal-program leaves, not operators.
    """
    compiled: list[CompiledDriveTerm] = []
    # A line's bands do not depend on the signal it delivers: every pulse and
    # crosstalk path reaching the line shares one operator per band, which
    # lets backends apply it once.
    line_bands: dict[tuple[str, str], list[tuple[_ResolvedDriveBand, CanonicalOperator]]] = {}
    for delivered_key, delivered in delivered_signals.items():
        drive = delivered.drive
        target = delivered.target
        line = (drive.label, target.label)
        if line not in line_bands:
            line_bands[line] = [
                (band, _apply_2pi_canonical(
                    backend, band.operator, dims=dims if space is None else space.solver_dims,
                    labels=subsystem_labels if space is None else ("excitation",), tag=band.tag))
                for band in _resolved_drive_bands(
                    chip,
                    drive,
                    target,
                    resolved_frame,
                    bases=bases,
                    dims=dims,
                    backend=backend,
                    approximation=approximation,
                    space=space,
                )
            ]
        for band, operator in line_bands[line]:
            compiled.append(
                CompiledDriveTerm(
                    operator=operator,
                    delivered_key=delivered_key,
                    hamiltonian_term_index=band.hamiltonian_term_index,
                    weight=band.weight,
                    device_frame_freq=band.frame_frequency,
                    filter_signal_bands=band.filter_signal_bands,
                    origin=("crosstalk" if delivered.origin == "crosstalk" else band.origin),
                    tag="crosstalk" if delivered.origin == "crosstalk" else band.tag,
                )
            )
    return tuple(compiled)


def _build_delivered_signals(
    chip: "Chip",
    drive_ops: list["ControlOp"],
) -> dict[SignalKey, _DeliveredSignal]:
    """Build, transform, and route complete signals to destination drives."""
    resolved = _resolve_drives(
        chip,
        [operation for operation in drive_ops if isinstance(operation, DriveOp)],
    )
    raw_signals: dict[SignalKey, AnalyticSignal] = {}
    for source_index, (drive, drive_op, target) in enumerate(resolved):
        signal = drive.signal(drive_op, target)
        if not isinstance(signal, AnalyticSignal):
            raise TypeError(f"{type(drive).__name__}.signal() must return AnalyticSignal, got {type(signal).__name__}.")
        raw_signals[(drive.label, source_index)] = signal

    equipment = chip.control_equipment
    transformed = (
        raw_signals if equipment is None or not equipment.signal_chain else equipment.apply_signal_chain(raw_signals)
    )
    if equipment is None:
        return {}

    line_map = {line.label: line for line in equipment.lines}
    delivered: dict[SignalKey, _DeliveredSignal] = {}
    for (destination_label, source_index), signal in transformed.items():
        destination = line_map.get(destination_label)
        if destination is None or destination._target is None:
            continue
        source_drive = resolved[source_index][0]
        delivered[(destination_label, source_index)] = _DeliveredSignal(
            drive=destination,
            target=destination._target,
            signal=signal,
            origin=("drive" if destination.label == source_drive.label else "crosstalk"),
        )
    return delivered


def _is_concrete_zero(value: Any) -> bool:
    """Return whether *value* is safely known to be scalar zero."""
    if contains_tracer(value):
        return False
    try:
        concrete = np.asarray(value)
    except Exception:
        return False
    return concrete.ndim == 0 and bool(concrete == 0)


def _is_concrete_zero_array(value: Any) -> bool:
    """Return whether an array payload is safely known to vanish."""
    if contains_tracer(value):
        return False
    try:
        return bool(np.all(np.asarray(value) == 0))
    except Exception:
        return False


def _compile_coherent_terms(
    slh: ResolvedSLH,
    control_ops: list["ControlOp"],
) -> tuple[CompiledCoherentTerm, ...]:
    """Compile operator skeletons for coherent-source composition.

    Cascading ``W_beta = (I, beta, 0)`` before ``(S, L, H)`` gives
    ``c = S beta``. Canonicalizing the displaced collapse operators back to
    the input-free ``L`` produces ``H_beta = i(c* L - c L dagger)``.
    """
    external = slh.external_channels
    exposure_index = {channel.key: index for index, channel in enumerate(external)}
    compiled: list[CompiledCoherentTerm] = []
    for operation_index, operation in enumerate(control_ops):
        if not isinstance(operation, CoherentOp):
            continue
        input_index = exposure_index.get(operation.exposure)
        if input_index is None:
            raise ValueError(
                f"Unknown coherent-input exposure {operation.exposure!r}. "
                f"Available exposures: {list(exposure_index)}."
            )
        for output_index, channel in enumerate(slh.channels):
            scattering = slh.S[output_index, input_index]
            if _is_concrete_zero(scattering):
                continue
            lowering = channel.coupling.with_metadata(
                tag=f"coherent_input:{operation.exposure}:{channel.key}:L"
            )
            dense = lowering.to_dense()
            xp = array_namespace(dense)
            raising = CanonicalOperator.from_dense(
                xp.conj(xp.swapaxes(dense, -1, -2)),
                dims=lowering.dims,
                basis=lowering.basis,
                subsystem_labels=lowering.subsystem_labels,
                tag=f"coherent_input:{operation.exposure}:{channel.key}:Ldagger",
            )
            compiled.append(
                CompiledCoherentTerm(
                    operation_index=operation_index,
                    exposure=operation.exposure,
                    scattering=scattering,
                    lowering=lowering,
                    raising=raising,
                    frame_frequency=channel.carrier,
                )
            )
    return tuple(compiled)


def _reject_inputs_to_mode_responses(chip: "Chip", control_ops: list["ControlOp"]) -> None:
    """Reject a coherent input that reaches the ports of a retained mode response.

    The Markov dynamics keep only the transformed port operators, so they omit
    the removed mode's own scattering of that field.
    """
    network = chip.port_network
    if network is None or not network._mode_responses:
        return
    compiled = network._compile()
    owners = {label: mode.label for mode in network._mode_responses for label in mode.ports}
    for operation in control_ops:
        if not isinstance(operation, CoherentOp):
            continue
        fed = network.fed_ports(chip, operation.exposure, _compiled=compiled)
        reached = sorted({owners[port.label] for port, _ in fed if port.label in owners})
        if reached:
            raise NotImplementedError(
                f"Coherent input {operation.exposure!r} reaches retained multiport mode responses {reached}. "
                "Time-domain dynamics omit their poles. Keep the boundary mode for transient field inputs."
            )


def _frame_shifted(program: SignalProgram, frequency: Any) -> SignalProgram:
    """Attach one operator-frame phase unless it is concretely zero."""
    if _is_concrete_zero(frequency):
        return program
    return Multiply((program, Carrier(freq=TWO_PI * frequency, sign=-1)))


def _instantiate_coherent_terms(
    template: HamiltonianTemplate,
    control_ops: list["ControlOp"],
) -> tuple[tuple[DynamicTerm, ...], tuple[BoundCoherentInput, ...]]:
    """Bind beta programs and materialize their solve-applied Hamiltonian."""
    external = {channel.key: channel for channel in template.base_result.slh.external_channels}
    boundary_signals: dict[int, AnalyticSignal] = {}
    bound: list[BoundCoherentInput] = []
    for operation_index, operation in enumerate(control_ops):
        if not isinstance(operation, CoherentOp):
            continue
        channel = external[operation.exposure]
        reference = operation.coherent_input.signal(operation, channel)
        if not isinstance(reference, AnalyticSignal):
            raise TypeError(
                f"{type(operation.coherent_input).__name__}.signal() must return "
                f"AnalyticSignal, got {type(reference).__name__}."
            )
        inbound_shift = time_shift(channel.reference.inbound)
        boundary = (
            reference if _is_concrete_zero(inbound_shift) else reference.shifted(inbound_shift)
        )
        if has_filter(channel.reference.inbound):
            carrier = 0.0 if reference.carrier is None else reference.carrier
            boundary = boundary.scaled(carrier_transfer(channel.reference.inbound, carrier))
        boundary_signals[operation_index] = boundary
        bound.append(
            BoundCoherentInput(
                exposure=operation.exposure,
                source_label=operation.coherent_input.label,
                beta=boundary.program,
                reference_beta=reference.program,
            )
        )

    dynamic: list[DynamicTerm] = []
    for compiled in template.coherent_terms:
        beta = boundary_signals[compiled.operation_index].program
        xp = array_namespace(compiled.scattering)
        scattering = xp.asarray(compiled.scattering)
        lowering_signal = _frame_shifted(
            Scale(Conjugate(beta), factor=1j * xp.conj(scattering)),
            compiled.frame_frequency,
        )
        raising_signal = _frame_shifted(
            Scale(beta, factor=-1j * scattering),
            -compiled.frame_frequency,
        )
        dynamic.extend(
            (
                DynamicTerm(
                    operator=compiled.lowering,
                    time_dependence=ScalarModulation(signal=lowering_signal),
                    origin="port",
                    tag=f"coherent_input:{compiled.exposure}",
                ),
                DynamicTerm(
                    operator=compiled.raising,
                    time_dependence=ScalarModulation(signal=raising_signal),
                    origin="port",
                    tag=f"coherent_input:{compiled.exposure}",
                ),
            )
        )
    return tuple(dynamic), tuple(bound)


def _drive_scalar_program(
    drive: BaseDrive,
    target: Any,
    signal: AnalyticSignal,
    term_index: int,
) -> SignalProgram:
    """Return one scalar signal factor from a drive-authored Hamiltonian."""
    terms = split_dynamic_hamiltonian(drive.hamiltonian(target, signal))
    try:
        scalar, _operator = terms[term_index]
    except IndexError as exc:
        raise ValueError(
            f"{type(drive).__name__}.hamiltonian() changed term topology between "
            "template compilation and instantiation."
        ) from exc
    return scalar_signal_program(scalar)


# -- Dropped-term aggregation --------------------------------------------


def _collect_dropped_terms(chip: "Chip", resolved_frame: "ResolvedFrame") -> tuple[DroppedTerm, ...]:
    """Gather coupling-declared advisory records outside engine band filtering.

    ``RWA()`` drops are generated inside :func:`_resolve_coupling_terms`;
    this pass collects whatever a coupling's own model elides. Records
    that declare ``band_weights`` and no ``frame_frequency`` get one
    resolved here as ``|Σ wᵢ·f_ref,i|``. This arithmetic keeps traced frame
    frequencies traced.
    """
    gathered: list[DroppedTerm] = []
    for coupling in chip.couplings:
        endpoint_labels = (coupling.device_a_label, coupling.device_b_label)
        for record in coupling.dropped_terms():
            weights = record.band_weights
            if record.frame_frequency is None and weights is not None and len(weights) == len(endpoint_labels):
                freq = _band_frame_frequency(
                    weights, [resolved_frame.frequencies.get(label, 0.0) for label in endpoint_labels],
                )
                record = replace(record, frame_frequency=abs(freq))
            gathered.append(record)
    return tuple(gathered)


# -- Template validation -------------------------------------------------


def _validate_variant_drive_ops(
    template: HamiltonianTemplate,
    drive_ops: list["ControlOp"],
) -> None:
    """Check that *drive_ops* match the template's drive, target, and envelope shape."""
    reference_ops = template.reference_drive_ops
    if len(drive_ops) != len(reference_ops):
        raise ValueError(
            "Homogeneous Hamiltonian template requires the same number of scheduled pulse entries. "
            f"Expected {len(reference_ops)}, got {len(drive_ops)}."
        )

    for index, (reference_op, variant_op) in enumerate(zip(reference_ops, drive_ops)):
        if type(variant_op) is not type(reference_op):
            raise ValueError(
                f"Variant control op {index} uses {type(variant_op).__name__}, "
                f"expected {type(reference_op).__name__}."
            )
        if variant_op.target_label != reference_op.target_label:
            raise ValueError(
                f"Variant drive op {index} targets '{variant_op.target_label}', expected '{reference_op.target_label}'."
            )
        if variant_op.drive_label != reference_op.drive_label:
            raise ValueError(
                f"Variant drive op {index} uses drive '{variant_op.drive_label}', "
                f"expected '{reference_op.drive_label}'."
            )
        if type(variant_op.envelope) is not type(reference_op.envelope):
            raise ValueError(
                f"Variant drive op {index} uses envelope '{type(variant_op.envelope).__name__}', "
                f"expected '{type(reference_op.envelope).__name__}'."
            )


# -- Public API ----------------------------------------------------------


def _template_from_engine_result(
    chip: "Chip",
    drive_ops: list["ControlOp"],
    base_result: EngineResult,
    resolved_frame: "ResolvedFrame",
) -> HamiltonianTemplate:
    """Attach scheduled-drive structure to an already resolved chip contract."""
    if base_result.resolved_frame is not resolved_frame:
        raise ValueError("The base EngineResult and scheduled drives must use the same resolved frame.")

    dims = base_result.dims if base_result.excitation_space is None else base_result.excitation_space.dims
    subsystem_labels = tuple(device.label for device in chip.devices)
    delivered = _build_delivered_signals(chip, drive_ops)
    drive_terms = _compile_drive_terms(
        chip,
        delivered,
        resolved_frame,
        chip.backend,
        bases=base_result.bases,
        dims=dims,
        subsystem_labels=subsystem_labels,
        approximation=base_result.approximation,
        space=base_result.excitation_space,
    )
    coherent_terms = _compile_coherent_terms(base_result.slh, drive_ops)
    _reject_inputs_to_mode_responses(chip, drive_ops)
    return HamiltonianTemplate(
        base_result=replace(base_result, metadata=dict(base_result.metadata)),
        drive_terms=drive_terms,
        coherent_terms=coherent_terms,
        reference_drive_ops=tuple(drive_ops),
        delivered_keys=frozenset(delivered),
    )


def _resolved_dressing_context(
    chip: "Chip",
    resolution: _SystemResolution,
) -> _ResolvedDressingContext:
    """Freeze assembly-time basis references for ``EngineResult.dress()``."""
    from quchip.engine.dressing import BareProductReference

    if resolution.space is not None:
        return _ResolvedDressingContext(backend=chip.backend, reference=BareProductReference(resolution.operator_dims))
    return _ResolvedDressingContext(
        backend=chip.backend,
        reference=BareProductReference(dims=tuple(resolution.dims), local_vectors=tuple(
            resolution.bases[device.label].energy_to_solver()
            for device in chip.devices)),
    )


def compile_hamiltonian_template(
    chip: "Chip",
    drive_ops: list["ControlOp"],
    *,
    resolved_frame: "ResolvedFrame",
    approximation: Approximation | None = None,
    _local_resolution: _SystemResolution | None = None,
    _base_result: EngineResult | None = None,
    _advisories: list[str] | None = None,
) -> HamiltonianTemplate:
    """Compile the invariant Hamiltonian skeleton (H₀, couplings, pre-embedded drive bands).

    The template holds everything invariant across a homogeneous sweep: the
    static Hamiltonian, static-coupling folds, invariant dynamic couplings, and
    band-decomposed drive operators pre-embedded and pre-scaled by 2π.
    Per-sweep instantiation (:func:`instantiate_engine_result`) rebuilds only
    the :class:`~quchip.engine.ir.SignalProgram` leaves. Envelope parameters,
    drive frequencies, phases, and frame scalars can therefore sweep through
    JAX without retracing operator tensors.
    """
    strategy = chip.approximation if approximation is None else require_approximation(approximation)
    if _base_result is not None:
        if _base_result.approximation != strategy:
            raise ValueError("The base EngineResult and scheduled drives must use the same approximation.")
        return _template_from_engine_result(
            chip,
            drive_ops,
            _base_result,
            resolved_frame,
        )

    backend = chip.backend
    resolution = _resolve_system(chip, backend) if _local_resolution is None else _local_resolution
    dims = resolution.operator_dims
    subsystem_labels = tuple(d.label for d in chip.devices) if resolution.space is None else ("excitation",)

    (
        coupling_h0,
        coupling_frame_static,
        coupling_td,
        coupling_dropped,
        coupling_supports,
    ) = _resolve_coupling_terms(
        chip,
        resolved_frame,
        backend,
        resolution,
        strategy,
        _advisories,
    )
    h0 = _build_static_h0(
        chip,
        resolved_frame,
        backend,
        resolution,
        coupling_h0,
    )
    if coupling_frame_static:
        h0 = backend.linear_combination([(1, h0), *((1, op) for op in coupling_frame_static)])

    # The coupling fold above cancels the lab-frame interaction out of H₀
    # exactly, leaving its diagonal offsets stored as explicit zeros; prune
    # them (tracer-guarded) so backends never integrate dead structure.
    static_terms = (
        StaticTerm(
            operator=prune_zero_diagonals(
                backend.to_canonical_operator(h0).with_metadata(
                    dims=dims,
                    subsystem_labels=subsystem_labels,
                    tag="H0",
                )
            ),
            coefficient=1.0,
            origin="device",
            metadata={"frame": str(resolved_frame)},
        ),
    )

    invariant_dynamic_terms: list[DynamicTerm] = []
    for op, td in coupling_td:
        invariant_dynamic_terms.append(
            DynamicTerm(
                operator=backend.to_canonical_operator(op).with_metadata(
                    dims=dims,
                    subsystem_labels=subsystem_labels,
                    tag="coupling",
                ),
                time_dependence=td,
                origin="coupling",
            )
        )

    component_dynamic, component_dropped, component_supports = _component_time_terms(
        chip,
        resolved_frame,
        backend,
        resolution,
        strategy,
    )
    invariant_dynamic_terms.extend(component_dynamic)
    coupling_dropped.extend(component_dropped)

    # Invariant signals never change across variants, so simplification
    # happens once here instead of on every instantiation.
    simplified_invariant = tuple(
        replace(term, time_dependence=ScalarModulation(signal=_simplify_signal(term.time_dependence.signal)))
        for term in invariant_dynamic_terms
    )

    collapse_terms, collapse_supports = _collect_collapse_terms(
        chip,
        resolved_frame,
        backend,
        resolution,
    )
    slh = ResolvedSLH.from_terms(
        static_terms=static_terms,
        dynamic_terms=simplified_invariant,
        collapse_terms=collapse_terms,
    )
    if chip.port_network is not None:
        slh = chip.port_network.resolve(slh, _compiled=resolution.network,
                                        _generated=_compact_network_hamiltonian(chip, resolution, backend))
    channel_supports = list(collapse_supports)
    for bath in chip.baths:
        targets = tuple(bath.resolve_targets(chip))
        if not bath.separable and len(targets) > 1:
            channel_supports.append(targets)
    network_supports = (
        ()
        if chip.port_network is None
        else chip.port_network.dynamical_supports(chip, _compiled=resolution.network)
    )
    dynamical_supports = tuple(
        dict.fromkeys(
            (*coupling_supports, *component_supports, *channel_supports, *network_supports)
        )
    )
    # Static terms are invariant across a homogeneous sweep, including any
    # Hamiltonian generated by network composition. Store the bound in GHz.
    resolved_static_span = _static_spectral_span(slh.H.static_terms)
    static_spectral_bound_ghz = (
        resolved_static_span / TWO_PI if resolved_static_span is not None else None
    )
    base_result = EngineResult(
        resolved_frame=resolved_frame,
        approximation=strategy,
        dims=dims,
        slh=slh,
        dropped_terms=_collect_dropped_terms(chip, resolved_frame) + tuple(coupling_dropped),
        metadata={"static_spectral_bound_ghz": static_spectral_bound_ghz},
        bases=resolution.bases,
        authored=chip.unresolved_hamiltonian(),
        _dressing_context=_resolved_dressing_context(chip, resolution),
        excitation_space=resolution.space,
        dynamical_supports=dynamical_supports,
    )
    return _template_from_engine_result(chip, drive_ops, base_result, resolved_frame)


def instantiate_engine_result(
    template: HamiltonianTemplate,
    drive_ops: list["ControlOp"],
    chip: "Chip",
) -> EngineResult:
    """Rebuild signal-program leaves from *drive_ops* and attach them to the template's operators."""
    _validate_variant_drive_ops(template, drive_ops)
    base_result = template.base_result

    delivered = _build_delivered_signals(chip, drive_ops)
    if delivered.keys() != template.delivered_keys:
        raise ValueError(
            "Signal routes changed since Hamiltonian template compilation. "
            "Compile a new template for the changed routing."
        )

    # Only the variant-specific terms built below need simplification;
    # the template's invariant terms were simplified at compile time.
    # Drive frequencies are per-op, so the fast partners the RWA drops
    # only become auditable here — their records join the template's.
    fresh_terms: list[DynamicTerm] = []
    fresh_dropped: list[DroppedTerm] = []
    for compiled in template.drive_terms:
        item = delivered[compiled.delivered_key]
        program = _drive_scalar_program(
            item.drive,
            item.target,
            item.signal,
            compiled.hamiltonian_term_index,
        )
        fresh_terms.append(
            DynamicTerm(
                operator=compiled.operator,
                time_dependence=ScalarModulation(
                    signal=resolve_drive_program(
                        base_result.approximation,
                        program,
                        weight=compiled.weight,
                        frame_frequency=compiled.device_frame_freq,
                        has_carrier=item.signal.carrier is not None,
                        filter_signal_bands=compiled.filter_signal_bands,
                    )
                ),
                origin=compiled.origin,
                tag=compiled.tag,
            )
        )
        if (
            compiled.filter_signal_bands
            and item.signal.carrier is not None
            and compiled.weight != 0
        ):
            fresh_dropped.append(
                DroppedTerm(
                    source=item.drive.label,
                    operator=(f"{compiled.origin} band w={compiled.weight:+d} on {item.target.label} (fast partner)"),
                    reason="counter-rotating drive component under RWA",
                    band_weights=(compiled.weight,),
                    frame_frequency=abs(item.signal.carrier
                                        + (1 if compiled.weight > 0 else -1) * compiled.device_frame_freq),
                )
            )

    applied_drive_terms = tuple(
        replace(
            term,
            time_dependence=ScalarModulation(signal=_simplify_signal(term.time_dependence.signal)),
        )
        for term in fresh_terms
    )
    coherent_terms, coherent_inputs = _instantiate_coherent_terms(template, drive_ops)
    applied_dynamic_terms = applied_drive_terms + coherent_terms
    dynamic_terms = base_result.slh.H.dynamic_terms + applied_dynamic_terms

    static_bound = base_result.metadata.get("static_spectral_bound_ghz")
    metadata: dict[str, Any] = {**base_result.metadata, "frame": str(base_result.resolved_frame)}
    if static_bound is not None:
        metadata["static_spectral_bound_ghz"] = static_bound
    metadata.update(_solver_hint_metadata(static_bound, dynamic_terms))

    return replace(
        base_result,
        applied_hamiltonian=HamiltonianProgram(dynamic_terms=applied_dynamic_terms),
        coherent_inputs=coherent_inputs,
        metadata=metadata,
        dropped_terms=base_result.dropped_terms + tuple(fresh_dropped),
        dissipation=True,
    )


def build_engine_result(
    chip: "Chip",
    drive_ops: list["ControlOp"],
    *,
    resolved_frame: "ResolvedFrame",
    approximation: Approximation | None = None,
    _local_resolution: _SystemResolution | None = None,
    _base_result: EngineResult | None = None,
    _advisories: list[str] | None = None,
) -> EngineResult:
    """Compile the template and instantiate one engine-result variant.

    Equivalent to :func:`compile_hamiltonian_template` followed by
    :func:`instantiate_engine_result` with the same ``drive_ops``. Prefer the
    two-step form when you solve many variants that share the same chip
    topology.

    Parameters
    ----------
    chip : Chip
        The chip whose device, coupling, and drive Hamiltonians are
        assembled (2π applied at this boundary).
    drive_ops : list of ControlOp
        Scheduled classical-drive or coherent-field operations to embed as dynamic terms.
    resolved_frame : ResolvedFrame
        Resolved frame carrying the per-device frame frequencies,
        demodulation frequencies, and frame mode.

    Returns
    -------
    EngineResult
        Static terms, dynamic terms, and dropped-term records for the
        single variant.
    """
    template = compile_hamiltonian_template(
        chip,
        drive_ops,
        resolved_frame=resolved_frame,
        approximation=approximation,
        _local_resolution=_local_resolution,
        _base_result=_base_result,
        _advisories=_advisories,
    )
    return instantiate_engine_result(template, drive_ops, chip)


def _build_static_analysis_result(
    chip: "Chip",
    *,
    approximation: Approximation,
    _local_resolution: _SystemResolution | None = None,
    _advisories: list[str] | None = None,
) -> EngineResult:
    """Resolve the chip in the lab frame and retain only its static model."""
    labels = [device.label for device in chip.devices]
    frame = ResolvedFrame(
        frequencies={label: 0.0 for label in labels},
        demod_freqs={label: 0.0 for label in labels},
        mode="lab",
    )
    result = build_engine_result(
        chip,
        [],
        resolved_frame=frame,
        approximation=approximation,
        _local_resolution=_local_resolution,
        _advisories=_advisories,
    )
    metadata = {key: value for key, value in result.metadata.items() if key in {"frame", "spectral_bound_ghz"}}
    return replace(
        result,
        slh=ResolvedSLH.from_terms(
            static_terms=result.static_terms,
            dynamic_terms=(),
            collapse_terms=(),
        ),
        applied_hamiltonian=HamiltonianProgram(),
        metadata=metadata,
    )


def _analysis_matrix_ghz(result: EngineResult, *, include_network: bool = True) -> Any:
    """Return the resolved static solver Hamiltonian in ordinary GHz.

    This internal path reads canonical arrays directly so JAX tracers never
    cross through a non-JAX inspection backend. Time-dependent terms are not
    part of a static dressed-state calculation. Set ``include_network=False``
    when a reduction regenerates network terms from transformed ports.
    """
    if not result.static_terms:
        raise ValueError("EngineResult contains no static Hamiltonian terms.")
    pairs = [(term.coefficient, term.operator.to_dense()) for term in result.static_terms
             if include_network or term.origin != "network"]
    if not pairs:
        matrix = result.static_terms[0].operator.to_dense()
        return array_namespace(matrix).zeros_like(matrix)
    xp = array_namespace(pairs[0][1])
    if not contains_tracer(pairs):
        # Concrete terms combine on the host and return in their own namespace.
        terms = [np.asarray(coefficient) * np.asarray(matrix) / TWO_PI for coefficient, matrix in pairs]
        return xp.asarray(sum(terms[1:], start=terms[0]))
    terms = [coefficient * matrix / TWO_PI for coefficient, matrix in pairs]
    return sum(terms[1:], start=terms[0])
