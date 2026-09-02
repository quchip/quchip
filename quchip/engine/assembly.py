"""Assemble an :class:`EngineResult` from chip, drive operations, and frame.

Responsibilities
----------------
This module owns the **2π boundary** of the engine: inputs are ordinary
GHz (ν), outputs are operators scaled by
ω = 2π·ν so backends can solve Schrödinger's equation with ``d|ψ⟩/dt =
-i H |ψ⟩`` in ns/rad units. The operator angular-scaling boundary lives
entirely here, in:

* :func:`_build_static_h0` — frame-subtracted bare Hamiltonian,
* :func:`_resolve_coupling_terms` — full interaction band-decomposed,
  each band handled by the chip's approximation strategy and folded into ``H₀``
  or carried, per band,
* :func:`_apply_2pi_canonical` — the single point that scales every
  embedded *dynamic* operator (drive, crosstalk, coupling-dynamic,
  device-dynamic).

The same ``2π`` convention also expresses signal-AST carrier and
rotating-frame-phase *frequencies* in rad/ns
(:func:`_single_tone_coefficient`, :func:`_direct_real_coefficient`) and
the observable-demodulation phase; those are frequencies inside the
time-dependence / observable bookkeeping, not a second Hamiltonian
boundary. :mod:`quchip.engine.solver_hints` divides by ``2π`` only to
report advisory hints back in ordinary GHz.

Physics
-------
Assembly performs three physically distinct operations on top of the 2π
scaling:

1. **Rotating-frame transformation.** Each device's number operator is
   shifted by its frame reference ω_ref so that the static Hamiltonian
   becomes ``H₀ − Σᵢ ω_ref,ᵢ nᵢ`` (see any standard cQED reference,
   e.g. Scully & Zubairy, *Quantum Optics*, CUP 1997, §5.1).

2. **Band decomposition / rotating-wave approximation (RWA).** Coupling
   and drive operators are split into excitation-change bands of weight
   ``w = col − row`` and attached to carriers ``exp(−i w·ω t)``.
   :class:`~quchip.approximations.Exact` retains them all.
   :class:`~quchip.approximations.RWA` retains total-excitation-conserving
   static bands and matches delivered-signal bands to operator bands (Jaynes & Cummings,
   *Proc. IEEE* **51**, 89 (1963); Walls & Milburn, *Quantum Optics*,
   Springer 2008, §10.3; for dispersive/structured cases see
   Gambetta et al., *PRA* **74**, 042318 (2006), and the
   cross-resonance treatment in Magesan & Gambetta, *PRA* **101**,
   052308 (2020)).

3. **Signal-program construction.** Time dependence is emitted as a
   :class:`~quchip.engine.ir.SignalProgram` AST — a pure, JAX-traceable
   description that backends lower into their native coefficient form.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass, replace
from math import prod
from typing import TYPE_CHECKING, Any, Mapping, cast

import jax
import jax.numpy as jnp
import numpy as np

from quchip.approximations import Approximation, Exact, RWA, require_approximation
from quchip.backend import _backend_context
from quchip.backend.protocol import Backend, Operator
from quchip.control.drive import BaseDrive, CouplingDrive
from quchip.control.signal import AnalyticSignal
from quchip.declarative.expr import (
    as_operator_expr,
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
from quchip.engine.basis import (
    BasisRecord,
    resolve_local_basis,
    semantic_to_solver_transform,
)
from quchip.engine.solver_hints import _solver_hint_metadata, _static_diagonal_span
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import (
    array_namespace,
    contains_tracer,
    maybe_concrete_scalar,
)
from quchip.engine.bands import (
    _decompose_product_canonical_bands,
    decompose_canonical_bands,
    decompose_two_body_canonical_bands,
    embed_on_support,
    embed_single_mode_bands,
    prune_zero_diagonals,
)

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.engine.ir import ControlOp


def _weight_zero_dropped_term(*, source: str, device_label: str, drive_freq: Any) -> DroppedTerm:
    """Describe a carrier-driven weight-zero band dropped by band RWA."""
    if drive_freq is None:
        raise ValueError(
            f"Carrier-driven weight-zero band on '{device_label}' (drive '{source}') has no carrier frequency."
        )
    return DroppedTerm(
        source=source,
        operator=f"drive band w=+0 on {device_label}",
        reason="no frame rotation cancels either carrier sideband under band RWA",
        band_weights=(0,),
        frequency=abs(drive_freq),
    )


# -- Static Hamiltonian --------------------------------------------------


@dataclass(frozen=True)
class _LocalResolution:
    bases: dict[str, BasisRecord]
    hamiltonians: tuple[Operator, ...]
    dims: tuple[int, ...]


def _support_semantic_transform(
    chip: "Chip",
    support: tuple[int, ...],
    bases: Mapping[str, BasisRecord],
) -> Any | None:
    transforms = [
        semantic_to_solver_transform(chip.devices[index], bases[chip.devices[index].label])
        for index in support
    ]
    if all(transform is None for transform in transforms):
        return None
    resolved = [
        jnp.eye(bases[chip.devices[index].label].resolved_dim, dtype=jnp.complex128)
        if transform is None
        else transform
        for index, transform in zip(support, transforms, strict=True)
    ]
    product_transform = resolved[0]
    for transform in resolved[1:]:
        product_transform = jnp.kron(product_transform, transform)
    return product_transform


def _resolve_local_system(chip: "Chip", backend: Backend) -> _LocalResolution:
    bases: dict[str, BasisRecord] = {}
    hamiltonians: list[Operator] = []
    dims: list[int] = []
    for device in chip.devices:
        authored = device.unresolved_hamiltonian()
        matrix = materialize_array(authored)
        policy = chip.resolve_basis(device)
        levels = device.resolved_dimension(chip.basis) if policy == "eigen" else None
        record = resolve_local_basis(matrix, basis=policy, levels=levels)
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
    return _LocalResolution(
        bases=bases,
        hamiltonians=tuple(hamiltonians),
        dims=tuple(dims),
    )


def _prepare_engine_assembly(
    chip: "Chip",
    frame_spec: Any,
) -> tuple[_LocalResolution, ResolvedFrame]:
    """Resolve local bases once, then use that static contract to resolve the frame."""
    from quchip.engine.frames import resolve_frame

    resolution = _resolve_local_system(chip, chip.backend)
    needs_dressed_references = any(device._reference_freq_override is None for device in chip.devices)
    dressed = (
        chip.analysis._dressed_frequencies(chip.analysis.engine_result(_local_resolution=resolution))
        if needs_dressed_references
        else {}
    )
    references = {
        device.label: (
            dressed[device.label] if device._reference_freq_override is None else device._reference_freq_override
        )
        for device in chip.devices
    }
    return resolution, resolve_frame(
        chip,
        frame_spec,
        reference_frequencies=references,
    )


def _project_on_support(
    chip: "Chip",
    operator: Any,
    support: tuple[int, ...],
    bases: Mapping[str, BasisRecord],
    backend: Backend,
) -> Operator:
    """Materialize and project a local operator into the resolved support basis."""
    local = materialize_expr(operator, backend)
    if len(support) == 1:
        record = bases[chip.devices[support[0]].label]
        if record.kind == "native":
            return local
        return backend.from_array(
            record.transform_operator(backend.to_array(local)),
            dims=[[record.resolved_dim], [record.resolved_dim]],
        )
    if len(support) >= 2:
        records = [bases[chip.devices[index].label] for index in support]
        if all(record.kind == "native" for record in records):
            return local
        xp = array_namespace(records[0].vectors)
        transform = records[0].vectors
        for record in records[1:]:
            transform = xp.kron(transform, record.vectors)
        matrix = transform.conj().T @ backend.to_array(local) @ transform
        resolved_dims = [record.resolved_dim for record in records]
        return backend.from_array(
            matrix,
            dims=[resolved_dims, resolved_dims],
        )

    if all(record.kind == "native" for record in bases.values()):
        return local
    matrix = backend.to_array(local)
    authored_dimension = prod(chip.authored_dims)
    if matrix.shape != (authored_dimension, authored_dimension):
        raise ValueError(
            "A support-free operator must span the full authored chip space; "
            f"expected {(authored_dimension, authored_dimension)}, got {matrix.shape}."
        )
    ordered = [bases[device.label] for device in chip.devices]
    xp = array_namespace(ordered[0].vectors)
    transform = ordered[0].vectors
    for record in ordered[1:]:
        transform = xp.kron(transform, record.vectors)
    projected = transform.conj().T @ matrix @ transform
    resolved_dims = [record.resolved_dim for record in ordered]
    return backend.from_array(projected, dims=[resolved_dims, resolved_dims])


def _build_static_h0(
    chip: "Chip",
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    resolution: _LocalResolution,
    static_couplings: list[Operator],
) -> Operator:
    """Build the frame-subtracted static Hamiltonian in angular units.

    .. math::
        H_0 \\;=\\; 2\\pi \\left( H_{\\text{bare}}
                 - \\sum_i \\omega^{\\text{ref}}_i \\, n_i \\right)

    where ``H_bare`` is the chip-level tensored sum of device and static
    coupling contributions (ordinary GHz), and ``ω^ref_i`` is the
    per-device frame reference. This function is one of the
    four places in the engine where the 2π boundary is crossed.
    """
    dims = resolution.dims
    h0: Operator | None = None
    for idx, local in enumerate(resolution.hamiltonians):
        embedded = backend.embed(local, idx, dims)
        h0 = embedded if h0 is None else h0 + embedded
    for embedded in static_couplings:
        h0 = embedded if h0 is None else h0 + embedded
    if h0 is None:
        raise ValueError("A chip must contain at least one device.")
    h0 = TWO_PI * h0
    for idx, dev in enumerate(chip.devices):
        omega_ref = resolved_frame.frequencies.get(dev.label, 0.0)
        concrete_omega = maybe_concrete_scalar(omega_ref)
        if concrete_omega is not None and concrete_omega == 0.0:
            continue
        with _backend_context(backend):
            record = resolution.bases[dev.label]
            level_operator = _resolved_frame_operator(dev, record, backend)
            n_emb = backend.embed(level_operator, idx, dims)
        h0 = h0 - TWO_PI * omega_ref * n_emb
    return h0


def _resolved_frame_operator(device: Any, record: BasisRecord, backend: Backend) -> Operator:
    """Return one device's frame generator in its resolved solver basis."""
    from quchip.devices.spaces import FockSpace

    space = device.local_space()
    if isinstance(space, FockSpace) and record.kind == "native":
        return space.operator("n", backend)
    return backend.from_array(
        _resolved_frame_matrix(device, record),
        dims=[[record.resolved_dim], [record.resolved_dim]],
    )


def _resolved_frame_matrix(device: Any, record: BasisRecord) -> Any:
    """Return one device's frame generator as a dense resolved matrix."""
    from quchip.devices.spaces import FockSpace

    space = device.local_space()
    return (
        record.transform_operator(space.matrix("n"))
        if isinstance(space, FockSpace)
        else record.level_operator()
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
    return backend.to_canonical_operator(TWO_PI * embedded).with_metadata(
        dims=dims,
        subsystem_labels=labels,
        tag=tag,
    )


def _collect_collapse_terms(
    chip: "Chip",
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    resolution: _LocalResolution,
) -> tuple[CollapseTerm, ...]:
    """Canonicalize every collapse channel and attach accessible-port metadata."""
    labels = tuple(device.label for device in chip.devices)
    terms: list[CollapseTerm] = []
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
            local = _project_on_support(chip, operator, support, resolution.bases, backend)
            resolved_rate = materialize_expr(rate, backend)
            embedded = embed_on_support(backend, local, support, resolution.dims)
            canonical = backend.to_canonical_operator(embedded).with_metadata(
                dims=resolution.dims,
                subsystem_labels=labels,
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
    return tuple(terms)


def _concrete_port_resolution(
    chip: "Chip",
    port: Any,
    backend: Backend,
    resolution: _LocalResolution,
) -> CanonicalOperator:
    """Materialize one explicit port in its semantic product basis."""
    labels = port.resolve_targets(chip)
    support = tuple(chip._label_to_index[label] for label in labels)
    records = resolution.bases
    traced = contains_tracer(
        tuple(value for record in records.values() for value in (record.vectors, record.energy_vectors))
    )
    context = jax.ensure_compile_time_eval() if traced else nullcontext()
    with context, _backend_context(backend):
        if traced:
            concrete_records: dict[str, BasisRecord] = {}
            for device in chip.devices:
                matrix = materialize_array(device.unresolved_hamiltonian())
                policy = chip.resolve_basis(device)
                levels = device.resolved_dimension(chip.basis) if policy == "eigen" else None
                concrete_records[device.label] = resolve_local_basis(
                    matrix,
                    basis=policy,
                    levels=levels,
                )
            records = concrete_records
        operator = _project_on_support(
            chip,
            port._authored_operator(chip),
            support,
            records,
            backend,
        )
        values = np.asarray(backend.to_array(operator), dtype=complex)
        transform = _support_semantic_transform(chip, support, records)
        if transform is not None:
            concrete_transform = np.asarray(transform, dtype=complex)
            values = concrete_transform.conj().T @ values @ concrete_transform
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


def _port_frame_frequency(
    chip: "Chip",
    port: Any,
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    resolution: _LocalResolution,
) -> Any:
    """Return the common frame phase of every band in one explicit port."""
    canonical = _concrete_port_resolution(chip, port, backend, resolution)
    bands = _decompose_product_canonical_bands(canonical, canonical.dims)
    if not bands:
        raise ValueError(f"Port {port.label!r} requires a nonzero coupling operator.")

    labels = port.resolve_targets(chip)
    frequencies = tuple(resolved_frame.frequencies[label] for label in labels)
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
            f"Port {port.label!r} has operator bands with different phases in the selected frame. "
            "Use a common frame for collective terms, use the lab frame, or use QuantumSequence for time evolution."
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


def _resolve_coupling_terms(
    chip: "Chip",
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    resolution: _LocalResolution,
    approximation: Approximation,
) -> tuple[
    list[Operator],
    list[Operator],
    list[tuple[Operator, ScalarModulation]],
    list[DroppedTerm],
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
    All emitted operators already carry the 2π factor.
    """
    interactions: list[Operator] = []
    frame_corrections: list[Operator] = []
    td_terms: list[tuple[Operator, ScalarModulation]] = []
    dropped: list[DroppedTerm] = []
    dims = resolution.dims
    label_to_index = {dev.label: i for i, dev in enumerate(chip.devices)}

    for coupling in chip.couplings:
        pair = (coupling.device_a_label, coupling.device_b_label)
        idx_a = label_to_index[pair[0]]
        idx_b = label_to_index[pair[1]]
        filters_terms = approximation.filters_terms
        omega_a = resolved_frame.frequencies.get(pair[0], 0.0)
        omega_b = resolved_frame.frequencies.get(pair[1], 0.0)

        h_full = _project_on_support(
            chip,
            coupling.interaction_hamiltonian(),
            (idx_a, idx_b),
            resolution.bases,
            backend,
        )

        # In a zero frame, an Exact interaction remains wholly static and
        # needs no band decomposition.
        if not filters_terms:
            conc_a = maybe_concrete_scalar(omega_a)
            conc_b = maybe_concrete_scalar(omega_b)
            if conc_a is not None and conc_a == 0.0 and conc_b is not None and conc_b == 0.0:
                interactions.append(backend.embed_two_body(h_full, idx_a, idx_b, dims))
                continue

        d_a = resolution.bases[pair[0]].resolved_dim
        d_b = resolution.bases[pair[1]].resolved_dim
        canonical = backend.to_canonical_operator(h_full).with_metadata(
            dims=(d_a, d_b),
            subsystem_labels=pair,
            tag="coupling_local",
        )
        sub_bands = decompose_two_body_canonical_bands(
            canonical,
            [d_a, d_b],
            semantic_to_solver=_support_semantic_transform(
                chip,
                (idx_a, idx_b),
                resolution.bases,
            ),
        )
        retained: list[Operator] = []
        for (delta_a, delta_b), band_canonical in sub_bands.items():
            osc_freq = delta_a * omega_a + delta_b * omega_b
            if not approximation.keeps_operator_band((delta_a, delta_b)):
                # The advisory amplitude is the dropped band's own largest
                # matrix element — the worst-case numerator of the
                # Bloch-Siegert smallness ratio — not the coupling's scalar
                # strength, which can differ per band in a multi-term
                # interaction. Raw arithmetic; stays traced if the payload is.
                band_values = band_canonical.values
                xp = array_namespace(band_values)
                dropped.append(
                    DroppedTerm(
                        source=coupling.label,
                        operator=f"coupling band (Δa={delta_a:+d}, Δb={delta_b:+d}) on {pair[0]}·{pair[1]}",
                        reason="counter-rotating under RWA",
                        band_weights=(delta_a, delta_b),
                        amplitude=xp.max(xp.abs(band_values)),
                        frequency=abs(osc_freq),
                    )
                )
                continue
            band_op = backend.from_canonical_operator(band_canonical)
            retained.append(band_op)
            concrete_osc = maybe_concrete_scalar(osc_freq)
            if concrete_osc is not None and concrete_osc == 0.0:
                continue
            embedded = backend.embed_two_body(band_op, idx_a, idx_b, dims)
            scaled = TWO_PI * embedded
            frame_corrections.append(-scaled)
            td_terms.append((scaled, ScalarModulation(signal=Carrier(freq=TWO_PI * osc_freq, sign=-1))))

        if filters_terms and sub_bands and not retained:
            warnings.warn(
                f"Coupling {coupling.label!r} vanishes entirely under RWA().",
                UserWarning,
                stacklevel=3,
            )
        if filters_terms:
            if retained:
                local_static = sum(retained[1:], start=retained[0])
                interactions.append(backend.embed_two_body(local_static, idx_a, idx_b, dims))
        else:
            interactions.append(backend.embed_two_body(h_full, idx_a, idx_b, dims))

    return interactions, frame_corrections, td_terms, dropped


def _component_time_terms(
    chip: "Chip",
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    resolution: _LocalResolution,
    approximation: Approximation,
) -> tuple[list[DynamicTerm], list[DroppedTerm]]:
    """Project component time terms, then apply frame carriers and coupling RWA."""
    labels = tuple(device.label for device in chip.devices)
    dynamic: list[DynamicTerm] = []
    dropped: list[DroppedTerm] = []

    for local_op, coefficient, support, owner, origin, tag in chip.dynamic_contributions():
        modulation = _as_time_coefficient(coefficient, owner=type(owner).__name__)
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
        projected = _project_on_support(chip, local_op, support, resolution.bases, backend)
        bands: dict[tuple[int, ...], CanonicalOperator]
        frequencies: tuple[Any, ...]
        if len(support) == 1:
            idx = support[0]
            device = chip.devices[idx]
            dimension = resolution.bases[device.label].resolved_dim
            canonical = backend.to_canonical_operator(projected).with_metadata(
                dims=(dimension,),
                subsystem_labels=(device.label,),
                tag=tag,
            )
            bands = {
                (weight,): band
                for weight, band in decompose_canonical_bands(
                    canonical,
                    dimension,
                    semantic_to_solver=semantic_to_solver_transform(
                        device,
                        resolution.bases[device.label],
                    ),
                ).items()
            }
            frequencies = (resolved_frame.frequencies.get(device.label, 0.0),)
        elif len(support) == 2:
            idx_a, idx_b = support
            device_a = chip.devices[idx_a]
            device_b = chip.devices[idx_b]
            dim_a = resolution.bases[device_a.label].resolved_dim
            dim_b = resolution.bases[device_b.label].resolved_dim
            canonical = backend.to_canonical_operator(projected).with_metadata(
                dims=(dim_a, dim_b),
                subsystem_labels=(device_a.label, device_b.label),
                tag=tag,
            )
            bands = {
                cast(tuple[int, ...], weights): band
                for weights, band in decompose_two_body_canonical_bands(
                    canonical,
                    [dim_a, dim_b],
                    semantic_to_solver=_support_semantic_transform(
                        chip,
                        (idx_a, idx_b),
                        resolution.bases,
                    ),
                ).items()
            }
            frequencies = (
                resolved_frame.frequencies.get(device_a.label, 0.0),
                resolved_frame.frequencies.get(device_b.label, 0.0),
            )
        else:
            raise ValueError(f"Time-dependent Hamiltonian terms require one or two supports, got {support!r}.")

        for weights, band in bands.items():
            oscillation = sum(weight * frequency for weight, frequency in zip(weights, frequencies))
            if len(support) == 2 and not approximation.keeps_operator_band(weights):
                values = band.values
                xp = array_namespace(values)
                dropped.append(
                    DroppedTerm(
                        source=owner.label,
                        operator=f"time-dependent coupling band {weights}",
                        reason="counter-rotating under RWA",
                        band_weights=weights,
                        amplitude=xp.max(xp.abs(values)),
                        frequency=abs(oscillation),
                    )
                )
                continue

            band_op = backend.from_canonical_operator(band)
            embedded = embed_on_support(backend, band_op, support, resolution.dims)
            signal = modulation.signal
            concrete = maybe_concrete_scalar(oscillation)
            if concrete is None or concrete != 0.0:
                signal = Multiply((signal, Carrier(freq=TWO_PI * oscillation, sign=-1)))
            dynamic.append(
                _dynamic_term(
                    backend,
                    embedded,
                    dims=resolution.dims,
                    labels=labels,
                    tag=tag,
                    origin=cast(TermOrigin, origin),
                    time_dependence=ScalarModulation(signal=signal),
                )
            )
    return dynamic, dropped


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


class _DeliveredSignal:
    """One transformed signal paired with its destination drive and target."""

    __slots__ = ("drive", "target", "signal", "origin")

    def __init__(
        self,
        drive: BaseDrive,
        target: Any,
        signal: AnalyticSignal,
        origin: TermOrigin,
    ) -> None:
        self.drive = drive
        self.target = target
        self.signal = signal
        self.origin = origin


# -- Drive term compilation ----------------------------------------------


@dataclass(frozen=True)
class CompiledDriveTerm:
    """One projected operator band from an authored drive Hamiltonian term."""

    operator: CanonicalOperator
    delivered_index: int
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
class _StructuralDrop:
    """Template-cached pointer to a carrier-driven weight-zero band drop.

    The structural decision needs no concrete carrier value. This pointer is
    resolved into a :class:`~quchip.engine.ir.DroppedTerm` after variant signal
    construction.
    """

    delivered_index: int
    device_label: str


@dataclass(frozen=True)
class _ResolvedDriveBand:
    operator: Operator
    hamiltonian_term_index: int
    weight: int
    frame_frequency: Any
    filter_signal_bands: bool
    origin: TermOrigin
    tag: str
    target_label: str


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
) -> list[_ResolvedDriveBand]:
    """Normalize authored drive Hamiltonian operators by target support."""
    probe = AnalyticSignal(program=Constant(1.0 + 0.0j))
    with _backend_context(backend):
        authored_terms = split_dynamic_hamiltonian(drive.hamiltonian(target, probe))

    if not isinstance(drive, CouplingDrive):
        device = target
        index = chip.device_index(device.label)
        frame_frequency = resolved_frame.frequencies.get(device.label, 0.0)
        resolved: list[_ResolvedDriveBand] = []
        for term_index, (_scalar, operator) in enumerate(authored_terms):
            authored = as_operator_expr(
                operator,
                labels=(device.label,),
                dims=(device.local_space().dimension,),
                name=rf"\hat H_{{{drive.label},{term_index}}}",
                owner=drive,
                scope=f"drive.{drive.label}",
            )
            local = _project_on_support(chip, authored, (index,), bases, backend)
            for weight, embedded in embed_single_mode_bands(
                backend,
                local,
                device_index=index,
                dim=bases[device.label].resolved_dim,
                label=device.label,
                dims=dims,
                semantic_to_solver=semantic_to_solver_transform(
                    device,
                    bases[device.label],
                ),
            ):
                if isinstance(approximation, RWA) and approximation.keep_bands is not None:
                    if not approximation.keeps_operator_band((weight,)):
                        continue
                resolved.append(
                    _ResolvedDriveBand(
                        operator=embedded,
                        hamiltonian_term_index=term_index,
                        weight=weight,
                        frame_frequency=frame_frequency,
                        filter_signal_bands=approximation.filters_terms,
                        origin="drive",
                        tag="drive",
                        target_label=device.label,
                    )
                )
        return resolved

    coupling = target
    idx_a = chip.device_index(coupling.device_a_label)
    idx_b = chip.device_index(coupling.device_b_label)
    d_a = bases[coupling.device_a_label].resolved_dim
    d_b = bases[coupling.device_b_label].resolved_dim
    omega_a = resolved_frame.frequencies.get(coupling.device_a_label, 0.0)
    omega_b = resolved_frame.frequencies.get(coupling.device_b_label, 0.0)
    resolved = []
    for term_index, (_scalar, operator) in enumerate(authored_terms):
        authored = as_operator_expr(
            operator,
            labels=(coupling.device_a_label, coupling.device_b_label),
            dims=(
                chip.devices[idx_a].local_space().dimension,
                chip.devices[idx_b].local_space().dimension,
            ),
            name=rf"\hat P_{{{coupling.label},{term_index}}}",
            owner=coupling,
            scope=coupling.label,
        )
        local_op = _project_on_support(chip, authored, (idx_a, idx_b), bases, backend)
        canonical = backend.to_canonical_operator(local_op).with_metadata(
            dims=(d_a, d_b),
            subsystem_labels=(coupling.device_a_label, coupling.device_b_label),
            tag="edge_pump_local",
        )
        for (delta_a, delta_b), band_canonical in decompose_two_body_canonical_bands(
            canonical,
            [d_a, d_b],
            semantic_to_solver=_support_semantic_transform(chip, (idx_a, idx_b), bases),
        ).items():
            if not approximation.keeps_operator_band((delta_a, delta_b)):
                continue
            osc_freq = delta_a * omega_a + delta_b * omega_b
            band_op = backend.from_canonical_operator(band_canonical)
            embedded = backend.embed_two_body(band_op, idx_a, idx_b, dims)
            resolved.append(
                _ResolvedDriveBand(
                    operator=embedded,
                    hamiltonian_term_index=term_index,
                    weight=1,
                    frame_frequency=osc_freq,
                    filter_signal_bands=False,
                    origin="coupling",
                    tag="edge_pump",
                    target_label=coupling.label,
                )
            )
    return resolved


def _compile_drive_terms(
    chip: "Chip",
    delivered_signals: list[_DeliveredSignal],
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    *,
    bases: Mapping[str, BasisRecord],
    dims: tuple[int, ...],
    subsystem_labels: tuple[str, ...],
    approximation: Approximation,
) -> tuple[tuple[CompiledDriveTerm, ...], tuple[_StructuralDrop, ...]]:
    """Band-decompose each drive channel into pre-embedded, 2π-scaled drive terms.

    For each authored Hamiltonian term the operator is split into
    single-subsystem excitation-change bands of weight ``w``
    (cf. :mod:`quchip.engine.bands`). The engine combines it with the complete
    delivered signal and a frame factor of the form
    ``exp(−i w ω_ref t)``, which
    is the standard rotating-wave form for a driven multi-level system
    (Jaynes & Cummings 1963; Scully & Zubairy, *Quantum Optics*, §5). A
    carrier-driven band at weight zero under ``RWA()`` is dropped because no
    operator-frame phase can cancel either sideband. A
    :class:`_StructuralDrop` pointer carries that decision to instantiation.

    The resulting :class:`CompiledDriveTerm` is template-cached so
    homogeneous sweeps only rebuild signal-program leaves, not operators.
    """
    compiled: list[CompiledDriveTerm] = []
    structural_drops: list[_StructuralDrop] = []
    for delivered_index, delivered in enumerate(delivered_signals):
        drive = delivered.drive
        target = delivered.target
        for band in _resolved_drive_bands(
            chip,
            drive,
            target,
            resolved_frame,
            bases=bases,
            dims=dims,
            backend=backend,
            approximation=approximation,
        ):
            if (
                band.filter_signal_bands
                and delivered.signal.carrier is not None
                and band.weight == 0
            ):
                structural_drops.append(
                    _StructuralDrop(
                        delivered_index=delivered_index,
                        device_label=band.target_label,
                    )
                )
                continue
            compiled.append(
                CompiledDriveTerm(
                    operator=_apply_2pi_canonical(
                        backend,
                        band.operator,
                        dims=dims,
                        labels=subsystem_labels,
                        tag=band.tag,
                    ),
                    delivered_index=delivered_index,
                    hamiltonian_term_index=band.hamiltonian_term_index,
                    weight=band.weight,
                    device_frame_freq=band.frame_frequency,
                    filter_signal_bands=band.filter_signal_bands,
                    origin=("crosstalk" if delivered.origin == "crosstalk" else band.origin),
                    tag="crosstalk" if delivered.origin == "crosstalk" else band.tag,
                )
            )
    return tuple(compiled), tuple(structural_drops)


def _build_delivered_signals(
    chip: "Chip",
    drive_ops: list["ControlOp"],
) -> list[_DeliveredSignal]:
    """Build, transform, and route complete signals to destination drives."""
    resolved = _resolve_drives(
        chip,
        [operation for operation in drive_ops if isinstance(operation, DriveOp)],
    )
    raw_signals: dict[tuple[str, int], AnalyticSignal] = {}
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
        return []

    line_map = {line.label: line for line in equipment.lines}
    delivered: list[_DeliveredSignal] = []
    for (destination_label, source_index), signal in transformed.items():
        destination = line_map.get(destination_label)
        if destination is None or destination._target is None:
            continue
        source_drive = resolved[source_index][0]
        delivered.append(
            _DeliveredSignal(
                drive=destination,
                target=destination._target,
                signal=signal,
                origin=("drive" if destination.label == source_drive.label else "crosstalk"),
            )
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
        for output_index, channel in enumerate(external):
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
                    frame_frequency=(
                        0.0
                        if channel.collapse.frame_frequency is None
                        else channel.collapse.frame_frequency
                    ),
                )
            )
    return tuple(compiled)


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
    external = {channel.key: channel for channel in template.slh.external_channels}
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
        boundary = (
            reference
            if _is_concrete_zero(channel.reference_delay)
            else reference.shifted(channel.reference_delay)
        )
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
    that declare ``band_weights`` without a frequency get the band's
    frame oscillation resolved here as ``|Σ wᵢ·f_ref,i|`` — raw
    arithmetic on possibly-traced frame frequencies, never concretized.
    """
    gathered: list[DroppedTerm] = []
    for coupling in chip.couplings:
        endpoint_labels = (coupling.device_a_label, coupling.device_b_label)
        for record in coupling.dropped_terms():
            weights = record.band_weights
            if record.frequency is None and weights is not None and len(weights) == len(endpoint_labels):
                freq = sum(w * resolved_frame.frequencies.get(lbl, 0.0) for w, lbl in zip(weights, endpoint_labels))
                record = replace(record, frequency=abs(freq))
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

    dims = base_result.dims
    subsystem_labels = tuple(device.label for device in chip.devices)
    drive_terms, weight_zero_drops = _compile_drive_terms(
        chip,
        _build_delivered_signals(chip, drive_ops),
        resolved_frame,
        chip.backend,
        bases=base_result.bases,
        dims=dims,
        subsystem_labels=subsystem_labels,
        approximation=base_result.approximation,
    )
    coherent_terms = _compile_coherent_terms(base_result.slh, drive_ops)
    return HamiltonianTemplate(
        resolved_frame=resolved_frame,
        approximation=base_result.approximation,
        dims=dims,
        slh=base_result.slh,
        static_terms=base_result.slh.H.static_terms,
        invariant_dynamic_terms=base_result.slh.H.dynamic_terms,
        drive_terms=drive_terms,
        coherent_terms=coherent_terms,
        reference_drive_ops=tuple(drive_ops),
        dropped_terms=base_result.dropped_terms,
        weight_zero_drops=weight_zero_drops,
        static_spectral_bound_ghz=base_result.metadata.get("static_spectral_bound_ghz"),
        collapse_terms=base_result.collapse_terms,
        bases=base_result.bases,
        authored=base_result.authored,
    )


def compile_hamiltonian_template(
    chip: "Chip",
    drive_ops: list["ControlOp"],
    *,
    resolved_frame: "ResolvedFrame",
    approximation: Approximation | None = None,
    _local_resolution: _LocalResolution | None = None,
    _base_result: EngineResult | None = None,
) -> HamiltonianTemplate:
    """Compile the invariant Hamiltonian skeleton (H₀, couplings, pre-embedded drive bands).

    Everything that does not change across a homogeneous sweep lives in
    the template: static Hamiltonian, static-coupling folds, invariant
    dynamic couplings, and band-decomposed drive operators pre-embedded
    and pre-scaled by 2π. Per-sweep instantiation
    (:func:`instantiate_engine_result`) rebuilds only the
    :class:`~quchip.engine.ir.SignalProgram` leaves, so envelope
    parameters, drive frequencies, phases, and frame scalars can sweep
    through JAX without retracing operator tensors.
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
    resolution = _resolve_local_system(chip, backend) if _local_resolution is None else _local_resolution
    dims = resolution.dims
    subsystem_labels = tuple(d.label for d in chip.devices)

    coupling_h0, coupling_frame_static, coupling_td, coupling_dropped = _resolve_coupling_terms(
        chip,
        resolved_frame,
        backend,
        resolution,
        strategy,
    )
    h0 = _build_static_h0(
        chip,
        resolved_frame,
        backend,
        resolution,
        coupling_h0,
    )
    for op in coupling_frame_static:
        h0 = h0 + op

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

    component_dynamic, component_dropped = _component_time_terms(
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

    drive_terms, weight_zero_drops = _compile_drive_terms(
        chip,
        _build_delivered_signals(chip, drive_ops),
        resolved_frame,
        backend,
        bases=resolution.bases,
        dims=dims,
        subsystem_labels=subsystem_labels,
        approximation=strategy,
    )

    collapse_terms = _collect_collapse_terms(chip, resolved_frame, backend, resolution)
    slh = ResolvedSLH.from_terms(
        static_terms=static_terms,
        dynamic_terms=simplified_invariant,
        collapse_terms=collapse_terms,
    )
    if chip.port_network is not None:
        slh = chip.port_network.resolve(slh)
    coherent_terms = _compile_coherent_terms(slh, drive_ops)
    # Static terms are invariant across a homogeneous sweep, including any
    # Hamiltonian generated by network composition. Store the bound in GHz.
    resolved_static_span = _static_diagonal_span(slh.H.static_terms)
    static_spectral_bound_ghz = (
        resolved_static_span / TWO_PI if resolved_static_span is not None else None
    )
    return HamiltonianTemplate(
        resolved_frame=resolved_frame,
        approximation=strategy,
        dims=dims,
        slh=slh,
        static_terms=slh.H.static_terms,
        invariant_dynamic_terms=slh.H.dynamic_terms,
        drive_terms=drive_terms,
        coherent_terms=coherent_terms,
        reference_drive_ops=tuple(drive_ops),
        dropped_terms=_collect_dropped_terms(chip, resolved_frame) + tuple(coupling_dropped),
        weight_zero_drops=weight_zero_drops,
        static_spectral_bound_ghz=static_spectral_bound_ghz,
        collapse_terms=slh.collapse_terms,
        bases=resolution.bases,
        authored=chip.unresolved_hamiltonian(),
    )


def instantiate_engine_result(
    template: HamiltonianTemplate,
    drive_ops: list["ControlOp"],
    chip: "Chip",
) -> EngineResult:
    """Rebuild signal-program leaves from *drive_ops* and attach them to the template's operators."""
    _validate_variant_drive_ops(template, drive_ops)
    dims = template.dims

    delivered = _build_delivered_signals(chip, drive_ops)

    # Only the variant-specific terms built below need simplification;
    # the template's invariant terms were simplified at compile time.
    # Drive frequencies are per-op, so the fast partners the RWA drops
    # only become auditable here — their records join the template's.
    fresh_terms: list[DynamicTerm] = []
    fresh_dropped: list[DroppedTerm] = []
    for compiled in template.drive_terms:
        item = delivered[compiled.delivered_index]
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
                        template.approximation,
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
                    frequency=(item.signal.carrier + abs(compiled.weight) * compiled.device_frame_freq),
                )
            )

    # Structural weight-zero drops carry no concrete frequency at compile
    # time; resolve each pointer against its delivered variant signal now.
    for drop in template.weight_zero_drops:
        item = delivered[drop.delivered_index]
        fresh_dropped.append(
            _weight_zero_dropped_term(
                source=item.drive.label,
                device_label=drop.device_label,
                drive_freq=item.signal.carrier,
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
    dynamic_terms = template.invariant_dynamic_terms + applied_dynamic_terms

    metadata: dict[str, Any] = {"frame": str(template.resolved_frame)}
    if template.static_spectral_bound_ghz is not None:
        metadata["static_spectral_bound_ghz"] = template.static_spectral_bound_ghz
    metadata.update(_solver_hint_metadata(template.static_spectral_bound_ghz, dynamic_terms))

    slh = replace(
        template.slh,
        hamiltonian=HamiltonianProgram(
            static_terms=template.static_terms,
            dynamic_terms=template.invariant_dynamic_terms,
        ),
    )

    return EngineResult(
        slh=slh,
        applied_hamiltonian=HamiltonianProgram(dynamic_terms=applied_dynamic_terms),
        coherent_inputs=coherent_inputs,
        dims=dims,
        metadata=metadata,
        dropped_terms=template.dropped_terms + tuple(fresh_dropped),
        bases=template.bases,
        authored=template.authored,
        resolved_frame=template.resolved_frame,
        approximation=template.approximation,
    )


def build_engine_result(
    chip: "Chip",
    drive_ops: list["ControlOp"],
    *,
    resolved_frame: "ResolvedFrame",
    approximation: Approximation | None = None,
    _local_resolution: _LocalResolution | None = None,
    _base_result: EngineResult | None = None,
) -> EngineResult:
    """Compile the template and instantiate one engine-result variant.

    Equivalent to
    :func:`compile_hamiltonian_template` followed by
    :func:`instantiate_engine_result` with the same
    ``drive_ops``. Prefer the two-step form when solving many variants
    that share the same chip topology.

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
    )
    return instantiate_engine_result(template, drive_ops, chip)


def _build_static_analysis_result(
    chip: "Chip",
    *,
    _local_resolution: _LocalResolution | None = None,
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
        approximation=Exact(),
        _local_resolution=_local_resolution,
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


def _analysis_matrix_ghz(result: EngineResult) -> Any:
    """Return the resolved static solver Hamiltonian in ordinary GHz.

    This internal path reads canonical arrays directly so JAX tracers never
    cross through a non-JAX inspection backend. Time-dependent terms are not
    part of a static dressed-state calculation.
    """
    terms = [term.coefficient * term.operator.to_dense() / TWO_PI for term in result.static_terms]
    if not terms:
        raise ValueError("EngineResult contains no static Hamiltonian terms.")
    return sum(terms[1:], start=terms[0])
