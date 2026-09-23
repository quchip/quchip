"""Backend-neutral input-output assembly for stationary port calculations.

Accessible channels are the port-marked subset of
:class:`~quchip.engine.ir.CollapseTerm`. This module derives coherent input Hamiltonians, output coupling
operators, and stationary-frame checks from those resolved terms. Numerical
Liouvillian construction and solution remain backend-owned.
"""

from __future__ import annotations

from typing import Any

from quchip.engine.frames import FrameConflict, plan_frame, planning_resolution, same_frequency, stationary_tones
from quchip.engine.ir import CanonicalOperator, EngineResult, StaticTerm
from quchip.engine.reference import cw_transfer


def resolve_stationary_engine(
    chip: Any,
    port_frequencies: tuple[tuple[str, Any], ...],
) -> EngineResult:
    """Resolve port-tone frequencies into one static engine description."""
    if not port_frequencies:
        raise ValueError("Stationary port analysis requires at least one tone frequency.")

    tail = "Use QuantumSequence for time evolution; periodic/Floquet steady states are not supported."
    resolution = planning_resolution(chip)
    tones = stationary_tones(chip, port_frequencies, resolution=resolution)
    try:
        plan = plan_frame(chip, tones, approximation=chip.approximation, strict=True, local_resolution=resolution)
    except FrameConflict as conflict:
        if conflict.kind == "port":
            raise ValueError(f"Ports address {conflict.devices[0]!r} with distinct stationary tones. {tail}") from None
        raise ValueError(
            f"Exchange-connected devices {list(conflict.cluster)!r} are addressed by distinct stationary tones. {tail}"
        ) from None
    engine = chip._resolve(frame=plan, _resolution=resolution)
    if engine.dynamic_terms:
        raise ValueError(
            f"The selected tones leave dynamic Hamiltonian terms after frame and approximation resolution. {tail}"
        )
    _validate_port_frequencies(engine, port_frequencies)
    return engine


def port_operators(engine: EngineResult, backend: Any) -> dict[str, CanonicalOperator]:
    """Return each resolved channel coupling ``L_p = exp(i phi) sqrt(kappa) A_p``."""
    operators: dict[str, CanonicalOperator] = {}
    for channel in engine.slh.channels:
        operators[channel.key] = channel.coupling.with_metadata(tag=f"port:{channel.key}")
    return operators


def add_port_inputs(
    engine: EngineResult,
    backend: Any,
    tones: tuple[tuple[str, Any, Any], ...],
) -> EngineResult:
    """Add stationary ``i(beta* L - beta L†)`` terms in angular units."""
    if not tones:
        return engine
    xp = backend.array_module
    external = engine.slh.external_channels
    operators = port_operators(engine, backend)
    exposure_index = {channel.key: index for index, channel in enumerate(external)}
    _validate_port_frequencies(
        engine,
        tuple((port_label, frequency) for port_label, frequency, _ in tones),
    )
    incident = [xp.asarray(0.0 + 0.0j) for _ in external]
    for port_label, frequency, amplitude in tones:
        input_index = exposure_index[port_label]
        incident[input_index] = incident[input_index] + xp.asarray(amplitude) * cw_transfer(
            external[input_index].reference.inbound, frequency, xp
        )

    coefficients = xp.asarray(engine.slh.S)[:, :len(external)] @ xp.asarray(incident)
    terms = list(engine.applied_hamiltonian.static_terms)
    for output_index, channel in enumerate(engine.slh.channels):
        coefficient = coefficients[output_index]
        coupling = operators[channel.key]
        values = coupling.to_dense()
        h_input = 1j * (
            xp.conj(coefficient) * values
            - coefficient * xp.conj(xp.swapaxes(values, -1, -2))
        )
        terms.append(
            StaticTerm(
                CanonicalOperator.from_dense(
                    h_input,
                    dims=coupling.dims,
                    basis=coupling.basis,
                    subsystem_labels=coupling.subsystem_labels,
                    tag=f"input:{channel.key}",
                ),
                origin="port",
                metadata={"port": channel.key},
            )
        )
    return engine.with_applied_hamiltonian_terms(static_terms=tuple(terms))


def _validate_port_frequencies(
    engine: EngineResult,
    port_frequencies: tuple[tuple[str, Any], ...],
) -> None:
    """Check that every channel a tone feeds is stationary in the tone's frame."""
    external = {channel.key: index for index, channel in enumerate(engine.slh.external_channels)}
    for port_label, frequency in port_frequencies:
        if port_label not in external:
            raise ValueError(f"Unknown resolved port {port_label!r}.")
        column = external[port_label]
        for row, channel in enumerate(engine.slh.channels):
            resolved = channel.collapse.frame_frequency
            if resolved is None or not engine.slh.feeds(row, column):
                continue
            if not same_frequency(resolved, frequency):
                raise ValueError(
                    f"Tone at {frequency!r} GHz entering {port_label!r} reaches channel "
                    f"{channel.key!r}, which is stationary at {resolved!r} GHz. "
                    "Use QuantumSequence for time evolution."
                )
