"""Weak-probe commutator sources for a frozen dressed reference."""
from __future__ import annotations

from typing import Any

from quchip.engine.ir import CoherenceResponseProblem, LinearResponseProblem
from quchip.engine.linear_response import (
    _UnsupportedLinearModel, _build_vacuum_response_problem, _sector_response_model,
)
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer


def build_conditional_response(
    chip: Any, frequencies: Any, labels: tuple[str, ...], reference: tuple[int, ...],
) -> tuple[LinearResponseProblem | CoherenceResponseProblem, dict[str, Any]]:
    """Build the closed adjacent-sector response, with the reference frozen only in its source."""
    try:
        model = _sector_response_model(chip, labels)
    except _UnsupportedLinearModel as error:
        raise ValueError(
            "Conditional VNA requires a static excitation-conserving model, lowering ports, "
            "lowering or conserving loss channels, vacuum inputs and no downstream output network."
        ) from error
    sectors, xp = model.sectors, chip.backend.array_module
    if any(level >= dim for level, dim in zip(reference, sectors.dims, strict=True)):
        raise ValueError("Conditional VNA reference exceeds a device's resolved level count.")
    total = sum(reference)
    metadata: dict[str, Any] = {
        "reference": tuple(zip((d.label for d in chip.devices), reference, strict=True)),
        "approximation": "none" if total == 0 else "frozen_reference",
    }
    if total == 0:
        problem = _build_vacuum_response_problem(chip, frequencies, plane_labels=labels)
        metadata.update(solver="vacuum_response", reference_overlap=1., reference_escape_rate=0.,
                        reference_stationarity_residual=0., mode_count=len(problem.mode_labels))
        return problem, metadata

    if chip.port_network._mode_responses:
        raise NotImplementedError(
            "Retained multiport mode responses support the vacuum reference only. "
            "Keep the boundary mode for an excited-reference VNA sweep."
        )
    system = sectors.eigensystem(total)
    row = system.space.row(reference)
    index = system.labeling.indices[row]
    overlap = xp.asarray(system.labeling.overlaps[row])
    vector = xp.asarray(system.eigenvectors)[:, index]
    invalid = (overlap <= 0.5 + 32 * xp.finfo(overlap.dtype).eps) | xp.any(system.labeling.duplicates)
    message = "Conditional VNA requires an unambiguous dressed reference with bare-state overlap greater than one half."
    if contains_tracer(invalid):
        import equinox as eqx

        vector = eqx.error_if(vector, invalid, message)
    elif bool(invalid):
        raise ValueError(message)
    rho = xp.outer(vector, vector.conj())
    energy = system.eigenvalues[index]
    sizes = tuple(sectors.space(n).size for n in range(total + 2))
    from quchip.utils.memory import dense_bytes, require_memory

    channel_count = len(model.channels)
    block_entries = sum(size**2 for size in sizes) * (len(model.conserving_indices) + 4)
    block_entries += sum(sizes[n] * sizes[n - 1] for n in range(1, len(sizes))) * (channel_count + len(labels))
    require_memory(dense_bytes(block_entries, copies=2), task="Conditional VNA sector operators",
                   remedy="Reduce the model or its local level counts.")
    hamiltonians = tuple(TWO_PI * (xp.asarray(sectors.hamiltonian(n)) - energy * xp.eye(size))
                         for n, size in enumerate(sizes))
    count = len(model.channels)
    lowering = (xp.zeros((count, 0, sizes[0]), dtype=complex),) + tuple(
        xp.stack([model.channel_block(k, n - 1, n) for k in range(count)]) for n in range(1, total + 2))
    conserving = tuple(xp.stack([model.channel_block(k, n, n) for k in model.conserving_indices])
                       if model.conserving_indices else xp.zeros((0, size, size), dtype=complex)
                       for n, size in enumerate(sizes))
    indices = list(model.plane_indices)
    mixing = model.scattering[:, indices]
    sources = [xp.zeros((sizes[0], 0, len(indices)), dtype=complex)]
    for n in range(1, total + 2):
        # K_i^dagger = sum_k S_ki L_k^dagger. The source is -[K_i^dagger, rho].
        dagger = xp.einsum("kab,ki->bai", lowering[n].conj(), mixing)
        sources.append(-xp.einsum("abi,bc->aci", dagger, rho) if n == total + 1
                       else xp.einsum("ab,bci->aci", rho, dagger))

    lower = lowering[total]
    kept = conserving[total]
    lower_state, kept_state = lower @ vector, kept @ vector
    means = xp.einsum("a,ka->k", vector.conj(), kept_state)
    escape = xp.real(xp.sum(xp.abs(lower_state)**2) + xp.sum(xp.abs(kept_state)**2) - xp.sum(xp.abs(means)**2))
    damping = xp.einsum("kab,kac->bc", lower.conj(), lower) + xp.einsum("kab,kac->bc", kept.conj(), kept)
    retained_derivative = xp.einsum("kab,bc,kdc->ad", kept, rho, kept.conj()) - (damping @ rho + rho @ damping) / 2
    escaped_derivative = xp.einsum("kab,bc,kdc->ad", lower, rho, lower.conj())
    metadata.update(
        solver="conditional_response", reference_overlap=overlap, reference_escape_rate=xp.maximum(escape, 0.),
        reference_stationarity_residual=xp.sqrt(xp.sum(xp.abs(retained_derivative)**2)
                                               + xp.sum(xp.abs(escaped_derivative)**2)),
        sector_dimensions=sizes, coherence_count=sum(sizes[n] * sizes[n - 1] for n in range(1, len(sizes))),
    )
    inbound, outbound = model.transfers(frequencies)
    return CoherenceResponseProblem(
        frequencies=frequencies, hamiltonians=hamiltonians, lowering=lowering, conserving=conserving,
        sources=tuple(sources), scattering=model.scattering, plane_indices=model.plane_indices,
        inbound_transfer=inbound, outbound_transfer=outbound,
    ), metadata
