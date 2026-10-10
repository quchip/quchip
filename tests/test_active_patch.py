"""Schedule-aware active-patch reduction (spec: 2026-07-11-partition-active-patch)."""

from __future__ import annotations

from quchip.approximations import RWA

import numpy as np
import pytest

from quchip import (
    Bath, Capacitive, ChargeDrive, Chip, DuffingTransmon, Gaussian, QuantumSequence, PortNetwork,
)
from quchip.chip.transformations.active_patch import coupling_adjacency, graph_distances


def _chain(n=4, g=0.004):
    qs = [DuffingTransmon(freq=5.0 + 0.35 * i, anharmonicity=-0.25, levels=3, label=f"q{i}") for i in range(n)]
    couplings = [Capacitive(qs[i], qs[i + 1], g=g, label=f"c{i}{i + 1}") for i in range(n - 1)]
    chip = Chip(qs, couplings=couplings, frame="rotating", approximation=RWA())
    drives = [ChargeDrive(target=q, label=f"d{i}") for i, q in enumerate(qs)]
    chip.wire(*drives)
    return chip, qs, drives


def _driven_pair_chain():
    chip, qs, drives = _chain()
    seq = QuantumSequence(chip)
    seq.schedule(drives[0], envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(qs[0]))
    return chip, seq


def test_graph_distances():
    """graph_distances reports BFS hop-distance from every source label."""
    chip, _, _ = _chain()
    adj = coupling_adjacency(chip)
    dist = graph_distances(adj, {"q0", "q1"})
    assert dist == {"q0": 0, "q1": 0, "q2": 1, "q3": 2}


def test_active_patch_trivial_when_everything_active():
    """An all-active patch has an independently editable model and schedule."""
    chip, seq = _driven_pair_chain()
    patch = seq.active_patch(hops=3)
    assert patch.eliminated_labels == ()
    patch.chip["q0"].freq = 6.1
    assert chip["q0"].freq == 5.0
    patch.sequence.delay("q0", duration=1.0)
    assert patch.sequence.total_duration == seq.total_duration + 1.0


def test_active_patch_stops_gracefully_on_an_unsupported_device_elimination():
    """active_patch downgrades a declined device elimination to a note and keeps that spectator on the patch chip."""
    # Joint elimination of a nonlinear accessible boundary is unsupported.
    # Keep that spectator and report why the reduction stopped.
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0", thermal_occupation=0.02)
    spec = DuffingTransmon(freq=5.4, anharmonicity=-0.25, levels=3, label="spec", T1=20_000.0)
    network = PortNetwork()
    network.port("probe", target=spec, rate=.01)
    chip = Chip([q0, spec], couplings=[Capacitive(q0, spec, g=0.004, label="c0s")],
                frame="rotating", port_network=network)
    drive = ChargeDrive(target=q0, label="d0")
    chip.wire(drive)
    seq = QuantumSequence(chip)
    seq.schedule(drive, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(q0))

    patch = seq.active_patch(hops=0)
    assert patch.eliminated_labels == ()
    assert {d.label for d in patch.chip.devices} == {"q0", "spec"}
    assert any("stopped eliminating" in note and "spec" in note for note in patch.notes)


def test_active_patch_matches_full_solve_in_dispersive_regime():
    """In the dispersive regime, the active-patch solve tracks the full solve on the driven qubit."""
    # _driven_pair_chain's spectators are far-detuned enough for good SW validity.
    chip, seq = _driven_pair_chain()
    tlist = np.linspace(0.0, 20.0, 41)

    full = seq.simulate(tlist=tlist, e_ops=chip.e_ops(q0="Z"), partition=False)
    patch = seq.active_patch(hops=1)
    # e_ops takes built operators (chip.e_ops(...)), not raw name strings —
    # decompose_eops (engine/observables.py) expects the former, and
    # the patch chip has its own local Hilbert space so the operators must
    # be built against patch.chip, not chip.
    reduced = patch.simulate(tlist=tlist, e_ops=patch.chip.e_ops(q0="Z"))

    z_full = np.asarray(full.expect("q0"))
    z_patch = np.asarray(reduced.expect("q0"))
    # Dispersive SW error at the active/spectator boundary (c12: q1-q2) scales as
    # (g/Delta)^2 = (0.004/0.35)^2 = 1.31e-4; 50x gives comfortable headroom over
    # that leading-order estimate (observed deviation is ~6e-7, far inside it).
    g = chip.coupling("c12").g
    delta = abs(chip.freq("q2") - chip.freq("q1"))
    tol = 50 * (g / delta) ** 2
    assert np.max(np.abs(z_full - z_patch)) < tol


def test_active_patch_warns_on_poor_sw_validity():
    """active_patch warns, without raising, when a fold's Schrieffer-Wolff validity comes back poor."""
    # g=0.05 at a 0.02 GHz detuning is near-resonant (g/Delta >> 0.1), so the fold triggers is_valid=False.
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    spec = DuffingTransmon(freq=5.02, anharmonicity=-0.25, levels=3, label="spec")
    chip = Chip([q0, spec], couplings=[Capacitive(q0, spec, g=0.05, label="c01")], frame="rotating")
    drive = ChargeDrive(target=q0, label="d0")
    chip.wire(drive)
    seq = QuantumSequence(chip)
    seq.schedule(drive, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(q0))

    with pytest.warns(UserWarning, match="Schrieffer-Wolff validity"):
        patch = seq.active_patch(hops=0)
    assert patch.eliminated_labels == ("spec",)
    assert not patch.validity["spec"]["c01"]["is_valid"]


def test_active_patch_retains_a_bath_targeting_a_spectator():
    chip, qs, drives = _chain()
    chip.add_bath(Bath("thermal", targets=[qs[3]], temperature=20.0))
    seq = QuantumSequence(chip)
    seq.schedule(drives[0], envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(qs[0]))
    patch = seq.active_patch(hops=1)
    assert qs[3].label in patch.eliminated_labels
    assert patch.chip.baths[0].resolve_targets(patch.chip) == [d.label for d in patch.chip.devices]
