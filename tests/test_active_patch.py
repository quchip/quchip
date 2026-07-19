"""Schedule-aware active-patch reduction (spec: 2026-07-11-partition-active-patch)."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import Bath, Capacitive, ChargeDrive, Chip, CrossKerr, DuffingTransmon, Gaussian, QuantumSequence
from quchip.chip.transformations import ActivePatchResult
from quchip.chip.transformations.active_patch import active_labels, coupling_adjacency, graph_distances


def _chain(n=4, g=0.004):
    qs = [DuffingTransmon(freq=5.0 + 0.35 * i, anharmonicity=-0.25, levels=3, label=f"q{i}") for i in range(n)]
    couplings = [Capacitive(qs[i], qs[i + 1], g=g, label=f"c{i}{i+1}") for i in range(n - 1)]
    chip = Chip(qs, couplings=couplings, frame="rotating", rwa=True)
    drives = [ChargeDrive(target=q, label=f"d{i}") for i, q in enumerate(qs)]
    chip.wire(*drives)
    return chip, qs, drives


def _driven_pair_chain():
    chip, qs, drives = _chain()
    seq = QuantumSequence(chip)
    seq.schedule(drives[0], envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(qs[0]))
    return chip, seq


def test_active_labels_from_schedule_plus_hops():
    """active_labels expands the scheduled targets by exactly ``hops`` coupling-graph steps."""
    _, seq = _driven_pair_chain()
    assert active_labels(seq, hops=0) == {"q0"}
    assert active_labels(seq, hops=1) == {"q0", "q1"}
    assert active_labels(seq, hops=3) == {"q0", "q1", "q2", "q3"}


def test_active_labels_empty_schedule_raises():
    """An empty schedule has no active patch to reduce to; active_labels raises ValueError."""
    chip, _, _ = _chain()
    seq = QuantumSequence(chip)
    with pytest.raises(ValueError, match="schedule"):
        active_labels(seq)


def test_graph_distances():
    """graph_distances reports BFS hop-distance from every source label."""
    chip, _, _ = _chain()
    adj = coupling_adjacency(chip)
    dist = graph_distances(adj, {"q0", "q1"})
    assert dist == {"q0": 0, "q1": 0, "q2": 1, "q3": 2}


def test_active_patch_eliminates_spectators_leaf_first():
    """active_patch eliminates spectators farthest-from-active first, leaving only active devices on the patch chip."""
    chip, seq = _driven_pair_chain()
    patch = seq.active_patch(hops=1)
    assert isinstance(patch, ActivePatchResult)
    assert patch.active_labels == ("q0", "q1")
    assert patch.eliminated_labels == ("q3", "q2")
    assert [d.label for d in patch.chip.devices] == ["q0", "q1"]
    assert set(patch.validity) == {"q3", "q2"}
    assert patch.chip is not chip and len(chip.devices) == 4


def test_active_patch_trivial_when_everything_active():
    """When every device is schedule-active, active_patch returns the source chip and sequence unchanged."""
    chip, seq = _driven_pair_chain()
    patch = seq.active_patch(hops=3)
    assert patch.eliminated_labels == ()
    assert patch.chip is chip
    assert patch.sequence is seq


def test_active_patch_strips_unused_spectator_lines():
    """active_patch drops control lines that only ever target an eliminated spectator."""
    chip, seq = _driven_pair_chain()
    patch = seq.active_patch(hops=1)
    surviving = [ln.label for ln in patch.chip.control_equipment.lines]
    assert "d0" in surviving and "d3" not in surviving
    assert any("d3" in note for note in patch.notes)


def test_active_patch_sequence_rebinds_and_simulates():
    """The patch sequence binds to the reduced chip and simulates directly on it."""
    chip, seq = _driven_pair_chain()
    patch = seq.active_patch(hops=1)
    # e_ops takes built operators (chip.e_ops(...)), not raw name strings —
    # decompose_eops (engine/stage3_observables.py) expects the former.
    e_ops = patch.chip.e_ops(q0="Z")
    result = patch.simulate(tlist=np.linspace(0.0, 20.0, 21), e_ops=e_ops)
    assert np.asarray(result.expect("q0")).shape == (21,)


def _spectator_diamond_with_pendant():
    """5-device star/kite: active bridges A,B which both bridge C, plus a leaf D off C.

    The spectator cycle forces chained folds through fold-created edges: the
    fold at C mediates a new exchange between A and B as a ``Capacitive``
    edge, so the later A and B eliminations must read that edge's ``g`` as a
    leg strength (A) and as the final leaf coupling (B, after A's fold updates
    the direct active-B edge in place).
    """
    active = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="active")
    a = DuffingTransmon(freq=5.4, anharmonicity=-0.25, levels=3, label="A")
    b = DuffingTransmon(freq=5.8, anharmonicity=-0.25, levels=3, label="B")
    c = DuffingTransmon(freq=6.2, anharmonicity=-0.25, levels=3, label="C")
    d = DuffingTransmon(freq=6.6, anharmonicity=-0.25, levels=3, label="D")
    couplings = [
        Capacitive(active, a, g=0.004, label="c_aA"),
        Capacitive(active, b, g=0.004, label="c_aB"),
        Capacitive(a, c, g=0.004, label="c_AC"),
        Capacitive(b, c, g=0.004, label="c_BC"),
        Capacitive(c, d, g=0.004, label="c_CD"),
    ]
    chip = Chip([active, a, b, c, d], couplings=couplings, frame="rotating", rwa=True)
    drive = ChargeDrive(target=active, label="d_active")
    chip.wire(drive)
    seq = QuantumSequence(chip)
    seq.schedule(drive, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(active))
    return chip, seq


def test_active_patch_folds_through_fold_created_edges():
    """Chained folds read bridging edges earlier folds created, reducing a spectator cycle to the driven device."""
    chip, seq = _spectator_diamond_with_pendant()
    patch = seq.active_patch(hops=0)
    assert patch.eliminated_labels == ("D", "C", "A", "B")
    assert [d.label for d in patch.chip.devices] == ["active"]
    assert not any("stopped eliminating" in note for note in patch.notes)
    assert set(patch.validity) == {"D", "C", "A", "B"}


def test_active_patch_stops_gracefully_on_an_unsupported_device_elimination():
    """active_patch downgrades a declined device elimination to a note and keeps that spectator on the patch chip."""
    # eliminate() declines (NotImplementedError) when a spectator's Purcell
    # decay would fold onto a survivor that carries thermal_population —
    # eliminate_device.py has no collapse-channel API to represent the
    # resulting rate without inventing thermal absorption that was never
    # physically present (see eliminate_device.py's T1/thermal_population
    # guard). active_patch must downgrade that to a "stopped eliminating"
    # note rather than raising.
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0", thermal_population=0.02)
    spec = DuffingTransmon(freq=5.4, anharmonicity=-0.25, levels=3, label="spec", T1=20_000.0)
    chip = Chip([q0, spec], couplings=[Capacitive(q0, spec, g=0.004, label="c0s")], frame="rotating", rwa=True)
    drive = ChargeDrive(target=q0, label="d0")
    chip.wire(drive)
    seq = QuantumSequence(chip)
    seq.schedule(drive, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(q0))

    patch = seq.active_patch(hops=0)
    assert patch.eliminated_labels == ()
    assert {d.label for d in patch.chip.devices} == {"q0", "spec"}
    assert any("stopped eliminating" in note and "spec" in note for note in patch.notes)


def test_active_patch_eliminates_a_crosskerr_coupled_spectator():
    """A CrossKerr-coupled spectator folds like any other; active_patch eliminates it with no stopped note."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    spec = DuffingTransmon(freq=6.0, anharmonicity=-0.25, levels=3, label="spec")
    chip = Chip([q0, spec], couplings=[CrossKerr(q0, spec, chi=-0.001, label="ck0s")], frame="rotating", rwa=True)
    drive = ChargeDrive(target=q0, label="d0")
    chip.wire(drive)
    seq = QuantumSequence(chip)
    seq.schedule(drive, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(q0))

    patch = seq.active_patch(hops=0)
    assert patch.eliminated_labels == ("spec",)
    assert not any("stopped eliminating" in note for note in patch.notes)


def test_active_patch_matches_full_solve_in_dispersive_regime():
    """In the dispersive regime, the active-patch solve tracks the full solve on the driven qubit."""
    # _driven_pair_chain's spectators are far-detuned enough for good SW validity.
    chip, seq = _driven_pair_chain()
    tlist = np.linspace(0.0, 20.0, 41)

    full = seq.simulate(tlist=tlist, e_ops=chip.e_ops(q0="Z"), partition=False)
    patch = seq.active_patch(hops=1)
    # e_ops takes built operators (chip.e_ops(...)), not raw name strings —
    # decompose_eops (engine/stage3_observables.py) expects the former, and
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
    chip = Chip([q0, spec], couplings=[Capacitive(q0, spec, g=0.05, label="c01")], frame="rotating", rwa=True)
    drive = ChargeDrive(target=q0, label="d0")
    chip.wire(drive)
    seq = QuantumSequence(chip)
    seq.schedule(drive, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(q0))

    with pytest.warns(UserWarning, match="Schrieffer-Wolff validity"):
        patch = seq.active_patch(hops=0)
    assert patch.eliminated_labels == ("spec",)
    assert patch.validity["spec"]["c01"]["is_valid"] is False


def test_active_patch_raises_when_bath_explicitly_targets_a_spectator():
    """A bath that explicitly targets a spectator is a real conflict; active_patch propagates eliminate()'s error."""
    # Eliminating that spectator would dangle the bath's target and raise
    # KeyError at solve time instead; active_patch's graceful-stop catch
    # must not convert this into a soft "stopped eliminating" note.
    chip, qs, drives = _chain()
    chip.add_bath(Bath("thermal", targets=[qs[3]], temperature=20.0))
    seq = QuantumSequence(chip)
    seq.schedule(drives[0], envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(qs[0]))
    with pytest.raises(ValueError, match="explicitly targets it"):
        seq.active_patch(hops=1)
