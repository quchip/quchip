"""Sentinel: active_patch + automatic partitioning reproduce a full multiplexed-readout solve.

Geometry: two driven qubit-resonator pairs (q0-r0, q1-r1) bridged through a
bus resonator to a spectator transmon (r0-bus-spec). ``hops=1`` from the
driven pair {q0, q1} only reaches their own resonators {r0, r1} — bus sits
two hops out and spec three, so both stay spectators and active_patch()
folds them away (farthest-first: spec, then bus). With bus gone, the patch
chip has no edge left between the two pairs, so QuantumSequence.simulate's
automatic partitioning splits the patch into two independent components.
"""

from __future__ import annotations

import numpy as np

from quchip import Capacitive, ChargeDrive, Chip, DuffingTransmon, Gaussian, QuantumSequence, Resonator
from quchip.results.partitioned import PartitionedSimulationResult


def test_patch_plus_partition_reproduces_full_readout_bank():
    """Patched-and-partitioned readout expectation values match the full multiplexed-readout solve."""
    # Pair 0: driven qubit + its own readout resonator.
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    r0 = Resonator(freq=7.0, levels=3, label="r0")
    # Pair 1: same shape, independently driven and detuned.
    q1 = DuffingTransmon(freq=5.4, anharmonicity=-0.24, levels=3, label="q1")
    r1 = Resonator(freq=7.3, levels=3, label="r1")
    # Bus resonator bridges the two pairs; a spectator transmon hangs off its far end.
    bus = Resonator(freq=8.5, levels=3, label="bus")
    spec = DuffingTransmon(freq=6.2, anharmonicity=-0.25, levels=3, label="spec")
    chip = Chip(
        [q0, r0, q1, r1, bus, spec],
        couplings=[
            Capacitive(q0, r0, g=0.05, label="qr0"),
            Capacitive(q1, r1, g=0.05, label="qr1"),
            Capacitive(r0, bus, g=0.004, label="rb"),
            Capacitive(bus, spec, g=0.004, label="bs"),
        ],
        frame="rotating", rwa=True,
    )
    d0 = ChargeDrive(target=q0, label="d0")
    d1 = ChargeDrive(target=q1, label="d1")
    chip.wire(d0, d1)
    seq = QuantumSequence(chip)
    seq.schedule(d0, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(q0))
    seq.schedule(d1, envelope=Gaussian(duration=20.0, sigmas=3, amplitude=0.02), freq=chip.freq(q1))
    tlist = np.linspace(0.0, 20.0, 41)

    full = seq.simulate(tlist=tlist, e_ops=chip.e_ops(q0="Z", q1="Z"), partition=False)

    # hops=1 from {q0, q1}: active = {q0, r0, q1, r1}; bus is 2 hops, spec 3.
    patch = seq.active_patch(hops=1)
    assert set(patch.eliminated_labels) == {"bus", "spec"}
    assert patch.eliminated_labels[0] == "spec"  # farthest first

    # e_ops takes built operators (chip.e_ops(...)), not raw name strings —
    # decompose_eops (engine/stage3_observables.py) expects the former, and
    # the patch chip has its own local Hilbert space so the operators must
    # be built against patch.chip, not chip.
    reduced = patch.simulate(tlist=tlist, e_ops=patch.chip.e_ops(q0="Z", q1="Z"))
    # After eliminating bus+spec the patch splits into the two pairs.
    assert isinstance(reduced, PartitionedSimulationResult)
    # Dispersive SW error at the active/spectator boundary (rb: r0-bus) scales as
    # (g/Delta)^2 = (0.004/1.5)^2 = 7.1e-6; 50x gives comfortable headroom over
    # that leading-order estimate (observed deviation is ~7e-6, well inside it).
    g = chip.coupling("rb").g
    delta = abs(chip.freq("bus") - chip.freq("r0"))
    tol = 50 * (g / delta) ** 2
    for key in ("q0", "q1"):
        assert np.max(np.abs(np.asarray(full.expect(key)) - np.asarray(reduced.expect(key)))) < tol
