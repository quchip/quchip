"""Stage 2 compiles scheduled edge pumps into per-band DynamicTerms (spec §5)."""

from __future__ import annotations

import numpy as np

from quchip import (
    Chip,
    ControlEquipment,
    DuffingTransmon,
    ParametricDrive,
    QuantumSequence,
    Square,
    TunableCapacitive,
)


def _problem(freq=None, frame="lab", rwa=False):
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    tc = TunableCapacitive(q0, q1, g_0=0.0, label="tc")
    pump = ParametricDrive(tc, label="pump")
    chip = Chip([q0, q1], couplings=[tc], frame=frame, rwa=rwa)
    chip.connect(ControlEquipment([pump]))
    seq = QuantumSequence(chip)
    seq.pump(tc, envelope=Square(duration=100.0, amplitude=0.005), freq=freq)
    return seq.build_problem(tlist=np.linspace(0.0, 100.0, 11))


def test_edge_pump_produces_dynamic_terms():
    """Edge pump scheduling produces dynamic terms tagged as coupling origin."""
    problem = _problem()
    pump_terms = [t for t in problem.hamiltonian.dynamic_terms if t.tag == "edge_pump"]
    assert pump_terms, "scheduled pump produced no dynamic terms"
    assert all(t.origin == "coupling" for t in pump_terms)


def test_baseband_and_tone_forms_differ():
    """Baseband and tone pump forms produce distinct signal structures."""
    base = _problem(freq=None)
    tone = _problem(freq=0.2)
    n_base = [t for t in base.hamiltonian.dynamic_terms if t.tag == "edge_pump"]
    n_tone = [t for t in tone.hamiltonian.dynamic_terms if t.tag == "edge_pump"]
    assert n_base and n_tone
    # Tone signals embed an extra Carrier; cheapest structural check is the repr tree.
    assert repr(n_base[0].time_dependence) != repr(n_tone[0].time_dependence)


def test_rwa_selects_beam_splitter_structure():
    """RWA selects the beam-splitter operator structure with fewer pump bands than the full interaction."""
    full = _problem(frame="rotating", rwa=False)
    rwa = _problem(frame="rotating", rwa=True)

    def pump_terms(problem):
        return [t for t in problem.hamiltonian.dynamic_terms if t.tag == "edge_pump"]

    # Full form carries more bands (counter-rotating (±1,±1) present) than the RWA form.
    assert len(pump_terms(full)) > len(pump_terms(rwa))
