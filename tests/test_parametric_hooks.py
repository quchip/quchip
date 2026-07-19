"""Parametric capability hooks on couplings (spec §4.2, §4.5)."""

from __future__ import annotations

import numpy as np

from quchip import Capacitive, Chip, CrossKerr, DuffingTransmon, TunableCapacitive


def _chip(coupling_cls, **kw):
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    c = coupling_cls(q0, q1, label="c", **kw)
    return Chip([q0, q1], couplings=[c], rwa=True), c


def _arr(op):
    return np.asarray(op.full() if hasattr(op, "full") else op)


def test_capacitive_is_not_modulable():
    """A static Capacitive coupling declares no parametric structure, so parametric_operator returns None."""
    chip, c = _chip(Capacitive, g=0.005)
    assert c.parametric_operator(chip) is None


def test_tunable_capacitive_rwa_vs_full_structures_differ():
    """TunableCapacitive's parametric operator differs between the RWA beam-splitter and full dipole-dipole forms."""
    chip_rwa, c_rwa = _chip(TunableCapacitive, g_0=0.0)
    chip_full = Chip(
        [d.copy() for d in chip_rwa.devices],
        couplings=[TunableCapacitive("q0", "q1", g_0=0.0, label="c")],
        rwa=False,
    )
    op_rwa = c_rwa.parametric_operator(chip_rwa)
    op_full = chip_full.coupling("c").parametric_operator(chip_full)
    assert op_rwa is not None and op_full is not None
    assert not np.allclose(_arr(op_rwa), _arr(op_full))


def test_crosskerr_interaction_is_diagonal_and_hooks_coincide():
    """CrossKerr's n̂n̂ interaction Hamiltonian is diagonal, so its RWA and full parametric operators coincide exactly."""
    chip, c = _chip(CrossKerr, chi=-0.0005)
    h = _arr(c.interaction_hamiltonian())
    assert np.allclose(h, np.diag(np.diag(h)))  # n̂n̂ is diagonal
    # RWA and full parametric structures coincide (band weight 0).
    full = _arr(c.parametric_operator(chip))
    chip_full = Chip(
        [d.copy() for d in chip.devices],
        couplings=[CrossKerr("q0", "q1", chi=-0.0005, label="c")],
        rwa=False,
    )
    assert np.allclose(full, _arr(chip_full.coupling("c").parametric_operator(chip_full)))


def test_crosskerr_declares_itself():
    """CrossKerr marks itself effective, and its physics_notes disclose the dispersive uniform-pull approximation."""
    chip, c = _chip(CrossKerr, chi=-0.0005)
    assert c.is_effective is True
    notes = " ".join(c.physics_notes()).lower()
    assert "dispersive" in notes and "uniform" in notes


def test_crosskerr_serialization_round_trip():
    """CrossKerr's to_dict/from_dict round trip preserves chi and label exactly."""
    chip, c = _chip(CrossKerr, chi=-0.0005)
    d = c.to_dict()
    q0, q1 = chip.devices[0], chip.devices[1]
    c2 = CrossKerr.from_dict(d, q0, q1)
    assert float(c2.chi) == float(c.chi)
    assert c2.label == "c"


def test_tunable_capacitive_has_no_modulation_surface():
    """TunableCapacitive has no modulation parameter or dynamic terms without a pump in its eliminated-coupler model."""
    import inspect

    from quchip import TunableCapacitive

    assert "modulation" not in inspect.signature(TunableCapacitive.__init__).parameters
    chip, c = _chip(TunableCapacitive, g_0=0.02)
    assert c.dynamic_interaction_terms(chip) == []
    assert c.is_effective is True
    notes = " ".join(c.physics_notes()).lower()
    assert "eliminated" in notes


def test_tunable_capacitive_serialization_round_trip():
    """TunableCapacitive's to_dict/from_dict round trip preserves g_0 exactly."""
    chip, c = _chip(TunableCapacitive, g_0=0.02)
    d = c.to_dict()
    c2 = TunableCapacitive.from_dict(d, chip.devices[0], chip.devices[1])
    assert float(c2.g_0) == 0.02
