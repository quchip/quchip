"""The public excitation-number record names what decides the sector routes."""

import warnings

import jax
import numpy as np
import pytest

from quchip import (
    Capacitive, ChargeDrive, Chip, DuffingTransmon, EffectiveTerms, Exact, ExcitationCutoff, RWA,
)
from quchip.engine.excitation import ExcitationTerm

pytestmark = pytest.mark.unit


def chain(n, *, entry=None, approximation=None):
    """Transmons in a chain, optionally with a retained correction on q0 that has a 0-1 entry."""
    qs = [DuffingTransmon(freq=5.0 + 0.1 * i, anharmonicity=-0.25, levels=3, label=f"q{i}") for i in range(n)]
    couplings = [Capacitive(a, b, g=0.01, label=f"{a.label}_{b.label}") for a, b in zip(qs, qs[1:])]
    effective = []
    if entry is not None:
        h = np.diag([0.0, 0.0, 0.0005])
        h[0, 1] = h[1, 0] = entry
        effective = [EffectiveTerms(("q0",), (3,), h, label="q0_correction")]
    return Chip(qs, couplings, effective_terms=effective, approximation=approximation or RWA())


def sector_line(chip):
    return next(line for line in chip.describe().splitlines() if line.startswith("Sectors"))


def test_conserving_chain_reports_its_sector_blocks():
    """Seven three-level transmons under RWA diagonalize at most 28 states for labels with N <= 2."""
    chip = chain(7)
    structure = chip.excitation_structure()

    assert structure.conserves and structure.changes == {0} and structure.breaking == ()
    assert structure.sector_sizes(2) == (1, 7, 28)
    assert structure.largest_block(2) == 28
    assert {term.changes for term in structure.terms} == {frozenset({0})}
    assert "N conserved" in sector_line(chip) and "28 states" in sector_line(chip)

    cutoff = chip.with_space(ExcitationCutoff(2)).excitation_structure()
    assert cutoff.classes(cutoff.cutoff) == ((0,), (1,), (2,))
    assert cutoff.largest_block(2) == 28


def test_round_off_entry_is_named_and_warns_without_changing_the_verdict():
    """A 1e-16 GHz entry between levels 0 and 1 breaks conservation, and dressed queries say why."""
    structure = chain(7, entry=1e-16).excitation_structure()

    assert not structure.conserves
    (term,) = structure.breaking
    assert (term.label, term.kind, term.changes) == ("q0_correction", "retained", frozenset({-1, 0, 1}))
    assert term.largest_entry == pytest.approx(1e-16)
    assert structure.largest_block(2) == 3**7

    chip = chain(3, entry=1e-16)
    with pytest.warns(UserWarning, match="q0_correction"):
        chip.freq("q1")
    assert not chip.excitation_structure().conserves
    assert "q0_correction (0, ±1)" in sector_line(chip)


def test_exact_record_names_each_counter_rotating_coupling_without_a_warning():
    """Under Exact(), each capacitive coupling changes N by 0 and ±2, through entries of physical size."""
    chip = chain(3, approximation=Exact())
    structure = chip.excitation_structure()

    assert [(term.label, term.kind, term.changes) for term in structure.breaking] == [
        ("q0_q1", "coupling", frozenset({-2, 0, 2})), ("q1_q2", "coupling", frozenset({-2, 0, 2})),
    ]
    # a^dagger b^dagger connects |11> and |22> with sqrt(2) * sqrt(2) * g.
    assert [term.largest_entry for term in structure.breaking] == pytest.approx([0.02, 0.02])
    assert structure.largest_block(2) == 27
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chip.freq("q1")
    assert not [warning for warning in caught if "excitation number" in str(warning.message)]


def test_record_reports_drives_and_collapse_channels():
    """A charge drive under RWA keeps changes of ±1, and relaxation lowers N by one."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, T1=20_000.0, label="q")
    chip = Chip([q], approximation=RWA())
    chip.wire(ChargeDrive(q, label="xy"))
    structure = chip.excitation_structure()

    assert structure.conserves
    assert structure.drives == (ExcitationTerm("xy", "drive", frozenset({-1, 1})),)
    assert {term.changes for term in structure.channels} == {frozenset({1})}


def test_record_is_a_snapshot_and_matches_its_traced_structure():
    """A record keeps its values after the chip changes, and tracing a coupling keeps the structure."""
    chip = chain(3, approximation=Exact())
    before = chip.excitation_structure()
    chip.couplings[0].g = 0.02
    after = chip.excitation_structure()
    assert before.terms[0].largest_entry == pytest.approx(0.02)
    assert after.terms[0].largest_entry == pytest.approx(0.04)

    seen = []

    def read(g):
        seen.append(chip.with_params({"q0_q1.g": g}).excitation_structure())
        return g

    jax.jit(read)(0.01)
    (traced,) = seen
    assert traced.conserves == after.conserves
    assert [(term.label, term.kind, term.changes) for term in traced.terms] == [
        (term.label, term.kind, term.changes) for term in after.terms
    ]
    assert traced.terms[0].largest_entry is None
