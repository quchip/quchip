"""Tests for post-construction noise and custom dissipation hooks.

Noise added, changed, or removed after chip construction must be fully reflected in the next
``simulate`` call, with no rebuild or cache poking required.
"""

from __future__ import annotations

import numpy as np
import pytest

from quchip import (
    Chip,
    CollapseChannel,
    DuffingTransmon,
    Resonator,
    Scalar,
    parameter,
    simulate,
)

TLIST = np.linspace(0.0, 300.0, 61)


def _lossless_pair() -> tuple[Chip, DuffingTransmon, Resonator]:
    """Noise-free q + r, deliberately uncoupled so single-device decay laws are exact.

    Three levels each with dynamics confined to {|0>, |1>} — the decay laws stay
    exact and the top level stays empty, so the truncation check stays quiet.
    """
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    return Chip([q, r]), q, r


# ---------------------------------------------------------------------------
# 1. Parameter mutation
# ---------------------------------------------------------------------------


def test_posthoc_t1_and_internal_quality_factor_reflected_in_next_simulate():
    """Setting T1/internal_quality_factor after construction is reflected in the next simulate call."""
    chip, q, r = _lossless_pair()
    excited = chip.bare_state({q: 1, r: 1})
    e_ops = {q: q.number_operator(), r: r.number_operator()}

    before = simulate(chip, [], TLIST, initial_state=excited, e_ops=e_ops)
    assert np.real(before.expect("q"))[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.real(before.expect("r"))[-1] == pytest.approx(1.0, abs=1e-9)

    assert before.solver == "sesolve"
    q.T1 = 51_600.0
    r.internal_quality_factor = 5_000.0

    after = simulate(chip, [], TLIST, initial_state=excited, e_ops=e_ops)
    assert after.solver == "mesolve"
    t_final = TLIST[-1]
    assert np.real(after.expect("q"))[-1] == pytest.approx(np.exp(-t_final / 51_600.0), rel=1e-3)
    kappa = 2 * np.pi * 7.0 / 5_000.0
    assert np.real(after.expect("r"))[-1] == pytest.approx(np.exp(-kappa * t_final), rel=1e-3)


def test_noise_removed_posthoc_restores_lossless_evolution():
    """Clearing T1/internal_quality_factor after the fact restores lossless evolution."""
    chip, q, r = _lossless_pair()
    q.T1 = 51_600.0
    r.internal_quality_factor = 5_000.0
    excited = chip.bare_state({q: 1, r: 1})
    e_ops = {q: q.number_operator(), r: r.number_operator()}

    noisy = simulate(chip, [], TLIST, initial_state=excited, e_ops=e_ops)
    assert np.real(noisy.expect("r"))[-1] < 0.5  # sanity: dissipation was active

    q.T1 = None
    r.internal_quality_factor = None

    clean = simulate(chip, [], TLIST, initial_state=excited, e_ops=e_ops)
    assert np.real(clean.expect("q"))[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.real(clean.expect("r"))[-1] == pytest.approx(1.0, abs=1e-9)


def test_posthoc_mutation_validated_like_constructor():
    """A post-construction noise-parameter write is validated identically to the constructor."""
    q = DuffingTransmon(
        freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=50_000.0, T2=80_000.0
    )

    with pytest.raises(ValueError, match="T2"):
        q.T2 = 150_000.0  # > 2*T1 — the same value the constructor refuses
    assert q.T2 == 80_000.0  # the rejected write must not stick

    with pytest.raises(ValueError, match="T1"):
        q.T1 = -5.0
    assert q.T1 == 50_000.0

    with pytest.raises(ValueError, match="thermal_occupation"):
        q.thermal_occupation = -0.1


def test_posthoc_declarative_sign_constraints_enforced():
    """A post-construction write to a declared positive parameter enforces its sign constraint."""
    r = Resonator(freq=7.0, levels=3, label="r", internal_quality_factor=5_000.0)

    with pytest.raises(ValueError, match="internal_quality_factor"):
        r.internal_quality_factor = -5_000.0
    assert r.internal_quality_factor == 5_000.0  # the rejected write must not stick

    r.internal_quality_factor = None  # None means "remove the channel" and must stay allowed
    assert r.internal_quality_factor is None


# ---------------------------------------------------------------------------
# 2. Baths on an existing chip
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. Custom device dissipation
# ---------------------------------------------------------------------------


class LeakyTransmon(DuffingTransmon):
    """DuffingTransmon plus one declared leakage-loss channel (test fixture)."""

    _type_prefix = "leaky"
    leakage_rate: Scalar = parameter(default=None, positive=True, noise=True)

    def dissipation(self, op, p):
        channels = super().dissipation(op, p)
        if self.leakage_rate is None:
            return channels
        return channels + (CollapseChannel(op.a, p.leakage_rate, "leakage"),)


def test_custom_channel_composes_with_inherited_ones():
    """A custom dissipation hook composes with inherited device channels."""
    quiet = LeakyTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="quiet")
    assert quiet.collapse_operators() == []  # parameter unset -> channel contributes nothing

    loud = LeakyTransmon(
        freq=5.0, anharmonicity=-0.25, levels=3, label="loud", T1=51_600.0, leakage_rate=0.01
    )
    assert len(loud.collapse_operators()) == 2  # inherited T1 channel + declared leakage


def test_custom_channel_rate_set_posthoc_reflected_in_next_simulate():
    """Setting a custom channel's rate after construction is reflected in the next simulate call."""
    q = LeakyTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q])
    excited = chip.bare_state({q: 1})
    e_ops = {q: q.number_operator()}

    before = simulate(chip, [], TLIST, initial_state=excited, e_ops=e_ops)
    assert np.real(before.expect("q"))[-1] == pytest.approx(1.0, abs=1e-9)

    q.leakage_rate = 0.01

    after = simulate(chip, [], TLIST, initial_state=excited, e_ops=e_ops)
    assert np.real(after.expect("q"))[-1] == pytest.approx(np.exp(-0.01 * TLIST[-1]), rel=1e-3)


def test_declared_channel_is_reflected_in_simulation():
    """A declared dissipation hook is reflected in simulation."""
    class DampedMode(Resonator):
        extra_loss_rate: Scalar = parameter(default=None, positive=True, noise=True)

        def dissipation(self, op, p):
            channels = super().dissipation(op, p)
            if self.extra_loss_rate is None:
                return channels
            return channels + (
                CollapseChannel(op.a, p.extra_loss_rate, "extra_loss"),
            )

    m = DampedMode(freq=5.0, levels=3, label="m", extra_loss_rate=0.01)
    chip = Chip([m])
    excited = chip.bare_state({m: 1})
    e_ops = {m: m.number_operator()}

    after = simulate(chip, [], TLIST, initial_state=excited, e_ops=e_ops)
    assert np.real(after.expect("m"))[-1] == pytest.approx(np.exp(-0.01 * TLIST[-1]), rel=1e-3)


def test_noise_channel_metadata_is_not_public():
    """Dissipation authors use CollapseChannel directly."""
    import quchip

    assert not hasattr(quchip, "NoiseChannel")
    assert "NoiseChannel" not in quchip.__all__


def test_noise_parameter_names_reflect_declared_channels():
    """noise_parameter_names() follows fields declared with noise=True."""
    assert DuffingTransmon.noise_parameter_names() == ("T1", "T2", "thermal_occupation")
    assert Resonator.noise_parameter_names() == ("T1", "T2", "thermal_occupation", "internal_quality_factor")
    assert LeakyTransmon.noise_parameter_names() == ("T1", "T2", "thermal_occupation", "leakage_rate")


# ---------------------------------------------------------------------------
# 4. chip.set_noise — one-call, replace-all noise configuration
# ---------------------------------------------------------------------------


def test_set_noise_is_the_complete_noise_description():
    """chip.set_noise() replaces the entire noise configuration; unmentioned params are cleared."""
    chip, q, r = _lossless_pair()
    q.T1 = 10_000.0                                                    # initial noise to replace

    chip.set_noise({"r": dict(internal_quality_factor=5_000.0)})
    assert q.T1 is None                       # unmentioned -> cleared (replace-all)
    assert r.internal_quality_factor == 5_000.0

    chip.set_noise()                          # no args == clear everything
    assert r.internal_quality_factor is None


def test_set_noise_matches_equivalent_attribute_writes():
    """chip.set_noise() produces identical dynamics to the equivalent per-attribute writes."""
    chip_a, qa, ra = _lossless_pair()
    chip_b, qb, rb = _lossless_pair()
    qa.T1 = 51_600.0
    ra.internal_quality_factor = 5_000.0
    chip_b.set_noise({qb: dict(T1=51_600.0), rb: dict(internal_quality_factor=5_000.0)})

    assert len(chip_a.collapse_contributions()) == len(chip_b.collapse_contributions()) == 2
    e_ops_a = {qa: qa.number_operator(), ra: ra.number_operator()}
    e_ops_b = {qb: qb.number_operator(), rb: rb.number_operator()}
    a = simulate(chip_a, [], TLIST, initial_state=chip_a.bare_state({qa: 1, ra: 1}), e_ops=e_ops_a)
    b = simulate(chip_b, [], TLIST, initial_state=chip_b.bare_state({qb: 1, rb: 1}), e_ops=e_ops_b)
    np.testing.assert_allclose(np.real(a.expect("q")), np.real(b.expect("q")), atol=1e-12)
    np.testing.assert_allclose(np.real(a.expect("r")), np.real(b.expect("r")), atol=1e-12)


def test_set_noise_can_reduce_relaxation_and_coherence_times_together():
    """Replace-all noise configuration validates the final joint decay model."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, T1=100.0, T2=150.0, label="q")
    chip = Chip([q])
    chip.set_noise({q: {"T1": 20.0, "T2": 30.0}})
    assert q.T1 == 20.0
    assert q.T2 == 30.0


def test_set_noise_is_atomic_on_invalid_input():
    """chip.set_noise() applies nothing when any target in the call is invalid."""
    chip, q, r = _lossless_pair()
    q.T1 = 50_000.0

    with pytest.raises(ValueError, match="T2"):
        chip.set_noise({q: dict(T1=50_000.0, T2=150_000.0)})          # target violates T2 <= 2*T1
    assert q.T1 == 50_000.0 and q.T2 is None                          # nothing applied

    with pytest.raises(ValueError, match="ghost"):
        chip.set_noise({q: dict(T1=1_000.0), "ghost": dict(T1=1_000.0)})
    assert q.T1 == 50_000.0                                           # bad target blocked the whole call


def test_set_noise_rejects_non_noise_and_unknown_keys():
    """chip.set_noise() rejects Hamiltonian params, unknown device keys, and duplicate targets."""
    chip, q, r = _lossless_pair()
    with pytest.raises(ValueError, match="freq"):
        chip.set_noise({q: dict(freq=4.9)})                           # Hamiltonian params unreachable
    with pytest.raises(ValueError, match="ghost"):
        chip.set_noise({"ghost": dict(T1=1_000.0)})
    with pytest.raises(ValueError, match="more than once"):
        chip.set_noise({q: dict(T1=1_000.0), "q": dict(T2=500.0)})


def test_set_noise_prints_changes_only_when_any(capsys):
    """chip.set_noise() prints a change summary only when the call actually changes something."""
    chip, q, r = _lossless_pair()
    chip.set_noise({q: dict(T1=51_600.0)})
    out = capsys.readouterr().out
    assert "q: T1 None" in out and "51600" in out
    chip.set_noise({q: dict(T1=51_600.0)})                            # identical call -> silent no-op
    assert capsys.readouterr().out == ""


def test_set_noise_covers_declared_custom_channel_params():
    """chip.set_noise() covers a custom noise=True rate parameter."""
    q = LeakyTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="lq")
    chip = Chip([q])
    chip.set_noise({q: dict(leakage_rate=0.01)})
    assert len(chip.collapse_contributions()) == 1
