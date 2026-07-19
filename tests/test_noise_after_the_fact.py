"""Tests for post-construction noise: parameter mutation, chip.add_bath, and custom NoiseChannels.

Noise added, changed, or removed after chip construction must be fully reflected in the next
``simulate`` call, with no rebuild or cache poking required.
"""

from __future__ import annotations

import numpy as np
import pytest

from quchip import Bath, Capacitive, Chip, DuffingTransmon, Resonator, Scalar, eliminate, parameter, simulate
from quchip.backend import get_default_backend
from quchip.devices.base import NoiseChannel
from quchip.utils.constants import k_B

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


def test_posthoc_t1_and_quality_factor_reflected_in_next_simulate():
    """Setting T1/quality_factor after construction is reflected in the next simulate call."""
    chip, q, r = _lossless_pair()
    excited = chip.bare_state({q: 1, r: 1})
    e_ops = {q: q.number_operator(), r: r.number_operator()}

    before = simulate(chip, [], TLIST, initial_state=excited, e_ops=e_ops)
    assert np.real(before.expect("q"))[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.real(before.expect("r"))[-1] == pytest.approx(1.0, abs=1e-9)

    q.T1 = 51_600.0
    r.quality_factor = 5_000.0

    after = simulate(chip, [], TLIST, initial_state=excited, e_ops=e_ops)
    t_final = TLIST[-1]
    assert np.real(after.expect("q"))[-1] == pytest.approx(np.exp(-t_final / 51_600.0), rel=1e-3)
    kappa = 2 * np.pi * 7.0 / 5_000.0
    assert np.real(after.expect("r"))[-1] == pytest.approx(np.exp(-kappa * t_final), rel=1e-3)


def test_posthoc_noise_flips_default_solver():
    """Adding T1 after construction alone flips the default solver from sesolve to mesolve."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q])
    tlist = np.linspace(0.0, 10.0, 5)
    excited = chip.bare_state({q: 1})

    before = simulate(chip, [], tlist, initial_state=excited)
    q.T1 = 51_600.0
    after = simulate(chip, [], tlist, initial_state=excited)

    assert before.solver == "sesolve"  # noise-free -> state-vector solve
    assert after.solver == "mesolve"  # the mutation alone selects the density-matrix solve


def test_noise_removed_posthoc_restores_lossless_evolution():
    """Clearing T1/quality_factor after the fact restores lossless evolution."""
    chip, q, r = _lossless_pair()
    q.T1 = 51_600.0
    r.quality_factor = 5_000.0
    excited = chip.bare_state({q: 1, r: 1})
    e_ops = {q: q.number_operator(), r: r.number_operator()}

    noisy = simulate(chip, [], TLIST, initial_state=excited, e_ops=e_ops)
    assert np.real(noisy.expect("r"))[-1] < 0.5  # sanity: dissipation was active

    q.T1 = None
    r.quality_factor = None

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

    with pytest.raises(ValueError, match="thermal_population"):
        q.thermal_population = -0.1


def test_posthoc_declarative_sign_constraints_enforced():
    """A post-construction write to a declared positive parameter enforces its sign constraint."""
    r = Resonator(freq=7.0, levels=3, label="r", quality_factor=5_000.0)

    with pytest.raises(ValueError, match="quality_factor"):
        r.quality_factor = -5_000.0
    assert r.quality_factor == 5_000.0  # the rejected write must not stick

    r.quality_factor = None  # None means "remove the channel" and must stay allowed
    assert r.quality_factor is None


# ---------------------------------------------------------------------------
# 2. Baths on an existing chip
# ---------------------------------------------------------------------------


def test_add_bath_reflected_in_next_simulate():
    """chip.add_bath() on an existing chip is reflected in the next simulate call."""
    mode = Resonator(freq=5.0, levels=5, label="m")
    chip = Chip([mode])
    tlist = np.linspace(0.0, 400.0, 81)
    e_ops = {mode: mode.number_operator()}

    before = simulate(chip, [], tlist, initial_state=chip.bare_state(), e_ops=e_ops)
    assert np.real(before.expect("m"))[-1] == pytest.approx(0.0, abs=1e-9)

    bath = chip.add_bath(Bath("thermal", temperature=100.0, rate=0.05))
    assert bath in chip.baths  # fluent return + visible on the chip

    after = simulate(chip, [], tlist, initial_state=chip.bare_state(), e_ops=e_ops)
    n_bar = 1.0 / np.expm1(5.0 / (k_B * 100.0))
    assert np.real(after.expect("m"))[-1] == pytest.approx(n_bar, rel=0.05)


def test_add_bath_matches_constructor_bath():
    """A bath added via add_bath() produces identical dynamics to the same bath in the constructor."""
    def make_chip(with_bath_in_ctor: bool) -> Chip:
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        baths = [Bath("thermal", temperature=25.0, rate=1e-3)] if with_bath_in_ctor else None
        return Chip([q], baths=baths)

    ctor = make_chip(True)
    posthoc = make_chip(False)
    posthoc.add_bath(Bath("thermal", temperature=25.0, rate=1e-3))

    assert len(ctor.collapse_contributions()) == len(posthoc.collapse_contributions()) == 2

    tlist = np.linspace(0.0, 200.0, 41)
    kw: dict = {"e_ops": {"q": ctor["q"].number_operator()}}
    a = simulate(ctor, [], tlist, initial_state=ctor.bare_state({"q": 1}), **kw)
    b = simulate(posthoc, [], tlist, initial_state=posthoc.bare_state({"q": 1}), **kw)
    np.testing.assert_allclose(np.real(a.expect("q")), np.real(b.expect("q")), atol=1e-10)


def test_add_bath_rejects_non_bath_immediately():
    """add_bath() rejects a non-Bath argument immediately, not as an AttributeError at solve time."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q])

    with pytest.raises(TypeError, match="Bath"):
        chip.add_bath(42)  # type: ignore[arg-type]
    assert chip.baths == ()  # the rejected attach must not stick


def test_bath_with_unknown_target_rejected_at_attach_time():
    """add_bath() with an unknown target label fails at attach time, not at solve time."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q])

    with pytest.raises(ValueError, match="ghost"):
        chip.add_bath(Bath("thermal", targets=["ghost"], temperature=30.0, rate=1e-3))
    assert chip.baths == ()

    with pytest.raises(ValueError, match="ghost"):
        Chip([q.copy()], baths=[Bath("thermal", targets=["ghost"], temperature=30.0, rate=1e-3)])


def test_clone_and_eliminate_do_not_share_bath_objects():
    """clone() and eliminate() copy baths, so mutating the copy never leaks back to the original."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    chip = Chip(
        [q, r],
        couplings=[Capacitive(q, r, g=0.05)],
        baths=[Bath("thermal", temperature=30.0, rate=1e-4)],
    )

    cloned = chip.clone()
    assert cloned.baths[0] is not chip.baths[0]
    cloned.baths[0].temperature = 500.0
    assert chip.baths[0].temperature == 30.0  # no leak back to the original

    reduced = eliminate(chip, r).chip
    assert reduced.baths[0] is not chip.baths[0]


def test_add_bath_round_trips_serialization():
    """A bath added via add_bath() round-trips through to_dict/from_dict."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    chip = Chip([q])
    chip.add_bath(Bath("thermal", targets=[q], temperature=25.0, rate=2e-3, label="fridge"))

    restored = Chip.from_dict(chip.to_dict())
    assert len(restored.baths) == 1
    fridge = restored.baths[0]
    assert fridge.recipe == "thermal" and fridge.label == "fridge"
    assert fridge.temperature == 25.0 and fridge.rate == 2e-3
    assert fridge.resolve_targets(restored) == ["q"]


# ---------------------------------------------------------------------------
# 3. Custom NoiseChannel on a device subclass
# ---------------------------------------------------------------------------


def _leakage_channel(device) -> list:
    """Extra loss channel ``sqrt(rate)·a`` — the documented one-declaration recipe."""
    if device.leakage_rate is None:
        return []
    xp = get_default_backend().array_module
    return [xp.sqrt(device.leakage_rate) * device.lowering_operator()]


class LeakyTransmon(DuffingTransmon):
    """DuffingTransmon plus one declared leakage-loss channel (test fixture)."""

    _type_prefix = "leaky"
    leakage_rate: Scalar = parameter(default=None, positive=True)
    _noise_channels = DuffingTransmon._noise_channels + (
        NoiseChannel("leakage", ("leakage_rate",), _leakage_channel),
    )


def test_custom_channel_composes_with_inherited_ones():
    """A declared custom NoiseChannel composes with the device's inherited noise channels."""
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


def _extra_loss_channel(device) -> list:
    if device.extra_loss_rate is None:
        return []
    xp = get_default_backend().array_module
    return [xp.sqrt(device.extra_loss_rate) * device.lowering_operator()]


def test_channel_attached_to_subclass_posthoc_reflected_in_next_simulate():
    """A NoiseChannel attached to a device class after instances already exist still applies."""
    class DampedMode(Resonator):
        extra_loss_rate: Scalar = parameter(default=None, positive=True)

    m = DampedMode(freq=5.0, levels=3, label="m", extra_loss_rate=0.01)
    chip = Chip([m])
    excited = chip.bare_state({m: 1})
    e_ops = {m: m.number_operator()}

    before = simulate(chip, [], TLIST, initial_state=excited, e_ops=e_ops)
    assert np.real(before.expect("m"))[-1] == pytest.approx(1.0, abs=1e-9)

    DampedMode._noise_channels = DampedMode._noise_channels + (
        NoiseChannel("extra_loss", ("extra_loss_rate",), _extra_loss_channel),
    )

    after = simulate(chip, [], TLIST, initial_state=excited, e_ops=e_ops)
    assert np.real(after.expect("m"))[-1] == pytest.approx(np.exp(-0.01 * TLIST[-1]), rel=1e-3)


def test_noise_channel_is_a_top_level_export():
    """NoiseChannel is exported from the top-level quchip package."""
    import quchip
    from quchip.devices.base import NoiseChannel as base_noise_channel

    assert quchip.NoiseChannel is base_noise_channel
    assert "NoiseChannel" in quchip.__all__


def test_noise_parameter_names_reflect_declared_channels():
    """noise_parameter_names() lists exactly the parameters of a class's declared noise channels."""
    assert DuffingTransmon.noise_parameter_names() == ("T1", "thermal_population", "T2")
    assert Resonator.noise_parameter_names() == ("T1", "thermal_population", "T2", "quality_factor")
    assert LeakyTransmon.noise_parameter_names() == ("T1", "thermal_population", "T2", "leakage_rate")


def test_instance_level_channel_attachment_raises_instead_of_silent_noop():
    """Assigning `_noise_channels` on an instance raises, since only the class tuple is composed."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")

    with pytest.raises(TypeError, match="class"):
        q._noise_channels = q._noise_channels + (
            NoiseChannel("extra", ("T1",), lambda device: []),
        )


# ---------------------------------------------------------------------------
# 4. chip.set_noise — one-call, replace-all noise configuration
# ---------------------------------------------------------------------------


def test_set_noise_is_the_complete_noise_description():
    """chip.set_noise() replaces the entire noise configuration; unmentioned params are cleared."""
    chip, q, r = _lossless_pair()
    q.T1 = 10_000.0                                                    # pre-existing noise...
    chip.add_bath(Bath("thermal", temperature=100.0, rate=1e-3))       # ...and a bath

    chip.set_noise(
        {"r": dict(quality_factor=5_000.0)},
        baths=[Bath("collective_decay", targets=[q, r], rate=1e-4, label="col")],
    )
    assert q.T1 is None                       # unmentioned -> cleared (replace-all)
    assert r.quality_factor == 5_000.0
    assert [b.label for b in chip.baths] == ["col"]

    chip.set_noise()                          # no args == clear everything
    assert r.quality_factor is None
    assert chip.baths == ()


def test_set_noise_matches_equivalent_attribute_writes():
    """chip.set_noise() produces identical dynamics to the equivalent per-attribute writes."""
    chip_a, qa, ra = _lossless_pair()
    chip_b, qb, rb = _lossless_pair()
    qa.T1 = 51_600.0
    ra.quality_factor = 5_000.0
    chip_b.set_noise({qb: dict(T1=51_600.0), rb: dict(quality_factor=5_000.0)})

    assert len(chip_a.collapse_contributions()) == len(chip_b.collapse_contributions()) == 2
    e_ops_a = {qa: qa.number_operator(), ra: ra.number_operator()}
    e_ops_b = {qb: qb.number_operator(), rb: rb.number_operator()}
    a = simulate(chip_a, [], TLIST, initial_state=chip_a.bare_state({qa: 1, ra: 1}), e_ops=e_ops_a)
    b = simulate(chip_b, [], TLIST, initial_state=chip_b.bare_state({qb: 1, rb: 1}), e_ops=e_ops_b)
    np.testing.assert_allclose(np.real(a.expect("q")), np.real(b.expect("q")), atol=1e-12)
    np.testing.assert_allclose(np.real(a.expect("r")), np.real(b.expect("r")), atol=1e-12)


def test_set_noise_is_atomic_on_invalid_input():
    """chip.set_noise() applies nothing when any target or bath in the call is invalid."""
    chip, q, r = _lossless_pair()
    q.T1 = 50_000.0

    with pytest.raises(ValueError, match="T2"):
        chip.set_noise(
            {q: dict(T1=50_000.0, T2=150_000.0)},                     # target violates T2 <= 2*T1
            baths=[Bath("thermal", temperature=100.0, rate=1e-3)],
        )
    assert q.T1 == 50_000.0 and q.T2 is None and chip.baths == ()     # nothing applied

    with pytest.raises(ValueError, match="ghost"):
        chip.set_noise(
            {q: dict(T1=1_000.0)},
            baths=[Bath("thermal", targets=["ghost"], temperature=100.0, rate=1e-3)],
        )
    assert q.T1 == 50_000.0                                           # bad bath blocked the whole call


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
    """chip.set_noise() covers a custom NoiseChannel's declared rate parameter."""
    q = LeakyTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="lq")
    chip = Chip([q])
    chip.set_noise({q: dict(leakage_rate=0.01)})
    assert len(chip.collapse_contributions()) == 1
