import numpy as np
import pytest

from quchip import Bath, Chip, DuffingTransmon, Resonator
from quchip.backend import _backend_context


def test_bath_autolabels_and_defaults_to_all_devices():
    """A bath with no explicit label auto-labels sequentially; with no explicit targets it covers every chip device."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([q, r], baths=[Bath("thermal", temperature=15.0, rate=1e-3)])
    bath = chip.baths[0]
    assert bath.label == "bath_0"
    # No explicit targets -> every device in the chip.
    assert set(bath.resolve_targets(chip)) == {"q", "r"}


def test_bath_targets_accept_label_or_object():
    """Bath targets given as a mix of device objects and labels resolve to the same ordered device labels."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    bath = Bath("thermal", targets=[q, "r"], temperature=15.0, rate=1e-3)
    chip = Chip([q, r], baths=[bath])
    assert bath.resolve_targets(chip) == ["q", "r"]


def _ops(bath, chip):
    with _backend_context(chip.backend):
        return bath.collapse_operators(chip)


def test_thermal_independent_emits_relaxation_and_absorption_per_device():
    """An independent finite-T bath emits a full-Hilbert-space relaxation/absorption operator pair per target."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([q, r], baths=[Bath("thermal", temperature=200.0, rate=1e-3)])
    ops = _ops(chip.baths[0], chip)
    # 2 devices x (relaxation + absorption) at finite T.
    assert len(ops) == 4
    full_dim = 3 * 4
    assert ops[0].shape == (full_dim, full_dim)


def test_collective_decay_is_a_single_summed_operator():
    """A collective decay bath produces a single summed jump operator across its targets rather than one per device."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q1")
    chip = Chip([q0, q1], baths=[Bath("collective_decay", rate=0.01)])
    ops = _ops(chip.baths[0], chip)
    assert len(ops) == 1  # ONE summed jump operator, not two independent ones.


def test_correlated_thermal_bath_raises_not_implemented():
    """A correlated thermal bath raises NotImplementedError because collective thermal noise is not implemented."""
    with pytest.raises(NotImplementedError, match="[Cc]ollective thermal"):
        Bath("thermal", temperature=15.0, rate=1e-3, correlated=True)


def test_chip_with_baths_survives_serialization_round_trip():
    """A chip's baths — recipe, parameters, label, and resolved targets — survive a to_dict/from_dict round trip."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip(
        [q, r],
        baths=[
            Bath("thermal", temperature=15.0, rate=1e-3),  # default (all) targets
            Bath("collective_decay", targets=[q, r], rate=0.01, label="collective"),
        ],
    )
    restored = Chip.from_dict(chip.to_dict())
    assert len(restored.baths) == 2
    thermal, collective = restored.baths
    assert thermal.recipe == "thermal" and thermal.temperature == 15.0 and thermal.rate == 1e-3
    assert thermal.resolve_targets(restored) == ["q", "r"]
    assert collective.recipe == "collective_decay" and collective.label == "collective"
    assert collective.resolve_targets(restored) == ["q", "r"]


def test_bath_rejects_negative_concrete_temperature_and_rate():
    """A concrete negative temperature or rate raises at construction."""
    with pytest.raises(ValueError, match="temperature"):
        Bath("thermal", temperature=-5.0, rate=1e-3)
    with pytest.raises(ValueError, match="rate"):
        Bath("thermal", temperature=15.0, rate=-1e-3)


def test_bath_allows_a_traced_negative_looking_temperature_or_rate():
    """A traced temperature/rate is never concretized for the sign check, so construction does not raise."""
    import jax

    def build(temperature):
        bath = Bath("thermal", temperature=temperature, rate=1e-3)
        return bath.temperature

    jax.jit(build)(-5.0)  # would raise ValueError under jit if the check forced concretization


def test_bath_zero_temperature_thermal_occupation_is_zero():
    """A concrete T=0 bath's thermal occupation is exactly 0, with no division-by-zero crash."""
    bath = Bath("thermal", temperature=0.0, rate=1e-3)
    n_bar = bath._bose(5.0, np)
    assert float(n_bar) == 0.0


def test_bath_zero_temperature_collapse_operators_relaxation_only():
    """A concrete T=0 thermal bath's relaxation operator is nonzero and its absorption operator is exactly zero."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], baths=[Bath("thermal", temperature=0.0, rate=1e-3)])
    relax, absorb = _ops(chip.baths[0], chip)
    relax_array = chip.backend.to_array(relax)
    absorb_array = chip.backend.to_array(absorb)
    assert np.linalg.norm(relax_array) > 0.0
    assert np.allclose(absorb_array, 0.0)


def test_baths_flow_into_collected_c_ops():
    """Collected chip collapse operators include each bath's ops and are empty for a chip with no baths."""
    from quchip.engine.stage4_problem import _collect_c_ops

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    no_bath = Chip([q])
    with_bath = Chip([q.copy()], baths=[Bath("thermal", temperature=200.0, rate=1e-3)])
    assert len(_collect_c_ops(no_bath)) == 0
    assert len(_collect_c_ops(with_bath)) == 2  # relaxation + absorption
