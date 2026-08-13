import numpy as np
import pytest

from quchip import Bath, Chip, CollapseChannel, DuffingTransmon, Resonator
from quchip.backend import _backend_context
from quchip.declarative.expr import materialize_expr


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


def _terms(bath, chip):
    with _backend_context(chip.backend):
        return bath.collapse_channels(chip)


def test_thermal_independent_emits_relaxation_and_absorption_per_device():
    """An independent finite-T bath emits a full-Hilbert-space relaxation/absorption operator pair per target."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([q, r], baths=[Bath("thermal", temperature=200.0, rate=1e-3)])
    terms = _terms(chip.baths[0], chip)
    # 2 devices x (relaxation + absorption) at finite T.
    assert len(terms) == 4
    assert terms[0].operator.matrix().shape == (12, 12)


def test_collective_decay_is_a_single_summed_operator():
    """A collective decay bath produces a single summed jump operator across its targets rather than one per device."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q1")
    chip = Chip([q0, q1], baths=[Bath("collective_decay", rate=0.01)])
    terms = _terms(chip.baths[0], chip)
    assert len(terms) == 1  # ONE summed jump operator, not two independent ones.
    assert float(materialize_expr(terms[0].rate, chip.backend)) == 0.01


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


def test_bath_zero_temperature_has_relaxation_rate_only():
    """A concrete T=0 thermal bath has a nonzero emission rate and zero absorption rate."""
    from quchip.declarative.expr import materialize_expr

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], baths=[Bath("thermal", temperature=0.0, rate=1e-3)])
    relax, absorb = _terms(chip.baths[0], chip)
    relax_array = chip.backend.to_array(materialize_expr(relax.operator, chip.backend))
    assert np.linalg.norm(relax_array) > 0.0
    assert float(materialize_expr(relax.rate, chip.backend)) == pytest.approx(1e-3)
    assert float(materialize_expr(absorb.rate, chip.backend)) == 0.0


def test_baths_flow_into_collected_c_ops():
    """Canonical engine results include each bath's collapse terms."""
    from quchip.engine import build_problem

    class CallableBath(Bath):
        def dissipation(self, chip, bases=None):
            del bases

            def lowering(rate):
                xp = chip.backend.array_module
                return xp.asarray([[0.0, rate * 0.0 + 1.0], [0.0, 0.0]])

            def decay_rate(rate):
                return rate

            return (
                CollapseChannel(
                    lowering,
                    decay_rate,
                    "callable_decay",
                ),
            )

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    no_bath = Chip([q])
    with_bath = Chip([q.copy()], baths=[Bath("thermal", temperature=200.0, rate=1e-3)])
    callable_bath = Chip(
        [q.copy()],
        baths=[CallableBath("collective_decay", rate=2e-3)],
    )
    tlist = np.asarray([0.0, 1.0])
    assert len(build_problem(no_bath, [], tlist).engine_result.collapse_terms) == 0
    assert len(build_problem(with_bath, [], tlist).engine_result.collapse_terms) == 2
    term = build_problem(callable_bath, [], tlist).engine_result.collapse_terms[0]
    assert term.channel == "callable_decay"
    assert float(term.rate) == pytest.approx(2e-3)
