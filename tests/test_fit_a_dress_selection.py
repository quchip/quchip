"""Tests for fit_a_dress's fit_parameters free-parameter selection."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, Resonator, TunableCapacitive, fit_a_dress
from quchip.chip.coupling_base import BaseCoupling
from quchip.declarative import DeviceModel, LocalOps, Scalar, parameter
from quchip.inverse_design.fit import _pack_initial_params, _resolve_fit_parameters


class _ToyDeviceModel(DeviceModel):
    """User-authored declarative DeviceModel that exposes its tunables via the derived default.

    Declares no explicit ``tunable_param_names``, so ``omega``/``anharm``
    reach :meth:`~quchip.devices.base.BaseDevice.tunable_params` through
    derivation (every declared ``parameter()`` field). Exercises the generic
    ``tunable_params``/``set_tunable_param``/``tunable_param_bounds`` seam
    :func:`~quchip.inverse_design.fit.fit_a_dress` already walks — no
    ``fit.py`` change is needed for a model it has never seen.
    """

    _type_prefix = "toy"
    _default_levels = 4

    omega: Scalar = parameter(positive=True, unit="GHz")
    anharm: Scalar = parameter(unit="GHz")

    def local_hamiltonian(self, op: LocalOps):
        return self.omega * op.n + (0.5 * self.anharm) * (op.n @ (op.n - op.I))

    def tunable_param_bounds(self, name: str, value: float) -> tuple[float, float]:
        if name == "omega":
            return (max(1e-6, 0.5 * value), 1.5 * value)
        if name == "anharm":
            return (2.0 * value, -1e-6) if value < 0 else (1e-6, 2.0 * value if value > 0 else 1.0)
        return super().tunable_param_bounds(name, value)


class _KappaCoupling(BaseCoupling):
    """Coupling whose scalar strength lives on ``.kappa``, not ``.g``.

    Declares ``coupling_strength_name`` explicitly, so selecting or
    freezing it through ``fit_parameters`` exercises the generic
    ``coupling_strength_name`` / ``set_coupling_strength`` seam rather than
    an assumed ``.g`` attribute.
    """

    _type_prefix = "kappa_coupling"

    def __init__(self, device_a, device_b, *, kappa, label=None) -> None:
        super().__init__(device_a, device_b, label=label)
        self.kappa = kappa

    @property
    def coupling_strength(self) -> float:
        return self.kappa

    @property
    def coupling_strength_name(self) -> str:
        return "kappa"

    def interaction_hamiltonian(self):
        from quchip.backend import get_default_backend

        backend = get_default_backend()
        return self.kappa * backend.tensor(self.device_a.number_operator(), self.device_b.number_operator())


def _simple_chip() -> tuple[DuffingTransmon, Resonator, Capacitive, Chip]:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=-1.2e-4, label="qr")
    chip = Chip([q, r], [coupling], frame="rotating")
    return q, r, coupling, chip


def test_fit_parameters_none_retains_old_free_set() -> None:
    """fit_parameters=None packs every declared device tunable and every coupling strength."""
    _, _, _, chip = _simple_chip()

    names_default, _ = _pack_initial_params(chip, ())
    names_explicit_none, _ = _pack_initial_params(chip, (), device_selection=None, coupling_selection=None)

    assert set(names_default) == {"q.freq", "q.anharmonicity", "r.freq", "qr.g"}
    assert names_default == names_explicit_none


def test_resolve_fit_parameters_treats_object_and_label_keys_identically() -> None:
    """A fit_parameters key given as a device/coupling object or its label resolves to the same selection."""
    q, r, coupling, chip = _simple_chip()

    by_object = _resolve_fit_parameters(chip, {q: ("freq",), coupling: ()})
    by_label = _resolve_fit_parameters(chip, {"q": ("freq",), "qr": ()})

    assert by_object == by_label


def test_fit_parameters_selection_moves_only_selected_and_freezes_the_rest() -> None:
    """Selected parameters move to hit the target; every unlisted device and coupling stays bit-identical."""
    q, r, coupling, chip = _simple_chip()

    result = fit_a_dress(chip, observable_targets={q: {"freq": 5.05}}, fit_parameters={q: ("freq",)})

    assert set(result.initial_params) == {"q.freq"}
    assert set(result.final_params) == {"q.freq"}
    assert result.final_params["q.freq"] != pytest.approx(result.initial_params["q.freq"])

    fitted_q = result.chip["q"]
    fitted_r = result.chip["r"]
    fitted_coupling = result.chip.couplings[0]
    assert fitted_q.anharmonicity == q.anharmonicity
    assert fitted_r.freq == r.freq
    assert fitted_coupling.g == coupling.g


def test_empty_selection_for_one_component_is_legal() -> None:
    """An explicit empty name-collection freezes that component while other listed components stay free."""
    q, r, coupling, chip = _simple_chip()

    result = fit_a_dress(
        chip,
        coupling_targets={coupling: "chi"},
        fit_parameters={q: (), coupling: (coupling.coupling_strength_name,)},
    )

    assert set(result.final_params) == {"qr.g"}
    assert result.chip["q"].freq == q.freq
    assert result.chip["q"].anharmonicity == q.anharmonicity


def test_zero_total_free_parameters_raises() -> None:
    """A fit_parameters mapping that freezes every listed component and omits the rest raises ValueError."""
    q, r, coupling, chip = _simple_chip()

    with pytest.raises(ValueError, match="zero free parameters"):
        fit_a_dress(chip, fit_parameters={q: (), r: (), coupling: ()})


def test_unknown_component_label_raises_with_available_choices() -> None:
    """An unresolvable fit_parameters key raises ValueError listing the chip's known labels."""
    _, _, _, chip = _simple_chip()

    with pytest.raises(ValueError, match="does not match any device or coupling") as exc_info:
        _resolve_fit_parameters(chip, {"not_a_label": ()})

    message = str(exc_info.value)
    assert "'q'" in message and "'r'" in message and "'qr'" in message


def test_unknown_device_parameter_name_raises_with_available_choices() -> None:
    """An unknown device parameter name raises ValueError listing the device's declared tunables."""
    q, _, _, chip = _simple_chip()

    with pytest.raises(ValueError, match="not tunable parameters") as exc_info:
        _resolve_fit_parameters(chip, {q: ("not_a_param",)})

    message = str(exc_info.value)
    assert "freq" in message and "anharmonicity" in message


def test_unknown_coupling_parameter_name_raises_with_available_choices() -> None:
    """An unknown coupling parameter name raises ValueError listing the coupling's own strength name."""
    _, _, coupling, chip = _simple_chip()

    with pytest.raises(ValueError, match="not the declared coupling-strength") as exc_info:
        _resolve_fit_parameters(chip, {coupling: ("not_g",)})

    assert "'g'" in str(exc_info.value)


def test_duplicate_resolved_keys_raise() -> None:
    """Two fit_parameters keys resolving to the same component label raise ValueError."""
    q, _, _, chip = _simple_chip()

    with pytest.raises(ValueError, match="duplicate"):
        _resolve_fit_parameters(chip, {q: ("freq",), "q": ("anharmonicity",)})


def test_bare_string_value_raises() -> None:
    """A bare string value (instead of a name collection) raises ValueError."""
    q, _, _, chip = _simple_chip()

    with pytest.raises(ValueError, match="collection of parameter names"):
        _resolve_fit_parameters(chip, {q: "freq"})


def test_underdetermined_fit_warns_and_records_solver_info_receipt() -> None:
    """An underdetermined-by-count fit warns and records the three identifiability receipt fields."""
    _, _, _, chip = _simple_chip()

    with pytest.warns(UserWarning, match="underdetermined by count"):
        result = fit_a_dress(chip)

    assert result.solver_info["n_free_parameters"] == 4
    assert result.solver_info["n_target_residuals"] == 3
    assert result.solver_info["underdetermined_by_count"] is True


def test_count_sufficient_fit_does_not_warn() -> None:
    """A fit_parameters selection with free parameters <= target residuals emits no underdetermined warning."""
    q, _, _, chip = _simple_chip()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_a_dress(chip, fit_parameters={q: ("freq", "anharmonicity")})

    assert not any("underdetermined" in str(w.message) for w in caught)
    assert result.solver_info["n_free_parameters"] == 2
    assert result.solver_info["n_target_residuals"] == 3
    assert result.solver_info["underdetermined_by_count"] is False


def test_custom_device_model_with_declared_tunables_fits_through_generic_seam() -> None:
    """A user-authored DeviceModel exposing custom tunable names fits without any fit.py change."""
    toy = _ToyDeviceModel(omega=5.0, anharm=-0.3, levels=4, label="toy")
    r = Resonator(freq=7.0, levels=6, label="r")
    coupling = Capacitive(toy, r, g=0.01, label="tr")
    chip = Chip([toy, r], [coupling], frame="rotating")
    seed_freq = float(chip.freq(toy))
    target_freq = seed_freq * 1.05

    result = fit_a_dress(
        chip,
        observable_targets={toy: {"freq": target_freq}},
        fit_parameters={toy: ("omega",)},
    )

    assert set(result.final_params) == {"toy.omega"}
    achieved = float(result.chip.freq(result.chip["toy"]))
    assert achieved == pytest.approx(target_freq, rel=5e-3, abs=5e-3)
    assert result.chip["toy"].anharm == toy.anharm


def test_custom_coupling_declared_strength_name_can_be_selected_or_frozen() -> None:
    """fit_parameters selects or freezes a custom coupling's own coupling_strength_name, not a stray '.g'."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    coupling = _KappaCoupling(q, r, kappa=0.01, label="kc")
    chip = Chip([q, r], [coupling], frame="rotating")

    frozen = fit_a_dress(chip, fit_parameters={q: ("freq",)})
    assert "kc.kappa" not in frozen.final_params
    assert frozen.chip.couplings[0].kappa == coupling.kappa

    selected = fit_a_dress(
        chip,
        observable_targets={coupling: {"g": 0.05}},
        fit_parameters={coupling: ("kappa",)},
    )
    assert set(selected.final_params) == {"kc.kappa"}
    assert selected.chip.couplings[0].kappa == pytest.approx(0.05, abs=5e-4)


def test_tunable_capacitive_g0_can_be_selected_via_fit_parameters() -> None:
    """A built-in coupling with a non-'g' strength attribute (TunableCapacitive.g_0) selects through its own name."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    coupling = TunableCapacitive(q, r, g_0=0.01, label="tc")
    chip = Chip([q, r], [coupling], frame="rotating")

    result = fit_a_dress(
        chip,
        observable_targets={coupling: {"g": 0.03}},
        fit_parameters={coupling: ("g_0",)},
    )

    assert set(result.final_params) == {"tc.g_0"}
    assert result.chip.couplings[0].g_0 == pytest.approx(0.03, abs=5e-4)


@pytest.mark.optional_backend
def test_jax_backed_selection_gets_exact_jacobian_sized_to_the_reduced_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JAX-native backend keeps the exact Jacobian path, sized to the fit_parameters-reduced vector."""
    pytest.importorskip("dynamiqs")
    from scipy.optimize import least_squares as scipy_least_squares

    from quchip.backend.dynamiqs import DynamiqsBackend
    from quchip.inverse_design import fit as fit_module

    checked = False

    def recording_least_squares(fun, *args, **kwargs):
        nonlocal checked
        x0 = np.asarray(kwargs["x0"], dtype=float)
        jacobian = np.asarray(kwargs["jac"](x0), dtype=float)
        step = 1e-6
        finite_difference = np.column_stack([
            (fun(x0 + step * np.eye(x0.size)[column]) - fun(x0 - step * np.eye(x0.size)[column]))
            / (2.0 * step)
            for column in range(x0.size)
        ])
        np.testing.assert_allclose(jacobian, finite_difference, rtol=2e-4, atol=2e-6)
        assert jacobian.shape[1] == 2
        checked = True
        return scipy_least_squares(fun, *args, **kwargs)

    monkeypatch.setattr(fit_module, "least_squares", recording_least_squares)

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    coupling = Capacitive(q, r, g=0.02, label="qr")
    chip = Chip([q, r], [coupling], frame="rotating", backend=DynamiqsBackend())

    result = fit_a_dress(
        chip,
        observable_targets={q: {"freq": 5.01, "anharmonicity": -0.25}, r: {"freq": 7.01}},
        fit_parameters={q: ("freq", "anharmonicity")},
    )

    assert checked
    assert set(result.final_params) == {"q.freq", "q.anharmonicity"}
    assert result.solver_info["jacobian"] == "jax"
