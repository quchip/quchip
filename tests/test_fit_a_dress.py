"""Tests for fit_a_dress: parameter packing, target fitting, and result introspection."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from quchip import (
    Exact,
    Capacitive,
    ChargeBasisTransmon,
    Chip,
    CrossKerr,
    DuffingTransmon,
    Resonator,
    TunableCapacitive,
    fit_a_dress,
)
from quchip.chip.coupling_base import BaseCoupling
from quchip.devices.base import BaseDevice
from quchip.inverse_design.fit import _estimate_bare_g, _pack_initial_params, _static_exchange_rate
from quchip.inverse_design import fit as fit_module
from quchip.inverse_design.observables import (
    TargetSpec,
    build_dressed_target_specs,
    build_target_specs,
)
from quchip.inverse_design.subsystems import build_local_subsystem, device_labels_for_local_eval
from quchip.inverse_design import FitADressResult, ObservableReport


class _StrengthOnlyCoupling(BaseCoupling):
    """A coupling whose scalar strength lives on ``.strength``, not ``.g``.

    Declares ``coupling_strength_name`` explicitly (unlike the default
    ``"g"``), so this is the general case ``set_coupling_strength`` must
    route through rather than assuming ``.g``.
    """

    _type_prefix = "strength_only"

    def __init__(self, device_a, device_b, *, strength, label=None) -> None:
        super().__init__(device_a, device_b, label=label)
        self.strength = strength

    @property
    def coupling_strength(self) -> float:
        return self.strength

    @property
    def coupling_strength_name(self) -> str:
        return "strength"

    def interaction_hamiltonian(self):
        from typing import cast

        from quchip.backend import get_default_backend

        backend = get_default_backend()
        a = cast(BaseDevice, self.device_a)
        b = cast(BaseDevice, self.device_b)
        return self.strength * backend.tensor(a.number_operator(), b.number_operator())


def test_pack_initial_params_uses_coupling_strength_not_g_attribute() -> None:
    """A user-authored coupling exposing only ``coupling_strength`` (no ``.g``) packs under its own name."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    coupling = _StrengthOnlyCoupling(q, r, strength=0.01, label="custom")
    chip = Chip([q, r], [coupling], frame="rotating")

    names, values = _pack_initial_params(chip, ())

    idx = names.index("custom.strength")
    assert values[idx] == pytest.approx(0.01)
    assert "custom.g" not in names


def test_fit_a_dress_writes_custom_coupling_strength_through_its_own_attribute() -> None:
    """fit_a_dress moves a custom coupling's declared coupling_strength_name attribute, not a stray .g."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    coupling = _StrengthOnlyCoupling(q, r, strength=0.01, label="custom")
    chip = Chip([q, r], [coupling], frame="rotating")

    result = fit_a_dress(chip, observable_targets={coupling: {"g": 0.05}})

    fitted_coupling = result.chip.couplings[0]
    assert fitted_coupling.strength == pytest.approx(0.05, abs=5e-4)
    assert not hasattr(fitted_coupling, "g")
    assert "custom.strength" in result.final_params
    assert "custom.g" not in result.final_params


def test_fit_a_dress_moves_tunable_capacitive_g0_with_no_stray_g_attribute() -> None:
    """fit_a_dress writes a TunableCapacitive's g_0 (not a stray .g) and reproduces the target."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    coupling = TunableCapacitive(q, r, g_0=0.01, label="tc")
    chip = Chip([q, r], [coupling], frame="rotating")

    result = fit_a_dress(chip, observable_targets={coupling: {"g": 0.03}})

    fitted_coupling = result.chip.couplings[0]
    assert fitted_coupling.g_0 == pytest.approx(0.03, abs=5e-4)
    assert not hasattr(fitted_coupling, "g")
    assert "tc.g_0" in result.final_params
    assert "tc.g" not in result.final_params


def test_fit_a_dress_moves_crosskerr_chi_with_no_stray_g_attribute() -> None:
    """fit_a_dress writes a CrossKerr's chi (not a stray .g) and reproduces the target."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    coupling = CrossKerr(q, r, chi=0.001, label="ck")
    chip = Chip([q, r], [coupling], frame="rotating")

    result = fit_a_dress(chip, observable_targets={coupling: {"g": 0.003}})

    fitted_coupling = result.chip.couplings[0]
    assert fitted_coupling.chi == pytest.approx(0.003, abs=5e-4)
    assert not hasattr(fitted_coupling, "g")
    assert "ck.chi" in result.final_params
    assert "ck.g" not in result.final_params


def test_estimate_bare_g_seed_subchip_preserves_chip_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The coupling seed sub-chip preserves basis, approximation, and backend intent."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=0.01, label="c")
    chip = Chip(
        [q, r],
        [coupling],
        frame="rotating",
        basis="eigen",
        backend="qutip",
        approximation=Exact(),
    )

    real_chip = fit_module.Chip
    captured: dict = {}

    def spy_chip(devices, couplings=None, **kwargs):
        captured["backend"] = kwargs.get("backend")
        captured["basis"] = kwargs.get("basis")
        captured["approximation"] = kwargs.get("approximation")
        return real_chip(devices, couplings, **kwargs)

    monkeypatch.setattr(fit_module, "Chip", spy_chip)

    _estimate_bare_g(chip, coupling, TargetSpec("chi", coupling.label, 1e-4))

    assert captured["backend"] is chip.backend
    assert captured["basis"] == "eigen"
    assert captured["approximation"] == Exact()


def test_local_fit_subsystem_inherits_chip_basis_policy() -> None:
    """Local fit evaluation retains an inherited energy-basis projection."""
    q = ChargeBasisTransmon(E_C=0.25, E_J=12.0, num_basis=9, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.02)], basis="eigen")

    local = build_local_subsystem(chip, ("q", "r"))
    resolved = local.resolve(frame="lab")

    assert local.basis == "eigen"
    assert resolved.dims == (3, 4)
    assert resolved.bases["q"].kind == "eigen"


def test_estimate_bare_g_raises_when_target_is_not_bracketed() -> None:
    """_estimate_bare_g raises ValueError (never a saturated endpoint) when the target is unreachable."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=0.001, label="c")
    chip = Chip([q, r], [coupling], frame="rotating")

    huge_target = 1000.0
    with pytest.raises(ValueError, match=r"1000\.0") as exc_info:
        _estimate_bare_g(chip, coupling, TargetSpec("chi", coupling.label, huge_target))

    message = str(exc_info.value)
    assert "1e-06, 0.25" in message


def test_estimate_bare_g_solves_correct_root_for_a_decreasing_observable(monkeypatch: pytest.MonkeyPatch) -> None:
    """_estimate_bare_g finds the true root even when the observable DECREASES with coupling strength.

    A bisection loop that always assumes "observable increases with
    strength" converges to the wrong endpoint on a decreasing
    observable (it moves the bracket in the wrong direction every
    iteration). The synthetic ``_chi`` below is monotonically
    decreasing on ``seed_strength_bounds`` with a known root, so any
    direction-dependent solver is caught red-handed; a direction-
    independent root solve (``scipy.optimize.brentq``) is not.
    """
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=0.01, label="c")
    chip = Chip([q, r], [coupling], frame="rotating")

    def decreasing_chi(sub_chip, sub_coupling):
        # chi(strength) = 0.5 - strength: root at strength=0.2 for target=0.3,
        # strictly decreasing and strictly positive over (1e-6, 0.25).
        return 0.5 - sub_coupling.coupling_strength

    monkeypatch.setattr(fit_module, "_chi", decreasing_chi)

    seed = _estimate_bare_g(chip, coupling, TargetSpec("chi", coupling.label, 0.3))

    assert seed == pytest.approx(0.2, abs=1e-8)


def test_fit_a_dress_public_exports() -> None:
    """fit_a_dress, FitADressResult, and ObservableReport form the public API surface."""
    assert callable(fit_a_dress)
    assert FitADressResult.__name__ == "FitADressResult"
    assert ObservableReport.__name__ == "ObservableReport"


def test_devices_declare_numeric_dressed_fit_defaults_without_dressing() -> None:
    """Common spectral devices map constructor numbers to dressed targets directly."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")

    assert q.default_dressed_targets() == {"freq": 5.0, "anharmonicity": -0.25}
    assert q.default_fit_parameters() == ("freq", "anharmonicity")
    assert r.default_dressed_targets() == {"freq": 7.0}
    assert r.default_fit_parameters() == ("freq",)


@pytest.mark.parametrize(
    "coupling",
    [
        lambda q, r: Capacitive(q, r, g=-0.00025, label="cap"),
        lambda q, r: TunableCapacitive(q, r, g_0=-0.00025, label="tunable"),
        lambda q, r: CrossKerr(q, r, chi=-0.00025, label="crosskerr"),
    ],
)
def test_dispersive_couplings_declare_cross_kerr_as_their_default_fit_target(coupling) -> None:
    """Qubit-bearing dispersive edges retain their cross-Kerr target."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    edge = coupling(q, r)

    assert edge.default_dressed_target() == ("cross_kerr", -0.00025)


def test_capacitive_between_noncomputational_modes_defaults_to_exchange_rate() -> None:
    """A non-computational edge targets its dressed exchange rate."""
    readout = Resonator(freq=7.0, levels=4, label="readout")
    filter_mode = Resonator(freq=7.2, levels=4, label="filter")
    edge = Capacitive(readout, filter_mode, g=0.03, label="readout-filter")
    desired = Chip([readout, filter_mode], [edge], frame="rotating")

    assert edge.default_dressed_target() == ("exchange_rate", 0.03)
    assert [
        (spec.kind, spec.label, spec.target)
        for spec in build_dressed_target_specs(desired)
        if spec.label == "readout-filter"
    ] == [("exchange_rate", "readout-filter", 0.03)]


def test_fit_a_dress_matches_noncomputational_capacitive_exchange_rate() -> None:
    """The automatic plan fits a resonator pair through dressed exchange."""
    readout = Resonator(freq=7.0, levels=4, label="readout")
    filter_mode = Resonator(freq=7.2, levels=4, label="filter")
    edge = Capacitive(readout, filter_mode, g=0.03, label="readout-filter")

    fit = fit_a_dress(Chip([readout, filter_mode], [edge], frame="rotating"), max_nfev=300)

    report = next(item for item in fit.final_targets if item.label == "readout-filter")
    assert report.kind == "exchange_rate"
    assert report.final == pytest.approx(0.03, abs=1e-8)
    assert float(_static_exchange_rate(fit.chip, ("readout", "filter"))) == pytest.approx(0.03, abs=1e-8)


def test_dressed_target_compilation_never_evaluates_the_desired_chip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The desired chip is a numeric specification, not a runnable seed model."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    edge = Capacitive(q, r, g=-0.00025, label="qr")
    desired = Chip([q, r], [edge], frame="rotating")

    def forbidden(*args, **kwargs):
        raise AssertionError("desired chip was evaluated")

    monkeypatch.setattr(desired, "freq", forbidden)
    monkeypatch.setattr(desired, "dressed_anharmonicity", forbidden)
    monkeypatch.setattr(desired, "static_zz", forbidden)

    specs = build_dressed_target_specs(desired)

    assert [(spec.kind, spec.label, spec.target) for spec in specs] == [
        ("freq", "q", 5.0),
        ("anharmonicity", "q", -0.25),
        ("freq", "r", 7.0),
        ("cross_kerr", "qr", -0.00025),
    ]
    assert [spec.source for spec in specs] == [
        "component default",
        "component default",
        "component default",
        "component default",
    ]


def test_explicit_constraints_extend_replace_and_remove_component_defaults() -> None:
    """Pair constraints are additive; same-edge values replace or remove defaults."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    edge = Capacitive(q0, bus, g=-0.00025, label="q0-bus")
    desired = Chip([q0, q1, bus], [edge], frame="rotating")

    specs = build_dressed_target_specs(
        desired,
        constraints={
            edge: {"cross_kerr": -0.0003},
            (q0, q1): {"exchange_rate": -0.0022},
            (q1, bus): {"zz": 0.00015},
        },
    )
    keyed = {(spec.kind, spec.label): spec.target for spec in specs}

    assert keyed[("cross_kerr", "q0-bus")] == -0.0003
    assert keyed[("exchange_rate", ("q0", "q1"))] == -0.0022
    assert keyed[("cross_kerr", ("q1", "bus"))] == 0.00015
    assert all(
        spec.source == "explicit"
        for spec in specs
        if (spec.kind, spec.label)
        in {
            ("cross_kerr", "q0-bus"),
            ("exchange_rate", ("q0", "q1")),
            ("cross_kerr", ("q1", "bus")),
        }
    )

    without_edge_default = build_dressed_target_specs(
        desired,
        constraints={edge: {"cross_kerr": None}},
    )
    assert ("cross_kerr", "q0-bus") not in {(spec.kind, spec.label) for spec in without_edge_default}


def test_fit_a_dress_treats_a_capacitive_scalar_as_a_cross_kerr_target() -> None:
    """The desired edge number is a dressed constraint, not the fitted bare g."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    edge = Capacitive(q, r, g=-0.00025, label="qr")
    desired = Chip([q, r], [edge], frame="rotating")

    fit = fit_a_dress(desired, max_nfev=300)

    assert fit.chip.static_zz("q", "r") == pytest.approx(-0.00025, abs=2e-6)
    assert fit.final_params["qr.g"] > 0.0
    assert fit.solver_info["n_free_parameters"] == 4
    assert fit.solver_info["n_target_residuals"] == 4
    assert fit.solver_info["input_contract"] == "desired-chip"
    assert {(report.kind, report.label): report.source for report in fit.final_targets}[
        ("cross_kerr", "qr")
    ] == "component default"
    coupling_report = next(report for report in fit.parameter_reports if report.name == "qr.g")
    assert coupling_report.seed_source == "isolated-pair root solve"
    assert coupling_report.sign_choice == "positive convention"
    assert coupling_report.lower_bound < 0.0 < coupling_report.upper_bound


def test_fit_a_dress_records_a_user_supplied_coupling_sign() -> None:
    """An explicit coupling start owns the sign branch and the receipt says so."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    edge = Capacitive(q, r, g=-0.00025, label="qr")
    desired = Chip([q, r], [edge], frame="rotating")

    fit = fit_a_dress(desired, start={"qr.g": -0.05})

    report = next(report for report in fit.parameter_reports if report.name == "qr.g")
    assert report.initial == pytest.approx(-0.05)
    assert report.final < 0.0
    assert report.seed_source == "user start"
    assert report.sign_choice == "user supplied"


def test_fit_a_dress_summary_is_a_compact_human_readable_receipt() -> None:
    """The common result inspection path is one summary, not several dict dumps."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    fit = fit_a_dress(Chip([q], frame="rotating"))

    summary = fit.summary()

    assert "fit_a_dress: converged" in summary
    assert "targets: 2" in summary
    assert "parameters: 2" in summary
    assert "identifiability: rank 2/2" in summary
    assert "q.freq" in summary
    assert "q.anharmonicity" in summary
    assert repr(fit) != summary


def test_parameter_report_is_part_of_the_public_inverse_design_api() -> None:
    """Parameter receipts are typed public data, not an undocumented solver-info blob."""
    from quchip.inverse_design import FitParameterReport

    report = FitParameterReport(
        name="q.freq",
        initial=4.9,
        final=5.0,
        lower_bound=0.0,
        upper_bound=10.0,
        seed_source="component declaration",
    )

    assert report.delta == pytest.approx(0.1)


def test_automatic_desired_chip_fit_rejects_a_rank_deficient_jacobian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal target/parameter counts do not make an automatic flat direction safe."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    desired = Chip([q], frame="rotating")

    def only_freq(candidate, spec, evaluator):
        del spec, evaluator
        return candidate["q"].freq

    monkeypatch.setattr(fit_module, "_evaluate_spec", only_freq)

    with pytest.raises(ValueError, match=r"Jacobian rank 1 for 2 free parameters"):
        fit_a_dress(desired)


def test_manual_rank_deficient_fit_returns_diagnostics_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit vary plan may return ambiguity, but it cannot hide it."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    desired = Chip([q], frame="rotating")

    def only_freq(candidate, spec, evaluator):
        del spec, evaluator
        return candidate["q"].freq

    monkeypatch.setattr(fit_module, "_evaluate_spec", only_freq)

    with pytest.warns(UserWarning, match=r"Jacobian rank 1 for 2 free parameters"):
        fit = fit_a_dress(
            desired,
            vary={q: ("freq", "anharmonicity")},
        )

    assert fit.solver_info["jacobian_rank"] == 1
    assert fit.solver_info["rank_deficient"] is True
    assert np.isinf(fit.solver_info["jacobian_condition_number"])
    weak = fit.solver_info["weak_parameter_directions"]
    assert weak
    assert abs(weak[0]["relative_weights"]["q.anharmonicity"]) == pytest.approx(1.0)


def test_automatic_desired_chip_fit_rejects_an_underdetermined_target_plan() -> None:
    """Removing a default target cannot leave an automatic free parameter floating."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    desired = Chip([q], frame="rotating")

    with pytest.raises(ValueError, match=r"2 free parameters but only 1 target residual"):
        fit_a_dress(desired, constraints={q: {"anharmonicity": None}})


def test_fit_a_dress_accepts_manual_vary_and_start_overrides() -> None:
    """Advanced users can replace automatic parameter selection and starting values."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    desired = Chip([q], frame="rotating")

    fit = fit_a_dress(
        desired,
        vary={q: ("freq",)},
        start={"q.freq": 4.8},
    )

    assert fit.initial_params == {"q.freq": 4.8}
    assert fit.final_params["q.freq"] == pytest.approx(5.0, abs=1e-8)
    assert fit.chip.dressed_anharmonicity("q") == pytest.approx(-0.25, abs=1e-8)


def test_desired_chip_selection_errors_name_vary_not_the_legacy_keyword() -> None:
    """Desired-chip errors use the vocabulary shown in its public signature."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    desired = Chip([q])

    with pytest.raises(ValueError, match=r"vary\['q'\]"):
        fit_a_dress(desired, vary={q: "freq"})


def test_compatibility_keywords_emit_one_caller_facing_deprecation_warning() -> None:
    """The old keyword family gives one actionable warning at the user's call site."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_a_dress(chip, fit_parameters={q: ("freq",)})

    deprecations = [item for item in caught if issubclass(item.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert deprecations[0].filename == __file__
    assert "deprecated since quchip 0.2.1" in str(deprecations[0].message)
    assert "removed in 0.3.0" in str(deprecations[0].message)
    assert "vary=" in str(deprecations[0].message)


def test_desired_chip_api_does_not_emit_a_deprecation_warning() -> None:
    """The replacement constraints/vary/start contract stays warning-free."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_a_dress(Chip([q], frame="rotating"))

    assert not any(issubclass(item.category, DeprecationWarning) for item in caught)


def test_fit_a_dress_retains_scipy_jacobian_without_dynamiqs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A QuTiP-only installation retains SciPy's numerical Jacobian."""

    def unavailable(*args, **kwargs):
        raise ImportError("dynamiqs unavailable")

    monkeypatch.setattr(fit_module, "_jax_residual_functions", unavailable)
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating")

    result = fit_a_dress(chip)

    assert result.solver_info["jacobian"] == "finite-difference"


def test_fit_a_dress_respects_a_qutip_chip_backend() -> None:
    """A QuTiP chip retains SciPy's numerical Jacobian when dynamiqs is installed."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating", backend="qutip")

    result = fit_a_dress(chip)

    assert result.solver_info["jacobian"] == "finite-difference"


def test_fit_a_dress_recovers_qr_target_chi_from_declared_coupling_value() -> None:
    """fit_a_dress recovers qubit-resonator chi from a coupling's declared g."""
    q = DuffingTransmon(freq=5.241031326, anharmonicity=-0.261031326, levels=4, label="q")
    r = Resonator(freq=6.653024480, levels=10, label="r")
    coupling = Capacitive(q, r, g=-646019e-9)
    chip = Chip([q, r], [coupling], frame="rotating")

    result = fit_a_dress(chip, coupling_targets={coupling: "chi"}, max_hilbert_dim=10_000)

    assert result.chip is not chip
    fitted_chip = result.chip
    fitted_q = fitted_chip["q"]
    fitted_r = fitted_chip["r"]
    fitted_c = fitted_chip.couplings[0]

    chi = (fitted_chip.freq(fitted_r, when={fitted_q: 1}) - fitted_chip.freq(fitted_r, when={fitted_q: 0})) / 2.0
    assert fitted_chip.freq(fitted_q) == pytest.approx(5.241031326, abs=5e-4)
    assert fitted_chip.freq(fitted_r) == pytest.approx(6.653024480, abs=5e-4)
    assert fitted_chip.dressed_anharmonicity(fitted_q) == pytest.approx(-0.261031326, abs=5e-4)
    assert chi == pytest.approx(-646019e-9, abs=5e-6)
    assert np.isfinite(fitted_c.g)


def test_fit_a_dress_recovers_qq_target_zz_from_declared_coupling_value() -> None:
    """fit_a_dress recovers static ZZ between two qubits from a coupling's declared g."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.18, anharmonicity=-0.24, levels=3, label="q1")
    coupling = Capacitive(q0, q1, g=0.0015)
    chip = Chip([q0, q1], [coupling], frame="rotating")

    result = fit_a_dress(chip, coupling_targets={coupling: "zz"}, max_hilbert_dim=10_000)

    fitted_chip = result.chip
    fitted_q0 = fitted_chip["q0"]
    fitted_q1 = fitted_chip["q1"]
    assert fitted_chip.freq(fitted_q0) == pytest.approx(5.0, abs=5e-4)
    assert fitted_chip.freq(fitted_q1) == pytest.approx(5.18, abs=5e-4)
    assert fitted_chip.static_zz(fitted_q0, fitted_q1) == pytest.approx(0.0015, abs=5e-5)


def test_fit_a_dress_does_not_mutate_input_chip() -> None:
    """fit_a_dress leaves the input chip's device and coupling parameters unmutated."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=-1.2e-4)
    chip = Chip([q, r], [coupling], frame="rotating")

    original = (q.freq, q.anharmonicity, coupling.g)
    _ = fit_a_dress(chip)

    assert (q.freq, q.anharmonicity, coupling.g) == original


def test_bare_g_seed_uses_isolated_subchip_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """_estimate_bare_g calls chip.freq safely even when repr() is invoked on a target device."""

    class ReprDevice(BaseDevice):
        _type_prefix = "repr_device"

        def __init__(self, freq: float, *, computational: bool, label: str, anharmonicity: float = 0.0) -> None:
            super().__init__(levels=3, label=label)
            self.freq = freq
            self.anharmonicity = anharmonicity
            self._computational = computational
            self._finish_init()

        def unresolved_hamiltonian(self):
            # A genuinely anharmonic (Duffing-like) diagonal spectrum: two purely
            # harmonic coupled devices have an exactly-zero dispersive shift for
            # any coupling strength, which would make the "chi" target below
            # unbracketable — not a repr-safety concern, just flat physics.
            import jax.numpy as jnp

            levels = jnp.arange(self.levels)
            energies = self.freq * levels + (self.anharmonicity / 2.0) * levels * (levels - 1)
            return jnp.diag(energies.astype(complex))

        @property
        def computational(self) -> bool:
            return self._computational

    q = ReprDevice(freq=5.0, computational=True, label="q", anharmonicity=-0.3)
    r = ReprDevice(freq=7.0, computational=False, label="r")
    coupling = Capacitive(q, r, g=0.001)
    chip = Chip([q, r], [coupling], frame="rotating")

    original_freq = Chip.freq

    def repr_then_freq(self, target=None, when=None):
        if target is not None:
            repr(target)
        return original_freq(self, target, when=when)

    monkeypatch.setattr(Chip, "freq", repr_then_freq)

    seed = _estimate_bare_g(chip, coupling, TargetSpec("chi", coupling.label, 1e-4))

    assert np.isfinite(seed)
    assert repr(q)


def test_base_device_repr_is_safe_with_multiple_chip_contexts() -> None:
    """A device's repr reports '<multiple chip contexts>' rather than raising when shared across chips."""

    class ReprDevice(BaseDevice):
        _type_prefix = "repr_device"

        def __init__(self, freq: float, label: str) -> None:
            super().__init__(levels=2, label=label)
            self.freq = freq
            self._finish_init()

        def unresolved_hamiltonian(self):
            return self.freq * self.number_operator()

    q = ReprDevice(freq=5.0, label="q")
    _ = Chip([q], label="a")
    _ = Chip([q], label="b")

    text = repr(q)

    assert "dressed_freq=<multiple chip contexts>" in text


def test_fit_a_dress_respects_coupling_target_override_to_g() -> None:
    """fit_a_dress fits a coupling's raw g directly when the target kind is 'g'."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.04)], frame="rotating")

    result = fit_a_dress(chip, coupling_targets={chip.couplings[0]: "g"})

    assert result.final_params[f"{chip.couplings[0].label}.g"] == pytest.approx(0.04, abs=5e-4)


def test_fit_a_dress_switches_to_local_subsystems_above_threshold() -> None:
    """fit_a_dress switches to local-subsystem evaluation once max_hilbert_dim is exceeded."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    r0 = Resonator(freq=7.0, levels=3, label="r0")
    r1 = Resonator(freq=7.3, levels=3, label="r1")
    chip = Chip(
        [q0, q1, r0, r1],
        [
            Capacitive(q0, r0, g=-1.0e-4),
            Capacitive(q1, r1, g=-1.2e-4),
            Capacitive(q0, q1, g=0.001),
        ],
        frame="rotating",
    )

    result = fit_a_dress(chip, max_hilbert_dim=50)

    assert any(report.evaluator == "local" for report in result.final_targets)


def test_fit_a_dress_returns_structured_result_fields() -> None:
    """fit_a_dress returns a result exposing history, loss, solver_info, and target/param snapshots."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=-1.2e-4)], frame="rotating")

    result = fit_a_dress(chip)

    assert result.history.shape[0] >= 1
    assert result.loss >= 0.0
    assert result.solver_info["method"] == "trf"
    assert result.solver_info["n_free_parameters"] == 4
    assert result.solver_info["n_target_residuals"] == 4
    assert result.solver_info["underdetermined_by_count"] is False
    assert result.solver_info["input_contract"] == "desired-chip"
    assert result.initial_targets
    assert result.final_targets
    assert result.initial_params
    assert result.final_params


def test_fit_a_dress_history_records_objective_evaluations() -> None:
    """The returned history supports a real convergence plot, not two endpoints."""
    q = DuffingTransmon(freq=4.8, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating", backend="qutip")

    result = fit_a_dress(
        chip,
        observable_targets={q: {"freq": 5.1}},
        fit_parameters={q: ("freq",)},
    )

    assert result.history.ndim == 1
    assert len(result.history) > 2
    assert result.history[0] > result.history[-1]
    assert result.history[-1] == pytest.approx(result.loss)
    assert result.solver_info["history_axis"] == "distinct residual evaluation"
    assert result.solver_info["n_recorded_evaluations"] == len(result.history)


def test_fit_rebind_returns_fitted_clones_for_seed_devices() -> None:
    """fit.rebind(*seeds) shortcircuits the ``chip.device_map[qb.label]`` ritual."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=-1.2e-4)], frame="rotating")

    result = fit_a_dress(chip)

    q_f, r_f = result.rebind(q, r)
    assert q_f is result.chip.device_map["q"]
    assert r_f is result.chip.device_map["r"]

    assert result.rebind(q) is result.chip.device_map["q"]
    assert result.rebind("r") is result.chip.device_map["r"]

    assert q_f is not q
    assert r_f is not r

    import pytest as _pytest

    with _pytest.raises(ValueError):
        result.rebind()


def test_build_target_specs_accepts_explicit_observables_and_suppresses_coupling_targets() -> None:
    """Explicit observable_targets suppress the coupling-implied chi/zz/g targets."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=-1.2e-4)
    chip = Chip([q, r], [coupling], frame="rotating")

    specs = build_target_specs(
        chip,
        {},
        {
            q: {"freq": 5.0},
            (q, r): {"custom_metric": 0.123},
        },
    )

    assert not any(spec.kind in {"chi", "zz", "g"} and spec.label == coupling.label for spec in specs)
    assert any(spec.kind == "freq" and spec.label == "q" and spec.target == pytest.approx(5.0) for spec in specs)
    assert any(
        spec.kind == "custom_metric" and spec.label == ("q", "r") and spec.target == pytest.approx(0.123)
        for spec in specs
    )


def test_fit_a_dress_accepts_explicit_observable_targets_with_object_labels() -> None:
    """fit_a_dress accepts device objects as observable_targets keys."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=-1.2e-4)
    chip = Chip([q, r], [coupling], frame="rotating")

    with pytest.warns(UserWarning, match="underdetermined by count"):
        result = fit_a_dress(
            chip,
            observable_targets={q: {"freq": 5.0}, r: {"freq": 7.0}},
        )

    assert not any(report.kind in {"chi", "zz", "g"} for report in result.final_targets)
    assert any(report.kind == "freq" and report.label == "q" for report in result.final_targets)
    assert any(report.kind == "freq" and report.label == "r" for report in result.final_targets)


def test_build_target_specs_explicit_observables_override_auto_device_targets() -> None:
    """Explicit observable_targets override the auto-generated device freq/anharmonicity targets."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    chip = Chip([q], frame="rotating")

    specs = build_target_specs(
        chip,
        {},
        {
            q: {"freq": 5.114, "anharmonicity": -0.330},
        },
    )

    freq_specs = [spec for spec in specs if spec.kind == "freq" and spec.label == "q"]
    anh_specs = [spec for spec in specs if spec.kind == "anharmonicity" and spec.label == "q"]

    assert len(freq_specs) == 1
    assert freq_specs[0].target == pytest.approx(5.114)
    assert len(anh_specs) == 1
    assert anh_specs[0].target == pytest.approx(-0.330)


def test_fit_a_dress_recovers_static_exchange_for_sheldon_style_bus_model() -> None:
    """fit_a_dress recovers a targeted static exchange coupling in a bus-mediated three-device system."""
    control = DuffingTransmon(freq=5.08, anharmonicity=-0.31, levels=4, label="control")
    target = DuffingTransmon(freq=4.95, anharmonicity=-0.35, levels=4, label="target")
    bus = Resonator(freq=6.28, levels=6, label="bus")
    c_bus = Capacitive(control, bus, g=0.020, label="c_bus")
    t_bus = Capacitive(target, bus, g=0.017, label="t_bus")
    chip = Chip([control, target, bus], [c_bus, t_bus], frame="rotating")

    # 7 free bare parameters (control freq/anharmonicity, target freq/anharmonicity, bus
    # freq, c_bus.g, t_bus.g) against 6 target residuals: underdetermined by count, yet
    # the fit converges because the exchange target and the two per-device anchors jointly
    # pin the coupling split closely enough from these seeds.
    with pytest.warns(UserWarning, match="underdetermined by count"):
        result = fit_a_dress(
            chip,
            observable_targets={
                control: {"freq": 5.114, "anharmonicity": -0.330},
                target: {"freq": 4.914, "anharmonicity": -0.330},
                bus: {"freq": 6.31},
                (control, target): {"exchange": 0.0038},
            },
            max_hilbert_dim=1_000,
        )

    fitted_chip = result.chip
    fitted_control = fitted_chip["control"]
    fitted_target = fitted_chip["target"]
    fitted_bus = fitted_chip["bus"]
    exchange_h = fitted_chip.effective_subspace_hamiltonian(
        ({fitted_control: 1, fitted_target: 0, fitted_bus: 0}, {fitted_control: 0, fitted_target: 1, fitted_bus: 0})
    )

    assert fitted_chip.freq(fitted_control) == pytest.approx(5.114, abs=1e-3)
    assert fitted_chip.freq(fitted_target) == pytest.approx(4.914, abs=1e-3)
    assert fitted_chip.freq(fitted_bus) == pytest.approx(6.31, abs=1e-3)
    assert fitted_chip.dressed_anharmonicity(fitted_control) == pytest.approx(-0.330, abs=2e-3)
    assert fitted_chip.dressed_anharmonicity(fitted_target) == pytest.approx(-0.330, abs=2e-3)
    assert exchange_h[0, 1] == pytest.approx(0.0038, abs=2e-4)


def test_fit_a_dress_respects_signed_exchange_target_for_direct_qq_system() -> None:
    """fit_a_dress preserves the sign of a targeted direct qubit-qubit exchange coupling."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.18, anharmonicity=-0.24, levels=3, label="q1")
    coupling = Capacitive(q0, q1, g=-0.0015, label="qq")
    chip = Chip([q0, q1], [coupling], frame="rotating")

    result = fit_a_dress(
        chip,
        observable_targets={
            q0: {"freq": 5.0, "anharmonicity": -0.25},
            q1: {"freq": 5.18, "anharmonicity": -0.24},
            (q0, q1): {"exchange": -0.0015},
        },
    )

    fitted_q0 = result.chip["q0"]
    fitted_q1 = result.chip["q1"]
    exchange_h = result.chip.effective_subspace_hamiltonian(
        ({fitted_q0: 1, fitted_q1: 0}, {fitted_q0: 0, fitted_q1: 1})
    )
    assert exchange_h[0, 1] == pytest.approx(-0.0015, abs=5e-5)


def test_fit_a_dress_recovers_explicit_pair_zz_target_for_direct_qq_system() -> None:
    """fit_a_dress recovers an explicit pair-level zz target for a direct qubit-qubit system."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.18, anharmonicity=-0.24, levels=3, label="q1")
    coupling = Capacitive(q0, q1, g=0.0015, label="qq")
    chip = Chip([q0, q1], [coupling], frame="rotating")

    result = fit_a_dress(
        chip,
        observable_targets={
            q0: {"freq": 5.0, "anharmonicity": -0.25},
            q1: {"freq": 5.18, "anharmonicity": -0.24},
            (q0, q1): {"zz": 0.0015},
        },
    )

    fitted_q0 = result.chip["q0"]
    fitted_q1 = result.chip["q1"]
    assert result.chip.static_zz(fitted_q0, fitted_q1) == pytest.approx(0.0015, abs=5e-5)


def test_device_labels_for_local_eval_stays_one_hop_for_pair_targets() -> None:
    """device_labels_for_local_eval expands only one hop beyond single or pair targets."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=3, label="q1")
    q2 = DuffingTransmon(freq=5.2, anharmonicity=-0.25, levels=3, label="q2")
    q3 = DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=3, label="q3")
    chip = Chip(
        [q0, q1, q2, q3],
        [Capacitive(q0, q1, g=0.001), Capacitive(q1, q2, g=0.001), Capacitive(q2, q3, g=0.001)],
        frame="rotating",
    )

    assert device_labels_for_local_eval(chip, "q1") == ("q0", "q1", "q2")
    assert device_labels_for_local_eval(chip, ("q1", "q2")) == ("q0", "q1", "q2", "q3")


def test_fit_a_dress_accepts_string_coupling_target_keys() -> None:
    """fit_a_dress accepts string labels as coupling_targets keys."""
    q = DuffingTransmon(freq=5.241031326, anharmonicity=-0.261031326, levels=4, label="q")
    r = Resonator(freq=6.653024480, levels=10, label="r")
    coupling = Capacitive(q, r, g=-646019e-9, label="qr")
    chip = Chip([q, r], [coupling], frame="rotating")

    result = fit_a_dress(chip, coupling_targets={"qr": "chi"}, max_hilbert_dim=10_000)

    assert any(report.kind == "chi" and report.label == "qr" for report in result.final_targets)


def test_static_exchange_rate_matches_pinned_value_on_bus_coupled_pair() -> None:
    """The static exchange seam agrees with independent effective-subspace analysis."""
    control = DuffingTransmon(freq=5.08, anharmonicity=-0.31, levels=4, label="control")
    target = DuffingTransmon(freq=4.95, anharmonicity=-0.35, levels=4, label="target")
    bus = Resonator(freq=6.28, levels=6, label="bus")
    c_bus = Capacitive(control, bus, g=0.020, label="c_bus")
    t_bus = Capacitive(target, bus, g=0.017, label="t_bus")
    chip = Chip([control, target, bus], [c_bus, t_bus], frame="rotating")

    got = float(_static_exchange_rate(chip, ("control", "target")))
    oracle = chip.effective_subspace_hamiltonian(({control: 1, target: 0, bus: 0}, {control: 0, target: 1, bus: 0}))
    assert got == pytest.approx(complex(oracle[0, 1]).real, abs=1e-10)


def test_build_target_specs_rejects_chi_target_with_both_endpoints_computational() -> None:
    """A 'chi' coupling target with both endpoints computational raises ValueError."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    coupling = Capacitive(q0, q1, g=0.01, label="qq")
    chip = Chip([q0, q1], [coupling], frame="rotating")

    with pytest.raises(ValueError, match="exactly one computational endpoint"):
        build_target_specs(chip, {coupling: "chi"}, None)


def test_build_target_specs_rejects_chi_target_with_neither_endpoint_computational() -> None:
    """A 'chi' coupling target with neither endpoint computational raises ValueError."""
    r0 = Resonator(freq=7.0, levels=4, label="r0")
    r1 = Resonator(freq=7.3, levels=4, label="r1")
    coupling = Capacitive(r0, r1, g=0.01, label="rr")
    chip = Chip([r0, r1], [coupling], frame="rotating")

    with pytest.raises(ValueError, match="exactly one computational endpoint"):
        build_target_specs(chip, {coupling: "chi"}, None)


def test_build_target_specs_rejects_explicit_observable_chi_target_with_both_endpoints_computational() -> None:
    """An explicit observable_targets 'chi' entry with both endpoints computational also raises.

    The validation must not be limited to coupling_targets-derived specs
    — the same 'chi' semantics apply regardless of how the TargetSpec
    was constructed.
    """
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    coupling = Capacitive(q0, q1, g=0.01, label="qq")
    chip = Chip([q0, q1], [coupling], frame="rotating")

    with pytest.raises(ValueError, match="exactly one computational endpoint"):
        build_target_specs(chip, {}, {coupling: {"chi": 1e-4}})


def test_build_target_specs_rejects_explicit_observable_chi_target_with_neither_endpoint_computational() -> None:
    """An explicit observable_targets 'chi' entry with neither endpoint computational also raises."""
    r0 = Resonator(freq=7.0, levels=4, label="r0")
    r1 = Resonator(freq=7.3, levels=4, label="r1")
    coupling = Capacitive(r0, r1, g=0.01, label="rr")
    chip = Chip([r0, r1], [coupling], frame="rotating")

    with pytest.raises(ValueError, match="exactly one computational endpoint"):
        build_target_specs(chip, {}, {coupling: {"chi": 1e-4}})
