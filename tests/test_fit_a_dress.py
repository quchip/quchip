"""Tests for fit_a_dress: parameter packing, target fitting, and result introspection."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import Capacitive, Chip, CrossKerr, DuffingTransmon, Resonator, TunableCapacitive, fit_a_dress
from quchip.chip.coupling_base import BaseCoupling
from quchip.devices.base import BaseDevice
from quchip.inverse_design.fit import _estimate_bare_g, _pack_initial_params, _static_exchange_rate
from quchip.inverse_design import fit as fit_module
from quchip.inverse_design.observables import TargetSpec, build_target_specs
from quchip.inverse_design.subsystems import device_labels_for_local_eval
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


def test_estimate_bare_g_seed_subchip_preserves_rwa_override_and_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """_estimate_bare_g's seed sub-chip preserves the coupling's rwa override and the chip's backend."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    coupling = Capacitive(q, r, g=0.01, label="c")
    coupling.rwa = False
    chip = Chip([q, r], [coupling], frame="rotating", backend="qutip")

    real_chip = fit_module.Chip
    captured: dict = {}

    def spy_chip(devices, couplings=None, **kwargs):
        captured["backend"] = kwargs.get("backend")
        captured["coupling_rwa"] = couplings[0].rwa if couplings else None
        return real_chip(devices, couplings, **kwargs)

    monkeypatch.setattr(fit_module, "Chip", spy_chip)

    _estimate_bare_g(chip, coupling, TargetSpec("chi", coupling.label, 1e-4))

    assert captured["backend"] is chip._backend
    assert captured["coupling_rwa"] is False


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
    with pytest.warns(UserWarning, match="underdetermined by count"):
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

        def hamiltonian(self):
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

        def hamiltonian(self):
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
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=4, label="q1")
    r0 = Resonator(freq=7.0, levels=10, label="r0")
    r1 = Resonator(freq=7.3, levels=10, label="r1")
    chip = Chip(
        [q0, q1, r0, r1],
        [Capacitive(q0, r0, g=-1.0e-4), Capacitive(q1, r1, g=-1.2e-4), Capacitive(q0, q1, g=-0.001)],
        frame="rotating",
    )

    with pytest.warns(UserWarning, match="underdetermined by count"):
        result = fit_a_dress(chip, max_hilbert_dim=100)

    assert any(report.evaluator == "local" for report in result.final_targets)


def test_fit_a_dress_returns_structured_result_fields() -> None:
    """fit_a_dress returns a result exposing history, loss, solver_info, and target/param snapshots."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=-1.2e-4)], frame="rotating")

    with pytest.warns(UserWarning, match="underdetermined by count"):
        result = fit_a_dress(chip)

    assert result.history.shape[0] >= 1
    assert result.loss >= 0.0
    assert result.solver_info["method"] == "trf"
    assert result.solver_info["n_free_parameters"] == 4
    assert result.solver_info["n_target_residuals"] == 3
    assert result.solver_info["underdetermined_by_count"] is True
    assert result.initial_targets
    assert result.final_targets
    assert result.initial_params
    assert result.final_params


def test_fit_rebind_returns_fitted_clones_for_seed_devices() -> None:
    """fit.rebind(*seeds) shortcircuits the ``chip.device_map[qb.label]`` ritual."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=10, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=-1.2e-4)], frame="rotating")

    with pytest.warns(UserWarning, match="underdetermined by count"):
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
    """_static_exchange_rate (via the public effective-H seam) matches its pre-refactor pinned value.

    Cross-checked against Chip.effective_subspace_hamiltonian — an independent
    implementation — as a second, non-circular confirmation of the math.
    """
    control = DuffingTransmon(freq=5.08, anharmonicity=-0.31, levels=4, label="control")
    target = DuffingTransmon(freq=4.95, anharmonicity=-0.35, levels=4, label="target")
    bus = Resonator(freq=6.28, levels=6, label="bus")
    c_bus = Capacitive(control, bus, g=0.020, label="c_bus")
    t_bus = Capacitive(target, bus, g=0.017, label="t_bus")
    chip = Chip([control, target, bus], [c_bus, t_bus], frame="rotating")

    got = float(_static_exchange_rate(chip, ("control", "target")))
    assert got == pytest.approx(-0.0002693680015177002, abs=1e-10)

    oracle = chip.effective_subspace_hamiltonian(
        ({control: 1, target: 0, bus: 0}, {control: 0, target: 1, bus: 0})
    )
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
