"""Compatibility checks for renamed physical quantities and network interfaces."""

import jax
import numpy as np
import pytest

from quchip import Chip, DuffingTransmon, PortNetwork, QuantumSequence, Resonator, parameter


def test_legacy_thermal_input_round_trips_to_one_canonical_parameter():
    """Old constructors and saved models retain the same bath without duplicate parameters."""
    with pytest.warns(DeprecationWarning, match="thermal_occupation"):
        q = DuffingTransmon(freq=5, anharmonicity=-.2, T1=100, thermal_population=.2, label="q")
    data = q.to_dict()
    assert data["thermal_occupation"] == .2
    assert "thermal_population" not in data
    data["thermal_population"] = data.pop("thermal_occupation")
    with pytest.warns(DeprecationWarning, match="thermal_occupation"):
        restored = DuffingTransmon.from_dict(data)
    assert restored.intrinsic_decay_rate() == pytest.approx(.012)
    assert "q.thermal_population" not in Chip([restored]).parameters
    with pytest.raises(TypeError, match="not both"):
        Resonator(freq=6, thermal_population=.1, thermal_occupation=.2)


def test_legacy_thermal_writes_validate_and_rebind_without_mutating_source():
    """Legacy writes share validation, cache tracking and atomic rebinding with the canonical field."""
    q = Resonator(freq=6, T1=100, thermal_occupation=.1, label="r")
    version = q.state_version
    with pytest.warns(DeprecationWarning, match="thermal_occupation"):
        q.thermal_population = .2
    assert q.state_version == version + 1
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="non-negative"):
        q.thermal_population = -.2
    assert q.thermal_occupation == .2
    chip = Chip([q])
    with pytest.warns(DeprecationWarning):
        changed = chip.with_params({"r.thermal_population": .3})
    assert changed["r"].thermal_occupation == .3
    assert q.thermal_occupation == .2
    with pytest.raises(TypeError, match="not both"):
        chip.with_params({"r.thermal_population": .3, "r.thermal_occupation": .4})
    with pytest.warns(DeprecationWarning):
        changed.set_noise({"r": {"T1": 100, "thermal_population": .4}})
    assert changed["r"].intrinsic_decay_rate() == pytest.approx(.014)


def test_thermal_occupation_remains_differentiable_through_parameter_binding():
    """Bath occupation changes the downward rate with the analytical derivative 1/T1."""
    chip = Chip([Resonator(freq=6, T1=100, thermal_occupation=.1, label="r")])
    derivative = jax.jit(jax.grad(
        lambda n: chip.with_params({"r.thermal_occupation": n})["r"].intrinsic_decay_rate()
    ))(.2)
    assert derivative == pytest.approx(.01)


def test_legacy_custom_thermal_declaration_and_symbolic_noise_share_one_field():
    """Existing custom thermal defaults and symbolic expressions use the canonical bath parameter."""
    with pytest.warns(DeprecationWarning):
        class ThermalResonator(Resonator):
            thermal_population: float = parameter(default=.2, nonnegative=True, noise=True, kw_only=True)
            tunable_param_names = ("freq", "thermal_population")

            def dissipation(self, op, p):
                from quchip import CollapseChannel

                return (CollapseChannel(op.a, p.thermal_population/100, "extra_loss"),)

    device = ThermalResonator(freq=6, T1=100, levels=3, label="r")
    assert device.thermal_occupation == .2
    assert "thermal_population" not in device.parameter_values()
    chip = Chip([device])
    with pytest.warns(DeprecationWarning):
        channels = chip.resolve().collapse_terms
    assert channels
    with pytest.warns(DeprecationWarning):
        derivative = jax.jit(jax.grad(
            lambda n: chip.with_params({"r.thermal_population": n})["r"].intrinsic_decay_rate()
        ))(.2)
    assert derivative == pytest.approx(.01)


def test_legacy_network_names_address_the_same_connected_ports():
    """Old connection handles retain identity and access to the same external network port."""
    network = PortNetwork()
    r = Resonator(freq=6)
    coupling = network.port("r", target=r, rate=.01)
    external = network.expose("readout", at=coupling)
    with pytest.warns(DeprecationWarning):
        assert network.exposure("readout") is external
    with pytest.warns(DeprecationWarning):
        assert network.exposures == network.external_ports
    circulator = network.circulator(label="circ")
    with pytest.warns(DeprecationWarning):
        assert circulator.side(1) == circulator.port(1)


def test_legacy_collapse_flux_returns_the_same_physical_jump_rate():
    """The old method retains the exponential relaxation rate from a saved quantum state."""
    q = DuffingTransmon(freq=5, anharmonicity=-.2, levels=2, T1=100, label="q")
    chip = Chip([q], frame="rotating")
    result = QuantumSequence(chip).simulate(
        tlist=[0, 10], initial_state=chip.state(q=1), )
    channel = result.collapse_channels[0]
    expected = np.exp(-np.asarray(result.times)/100)/100
    np.testing.assert_allclose(result.jump_rate(channel), expected, rtol=1e-6)
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(result.collapse_flux(channel), result.jump_rate(channel))
