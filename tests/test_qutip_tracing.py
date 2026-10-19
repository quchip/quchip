"""Actionable errors at QuTiP lowering and supported dynamiqs overrides."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import (
    RWA, VNA, Capacitive, ChargeBasisTransmon, ChargeDrive, Chip,
    DuffingTransmon, Gaussian, PhysicsExpr, PortNetwork, QuantumSequence, Resonator,
)
from quchip.engine.linear_response import _scalar_value


def _chip(backend: str) -> Chip:
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=2000.0)
    readout = Resonator(freq=6.0, levels=3, label="r")
    network = PortNetwork(label="m")
    network.port("r", target=readout, rate=0.05)
    chip = Chip(
        [qubit, readout], [Capacitive(qubit, readout, g=0.06, label="qr")],
        port_network=network, approximation=RWA(), backend=backend,
    )
    chip.wire(ChargeDrive(chip["q"], label="xy"))
    return chip


def _population(
    chip: Chip, *, backend: str | None = None, amplitude=0.02, carrier_frequency=4.99,
):
    sequence = QuantumSequence(chip)
    sequence.schedule(
        "xy", envelope=Gaussian(duration=40.0, sigmas=3.0, amplitude=amplitude), freq=carrier_frequency,
    )
    result = sequence.simulate(
        tlist=np.linspace(0.0, 60.0, 121), initial_state={"q": 0}, backend=backend,
    )
    return jnp.real(result.population("q", level=1)[-1])


_QUERIES = {
    "frequency": lambda chip: jnp.asarray(chip.freq("q")),
    "dispersive_shift": lambda chip: jnp.asarray(chip.dispersive_shift("q", "r")),
    "vna": lambda chip: jnp.abs(VNA(chip).sweep(np.array([6.0])).matrix[0, 0, 0]) ** 2,
    "pulse": _population,
}


@pytest.mark.parametrize("query", _QUERIES)
@pytest.mark.parametrize("transform", [jax.grad, jax.jit], ids=["grad", "jit"])
def test_qutip_traced_queries_name_supported_backend(query, transform):
    """Traced device frequencies receive the backend remedy on each public route."""
    chip = _chip("qutip")
    with pytest.raises(ValueError, match="JAX tracing requires backend='dynamiqs'.*'qutip'"):
        transform(lambda frequency: _QUERIES[query](chip.with_params({"q.freq": frequency})))(5.0)


@pytest.mark.parametrize("parameter,value,error", [("qr.g", 0.06, ValueError), ("q.T1", 2000.0, RuntimeError)])
def test_qutip_traced_coupling_and_loss_name_supported_backend(parameter, value, error):
    """The lowering check also covers coupling strengths and collapse rates."""
    chip = _chip("qutip")
    with pytest.raises(error, match="JAX tracing requires backend='dynamiqs'.*'qutip'"):
        jax.grad(lambda bound: _population(chip.with_params({parameter: bound})))(value)


def test_qutip_traced_eigenbasis_names_supported_backend():
    """Traced projected matrices receive the same remedy as native-basis scalars."""
    def frequency(josephson_energy):
        qubit = ChargeBasisTransmon(E_C=0.25, E_J=josephson_energy, num_basis=7, levels=3, label="q")
        return Chip([qubit], backend="qutip").freq("q")

    with pytest.raises(ValueError, match="JAX tracing requires backend='dynamiqs'.*'qutip'"):
        jax.grad(frequency)(12.5)


@pytest.mark.unit
@pytest.mark.parametrize("symbolic", [False, True], ids=["raw", "expression"])
def test_compact_scalar_lowering_validates_raw_and_evaluated_values(backend, symbolic):
    """Every compact scalar receives the same guard before native conversion."""
    def evaluate(value):
        expression = PhysicsExpr.parameter(scope="response", name="value") if symbolic else value
        return _scalar_value(expression, backend, bindings={"response.value": value})

    assert evaluate(0.25) == 0.25
    with pytest.raises(ValueError, match="JAX tracing requires backend='dynamiqs'.*'qutip'"):
        jax.grad(evaluate)(0.25)


@pytest.mark.parametrize("parameter,value", [("rate", 0.05), ("phase", 0.2)])
@pytest.mark.parametrize("transform", [jax.grad, jax.jit], ids=["grad", "jit"])
def test_qutip_traced_port_scalars_name_supported_backend(parameter, value, transform):
    """A harmonic response rejects traced port parameters at scalar lowering."""
    readout = Resonator(freq=6.0, levels=3, label="r")
    network = PortNetwork(label="m")
    network.port("r", target=readout, rate=0.05, phase=0.2)
    chip = Chip([readout], port_network=network, approximation=RWA(), backend="qutip")
    frequencies = np.array([6.01])

    def reflection(bound):
        response = VNA(chip.with_params({f"port.r.{parameter}": bound})).sweep(frequencies)
        return jnp.real(response.matrix[0, 0, 0])

    with pytest.raises(ValueError, match="JAX tracing requires backend='dynamiqs'.*'qutip'"):
        transform(reflection)(value)


@pytest.mark.parametrize("parameter,value", [("amplitude", 0.02), ("carrier_frequency", 4.99)])
@pytest.mark.parametrize("transform,error_type", [(jax.grad, RuntimeError), (jax.jit, ValueError)], ids=["grad", "jit"])
def test_qutip_traced_pulse_coefficients_name_supported_backend(parameter, value, transform, error_type):
    """Traced pulse signals retain the backend remedy and solver exception cause."""
    chip = _chip("qutip")
    with pytest.raises(error_type, match="JAX tracing requires backend='dynamiqs'.*'qutip'") as error:
        transform(lambda bound: _population(chip, **{parameter: bound}))(value)
    # JIT traces initial-state assembly before the solver; grad reaches native coefficients.
    if error_type is RuntimeError:
        assert isinstance(error.value.__cause__, ValueError)
        assert "backend='dynamiqs'" in str(error.value.__cause__)


@pytest.mark.optional_backend
@pytest.mark.parametrize("query", ["frequency", "dispersive_shift", "vna"])
def test_dynamiqs_static_gradients_and_eager_qutip_agree(query):
    """Supported static derivatives match finite differences and eager QuTiP values."""
    pytest.importorskip("dynamiqs")
    chip = _chip("dynamiqs")
    call = _QUERIES[query]

    def evaluate(frequency):
        return call(chip.with_params({"q.freq": frequency}))

    # O(h^2) central differences at h=1e-4 resolve the nonzero dispersive derivative.
    step = 1e-4
    difference = (evaluate(5.0 + step) - evaluate(5.0 - step)) / (2 * step)
    derivative = jax.jit(jax.grad(evaluate))(5.0)
    np.testing.assert_allclose(derivative, difference, rtol=2e-5, atol=1e-9)
    np.testing.assert_allclose(evaluate(5.0), call(_chip("qutip")), rtol=1e-10, atol=1e-12)


@pytest.mark.validation
@pytest.mark.optional_backend
@pytest.mark.parametrize("parameter,value", [("q.freq", 5.0), ("amplitude", 0.02), ("carrier_frequency", 4.99)])
def test_dynamiqs_override_preserves_traced_binding_and_pulse_gradient(parameter, value):
    """A dynamiqs solve override keeps the same population and finite-difference derivative."""
    pytest.importorskip("dynamiqs")
    declared_qutip = _chip("qutip")
    declared_dynamiqs = _chip("dynamiqs")

    def evaluate(chip, bound, *, backend=None):
        if parameter == "q.freq":
            return _population(chip.with_params({parameter: bound}), backend=backend)
        return _population(chip, backend=backend, **{parameter: bound})

    def override(bound):
        return evaluate(declared_qutip, bound, backend="dynamiqs")

    def direct(bound):
        return evaluate(declared_dynamiqs, bound)

    direct_value, direct_gradient = jax.value_and_grad(direct)(value)
    override_value, override_gradient = jax.jit(jax.value_and_grad(override))(value)
    # The identical dynamiqs problem must agree within solver accuracy after JIT.
    np.testing.assert_allclose(override_value, direct_value, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(override_gradient, direct_gradient, rtol=1e-7, atol=1e-7)
    step = 1e-5
    difference = (override(value + step) - override(value - step)) / (2 * step)
    # O(h^2) truncation over a 60 ns pulse dominates the 1e-9 solver tolerance.
    np.testing.assert_allclose(override_gradient, difference, rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(evaluate(declared_qutip, value), direct_value, rtol=1e-5, atol=1e-7)
