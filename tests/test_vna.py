"""Continuous-wave input-output response through declared ports."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from quchip import (
    Capacitive,
    Chip,
    CrossKerr,
    DuffingTransmon,
    KerrCavity,
    Port,
    PortNetwork,
    Resonator,
    Sweep,
    VNA,
)


def _network(*ports: Port) -> PortNetwork:
    """Return the explicit identity boundary for the supplied ports."""
    return PortNetwork.from_ports(ports)


def _linear_resonator(*, kappa_in: float, kappa_out: float = 0.0):
    resonator = Resonator(freq=6.0, levels=8, label="r")
    input_port = Port(resonator, rate=kappa_in, label="in")
    ports = [input_port]
    output_port = None
    if kappa_out:
        output_port = Port(resonator, rate=kappa_out, label="out")
        ports.append(output_port)
    return resonator, input_port, output_port, Chip([resonator], port_network=_network(*ports))


def test_coherent_port_input_uses_standard_slh_hamiltonian_sign() -> None:
    """A coherent source composes as i(beta* L - beta L-dagger)."""
    from quchip.engine.input_output import (
        add_port_inputs,
        port_operators,
        resolve_stationary_engine,
    )

    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    engine = resolve_stationary_engine(chip, ((input_port.label, 6.0),))
    beta = 0.02 + 0.01j
    coupling = port_operators(engine, chip.backend)[input_port.label].to_dense()

    driven = add_port_inputs(engine, chip.backend, ((input_port.label, 6.0, beta),))

    expected = 1j * (np.conj(beta) * coupling - beta * coupling.conj().T)
    np.testing.assert_allclose(driven.static_terms[-1].operator.to_dense(), expected)


def test_coherent_port_input_leaves_resolved_slh_input_free() -> None:
    """Binding beta adds solve physics beside the immutable resolved SLH value."""
    from quchip.engine.input_output import add_port_inputs, resolve_stationary_engine

    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    engine = resolve_stationary_engine(chip, ((input_port.label, 6.0),))

    driven = add_port_inputs(
        engine,
        chip.backend,
        ((input_port.label, 6.0, 0.02 + 0.01j),),
    )

    assert driven.slh is engine.slh
    assert driven.slh.H == engine.slh.H
    assert len(driven.static_terms) == len(engine.static_terms) + 1


def test_output_field_uses_standard_slh_plus_sign() -> None:
    """The reported field is b_out = beta I + L at the reference plane."""
    from quchip.analysis.vna import _output_field_matrix
    from quchip.engine.input_output import port_operators, resolve_stationary_engine

    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    engine = resolve_stationary_engine(chip, ((input_port.label, 6.0),))
    coupling = port_operators(engine, chip.backend)[input_port.label]
    beta = 0.02 + 0.01j

    output = _output_field_matrix(coupling, beta, np)

    expected = beta * np.eye(coupling.shape[0], dtype=complex) + coupling.to_dense()
    np.testing.assert_allclose(output, expected)


def test_one_sided_small_signal_reflection_matches_analytic_response() -> None:
    resonator, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    frequencies = np.array([5.98, 6.0, 6.03])

    result = VNA(chip, input=input_port, outputs=[input_port]).sweep(frequencies)

    detuning = 2 * np.pi * (resonator.freq - frequencies)
    expected = 1.0 - 0.04 / (0.02 + 1j * detuning)
    np.testing.assert_allclose(result.s11, expected, atol=2e-8)
    np.testing.assert_allclose(result.s("in", "in"), expected, atol=2e-8)
    np.testing.assert_allclose(result.frequencies, frequencies)


def test_two_sided_transmission_is_unit_magnitude_on_resonance() -> None:
    _, input_port, output_port, chip = _linear_resonator(kappa_in=0.03, kappa_out=0.03)
    assert output_port is not None

    result = VNA(chip, input=input_port, outputs=[input_port, output_port]).sweep([6.0])

    np.testing.assert_allclose(result.s11, [0.0], atol=2e-8)
    np.testing.assert_allclose(result.s21, [-1.0], atol=2e-8)
    np.testing.assert_allclose(np.abs(result.s11) ** 2 + np.abs(result.s21) ** 2, [1.0], atol=2e-8)


def test_vna_uses_network_exposure_labels_and_scattering_background() -> None:
    """VNA queries named network exposures and retains direct scattering."""
    resonator = Resonator(freq=6.0, levels=6, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.04)
    phase = network.phase_shift("phase", phase=np.pi / 2)
    network.cascade(port, phase)
    network.expose("readout", input=port.input, output=phase.output)
    chip = Chip([resonator], port_network=network)

    result = VNA(chip, input="readout", outputs=["readout"]).sweep([5.98, 6.0, 6.02])

    detuning = 2 * np.pi * (resonator.freq - np.asarray([5.98, 6.0, 6.02]))
    expected = 1j * (1.0 - 0.04 / (0.02 + 1j * detuning))
    np.testing.assert_allclose(result.s11, expected, atol=2e-8)


def test_vna_reference_delay_is_reciprocal() -> None:
    """VNA phase accumulates the reciprocal external reference-plane delay."""
    def response(delay: float) -> complex:
        resonator = Resonator(freq=6.0, levels=5, label="r")
        network = PortNetwork(label="line")
        port = network.port("chip_port", target=resonator, rate=0.04)
        network.expose(
            "readout",
            input=port.input,
            output=port.output,
            delay=delay,
        )
        chip = Chip([resonator], port_network=network)
        return complex(VNA(chip, input="readout", outputs=["readout"]).sweep([6.01]).s11[0])

    delay = 0.125
    expected_phase = np.exp(1j * 2.0 * 2.0 * np.pi * 6.01 * delay)
    assert response(delay) == pytest.approx(expected_phase * response(0.0), abs=2e-8)


@pytest.mark.parametrize("internal_rate,external_rate", [(0.04, 0.02), (0.02, 0.02), (0.01, 0.03)])
def test_resonance_distinguishes_undercritical_and_overcoupling(
    internal_rate: float,
    external_rate: float,
) -> None:
    resonator = Resonator(freq=6.0, levels=5, label="r", T1=1.0 / internal_rate)
    port = Port(resonator, rate=external_rate, label="p")
    result = VNA(
        Chip([resonator], port_network=_network(port)),
        input=port,
        outputs=[port],
    ).sweep([6.0])

    expected = (internal_rate - external_rate) / (internal_rate + external_rate)
    np.testing.assert_allclose(result.s11, [expected], atol=2e-8)


def test_vna_probe_has_no_finite_amplitude_mode() -> None:
    """Finite-power spectroscopy belongs to external-plane input simulations."""
    assert "amplitude" not in inspect.signature(VNA.sweep).parameters


def test_vna_rejects_sweep_axes_it_does_not_own() -> None:
    """Ordinary chip parameters cannot be silently ignored as tone variations."""
    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    vna = VNA(chip, input=input_port, outputs=[input_port])

    with pytest.raises(ValueError, match="created by this VNA"):
        vna.sweep([6.0], Sweep([5.9, 6.0], name="r.freq"))


def test_vna_result_does_not_depend_on_chip_default_frame() -> None:
    """VNA resolves the stationary tone frame explicitly."""
    def response(frame):
        resonator = Resonator(freq=6.0, levels=5, label="r")
        port = Port(resonator, rate=0.04, label="p")
        chip = Chip([resonator], port_network=_network(port), frame=frame)
        return VNA(chip, input=port, outputs=[port]).sweep([5.98, 6.0, 6.02]).s11

    np.testing.assert_allclose(response("lab"), response("rotating"), atol=2e-8)


def test_one_carrier_probes_passive_modes_behind_the_port() -> None:
    """The probe frame covers a passive filter-readout network, not only the port target."""
    readout = Resonator(freq=6.0, levels=2, label="readout")
    purcell_filter = Resonator(freq=6.02, levels=2, label="filter")
    feedline = Port(purcell_filter, rate=0.02, label="feedline")
    chip = Chip(
        [purcell_filter, readout],
        couplings=[Capacitive(purcell_filter, readout, g=0.01)],
        port_network=_network(feedline),
    )

    result = VNA(chip, input=feedline, outputs=[feedline]).sweep([5.99, 6.0, 6.02])

    assert result.s11.shape == (3,)
    assert np.all(np.isfinite(result.s11))


def test_fixed_tone_variation_uses_pump_axes_without_new_drive_physics() -> None:
    readout = Resonator(freq=6.0, levels=4, label="readout")
    auxiliary = Resonator(freq=5.0, levels=3, label="aux")
    readout_port = Port(readout, rate=0.03, label="readout_port")
    pump_port = Port(auxiliary, rate=0.04, label="pump_port")
    chip = Chip([readout, auxiliary], port_network=_network(readout_port, pump_port))
    vna = VNA(chip, input=readout_port, outputs=[readout_port])
    pump = vna.pump(pump_port, freq=5.0, amplitude=0.02)

    result = vna.sweep(
        np.array([5.99, 6.0]),
        vna.vary(pump, "freq", np.array([4.98, 5.02, 5.04])),
    )

    assert result.shape == (3, 2)
    assert result.axis_names == ("pump_port.freq", "frequency")


def test_distinct_stationary_tones_on_one_mode_require_time_evolution() -> None:
    resonator = Resonator(freq=6.0, levels=4, label="r")
    probe = Port(resonator, rate=0.02, label="probe")
    pump_port = Port(resonator, rate=0.02, label="pump")
    chip = Chip([resonator], port_network=_network(probe, pump_port))
    vna = VNA(chip, input=probe, outputs=[probe])
    vna.pump(pump_port, freq=5.9, amplitude=0.01)

    with pytest.raises(ValueError, match="QuantumSequence"):
        vna.sweep([6.0])


def test_qutip_and_dynamiqs_vna_response_agree() -> None:
    pytest.importorskip("dynamiqs")

    def response(backend: str):
        resonator = Resonator(freq=6.0, levels=5, label="r")
        input_port = Port(resonator, rate=0.02, label="in")
        output_port = Port(resonator, rate=0.03, label="out")
        chip = Chip(
            [resonator],
            port_network=_network(input_port, output_port),
            backend=backend,
        )
        return VNA(chip, input=input_port, outputs=[output_port]).sweep([5.98, 6.0, 6.02]).s21

    np.testing.assert_allclose(np.asarray(response("dynamiqs")), response("qutip"), atol=2e-8)


def test_dynamiqs_pumped_small_signal_response_is_jittable_and_differentiable() -> None:
    """Pumped Dynamiqs small-signal response is JIT-safe and differentiable."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    resonator = Resonator(freq=6.0, levels=3, label="r")
    port = Port(resonator, rate=0.04, label="p")
    chip = Chip([resonator], port_network=_network(port), backend="dynamiqs")

    def reflection(amplitude):
        vna = VNA(chip, input=port, outputs=[port])
        vna.pump(port, freq=6.0, amplitude=amplitude)
        return jnp.real(vna.sweep([6.0]).s11[0])

    value = jax.jit(reflection)(jnp.asarray(0.01))
    gradient = jax.grad(reflection)(jnp.asarray(0.01))

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_dynamiqs_explicit_port_frequency_is_jittable_and_differentiable(
) -> None:
    """Small-signal VNA preserves an explicit port's traced frequency."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    resonator = Resonator(freq=6.0, levels=3, label="r")
    lowering = np.diag(np.sqrt(np.arange(1, 3)), k=1).astype(complex)
    port = Port(resonator, rate=0.04, operator=lowering, label="p")
    vna = VNA(
        Chip([resonator], port_network=_network(port), backend="dynamiqs"),
        input=port,
        outputs=[port],
    )

    def reflection(frequency):
        return jnp.real(vna.sweep(jnp.asarray([frequency])).s11[0])

    value, gradient = jax.jit(jax.value_and_grad(reflection))(jnp.asarray(6.01))

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_resonantly_driven_two_level_population_matches_optical_bloch_solution() -> None:
    decay_rate = 0.05
    amplitude = 0.03
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    port = Port(qubit, rate=decay_rate, label="drive")
    chip = Chip([qubit], port_network=_network(port))

    vna = VNA(chip, input=port, outputs=[port])
    vna.pump(port, freq=5.0, amplitude=amplitude)
    result = vna.sweep([5.0])
    state = np.asarray(result.steady_states[0].state.full())
    excited_population = np.real(state[1, 1])
    expected = 4 * amplitude**2 / (decay_rate + 8 * amplitude**2)

    np.testing.assert_allclose(excited_population, expected, atol=2e-8)


def test_two_tone_cross_kerr_model_produces_a_pump_frequency_axis() -> None:
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    resonator = Resonator(freq=6.0, levels=4, label="r")
    qubit_port = Port(qubit, rate=0.04, label="qubit_port")
    readout_port = Port(resonator, rate=0.03, label="readout_port")
    chip = Chip(
        [qubit, resonator],
        [CrossKerr(qubit, resonator, chi=-0.03)],
        port_network=_network(qubit_port, readout_port),
    )
    vna = VNA(chip, input=readout_port, outputs=[readout_port])
    pump = vna.pump(qubit_port, freq=5.0, amplitude=0.04)

    result = vna.sweep(
        np.array([5.99, 6.0]),
        vna.vary(pump, "freq", np.array([4.97, 5.0, 5.03])),
    )

    assert result.s11.shape == (3, 2)
    assert result.axis_names == ("qubit_port.freq", "frequency")
    assert not np.allclose(result.s11[0], result.s11[1])


def test_two_tone_probe_frame_propagates_through_passive_filter_network() -> None:
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    resonator = Resonator(freq=6.0, levels=2, label="r")
    purcell_filter = Resonator(freq=6.02, levels=2, label="filter")
    qubit_port = Port(qubit, rate=0.04, label="qubit_port")
    feedline = Port(purcell_filter, rate=0.20, label="feedline")
    chip = Chip(
        [qubit, resonator, purcell_filter],
        [
            CrossKerr(qubit, resonator, chi=-0.003),
            Capacitive(purcell_filter, resonator, g=0.010),
        ],
        port_network=_network(qubit_port, feedline),
    )
    vna = VNA(chip, input=feedline, outputs=[feedline])
    pump = vna.pump(qubit_port, freq=5.0, amplitude=0.02)

    result = vna.sweep(
        np.array([5.99, 6.00]),
        vna.vary(pump, "freq", np.array([4.98, 5.00, 5.02])),
    )

    assert result.s11.shape == (3, 2)
    assert result.axis_names == ("qubit_port.freq", "frequency")
    assert np.all(np.isfinite(result.s11))


def test_nonlinear_stationary_state_matches_long_time_master_equation() -> None:
    import qutip

    amplitude = 0.05
    cavity = KerrCavity(freq=6.0, kerr=0.03, levels=8, label="c")
    port = Port(cavity, rate=0.05, label="p")
    chip = Chip([cavity], port_network=_network(port), backend="qutip")
    vna = VNA(chip, input=port, outputs=[port])
    vna.pump(port, freq=6.0, amplitude=amplitude)
    stationary = vna.sweep([6.0])

    engine = chip.resolve(frame={"c": 6.0})
    backend = chip.backend
    hamiltonian = sum(
        (
            term.coefficient * backend.from_canonical_operator(term.operator)
            for term in engine.static_terms
        ),
        start=0,
    )
    port_term = engine.port_terms[0]
    coupling = np.exp(1j * port_term.phase) * np.sqrt(port_term.rate) * backend.from_canonical_operator(
        port_term.operator
    )
    hamiltonian = hamiltonian + 1j * (amplitude.conjugate() * coupling - amplitude * coupling.dag())
    collapse = [
        np.sqrt(term.rate) * backend.from_canonical_operator(term.operator)
        for term in engine.collapse_terms
    ]
    initial = qutip.ket2dm(qutip.basis(cavity.levels, 0))
    evolved = qutip.mesolve(
        hamiltonian,
        initial,
        [0.0, 1000.0],
        collapse,
        options={"method": "diag"},
    ).states[-1]

    np.testing.assert_allclose(stationary.steady_states[0].state.full(), evolved.full(), atol=2e-7)
