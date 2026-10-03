"""Continuous-wave input-output response through declared ports."""

from __future__ import annotations


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


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("options", [None, {}])
def test_instrument_reflection_convention_with_complex_probe_and_delay(backend, options) -> None:
    """Both response routes report engineering reflection, cable phase, and IQ means."""
    from quchip import IQReceiver

    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    r = Resonator(freq=6., levels=3, internal_quality_factor=30000, label="r")
    network = PortNetwork()
    port = network.port("p", target=r, external_quality_factor=10000)
    cable = network.delay("cable", duration=0.123)
    network.link(port, cable)
    network.expose("vna", at=cable.port(2))
    vna = VNA(Chip([r], port_network=network, backend=backend))
    frequencies = np.array([5.9996, 6., 6.0004])
    expected = np.array([.25-.75j, -.5, .25+.75j]) * np.exp(-4j*np.pi*frequencies*.123)
    beta = 1e-5 * (1+2j)
    measured = vna.measure(frequencies, beta, options=options, noise_frequencies=[-.1, -.01, 0., .01, .1])
    np.testing.assert_allclose(vna.sweep(frequencies, options=options).s11, expected, atol=1e-8)
    np.testing.assert_allclose(measured.ratio("vna"), expected, atol=1e-8)
    np.testing.assert_allclose(vna.finite_power(frequencies, beta).ratio("vna"), expected, atol=1e-8)
    stats = measured.statistics(receiver=IQReceiver(integration_time=1000.))
    np.testing.assert_allclose(stats.mean("vna"), beta*expected, atol=1e-12)


def test_vna_reflection_fit_recovers_resonance_and_quality_factors() -> None:
    """Complex VNA data fits the engineering reflection model without conjugation."""
    from scipy.optimize import least_squares

    r = Resonator(freq=6., levels=3, internal_quality_factor=30000, label="r")
    network = PortNetwork()
    network.port("p", target=r, external_quality_factor=10000)
    frequencies = np.linspace(5.997, 6.003, 61)
    data = VNA(Chip([r], port_network=network)).sweep(frequencies).s11

    def residual(parameters):
        f0, qi, qe = parameters
        ql = 1/(1/qi + 1/qe)
        model = 1 - (2*ql/qe)/(1 + 2j*ql*(frequencies-f0)/f0)
        error = model-data
        return np.concatenate((error.real, error.imag))

    fit = least_squares(residual, [6.0001, 25000., 12000.], x_scale="jac",
                        bounds=([5.99, 100., 100.], [6.01, 1e6, 1e6]), gtol=1e-12)
    assert fit.success
    np.testing.assert_allclose(fit.x, [6., 30000., 10000.], rtol=1e-7)


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

    result = VNA(chip, ports=[input_port]).sweep(frequencies)

    detuning = 2 * np.pi * (frequencies - resonator.freq)
    expected = 1.0 - 0.04 / (0.02 + 1j * detuning)
    np.testing.assert_allclose(result.s11, expected, atol=2e-8)
    np.testing.assert_allclose(result.s("in", "in"), expected, atol=2e-8)
    np.testing.assert_allclose(result.frequencies, frequencies)


def test_vna_returns_the_full_port_matrix_in_one_sweep() -> None:
    """All S_ji between the selected ports come from one solve per frequency."""
    resonator, input_port, output_port, chip = _linear_resonator(kappa_in=0.04, kappa_out=0.02)
    frequencies = np.array([5.98, 6.0, 6.03])

    result = VNA(chip).sweep(frequencies)
    single = VNA(chip, ports=[output_port]).sweep(frequencies)

    assert result.ports == ("in", "out")
    assert result.matrix.shape == (3, 2, 2)
    np.testing.assert_allclose(result.s21, result.matrix[:, 1, 0])
    np.testing.assert_allclose(result.s("in", output_port), result.matrix[:, 0, 1])
    np.testing.assert_allclose(result.s(output_port, output_port), single.s11, atol=1e-12)
    power = np.sum(np.abs(result.matrix) ** 2, axis=1)
    np.testing.assert_allclose(power, np.ones((3, 2)), atol=1e-8)
    np.testing.assert_allclose(np.asarray(result), result.matrix)
    with pytest.raises(AttributeError, match="s21"):
        _ = single.s21
    with pytest.raises(TypeError, match="sequence"):
        VNA(chip, ports="in")


def test_stationary_matrix_matches_mode_space_matrix() -> None:
    """The multi-source Liouvillian path agrees with the passive-linear matrix."""
    _, _, _, chip = _linear_resonator(kappa_in=0.04, kappa_out=0.02)
    frequencies = np.array([5.99, 6.0, 6.01])
    linear = VNA(chip).sweep(frequencies)
    general = VNA(chip).sweep(frequencies, options={"method": "direct"})

    assert {item["solver"] for item in linear.diagnostics} == {"linear_response"}
    assert {item["solver"] for item in general.diagnostics} == {"stationary_resolvent"}
    np.testing.assert_allclose(general.matrix, linear.matrix, atol=2e-8)


def test_hidden_dilation_channels_carry_probe_and_pump_fields() -> None:
    """Coherent sources sum conj(S) L over every channel, hidden vacuum outputs included."""
    resonator = Resonator(freq=6.0, levels=8, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=resonator, rate=0.04)
    loss = network.attenuator("cold_loss", eta=0.64)
    network.link(port, loss)
    network.expose("readout", at=loss.port(2))
    chip = Chip([resonator], port_network=network)
    frequencies = np.array([5.99, 6.0, 6.01])

    linear = VNA(chip).sweep(frequencies)
    general = VNA(chip).sweep(frequencies, options={"method": "direct"})
    np.testing.assert_allclose(general.matrix, linear.matrix, atol=2e-8)

    bare = Resonator(freq=6.0, levels=8, label="r")
    bare_network = PortNetwork(label="bare")
    bare_network.port("coupler", target=bare, rate=0.04)
    bare_chip = Chip([bare], port_network=bare_network)
    attenuated = VNA(chip, ports=["readout"])
    attenuated.pump("readout", freq=6.0, amplitude=0.05)
    direct = VNA(bare_chip, ports=["coupler"])
    direct.pump("coupler", freq=6.0, amplitude=0.8 * 0.05)
    lossy_state = attenuated._stationary_output("readout", None).state.state.full()
    direct_state = direct._stationary_output("coupler", None).state.state.full()
    np.testing.assert_allclose(lossy_state, direct_state, atol=1e-8)


def _lowpass(frequency, *, cutoff, order):
    return 1.0 / (1.0 + 1j * (frequency / cutoff) ** order)


def test_filter_section_is_exact_for_continuous_waves_and_sweepable() -> None:
    """A reciprocal filter multiplies both legs by H(f); its parameters are chip paths."""
    resonator = Resonator(freq=6.0, levels=8, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=resonator, rate=0.04)
    lowpass = network.filter("lowpass", transfer=_lowpass, cutoff=6.5, order=2)
    network.link(port, lowpass)
    network.expose("readout", at=lowpass.port(2))
    chip = Chip([resonator], port_network=network)
    frequencies = np.array([5.98, 6.0, 6.02])

    bare = Resonator(freq=6.0, levels=8, label="r")
    bare_network = PortNetwork(label="bare")
    bare_port = bare_network.port("coupler", target=bare, rate=0.04)
    bare_chip = Chip([bare], port_network=bare_network)

    filtered = VNA(chip).sweep(frequencies)
    reference = VNA(bare_chip, ports=[bare_port]).sweep(frequencies)
    expected = _lowpass(frequencies, cutoff=6.5, order=2).conj() ** 2 * reference.s11
    np.testing.assert_allclose(filtered.s11, expected, atol=1e-10)
    np.testing.assert_allclose(
        VNA(chip).sweep(frequencies, options={"method": "direct"}).s11, expected, atol=2e-8
    )

    rebound = VNA(chip).sweep(frequencies, Sweep([6.5, 8.0], name="network.component.lowpass.cutoff"))
    np.testing.assert_allclose(rebound.s11[0], expected, atol=1e-10)
    np.testing.assert_allclose(
        rebound.s11[1], _lowpass(frequencies, cutoff=8.0, order=2).conj() ** 2 * reference.s11, atol=1e-10
    )
    with pytest.raises(TypeError, match="lowpass"):
        chip.to_dict()


def test_filter_section_must_be_passive() -> None:
    """A concretely evaluated |H| above one is gain and is rejected."""
    resonator = Resonator(freq=6.0, levels=4, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=resonator, rate=0.04)
    amplifier = network.filter("gain", transfer=lambda frequency, *, gain: gain, gain=2.0)
    network.link(port, amplifier)
    network.expose("readout", at=amplifier.port(2))

    with pytest.raises(ValueError, match="passive"):
        VNA(Chip([resonator], port_network=network)).sweep([6.0])


def test_output_spectrum_is_filtered_at_the_offset_frequency() -> None:
    """An outbound filter scales the fluctuation spectrum by |H(f_frame + nu)|^2."""

    def build(with_filter: bool):
        resonator = DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="q")
        network = PortNetwork(label="line")
        port = network.port("coupler", target=resonator, rate=0.04)
        if with_filter:
            section = network.filter("lowpass", transfer=_lowpass, cutoff=6.02, order=1)
            network.link(port, section)
            network.expose("readout", at=section.port(2))
        else:
            network.expose("readout", at=port)
        vna = VNA(Chip([resonator], port_network=network))
        inbound = 1.0 if with_filter else abs(_lowpass(6.0, cutoff=6.02, order=1))
        vna.pump("readout", freq=6.0, amplitude=0.02 * inbound)
        return vna

    offsets = np.array([-0.02, 0.0, 0.03])
    filtered = build(True).output_spectrum("readout", frequencies=offsets)
    plain = build(False).output_spectrum("readout", frequencies=offsets)
    gain = np.abs(_lowpass(6.0 + offsets, cutoff=6.02, order=1)) ** 2
    np.testing.assert_allclose(filtered.total_fluctuation_spectrum, gain * plain.total_fluctuation_spectrum, rtol=1e-8)


def test_output_notch_filter_keeps_the_sideband_spectrum() -> None:
    """A filter that nulls the carrier still passes the sidebands with |H(f_c + nu)|^2."""

    def notch(frequency, *, center, width):
        detuning = (frequency - center) / width
        return 1j * detuning / (1.0 + 1j * detuning)

    def build(with_filter: bool):
        qubit = DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="q")
        network = PortNetwork(label="fridge")
        port = network.port("coupler", target=qubit, rate=0.04)
        circulator = network.circulator("circ")
        network.link(port, circulator.port(2))
        network.expose("drive", at=circulator.port(1))
        if with_filter:
            section = network.filter("notch", transfer=notch, center=6.0, width=0.01)
            network.link(circulator.port(3), section)
            network.expose("readout", at=section.port(2))
        else:
            network.expose("readout", at=circulator.port(3))
        vna = VNA(Chip([qubit], port_network=network))
        vna.pump("drive", freq=6.0, amplitude=0.02)
        return vna

    offsets = np.array([-0.02, 0.0, 0.03])
    filtered = build(True).output_spectrum("readout", frequencies=offsets)
    plain = build(False).output_spectrum("readout", frequencies=offsets)
    gain = np.abs(notch(6.0 + offsets, center=6.0, width=0.01)) ** 2
    np.testing.assert_allclose(filtered.total_fluctuation_spectrum, gain * plain.total_fluctuation_spectrum, atol=1e-12)
    assert filtered.total_fluctuation_spectrum[1] == pytest.approx(0.0, abs=1e-12)
    assert filtered.total_fluctuation_spectrum[2] > 0.0


def test_coupling_free_output_takes_its_carrier_from_the_feeding_tone() -> None:
    """A circulator's drive-side output has no coupling; its filter follows the unique feeding tone."""

    def tilt(frequency, *, slope):
        return 0.5 * np.exp(1j * slope * frequency)

    def build(pumped: bool, *, delay_only: bool = False):
        qubit = DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="q")
        network = PortNetwork(label="fridge")
        port = network.port("coupler", target=qubit, rate=0.04)
        circulator = network.circulator("circ")
        line = (
            network.delay("line", duration=0.3)
            if delay_only
            else network.filter("line", transfer=tilt, slope=0.7)
        )
        network.link(line, circulator.port(1))
        network.link(port, circulator.port(2))
        network.expose("drive", at=line.port(1))
        network.expose("readout", at=circulator.port(3))
        vna = VNA(Chip([qubit], port_network=network))
        if pumped:
            vna.pump("readout", freq=6.0, amplitude=0.03)
        return vna

    spectrum = build(True).output_spectrum("drive", frequencies=np.array([0.0]))
    assert spectrum.signal_coherent_flux == pytest.approx(abs(tilt(6.0, slope=0.7)) ** 2 * 0.03**2)
    with pytest.raises(ValueError, match="carrier"):
        build(False).output_spectrum("drive", frequencies=np.array([0.0]))
    delayed = build(True, delay_only=True).output_spectrum("drive", frequencies=np.array([0.0]))
    assert delayed.signal_coherent_flux == pytest.approx(0.03**2)
    with pytest.raises(ValueError, match="carrier"):
        build(False, delay_only=True).output_spectrum("drive", frequencies=np.array([0.0]))
    quiet = build(False).output_spectrum("readout", frequencies=np.array([0.0]))
    assert quiet.signal_coherent_flux == pytest.approx(0.0)


@pytest.mark.validation
def test_distinct_tones_stay_valid_with_traced_network_scattering() -> None:
    """Reachability comes from the compiled structure, not from traced S values."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("dynamiqs")
    import jax.numpy as jnp

    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    resonator = Resonator(freq=6.0, levels=4, label="r")
    network = PortNetwork(label="fridge")
    qubit_port = network.port("qubit_port", target=qubit, rate=0.04)
    readout_port = network.port("readout_port", target=resonator, rate=0.03)
    loss = network.attenuator("loss", eta=0.5)
    network.link(readout_port, loss)
    network.expose("readout", at=loss.port(2))
    chip = Chip(
        [qubit, resonator],
        [CrossKerr(qubit, resonator, chi=-0.03)],
        port_network=network,
        backend="dynamiqs",
    )

    def response(eta):
        vna = VNA(chip.with_params({"network.component.loss.eta": eta}), ports=["readout"])
        vna.pump(qubit_port, freq=5.0, amplitude=0.02)
        return jnp.abs(vna.sweep([6.0]).s11[0])

    assert jnp.isfinite(jax.jit(response)(jnp.asarray(0.5)))


@pytest.mark.validation
def test_cross_kerr_vna_keeps_zero_band_with_traced_local_frequency():
    import jax
    import jax.numpy as jnp

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    r = Resonator(freq=6.0, levels=4, label="r")
    qport = Port(q, rate=0.04, label="pump")
    rport = Port(r, rate=0.03, label="readout")
    chip = Chip([q, r], [CrossKerr(q, r, chi=-0.03)],
                port_network=_network(qport, rport), backend="dynamiqs")

    def response(frequency):
        vna = VNA(chip.with_params({"q.freq": frequency}), ports=[rport])
        vna.pump(qport, freq=5.0, amplitude=0.02)
        return jnp.abs(vna.sweep([6.0]).s11[0])

    step = 1e-5
    expected = response(5.01)
    finite_difference = (response(5.01 + step) - response(5.01 - step)) / (2 * step)
    value, derivative = jax.jit(jax.value_and_grad(response))(jnp.asarray(5.01))
    np.testing.assert_allclose(value, expected, atol=1e-10)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-5, atol=1e-7)


def _amplified(
    gain: float | None = None,
    added_noise: float = 0.0,
    *,
    second: tuple[float, float] | None = None,
    reverse: bool = False,
):
    qubit = DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="q")
    network = PortNetwork(label="fridge")
    port = network.port("coupler", target=qubit, rate=0.04)
    circulator = network.circulator("circ")
    network.link(port, circulator.port(2))
    network.expose("drive", at=circulator.port(1))
    tail = circulator.port(3)
    if gain is not None:
        amplifier = network.amplifier("hemt", gain=gain, added_noise=added_noise)
        if reverse:
            network.link(tail, amplifier.port(2))
            tail = amplifier.port(1)
        else:
            network.link(tail, amplifier)
            tail = amplifier.port(2)
    if second is not None:
        booster = network.amplifier("booster", gain=second[0], added_noise=second[1])
        network.link(tail, booster)
        tail = booster.port(2)
    network.expose("readout", at=tail)
    return Chip([qubit], port_network=network)


def test_amplifier_adds_gain_to_scattering_and_noise_to_spectra() -> None:
    """A forward amplifier multiplies the mean field by sqrt(G) and adds output-referred noise."""
    gain, added = 100.0, 1.0
    frequencies = np.array([5.99, 6.0, 6.02])
    amplified = VNA(_amplified(gain, added)).sweep(frequencies)
    plain = VNA(_amplified()).sweep(frequencies)
    np.testing.assert_allclose(amplified.s("readout", "drive"), np.sqrt(gain) * plain.s("readout", "drive"), atol=1e-9)
    np.testing.assert_allclose(amplified.s("drive", "drive"), plain.s("drive", "drive"), atol=1e-9)
    stationary = VNA(_amplified(gain, added)).sweep(frequencies, options={"method": "direct"})
    np.testing.assert_allclose(stationary.matrix, amplified.matrix, atol=2e-8)

    offsets = np.array([-0.02, 0.0, 0.03])
    vna = VNA(_amplified(gain, added))
    vna.pump("drive", freq=6.0, amplitude=0.02)
    reference = VNA(_amplified())
    reference.pump("drive", freq=6.0, amplitude=0.02)
    loud = vna.output_spectrum("readout", frequencies=offsets)
    quiet = reference.output_spectrum("readout", frequencies=offsets)
    noise = gain * added + (gain - 1.0) / 2.0
    np.testing.assert_allclose(loud.added_noise_spectrum, noise)
    np.testing.assert_allclose(
        loud.total_fluctuation_spectrum, gain * quiet.total_fluctuation_spectrum + noise, rtol=1e-8
    )
    np.testing.assert_allclose(loud.signal_coherent_flux, gain * quiet.signal_coherent_flux, rtol=1e-8)
    with pytest.raises(NotImplementedError, match="amplifier"):
        vna.g1("readout", delays=np.array([0.0, 1.0]))


def test_amplifier_noise_composes_by_friis_and_respects_the_quantum_limit() -> None:
    """Two amplifiers in series follow Friis; the added-noise floor is (1 - 1/G)/2."""
    g1, n1, g2, n2 = 10.0, 1.5, 100.0, 4.0
    vna = VNA(_amplified(g1, n1, second=(g2, n2)))
    vna.pump("drive", freq=6.0, amplitude=0.02)
    spectrum = vna.output_spectrum("readout", frequencies=np.array([0.0]))
    total_gain = g1 * g2
    expected = total_gain * (n1 + n2 / g1) + (total_gain - 1.0) / 2.0
    np.testing.assert_allclose(spectrum.added_noise_spectrum, expected)

    floor = (1.0 - 1.0 / g1) / 2.0
    limited = VNA(_amplified(g1, floor))
    limited.pump("drive", freq=6.0, amplitude=0.02)
    np.testing.assert_allclose(
        limited.output_spectrum("readout", frequencies=np.array([0.0])).added_noise_spectrum, g1 - 1.0
    )
    with pytest.raises(ValueError, match="quantum"):
        _amplified(g1, 0.9 * floor)
    with pytest.raises(ValueError, match="gain"):
        _amplified(0.5, 1.0)
    with pytest.raises(ValueError, match="gain"):
        _amplified(g1, n1).with_params({"network.component.hemt.gain": 0.5}).resolve()
    with pytest.raises(ValueError, match="quantum"):
        _amplified(g1, n1).with_params({"network.component.hemt.added_noise": 0.0}).resolve()


def test_traced_amplifier_gain_flows_through_the_linear_response() -> None:
    """A traced gain skips concrete validation and stays differentiable in mode space."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("dynamiqs")
    import jax.numpy as jnp

    resonator = Resonator(freq=6.0, levels=4, label="r")
    network = PortNetwork(label="fridge")
    port = network.port("coupler", target=resonator, rate=0.04)
    circulator = network.circulator("circ")
    amplifier = network.amplifier("hemt", gain=10.0, added_noise=1.0)
    network.link(port, circulator.port(2))
    network.link(circulator.port(3), amplifier)
    network.expose("drive", at=circulator.port(1))
    network.expose("readout", at=amplifier.port(2))
    chip = Chip([resonator], port_network=network, backend="dynamiqs")

    def transmission(gain):
        rebound = chip.with_params({"network.component.hemt.gain": gain})
        return jnp.abs(VNA(rebound).sweep([6.02]).s("readout", "drive")[0])

    value, gradient = jax.jit(jax.value_and_grad(transmission))(jnp.asarray(10.0))
    assert jnp.isfinite(value) and jnp.isfinite(gradient)
    np.testing.assert_allclose(gradient, value / 20.0, rtol=1e-6)


def test_amplifier_orientation_is_structural() -> None:
    """An amplifier traversed backwards or sitting on an inbound run is rejected."""
    with pytest.raises(ValueError, match="output"):
        _amplified(10.0, 1.0, reverse=True).resolve()

    resonator = Resonator(freq=6.0, levels=3, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=resonator, rate=0.04)
    amplifier = network.amplifier("hemt", gain=10.0, added_noise=1.0)
    network.link(port, amplifier.port(2))
    network.expose("drive", at=amplifier.port(1))
    with pytest.raises(ValueError, match="output"):
        Chip([resonator], port_network=network).resolve()


def test_finite_power_converges_to_small_signal_and_then_saturates() -> None:
    """The mean-field ratio tends to S as beta -> 0 and departs from it at strong drive."""
    qubit = DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="q")
    port = Port(qubit, rate=0.04, label="p")
    chip = Chip([qubit], port_network=_network(port))
    vna = VNA(chip, ports=[port])
    amplitudes = np.array([0.0, 1e-5, 0.05, 0.3])
    frequencies = np.array([5.99, 6.0, 6.01])

    power = vna.finite_power(frequencies, amplitudes)
    small = vna.sweep(frequencies)

    assert power.shape == (4, 3)
    assert power.axis_names == ("amplitude", "frequency")
    assert power.ports == ("p",) and power.input == "p"
    assert np.all(np.isnan(power.ratio(port)[0]))
    np.testing.assert_allclose(power.mean(port)[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(power.ratio(port)[1], small.s11, rtol=1e-6, atol=1e-8)
    assert np.all(np.abs(power.ratio(port)[3] - small.s11) > 1e-2)
    np.testing.assert_allclose(power.mean(port)[1:], power.ratio(port)[1:] * power.incident[1:])

    single = vna.finite_power(6.0, 0.05)
    assert single.shape == ()
    assert complex(single.mean(port)) == pytest.approx(complex(power.mean(port)[2, 1]))


def test_finite_power_matches_small_signal_for_a_linear_resonator() -> None:
    """A harmonic mode responds linearly, so mean/beta equals S at every amplitude."""
    _, input_port, output_port, chip = _linear_resonator(kappa_in=0.04, kappa_out=0.02)
    vna = VNA(chip)
    frequencies = np.array([5.98, 6.0, 6.03])

    power = vna.finite_power(frequencies, np.array([0.01, 0.03]), input=input_port)
    small = vna.sweep(frequencies)

    for row in range(2):
        np.testing.assert_allclose(power.ratio(output_port)[row], small.s21, atol=1e-8)
        np.testing.assert_allclose(power.ratio(input_port)[row], small.s11, atol=1e-8)


def test_finite_power_rejects_ambiguous_probes() -> None:
    """The probe plane must be explicit, unpumped, and free of axis-name collisions."""
    _, input_port, output_port, chip = _linear_resonator(kappa_in=0.04, kappa_out=0.02)
    vna = VNA(chip)
    with pytest.raises(ValueError, match="input"):
        vna.finite_power(6.0, 0.01)
    pumped = VNA(chip, ports=[input_port])
    pumped.pump(input_port, freq=6.0, amplitude=0.01)
    with pytest.raises(ValueError, match="pump"):
        pumped.finite_power(6.0, 0.01)
    other = VNA(chip)
    pump = other.pump(output_port, freq=6.0, amplitude=0.01)
    with pytest.raises(ValueError, match="unique"):
        other.finite_power(6.0, [0.01, 0.02], pump.vary("amplitude", [0.01, 0.02], name="amplitude"), input=input_port)


@pytest.mark.validation
def test_dynamiqs_finite_power_is_differentiable_in_amplitude() -> None:
    """The finite-power mean field stays differentiable through the stationary solve."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("dynamiqs")
    import jax.numpy as jnp

    qubit = DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="q")
    port = Port(qubit, rate=0.04, label="p")
    chip = Chip([qubit], port_network=_network(port), backend="dynamiqs")

    def reflection(amplitude):
        return jnp.abs(VNA(chip, ports=[port]).finite_power(6.0, amplitude).ratio(port))

    value, gradient = jax.jit(jax.value_and_grad(reflection))(jnp.asarray(0.05))
    assert jnp.isfinite(value) and jnp.isfinite(gradient)
    assert gradient != 0.0


def test_vna_uses_network_exposure_labels_and_scattering_background() -> None:
    """VNA queries named network exposures and retains direct scattering."""
    resonator = Resonator(freq=6.0, levels=6, label="r")
    network = PortNetwork(label="line")
    port = network.port("chip_port", target=resonator, rate=0.04)
    phase = network.phase_shift("phase", phase=np.pi / 2)
    network.cascade(port, phase)
    network.expose("readout", input=port.input, output=phase.output)
    chip = Chip([resonator], port_network=network)

    result = VNA(chip, ports=["readout"]).sweep([5.98, 6.0, 6.02])

    detuning = 2 * np.pi * (resonator.freq - np.asarray([5.98, 6.0, 6.02]))
    expected = -1j * (1.0 - 0.04 / (0.02 - 1j * detuning))
    np.testing.assert_allclose(result.s11, expected, atol=2e-8)


def test_vna_reference_delay_is_reciprocal() -> None:
    """VNA phase accumulates the reciprocal external reference-plane delay."""
    def response(delay: float) -> complex:
        resonator = Resonator(freq=6.0, levels=5, label="r")
        network = PortNetwork(label="line")
        port = network.port("chip_port", target=resonator, rate=0.04)
        cable = network.delay("cable", duration=delay)
        network.link(port, cable)
        network.expose("readout", at=cable.port(2))
        chip = Chip([resonator], port_network=network)
        return complex(VNA(chip, ports=["readout"]).sweep([6.01]).s11[0])

    delay = 0.125
    expected_phase = np.exp(-1j * 2.0 * 2.0 * np.pi * 6.01 * delay)
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
        ports=[port],
    ).sweep([6.0])

    expected = (internal_rate - external_rate) / (internal_rate + external_rate)
    np.testing.assert_allclose(result.s11, [expected], atol=2e-8)


def test_chip_parameter_sweeps_are_vna_axes() -> None:
    """Ordinary chip parameter sweeps rebind the chip per grid point beside tone axes."""
    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    frequencies = np.array([5.98, 6.0, 6.02])
    vna = VNA(chip, ports=[input_port])

    result = vna.sweep(frequencies, Sweep([5.98, 6.02], name="r.freq"))

    assert result.shape == (2, 3)
    assert result.axis_names == ("r.freq", "frequency")
    for row, freq in zip(result.s11, [5.98, 6.02], strict=True):
        shifted = VNA(chip.with_params({"r.freq": freq}), ports=[input_port]).sweep(frequencies)
        np.testing.assert_allclose(row, shifted.s11, atol=1e-12)


def test_chip_paths_resembling_tone_keys_still_rebind() -> None:
    """Chip bindings are classified by exact tone keys, not by a label prefix."""
    resonator = Resonator(freq=6.0, levels=6, label="__vna_tone_0")
    port = Port(resonator, rate=0.04, label="p")
    chip = Chip([resonator], port_network=_network(port))

    result = VNA(chip).sweep(np.array([5.9, 6.1]), Sweep([5.9, 6.1], name="__vna_tone_0.freq"))

    np.testing.assert_allclose(np.abs(result.s11[0, 0]), np.abs(result.s11[1, 1]), atol=1e-8)
    assert not np.allclose(result.s11[0, 0], result.s11[0, 1])


def test_vna_rejects_unknown_paths_and_foreign_tone_axes() -> None:
    """Unknown parameter paths and another VNA's tone axes fail loudly."""
    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    vna = VNA(chip, ports=[input_port])

    with pytest.raises(KeyError, match="r.frequency"):
        vna.sweep([6.0], Sweep([5.9], name="r.frequency"))
    other = VNA(chip, ports=[input_port]).pump(input_port, freq=5.0, amplitude=0.01)
    with pytest.raises(ValueError, match="this VNA"):
        vna.sweep([6.0], other.vary("freq", [4.9]))


def test_chip_and_pump_axes_zip_together() -> None:
    """A zipped chip parameter and pump axis step through the grid element by element."""
    readout = Resonator(freq=6.0, levels=4, label="readout")
    auxiliary = Resonator(freq=5.0, levels=3, label="aux")
    readout_port = Port(readout, rate=0.03, label="readout_port")
    pump_port = Port(auxiliary, rate=0.04, label="pump_port")
    chip = Chip([readout, auxiliary], port_network=_network(readout_port, pump_port))
    vna = VNA(chip, ports=[readout_port])
    pump = vna.pump(pump_port, freq=5.0, amplitude=0.02)

    result = vna.sweep(
        np.array([5.99, 6.0]),
        vna.zip(pump.vary("amplitude", [0.01, 0.02]), Sweep([0.03, 0.05], name="port.readout_port.rate")),
    )

    assert result.shape == (2, 2)
    assert result.axis_names == ("pump_port.amplitude/port.readout_port.rate", "frequency")
    assert np.all(np.isfinite(result.s11))


def test_vna_rejects_duplicate_public_axis_names() -> None:
    """Custom axis names cannot collide with each other or with the frequency axis."""
    _, input_port, _, chip = _linear_resonator(kappa_in=0.04)
    vna = VNA(chip, ports=[input_port])
    pump = vna.pump(input_port, freq=5.0, amplitude=0.01)

    with pytest.raises(ValueError, match="unique"):
        vna.sweep(
            [6.0, 6.01],
            vna.zip(pump.vary("freq", [4.9, 5.1], name="power"), pump.vary("amplitude", [0.01, 0.02], name="power")),
        )
    with pytest.raises(ValueError, match="unique"):
        vna.sweep([6.0, 6.01], pump.vary("amplitude", [0.01, 0.02], name="frequency"))
    with pytest.raises(ValueError, match="unique"):
        vna.sweep(
            6.0,
            vna.zip(pump.vary("freq", [4.9, 5.1], name="gain"), pump.vary("amplitude", [0.01, 0.02], name="phase")),
            pump.vary("amplitude", [0.03, 0.04], name="gain/phase"),
        )


def test_vna_result_does_not_depend_on_chip_default_frame() -> None:
    """VNA resolves the stationary tone frame explicitly."""
    def response(frame):
        resonator = Resonator(freq=6.0, levels=5, label="r")
        port = Port(resonator, rate=0.04, label="p")
        chip = Chip([resonator], port_network=_network(port), frame=frame)
        return VNA(chip, ports=[port]).sweep([5.98, 6.0, 6.02]).s11

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

    result = VNA(chip, ports=[feedline]).sweep([5.99, 6.0, 6.02])

    assert result.s11.shape == (3,)
    assert np.all(np.isfinite(result.s11))


def test_fixed_tone_variation_uses_pump_axes_without_new_drive_physics() -> None:
    readout = Resonator(freq=6.0, levels=4, label="readout")
    auxiliary = Resonator(freq=5.0, levels=3, label="aux")
    readout_port = Port(readout, rate=0.03, label="readout_port")
    pump_port = Port(auxiliary, rate=0.04, label="pump_port")
    chip = Chip([readout, auxiliary], port_network=_network(readout_port, pump_port))
    vna = VNA(chip, ports=[readout_port])
    pump = vna.pump(pump_port, freq=5.0, amplitude=0.02)

    result = vna.sweep(
        np.array([5.99, 6.0]),
        pump.vary("freq", np.array([4.98, 5.02, 5.04]), name="pump_freq"),
    )

    assert result.shape == (3, 2)
    assert result.axis_names == ("pump_freq", "frequency")


def test_distinct_stationary_tones_on_one_mode_require_time_evolution() -> None:
    resonator = Resonator(freq=6.0, levels=4, label="r")
    probe = Port(resonator, rate=0.02, label="probe")
    pump_port = Port(resonator, rate=0.02, label="pump")
    chip = Chip([resonator], port_network=_network(probe, pump_port))
    vna = VNA(chip, ports=[probe])
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
        return VNA(chip, ports=[input_port, output_port]).sweep([5.98, 6.0, 6.02]).s21

    np.testing.assert_allclose(np.asarray(response("dynamiqs")), response("qutip"), atol=2e-8)


def test_eight_resonator_cascade_matches_exact_series_product() -> None:
    """An eight-mode feedline stays in mode space and reproduces SLH series response."""
    resonances = np.asarray([6.42, 6.505, 6.595, 6.69, 6.79, 6.895, 7.005, 7.12])
    external_rates = np.asarray([0.0060, 0.0082, 0.0052, 0.0105, 0.0071, 0.0120, 0.0090, 0.0066])
    qualities = np.asarray([28_000, 42_000, 22_000, 55_000, 31_000, 47_000, 26_000, 60_000])
    phases = np.asarray([0.18, 0.27, 0.16, 0.31, 0.22, 0.28, 0.19])
    resonators = [
        Resonator(
            freq=frequency,
            levels=32,
            internal_quality_factor=quality,
            label=f"r{index}",
        )
        for index, (frequency, quality) in enumerate(zip(resonances, qualities, strict=True))
    ]
    network = PortNetwork(label="feedline")
    ports = [
        network.port(f"p{index}", target=resonator, rate=rate)
        for index, (resonator, rate) in enumerate(zip(resonators, external_rates, strict=True))
    ]
    previous = ports[0]
    for index, (phase, following) in enumerate(zip(phases, ports[1:], strict=True)):
        section = network.phase_shift(f"section{index}", phase=phase)
        network.cascade(previous, section, following)
        previous = following
    input_line = network.delay("input_line", duration=0.08)
    output_line = network.delay("output_line", duration=0.08)
    network.connect(input_line.output_terminal("2"), ports[0].input)
    network.connect(ports[-1].output, output_line.input_terminal("1"))
    network.expose(
        "readout",
        input=input_line.input_terminal("1"),
        output=output_line.output_terminal("2"),
    )
    frequencies = np.asarray([6.42, 6.69, 7.12])

    result = VNA(
        Chip(resonators, port_network=network),
        ports=["readout"],
    ).sweep(frequencies)

    expected = np.exp(-1j * np.sum(phases)) * np.exp(-1j * 4 * np.pi * frequencies * 0.08)
    for resonance, external_rate, quality in zip(resonances, external_rates, qualities, strict=True):
        internal_rate = 2 * np.pi * resonance / quality
        expected *= 1.0 - external_rate / (
            0.5 * (external_rate + internal_rate) + 1j * 2 * np.pi * (frequencies - resonance)
        )
    np.testing.assert_allclose(result.s11, expected, atol=2e-11)
    assert all(diagnostic["mode_count"] == 8 for diagnostic in result.diagnostics)


def test_coupled_mode_response_matches_general_liouvillian_fallback() -> None:
    """Mode-space, one-excitation and general-solver scattering agree for a coupled linear chip."""
    frequencies = np.asarray([5.97, 6.0, 6.04])

    def response(*, explicit_operator: bool, options: dict | None = None):
        first = Resonator(freq=6.0, levels=3, label="a")
        second = Resonator(freq=6.035, levels=3, label="b")
        operator = (
            np.diag(np.sqrt(np.arange(1, first.levels)), k=1).astype(complex)
            if explicit_operator
            else None
        )
        port = Port(first, rate=0.035, operator=operator, label="readout")
        chip = Chip(
            [first, second],
            [Capacitive(first, second, g=0.012)],
            port_network=_network(port),
        )
        return VNA(chip, ports=[port]).sweep(frequencies, options=options)

    linear = response(explicit_operator=False)
    projected = response(explicit_operator=True)
    general = response(explicit_operator=True, options={"method": "direct"})

    np.testing.assert_allclose(linear.s11, general.s11, atol=2e-10)
    np.testing.assert_allclose(projected.s11, general.s11, atol=2e-10)
    assert {item["solver"] for item in linear.diagnostics} == {"linear_response"}
    assert {item["solver"] for item in projected.diagnostics} == {"vacuum_response"}
    assert {item["solver"] for item in general.diagnostics} == {"stationary_resolvent"}


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_harmonic_level_shift_retains_compact_response(backend) -> None:
    """An energy-level shift of an ordered harmonic mode remains quadratic."""
    from quchip.declarative import CouplingModel

    class LevelShift(CouplingModel):
        def interaction(self, a, b, p):
            return 0.02 * a.level * b.I

    first = Resonator(freq=6.0, levels=3, label="a")
    second = Resonator(freq=7.0, levels=3, label="b")
    ports = [Port(first, rate=0.04, label="readout"), Port(second, rate=0.03, label="loss")]
    chip = Chip([first, second], [LevelShift(first, second)], port_network=_network(*ports), backend=backend)
    frequencies = np.asarray([6.01, 6.02, 6.03])
    result = VNA(chip, ports=["readout"]).sweep(frequencies)
    expected = 1 - 0.04 / (0.02 + 2j * np.pi * (frequencies - 6.02))
    np.testing.assert_allclose(result.s11, expected, atol=2e-10)
    assert {item["solver"] for item in result.diagnostics} == {"linear_response"}


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("custom_space", [False, True])
def test_reordered_saved_level_uses_its_physical_spectrum(backend, custom_space) -> None:
    """A saved descending spectrum must not become a positive number shift."""
    from quchip.declarative import CouplingModel, DeviceModel
    from quchip.declarative.ops import LocalOps
    from quchip.devices.spaces import CustomSpace

    class DescendingMode(DeviceModel):
        def local_hamiltonian(self, op, p):
            return (6.0 if custom_space else -6.0) * op.n

        def local_space(self):
            if custom_space:
                return CustomSpace(3, {"n": np.diag([2, 1, 0]), "I": np.eye(3)})
            return super().local_space()

    source = DescendingMode(levels=3, label="a")
    saved = LocalOps(label="a", space=source.local_space(), device=source).level

    class SavedShift(CouplingModel):
        def interaction(self, a, b, p):
            return 0.02 * saved * b.I

    first = Resonator(freq=6.0, levels=3, label="a")
    second = Resonator(freq=7.0, levels=3, label="b")
    ports = [Port(first, rate=0.04, label="readout"), Port(second, rate=0.03, label="loss")]
    chip = Chip([first, second], [SavedShift(first, second)], port_network=_network(*ports), backend=backend)
    frequencies = np.asarray([5.97, 5.98, 5.99])
    result = VNA(chip, ports=["readout"]).sweep(frequencies)
    expected = 1 - 0.04 / (0.02 + 2j * np.pi * (frequencies - 5.98))
    np.testing.assert_allclose(result.s11, expected, atol=2e-10)
    assert {item["solver"] for item in result.diagnostics} == {"vacuum_response"}


def test_nonlinear_mode_weak_probe_uses_its_one_excitation_block() -> None:
    """A Kerr mode's weak-probe reflection is the linear cavity's, from the one-excitation block."""
    cavity = KerrCavity(freq=6.0, kerr=0.02, levels=3, label="c")
    port = Port(cavity, rate=0.04, label="readout")
    vna = VNA(Chip([cavity], port_network=_network(port)), ports=[port])
    frequencies = np.asarray([5.99, 6.0, 6.01])

    weak = vna.sweep(frequencies)
    general = vna.sweep(frequencies, options={"method": "direct"})

    np.testing.assert_allclose(weak.s11, 1 - 0.04 / (0.02 + 2j * np.pi * (frequencies - 6.0)), atol=1e-12)
    np.testing.assert_allclose(general.s11, weak.s11, atol=2e-8)
    assert {item["solver"] for item in weak.diagnostics} == {"vacuum_response"}
    assert {item["solver"] for item in general.diagnostics} == {"stationary_resolvent"}


def test_dynamiqs_linear_response_is_jittable_and_differentiable() -> None:
    """The mode-space VNA path preserves JIT and gradients through chip parameters."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    resonator = Resonator(freq=6.0, levels=16, label="r")
    port = Port(resonator, rate=0.04, label="readout")
    chip = Chip([resonator], port_network=_network(port), backend="dynamiqs")

    def reflection(resonance):
        shifted = chip.with_params({"r.freq": resonance})
        return jnp.real(VNA(shifted, ports=["readout"]).sweep([6.01]).s11[0])

    value, gradient = jax.jit(jax.value_and_grad(reflection))(jnp.asarray(6.0))

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


@pytest.mark.validation
def test_dynamiqs_pumped_small_signal_response_is_jittable_and_differentiable() -> None:
    """Pumped Dynamiqs small-signal response is JIT-safe and differentiable."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    resonator = Resonator(freq=6.0, levels=3, label="r")
    port = Port(resonator, rate=0.04, label="p")
    chip = Chip([resonator], port_network=_network(port), backend="dynamiqs")

    def reflection(amplitude):
        vna = VNA(chip, ports=[port])
        vna.pump(port, freq=6.0, amplitude=amplitude)
        return jnp.real(vna.sweep([6.0]).s11[0])

    value, gradient = jax.jit(jax.value_and_grad(reflection))(jnp.asarray(0.01))

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
        ports=[port],
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

    vna = VNA(chip, ports=[port])
    vna.pump(port, freq=5.0, amplitude=amplitude)
    state = np.asarray(vna._stationary_output(port.label, None).state.state.full())
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
    vna = VNA(chip, ports=[readout_port])
    pump = vna.pump(qubit_port, freq=5.0, amplitude=0.04)

    result = vna.sweep(
        np.array([5.99, 6.0]),
        pump.vary("freq", np.array([4.97, 5.0, 5.03])),
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
    vna = VNA(chip, ports=[feedline])
    pump = vna.pump(qubit_port, freq=5.0, amplitude=0.02)

    result = vna.sweep(
        np.array([5.99, 6.00]),
        pump.vary("freq", np.array([4.98, 5.00, 5.02])),
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
    vna = VNA(chip, ports=[port])
    vna.pump(port, freq=6.0, amplitude=amplitude)
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

    np.testing.assert_allclose(vna._stationary_output(port.label, None).state.state.full(), evolved.full(), atol=2e-7)


def test_pumped_vna_reports_phase_conjugate_response() -> None:
    """Around a pumped Kerr operating point, S and T reproduce the real and imaginary probe derivatives."""
    from quchip.devices.kerr_cavity import KerrCavity

    cavity = KerrCavity(freq=6.0, kerr=0.02, levels=12, label="k")
    network = PortNetwork(label="combiner")
    port = network.port("p", target=cavity, rate=0.05)
    splitter = network.beam_splitter("bs", eta=0.5)
    network.cascade(splitter.output_terminal("left"), port)
    network.expose("pump", input=splitter.input_terminal("left"), output=splitter.output_terminal("right"))
    network.expose("probe", input=splitter.input_terminal("right"), output=port.output)
    vna = VNA(Chip([cavity], port_network=network), ports=["probe"])
    vna.pump("pump", freq=6.0, amplitude=0.3 + 0.1j)

    small_signal = vna.sweep([6.0])
    s_value = complex(np.asarray(small_signal.s("probe", "probe"))[0])
    t_value = complex(np.asarray(small_signal.t("probe", "probe"))[0])
    assert abs(t_value) > 1e-3

    step = 1e-4
    means = np.asarray(vna.finite_power([6.0], [step, -step, 1j * step, -1j * step], input="probe").mean("probe"))
    real_slope = (means[0, 0] - means[1, 0]) / (2 * step)
    imag_slope = (means[2, 0] - means[3, 0]) / (2j * step)
    np.testing.assert_allclose(real_slope, s_value + t_value, atol=1e-5)
    np.testing.assert_allclose(imag_slope, s_value - t_value, atol=1e-5)

    linear = Resonator(freq=6.0, levels=4, label="r")
    passive = VNA(Chip([linear], port_network=PortNetwork.from_ports([Port(linear, rate=0.05, label="p")])))
    np.testing.assert_allclose(passive.sweep([6.0]).conjugate_matrix, 0.0)


def test_stationary_route_keeps_plane_labels_out_of_source_keys() -> None:
    """Plane labels that look like internal source keys do not collide in the stationary solve."""
    resonator = Resonator(freq=6.0, levels=4, label="r")
    network = PortNetwork(label="odd")
    network.port("x", target=resonator, rate=0.04)
    network.port("conjugate:x", target=resonator, rate=0.02)
    chip = Chip([resonator], port_network=network)
    frequencies = np.linspace(5.98, 6.02, 5)

    passive = VNA(chip).sweep(frequencies)
    stationary = VNA(chip).sweep(frequencies, options={})

    assert {item["solver"] for item in stationary.diagnostics} == {"stationary_resolvent"}
    np.testing.assert_allclose(stationary.matrix, passive.matrix, atol=1e-8)
    np.testing.assert_allclose(stationary.conjugate_matrix, 0.0, atol=1e-10)
