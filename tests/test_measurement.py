"""Stationary field capture and receiver processing have separate lifetimes."""
import numpy as np
import pytest

from quchip import Chip, IQReceiver, PortNetwork, Resonator, VNA


def line(*, thermal_occupation=None, backend="qutip", mixed=False):
    r = Resonator(freq=6.0, levels=8, label="r")
    net = PortNetwork(label="line")
    port = net.port("coupler", target=r, rate=0.04)
    circ = net.circulator("circ")
    iso = net.isolator("iso", thermal_occupation=thermal_occupation)
    amp = net.amplifier("hemt", gain=100.0, added_noise=1.0)
    net.link(port, circ.port(2))
    if mixed:
        loss = net.attenuator("loss", eta=0.25, thermal_occupation=0.2)
        second = net.amplifier("second", gain=10.0, added_noise=2.0)
        net.link(circ.port(3), amp, iso, loss, second)
        tail = second
    else:
        net.link(circ.port(3), iso, amp)
        tail = amp
    drive = net.expose("drive", at=circ.port(1))
    readout = net.expose("readout", at=tail.port(2))
    return Chip([r], port_network=net, backend=backend), drive, readout


def test_receiver_samples_capture_correct_mean_variance_and_do_not_resolve_again(monkeypatch) -> None:
    """A coherent line produces quantum plus amplifier noise with 1/T variance."""
    chip, drive, readout = line()
    vna = VNA(chip)
    measurement = vna.measure([5.99, 6.0, 6.01], 0.02, input=drive, outputs=[readout])
    np.testing.assert_allclose(measurement.ratio(readout), vna.sweep([5.99, 6.0, 6.01]).s(readout, drive), atol=1e-7)

    def forbidden(*args, **kwargs):
        raise AssertionError("Receiver postprocessing invoked a solver")
    monkeypatch.setattr(chip.backend, "stationary_resolvent", forbidden)
    monkeypatch.setattr(chip.backend, "steadystate", forbidden)
    stats = measurement.statistics(receiver=IQReceiver(integration_time=200_000))
    # G*n_add+(G-1)/2 = 149.5, plus one heterodyne quantum, split into I,Q.
    expected = 150.5 / (2 * 200_000)
    np.testing.assert_allclose(stats.covariance(readout), np.broadcast_to(np.eye(2)*expected, (3, 2, 2)), atol=1e-10)
    longer = measurement.statistics(receiver=IQReceiver(integration_time=2_000_000))
    np.testing.assert_allclose(stats.covariance(readout), 10*longer.covariance(readout), rtol=1e-7, atol=1e-12)
    samples = stats.sample(100_000, seed=7)
    delta = samples.field(readout)[:, 1] - stats.mean(readout)[1]
    np.testing.assert_allclose([np.var(delta.real), np.var(delta.imag)], expected, rtol=0.015)
    np.testing.assert_array_equal(samples.field(readout), stats.sample(100_000, seed=7).field(readout))
    assert samples.ratio(readout).shape == (100_000, 3)
    with pytest.raises(ValueError):
        measurement.values[0, 0] = 2


def test_mixed_amplifier_isolator_loss_chain_composes_noise_in_order() -> None:
    """Downstream loss attenuates upstream amplifier noise and adds thermal noise."""
    chip, drive, readout = line(mixed=True)
    vna = VNA(chip)
    vna.pump(drive, freq=6.0, amplitude=0.01)
    spectrum = vna.output_spectrum(readout, frequencies=[-0.01, 0, 0.01])
    expected = 10 * (0.25 * 149.5 + 0.75 * 0.2) + 24.5
    np.testing.assert_allclose(spectrum.added_noise_spectrum, expected)
    measured = VNA(chip).measure(6.0, 0.01, input=drive, outputs=[readout])
    stats = measured.statistics(receiver=IQReceiver(integration_time=1000))
    np.testing.assert_allclose(stats.covariance(readout), np.eye(2)*(expected+1)/2000, atol=1e-8)
    budget = stats.noise_contributions(readout)
    assert any("loss" in name for name in budget)
    np.testing.assert_allclose(sum(budget.values()), stats.covariance(readout))


def test_circulated_finite_power_can_report_the_zero_source_output() -> None:
    """A dark reflected source output is well-defined at the probe carrier."""
    chip, drive, readout = line()
    result = VNA(chip).finite_power([6.0], 0.01, input=drive)
    np.testing.assert_allclose(result.mean(drive), 0, atol=1e-12)
    assert abs(result.mean(readout)[0]) > 0


def test_calibration_transforms_both_covariance_axes() -> None:
    """Complex output calibration rotates and scales the captured distribution."""
    chip, drive, readout = line()
    measured = VNA(chip).measure(6.0, 0.01, input=drive, outputs=[readout])
    stats = measured.statistics(receiver=IQReceiver(integration_time=1000))
    calibrated = stats.calibrate({readout: 0.1j})
    np.testing.assert_allclose(calibrated.mean(readout), 0.1j*stats.mean(readout))
    np.testing.assert_allclose(calibrated.covariance(readout), 0.01*stats.covariance(readout), atol=1e-12)


def test_receiver_requires_resolved_spectral_grid() -> None:
    """Invalid spectral support and receiver settings fail explicitly."""
    chip, drive, readout = line()
    for offsets in ([0, 1], [-1, -0.1, 0, 0.2, 1]):
        with pytest.raises(ValueError, match="noise_frequencies"):
            VNA(chip).measure(6, 0.01, input=drive, outputs=[readout], noise_frequencies=offsets)
    with pytest.raises(ValueError):
        IQReceiver(integration_time=0)


def test_jax_key_and_receiver_time_gradient_on_captured_numpy_data() -> None:
    """Keyed draws and receiver derivatives use captured data without host conversion."""
    import jax
    import jax.numpy as jnp
    chip, drive, readout = line()
    measured = VNA(chip).measure(6.0, 0.01, input=drive, outputs=[readout])
    def variance(time):
        return measured.statistics(receiver=IQReceiver(integration_time=time)).covariance(readout)[0, 0]
    value, gradient = jax.jit(jax.value_and_grad(variance))(jnp.asarray(1000.0))
    np.testing.assert_allclose([value, gradient], [150.5/2000, -150.5/2e6], rtol=1e-6)
    stats = measured.statistics(receiver=IQReceiver(integration_time=1000))
    draw = jax.jit(lambda key: stats.sample(3, key=key).field(readout))
    np.testing.assert_array_equal(draw(jax.random.key(2)), draw(jax.random.key(2)))


def test_squeezed_output_retains_anomalous_quadrature_noise() -> None:
    """A subthreshold parametric oscillator has the analytic squeezed output spectrum."""
    from quchip import DeviceModel, parameter
    class ParametricMode(DeviceModel):
        epsilon: float = parameter(default=0.005)
        freq: float = parameter(default=0.0)
        approximation = None
        def local_hamiltonian(self, op, p):
            return p.freq*op.n + 1j*p.epsilon/(4*np.pi)*(op.adag @ op.adag - op.a @ op.a)
    mode = ParametricMode(levels=14, label="m")
    net = PortNetwork()
    port = net.port("p", target=mode, rate=0.04)
    measured = VNA(Chip([mode], port_network=net, frame="lab")).measure(
        0.0, 0.0, input=port, outputs=[port], noise_frequencies=[-0.1, -0.01, 0, 0.01, 0.1])
    physical = sum(measured.noise_contributions(port).values())[2]
    ratio = (0.02 + 0.005)/(0.02 - 0.005)
    # Heterodyne adds a half per quadrature to normal-order excess.
    expected = np.diag([(ratio**2 + 1)/4, (ratio**-2 + 1)/4])
    np.testing.assert_allclose(physical + np.eye(2)/2, expected, atol=2e-7)
    np.testing.assert_allclose(measured.mode_amplitude(mode), 0., atol=1e-12)
    np.testing.assert_allclose(measured.photon_number(mode),
                               2*.005**2/(.04**2-4*.005**2), atol=2e-8)


def test_internal_moments_retain_projected_authored_operators():
    """Mode capture matches stationary observables after a nontrivial basis projection."""
    from quchip import DeviceModel, parameter
    class DisplacedMode(DeviceModel):
        freq: float = parameter(default=.06)
        force: float = parameter(default=.01)
        approximation = None
        def local_hamiltonian(self, op, p):
            return p.freq*op.n + p.force*(op.a + op.adag)
    mode = DisplacedMode(levels=6, label="r")
    mode.basis = "eigen"
    mode.projection_levels = 3
    net = PortNetwork()
    port = net.port("p", target=mode, rate=.04)
    chip = Chip([mode], port_network=net, frame="lab")
    state = chip.steadystate(e_ops={mode: ["a", "n"]})
    measured = VNA(chip).measure(0., 0., input=port, outputs=[port],
                                noise_frequencies=[-.04, -.004, 0., .004, .04])
    np.testing.assert_allclose(measured.mode_amplitude(mode), state.expect(mode, index=0), atol=1e-12)
    np.testing.assert_allclose(measured.photon_number(mode), state.expect(mode, index=1).real, atol=1e-12)
    assert measured.photon_number(mode) > .02
    assert measured.mode_frequency(mode) == 0.


def split_thermal(*, delay=0.0, backend="qutip"):
    """Mix a warm load with vacuum and observe both outputs simultaneously."""
    mode = Resonator(freq=6.0, levels=2, label="unused", T1=100)
    net = PortNetwork()
    net.port("quiet", target=mode, rate=0.01)
    split = net.beam_splitter("split", eta=0.25)
    load = net.termination("warm", thermal_occupation=2.0)
    cable = net.delay("cable", duration=delay)
    net.connect(load.output_terminal("1"), split.input_terminal("right"))
    net.connect(split.output_terminal("right"), cable.input_terminal("1"))
    net.connect(cable.output_terminal("1"), load.input_terminal("1"))
    left = net.expose("left", input=split.input_terminal("left"), output=split.output_terminal("left"))
    right = net.expose("right", at=cable.port(2))
    return Chip([mode], port_network=net, backend=backend), left, right


def test_joint_thermal_outputs_keep_phase_delay_and_digital_filter() -> None:
    """A split thermal field keeps cross covariance and loses overlap after a delay."""
    chip, left, right = split_thermal(delay=100.0)
    grid = np.linspace(-0.05, 0.05, 4001)
    measured = VNA(chip).measure(0.0, 0.0, input=left, outputs=[left, right], noise_frequencies=grid)
    plain = measured.statistics(receiver=IQReceiver(integration_time=200))
    # Independent thermal input enters the second column of the beam splitter.
    source = np.array([np.sqrt(0.75), 0.5])
    normal = 2 * source[:, None]*source.conj()[None, :]
    expected_cross = np.array([[normal[0, 1].real, -normal[0, 1].imag],
                               [normal[0, 1].imag, normal[0, 1].real]])/2
    np.testing.assert_allclose(plain.covariance(left, right), expected_cross * 0.5 / 200, atol=1e-12)
    receiver = IQReceiver(integration_time=200, transfer=lambda f: np.exp(-(f/0.01)**2))
    filtered = measured.statistics(receiver=receiver)
    integral = np.trapezoid(np.sinc(grid*200)**2 * np.exp(-2*(grid/0.01)**2)
                           * np.cos(2*np.pi*grid*100), grid)
    np.testing.assert_allclose(filtered.covariance(left, right), expected_cross*integral, atol=1e-10)

    # An asymmetric passband checks the sign of the delayed cross-spectrum;
    # a symmetric boxcar alone cannot distinguish the two Fourier conventions.
    receiver = IQReceiver(integration_time=200, transfer=lambda f: np.exp(-((f-.003)/.01)**2))
    filtered = measured.statistics(receiver=receiver)
    cross = normal[0, 1] * np.trapezoid(np.sinc(grid*200)**2 * np.abs(receiver.transfer(grid))**2
                                       * np.exp(2j*np.pi*grid*100), grid)/2
    np.testing.assert_allclose(filtered.covariance(left, right),
                               [[cross.real, -cross.imag], [cross.imag, cross.real]], atol=1e-10)


def test_amplifier_before_splitter_preserves_correlated_noise_and_signal() -> None:
    """An upstream amplifier contributes one shared noise field to both splitter outputs."""
    mode = Resonator(freq=6.0, levels=8, label="r")
    net = PortNetwork()
    port = net.port("p", target=mode, rate=0.04)
    amp = net.amplifier("amp", gain=100.0, added_noise=1.0)
    split = net.hybrid90("split")
    net.link(port, amp)
    net.connect(amp.output_terminal("2"), split.input_terminal("left"))
    left = net.expose("left", input=amp.input_terminal("2"), output=split.output_terminal("left"))
    right = net.expose("right", input=split.input_terminal("right"), output=split.output_terminal("right"))
    chip = Chip([mode], port_network=net)
    measured = VNA(chip).measure(6.0, 0.01, input=left, outputs=[left, right])
    np.testing.assert_allclose(measured.values, -0.1*np.array([1, -1j])/np.sqrt(2), atol=1e-9)
    deterministic = VNA(chip).sweep([6.0])
    np.testing.assert_allclose(measured.ratio(left), deterministic.s(left, left)[0], atol=1e-7)
    stats = measured.statistics(receiver=IQReceiver(integration_time=1000))
    np.testing.assert_allclose(stats.covariance(left), np.eye(2)*(149.5/2+1)/2000, atol=1e-10)
    np.testing.assert_allclose(stats.covariance(left, right), [[0,-149.5/4000],[149.5/4000,0]], atol=1e-10)
    vna = VNA(chip)
    vna.pump(left, freq=6.0, amplitude=0.01)
    np.testing.assert_allclose(vna.output_spectrum(right, frequencies=[0.0]).added_noise_spectrum, [149.5/2])


@pytest.mark.optional_backend
def test_physical_gain_gradient_matches_amplifier_formula() -> None:
    """Physical covariance gradients agree with independent amplifier noise derivatives."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp
    chip, drive, readout = line(backend="dynamiqs")
    def variance(gain):
        rebound = chip.with_params({"network.component.hemt.gain": gain})
        measurement = VNA(rebound).measure(6.0, 0.0, input=drive, outputs=[readout],
                                           noise_frequencies=[-0.1, -0.01, 0, 0.01, 0.1])
        return measurement.statistics(receiver=IQReceiver(integration_time=1000)).covariance(readout)[0, 0]
    value, gradient = jax.jit(jax.value_and_grad(variance))(jnp.asarray(100.0))
    np.testing.assert_allclose([value, gradient], [150.5/2000, 1.5/2000], rtol=1e-7)


@pytest.mark.parametrize("kind", ["delay", "attenuator", "filter"])
def test_branched_chain_preserves_independently_peeled_reverse_leg(kind) -> None:
    """An exposed reverse leg and an internal forward leg retain separate reference actions."""
    mode = Resonator(freq=6.0, levels=8, label="r")
    net = PortNetwork()
    port = net.port("p", target=mode, rate=0.04)
    amp = net.amplifier("amp", gain=100.0, added_noise=1.0)
    if kind == "delay":
        section = net.delay("section", duration=0.123)
        transmission = np.exp(2j*np.pi*6.0*0.123)
    elif kind == "attenuator":
        section = net.attenuator("section", eta=0.25, thermal_occupation=0.0)
        transmission = 0.5
    else:
        section = net.filter("section", transfer=lambda f: 0.5+np.zeros_like(f))
        transmission = 0.5
    splitter = net.hybrid90("split")
    net.link(port, amp, section)
    net.connect(section.output_terminal("2"), splitter.input_terminal("left"))
    left = net.expose("left", input=section.input_terminal("2"), output=splitter.output_terminal("left"))
    right = net.expose("right", input=splitter.input_terminal("right"), output=splitter.output_terminal("right"))
    measured = VNA(Chip([mode], port_network=net)).measure(6.0, 0.01, input=left, outputs=[left, right])
    np.testing.assert_allclose(measured.values, -0.1*np.conj(transmission)**2*np.array([1,-1j])/np.sqrt(2), atol=1e-8)
    noise = measured.noise_contributions(left)
    normal = sum(noise.values())
    np.testing.assert_allclose(np.real(np.trace(normal, axis1=-2, axis2=-1)),
                               149.5*abs(transmission)**2/2, atol=1e-8)


def test_measurement_sweeps_and_parameter_capture_are_stable() -> None:
    """Gain, amplitude, frequency and sample axes keep their order and captured parameters."""
    from quchip import Sweep
    chip, drive, readout = line()
    measured = VNA(chip).measure([5.99, 6.0], [0.0, 0.01],
        Sweep([100.0, 200.0], name="network.component.hemt.gain"), input=drive, outputs=[readout])
    assert measured.shape == (2, 2, 2)
    assert measured.parameters[0]["network.component.hemt.gain"] == 100.0
    assert measured.parameters[-1]["network.component.hemt.gain"] == 200.0
    sample = measured.sample(3, receiver=IQReceiver(integration_time=1000), seed=3)
    assert sample.field(readout).shape == (3, 2, 2, 2)
    assert np.isnan(sample.ratio(readout)[:, :, 0]).all()
    with pytest.raises(TypeError):
        measured.parameters[0]["network.component.hemt.gain"] = 3.0
