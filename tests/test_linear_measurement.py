"""Compact acquisition preserves noise quanta and power gain conventions."""

import numpy as np
import pytest

from quchip import Capacitive, Chip, IQReceiver, PortNetwork, Resonator, RWA, VNA


@pytest.mark.parametrize("operator", [None, "a"])
def test_custom_lowering_port_keeps_its_normalization(operator):
    """A harmonic Hamiltonian does not imply a unit-amplitude port operator."""
    class DipoleMode(Resonator):
        def lowering_operator(self):
            return 2 * super().lowering_operator()

    def reflection(mode, rate):
        network = PortNetwork()
        port = network.port("p", target=mode, operator=operator, rate=rate)
        probe = network.expose("probe", at=port)
        chip = Chip([mode], port_network=network)
        return VNA(chip).sweep([5.99, 6., 6.01]).s(probe, probe)

    np.testing.assert_allclose(reflection(DipoleMode(6., levels=3), .01),
                               reflection(Resonator(6., levels=3), .04), atol=1e-10)


def thermal_fridge(*, backend="qutip", filter_first=True):
    r = Resonator(freq=6.0, levels=8, internal_quality_factor=10_000, label="r")
    net = PortNetwork(label="fridge")
    port = net.port("p", target=r, rate=.04)
    filt = net.filter("filter", transfer=lambda f: 1/(1-1j*(f-6)/.1))
    att = net.attenuator("att", loss_db=10., thermal_occupation=.02)
    circ = net.circulator("circ")
    amp = net.amplifier("amp", gain_db=20., added_noise=7.)
    chain = (filt, att) if filter_first else (att, filt)
    lead = net.delay("lead", duration=0.)
    net.link(lead, *chain, circ.port(1))
    net.link(circ.port(2), port)
    net.link(circ.port(3), amp)
    drive = net.expose("drive", at=lead.port(1))
    readout = net.expose("readout", at=amp.port(2))
    return Chip([r], port_network=net, backend=backend), drive, readout


def test_compact_capture_matches_density_matrix_with_thermal_input_and_upstream_filter():
    """A filter preceding the occupied sources permits exact Markov acquisition."""
    chip, drive, readout = thermal_fridge()
    offsets = [-.04, -.004, 0., .004, .04]
    vna = VNA(chip)
    compact = vna.measure([5.997, 6., 6.003], .002, input=drive, outputs=[readout], noise_frequencies=offsets)
    general = vna.measure([5.997, 6., 6.003], .002, input=drive, outputs=[readout],
                          noise_frequencies=offsets, options={})
    np.testing.assert_allclose(compact.values, general.values, atol=1e-8)
    np.testing.assert_allclose(compact.mode_amplitude("r"), general.mode_amplitude("r"), atol=1e-8)
    np.testing.assert_allclose(compact.photon_number("r"), general.photon_number("r"), atol=1e-8)
    np.testing.assert_allclose(compact.mode_frequency("r"), general.mode_frequency("r"), atol=1e-12)
    assert compact.noise_components.keys() == general.noise_components.keys()
    for source in compact.noise_components:
        np.testing.assert_allclose(compact.noise_components[source][0], general.noise_components[source][0], atol=1e-8)
        np.testing.assert_allclose(compact.noise_components[source][1], general.noise_components[source][1], atol=1e-7)


def test_filter_after_thermal_source_still_rejects_colored_backaction():
    """Changing propagation order does not turn a colored bath into a Markov input."""
    chip, drive, readout = thermal_fridge(filter_first=False)
    for options in (None, {}):
        with pytest.raises(ValueError, match="colored thermal backaction"):
            VNA(chip).measure(6., .002, input=drive, outputs=[readout], options=options)


def test_lossless_thermal_attenuator_before_filter_emits_no_colored_noise():
    """A zero-loss endpoint is valid even when its load has thermal quanta."""
    chip, drive, readout = thermal_fridge(filter_first=False)
    chip = chip.with_params({"network.component.att.loss_db": 0.})
    for options in (None, {}):
        measured = VNA(chip).measure(6., .001, input=drive, outputs=[readout], options=options,
                                   noise_frequencies=[-.1, -.01, 0., .01, .1])
        np.testing.assert_allclose(measured.noise_contributions(readout)["input.drive.att"], 0., atol=1e-12)


def test_filtered_auxiliary_field_mixed_after_device_does_not_create_backaction():
    """Input-system coupling S†L distinguishes downstream mixing from colored heating."""
    r = Resonator(freq=6., levels=4, label="r")
    net = PortNetwork()
    port = net.port("p", target=r, rate=.04)
    split = net.hybrid90("split")
    filt = net.filter("aux", transfer=lambda f: np.sqrt(.6)/(1-1j*(f-6.)/.02), thermal_occupation=.2)
    net.connect(port.output, split.input_terminal("left"))
    net.connect(filt.output_terminal("2"), split.input_terminal("right"))
    net.connect(split.output_terminal("right"), filt.input_terminal("2"))
    left = net.expose("left", input=port.input, output=split.output_terminal("left"))
    right = net.expose("right", at=filt.port(1))
    vna = VNA(Chip([r], port_network=net))
    for options in (None, {}):
        measured = vna.measure(6., 0., input=left, outputs=[left, right],
                               noise_frequencies=[-.04, -.004, 0., .004, .04], options=options)
        np.testing.assert_allclose(measured.noise_spectrum(left)[2], .04, atol=1e-10)
        np.testing.assert_allclose(measured.noise_spectrum(right)[2], .104, atol=1e-10)


def test_uncoupled_lossless_mode_has_no_unique_stationary_capture():
    """A dark mode cannot be silently assigned a stationary state by the compact route."""
    r = Resonator(freq=6., label="r")
    dark = Resonator(freq=7., label="dark")
    net = PortNetwork()
    port = net.port("p", target=r, rate=.04)
    with pytest.raises(ValueError, match="all modes to decay"):
        VNA(Chip([r, dark], port_network=net)).measure(6., .001, input=port, outputs=[port])


def test_nine_mode_measurement_matches_independent_star_response_without_density_matrix(monkeypatch):
    """Nine harmonic modes acquire means and noise without tensor-product operators."""
    frequencies = np.linspace(6., 7., 8)
    bus = Resonator(freq=6.5, levels=8, internal_quality_factor=100_000, label="bus")
    modes = [Resonator(freq=f, levels=8, internal_quality_factor=8000+2000*i, label=f"r{i}")
             for i, f in enumerate(frequencies)]
    couplings = [Capacitive(bus, mode, g=.025+.002*i) for i, mode in enumerate(modes)]
    net = PortNetwork()
    port = net.port("bus", target=bus, rate=2*np.pi)
    amp = net.amplifier("amp", gain_db=40., added_noise=8.)
    net.link(port, amp)
    output = net.expose("out", at=amp.port(2))
    chip = Chip([bus, *modes], couplings, port_network=net, approximation=RWA())
    def forbidden(*args, **kwargs):
        raise AssertionError("harmonic acquisition constructed a density matrix")
    monkeypatch.setattr(chip.backend, "steadystate", forbidden)
    monkeypatch.setattr(chip.backend, "stationary_resolvent", forbidden)
    probe = np.linspace(5.98, 7.02, 51)
    measured = VNA(chip).measure(probe, .02, input=output, outputs=[output],
                                noise_frequencies=[-.1, -.01, 0., .01, .1])
    den = 2*np.pi*np.array([m.freq/m.internal_quality_factor for m in modes])/2
    den = den+2j*np.pi*(frequencies[None, :]-probe[:, None])
    bus_den = (2*np.pi+2*np.pi*bus.freq/bus.internal_quality_factor)/2+2j*np.pi*(bus.freq-probe)
    bus_den += np.sum((2*np.pi*np.array([c.g for c in couplings]))**2/den, axis=1)
    np.testing.assert_allclose(measured.ratio(output), 100*(1-2*np.pi/bus_den).conj(), atol=1e-9)
    covariance = measured.statistics(receiver=IQReceiver(integration_time=1e6)).covariance(output)
    np.testing.assert_allclose(covariance, np.broadcast_to(np.eye(2)*(85000.5/2e6), covariance.shape), atol=1e-10)


def test_noise_quanta_and_db_parameters_rebind_and_roundtrip():
    """dB and noise quanta remain authored parameters after capture and serialization."""
    r = Resonator(freq=6., levels=4, label="r")
    net = PortNetwork()
    port = net.port("p", target=r, rate=.04)
    loss = net.attenuator("loss", loss_db=10.)
    amp = net.amplifier("amp", gain_db=20., added_noise=2.)
    net.link(port, loss, amp)
    net.expose("out", at=amp.port(2))
    chip = Chip([r], port_network=net)
    restored = Chip.from_dict(chip.to_dict())
    changed = restored.with_params({"network.component.amp.gain_db": 30., "network.component.loss.loss_db": 20.,
                                    "network.component.amp.added_noise": 3.})
    point = VNA(changed).measure(6., .001, input="out", outputs=["out"])
    assert point.parameters[0]["network.component.amp.gain_db"] == 30.
    expected = (1000*3.+(1000-1)/2+1)/2e6
    np.testing.assert_allclose(point.statistics(receiver=IQReceiver(integration_time=1e6)).covariance("out"),
                               np.eye(2)*expected, atol=1e-10)
    physical = 2e6*expected-1
    np.testing.assert_allclose(point.noise_spectrum("out"), physical, rtol=1e-12)
    watts = 6.62607015e-34*(6.+point.noise_frequencies)*1e9*physical
    np.testing.assert_allclose(point.noise_spectrum("out", unit="W/Hz"), watts, rtol=1e-12)
    np.testing.assert_allclose(point.noise_spectrum("out", unit="dBm/Hz"), 10*np.log10(watts/1e-3), atol=1e-12)


@pytest.mark.parametrize("kwargs", [dict(gain=100., gain_db=20., added_noise=1.),
    dict(gain_db=20., added_noise=.1), dict(gain=.5, added_noise=1.),
    dict(gain_db=20., added_noise=np.nan), dict(gain_db=20., added_noise=np.inf)])
def test_invalid_amplifier_declarations_fail(kwargs):
    """Gain is unambiguous and finite added quanta satisfy the quantum floor."""
    with pytest.raises(ValueError):
        PortNetwork().amplifier("amp", **kwargs)


@pytest.mark.optional_backend
def test_linear_measurement_added_noise_gradient():
    """Added quanta differentiate through harmonic acquisition and receiver."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("dynamiqs")
    chip, drive, readout = thermal_fridge(backend="dynamiqs")
    offsets = np.array([-.1, -.01, 0., .01, .1])
    def variance(added_noise):
        changed = chip.with_params({"network.component.amp.added_noise": added_noise})
        result = VNA(changed).measure(6., .002, input=drive, outputs=[readout],
                                     noise_frequencies=offsets)
        return result.statistics(receiver=IQReceiver(integration_time=1e6)).covariance(readout)[0, 0]
    gradient = jax.jit(jax.grad(variance))(7.)
    expected = 100/(2e6)
    np.testing.assert_allclose(gradient, expected, rtol=1e-9)


def test_internal_occupation_includes_filtered_drive_and_thermal_bath(monkeypatch):
    """Captured mode moments obey the cavity susceptibility and thermal balance."""
    chip, drive, readout = thermal_fridge()
    mode = chip.devices[0]
    frequencies = np.array([5.99, 6., 6.01])
    amplitudes = np.array([0., .002, .01j])
    result = VNA(chip).measure(frequencies, amplitudes, input=drive, outputs=[readout],
                               noise_frequencies=[-.04, -.004, 0., .004, .04])
    kappa = .04 + 2*np.pi*6/10_000
    field = -np.sqrt(.04*.1)*amplitudes.conj()[:, None]/(
        (1-1j*(frequencies-6)/.1)*(kappa/2+2j*np.pi*(6-frequencies)))
    thermal = .04*.9*.02/kappa
    np.testing.assert_allclose(result.mode_amplitude(mode), field, atol=1e-12)
    np.testing.assert_allclose(result.photon_number(mode), abs(field)**2+thermal, atol=1e-12)

    def forbidden(*args, **kwargs):
        raise AssertionError("Captured observable invoked a physical solve")
    monkeypatch.setattr(chip.backend, "linear_response", forbidden)
    monkeypatch.setattr(chip.backend, "steadystate", forbidden)
    mode.freq = 8.
    np.testing.assert_allclose(result.photon_number("r"), abs(field)**2+thermal, atol=1e-12)
    np.testing.assert_allclose(result.mode_frequency("r"), np.broadcast_to(frequencies, (3, 3)))
    for values in (result.mode_amplitudes, result.photon_numbers, result.mode_frequencies):
        with pytest.raises(ValueError):
            values[0, 0, 0] = 1
    with pytest.raises(KeyError, match="Unknown Fock mode"):
        result.photon_number("missing")


def test_coupled_modes_share_thermal_noise_and_match_general_solver():
    """Thermal correlations through a bus give the same occupations in both solvers."""
    bus = Resonator(freq=6., levels=5, label="bus")
    r = Resonator(freq=6.01, levels=5, internal_quality_factor=6000, label="r")
    net = PortNetwork()
    port = net.port("p", target=bus, rate=.04)
    att = net.attenuator("att", loss_db=10., thermal_occupation=.01)
    net.link(att, port)
    drive = net.expose("drive", at=att.port(1))
    chip = Chip([bus, r], [Capacitive(bus, r, g=.005)], port_network=net, approximation=RWA())
    results = [VNA(chip).measure(6.005, .003, input=drive, outputs=[drive],
                                noise_frequencies=[-.04, -.004, 0., .004, .04], options=options)
               for options in (None, {})]
    # Five levels leave O(n^5) thermal tails at n < .01.
    for mode in (bus, r):
        np.testing.assert_allclose(results[0].photon_number(mode), results[1].photon_number(mode), atol=2e-8)
        np.testing.assert_allclose(results[0].mode_amplitude(mode), results[1].mode_amplitude(mode), atol=2e-8)
        assert results[0].photon_number(mode) > abs(results[0].mode_amplitude(mode))**2


@pytest.mark.optional_backend
def test_internal_photon_number_jit_gradients_follow_drive_and_thermal_balance():
    """JAX mode occupations differentiate through drive, loss, and bath diffusion."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("dynamiqs")
    chip, drive, readout = thermal_fridge(backend="dynamiqs")
    def occupation(amplitude, bath):
        changed = chip.with_params({"network.component.att.thermal_occupation": bath})
        result = VNA(changed).measure(6., amplitude, input=drive, outputs=[readout],
                                     noise_frequencies=[-.04, -.004, 0., .004, .04])
        return result.photon_number("r")
    value, gradients = jax.jit(jax.value_and_grad(occupation, argnums=(0, 1)))(.002, .02)
    kappa = .04 + 2*np.pi*6/10_000
    expected = .04*.1*.002**2/(kappa/2)**2 + .04*.9*.02/kappa
    np.testing.assert_allclose(value, expected, rtol=1e-10)
    np.testing.assert_allclose(gradients, [2*.04*.1*.002/(kappa/2)**2, .04*.9/kappa], rtol=1e-10)
