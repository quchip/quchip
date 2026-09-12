"""Terminal detector models preserve quantum preparation and joint Born statistics."""
import numpy as np
import pytest

from quchip import (
    ChargeDrive, Chip, DeviceModel, DuffingTransmon, IQReadout, IQReceiver,
    PortNetwork, QuantumSequence, Resonator, Square, VNA, parameter,
)
from quchip.results import SimulationBatchResult


def preparation(backend="qutip", *, mixed=False, states="final"):
    q = DuffingTransmon(freq=5., anharmonicity=-.2, levels=2, label="q")
    chip = Chip([q], backend=backend, frame="rotating")
    state = np.array([np.sqrt(.7), 1j*np.sqrt(.3)])[:, None]
    if mixed:
        state = chip.backend.from_array(state @ state.conj().T, dims=[list(chip.dims), list(chip.dims)])
    result = QuantumSequence(chip).simulate(tlist=[0., 1.], initial_state=state,
        states=states)
    return chip, q, result


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("mixed", [False, True])
def test_born_probabilities_and_counts_preserve_ket_dm_semantics(backend, mixed):
    """Either solver yields the same probabilities and reproducible multinomial counts."""
    chip, q, result = preparation(backend, mixed=mixed)
    measurement = result.measure(q)
    np.testing.assert_allclose(measurement.probabilities, [.7, .3], atol=1e-10)
    assert result.solver == ("mesolve" if mixed else "sesolve")
    shots = measurement.sample(100_000, seed=17)
    np.testing.assert_array_equal(shots.indices, measurement.sample(100_000, seed=17).indices)
    assert abs(shots.counts()[(1,)] / 100_000 - .3) < 5*np.sqrt(.3*.7/100_000)
    assert shots.iq is None
    assert measurement.devices == ("q",)
    np.testing.assert_allclose(result.measure("q", t=1).probabilities, measurement.probabilities)


@pytest.mark.parametrize("mixed", [False, True])
def test_bell_joint_measurement_and_custom_complex_bases(mixed):
    """Joint draws retain Bell correlations in energy and complex rotated local bases."""
    devices = [DuffingTransmon(freq=5+i, anharmonicity=-.2, levels=2, label=name)
               for i, name in enumerate(("a", "b"))]
    chip = Chip(devices, frame="rotating")
    state = np.array([1., 0., 0., 1.])[:, None]/np.sqrt(2)
    if mixed:
        state = chip.backend.from_array(state @ state.conj().T, dims=[list(chip.dims), list(chip.dims)])
    result = QuantumSequence(chip).simulate(tlist=[0., 1.], initial_state=state,
        partition=False)
    joint = result.measure(*devices)
    np.testing.assert_allclose(joint.probabilities, [.5, 0, 0, .5], atol=1e-10)
    assert set(np.asarray(joint.sample(500, seed=1).indices)) == {0, 3}
    y = np.array([[1, 1], [1j, -1j]])/np.sqrt(2)
    np.testing.assert_allclose(result.measure(*devices, basis={d: y for d in devices}).probabilities,
                               [0, .5, .5, 0], atol=1e-10)
    np.testing.assert_allclose(result.measure(devices[1]).probabilities, [.5, .5], atol=1e-10)


def test_energy_basis_snapshot_and_custom_basis_validation():
    """Energy measurement uses captured vectors after the source Hamiltonian changes."""
    class TiltedQubit(DeviceModel):
        tilt: float = parameter(default=.04)
        def local_hamiltonian(self, op, p):
            return .1*op.n + p.tilt*(op.a + op.adag)
    q = TiltedQubit(levels=2, label="q")
    chip = Chip([q], frame="lab")
    result = QuantumSequence(chip).simulate(tlist=[0., .1], initial_state=chip.bare_state(q=0))
    q.tilt = -.03
    np.testing.assert_allclose(result.measure(q).probabilities, [1, 0], atol=1e-12)
    assert result.measure(q, basis="solver").probabilities[1] > .05
    with pytest.raises(ValueError, match="orthonormal"):
        result.measure(q, basis=np.ones((2, 2)))
    with pytest.raises(ValueError, match="distinct"):
        result.measure(q, q)
    with pytest.raises(ValueError, match="basis"):
        result.measure(q, basis="bad")


def test_assignment_orientation_and_conditional_iq_mixture():
    """Assignment columns and conditional mixture moments describe distinct detector models."""
    _, q, result = preparation()
    m = result.measure(q)
    assignment = np.array([[.9, .2], [.1, .8]])
    np.testing.assert_allclose(m.recorded_probabilities(assignment), [.69, .31])
    draws = m.sample(100_000, assignment=assignment, seed=7)
    for physical in (0, 1):
        conditional = np.asarray(draws.indices)[draws.physical_indices == physical]
        assert abs(np.mean(conditional == 1) - assignment[1, physical]) < .01
    readout = IQReadout([-1+.2j, 2-.4j], [[[.1, .02], [.02, .2]], [[.3, 0], [0, .1]]])
    draws = m.sample(100_000, readout=readout, seed=7)
    measured = np.stack((draws.iq.real, draws.iq.imag), axis=-1)
    np.testing.assert_allclose(np.mean(draws.iq), readout.mean(m.probabilities), atol=.015)
    np.testing.assert_allclose(np.cov(measured.T), readout.covariance(m.probabilities), atol=.015)
    with pytest.raises(ValueError, match="columns"):
        m.recorded_probabilities(assignment.T)
    with pytest.raises(ValueError, match="Choose"):
        m.sample(10, readout=readout, assignment=assignment)
    with pytest.raises(ValueError, match="one distribution"):
        m.sample(10, readout=IQReadout([1], np.eye(2)))
    with pytest.raises(ValueError):
        readout.means[0] = 99


def test_batch_coordinates_and_partitioned_measurement_avoid_joint_state(monkeypatch):
    """Partition probabilities combine only independent components and retain batch axes."""
    a = DuffingTransmon(freq=5., anharmonicity=-.2, levels=2, label="a")
    b = Resonator(freq=6., levels=3, label="b")
    chip = Chip([a, b], frame="rotating")
    result = QuantumSequence(chip).simulate(tlist=[0., 1.], initial_state={"a": 1, "b": 2})
    def forbidden(*args, **kwargs):
        raise AssertionError("Rebuilt an unnecessary joint state")
    monkeypatch.setattr(type(result), "final_state", property(forbidden))
    m = result.measure(b, a)
    assert m.outcomes == ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))
    np.testing.assert_allclose(m.probabilities, [0, 0, 0, 0, 0, 1])
    _, q, single = preparation()
    batch = SimulationBatchResult([single, single], shape=(2,), axes=(("pulse.amplitude", [.1, .2]),))
    measured = batch.measure(q)
    assert measured.axes[0][0] == "pulse.amplitude"
    assert measured.sample(7).indices.shape == (7, 2)
    np.testing.assert_allclose(measured.probabilities, [[.7, .3], [.7, .3]])


def test_missing_states_and_invalid_shots_are_explicit():
    """Unsaved states cannot be recovered from a population trace or silently interpolated."""
    _, q, result = preparation(states="none")
    with pytest.raises(RuntimeError, match="state"):
        result.measure(q)
    _, q, result = preparation()
    with pytest.raises(RuntimeError, match="states"):
        result.measure(q, t=0)
    with pytest.raises(ValueError, match="saved time"):
        result.measure(q, t=.5)
    for count in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            result.measure(q).sample(count)


def test_closed_rabi_needs_no_readout_pulse_or_master_equation():
    """Postprocessing a driven ket agrees with stored populations and leaves evolution intact."""
    q = DuffingTransmon(freq=5., anharmonicity=-.2, levels=2, label="q")
    chip = Chip([q], frame="rotating")
    line = ChargeDrive(q, label="xy")
    chip.wire(line)
    sequence = QuantumSequence(chip)
    sequence.schedule(line, envelope=Square(duration=20., amplitude=.025), freq=5.)
    result = sequence.simulate(tlist=np.linspace(0, 20, 21))
    assert result.solver == "sesolve"
    for i in (0, 5, 10, 20):
        np.testing.assert_allclose(result.measure(q, t=result.times[i]).probabilities[1],
                                   result.population(q, 1)[i], atol=1e-12)
    assert result.measure(q).probabilities[1] > .99


def wired_result(*, backend="qutip", gain=100., noise=1., loss=.25):
    r = Resonator(freq=6., levels=2, label="r")
    net = PortNetwork()
    port = net.port("p", target=r, rate=.04)
    circ = net.circulator("circ")
    amp = net.amplifier("hemt", gain=gain, added_noise=noise)
    attenuator = net.attenuator("cable", eta=loss, thermal_occupation=.2)
    net.link(port, circ.port(2))
    net.link(circ.port(3), amp, attenuator)
    drive = net.expose("drive", at=circ.port(1))
    out = net.expose("out", at=attenuator.port(2))
    chip = Chip([r], port_network=net, frame="rotating", backend=backend)
    result = QuantumSequence(chip).simulate(tlist=[0., 1.], states="final")
    return chip, out, drive, result


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_wiring_readout_matches_vna_noise_and_captures_model(backend):
    """Coherent templates use the same downstream noise, gain and receiver as VNA."""
    chip, out, drive, result = wired_result(backend=backend)
    receiver = IQReceiver(1000)
    detector = result.iq_readout(out, means=[-.1, .1], frequency=6., receiver=receiver)
    np.testing.assert_allclose(detector.means, [-.5, .5])
    expected = (.25*(100+99/2)+.75*.2+1)/2000
    np.testing.assert_allclose(detector.iq_covariance[0], np.eye(2)*expected, atol=1e-12)
    stationary = VNA(chip).measure(6., .001, input=drive, outputs=[out])
    np.testing.assert_allclose(detector.iq_covariance[0], stationary.statistics(receiver=receiver).covariance(out),
                               atol=1e-10)
    np.testing.assert_allclose(sum(detector.contributions.values()), detector.iq_covariance[0])
    chip.disconnect_network()
    captured = result.iq_readout(out, means=[-.1, .1], frequency=6., receiver=receiver)
    np.testing.assert_allclose(captured.means, detector.means)
    np.testing.assert_allclose(captured.iq_covariance, detector.iq_covariance)


def test_jax_measurement_gradients_and_keyed_detector_sampling():
    """Born and mixture statistics retain analytical gradients; keyed labels are reproducible."""
    import jax
    import jax.numpy as jnp
    from quchip.results.terminal import StateMeasurement
    from quchip.backend.containers import SolverResult
    from quchip.results import SimulationResult
    from quchip.backend.dynamiqs import DynamiqsBackend
    backend = DynamiqsBackend()
    def probability(angle):
        state = jnp.array([jnp.cos(angle), 1j*jnp.sin(angle)])[:, None]
        raw = SolverResult(times=jnp.array([0.]), states=None, final_state=state, solver="sesolve")
        result = SimulationResult(raw, backend, [2], device_info=[("q", False)])
        return result.measure("q", basis="solver").probabilities[1]
    value, gradient = jax.jit(jax.value_and_grad(probability))(.4)
    np.testing.assert_allclose([value, gradient], [np.sin(.4)**2, np.sin(.8)], atol=1e-12)
    m = StateMeasurement(("q",), ((0,), (1,)), jnp.array([.7, .3]))
    detector = IQReadout([-1., 1.], np.eye(2)*.1)
    sample = jax.jit(lambda key: m.sample(30, key=key, readout=detector).iq)
    np.testing.assert_array_equal(sample(jax.random.key(2)), sample(jax.random.key(2)))
    def mixture(p):
        return jnp.real(detector.mean(jnp.array([1-p, p])))
    assert jax.jit(jax.grad(mixture))(.3) == pytest.approx(2.)
    _, out, _, result = wired_result()
    def variance(time):
        return result.iq_readout(out, means=[-.1, .1], frequency=6.,
                                receiver=IQReceiver(time)).iq_covariance[0, 0, 0]
    v, g = jax.jit(jax.value_and_grad(variance))(1000.)
    assert g == pytest.approx(-v/1000)


def test_partitioned_custom_basis_validation_and_phase_convention():
    """Custom bases have the same contract across partitions and act in the stored integration frame."""
    a = DuffingTransmon(freq=5., anharmonicity=-.2, levels=2, label="a")
    b = DuffingTransmon(freq=6., anharmonicity=-.2, levels=2, label="b")
    hadamard = np.array([[1., 1.], [1., -1.]])/np.sqrt(2)
    for partition in (True, False):
        result = QuantumSequence(Chip([a, b], frame="rotating")).simulate(
            tlist=[0., .1], partition=partition)
        with pytest.raises(ValueError, match="mapping"):
            result.measure(a, b, basis=hadamard)
        with pytest.raises(ValueError, match="unmeasured"):
            result.measure(a, b, basis={"typo": hadamard})
        np.testing.assert_allclose(result.measure(b, a, basis={a: hadamard}).probabilities,
                                   [.5, .5, 0, 0], atol=1e-12)
    for frame, expected in (("lab", [0, 1]), ("rotating", [1, 0])):
        chip = Chip([a], frame=frame)
        result = QuantumSequence(chip).simulate(tlist=[0., .1],
            initial_state=np.ones((2, 1))/np.sqrt(2))
        np.testing.assert_allclose(result.measure(a, basis=hadamard).probabilities, expected, atol=1e-12)


def test_thermal_evolution_and_downstream_noise_have_separate_owners():
    """A bath changes quantum populations while HEMT noise changes only a supplied readout model."""
    first_chip, out, _, first = wired_result(noise=1.)
    _, second_out, _, second = wired_result(noise=3.)
    np.testing.assert_allclose(first.measure("r").probabilities, second.measure("r").probabilities, atol=1e-12)
    receiver = IQReceiver(1000)
    first_detector = first.iq_readout(out, means=[-.1, .1], frequency=6., receiver=receiver)
    second_detector = second.iq_readout(second_out, means=[-.1, .1], frequency=6., receiver=receiver)
    assert second_detector.iq_covariance[0, 0, 0] > first_detector.iq_covariance[0, 0, 0]
    r = first_chip["r"]
    r.T1 = 10.
    r.thermal_occupation = .2
    warm = QuantumSequence(first_chip).simulate(tlist=[0., 10.])
    assert warm.solver == "mesolve"
    assert warm.measure(r).probabilities[1] > .05


def test_wiring_noise_retains_differentiated_gain():
    """Resolved amplifier gain remains differentiable in the captured readout calculation."""
    import jax
    import jax.numpy as jnp
    from quchip.analysis.field_noise import ReadoutWiring
    chip, out, _, _ = wired_result(backend="dynamiqs")
    def variance(gain):
        candidate = chip.with_params({"network.component.hemt.gain": gain})
        wiring = ReadoutWiring.capture(candidate.resolve().slh, candidate.backend.array_module)
        return wiring.iq_readout(out, means=[-.1, .1], frequency=6., receiver=IQReceiver(1000)).iq_covariance[0, 0, 0]
    v, g = jax.jit(jax.value_and_grad(variance))(jnp.asarray(100.))
    np.testing.assert_allclose([v, g], [(37.375+.15+1)/2000, .25*1.5/2000], atol=1e-12)


def test_custom_detector_basis_can_be_differentiated_after_qutip_evolution():
    """A numerical detector rotation is differentiable even when quantum preparation used QuTiP."""
    import jax
    import jax.numpy as jnp
    _, q, result = preparation()
    def probability(theta):
        rotation = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
        return result.measure(q, basis={q: rotation}).probabilities[1]
    v, g = jax.jit(jax.value_and_grad(probability))(.4)
    np.testing.assert_allclose([v, g], [.3+.4*np.sin(.4)**2, .4*np.sin(.8)], atol=1e-12)
    batch = SimulationBatchResult([result, result])
    def batch_probability(theta):
        rotation = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
        return jnp.sum(batch.measure(q, basis=rotation).probabilities[:, 1])
    np.testing.assert_allclose(jax.jit(jax.value_and_grad(batch_probability))(.4), [2*v, 2*g], atol=1e-12)


def test_bandpass_noise_and_partitioned_wiring_forwarding():
    """A passive bandpass rejects the supplied signal and replaces its loss with declared thermal noise."""
    q = DuffingTransmon(freq=5., anharmonicity=-.2, levels=2, label="q")
    r = Resonator(freq=6., levels=2, label="r")
    net = PortNetwork()
    p = net.port("p", target=r, rate=.04)
    circ = net.circulator("circ")
    filt = net.filter("bandpass", transfer=lambda f: np.where((f >= 4) & (f <= 8), 1., 0.),
                      thermal_occupation=.2)
    amp = net.amplifier("hemt", gain=100., added_noise=1.)
    net.link(p, circ.port(2))
    net.link(circ.port(3), filt, amp)
    net.expose("drive", at=circ.port(1))
    out = net.expose("out", at=amp.port(2))
    result = QuantumSequence(Chip([q, r], port_network=net, frame="rotating")).simulate(
        tlist=[0., 1.])
    receiver = IQReceiver(1000)
    passing = result.iq_readout(out, means=[-.1, .1], frequency=6., receiver=receiver)
    stopped = result.iq_readout(out, means=[-.1, .1], frequency=9., receiver=receiver)
    np.testing.assert_allclose(passing.means, [-1, 1])
    np.testing.assert_allclose(stopped.means, [0, 0])
    # Colored spectra use numerical integration at the receiver's declared accuracy.
    np.testing.assert_allclose(stopped.iq_covariance[0]-passing.iq_covariance[0], np.eye(2)*20/2000,
                               rtol=receiver.tolerance, atol=1e-12)
    assert result.measure(q).sample(10, readout=passing).iq.shape == (10,)


def test_detector_can_resolve_wiring_without_quantum_evolution(monkeypatch):
    """Direct wiring calibration and saved-result calibration share the same physical calculation."""
    chip, out, _, result = wired_result()
    def forbidden(*args, **kwargs):
        raise AssertionError("Detector construction invoked quantum evolution")
    monkeypatch.setattr(chip.backend, "sesolve", forbidden)
    monkeypatch.setattr(chip.backend, "mesolve", forbidden)
    receiver = IQReceiver(1000)
    direct = IQReadout.from_wiring(chip, out, means=[-.1, .1], frequency=6., receiver=receiver)
    saved = result.iq_readout(out, means=[-.1, .1], frequency=6., receiver=receiver)
    np.testing.assert_allclose(direct.means, saved.means)
    np.testing.assert_allclose(direct.iq_covariance, saved.iq_covariance)
    chip.disconnect_network()
    np.testing.assert_allclose(direct.means, [-.5, .5])
