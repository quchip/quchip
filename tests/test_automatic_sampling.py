"""Automatic grids resolve physical and pulse scales without changing explicit grids."""

import numpy as np
import pytest

from quchip import Chip, DuffingTransmon, QuantumSequence


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_stationary_rotating_qubit_needs_only_interval_endpoints(backend):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    problem = QuantumSequence(Chip([q], backend=backend, frame="rotating")).build_problem(duration=1000.0)
    np.testing.assert_array_equal(problem.tlist, [0.0, 1000.0])


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_automatic_idle_grid_resolves_fast_population_exchange(backend):
    from quchip.declarative import DeviceModel, parameter

    class Exchange(DeviceModel):
        rate: float = parameter(default=9.7)

        def local_hamiltonian(self, op, p):
            return p.rate * (op.a + op.adag)

    q = Exchange(label="q", levels=2)
    q.reference_freq = 0.0
    chip = Chip([q], backend=backend, frame="lab")
    options = {"rtol": 1e-9, "atol": 1e-11}
    if backend == "dynamiqs":
        import dynamiqs as dq
        options = {"method": dq.method.Tsit5(rtol=1e-9, atol=1e-11)}
    result = QuantumSequence(chip).simulate(options=options, duration=1.0, initial_state=chip.backend.basis(2, 0),
                                           partition=False, e_ops={"q": q.number_operator()})
    times = np.asarray(result.times)
    assert np.max(np.diff(times)) <= 1 / (32 * 2 * q.rate)
    np.testing.assert_allclose(result.expect("q"),
                               np.sin(2 * np.pi * q.rate * times) ** 2, atol=2e-5)
    query = np.linspace(0.0, 1.0, 4001)
    interpolated = np.interp(query, times, np.real(result.expect("q")))
    np.testing.assert_allclose(interpolated, np.sin(2 * np.pi * q.rate * query) ** 2, atol=0.005)


def test_fast_decay_is_sampled_without_a_hamiltonian_frequency():
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=0.001, label="q")
    chip = Chip([q], frame="rotating")
    sequence = QuantumSequence(chip)
    problem = sequence.build_problem(duration=0.01, initial_state={"q": 1})
    assert np.max(np.diff(problem.tlist)) <= q.T1 / 16
    closed = sequence.build_problem(duration=0.01, dissipation=False)
    np.testing.assert_array_equal(closed.tlist, [0.0, 0.01])


def test_narrow_gaussian_features_are_local_to_pulse():
    from quchip import ChargeDrive, Gaussian

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    chip = Chip([q], frame="rotating")
    drive = ChargeDrive(q)
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(drive, envelope=Gaussian(duration=0.01, sigmas=12.0, amplitude=0.01),
                      start_time=500.0, freq=5.0)
    times = np.asarray(sequence.build_problem(duration=1000.0).tlist)
    assert len(times) < 1000
    assert sum((times > 500.0) & (times < 500.01)) >= 60
    assert sum(times < 500.0) <= 2


def test_amplitude_sweep_uses_one_grid_without_growth_per_point():
    from quchip import ChargeDrive, Square

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    chip = Chip([q], frame="rotating")
    drive = ChargeDrive(q)
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule(drive, envelope=Square(duration=10.0, amplitude=0.01), freq=5.0)
    amplitudes = np.linspace(0.01, 0.2, 20)
    many = sequence.build_batch(pulse.vary("amplitude", amplitudes))
    largest = sequence.build_batch(pulse.vary("amplitude", amplitudes[-1:]))
    assert many.has_shared_tlist
    np.testing.assert_array_equal(many.tlist, largest.tlist)


def test_explicit_grid_keeps_its_values_despite_fast_dynamics():
    q = DuffingTransmon(freq=20.0, anharmonicity=-0.2, levels=3, label="q")
    times = np.array([0.0, 0.2, 3.0])
    problem = QuantumSequence(Chip([q], frame="lab")).build_problem(times)
    np.testing.assert_array_equal(problem.tlist, times)


def test_sparse_spectral_span_covers_off_diagonal_and_ignores_energy_origin():
    from scipy import sparse
    from quchip.engine.ir import CanonicalOperator, StaticTerm
    from quchip.engine.solver_hints import _static_spectral_span

    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    metadata = dict(dims=(2,), basis="fock", subsystem_labels=("q",))
    csr, dia = sparse.csr_matrix(matrix), sparse.dia_matrix(matrix)
    operators = [CanonicalOperator.from_dense(matrix, **metadata),
                 CanonicalOperator.from_csr(csr.data, csr.indices, csr.indptr, shape=csr.shape, **metadata),
                 CanonicalOperator.from_dia(dia.data, dia.offsets, shape=dia.shape, **metadata)]
    for operator in operators:
        shift = CanonicalOperator.from_dense(np.eye(2) * 1000.0, dims=(2,), basis="fock", subsystem_labels=("q",))
        assert _static_spectral_span((StaticTerm(operator), StaticTerm(shift))) == pytest.approx(2.0)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_narrow_gaussian_trace_matches_integrated_pulse_area(backend):
    from scipy.special import erf
    from quchip import ChargeDrive, Gaussian

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    chip = Chip([q], backend=backend, frame="rotating")
    drive = ChargeDrive(q)
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    start, duration, sigmas, amplitude = 3.0, 0.1, 12.0, 50.0
    sequence.schedule(drive, envelope=Gaussian(duration=duration, sigmas=sigmas, amplitude=amplitude),
                      start_time=start, freq=5.0)
    result = sequence.simulate(duration=7.0, states="none",
                               e_ops={"q": q.number_operator()})
    times = np.asarray(result.times)
    local = np.clip(times - start, 0, duration)
    sigma = duration / (2 * sigmas)
    area = amplitude * sigma * np.sqrt(np.pi / 2) * (
        erf((local - duration / 2) / (np.sqrt(2) * sigma)) + erf(sigmas / np.sqrt(2))
    )
    np.testing.assert_allclose(result.expect("q"), np.sin(np.pi * area) ** 2, atol=0.002)
    assert len(times) < 500


def test_readout_demodulation_is_sampled_even_when_solver_state_is_stationary():
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    q.reference_freq = 0.0
    chip = Chip([q], frame={"q": 5.0})
    initial = (chip.backend.basis(2, 0) + chip.backend.basis(2, 1)) / np.sqrt(2)
    result = QuantumSequence(chip).simulate(duration=1.0, initial_state=initial,
                                           e_ops=chip.e_ops(q="X"))
    assert len(result.times) >= 161
    np.testing.assert_allclose(result.expect("q"), np.cos(2 * np.pi * 5 * result.times), atol=1e-7)


def test_carrier_hint_uses_product_and_power_frequencies():
    from quchip.engine.ir import CanonicalOperator, Carrier, DynamicTerm, Multiply, ScalarModulation, SignalPower
    from quchip.engine.solver_hints import _max_abs_carrier_freq

    operator = CanonicalOperator.from_dense(np.eye(2), dims=(2,), basis="fock", subsystem_labels=("q",))
    for signal, expected in [(Multiply((Carrier(3.0), Carrier(5.0))), 8.0),
                             (SignalPower(Carrier(3.0), 2), 6.0)]:
        term = DynamicTerm(operator, ScalarModulation(signal))
        assert _max_abs_carrier_freq((term,)) == pytest.approx(expected)


def test_large_sparse_spectral_bound_does_not_need_dense_materialization(monkeypatch):
    from quchip.engine.ir import CanonicalOperator, StaticTerm
    from quchip.engine.solver_hints import _static_spectral_span

    size = 4096
    operator = CanonicalOperator.from_dia(np.ones((2, size)), np.array([-1, 1]), shape=(size, size),
                                         dims=(size,), basis="fock", subsystem_labels=("q",))
    def reject_dense(self):
        raise AssertionError("Sampling must not materialize the global sparse Hamiltonian")
    monkeypatch.setattr(CanonicalOperator, "to_dense", reject_dense)
    assert _static_spectral_span((StaticTerm(operator),)) == pytest.approx(4.0)


def test_automatic_grid_can_be_reused_for_differentiation():
    import jax
    import jax.numpy as jnp
    from quchip import ChargeDrive, Square

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    chip = Chip([q], backend="dynamiqs", frame="rotating")
    drive = ChargeDrive(q)
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(drive, envelope=Square(duration=3.0, amplitude=0.07), freq=5.0)
    times = sequence.build_problem().tlist

    @jax.jit
    @jax.value_and_grad
    def objective(amplitude):
        variant = sequence.with_params({"pulse.0.amplitude": amplitude})
        result = variant.simulate(tlist=times, e_ops={"q": q.number_operator()}, states="none",
                                  partition=False)
        return jnp.real(result.expect("q")[-1])

    value, gradient = objective(jnp.asarray(0.07))
    assert value == pytest.approx(np.sin(np.pi * 0.07 * 3) ** 2, abs=2e-6)
    assert gradient == pytest.approx(3 * np.pi * np.sin(2 * np.pi * 0.07 * 3), abs=2e-5)


def test_partitioned_automatic_grid_supports_simultaneous_correlations():
    from quchip import ChargeDrive, Square
    from quchip.results.partitioned import PartitionedSimulationResult

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q0")
    q1 = DuffingTransmon(freq=6.0, anharmonicity=-0.2, levels=2, label="q1")
    chip = Chip([q0, q1], frame="rotating")
    chip.wire(ChargeDrive(q0), ChargeDrive(q1))
    sequence = QuantumSequence(chip)
    sequence.charge(q0, envelope=Square(duration=10.0, amplitude=0.07))
    sequence.charge(q1, envelope=Square(duration=10.0, amplitude=0.13))
    e_ops = chip.e_ops(correlators={("q0", "q1"): ("Z", "Z")})
    result = sequence.simulate(e_ops=e_ops)
    assert isinstance(result, PartitionedSimulationResult)
    for component in result.components:
        np.testing.assert_array_equal(component.times, result.times)
    expected = np.cos(2 * np.pi * 0.07 * result.times) * np.cos(2 * np.pi * 0.13 * result.times)
    np.testing.assert_allclose(result.expect(("q0", "q1")), expected, atol=2e-5)
    joint = sequence.simulate(tlist=result.times, e_ops=e_ops, partition=False)
    np.testing.assert_allclose(result.expect(("q0", "q1")), joint.expect(("q0", "q1")), atol=2e-5)


def test_native_cutoff_reconstruction_is_sampled_in_rotating_frame():
    from quchip.declarative import DeviceModel, parameter

    class Exchange(DeviceModel):
        rate: float = parameter(default=9.7)

        def local_hamiltonian(self, op, p):
            return p.rate * (op.a + op.adag)

    q = Exchange(label="q", levels=2)
    chip = Chip([q], frame="rotating")
    from quchip import with_truncation
    from quchip.engine import solve_problem
    from quchip.engine.sampling import automatic_tlist

    problem = with_truncation(QuantumSequence(chip).build_problem(
        [0.0, 1.0], initial_state=chip.backend.basis(2, 0), states="none"))
    from dataclasses import replace
    result = solve_problem(replace(problem, tlist=automatic_tlist([problem])))
    assert result.check_truncation(threshold=2.0)["q"] > 0.997
    np.testing.assert_allclose(result._boundary_traces[0],
                               np.sin(2 * np.pi * q.rate * result.times) ** 2, atol=1e-6)


def test_requested_output_carrier_is_resolved_in_rotating_frame():
    from quchip import PortNetwork, Resonator

    mode = Resonator(freq=5.0, levels=2, label="r")
    network = PortNetwork(label="line")
    network.port("coupler", target=mode, rate=0.001)
    chip = Chip([mode], port_network=network, frame="rotating")
    plane = network.external_port("coupler")
    initial = (chip.backend.basis(2, 0) + chip.backend.basis(2, 1)) / np.sqrt(2)
    result = QuantumSequence(chip).simulate(duration=1.0, initial_state=initial,
                                           e_ops={plane: plane.output}, states="none")
    assert len(result.times) >= 161
    expected = np.sqrt(0.001) / 2 * np.exp((-0.001 / 2 - 2j * np.pi * 5) * result.times)
    np.testing.assert_allclose(result.output(plane).amplitude, expected, atol=1e-8)


def test_narrow_output_pulse_survives_reference_plane_delays():
    from quchip import PortNetwork, Resonator, Square

    mode = Resonator(freq=0.2, levels=4, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=mode, rate=0.001)
    cable = network.delay("cable", duration=1.0)
    network.link(port, cable)
    plane = network.expose("readout", at=cable.port(2))
    chip = Chip([mode], port_network=network, frame="rotating")
    sequence = QuantumSequence(chip)
    sequence.schedule(plane.input, envelope=Square(duration=0.01, amplitude=0.1))
    result = sequence.simulate(duration=3.0, e_ops={plane: plane.output},
                               states="none")
    field = result.output(plane)
    assert len(result.times) < 1000
    assert np.max(field.photon_flux) == pytest.approx(0.01, abs=1e-6)
    peak_time = np.asarray(result.times)[np.argmax(field.photon_flux)]
    assert 2.0 <= peak_time <= 2.01


def test_excessive_automatic_grid_explains_an_explicit_choice():
    q = DuffingTransmon(freq=50.0, anharmonicity=-0.2, levels=2, label="q")
    with pytest.raises(ValueError, match="Automatic sampling would require.*explicit tlist"):
        QuantumSequence(Chip([q], frame="lab")).build_problem(duration=1000.0)


def test_value_dependent_automatic_grid_explains_the_jax_boundary():
    import jax
    import jax.numpy as jnp

    @jax.jit
    def build(frequency):
        q = DuffingTransmon(freq=frequency, anharmonicity=-0.2, levels=2, label="q")
        return QuantumSequence(Chip([q], backend="dynamiqs", frame="lab")).build_problem(duration=1.0).tlist

    with pytest.raises(ValueError, match="Automatic sampling requires concrete"):
        build(jnp.asarray(5.0))


def test_single_point_frequency_sweep_resolves_its_own_auto_frame():
    from quchip import ChargeDrive, Square

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    chip = Chip([q], frame="auto")
    drive = ChargeDrive(q)
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule(drive, envelope=Square(duration=10.0, amplitude=0.2), freq=5.0)
    batch = sequence.build_batch(pulse.vary("freq", [6.0]))
    independent = sequence.with_params({"pulse.0.freq": 6.0}).build_problem()
    assert batch.element(0).resolved_frame.frequencies == independent.resolved_frame.frequencies
    np.testing.assert_array_equal(batch.tlist, independent.tlist)


def test_custom_envelope_can_supply_narrow_feature_times():
    from quchip import ChargeDrive, Envelope
    from quchip.declarative import parameter, qnp

    class Peak(Envelope):
        duration: float = parameter(default=1.0, positive=True)
        amplitude: float = parameter(default=0.01)

        def value(self, t):
            return self.amplitude * qnp.exp(-((t - 0.123) / 0.0001) ** 2)

        def sampling_times(self):
            return qnp.linspace(0.1224, 0.1236, 65)

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
    chip = Chip([q], frame="rotating")
    drive = ChargeDrive(q)
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(drive, envelope=Peak(), freq=5.0)
    times = np.asarray(sequence.build_problem().tlist)
    assert sum((times > 0.1224) & (times < 0.1236)) >= 63
    assert len(times) < 200


def test_delayed_emission_resolves_an_ordinary_drive_transition():
    from quchip import ChargeDrive, PortNetwork, Resonator, Square

    mode = Resonator(freq=0.2, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=mode, rate=0.001)
    cable = network.delay("cable", duration=1.0)
    network.link(port, cable)
    plane = network.expose("readout", at=cable.port(2))
    chip = Chip([mode], port_network=network, frame="rotating")
    drive = ChargeDrive(mode)
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(drive, envelope=Square(duration=0.01, amplitude=25), start_time=0.3, freq=0.2)
    options = dict(e_ops={plane: plane.output}, states="none")
    result = sequence.simulate(duration=3.0, **options)
    dense = sequence.simulate(tlist=np.linspace(0, 3, 6001), **options)
    field = result.output(plane).amplitude
    interpolated = np.interp(dense.times, result.times, field)
    np.testing.assert_allclose(interpolated, dense.output(plane).amplitude, atol=2e-4)
    assert sum((result.times > 1.3) & (result.times < 1.31)) >= 8
    assert len(result.times) < 1000


def test_delayed_initial_emission_preserves_zero_fill_onset():
    from quchip import PortNetwork, Resonator

    mode = Resonator(freq=0.2, levels=2, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=mode, rate=0.001)
    cable = network.delay("cable", duration=1.0)
    network.link(port, cable)
    plane = network.expose("readout", at=cable.port(2))
    chip = Chip([mode], port_network=network, frame="rotating")
    initial = (chip.backend.basis(2, 0) + chip.backend.basis(2, 1)) / np.sqrt(2)
    result = QuantumSequence(chip).simulate(duration=3.0, initial_state=initial,
                                            e_ops={plane: plane.output}, states="none")
    assert 1.0 in result.times
    assert np.nextafter(1.0, 0.0) in result.times
    field = result.output(plane).amplitude
    np.testing.assert_array_equal(field[result.times < 1.0], 0.0)
    assert abs(field[np.flatnonzero(result.times == 1.0)[0]]) == pytest.approx(np.sqrt(0.001) / 2)
