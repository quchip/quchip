"""An excitation cutoff keeps P H P and P L P on the product labels with N <= K."""

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from quchip import (
    RWA,
    VNA,
    Capacitive,
    ChargeDrive,
    Chip,
    DuffingTransmon,
    Exact,
    ExcitationCutoff,
    Gaussian,
    PortNetwork,
    QuantumSequence,
    Resonator,
    check_truncation,
    eliminate,
    fit_a_dress,
    with_truncation,
)
from quchip.engine.excitation import excitation_classes


def ladder(levels):
    """Return a truncated harmonic lowering operator."""
    return np.diag(np.sqrt(np.arange(1, levels)), 1)


def retained(dims, max_total):
    """Return the product labels with N <= max_total and their row-major product indices."""
    levels = np.array(np.unravel_index(np.arange(int(np.prod(dims))), dims)).T
    keep = np.flatnonzero(levels.sum(axis=1) <= max_total)
    return [tuple(int(n) for n in row) for row in levels[keep]], keep


def assigned(matrix, labels, label):
    """Return the eigenpair with the largest weight on ``label``, with a real nonnegative overlap."""
    values, vectors = np.linalg.eigh(matrix)
    row = labels.index(label)
    column = int(np.argmax(np.abs(vectors[row]) ** 2))
    vector = vectors[:, column]
    return values[column], vector * np.conj(vector[row]) / abs(vector[row])


def pair(*, loss=False, **options):
    """Return a transmon and a resonator with a counter-rotating charge coupling and one drive."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q",
                        **({"T1": 400.0, "thermal_occupation": 0.05} if loss else {}))
    r = Resonator(freq=7.0, levels=5, label="r", **({"T1": 150.0} if loss else {}))
    chip = Chip([q, r], [Capacitive(q, r, g=0.1, label="qr")], approximation=Exact(), **options)
    chip.wire(ChargeDrive(q, label="xy"))
    return chip


def pair_matrices():
    """Return the independent lab-frame Hamiltonian of ``pair`` in GHz and the transmon charge operator."""
    b, a = np.kron(ladder(4), np.eye(5)), np.kron(np.eye(4), ladder(5))
    nq, nr = b.T @ b, a.T @ a
    charge_q, charge_r = 1j * (b.T - b), 1j * (a.T - a)
    hamiltonian = 5.0 * nq - 0.125 * nq @ (nq - np.eye(20)) + 7.0 * nr + 0.1 * charge_q @ charge_r
    return hamiltonian, charge_q


def tolerances(backend):
    """Return tight integrator options for ``backend``."""
    if backend == "dynamiqs":
        import dynamiqs as dq

        return {"method": dq.method.Tsit5(rtol=1e-10, atol=1e-12)}
    return {"atol": 1e-12, "rtol": 1e-10}


def schedule(chip):
    """Return a sequence with one Gaussian pulse near the transmon frequency."""
    sequence = QuantumSequence(chip)
    sequence.schedule("xy", envelope=Gaussian(duration=16.0, sigmas=3.0, amplitude=0.06), freq=4.99)
    return sequence


def lindblad_reference(problem, keep, rho0, tlist):
    """Integrate the sliced product-space generator of ``problem`` with an independent integrator."""
    engine = problem.engine_result
    window = np.ix_(keep, keep)
    static = sum(term.coefficient * term.operator.to_dense() for term in engine.static_terms)[window]
    dynamic = [(term.operator.to_dense()[window], term.time_dependence.signal) for term in engine.dynamic_terms]
    jumps = [(1 if term.phase is None else np.exp(1j * term.phase)) * np.sqrt(term.rate)
             * term.operator.to_dense()[window] for term in engine.collapse_terms]
    size = keep.size

    def rhs(t, y):
        rho = y.reshape(size, size)
        h = static + sum(np.asarray(signal.evaluate(t, xp=np)) * op for op, signal in dynamic)
        out = -1j * (h @ rho - rho @ h)
        for jump in jumps:
            decay = jump.conj().T @ jump
            out += jump @ rho @ jump.conj().T - 0.5 * (decay @ rho + rho @ decay)
        return out.reshape(-1)

    solution = solve_ivp(rhs, (tlist[0], tlist[-1]), rho0.reshape(-1), t_eval=tlist, rtol=1e-9, atol=1e-11,
                         method="DOP853")
    return [solution.y[:, index].reshape(size, size) for index in range(len(tlist))]


@pytest.mark.unit
def test_policy_validates_and_survives_copies_rebinding_and_serialization():
    """The policy rejects invalid budgets and stays on every chip derived from a limited chip."""
    for value in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="positive integer"):
            ExcitationCutoff(value)
    policy = ExcitationCutoff(np.int64(2))
    assert type(policy.max_total) is int and ExcitationCutoff.from_dict(policy.to_dict()) == policy
    chip = pair()
    limited = chip.with_space(policy)
    assert chip.space is None and limited.space == policy and limited.with_space(None).space is None
    for derived in (limited.clone(), limited.with_params({"qr.g": 0.05}), Chip.from_dict(limited.to_dict())):
        assert derived.space == policy
    assert Chip.from_dict(chip.to_dict()).space is None
    assert "space=ExcitationCutoff(max_total=2)" in repr(limited)
    with pytest.raises(TypeError, match="space must be"):
        chip.with_space(2)


@pytest.mark.unit
def test_static_changes_split_the_budget_into_classes():
    """Changes of two give parity classes, and unknown changes keep one class."""
    assert excitation_classes(frozenset({-1, 0, 1}), 3) == ((0, 1, 2, 3),)
    assert excitation_classes(frozenset({-2, 0, 2}), 3) == ((0, 2), (1, 3))
    assert excitation_classes(None, 2) == ((0, 1, 2),)


@pytest.mark.unit
def test_cutoff_that_keeps_every_label_leaves_results_unchanged():
    """A budget of at least sum_i (d_i - 1) is inactive, so the full dressed spectrum stays available."""
    chip = pair()
    same = chip.with_space(ExcitationCutoff(7))
    np.testing.assert_array_equal(np.asarray(same.dress().eigenvalues), np.asarray(chip.dress().eigenvalues))
    assert float(same.freq("q")) == float(chip.freq("q"))


@pytest.mark.unit
def test_labels_and_states_outside_the_cutoff_raise():
    """quchip never projects or renormalizes support above the cutoff."""
    chip = pair().with_space(ExcitationCutoff(1))
    for query in (lambda: chip.energy({"q": 1, "r": 1}), chip.kerr_matrix, lambda: chip.compact_state({"q": 2})):
        with pytest.raises(ValueError, match="outside the excitation cutoff"):
            query()
    with pytest.raises(ValueError, match="not available under an excitation cutoff"):
        chip.dress()
    sequence = QuantumSequence(chip)
    for state in ({"q": 1, "r": 1}, chip.with_space(None).bare_state({"r": 2})):
        with pytest.raises(ValueError, match="outside the excitation cutoff"):
            sequence.simulate(tlist=[0.0, 1.0], initial_state=state)


def _ported_pair():
    """Return ``pair`` at K = 1 with an external port on the resonator."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    r = Resonator(freq=7.0, levels=5, label="r")
    network = PortNetwork(label="line")
    network.port("out", target=r, rate=0.01)
    chip = Chip([q, r], [Capacitive(q, r, g=0.1, label="qr")], approximation=Exact(), port_network=network,
                space=ExcitationCutoff(1))
    chip.wire(ChargeDrive(q, label="xy"))
    return chip


@pytest.mark.unit
@pytest.mark.parametrize("workflow", [
    lambda chip: VNA(chip).sweep(7.0),
    lambda chip: eliminate(chip, "r"),
    lambda chip: chip.partition(),
    lambda chip: QuantumSequence(chip).active_patch(),
    lambda chip: QuantumSequence(chip).simulate(tlist=[0.0, 1.0], solver="mcsolve"),
    lambda chip: fit_a_dress(chip, constraints={"q": {"freq": 5.0}}, vary={"q": ["freq"]}, evaluator="local"),
], ids=["vna_sweep", "eliminate", "partition", "active_patch", "mcsolve", "local_fit"])
def test_workflows_without_cutoff_support_raise(workflow):
    """Each excluded workflow raises and names the full product space instead of ignoring the cutoff."""
    with pytest.raises(ValueError, match=r"with_space\(None\)"):
        workflow(_ported_pair())


def test_counter_rotating_spectrum_matches_the_truncated_matrix():
    """Each cutoff diagonalizes P H P with its counter-rotating elements, and the frequency need not be monotone."""
    a, b = Resonator(freq=5.0, levels=3, label="a"), Resonator(freq=6.0, levels=3, label="b")
    chip = Chip([a, b], [Capacitive(a, b, g=0.1)], approximation=Exact())
    lower_a, lower_b = np.kron(ladder(3), np.eye(3)), np.kron(np.eye(3), ladder(3))
    hamiltonian = (5.0 * lower_a.T @ lower_a + 6.0 * lower_b.T @ lower_b
                   + 0.1 * (lower_a + lower_a.T) @ (lower_b + lower_b.T))
    frequencies = []
    for max_total in (1, 2, 3):
        labels, keep = retained((3, 3), max_total)
        block = hamiltonian[np.ix_(keep, keep)]
        vacuum = assigned(block, labels, (0, 0))[0]
        limited = chip.with_space(ExcitationCutoff(max_total))
        for device, label in (("a", (1, 0)), ("b", (0, 1))):
            assert float(limited.freq(device)) == pytest.approx(assigned(block, labels, label)[0] - vacuum, abs=1e-12)
        frequencies.append(float(limited.freq("a")))
    # At K = 1 no retained label pairs with the vacuum, so the exchange doublet alone sets the frequency.
    assert frequencies[0] == pytest.approx(5.5 - math.sqrt(0.5**2 + 0.1**2), abs=1e-12)
    assert frequencies[0] < frequencies[1] > frequencies[2]


def test_dressed_statics_and_states_use_the_truncated_eigenvectors():
    """Energies, Kerr terms, drive elements, and states match an independent diagonalization of P H P."""
    chip = pair()
    hamiltonian, charge = pair_matrices()
    bare = Chip([DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")])
    bare.wire(ChargeDrive(bare["q"], label="xy"))
    unit = complex(bare.drive_matrix_elements("q")["xy"])
    for max_total in (2, 3):
        limited = chip.with_space(ExcitationCutoff(max_total))
        labels, keep = retained((4, 5), max_total)
        window = np.ix_(keep, keep)
        pairs = {label: assigned(hamiltonian[window], labels, label)
                 for label in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1))}
        energy = {label: value for label, (value, _) in pairs.items()}
        assert float(limited.energy({"q": 1})) == pytest.approx(energy[1, 0], abs=1e-11)
        assert float(limited.freq("q")) == pytest.approx(energy[1, 0] - energy[0, 0], abs=1e-11)
        assert float(limited.dressed_anharmonicity("q")) == pytest.approx(
            energy[2, 0] - 2 * energy[1, 0] + energy[0, 0], abs=1e-11)
        assert float(limited.dispersive_shift("q", "r")) == pytest.approx(
            energy[1, 1] - energy[1, 0] - energy[0, 1] + energy[0, 0], abs=1e-11)
        # The drive element scales with the charge element, whose bare value is <1|n|0> = i.
        element = pairs[1, 0][1].conj() @ charge[window] @ pairs[0, 0][1] / 1j
        assert complex(limited.drive_matrix_elements("q")["xy"]) == pytest.approx(unit * element, abs=1e-11)
        state = np.asarray(limited.backend.to_array(limited.state({"q": 1}))).reshape(-1)
        expected = np.zeros(20, dtype=complex)
        expected[keep] = pairs[1, 0][1]
        np.testing.assert_allclose(state, expected, atol=1e-10)
        compact = limited.compact_state({"q": 1}, dressed=True)
        coordinates = [pairs[1, 0][1][labels.index(tuple(int(n) for n in level))] for level in compact.space.levels]
        np.testing.assert_allclose(np.asarray(compact.values).reshape(-1), coordinates, atol=1e-10)
    with pytest.raises(ValueError, match="requires proven static excitation conservation"):
        chip.compact_state({"q": 1}, dressed=True)


def test_driven_dissipative_solve_matches_a_sliced_lindblad_reference():
    """A pulse with loss and heating evolves P H(t) P with jumps P L P, and the boundary population is reported."""
    chip = pair(loss=True, frame={"q": 5.0, "r": 7.0})
    limited = chip.with_space(ExcitationCutoff(3))
    tlist = np.linspace(0.0, 16.0, 33)
    result = schedule(limited).simulate(tlist=tlist, solver="mesolve", options=tolerances("qutip"))
    stats = result.stats["excitation_reduction"]
    assert (stats["cutoff"], stats["cutoff_dimension"]) == (3, 10)
    # The reference integrates quchip's own product-space terms, so it checks the projection and the
    # integration, not the drive and rate conventions.
    reference_problem = schedule(chip).build_problem(tlist, solver="mesolve", initial_state={"q": 0})
    labels, keep = retained((4, 5), 3)
    hamiltonian, _ = pair_matrices()
    # The default state is the vacuum-assigned eigenvector of the truncated lab-frame static model.
    vacuum = assigned(hamiltonian[np.ix_(keep, keep)], labels, (0, 0))[1]
    references = lindblad_reference(reference_problem, keep, np.outer(vacuum, vacuum.conj()), tlist)
    space = result._excitation_space
    order = [space.rows[label] for label in labels]
    boundary = [index for index, label in enumerate(labels) if sum(label) in stats["boundary"]]
    sampled = []
    for time, reference in zip(tlist, references):
        state = np.asarray(result.compact_state(time).values)
        rho = (state @ state.conj().T if state.shape[1] == 1 else state)[np.ix_(order, order)]
        # QuTiP interpolates the windowed envelope linearly at 40 samples per ns. The 2e-6 difference falls
        # with the square of that spacing and is the same without a cutoff.
        np.testing.assert_allclose(rho, reference, atol=1e-5)
        assert np.trace(rho).real == pytest.approx(1, abs=1e-8)
        assert np.linalg.eigvalsh(rho).min() > -1e-9
        sampled.append(np.trace(rho[np.ix_(boundary, boundary)]).real)
    observed = check_truncation(result, threshold=1.0)[ExcitationCutoff(3)]
    assert float(observed) == pytest.approx(max(sampled), abs=1e-10)
    with pytest.warns(UserWarning, match="not an error bound"):
        check_truncation(result, threshold=0.0)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_cutoff_check_finds_the_boundary_rows_when_it_runs(backend):
    """A request keeps the cutoff and its boundary totals, but no compact rows or matrices.

    The check reads the boundary sectors from the saved diagonals, and with_truncation samples the same
    projector during the solve.
    """
    chip = pair(loss=True, backend=backend, space=ExcitationCutoff(3))
    problem = schedule(chip).build_problem(np.linspace(0.0, 16.0, 9), solver="mesolve", options=tolerances(backend))
    plan = problem.truncation
    # The counter-rotating coupling raises N by 2, and the drive and the heating raise it by 1.
    assert plan.excitation == (3, problem.engine_result.metadata["excitation_reduction"]["boundary"]) == (3, (2, 3))
    assert plan.space is problem.engine_result.excitation_space
    assert (plan.operators, plan.metadata, plan.sampled) == ((), (), False)
    saved = check_truncation(chip.solve(problem), threshold=1.0)
    sampled = check_truncation(chip.solve(with_truncation(replace(problem, states="none"))), threshold=1.0)
    assert saved.keys() == sampled.keys() == {"q", "r", ExcitationCutoff(3)}
    for key, value in saved.items():
        assert float(sampled[key]) == pytest.approx(float(value), abs=1e-12)
    assert float(saved[ExcitationCutoff(3)]) > 1e-2


def test_cascade_at_the_one_excitation_cutoff_matches_the_exact_space():
    """With and without dissipation, K = 1 reproduces the exact one-excitation solve of a cascade."""
    r1, r2 = Resonator(freq=6.0, levels=3, label="r1"), Resonator(freq=6.0, levels=3, label="r2")
    network = PortNetwork(label="line")
    rate = 2 * math.pi * 0.01
    first, second = network.port("p1", target=r1, rate=rate), network.port("p2", target=r2, rate=rate)
    network.cascade(first, second)
    network.expose("out", input=first.input, output=second.output)
    chip = Chip([r1, r2], port_network=network)
    tlist = np.linspace(0.0, 40.0, 81)
    for dissipation in (False, True):
        unrestricted, limited = (
            QuantumSequence(target).simulate(tlist=tlist, initial_state={"r1": 1}, dissipation=dissipation,
                                             options=tolerances("qutip"))
            for target in (chip, chip.with_space(ExcitationCutoff(1))))
        np.testing.assert_allclose(limited.population("r2", 1), unrestricted.population("r2", 1), atol=1e-7)


def test_raising_ports_keep_the_exchange_through_a_label_above_the_cutoff():
    """P H_gen P keeps the |01> to |10> element of raising ports, which passes through |11>."""
    u, d = Resonator(freq=6.0, levels=2, label="u"), Resonator(freq=6.0, levels=2, label="d")
    network = PortNetwork(label="line")
    rate = 0.05
    first = network.port("pu", target=u, rate=rate, operator="a_dag")
    second = network.port("pd", target=d, rate=rate, operator="a_dag")
    network.cascade(first, second)
    network.expose("out", input=first.input, output=second.output)
    chip = Chip([u, d], port_network=network, space=ExcitationCutoff(1))
    tlist = np.linspace(0.0, 40.0, 41)
    result = QuantumSequence(chip).simulate(tlist=tlist, initial_state={"d": 1}, dissipation=False,
                                            options=tolerances("qutip"))
    np.testing.assert_allclose(result.population("u", 1), np.sin(rate * tlist / 2) ** 2, atol=1e-6)


def test_global_budget_couples_uncoupled_driven_devices():
    """At K = 1 two driven, uncoupled qubits form one V system, so the solve stays joint."""
    devices = [DuffingTransmon(freq=5.0 + 0.3 * i, anharmonicity=-0.25, levels=2, label=f"q{i}") for i in range(2)]
    chip = Chip(devices, approximation=RWA(), frame="rotating")
    chip.wire(*(ChargeDrive(device, label=f"xy{device.label[1]}") for device in devices))

    def excited(target):
        sequence = QuantumSequence(target)
        for device in devices:
            sequence.schedule(f"xy{device.label[1]}", envelope=Gaussian(duration=20.0, sigmas=3.0, amplitude=0.03),
                              freq=device.freq)
        result = sequence.simulate(tlist=[0.0, 20.0], options=tolerances("qutip"))
        return float(np.real(result.population("q0", 1)[-1]))

    # Each qubit alone turns by theta, so P = sin^2(theta / 2). The bright state turns by sqrt(2) theta.
    theta = 2 * math.asin(math.sqrt(excited(chip)))
    assert excited(chip.with_space(ExcitationCutoff(1))) == pytest.approx(math.sin(theta / math.sqrt(2)) ** 2 / 2,
                                                                          abs=1e-7)
    with pytest.raises(ValueError, match="partition"):
        chip.with_space(ExcitationCutoff(1)).partition()


def test_compact_replay_measurement_and_batches_keep_the_cutoff():
    """A saved compact state continues the solve, and each batch point equals its single solve."""
    chip = pair(loss=True, frame={"q": 5.0, "r": 7.0}, space=ExcitationCutoff(2))
    whole = QuantumSequence(chip).simulate(tlist=np.linspace(0.0, 20.0, 5), initial_state={"q": 1})
    first = QuantumSequence(chip).simulate(tlist=np.linspace(0.0, 10.0, 3), initial_state={"q": 1})
    replay = QuantumSequence(chip).simulate(tlist=np.linspace(10.0, 20.0, 3), initial_state=first.compact_state())
    np.testing.assert_allclose(replay.population("q", 1)[-1], whole.population("q", 1)[-1], atol=1e-7)
    native = QuantumSequence(chip).simulate(tlist=np.linspace(0.0, 20.0, 5),
                                            initial_state=chip.with_space(None).bare_state({"q": 1}))
    np.testing.assert_allclose(native.population("q", 1), whole.population("q", 1), atol=1e-12)
    probabilities = np.asarray(whole.measure("q").probabilities)
    assert np.sum(probabilities) == pytest.approx(1, abs=1e-8)
    sequence = schedule(chip)
    batch = sequence.simulate_batch(sequence.vary("qr.g", np.array([0.05, 0.1])), tlist=[0.0, 20.0], progress=False)
    singles = [schedule(chip.with_params({"qr.g": g})).simulate(tlist=[0.0, 20.0]).population("q", 1)[-1]
               for g in (0.05, 0.1)]
    np.testing.assert_allclose(np.asarray(batch.population("q", level=1, reduce="last")), singles, atol=1e-9)


def test_fit_recovers_a_bare_frequency_under_the_same_cutoff():
    """fit_a_dress fits the truncated model, keeps the policy, and bounds the class it diagonalizes."""
    source = pair()
    truth = source.with_params({"q.freq": 5.03}).with_space(ExcitationCutoff(3))
    target = float(truth.freq("q"))
    chip = source.with_space(ExcitationCutoff(3))
    constraints = {"q": {"freq": target, "anharmonicity": None}, "r": {"freq": None}, "qr": {"cross_kerr": None}}
    # The class of the N = 1 label holds N = 1 and N = 3: 2 + 4 product labels.
    result = fit_a_dress(chip, constraints=constraints, vary={"q": ["freq"]}, max_hilbert_dim=6)
    assert result.chip.space == ExcitationCutoff(3)
    assert float(result.chip.freq("q")) == pytest.approx(target, abs=1e-9)
    assert float(result.chip["q"].freq) == pytest.approx(5.03, abs=1e-8)
    with pytest.raises(ValueError, match="max_hilbert_dim=5"):
        fit_a_dress(chip, constraints=constraints, vary={"q": ["freq"]}, max_hilbert_dim=5)
    with pytest.raises(ValueError, match="cutoff"):
        fit_a_dress(source.with_space(ExcitationCutoff(1)), constraints={**constraints, "q": {"anharmonicity": -0.25}},
                    vary={"q": ["anharmonicity"]})


@pytest.mark.optional_backend
@pytest.mark.validation
def test_cutoff_derivatives_match_central_differences():
    """JIT derivatives of a truncated frequency and of pulse populations from prepared states match differences."""
    pytest.importorskip("dynamiqs")
    chips = {backend: pair(loss=True, frame={"q": 5.0, "r": 7.0}, backend=backend, space=ExcitationCutoff(3))
             for backend in ("dynamiqs", "qutip")}

    def frequency(g):
        return chips["dynamiqs"].with_params({"qr.g": g}).freq("q")

    def population(g, *, dressed, backend="dynamiqs"):
        bound = chips[backend].with_params({"qr.g": g})
        # The default state is the vacuum-assigned eigenvector of the traced truncated model.
        state = bound.compact_state({"q": 1}, dressed=True) if dressed else None
        result = schedule(bound).simulate(tlist=np.linspace(0.0, 16.0, 9), initial_state=state,
                                          options=tolerances(backend))
        return jnp.real(result.population("q", 1)[-1])

    for output in (frequency, lambda g: population(g, dressed=False), lambda g: population(g, dressed=True)):
        value, derivative = jax.jit(jax.value_and_grad(output))(0.1)
        step = 1e-4
        # The central-difference truncation error at this step is about 1e-6 of the derivative.
        estimate = (output(0.1 + step) - output(0.1 - step)) / (2 * step)
        assert float(value) == pytest.approx(float(output(0.1)), abs=1e-9)
        assert float(derivative) == pytest.approx(float(estimate), rel=1e-5)
    # QuTiP solves the same truncated model with interpolated drive samples.
    assert float(population(0.1, dressed=False)) == pytest.approx(
        float(population(0.1, dressed=False, backend="qutip")), abs=1e-5)


def _ring(approximation):
    """Return eight qubits, eight couplers, and eight readout resonators coupled in a ring."""
    qubits = [DuffingTransmon(freq=4.8 + 0.07 * i, anharmonicity=-0.22, levels=3, label=f"q{i}") for i in range(8)]
    couplers = [DuffingTransmon(freq=6.1 + 0.05 * i, anharmonicity=-0.12, levels=3, label=f"c{i}")
                for i in range(8)]
    readouts = [Resonator(freq=7.0 + 0.04 * i, levels=3, label=f"r{i}") for i in range(8)]
    couplings = []
    for i in range(8):
        couplings += [Capacitive(qubits[i], couplers[i], g=0.08), Capacitive(couplers[i], qubits[(i + 1) % 8], g=0.08),
                      Capacitive(qubits[i], readouts[i], g=0.06)]
    return Chip(qubits + couplers + readouts, couplings, approximation=approximation)


def test_state_of_a_large_cutoff_chip_raises_before_the_product_vector():
    """chip.state() of the 24-device ring needs a 3**24 product vector, so the memory check raises first."""
    chip = _ring(Exact()).with_space(ExcitationCutoff(3))
    with pytest.raises(MemoryError, match=r"Product-space state.*compact_state"):
        chip.state({"q0": 0})


@pytest.mark.unit
def test_compact_operators_are_stored_sparse_on_qutip():
    """On QuTiP, every compact operator is CSR, so its memory follows its entries instead of the dimension squared."""
    limited = pair(loss=True).with_space(ExcitationCutoff(3))
    engine = schedule(limited).build_problem(np.linspace(0.0, 16.0, 5), solver="mesolve").engine_result
    terms = (*engine.static_terms, *engine.dynamic_terms, *engine.collapse_terms)
    assert len(terms) == 6 and all(term.operator.layout == "csr" for term in terms)


@pytest.mark.unit
@pytest.mark.optional_backend
def test_a_dense_compact_operator_that_cannot_fit_raises(monkeypatch):
    """dynamiqs holds compact operators densely, so the memory check runs before each one is allocated."""
    pytest.importorskip("dynamiqs")
    from quchip.utils import memory as _memory

    devices = [Resonator(freq=0.25, levels=3, label=f"r{i}") for i in range(4)]
    chain = [Capacitive(a, b, g=0.01) for a, b in zip(devices, devices[1:])]
    chip = Chip(devices, couplings=chain, approximation=Exact(), backend="dynamiqs").with_space(ExcitationCutoff(3))
    sequence, tlist = QuantumSequence(chip), np.linspace(0.0, 3.0, 7)
    # The first build keeps the dressed excitation classes, whose eigenvectors need more memory than one operator.
    sequence.build_problem(tlist, initial_state={"r0": 1})
    # The odd sectors keep 20 labels, so one dense operator and its fill take 12.8 kB.
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 10**4)
    with pytest.raises(MemoryError, match=r"dense compact operator \(dimension 20\)"):
        sequence.build_problem(tlist, initial_state={"r0": 1})


@pytest.mark.validation
def test_ring_pilot_keeps_exact_classes_without_a_convergence_claim():
    """A 24-device ring diagonalizes only its parity classes, and K = 1 reproduces the RWA spectrum."""
    exact, rwa = _ring(Exact()), _ring(RWA())
    reference = rwa.freq()
    sizes = {1: ((1,), (24,)), 2: ((301,), (24,)), 3: ((301,), (2600,))}
    for max_total, (even, odd) in sizes.items():
        limited = exact.with_space(ExcitationCutoff(max_total))
        frequencies = limited.freq()
        model = limited.analysis._sector_model()
        assert tuple(model.space(group).size for group in model.classes) == even + odd
        assert all(np.isfinite(float(value)) for value in frequencies.values())
        if max_total == 1:
            # Without a retained partner for the vacuum, only exchange terms act inside N = 1.
            for label, value in reference.items():
                assert float(frequencies[label]) == pytest.approx(float(value), abs=1e-12)
    truncated = rwa.with_space(ExcitationCutoff(3)).freq()
    assert all(float(truncated[label]) == pytest.approx(float(value), abs=1e-12) for label, value in reference.items())
