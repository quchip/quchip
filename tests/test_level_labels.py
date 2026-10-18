"""A level label names its dressed state as an initial state, as a target and in a superposition."""

import numpy as np
import pytest

from quchip import RWA, Capacitive, Chip, DuffingTransmon, Exact, QuantumSequence

# The dynamiqs runs compile each tight solve, so they run with validation, and QuTiP keeps the E2E check.
BACKENDS = ["qutip", pytest.param("dynamiqs", marks=[pytest.mark.optional_backend, pytest.mark.validation])]

# The one-excitation block under RWA() has detuning -0.1 GHz and coupling 0.03 GHz, so tan 2θ = 0.6.
RABI = np.hypot(0.1, 0.06)
COS2 = 0.5 * (1 + 0.1 / RABI)


def pair(approximation=None, frame="lab", backend="qutip"):
    """Return two transmons 100 MHz apart with g = 30 MHz."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=3, label="q1")
    return Chip([q0, q1], [Capacitive(q0, q1, g=0.03, label="c")], approximation=approximation or RWA(),
                frame=frame, backend=backend)


def tight(backend):
    """Solver tolerances far below the checked differences."""
    if backend == "dynamiqs":
        dq = pytest.importorskip("dynamiqs")
        return {"method": dq.method.Tsit5(rtol=1e-10, atol=1e-12)}
    return {"atol": 1e-12, "rtol": 1e-10}


def vector(chip, state):
    return np.asarray(chip.backend.to_array(state)).reshape(-1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_label_starts_in_its_dressed_state_at_the_start_time(backend):
    """At t0 = 3.7 ns, {"q0": 1} is c|10> + s exp(i 2π (f1 - f0) t0)|01> in each frame, and stays in its label."""
    t0 = 3.7
    times = t0 + np.linspace(0.0, 50.0, 201)
    for frame in ("lab", 5.0, "rotating"):
        chip = pair(frame=frame, backend=backend)
        result = QuantumSequence(chip).simulate(times, initial_state={"q0": 1}, dissipation=False,
                                                options=tight(backend))
        f0, f1 = (float(result.frame.frequencies[label]) for label in ("q0", "q1"))
        expected = np.zeros((3, 3), dtype=complex)
        expected[1, 0] = np.sqrt(COS2)
        expected[0, 1] = -np.sqrt(1 - COS2) * np.exp(2j * np.pi * (f1 - f0) * t0)
        np.testing.assert_allclose(vector(chip, result.state(t0)), expected.reshape(-1), atol=1e-12, rtol=0)
        # The label's amplitude starts at 1 and turns at its frame energy minus its dressed energy.
        energy = float(chip.energy({"q0": 1}))
        np.testing.assert_allclose(result.amplitude_array({"q0": 1}),
                                   np.exp(2j * np.pi * (f0 - energy) * (times - t0)), atol=1e-7, rtol=0)


@pytest.mark.unit
def test_an_omitted_state_is_the_all_ground_label():
    """Under Exact(), {} and an omitted state are the dressed ground, which is not the bare vacuum."""
    chip = pair(Exact())
    chip.set_state_order("q0", "q1")

    def start(spec):
        problem = QuantumSequence(chip).build_problem(np.array([0.0, 1.0]), initial_state=spec)
        return vector(chip, problem.initial_state)

    omitted = start(None)
    for spec in ({}, "00"):
        np.testing.assert_allclose(start(spec), omitted, atol=1e-14, rtol=0)
    # Counter-rotating terms put about (g / (f0 + f1))² of the ground on |11>.
    assert 1e-6 < 1 - abs(omitted[0]) ** 2 < 1e-4
    with pytest.raises(TypeError, match=r"Use chip\.bare_state"):
        QuantumSequence(chip).build_problem(np.array([0.0, 1.0]), initial_state={"q0": chip.bare_state(q0=1)})


def test_a_bare_start_exchanges_and_a_dressed_start_stays():
    """From the bare |10>, population falls to 1 - sin²2θ; the label overlap stays cos²θ, as it does from {"q0": 1}."""
    chip = pair()
    times = np.linspace(0.0, 1 / RABI, 201)
    bare = QuantumSequence(chip).simulate(times, initial_state=chip.bare_state(q0=1), options=tight("qutip"))
    dressed = QuantumSequence(chip).simulate(times, initial_state={"q0": 1}, options=tight("qutip"))
    # The exchange is deepest at half a period, 1 / (2 Rabi).
    assert float(np.asarray(bare.population("q0", 1))[100]) == pytest.approx(1 - 0.6**2 / 1.36, abs=1e-8)
    np.testing.assert_allclose(bare.overlap({"q0": 1}), COS2, atol=1e-8, rtol=0)
    np.testing.assert_allclose(dressed.overlap({"q0": 1}), 1.0, atol=1e-8, rtol=0)
    # Population is the local projector, and the bare ket target gives the bare overlap.
    np.testing.assert_allclose(dressed.population("q0", 1), COS2, atol=1e-8, rtol=0)
    np.testing.assert_allclose(dressed.overlap(chip.bare_state(q0=1)), COS2, atol=1e-8, rtol=0)


@pytest.mark.unit
def test_superpositions_are_dressed_and_bare_state_takes_pairs():
    """superposition() combines dressed states, bare_state() takes the same pairs, and a zero sum raises."""
    chip = pair()
    ground, excited = vector(chip, chip.state({})), vector(chip, chip.state(q0=1))
    np.testing.assert_allclose(vector(chip, chip.superposition({}, (1j, {"q0": 1}))),
                               (ground + 1j * excited) / np.sqrt(2), atol=1e-14, rtol=0)
    expected = np.zeros(9, dtype=complex)
    expected[[0, 3]] = 0.6, 0.8
    np.testing.assert_allclose(vector(chip, chip.bare_state((0.6, {}), (0.8, {"q0": 1}))), expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(vector(chip, chip.bare_state({"q0": 1})), np.eye(9)[3], atol=0, rtol=0)
    with pytest.raises(ValueError, match="zero norm"):
        chip.superposition((1, {"q0": 1}), (-1, {"q0": 1}))
    with pytest.raises(TypeError, match="Keyword levels name one product state"):
        chip.bare_state({}, {"q0": 1}, q1=1)
    compact = chip.compact_state({"q0": 1})
    rows = [excited[3 * n0 + n1] for n0, n1 in compact.space.levels]
    np.testing.assert_allclose(np.asarray(compact.values).reshape(-1), rows, atol=1e-14, rtol=0)
    bare = chip.compact_state({"q0": 1}, dressed=False)
    assert np.asarray(bare.values).reshape(-1).tolist() == [1.0 if tuple(level) == (1, 0) else 0.0
                                                             for level in bare.space.levels]


@pytest.mark.unit
def test_compact_bare_products_take_local_kets_inside_the_cutoff():
    """compact_state(dressed=False) reads a product of local kets on its labels, and the cutoff bounds its weight."""
    from quchip import ChargeBasisTransmon, ExcitationCutoff, Resonator

    chip = pair()
    plus = (chip["q0"].basis_state(0) + chip["q0"].basis_state(1)).unit()
    compact = chip.compact_state({"q0": plus, "q1": 1}, dressed=False)
    assert compact.space.totals == (1, 2)
    native = vector(chip, chip.bare_state(q0=plus, q1=1)).reshape(3, 3)
    np.testing.assert_allclose(np.asarray(compact.values).reshape(-1),
                               [native[tuple(level)] for level in compact.space.levels], atol=1e-14, rtol=0)
    with pytest.raises(TypeError, match="dressed=False"):
        chip.compact_state({"q0": plus})
    # Half of |+>|1> has N = 2, above the cutoff.
    with pytest.raises(ValueError, match="weight 0.5"):
        chip.with_space(ExcitationCutoff(1)).compact_state({"q0": plus, "q1": 1}, dressed=False)
    # The ground levels of native-basis devices add no round-off support.
    r = Resonator(freq=6.0, levels=3, label="r")
    devices = [ChargeBasisTransmon(E_C=0.25, E_J=12.0, n_g=0.1, num_basis=7, basis="native", label=f"t{i}")
               for i in range(3)]
    plus = (r.basis_state(0) + r.basis_state(1)).unit()
    assert Chip([r, *devices]).compact_state({"r": plus}, dressed=False).space.totals == (0, 1)


@pytest.mark.unit
@pytest.mark.optional_backend
def test_a_traced_local_ket_keeps_the_support_of_integer_levels():
    """Under jit and a cutoff, a traced ket counts all its levels, and an integer level counts only itself."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    from quchip import ExcitationCutoff, Resonator

    chip = Chip([DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="a"),
                 Resonator(freq=6.0, levels=3, label="r")], backend="dynamiqs", space=ExcitationCutoff(1))

    @jax.jit
    def values(x):
        ket = jnp.array([1.0, x], dtype=complex) / jnp.sqrt(1 + x**2)
        return jnp.asarray(chip.compact_state({"a": ket, "r": 0}, dressed=False).values).reshape(-1)

    # The cutoff keeps |00>, |01> and |10>, and the resonator stays in level 0.
    np.testing.assert_allclose(np.asarray(values(0.5)), np.array([2, 0, 1]) / np.sqrt(5), atol=1e-14, rtol=0)


@pytest.mark.unit
# The 1e-12 GHz pair terms also raise the round-off conservation advisory.
@pytest.mark.filterwarnings("ignore:The total excitation number N changes only")
def test_every_label_query_takes_the_assigned_state_and_warns_when_it_is_mixed():
    """Near a degeneracy a label starts in the state of chip.state(), and a weak assignment warns on each path."""
    from quchip import ExcitationCutoff

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q1")
    # At g = 1e-12 GHz the bare |10> nearly solves H, but the resonant dressed states are equal mixtures.
    chip = Chip([q0, q1], [Capacitive(q0, q1, g=1e-12, label="c")], approximation=Exact())
    with pytest.warns(UserWarning, match="assignment overlap 0.50"):
        problem = QuantumSequence(chip).build_problem(np.array([0.0, 1.0]), initial_state={"q0": 1})
    with pytest.warns(UserWarning, match="assignment overlap 0.50"):
        expected = vector(chip, chip.state(q0=1))
    start = vector(chip, problem.initial_state)
    np.testing.assert_allclose(start, expected, atol=1e-14, rtol=0)
    assert abs(start[3]) ** 2 == pytest.approx(0.5, abs=0.01)
    # Under a cutoff, the label takes the block of its class, and its overlap of 0.820 warns there too.
    mixed = pair().with_params({"c.g": 0.06}).with_space(ExcitationCutoff(1))
    with pytest.warns(UserWarning, match="assignment overlap 0.820"):
        QuantumSequence(mixed).build_problem(np.array([0.0, 1.0]), initial_state={"q0": 1})


@pytest.mark.validation
@pytest.mark.optional_backend
def test_label_start_derivatives_keep_the_label_gauge():
    """On a loop with a complex link, the derivative of a label start matches central differences."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    from quchip import Coupling

    devices = [DuffingTransmon(freq=5.0 + 0.1 * i, anharmonicity=-0.25, levels=2, label=f"q{i}") for i in range(3)]

    def link(a, b, backend):
        hop = backend.tensor(backend.dag(a.lowering_operator()), b.lowering_operator())
        return 1j * hop - 1j * backend.dag(hop)

    # The loop phase makes the relative phases of the dressed state depend on the real coupling g01.
    chip = Chip(devices, [Capacitive(devices[0], devices[1], g=0.03, label="c01"),
                          Capacitive(devices[1], devices[2], g=0.02, label="c12"),
                          Coupling(devices[0], devices[2], g=0.015, interaction=link, label="c02")],
                approximation=RWA(), backend="dynamiqs")

    def start(g):
        bound = chip.with_params({"c01.g": g})
        problem = QuantumSequence(bound).build_problem(np.array([0.0, 1.0]), initial_state={"q0": 1},
                                                       dissipation=False)
        return jnp.asarray(bound.backend.to_array(problem.initial_state)).reshape(-1)

    g, step = 0.03, 1e-5
    derivative = np.asarray(jax.jacfwd(start)(jnp.asarray(g)))
    difference = np.asarray((start(g + step) - start(g - step)) / (2 * step))
    # The central difference errs by about step² times the third derivative, below 1e-7 here.
    np.testing.assert_allclose(derivative, difference, atol=1e-6, rtol=0)
