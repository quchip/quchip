"""Collapse support remains structural through bases and retained coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import (
    RWA, Bath, Capacitive, Chip, DuffingTransmon, EffectiveTerms, PortNetwork,
    QuantumSequence, Resonator, eliminate, set_default_backend,
)
from quchip.chip.effective import OperatorProjection
from quchip.declarative import CollapseChannel, CouplingModel, DeviceModel, Scalar, parameter
from quchip.interop.eigenbasis import EigenbasisDevice


class Diagonal(DeviceModel):
    freq: Scalar = parameter()

    def local_hamiltonian(self, op, p):
        return p.freq * op.n - 3. * (op.n @ op.n)


def _readout_chip(*, loss=False, backend="dynamiqs"):
    a = DuffingTransmon(freq=5., anharmonicity=-.25, levels=2, label="a", T1=200. if loss else None)
    b = DuffingTransmon(freq=5.2, anharmonicity=-.25, levels=2, label="b")
    bus = Resonator(freq=6.3, levels=2, label="bus")
    readout = Resonator(freq=7., levels=2, label="readout")
    network = PortNetwork(label="m")
    network.port("out", target=readout, rate=.05)
    return Chip([a, b, bus, readout], [Capacitive(a, bus, g=.08, label="leg"),
                Capacitive(b, bus, g=.08), Capacitive(a, readout, g=.06)],
                port_network=network, approximation=RWA(), backend=backend, frame="rotating")


def test_sw_spectator_port_keeps_its_operator_after_serialization():
    """An operator outside the SW generator support is unchanged in independent frames."""
    source = _readout_chip(backend="qutip")
    result = eliminate(source, "bus")
    embedding = np.asarray(result.mapping.embedding)
    original = np.asarray(source.resolve(frame="lab").collapse_terms[0].operator.to_dense())
    expected = embedding.conj().T @ original @ embedding
    for variant in (result.chip, Chip.from_dict(result.chip.to_dict())):
        actual = np.asarray(variant.resolve().collapse_terms[0].operator.to_dense())
        # The embedding is the identity on the spectator. The bound covers round-off in 16-dimensional products.
        np.testing.assert_allclose(actual, expected, atol=2e-14)


@pytest.mark.optional_backend
def test_sw_spectator_port_accepts_traced_coupling():
    """A spectator port keeps its single phase when the SW rotation is differentiated."""
    pytest.importorskip("dynamiqs")
    source = _readout_chip()

    def amplitude(g):
        reduced = eliminate(source.with_params({"leg.g": g}), "bus").chip
        return jnp.real(reduced.resolve().collapse_terms[0].operator.to_dense()[0, 1])

    value, derivative = jax.jit(jax.value_and_grad(amplitude))(.08)
    # The spectator port keeps its authored unit matrix element, so only round-off remains.
    assert float(value) == pytest.approx(float(amplitude(.08)), abs=2e-14)
    assert float(value) == pytest.approx(1., abs=2e-14)
    assert float(derivative) == 0.


@pytest.mark.optional_backend
def test_isolated_three_level_loss_keeps_spectator_under_compiled_local_elimination():
    """Compiled local elimination of an isolated lossy mode keeps the analytic spectator and loss."""
    pytest.importorskip("dynamiqs")
    spectator = Resonator(freq=5., levels=3, label="q")
    removed = Resonator(freq=7., levels=3, label="m", T1=100.)
    chip = Chip([spectator, removed], backend="dynamiqs")

    def observables(frequencies):
        reduced = eliminate(chip.with_params({"q.freq": frequencies[0], "m.freq": frequencies[1]}),
                            "m", local=True).chip
        resolved = reduced.resolve(frame="lab")
        assert resolved.dims == (3,)
        channel, = resolved.collapse_terms
        energies = jnp.real(jnp.diag(resolved.hamiltonian().matrix(backend=reduced.backend)))
        strength = jnp.sum(jnp.abs(channel.operator.to_dense()) ** 2)
        return jnp.concatenate((energies, jnp.array([strength, channel.rate])))

    frequencies = jnp.array([5., 7.])
    value, derivative = jax.jit(lambda x: jax.jvp(observables, (x,), (jnp.ones(2),)))(frequencies)
    # Uncoupled modes give analytic values: levels 0, f and 2f, no inherited amplitude
    # and the rate 1/T1. The bound covers round-off near 10 GHz.
    np.testing.assert_allclose(value, [0., 5., 10., 0., .01], atol=1e-12)
    np.testing.assert_allclose(derivative, [0., 1., 2., 0., 0.], atol=1e-12)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_collective_bath_frequency_gradient_matches_one_excitation_solution():
    """Collective decay in a common frame matches the independent two-amplitude solution."""
    from scipy.linalg import expm

    pytest.importorskip("dynamiqs")
    a = DuffingTransmon(freq=5., anharmonicity=-.25, levels=2, label="a")
    b = DuffingTransmon(freq=5.05, anharmonicity=-.25, levels=2, label="b")
    source = Chip([a, b], baths=[Bath("collective_decay", targets=[a, b], rate=.1, label="shared")],
                  frame=5., approximation=RWA(), backend="dynamiqs")

    def population(frequency):
        chip = source.with_params({"a.freq": frequency})
        result = QuantumSequence(chip).simulate(tlist=jnp.linspace(0., 5., 11), initial_state={"a": 1})
        return jnp.real(result.population("a", level=1)[-1])

    def reference(frequency):
        generator = -2j * np.pi * np.diag([frequency - 5., .05]) - .05 * np.ones((2, 2))
        return abs((expm(5. * generator) @ [1., 0.])[0]) ** 2

    point, step = 4.97, 1e-5
    plain = population(point)
    compiled, compiled_derivative = jax.jit(jax.value_and_grad(population))(point)
    # dynamiqs integrates at rtol 1e-9 and atol 1e-11 by default. The bound is ten times that rtol.
    np.testing.assert_allclose([plain, compiled], reference(point), rtol=1e-8, atol=1e-11)
    expected = (reference(point + step) - reference(point - step)) / (2 * step)
    # The central difference of the exact solution has O(step^2) error near 1e-10, so solver error dominates.
    np.testing.assert_allclose(compiled_derivative, expected, rtol=1e-7, atol=1e-11)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_reduced_collective_bath_frequency_derivative_survives_serialization():
    """A collective Bath on a reduced chip keeps one phase in a common frame, also after a round trip."""
    pytest.importorskip("dynamiqs")
    a = DuffingTransmon(freq=5., anharmonicity=-.25, levels=3, label="a")
    b = DuffingTransmon(freq=5.05, anharmonicity=-.25, levels=3, label="b")
    bus = Resonator(freq=6.3, levels=2, label="bus")
    source = Chip([a, b, bus], [Capacitive(a, bus, g=.05), Capacitive(b, bus, g=.05)],
                  baths=[Bath("collective_decay", targets=[a, b], rate=.1, label="shared")],
                  frame=5., approximation=RWA(), backend="dynamiqs")

    def amplitude(frequency, reduce=True):
        bound = source.with_params({"a.freq": frequency})
        chip = eliminate(bound, "bus").chip if reduce else restored.with_params({"a.freq": frequency})
        channel = next(term for term in chip.resolve().collapse_terms if term.source == "shared")
        return jnp.real(channel.operator.to_dense()[0, 3])

    value, derivative = jax.jit(jax.value_and_grad(amplitude))(5.)
    step = 1e-4
    reference = (amplitude(5. + step) - amplitude(5. - step)) / (2 * step)
    # The compiled call differs from plain evaluation only by reassociation round-off.
    assert float(value) == pytest.approx(float(amplitude(5.)), abs=2e-13)
    # The central difference has O(step²) truncation error and round-off divided by the step.
    assert float(derivative) == pytest.approx(float(reference), rel=2e-6, abs=1e-10)
    assert abs(float(derivative)) > 1e-4
    # Serialization does not store the backend.
    set_default_backend("dynamiqs")
    restored = Chip.from_dict(eliminate(source, "bus").chip.to_dict())
    # The restored operators are fixed matrices, so both calls build the same exact entries.
    np.testing.assert_allclose(jax.jit(lambda f: amplitude(f, reduce=False))(5.), amplitude(5., reduce=False),
                               atol=1e-15)


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_carried_jump_names_its_devices_and_the_frames_that_work(method):
    """A carried jump with unequal band phases names its devices and the frames that give them one phase."""
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-.25, levels=3, T1=60_000., label="q1")
    q2 = DuffingTransmon(freq=5.4, anharmonicity=-.25, levels=3, T1=60_000., label="q2")
    bus = Resonator(freq=6.4, levels=3, label="b")
    chip = Chip([q1, q2, bus], [Capacitive(q1, bus, g=.07), Capacitive(q2, bus, g=.07)],
                approximation=RWA(), frame="rotating")
    reduced = eliminate(chip, "b", method=method).chip
    with pytest.raises(ValueError, match=r"'q1'/'thermal_emission'.*devices \['q1', 'q2'\]") as error:
        reduced.resolve()
    for phrase in ('frame="lab"', "common frame", 'frame="auto"', "Time-dependent collapse operators"):
        assert phrase in str(error.value)
    # The undriven exchange-coupled pair rotates together in the "auto" frame, so the carried jump is static.
    frequencies = reduced.resolve(frame="auto").resolved_frame.frequencies
    assert frequencies["q1"] == frequencies["q2"]


@pytest.mark.optional_backend
def test_sw_induced_loss_still_rejects_incompatible_frames():
    """The induced loss on the other survivor cannot share a phase in independent frames."""
    pytest.importorskip("dynamiqs")
    source = _readout_chip(loss=True)

    def amplitude(g):
        reduced = eliminate(source.with_params({"leg.g": g}), "bus").chip
        return jnp.real(reduced.resolve().collapse_terms[0].operator.to_dense()[0, 1])

    with pytest.raises(ValueError, match="operator bands with different phases"):
        jax.jit(amplitude)(.08)


@pytest.mark.optional_backend
def test_traced_diagonal_basis_keeps_authored_order_and_rejects_reordered_levels():
    """Traced energies keep a diagonal model's authored level order, and execution checks that order."""
    pytest.importorskip("dynamiqs")

    def operator(frequency):
        a = Diagonal(freq=frequency, levels=3, label="a")
        b = Diagonal(freq=frequency, levels=3, label="b")
        chip = Chip([a, b], baths=[Bath("collective_decay", rate=.1)], frame=1., backend="dynamiqs")
        return chip.resolve().collapse_terms[0].operator.to_dense()

    compiled = jax.jit(operator)
    # At f=10 the energies 0, 7 and 8 are in authored order. Both calls build the same exact entries.
    np.testing.assert_allclose(compiled(10.), operator(10.), atol=2e-14)
    # At f=9 the energies 0, 6 and 6 tie. The stable sort keeps their authored order, as tracing does.
    np.testing.assert_allclose(compiled(9.), operator(9.), atol=2e-14)
    # At f=-1 the energies 0, -4 and -14 are reversed. Plain evaluation sorts them, and one phase remains.
    operator(-1.)
    # At f=5 the energies 0, 2 and -2 give the native lowering edges level changes 1 and -2.
    with pytest.raises(ValueError, match="operator bands with different phases"):
        operator(5.)
    for frequency in (-1., 5.):
        with pytest.raises(jax.errors.JaxRuntimeError, match="out of their authored order"):
            compiled(frequency).block_until_ready()
        with pytest.raises(ValueError, match="out of their authored order"):
            jax.grad(lambda f: jnp.real(operator(f)[1, 0]))(frequency)


@pytest.mark.optional_backend
def test_traced_eigen_truncation_keeps_only_the_lowest_levels():
    """A traced eigen truncation keeps the levels of plain evaluation, and a lower dropped level raises."""
    pytest.importorskip("dynamiqs")
    q = DuffingTransmon(freq=1., anharmonicity=-.9, levels=5, label="q")
    q.basis, q.projection_levels = "eigen", 3
    chip = Chip([q], backend="dynamiqs")

    def energies(frequency):
        return chip.with_params({"q.freq": frequency}).resolve(frame="lab").bases["q"].energies

    compiled = jax.jit(energies)
    # At f=3 the energies 0, 3, 5.1, 6.3 and 6.6 increase, so both calls keep the first three exactly.
    np.testing.assert_allclose(compiled(3.), energies(3.), atol=1e-14)
    # At f=1 the energies are 0, 1, 1.1, 0.3 and -1.4. Plain evaluation keeps levels 4, 0 and 3.
    np.testing.assert_allclose(energies(1.), [-1.4, 0., .3], atol=1e-14)
    with pytest.raises(jax.errors.JaxRuntimeError, match="out of their authored order"):
        compiled(1.).block_until_ready()


def test_concrete_device_values_keep_sorted_levels_inside_a_trace():
    """Concrete device values inside a JAX trace give the sorted levels of plain evaluation."""
    # Above level 21 the authored energies of this model decrease, so a traced order check would fail.
    q = DuffingTransmon(freq=5., anharmonicity=-.25, levels=24, label="q")
    gap = jax.jit(lambda scale: scale * q.transition_frequency(0, 1))(1.)
    # E_1 - E_0 is exactly the frequency in floating point.
    assert float(gap) == 5.


@pytest.mark.optional_backend
def test_jitted_bath_rate_keeps_the_common_frame_check_of_stored_spectra():
    """A collective bath on stored spectra passes the common-frame check under jit, as in plain evaluation."""
    pytest.importorskip("dynamiqs")
    a = EigenbasisDevice(np.array([0., 5., 9.8]), label="a")
    b = EigenbasisDevice(np.array([0., 5., 9.7]), label="b")
    chip = Chip([a, b], baths=[Bath("collective_decay", targets=[a, b], rate=.1, label="shared")],
                approximation=RWA(), frame=5., backend="dynamiqs")

    def rate(value):
        return sum(term.rate for term in chip.with_params({"bath.shared.rate": value}).resolve().collapse_terms)

    assert float(jax.jit(rate)(.1)) == pytest.approx(float(rate(.1)), abs=1e-15)


def test_reduced_bath_rechecks_bands_after_an_undeclared_map():
    """A bath operator that crosses a map with no declared level changes keeps its retained bands."""
    q = Resonator(freq=5., levels=2, label="q")
    r = Resonator(freq=5.5, levels=2, label="r")
    bus = Resonator(freq=7., levels=2, label="bus")
    # The earlier map mixes |00> and |11>, so it adds total level changes of two.
    mixing = np.eye(4, dtype=complex)
    mixing[np.ix_([0, 3], [0, 3])] = [[np.cos(.3), np.sin(.3)], [-np.sin(.3), np.cos(.3)]]
    projection = OperatorProjection(("q", "r"), (2, 2), ("q", "r"), (2, 2), mixing)
    terms = EffectiveTerms(("q", "r"), (2, 2), np.zeros((4, 4)), projection=projection, label="user")
    source = Chip([q, r, bus], [Capacitive(r, bus, g=.05)], effective_terms=(terms,),
                  baths=[Bath("collective_decay", targets=[q, r], rate=.1, label="shared")],
                  approximation=RWA(), frame=5.)
    for chip in (source, eliminate(source, "bus").chip):
        with pytest.raises(ValueError, match="operator bands with different phases"):
            chip.resolve()


def test_common_frame_check_prunes_a_zero_structural_band():
    """A zero-amplitude raising term adds no second phase to a channel in a common frame."""

    class LeakyDecay(CouplingModel):
        g: Scalar = parameter(unit="GHz")
        leak: Scalar = parameter()

        def interaction(self, a, b, p):
            return p.g * (a.adag * b.a + a.a * b.adag)

        def dissipation(self, a, b, p):
            return (CollapseChannel(a.a * b.I + p.leak * (a.I * b.adag), .1, "leaky_decay"),)

    a = DuffingTransmon(freq=5., anharmonicity=-.25, levels=2, label="a")
    b = DuffingTransmon(freq=5.05, anharmonicity=-.25, levels=2, label="b")
    channel, = Chip([a, b], [LeakyDecay(a, b, g=.01, leak=0.)], frame=5.).resolve().collapse_terms
    # The channel keeps its authored entries, which are exactly zero or one.
    np.testing.assert_allclose(channel.operator.to_dense(), np.kron(np.diag([1.], 1), np.eye(2)), atol=1e-15)
    with pytest.raises(ValueError, match="operator bands with different phases"):
        Chip([a, b], [LeakyDecay(a, b, g=.01, leak=.2)], frame=5.).resolve()
