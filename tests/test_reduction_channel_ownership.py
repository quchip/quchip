"""Surviving dissipative owners remain live in the reduced coordinates."""
import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, Exact, Resonator, eliminate


def _source(backend="qutip"):
    q = DuffingTransmon(freq=5., anharmonicity=-.3, levels=3, T1=40., label="q")
    r = Resonator(freq=7., levels=3, T1=30., label="r")
    return Chip([q, r], [Capacitive(q, r, g=.1, label="edge")],
                approximation=Exact(), backend=backend)


def _jumps(chip):
    return [np.asarray(chip.backend.to_array(c)) for c in
            chip.backend._collapse_operators(chip.resolve(frame="lab"))]


@pytest.mark.validation
@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_exact_edge_transforms_surviving_channels_with_the_public_map(backend):
    source = _source(backend)
    result = eliminate(source, "edge", method="exact")
    u = np.asarray(result.mapping.embedding)
    before, after = _jumps(source), _jumps(result.chip)
    assert len(before) == len(after) == 2
    for c, transformed in zip(before, after):
        np.testing.assert_allclose(transformed, u.conj().T @ c @ u, atol=1e-12)
    changed = result.chip.with_params({"q.T1": 160.})
    altered = _jumps(changed)
    np.testing.assert_allclose(altered[0], after[0] / 2, atol=1e-12)
    np.testing.assert_allclose(altered[1], after[1], atol=1e-12)
    assert result.chip["q"].T1 == 40.


def test_exact_mode_transforms_surviving_channel_and_keeps_removed_loss_separate():
    source = _source()
    result = eliminate(source, "r", method="exact")
    u = np.asarray(result.mapping.embedding)
    before, after = _jumps(source), _jumps(result.chip)
    assert len(before) == len(after) == 2
    for c, transformed in zip(before, after):
        np.testing.assert_allclose(transformed, u.conj().T @ c @ u, atol=1e-12)


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_repeated_reductions_compose_channel_coordinates(method):
    source = _source()
    first = eliminate(source, "edge", method=method)
    second = eliminate(first.chip, "r", method=method)
    u = np.asarray(first.mapping.embedding @ second.mapping.embedding)
    for original, transformed in zip(_jumps(source), _jumps(second.chip)):
        np.testing.assert_allclose(transformed, u.conj().T @ original @ u, atol=1e-12)


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_projection_survives_clone_serialization_and_new_noise_activation(method):
    source = _source()
    source["q"].T1 = None
    result = eliminate(source, "edge", method=method)
    expected_source = _source()
    u = np.asarray(result.mapping.embedding)
    for variant in (result.chip.clone(), Chip.from_dict(result.chip.to_dict())):
        variant["q"].T1 = 40.
        for original, transformed in zip(_jumps(expected_source), _jumps(variant)):
            np.testing.assert_allclose(transformed, u.conj().T @ original @ u, atol=1e-12)


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_bath_channels_follow_reduction_and_keep_their_rate_parameter(method):
    from quchip import Bath

    source = _source()
    source["q"].T1 = source["r"].T1 = None
    source.add_bath(Bath("collective_decay", rate=.01, label="common"))
    edge = eliminate(source, "edge", method=method)
    result = eliminate(edge.chip, "r", method=method)
    u = np.asarray(edge.mapping.embedding @ result.mapping.embedding)
    before, after = _jumps(source), _jumps(Chip.from_dict(result.chip.to_dict()))
    assert len(before) == len(after) == 1
    for original, transformed in zip(before, after):
        np.testing.assert_allclose(transformed, u.conj().T @ original @ u, atol=1e-12)
    changed = result.chip.with_params({"bath.common.rate": .04})
    np.testing.assert_allclose(_jumps(changed)[-1], 2 * after[-1], atol=1e-12)


@pytest.mark.validation
@pytest.mark.parametrize("basis", ["native", "eigen"])
def test_surviving_channel_projection_uses_authored_coordinates(basis):
    from quchip import CollapseChannel
    from quchip.declarative import DeviceModel, Scalar, parameter

    class Tilted(DeviceModel):
        freq: Scalar = parameter(default=5.)

        def local_hamiltonian(self, op, p):
            return p.freq * op.n + .3 * op.sigma_x + .2 * op.sigma_y

        def dissipation(self, op, p):
            return (CollapseChannel(op.a, .01, "loss"),)

    q = Tilted(levels=3, label="q")
    q.projection_levels = 2
    r = Resonator(freq=7., levels=3, label="r")
    source = Chip([q, r], [Capacitive(q, r, g=.05)], basis=basis, approximation=Exact())
    result = eliminate(source, "r", method="exact")
    u = np.asarray(result.mapping.embedding)
    np.testing.assert_allclose(_jumps(result.chip)[0], u.conj().T @ _jumps(source)[0] @ u, atol=1e-12)


def test_exact_surviving_noise_preserves_the_full_lindblad_generator():
    source = _source()
    result = eliminate(source, "edge", method="exact")
    u = np.asarray(result.mapping.embedding)
    rho = np.ones((9, 9), dtype=complex) / 9

    def derivative(chip, state):
        h = np.asarray(chip.hamiltonian().matrix(backend=chip.backend))
        value = -2j * np.pi * (h @ state - state @ h)
        for c in _jumps(chip):
            cc = c.conj().T @ c
            value += c @ state @ c.conj().T - .5 * (cc @ state + state @ cc)
        return value

    expected = u.conj().T @ derivative(source, u @ rho @ u.conj().T) @ u
    np.testing.assert_allclose(derivative(result.chip, rho), expected, atol=1e-12)


@pytest.mark.validation
@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("method", ["sw", "exact"])
@pytest.mark.parametrize("target", ["edge", "r"])
def test_surviving_and_retargeted_ports_are_transformed_exactly_once(backend, method, target):
    from quchip import Port, PortNetwork

    q = Resonator(freq=5., levels=3, label="q")
    r = Resonator(freq=7., levels=3, label="r")
    source = Chip([q, r], [Capacitive(q, r, g=.1, label="edge")], approximation=Exact(), backend=backend,
                  port_network=PortNetwork.from_ports([Port(r, rate=.03, label="readout")]))
    result = eliminate(source, target, method=method)
    u = np.asarray(result.mapping.embedding)
    expected = u.conj().T @ _jumps(source)[0] @ u
    for variant in (result.chip, result.chip.clone(), Chip.from_dict(result.chip.to_dict())):
        np.testing.assert_allclose(_jumps(variant)[0], expected, atol=1e-12)


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_surviving_coupling_channel_keeps_its_nonadjacent_source_support(method):
    from quchip import CollapseChannel

    class NoisyCapacitive(Capacitive):
        def dissipation(self, a, b, p):
            return (CollapseChannel(a.n * b.a, .01, "loss"),)

    a = DuffingTransmon(freq=5., anharmonicity=-.3, levels=3, label="a")
    b = DuffingTransmon(freq=5.3, anharmonicity=-.2, levels=3, label="b")
    bus = Resonator(freq=7., levels=3, label="bus")
    source = Chip([a, bus, b], [NoisyCapacitive(a, b, g=.004, label="direct"),
                              Capacitive(a, bus, g=.05), Capacitive(b, bus, g=.04)], approximation=Exact())
    result = eliminate(source, "bus", method=method)
    u = np.asarray(result.mapping.embedding)
    np.testing.assert_allclose(_jumps(result.chip)[0], u.conj().T @ _jumps(source)[0] @ u, atol=1e-12)


def test_survivor_relaxation_matches_exact_exchange_rotation_through_second_order():
    from quchip.declarative import CouplingModel, Scalar, parameter

    class Exchange(CouplingModel):
        g: Scalar = parameter()

        def interaction(self, a, b, p):
            return p.g * (a.adag * b.a + a.a * b.adag)

    errors = []
    for g in (.16, .08, .04):
        q = Resonator(freq=5., levels=2, T1=40., label="q")
        r = Resonator(freq=7., levels=2, label="r")
        source = Chip([q, r], [Exchange(q, r, g=g)], approximation=Exact())
        result = eliminate(source, "r", method="sw")
        actual = _jumps(result.chip)[0][0, 1].real
        # The exact two-level exchange eigenstate rotates by half arctan(2g/Delta).
        exact_amplitude = np.cos(.5 * np.arctan(2 * g / 2.)) / np.sqrt(40.)
        errors.append(abs(actual - exact_amplitude))
    # An untransformed survivor jump has a second-order error. The shared SW
    # coordinate map includes that correction; its residual here is fourth order.
    assert 15.4 < errors[0] / errors[1] < 16.1
    assert 15.4 < errors[1] / errors[2] < 16.1
