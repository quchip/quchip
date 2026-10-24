"""Edge reduction changes coordinates without double-counting retained edges."""
import numpy as np
import pytest
from scipy.linalg import expm

from quchip import Capacitive, Chip, DuffingTransmon, Exact, Resonator, eliminate


def _chip(backend="qutip", g=.05, topology="parallel", scale=1.):
    q = DuffingTransmon(freq=5., anharmonicity=-.3, levels=3, label="q")
    r = Resonator(freq=7., levels=3, label="r")
    devices = [q, r]
    couplings = [Capacitive(q, r, g=g * scale, label="selected")]
    if topology == "parallel":
        couplings.append(Capacitive(q, r, g=.08 * scale, label="remaining"))
    else:
        s = Resonator(freq=8.4, levels=2, label="s")
        devices.insert(1, s)  # Nonadjacent pair exercises support ordering.
        couplings.append(Capacitive(q, s, g=.08 * scale, label="remaining"))
    return Chip(devices, couplings, backend=backend, approximation=Exact())


def _matrix(chip):
    return np.asarray(chip.resolve(frame="lab").hamiltonian().matrix(backend=chip.backend))


@pytest.mark.validation
@pytest.mark.parametrize("method", ["sw", "exact"])
@pytest.mark.parametrize("topology", ["parallel", "spectator"])
def test_zero_edge_is_identity_in_presence_of_other_interactions(method, topology):
    backend = "qutip"
    source = _chip(backend, g=0., topology=topology)
    result = eliminate(source, "selected", method=method)
    np.testing.assert_allclose(_matrix(result.chip), _matrix(source), atol=1e-12)
    np.testing.assert_allclose(result.mapping.embedding, np.eye(np.prod(source.dims)), atol=1e-12)
    assert result.chip["q"].freq == 5.
    assert result.chip.coupling_map["remaining"].g == .08
    assert "selected" not in result.chip.coupling_map


@pytest.mark.validation
@pytest.mark.parametrize("topology", ["parallel", "spectator"])
def test_exact_edge_rotation_preserves_entire_hamiltonian_and_dynamics(topology):
    source = _chip("qutip", topology=topology)
    result = eliminate(source, "selected", method="exact")
    h, reduced_h = _matrix(source), _matrix(result.chip)
    u = np.asarray(result.mapping.embedding)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(len(h)), atol=1e-12)
    np.testing.assert_allclose(u.conj().T @ h @ u, reduced_h, atol=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(reduced_h), np.linalg.eigvalsh(h), atol=1e-12)
    state = np.arange(1, len(h) + 1) + 1j
    state /= np.linalg.norm(state)
    np.testing.assert_allclose(expm(-2j * np.pi * h * .7) @ u @ state,
                               u @ expm(-2j * np.pi * reduced_h * .7) @ state, atol=1e-12)
    source["q"].freq = 6.
    result.chip["q"].freq = 4.
    np.testing.assert_allclose(result.mapping.embedding, u, atol=1e-12)


@pytest.mark.parametrize("topology", ["parallel", "spectator"])
def test_sw_retains_second_order_cross_terms(topology):
    errors = []
    for scale in [1., .5, .25]:
        source = _chip(topology=topology, scale=scale)
        result = eliminate(source, "selected", method="sw")
        u = np.asarray(result.mapping.embedding)
        errors.append(np.linalg.norm(u.conj().T @ _matrix(source) @ u - _matrix(result.chip)))
    # The difference from exponentiating the same first-order generator is
    # third order, including the remaining parallel/spectator interaction.
    assert 7.8 < errors[0] / errors[1] < 8.2
    assert 7.8 < errors[1] / errors[2] < 8.2


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_exact_edge_reduction_transforms_removed_channels_and_the_full_generator(backend):
    from quchip import CollapseChannel

    class NoisyEdge(Capacitive):
        def dissipation(self, a, b, p):
            return (CollapseChannel(a.a * b.I + a.I * b.a, .01, "collective"),)

    a = DuffingTransmon(freq=5., anharmonicity=-.3, levels=3, label="a")
    b = Resonator(freq=7., levels=3, label="b")
    source = Chip([a, b], [NoisyEdge(a, b, g=.08, label="edge")], backend=backend, approximation=Exact())
    result = eliminate(source, "edge", method="exact")
    u = np.asarray(result.mapping.embedding)

    def jumps(chip):
        return [np.asarray(chip.backend.to_array(c)) for c in
                chip.backend._collapse_operators(chip.resolve(frame="lab"))]

    source_jumps, retained_jumps = jumps(source), jumps(result.chip)
    assert len(source_jumps) == len(retained_jumps) == 1
    np.testing.assert_allclose(retained_jumps[0], u.conj().T @ source_jumps[0] @ u, atol=1e-12)
    state = np.arange(1, 10) + 1j
    state /= np.linalg.norm(state)
    rho = np.outer(state, state.conj())

    def derivative(h, channels, rho):
        value = -2j * np.pi * (h @ rho - rho @ h)
        for c in channels:
            cc = c.conj().T @ c
            value += c @ rho @ c.conj().T - .5 * (cc @ rho + rho @ cc)
        return value

    expected = u.conj().T @ derivative(_matrix(source), source_jumps, u @ rho @ u.conj().T) @ u
    np.testing.assert_allclose(derivative(_matrix(result.chip), retained_jumps, rho), expected, atol=1e-12)


def test_repeated_exact_edge_reductions_compose_without_losing_prior_corrections():
    source = _chip(topology="spectator")
    first = eliminate(source, "selected", method="exact")
    second = eliminate(first.chip, "remaining", method="exact")
    embedding = np.asarray(first.mapping.embedding @ second.mapping.embedding)
    np.testing.assert_allclose(embedding.conj().T @ _matrix(source) @ embedding,
                               _matrix(second.chip), atol=1e-11)


def test_intrinsic_dynamic_edge_cannot_be_silently_discarded():
    from quchip import CosineCoefficient, TimeDependentTerm

    class ModulatedEdge(Capacitive):
        def time_terms(self, a, b, p):
            return (TimeDependentTerm(a.x * b.x, CosineCoefficient(amplitude=.01, frequency=.1)),)

    a = Resonator(freq=5., levels=3, label="a")
    b = Resonator(freq=7., levels=3, label="b")
    source = Chip([a, b], [ModulatedEdge(a, b, g=.05, label="edge")])
    with pytest.raises(NotImplementedError, match="intrinsic time-dependent"):
        eliminate(source, "edge")


@pytest.mark.validation
def test_zero_edge_does_not_rotate_degenerate_native_charge_levels():
    method = "exact"
    from quchip import ChargeBasisTransmon

    q = ChargeBasisTransmon(E_C=.25, E_J=12., levels=4, num_basis=61, label="q")
    r = Resonator(freq=7.1, levels=2, label="r")
    source = Chip([q, r], [Capacitive(q, r, g=0., label="edge")])
    result = eliminate(source, "edge", method=method)
    np.testing.assert_allclose(_matrix(result.chip), _matrix(source), atol=2e-11)
    np.testing.assert_allclose(result.mapping.embedding, np.eye(122), atol=1e-12)
