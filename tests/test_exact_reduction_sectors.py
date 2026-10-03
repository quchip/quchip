"""Reductions of excitation-conserving chips keep exact sectors through jit, gradients and backends."""
import jax
import numpy as np
import pytest

from quchip import (
    Capacitive, ChargeDrive, Chip, DuffingTransmon, Gaussian, PortNetwork, QuantumSequence, Resonator, RWA,
    eliminate, simulate,
)
from quchip.chip.sw import excitation_sectors

pytestmark = pytest.mark.optional_backend
pytest.importorskip("dynamiqs")


def _bus_chip(alphas=(-0.262, -0.264), levels=(3, 3, 2), g=0.03, backend="dynamiqs"):
    q1 = DuffingTransmon(freq=5.326, anharmonicity=alphas[0], levels=levels[0], label="q1", T1=30000.)
    q2 = DuffingTransmon(freq=5.192, anharmonicity=alphas[1], levels=levels[1], label="q2", T1=30000.)
    bus = Resonator(freq=6.298, levels=levels[2], label="bus", internal_quality_factor=2e5)
    return Chip([q1, q2, bus], [Capacitive(q1, bus, g=g), Capacitive(q2, bus, g=0.03)],
                frame=5.2, approximation=RWA(), backend=backend)


def _readout_chip(alphas=(-0.262, -0.264), levels=(4, 4, 4, 3, 3), g=0.04, backend="dynamiqs", ports=False):
    q1 = DuffingTransmon(freq=5.326, anharmonicity=alphas[0], levels=levels[0], label="q1", T1=30000.)
    q2 = DuffingTransmon(freq=5.192, anharmonicity=alphas[1], levels=levels[1], label="q2", T1=30000.)
    bus = Resonator(freq=6.298, levels=levels[2], label="bus", internal_quality_factor=2e5)
    r1 = Resonator(freq=6.558, levels=levels[3], label="r1", internal_quality_factor=8e4)
    r2 = Resonator(freq=6.657, levels=levels[4], label="r2", internal_quality_factor=1.2e5)
    couplings = [Capacitive(q1, bus, g=0.03), Capacitive(q2, bus, g=0.03),
                 Capacitive(q1, r1, g=g, label="q1_r1"), Capacitive(q2, r2, g=0.04)]
    network = None
    if ports:
        network = PortNetwork(label="feed")
        network.port("r1", target=r1, rate=0.01)
        network.port("r2", target=r2, rate=0.01)
    return Chip([q1, q2, bus, r1, r2], couplings, port_network=network, frame=5.2, approximation=RWA(),
                backend=backend)


def _off_band(operator, dims, changes):
    """Entries whose total level change, column minus row, is not in ``changes``."""
    levels = excitation_sectors(tuple(dims))
    change = levels[None, :] - levels[:, None]
    return np.abs(np.asarray(operator))[~np.isin(change, sorted(changes))]


def _jumps(chip, resolved):
    return [np.asarray(chip.backend.to_array(op)) for op in chip.backend._collapse_operators(resolved)]


def _assert_sector_exact(reduced):
    (terms,) = reduced.effective_terms
    assert terms.excitation_changes == {"internal_photon_loss": {1}}
    assert np.all(_off_band(terms.hamiltonian, terms.dims, {0}) == 0.0)
    for channel in terms.channels:
        assert np.all(_off_band(channel.operator, terms.dims, terms.excitation_changes[channel.name]) == 0.0)
    resolved = reduced.resolve()
    jumps = _jumps(reduced, resolved)
    assert len(jumps) == 3
    for jump in jumps:
        # Surviving T1 and inherited bus loss each lower the total excitation by one.
        assert np.all(_off_band(jump, reduced.dims, {1}) == 0.0)
    return resolved


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_bus_reduction_keeps_every_retained_operator_inside_its_sector(method):
    """Retained terms and resolved jumps of an RWA reduction have exact zeros outside their sectors."""
    _assert_sector_exact(eliminate(_bus_chip(), "bus", method=method).chip)


def test_exact_reduction_resolves_identically_inside_jit():
    """An exactly reduced chip resolves inside jit to the eager Hamiltonian and jump operators."""
    reduced = eliminate(_bus_chip(), "bus", method="exact").chip
    eager = reduced.resolve()
    eager_h = np.asarray(eager.hamiltonian().matrix(backend=reduced.backend))
    eager_jumps = _jumps(reduced, eager)

    unresolved = eliminate(_bus_chip(), "bus", method="exact").chip
    assert jax.jit(lambda x: (unresolved.resolve(), x)[1])(1.0) == 1.0

    fresh = eliminate(_bus_chip(), "bus", method="exact").chip

    def resolved_at(t1):
        resolved = fresh.with_params({"q1.T1": t1}).resolve()
        jumps = [fresh.backend.to_array(op) for op in fresh.backend._collapse_operators(resolved)]
        return resolved.hamiltonian().matrix(backend=fresh.backend), jumps

    h, jumps = jax.jit(resolved_at)(30000.)
    np.testing.assert_allclose(h, eager_h, atol=1e-12)
    for jump, expected in zip(jumps, eager_jumps, strict=True):
        np.testing.assert_allclose(jump, expected, atol=1e-12)


def _resolved_matrices(chip):
    resolved = chip.resolve()
    jumps = [chip.backend.to_array(op) for op in chip.backend._collapse_operators(resolved)]
    return resolved.hamiltonian().matrix(backend=chip.backend), jumps


def _assert_jit_matches_eager(build, value):
    eager_h, eager_jumps = _resolved_matrices(build(value))
    h, jumps = jax.jit(lambda x: _resolved_matrices(build(x)))(value)
    np.testing.assert_allclose(h, eager_h, atol=1e-12)
    assert len(jumps) == len(eager_jumps)
    for jump, expected in zip(jumps, eager_jumps, strict=True):
        np.testing.assert_allclose(jump, expected, atol=1e-12)


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_kept_ports_resolve_identically_inside_jit(method):
    """Ports on surviving modes keep the lowering change of their authored operator inside jit."""
    reduced = eliminate(_readout_chip(levels=(2, 2, 2, 2, 2), ports=True), "bus", method=method).chip
    # Nothing traced but the resolve itself, then a traced survivor parameter.
    assert jax.jit(lambda x: (reduced.resolve(), x)[1])(1.0) == 1.0
    _assert_jit_matches_eager(lambda t1: reduced.with_params({"q1.T1": t1}), 30000.)


def test_driven_reduction_in_a_common_frame_has_only_its_drive_carriers():
    """Exactly static retained bands stay static, and drive bands sharing a carrier lower to one term."""
    reduced = eliminate(_readout_chip(levels=(4, 4, 2, 2, 2)), "bus", method="exact").chip
    drives = [ChargeDrive(target=reduced[label], label=f"drive_{label}") for label in ("q1", "q2")]
    reduced.wire(*drives)
    sequence = QuantumSequence(reduced)
    for drive in drives:
        sequence.schedule(drive, envelope=Gaussian(duration=40.0, amplitude=0.02, sigmas=4.0),
                          freq=float(reduced.freq(reduced[drive.target_label])))

    rotating, lab = sequence.resolve(), sequence.resolve(frame="lab")

    # Level changes up to ±3 used to leave 1e-14 GHz carriers on static
    # couplings and split equal drive carriers into separate terms.
    assert not rotating.slh.H.dynamic_terms
    assert len(rotating.applied_hamiltonian.dynamic_terms) == len(lab.applied_hamiltonian.dynamic_terms)


def test_transformed_port_declares_its_excitation_change_inside_jit():
    """A port transformed by an elimination traced inside jit resolves in the rotating frame."""
    def reduced(g):
        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=20000.)
        resonator = Resonator(freq=6.0, levels=3, label="r")
        network = PortNetwork(label="m")
        network.port("r", target=resonator, rate=0.05)
        chip = Chip([qubit, resonator], [Capacitive(qubit, resonator, g=g)], port_network=network,
                    frame=5.0, approximation=RWA(), backend="dynamiqs")
        return eliminate(chip, "r", method="exact").chip

    _assert_jit_matches_eager(reduced, 0.06)


@pytest.mark.validation
def test_issue_operating_points_reduce_exactly_and_resolve():
    """Operating points that previously left stray bands reduce exactly and resolve in the rotating frame."""
    # Before sector-wise diagonalization this reported point left 2.6e-12
    # off-band entries that made the rotating-frame collapse check fail.
    alphas = (-0.2618228086648377, -0.2636511734115886)
    _assert_sector_exact_readout(eliminate(_readout_chip(alphas), "bus", method="exact").chip)


def _assert_sector_exact_readout(reduced):
    (terms,) = reduced.effective_terms
    assert np.all(_off_band(terms.hamiltonian, terms.dims, {0}) == 0.0)
    for channel in terms.channels:
        assert np.all(_off_band(channel.operator, terms.dims, terms.excitation_changes[channel.name]) == 0.0)
    for jump in _jumps(reduced, reduced.resolve()):
        assert np.all(_off_band(jump, reduced.dims, {1}) == 0.0)


@pytest.mark.validation
def test_gradient_through_exact_elimination_matches_finite_difference():
    """A Hamiltonian element differentiated through exact elimination matches a central difference."""
    def element(g):
        reduced = eliminate(_bus_chip(g=g), "bus", method="exact").chip
        h = reduced.resolve().hamiltonian().matrix(backend=reduced.backend)
        q1 = int(np.ravel_multi_index((1, 0), tuple(reduced.dims)))
        q2 = int(np.ravel_multi_index((0, 1), tuple(reduced.dims)))
        return h[q1, q1].real + h[q1, q2].real

    g, step = 0.03, 1e-5
    finite_difference = (element(g + step) - element(g - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(element)(g), finite_difference, rtol=1e-7)


@pytest.mark.validation
def test_qutip_reduced_master_equation_stays_sparse_and_matches_dynamiqs(monkeypatch):
    """QuTiP stores reduced-model jumps as CSR and its master equation matches dense storage and dynamiqs."""
    import dynamiqs
    import quchip.backend.qutip as qutip_backend

    def final_state(backend, options):
        reduced = eliminate(_readout_chip(levels=(2, 2, 2, 2, 2), backend=backend), "bus", method="exact").chip
        psi = np.zeros(int(np.prod(reduced.dims)), dtype=complex)
        psi[int(np.ravel_multi_index((1, 0, 0, 0), tuple(reduced.dims)))] = 1.0
        result = simulate(reduced, [], np.linspace(0.0, 20.0, 5), solver="mesolve", initial_state=psi,
                          states="final", options=options)
        return reduced, np.asarray(reduced.backend.to_array(result.state_at(result.times[-1])))

    tight = {"atol": 1e-12, "rtol": 1e-12, "nsteps": 10**7}
    reduced, sparse = final_state("qutip", tight)
    jumps = reduced.backend._collapse_operators(reduced.resolve())
    assert {type(jump.data).__name__ for jump in jumps} == {"CSR"}
    monkeypatch.setattr(qutip_backend, "_CSR_MAX_FILL", -1.0)
    _, dense = final_state("qutip", tight)
    _, reference = final_state("dynamiqs", {"method": dynamiqs.method.Tsit5(atol=1e-12, rtol=1e-12, max_steps=10**7)})
    np.testing.assert_allclose(sparse, dense, atol=1e-11)
    np.testing.assert_allclose(sparse, reference, atol=1e-7)
