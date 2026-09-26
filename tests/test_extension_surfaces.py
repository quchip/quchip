"""Tests for declarative device, dissipation, and batch-handle extensions."""

from __future__ import annotations

from quchip.approximations import RWA

import numpy as np
import pytest

from quchip import (
    Chip, ChargeDrive, CollapseChannel, CouplingModel, CustomSpace, DuffingTransmon,
    Exact, Gaussian, PortNetwork, QuantumSequence, Resonator, eliminate,
)
from quchip.extensions import SpinHalf as ReferenceSpin
from quchip.declarative.models import DeviceModel
from quchip.declarative.parameters import Scalar, parameter


class SpinHalf(DeviceModel):
    """Minimal spin-like device authored purely on the sigma surface."""

    _type_prefix = "spin"
    _default_levels = 2

    freq: Scalar = parameter(positive=True)

    def local_hamiltonian(self, op, p):
        # H = -(freq/2)·sigma_z, so |1> sits `freq` above |0>.
        return (-0.5 * p.freq) * op.sigma_z


def test_sigma_ops_author_a_spin_device():
    """A sigma_z-authored device gives a diagonal Hamiltonian split by ±freq/2."""
    spin = SpinHalf(freq=4.0, levels=2)
    chip = Chip([spin])
    h = np.asarray(chip.hamiltonian().matrix(backend=chip.backend))
    np.testing.assert_allclose(h, np.diag([-2.0, 2.0]), atol=1e-12)


class DampedTransmon(DuffingTransmon):
    """Transmon with an extra device-declared two-photon-loss channel."""

    _type_prefix = "damped"

    two_photon_rate: Scalar = parameter(
        default=None,
        positive=True,
        noise=True,
    )

    def dissipation(self, op, p):
        channels = super().dissipation(op, p)
        if self.two_photon_rate is None:
            return channels
        return channels + (CollapseChannel(op.a @ op.a, p.two_photon_rate, "two_photon_loss"),)


def test_dissipation_hook_composes_with_common_device_channels():
    """A device dissipation hook composes with the built-in T1 channel."""
    quiet = DampedTransmon(freq=5.0, anharmonicity=-0.3, levels=3)
    assert quiet.collapse_operators() == []

    noisy = DampedTransmon(freq=5.0, anharmonicity=-0.3, levels=3, T1=10_000.0, two_photon_rate=1e-4)
    ops = noisy.collapse_operators()
    # Built-in T1 relaxation channel plus the declared two-photon channel.
    assert len(ops) == 2
    from quchip.backend import get_default_backend

    a2 = np.asarray(get_default_backend().to_array(ops[-1]))
    lowering = np.diag(np.sqrt(np.arange(1, 3)).astype(complex), k=1)
    expected = np.sqrt(1e-4) * (lowering @ lowering)
    np.testing.assert_allclose(a2, expected, atol=1e-12)


def _sequence_with_pulse():
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3)
    chip = Chip([q], frame="rotating", approximation=RWA())
    drive = ChargeDrive(target=q)
    chip.wire(drive)
    seq = QuantumSequence(chip)
    handle = seq.schedule(drive, envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=3.0), freq=5.0)
    return q, drive, seq, handle


def test_stale_handle_is_rejected_not_silently_misapplied():
    """A batch handle raises once its owning sequence has been mutated out from under it."""
    q, drive, seq, handle = _sequence_with_pulse()
    seq.delay(q, 5.0)
    seq._entries.pop(0)  # shifts entry indices out from under the handle
    with pytest.raises(RuntimeError, match="modified after the handle"):
        handle.vary("amplitude", [0.01, 0.02])


def test_duplicate_axis_names_rejected_upfront():
    """Building a batch with duplicate axis names across handles raises before solving."""
    q, drive, seq, handle = _sequence_with_pulse()
    ax1 = handle.vary("amplitude", [0.01, 0.02], name="amps")
    ax2 = handle.vary("freq", [4.9, 5.1], name="amps")
    with pytest.raises(ValueError, match="Duplicate batch axis name"):
        seq.build_batch(ax1, ax2, tlist=np.linspace(0.0, 20.0, 11))


# No oscillator or circuit aliases: these names are the model's vocabulary.
class AtomicSpin(ReferenceSpin):
    def local_space(self):
        operators = super().local_space().operators
        return CustomSpace(2, {name: value for name, value in operators.items()
                               if name.startswith('sigma_') or name == 'I'})


class AtomCavityExchange(CouplingModel):
    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        return p.g * (a.sigma_minus * b.adag + a.sigma_plus * b.a)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_custom_operator_names_supply_noise_ports_and_observables(backend):
    """An alias-free spin uses its declared transitions for T1, T2 and ports."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    spin = AtomicSpin(5.0, T1=100.0, T2=150.0, thermal_occupation=0.2, label="atom")
    network = PortNetwork()
    network.port("default", target=spin, rate=0.03)
    network.port("named", target=spin, operator="sigma_minus", rate=0.04)
    chip = Chip([spin], port_network=network, backend=backend)
    resolved = chip.resolve()
    channels = sorted(resolved.collapse_terms, key=lambda term: term.source != "atom")
    matrices = [np.asarray(channel.operator.to_dense()) for channel in channels]
    rates = [float(channel.rate) for channel in channels]
    down = np.array([[0, 1], [0, 0]])
    np.testing.assert_allclose(matrices[:3], [down, down.T, np.diag([0, 1])], atol=1e-14)
    np.testing.assert_allclose(rates[:3], [0.012, 0.002, 2 * (1 / 150 - 1 / 200)])
    np.testing.assert_allclose(matrices[3:], [down, down], atol=1e-14)
    np.testing.assert_allclose(chip.backend.to_array(chip.observable(spin, "sigma_minus")), down)


@pytest.mark.parametrize("method", ["sw", "exact"])
def test_alias_free_atoms_retain_cavity_mediated_exchange(method):
    """Generic mediated edges preserve the retained matrix and serialize with the chip."""
    a = AtomicSpin(5.0, label="a")
    b = AtomicSpin(5.1, label="b")
    cavity = Resonator(7.0, levels=3, label="c")
    chip = Chip([a, b, cavity], [AtomCavityExchange(a, cavity, g=0.02),
                               AtomCavityExchange(b, cavity, g=0.03)])
    result = eliminate(chip, cavity, method=method)
    h = np.asarray(result.chip.hamiltonian().matrix())
    j = result.effective_params["exchange"]["j_eff"]
    np.testing.assert_allclose(h[1, 2], j, atol=1e-12)
    if method == "sw":
        np.testing.assert_allclose(j, .02 * .03 / 2 * (1 / -2 + 1 / -1.9), atol=1e-12)
    else:
        projected = result.mapping.project_operator(chip.resolve(approximation=Exact()).hamiltonian().matrix())
        np.testing.assert_allclose(h, chip.backend.to_array(projected), atol=1e-12)
    restored = Chip.from_dict(result.chip.to_dict())
    np.testing.assert_allclose(restored.hamiltonian().matrix(), h, atol=1e-12)


class RadiatingAtom(AtomicSpin):
    def dissipation(self, op, p):
        return (CollapseChannel(2 * op.sigma_minus, 1 / p.T1, "dipole_decay"),
                CollapseChannel(op.sigma_z, 0.03, "dephasing"))


def test_elimination_diagnostics_use_declared_channels():
    """The decay summary and inherited jumps share their operator and normalization."""
    atom = RadiatingAtom(7.0, T1=100.0, label="atom")
    mode = Resonator(5.0, levels=2, label="mode")
    chip = Chip([atom, mode], [AtomCavityExchange(atom, mode, g=0.02)])
    result = eliminate(chip, atom)
    diagnostics = result.effective_params[mode]
    np.testing.assert_allclose(diagnostics["kappa"], 0.04, atol=1e-14)
    np.testing.assert_allclose(diagnostics["purcell_rate"], 0.04 * np.sin(.01) ** 2, rtol=1e-12)
    channels = result.chip.effective_terms[-1].channels
    for original, inherited in zip(chip.resolve().collapse_terms, channels, strict=True):
        expected = result.mapping.project_operator(original.operator.to_dense())
        np.testing.assert_allclose(inherited.operator, chip.backend.to_array(expected), atol=1e-12)
        np.testing.assert_allclose(inherited.rate, original.rate)


class ThreeLevelAtom(DeviceModel):
    """Unequal dipole elements in an authored basis ordered excited, upper, ground."""

    freq: Scalar = parameter(positive=True)
    dipole: Scalar = parameter(default=0.4)

    def local_space(self):
        from quchip import qnp

        down = qnp.asarray([[0., self.dipole, 0.], [0., 0., 0.], [1., 0., 0.]])
        return CustomSpace(3, {"I": np.eye(3), "energy": np.diag([1., 1.7, 0.]),
                               "dipole": down, "excitation": np.diag([1., 2., 0.])})

    def local_hamiltonian(self, op, p):
        return p.freq * op["energy"]

    def lowering_operator(self):
        return self.local_operator("dipole")

    def number_operator(self):
        return self.local_operator("excitation")


@pytest.mark.optional_backend
def test_multilevel_dipole_rates_and_gradients_keep_declared_matrix_elements():
    """Energy ordering does not replace a multilevel dipole with an oscillator ladder."""
    import jax
    import jax.numpy as jnp

    pytest.importorskip("dynamiqs")

    def rates(dipole, basis="native"):
        atom = ThreeLevelAtom(1.0, dipole=dipole, levels=3, T1=100, label="atom")
        resolved = Chip([atom], basis=basis, backend="dynamiqs").resolve()
        channel = resolved.collapse_terms[0]
        vectors = resolved.bases["atom"].energy_to_solver()
        matrix = channel.operator.to_dense()
        if vectors is not None:
            matrix = vectors.conj().T @ matrix @ vectors
        return channel.rate * jnp.abs(matrix) ** 2

    for basis in ("native", "eigen"):
        expected = np.zeros((3, 3))
        expected[0, 1], expected[1, 2] = .01, .0016
        np.testing.assert_allclose(rates(.4, basis), expected, atol=1e-14)
    derivative = jax.jit(jax.grad(lambda dipole: rates(dipole)[1, 2]))(.4)
    np.testing.assert_allclose(derivative, 2 * .4 / 100, rtol=1e-12)

    # Compiled batches must rebind the dipole, while built problems retain it.
    from quchip.engine import solve_problem

    atom = ThreeLevelAtom(1.0, levels=3, T1=100, label="atom")
    sequence = QuantumSequence(Chip([atom], basis="eigen", frame="rotating", backend="dynamiqs"))
    problem = sequence.build_problem([0., 10.], initial_state={"atom": 2})
    results = sequence.simulate_batch(sequence.vary("atom.dipole", [.4, .8]),
                                     tlist=[0., 10.], initial_state={"atom": 2}, progress=False)
    np.testing.assert_allclose(results.population("atom", level=2, reduce="last"),
                               np.exp(-10 * np.array([.4, .8]) ** 2 / 100), atol=2e-7)
    atom.dipole = .9
    np.testing.assert_allclose(solve_problem(problem).population("atom", 2)[-1],
                               np.exp(-10 * .4 ** 2 / 100), atol=2e-7)
