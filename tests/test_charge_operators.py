"""Capacitive couplings and charge drives act through each device's charge operator."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import (
    Capacitive,
    ChargeBasisTransmon,
    Chip,
    DuffingTransmon,
    EigenbasisDevice,
    Exact,
    FluxTunableTransmon,
    Port,
    PortNetwork,
    RWA,
    Resonator,
)
from quchip.control.drive import ChargeDrive, FluxDrive, PhaseDrive
from quchip.control.signal import AnalyticSignal
from quchip.declarative import CouplingModel, DeviceModel, LocalOps, Scalar, parameter
from quchip.declarative.expr import PhysicsExpr
from quchip.engine.ir import Constant


pytestmark = pytest.mark.unit


_UNIT_SIGNAL = AnalyticSignal(Constant(1.0 + 0.0j))


def _lowering(levels: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1.0, levels)), 1)


def _fock_charge(levels: int) -> np.ndarray:
    lowering = _lowering(levels)
    return 1j * (lowering - lowering.T)


def _charge_scale(flux: float, asymmetry: float) -> float:
    return (np.cos(np.pi * flux) ** 2 + asymmetry**2 * np.sin(np.pi * flux) ** 2) ** 0.125


def _drive_operator(drive_type: type, device: object) -> np.ndarray:
    return np.asarray(drive_type(target=device).hamiltonian(device, _UNIT_SIGNAL).matrix(t=0.0))


class _PlainMode(DeviceModel):
    """Fock mode that declares no charge operator."""

    freq: Scalar = parameter(positive=True, unit="GHz")

    def local_hamiltonian(self, op: LocalOps, p: object):
        return p.freq * op.n


class _DoubledChargeTransmon(DuffingTransmon):
    """Duffing transmon whose declared charge operator is twice the default."""

    def charge_coupling_operator(self):
        return 2.0 * super().charge_coupling_operator()


class _QuadratureCoupling(CouplingModel):
    """Reference coupling ``g (a + a†)(b + b†)`` that ignores the devices' charge operators."""

    _type_prefix = "quadrature"

    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        return p.g * a.x * b.x


_DEVICES = {
    "duffing": (
        lambda: DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q"),
        lambda: _fock_charge(3),
    ),
    "resonator": (
        lambda: Resonator(freq=6.5, levels=3, label="q"),
        lambda: _fock_charge(3),
    ),
    "flux_tunable": (
        lambda: FluxTunableTransmon(
            freq=5.0, anharmonicity=-0.25, flux_bias=0.3, asymmetry=0.2, levels=3, label="q"
        ),
        lambda: _charge_scale(0.3, 0.2) * _fock_charge(3),
    ),
    "charge_basis": (
        lambda: ChargeBasisTransmon(E_C=0.2, E_J=10.0, levels=3, num_basis=21, label="q"),
        lambda: np.diag(np.arange(-10.0, 11.0)),
    ),
}


@pytest.mark.parametrize("make_device, expected_charge", _DEVICES.values(), ids=_DEVICES.keys())
def test_capacitive_coupling_and_charge_drive_share_the_device_charge(make_device, expected_charge):
    """A capacitive coupling and a charge drive both act through the device's charge operator."""
    device = make_device()
    partner = Resonator(freq=7.0, levels=2, label="r")
    charge = expected_charge()

    coupling = Capacitive(device, partner, g=0.02, label="qr")

    np.testing.assert_allclose(_drive_operator(ChargeDrive, device), charge, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(coupling.interaction_hamiltonian().matrix()),
        0.02 * np.kron(charge, _fock_charge(2)),
        atol=1e-12,
    )


def test_flux_tunable_phase_operator_takes_the_inverse_charge_scale():
    """Away from zero flux the phase operator is (a + a†)/s and the flux operator stays n."""
    device = FluxTunableTransmon(
        freq=5.0, anharmonicity=-0.25, flux_bias=0.3, asymmetry=0.2, levels=3, label="q"
    )
    lowering = _lowering(3)

    np.testing.assert_allclose(
        _drive_operator(PhaseDrive, device),
        (lowering + lowering.T) / _charge_scale(0.3, 0.2),
        atol=1e-12,
    )
    np.testing.assert_allclose(_drive_operator(FluxDrive, device), np.diag([0.0, 1.0, 2.0]), atol=1e-12)


@pytest.mark.parametrize("name", ["charge", "phase"])
def test_flux_tunable_named_observable_and_port_take_the_charge_scale(name):
    """An observable and a port named "charge" or "phase" use the flux-scaled operator."""
    device = FluxTunableTransmon(
        freq=5.0, anharmonicity=-0.25, flux_bias=0.3, asymmetry=0.2, levels=3, label="q"
    )
    lowering = _lowering(3)
    scale = _charge_scale(0.3, 0.2)
    expected = scale * _fock_charge(3) if name == "charge" else (lowering + lowering.T) / scale
    port = Port(device, rate=0.02, operator=name, label="p")
    chip = Chip([device], port_network=PortNetwork.from_ports([port]))

    observable = chip.backend.to_array(chip.observable("q", name))
    port_operator = chip.resolve(frame="lab").port_terms[0].operator.to_dense()

    np.testing.assert_allclose(np.asarray(observable), expected, atol=1e-12)
    np.testing.assert_allclose(np.asarray(port_operator), expected, atol=1e-12)


def test_declared_charge_operator_reaches_capacitive_coupling():
    """A device that overrides charge_coupling_operator() couples capacitively through its override."""
    device = _DoubledChargeTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    partner = Resonator(freq=7.0, levels=2, label="r")

    interaction = Capacitive(device, partner, g=0.02).interaction_hamiltonian().matrix()

    np.testing.assert_allclose(
        np.asarray(interaction), 0.02 * np.kron(2.0 * _fock_charge(3), _fock_charge(2)), atol=1e-12
    )


def test_device_without_charge_declaration_couples_through_its_space_quadrature():
    """A Fock device without charge_coupling_operator() couples through a + a†."""
    device = _PlainMode(freq=5.0, levels=3, label="m")
    partner = Resonator(freq=7.0, levels=2, label="r")
    lowering = _lowering(3)

    interaction = Capacitive(device, partner, g=0.02).interaction_hamiltonian().matrix()

    np.testing.assert_allclose(
        np.asarray(interaction), 0.02 * np.kron(lowering + lowering.T, _fock_charge(2)), atol=1e-12
    )


def test_capacitive_coupling_requires_an_imported_charge_operator():
    """An imported device couples through its supplied charge operator and rejects a missing one."""
    partner = Resonator(freq=7.0, levels=2, label="r")
    charge = np.array([[0.0, 0.6, 0.0], [0.6, 0.0, 0.9], [0.0, 0.9, 0.0]])
    imported = EigenbasisDevice([0.0, 5.0, 9.8], charge_operator=charge, label="imported")
    bare = EigenbasisDevice([0.0, 5.0, 9.8], label="bare")

    interaction = Capacitive(imported, partner, g=0.02).interaction_hamiltonian().matrix()

    np.testing.assert_allclose(np.asarray(interaction), 0.02 * np.kron(charge, _fock_charge(2)), atol=1e-12)
    with pytest.raises(ValueError, match="capacitive couplings require"):
        Capacitive(bare, partner, g=0.02).interaction_hamiltonian()


class _ChargeQuadratureCoupling(CouplingModel):
    """Reference coupling ``g Q_a (b + b†)`` with a Fock quadrature on the second device."""

    _type_prefix = "charge_quadrature"

    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        return p.g * a.charge * b.x


def _pair_hamiltonian(make_first, coupling_type: type, approximation: object) -> np.ndarray:
    first = make_first()
    second = DuffingTransmon(freq=6.2, anharmonicity=-0.3, levels=3, label="b")
    chip = Chip([first, second], [coupling_type(first, second, g=0.05)], frame="lab", approximation=approximation)
    return np.asarray(chip.hamiltonian().matrix())


def _duffing_pair_first():
    return DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="a")


def _charge_basis_pair_first():
    return ChargeBasisTransmon(E_C=0.25, E_J=12.5, levels=3, num_basis=21, basis="eigen", label="a")


@pytest.mark.parametrize(
    "make_first, reference_type",
    [(_duffing_pair_first, _QuadratureCoupling), (_charge_basis_pair_first, _ChargeQuadratureCoupling)],
    ids=["fock", "charge_basis"],
)
@pytest.mark.parametrize("approximation", [Exact(), RWA()], ids=["exact", "rwa"])
def test_capacitive_spectra_match_the_quadrature_coupling(make_first, reference_type, approximation):
    """The Fock charge i(b - b†) leaves the spectrum of the (b + b†) quadrature coupling unchanged."""
    capacitive = _pair_hamiltonian(make_first, Capacitive, approximation)
    reference = _pair_hamiltonian(make_first, reference_type, approximation)

    np.testing.assert_allclose(np.linalg.eigvalsh(capacitive), np.linalg.eigvalsh(reference), rtol=0, atol=1e-12)


def test_fock_capacitive_coupling_flips_only_the_counter_rotating_sign():
    """Between Fock devices the RWA Hamiltonian is unchanged and Exact negates the a b and a†b† terms."""
    lowering = _lowering(3)
    counter = np.kron(lowering, lowering) + np.kron(lowering.T, lowering.T)

    np.testing.assert_allclose(
        _pair_hamiltonian(_duffing_pair_first, Capacitive, RWA()),
        _pair_hamiltonian(_duffing_pair_first, _QuadratureCoupling, RWA()),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _pair_hamiltonian(_duffing_pair_first, Capacitive, Exact())
        - _pair_hamiltonian(_duffing_pair_first, _QuadratureCoupling, Exact()),
        -2.0 * 0.05 * counter,
        atol=1e-12,
    )


def test_complex_literals_render_with_the_imaginary_unit():
    """LaTeX shows complex coefficients with i rather than Python's j suffix."""
    device = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")

    assert device.charge_coupling_operator().latex() == r"i\,(\hat a_{q} - \hat a^\dagger_{q})"
    assert [PhysicsExpr.literal(value).latex() for value in (1j, -1j, 2.5j, 1 - 2j, 0.5 + 0j)] == [
        "i",
        "-i",
        "2.5i",
        "(1-2i)",
        "0.5",
    ]
