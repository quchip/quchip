"""Projection of multi-device operators into captured local eigenbases."""

from __future__ import annotations

import tracemalloc

import jax
import numpy as np
import pytest

from quchip import Bath, Capacitive, Chip, Exact, Fluxonium, Resonator
from quchip.backend import _memory
from quchip.declarative import CouplingModel, Scalar, parameter


class _MixedCoupling(CouplingModel):
    """Sums, scalar factors and same-device products of single-device operators."""

    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        exchange = a.charge * b.charge
        squared = (a.phi * b.charge) @ (a.phi * b.charge)
        return p.g * exchange - 0.3 * p.g * squared + 0.1 * (a.n2 * b.I)


class _OpaqueCapacitive(CouplingModel):
    """The capacitive interaction authored as one opaque two-device callable."""

    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        del p

        def charge_product(g):
            return g * np.kron(np.asarray(a.space.matrix("n")), np.asarray(b.space.matrix("n")))

        return charge_product


def _fluxonium_pair(**options) -> tuple[Fluxonium, Fluxonium]:
    return (
        Fluxonium(E_C=1.0, E_J=4.0, E_L=0.5, phi_ext=0.5, levels=5, label="fa", **options),
        Fluxonium(E_C=1.0, E_J=5.0, E_L=0.5, phi_ext=0.5, levels=5, label="fb", **options),
    )


def _dense_projection(chip: Chip) -> np.ndarray:
    """Project a two-device lab-frame Hamiltonian through its dense native product space."""
    first, second = (np.asarray(device.unresolved_hamiltonian().matrix()) for device in chip.devices)
    (coupling,) = chip.couplings
    native = (
        np.kron(first, np.eye(len(second)))
        + np.kron(np.eye(len(first)), second)
        + np.asarray(coupling.interaction_hamiltonian().matrix())
    )
    bases = chip.resolve().bases
    transform = np.kron(*(np.asarray(bases[device.label].vectors) for device in chip.devices))
    return transform.conj().T @ native @ transform


@pytest.mark.parametrize("num_basis", [41, pytest.param(61, marks=pytest.mark.validation)])
def test_eigen_capacitive_coupling_equals_the_dense_native_projection(num_basis: int) -> None:
    """A capacitive coupling between eigen-basis fluxoniums equals its dense native-space projection."""
    fa, fb = _fluxonium_pair(num_basis=num_basis)
    chip = Chip([fa, fb], couplings=[Capacitive(fa, fb, g=0.5)], basis="eigen")

    # The native spectrum spans about 100 GHz, so float64 round-off in either projection stays near 1e-14.
    np.testing.assert_allclose(
        np.asarray(chip.resolve(approximation=Exact()).hamiltonian().matrix()),
        _dense_projection(chip),
        rtol=0.0,
        atol=1e-12,
    )


def test_sums_and_same_device_products_equal_the_dense_native_projection() -> None:
    """Each device multiplies its factors in the native basis before projection, beside a native partner."""
    flux = Fluxonium(E_C=1.0, E_J=4.0, E_L=0.5, phi_ext=0.5, levels=5, num_basis=41, basis="eigen", label="fa")
    resonator = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([flux, resonator], couplings=[_MixedCoupling(flux, resonator, g=0.05)])

    assert chip.resolve().bases["r"].kind == "native"
    # The native spectrum spans about 100 GHz, so float64 round-off in either projection stays near 1e-14.
    np.testing.assert_allclose(
        np.asarray(chip.resolve(approximation=Exact()).hamiltonian().matrix()),
        _dense_projection(chip),
        rtol=0.0,
        atol=1e-12,
    )


def test_unfactored_coupling_keeps_the_dense_projection() -> None:
    """An opaque two-device interaction resolves to the same Hamiltonian as its factorized form."""
    fa, fb = _fluxonium_pair(num_basis=41)
    factorized = Chip([fa, fb], couplings=[Capacitive(fa, fb, g=0.5)], basis="eigen")
    opaque = Chip([fa, fb], couplings=[_OpaqueCapacitive(fa, fb, g=0.5)], basis="eigen")

    # The dense and factorized routes differ only by float64 round-off on a 100 GHz native spectrum.
    np.testing.assert_allclose(
        np.asarray(opaque.resolve(approximation=Exact()).hamiltonian().matrix()),
        np.asarray(factorized.resolve(approximation=Exact()).hamiltonian().matrix()),
        rtol=0.0,
        atol=1e-12,
    )


def test_unfactored_coupling_checks_memory_before_the_dense_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only an operator that does not factor needs the dense native product space."""
    fa, fb = _fluxonium_pair(num_basis=41)
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 10**6)

    factorized = Chip([fa, fb], couplings=[Capacitive(fa, fb, g=0.5)], basis="eigen")
    assert factorized.resolve().dims == (5, 5)
    opaque = Chip([fa, fb], couplings=[_OpaqueCapacitive(fa, fb, g=0.5)], basis="eigen")
    with pytest.raises(MemoryError, match="dense projection of an operator on fa, fb at native dimension N = 1681"):
        opaque.resolve()


def test_collective_bath_operators_project_to_level_ladders() -> None:
    """Collective bath sums over eigen-basis devices project to sums of energy-level ladders."""
    fa, fb = _fluxonium_pair(num_basis=41)
    chip = Chip(
        [fa, fb],
        basis="eigen",
        baths=[Bath("collective_decay", rate=1e-4), Bath("correlated_dephasing", rate=1e-4)],
    )
    operators = {term.channel: np.asarray(term.operator.to_dense()) for term in chip.resolve().collapse_terms}

    identity = np.eye(5)
    lowering = np.diag(np.sqrt(np.arange(1.0, 5.0)), 1)
    number = np.diag(np.arange(5.0))
    # Each bath operator is authored as V L V† in the captured basis, so its projection is exact to round-off.
    np.testing.assert_allclose(
        operators["collective_decay"], np.kron(lowering, identity) + np.kron(identity, lowering), atol=1e-12
    )
    np.testing.assert_allclose(
        operators["correlated_dephasing"], np.kron(number, identity) + np.kron(identity, number), atol=1e-12
    )


def test_default_fluxonium_pair_resolves_without_the_dense_native_product_space() -> None:
    """Two coupled default fluxoniums need memory of order num_basis², not num_basis⁴."""
    fa, fb = _fluxonium_pair()
    chip = Chip([fa, fb], couplings=[Capacitive(fa, fb, g=0.5)], basis="eigen")

    tracemalloc.start()
    try:
        chip.freq()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # One dense native product-space array alone holds 16·num_basis⁴ bytes, 410 GB here.
    assert peak < 100 * 16 * fa.num_basis**2

    result = chip.resolve(approximation=Exact())
    local = {}
    for device in (fa, fb):
        vectors = np.asarray(result.bases[device.label].vectors)
        local[device.label] = [
            vectors.conj().T @ np.asarray(matrix) @ vectors
            for matrix in (device.unresolved_hamiltonian().matrix(), device.local_space().matrix("n"))
        ]
    (hamiltonian_a, charge_a), (hamiltonian_b, charge_b) = local["fa"], local["fb"]
    identity = np.eye(5)
    expected = np.kron(hamiltonian_a, identity) + np.kron(identity, hamiltonian_b) + 0.5 * np.kron(charge_a, charge_b)
    # The native spectrum spans 2700 GHz, so float64 round-off in either projection stays near 1e-13.
    np.testing.assert_allclose(np.asarray(result.hamiltonian().matrix()), expected, rtol=0.0, atol=1e-11)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_factorized_projection_derivatives_match_finite_differences() -> None:
    """Derivatives through the coupling scale and the partner's eigenvectors match central differences."""
    pytest.importorskip("dynamiqs")
    fa, fb = _fluxonium_pair(num_basis=31)
    chip = Chip([fa, fb], couplings=[Capacitive(fa, fb, g=0.3, label="c")], basis="eigen", backend="dynamiqs")

    def frequency(path, value):
        return chip.with_params({path: value}).freq("fa")

    step = 1e-4
    for path, value in (("c.g", 0.3), ("fb.E_J", 5.0)):
        reference = (float(frequency(path, value + step)) - float(frequency(path, value - step))) / (2 * step)
        # A central difference with step 1e-4 has a relative truncation error near 1e-7.
        assert float(jax.grad(lambda x: frequency(path, x))(value)) == pytest.approx(reference, rel=1e-5)
        assert abs(reference) > 1e-6
