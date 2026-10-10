"""Projection of multi-device operators into captured local eigenbases."""

from __future__ import annotations

import tracemalloc

import jax
import numpy as np
import pytest

from quchip import Capacitive, Chip, Exact, Fluxonium
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


class _PowerOfSum(CouplingModel):
    """The power (φ_a + φ_b)^16, which expands into 65536 single-device products."""

    g: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        power = a.phi * b.I + a.I * b.phi
        for _ in range(4):
            power = power @ power
        return p.g * power


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


def test_factorized_projection_equals_the_dense_native_projection() -> None:
    """Sums, scalar factors and same-device products project as on the dense native product space."""
    fa, fb = _fluxonium_pair(num_basis=41)
    chip = Chip([fa, fb], couplings=[_MixedCoupling(fa, fb, g=0.05)], basis="eigen")

    # The native spectrum spans about 100 GHz, so float64 round-off in either projection stays near 1e-14.
    np.testing.assert_allclose(
        np.asarray(chip.resolve(approximation=Exact()).hamiltonian().matrix()),
        _dense_projection(chip),
        rtol=0.0,
        atol=1e-12,
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


def test_dense_route_checks_memory_before_the_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators that do not factor, or that expand past the dense operator, take the memory-checked dense route."""
    fa, fb = _fluxonium_pair(num_basis=41)
    monkeypatch.setattr(_memory, "available_memory_bytes", lambda: 10**6)

    factorized = Chip([fa, fb], couplings=[Capacitive(fa, fb, g=0.5)], basis="eigen")
    assert factorized.resolve().dims == (5, 5)
    # The 65536 products of (φ_a + φ_b)^16 would hold 3.5 GB, and the dense operator holds 45 MB.
    for coupling in (_OpaqueCapacitive(fa, fb, g=0.5), _PowerOfSum(fa, fb, g=1e-6)):
        chip = Chip([fa, fb], couplings=[coupling], basis="eigen")
        with pytest.raises(MemoryError, match="dense projection of an operator on fa, fb at native dimension N = 1681"):
            chip.resolve()


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
