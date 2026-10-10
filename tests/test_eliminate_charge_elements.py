"""Device elimination reports exchange and validity in the units of an edge between survivors."""

import numpy as np
import pytest

from quchip import Capacitive, ChargeBasisTransmon, Chip, DuffingTransmon, Fluxonium, Resonator, eliminate
from quchip.chip.effective import EffectiveTerms

# |<0|n|1>| is about 1.15 for this charge-basis transmon. The fluxonium's
# <0|n|1> is imaginary.
_PARTNERS = {
    "charge_basis": lambda: ChargeBasisTransmon(E_C=.24, E_J=15., levels=3, basis="eigen", num_basis=31, label="b"),
    "fluxonium": lambda: Fluxonium(E_C=1., E_J=4., E_L=1., levels=3, basis="eigen", num_basis=60, label="b"),
}
_BUSES = {
    "resonator": lambda: Resonator(freq=6.5, levels=3, label="bus"),
    "charge_basis": lambda: ChargeBasisTransmon(E_C=.24, E_J=30., levels=3, basis="eigen", num_basis=31, label="bus"),
}


def _bus_chip(partner, bus="resonator", effective_terms=()):
    a = DuffingTransmon(freq=5., anharmonicity=-.25, levels=3, label="a")
    b = _PARTNERS[partner]()
    mode = _BUSES[bus]()
    couplings = [Capacitive(a, mode, g=.06, label="a_bus"), Capacitive(b, mode, g=.06, label="b_bus")]
    return Chip([a, b, mode], couplings, effective_terms=effective_terms)


def _element(chip, row, column):
    """``<1_row|H|1_column>`` in the devices' energy coordinates, from the resolved chip."""
    resolved = chip.resolve(frame="lab")
    h = np.asarray(resolved.hamiltonian().matrix())
    labels = [device.label for device in chip.devices]

    def index(label):
        return np.ravel_multi_index(tuple(int(name == label) for name in labels), resolved.dims)

    return h[index(row), index(column)]


def _edge_only(result):
    """The reduced survivors joined only by the emitted mediated edge."""
    edge = result.chip.coupling(result.effective_params["exchange"]["coupling"])
    devices = [device.copy() for device in result.chip.devices]
    return Chip(devices, [edge.copy({device.label: device for device in devices})])


@pytest.mark.parametrize("partner", ["charge_basis", "fluxonium"])
def test_mediated_edge_alone_carries_the_reduced_exchange(partner):
    """The emitted edge reproduces the reduced chip's exchange element when a survivor's charge element is not 1."""
    result = eliminate(_bus_chip(partner), "bus")
    reduced = _element(result.chip, "a", "b")
    assert abs(reduced) > 1e-3
    # Both sides hold the same GHz-scale matrices up to double-precision round-off.
    np.testing.assert_allclose(_element(_edge_only(result), "a", "b"), reduced, rtol=0, atol=1e-13)


def test_g_over_delta_uses_the_resolved_exchange_element():
    """g_over_delta is the resolved element over the bare detuning, for a device or an edge target."""
    source = _bus_chip("charge_basis")
    validity = eliminate(source, "bus").validity
    leg = _element(source, "b", "bus")
    detuning = (_element(source, "b", "b") - _element(source, "bus", "bus")).real
    assert abs(leg) != pytest.approx(.06, rel=.1)
    assert float(validity["b_bus"]["g_over_delta"]) == pytest.approx(abs(leg / detuning), rel=1e-12)
    # A unit Duffing leg keeps |g/Δ|.
    assert float(validity["a_bus"]["g_over_delta"]) == pytest.approx(.06 / 1.5, rel=1e-12)

    a, b = DuffingTransmon(freq=6., anharmonicity=-.25, levels=3, label="a"), _PARTNERS["charge_basis"]()
    pair = Chip([a, b], [Capacitive(a, b, g=.03, label="ab")])
    detuning = (_element(pair, "a", "a") - _element(pair, "b", "b")).real
    assert float(eliminate(pair, "ab").validity["ab"]["g_over_delta"]) == pytest.approx(
        abs(_element(pair, "a", "b") / detuning), rel=1e-12)


def test_flux_gain_is_the_coupler_frequency_derivative_of_the_edge_strength():
    """dJ_domega_c equals the derivative of j_eff with respect to the coupler's first transition frequency.

    The fluxonium's charge element is imaginary, and the charge-basis coupler's is not 1.
    """
    partner, bus = "fluxonium", "charge_basis"
    base = _bus_chip(partner, bus)
    vector = np.asarray(base.resolve(frame="lab").bases["bus"].energy_vectors)[:, 1]

    def exchange(shift):
        # Raise only the coupler's first excited level: its charge elements stay fixed.
        terms = EffectiveTerms(("bus",), (vector.size,), shift * np.outer(vector, vector.conj()), label="shift")
        return eliminate(_bus_chip(partner, bus, (terms,)), "bus").effective_params["exchange"]

    step = 1e-4
    derivative = (float(exchange(step)["j_eff"]) - float(exchange(-step)["j_eff"])) / (2 * step)
    # Second-order SW exchange through the single coupler excitation: the central
    # difference has truncation error (step/Δ)² ≈ 2e-9 and round-off near 1e-8.
    assert derivative == pytest.approx(float(exchange(0.)["dJ_domega_c"]), rel=1e-6)


@pytest.mark.validation
@pytest.mark.optional_backend
def test_edge_strength_gradient_with_an_imaginary_charge_element():
    """d j_eff / d g_b is g_a/2 (1/Δ_a + 1/Δ_b) when the partner's 0-1 charge element is imaginary."""
    pytest.importorskip("dynamiqs")
    import jax

    def chip(g):
        a = DuffingTransmon(freq=5., anharmonicity=-.25, levels=3, label="a")
        b = _PARTNERS["fluxonium"]()
        bus = Resonator(freq=6.5, levels=3, label="bus")
        couplings = [Capacitive(a, bus, g=.06, label="a_bus"), Capacitive(b, bus, g=g, label="b_bus")]
        return Chip([a, b, bus], couplings, backend="dynamiqs")

    def j_eff(g):
        return eliminate(chip(g), "bus").effective_params["exchange"]["j_eff"]

    energies = np.asarray(chip(.06).resolve(frame="lab").bases["b"].energies)
    expected = .06 / 2 * (1 / (5. - 6.5) + 1 / (energies[1] - energies[0] - 6.5))
    # The second-order exchange is linear in g_b, so the derivative is exact.
    assert float(jax.jit(jax.grad(j_eff))(.06)) == pytest.approx(expected, rel=1e-10)
