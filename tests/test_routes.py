"""Each stationary calculation records its numerical route and the reason for it."""

import warnings

import numpy as np
import pytest

from quchip import RWA, VNA, Capacitive, Chip, DuffingTransmon, Exact, ExcitationCutoff, PortNetwork, Resonator
from quchip.engine.routes import CUTOFF_TOLERANCE, automatic_cutoff


def readout_pairs(count, backend="qutip", approximation=None, frame=5.2):
    """Return ``count`` transmon-resonator pairs on one cascaded feedline, with ``9**count`` product states."""
    qubits = [DuffingTransmon(freq=5.3 - 0.1 * i, anharmonicity=-0.26, levels=3, label=f"q{i}", T1=3e4)
              for i in range(count)]
    resonators = [Resonator(freq=6.5 + 0.1 * i, levels=3, label=f"r{i}", internal_quality_factor=2e5)
                  for i in range(count)]
    network = PortNetwork(label="feedline")
    ports = [network.port(r.label, target=r, rate=2 * np.pi * 0.002) for r in resonators]
    if count > 1:
        network.cascade(*ports)
    network.expose("feed", input=ports[0], output=ports[-1])
    couplings = [Capacitive(q, r, g=0.04) for q, r in zip(qubits, resonators, strict=True)]
    return Chip(qubits + resonators, couplings, port_network=network, frame=frame,
                approximation=RWA() if approximation is None else approximation, backend=backend)


def test_each_stationary_call_reports_its_route():
    """On 81 product states, each call names its route and its reason."""
    chip = readout_pairs(2)
    frequency = float(chip.freq("r0"))
    pumped = VNA(chip)
    pumped.pump("feed", freq=frequency, amplitude=1e-3)
    with warnings.catch_warnings():
        # The automatic cutoffs warn with their choice.
        warnings.simplefilter("ignore", UserWarning)
        sweep = VNA(chip).sweep(frequency).diagnostics[0]
        sectors = [VNA(chip).finite_power(frequency, 1e-3).diagnostics[0], chip.steadystate().diagnostics,
                   pumped.output_spectrum("feed", frequencies=[-0.01, 0.01]).diagnostics,
                   pumped.g1("feed", [0.0, 10.0]).diagnostics, pumped.g2("feed", [0.0, 10.0]).diagnostics]
    assert sweep["solver"] == sweep["route"].name == "vacuum_response"
    assert sweep["route"].dimension == 4 and "one-excitation states" in sweep["route"].reason
    for diagnostics in sectors:
        route = diagnostics["route"]
        assert diagnostics["solver"] == route.name == "sector_stationary"
        assert route.dimension == diagnostics["retained_dimension"] and route.arrays == "numpy"
        assert "81 product states, more than 64" in route.reason
    # The automatic cutoffs say how they chose K; the vacuum needs no cutoff.
    assert all("This is a convergence check" in diagnostics["route"].reason
               for diagnostics in (sectors[0], *sectors[2:]))


@pytest.mark.parametrize("backend", [
    "qutip", pytest.param("dynamiqs", marks=[pytest.mark.optional_backend, pytest.mark.validation]),
])
def test_a_small_chip_solves_the_product_space_and_says_why(backend):
    """At most 64 product states use the backend's Liouvillian in its own arrays, with a memory estimate."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    route = readout_pairs(1, backend).steadystate().diagnostics["route"]
    assert route.name == "stationary_resolvent" and route.dimension == 9
    assert "9 product states, at most 64" in route.reason
    assert route.arrays == ("numpy" if backend == "qutip" else "jax") and route.peak_bytes > 0


def test_the_vacuum_needs_no_static_frame():
    """Each device rotates at its own frequency, which makes terms dynamic; the vacuum is the same in every frame."""
    chip = readout_pairs(2, frame="rotating")
    result = chip.steadystate()
    assert result.diagnostics["route"].name == "sector_stationary"
    assert result.frame.frequencies["q0"] != result.frame.frequencies["r0"]
    np.testing.assert_allclose(chip.backend.to_array(result.reduced_state("r0")), np.diag([1.0, 0.0, 0.0]), atol=1e-15)
    with pytest.raises(ValueError, match="statically compose"):
        chip.steadystate(options={})


def test_product_routes_name_what_rules_out_the_sectors():
    """Solver options and a model that does not conserve N each select the product route."""
    options = readout_pairs(2).steadystate(options={}).diagnostics["route"]
    assert options.name == "stationary_resolvent" and options.reason.startswith("Solver options")
    # Without the RWA the static model is static only in the lab frame.
    exact = readout_pairs(1, approximation=Exact(), frame="lab").with_space(ExcitationCutoff(max_total=2))
    broken = exact.steadystate().diagnostics["route"]
    assert broken.name == "stationary_resolvent" and broken.dimension == 6
    assert "does not conserve the total excitation N" in broken.reason and "N <= 2" in broken.reason


@pytest.mark.unit
def test_the_automatic_cutoff_keeps_the_larger_cutoff_of_the_first_converged_pair():
    """K rises while K + 1 changes the results, K + 1 is kept, and the check records why it stopped."""
    values = {2: 1.0, 3: 1.1, 4: 1.1 + 1e-8, 5: 1.1 + 1e-8}
    calls = []

    def solve(cutoff, lower):
        calls.append((cutoff, lower))
        return values[cutoff]

    def change(lower, upper):
        return abs(lower - upper) / max(abs(lower), abs(upper))

    cutoff, result, check = automatic_cutoff(9, solve, change, fits=lambda total: True, states=lambda total: 10 * total)
    assert (cutoff, result) == (4, 1.1 + 1e-8) and calls == [(2, None), (3, 1.0), (4, 1.1)]
    assert check.status == "passed" and check.compared_with == 3 and check.change <= CUTOFF_TOLERANCE
    assert check.states == 40
    # Results solved in part keep the smaller cutoff of the pair, and a later start skips the cutoffs below it.
    cutoff, result, check = automatic_cutoff(9, solve, change, fits=lambda total: True, states=lambda total: total,
                                             partial=lambda result: True, start=3)
    assert (cutoff, result, check.compared_with) == (3, 1.1, 4) and calls[3:] == [(3, None), (4, 1.1)]
    # A comparison that does not fit in memory does not run, and the record says so.
    cutoff, _, check = automatic_cutoff(9, solve, change, fits=lambda total: total < 4, states=lambda total: total)
    assert cutoff == 3 and check.status == "not run" and check.change is None
    # A cutoff that holds every label needs no comparison.
    cutoff, _, check = automatic_cutoff(3, solve, change, fits=lambda total: True, states=lambda total: total)
    assert cutoff == 3 and check is None
