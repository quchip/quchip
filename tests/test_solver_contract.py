"""Explicit numerical choices preserve requested physics and reject silent omissions."""

import numpy as np
import pytest

from quchip import Chip, DuffingTransmon, QuantumSequence
from quchip.engine import solve_problem


def _sequence(backend="qutip", *, noisy=False):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q", T1=20.0 if noisy else None)
    return QuantumSequence(Chip([q], frame="rotating", backend=backend))


def test_built_request_rejects_unknown_solver():
    """Direct request construction cannot route an unknown name to a different solver."""
    with pytest.raises(ValueError, match="Unknown solver"):
        _sequence().build_problem([0.0, 1.0], solver="typo")


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_explicit_schrodinger_solver_cannot_discard_dissipation(backend):
    """Selecting ket evolution explicitly must not silently remove declared decay."""
    problem = _sequence(backend, noisy=True).build_problem([0.0, 1.0], solver="sesolve")
    with pytest.raises(ValueError, match="dissipation"):
        solve_problem(problem)


@pytest.mark.parametrize("options", [{"atol": 1e-10}, {"max_step": 0.01}, {"typo": 1}])
def test_dynamiqs_rejects_unsupported_options(options):
    """Unsupported flat options fail rather than appearing to configure the native solver."""
    problem = _sequence("dynamiqs").build_problem([0.0, 1.0], options=options)
    with pytest.raises(ValueError, match="[Uu]nsupported.*option"):
        solve_problem(problem)


def test_qutip_explicit_diag_rejects_adaptive_controls():
    """Explicit diagonal propagation cannot claim to apply adaptive tolerances."""
    problem = _sequence().build_problem([0.0, 1.0], options={"method": "diag", "rtol": 1e-8})
    with pytest.raises(ValueError, match="diag.*option"):
        solve_problem(problem)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_dissipation_exclusion_is_explicit_and_captured(backend):
    """A Hamiltonian-only calculation retains excitation and records its choice."""
    seq = _sequence(backend, noisy=True)
    times = np.linspace(0.0, 2.0, 5)
    problem = seq.build_problem(times, dissipation=False, solver="sesolve", initial_state={"q": 1})
    result = solve_problem(problem)
    assert problem.dissipation is False
    assert result.dissipation is False
    assert not problem.engine_result.collapse_terms
    np.testing.assert_allclose(result.population("q", 1), 1.0, atol=1e-8)
    with pytest.raises((KeyError, ValueError)):
        result.jump_rate("hidden.q.thermal_emission")
    assert seq.build_problem(times).engine_result.collapse_terms


def test_dissipation_exclusion_retains_network_hamiltonian():
    """Excluding jumps leaves series-composition Hamiltonian terms exactly intact."""
    from quchip import PortNetwork, Resonator

    first, second = Resonator(freq=5.0, levels=2, label="a"), Resonator(freq=5.0, levels=2, label="b")
    network = PortNetwork(label="line")
    a = network.port("a_port", target=first, rate=0.04)
    b = network.port("b_port", target=second, rate=0.09)
    network.cascade(a, b)
    network.expose("feedline", input=a.input, output=b.output)
    seq = QuantumSequence(Chip([first, second], port_network=network, frame="lab"))
    normal = seq.build_problem([0.0, 1.0])
    closed = seq.build_problem([0.0, 1.0], dissipation=False)
    assert any(term.origin == "network" for term in closed.engine_result.static_terms)
    np.testing.assert_allclose(
        closed.engine_result.hamiltonian().matrix(), normal.engine_result.hamiltonian().matrix(), atol=1e-12,
    )
    assert normal.engine_result.collapse_terms
    assert closed.engine_result.collapse_terms == ()
    assert closed.engine_result.port_terms == ()
    assert normal.engine_result.port_terms


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_batch_retains_dissipation_choice_and_native_method(backend):
    """Batch construction and dispatch preserve exclusion and an explicit native method."""
    seq = _sequence(backend, noisy=True)
    if backend == "dynamiqs":
        dq = pytest.importorskip("dynamiqs")
        options = {"method": dq.method.Tsit5(atol=1e-10, rtol=1e-10)}
    else:
        options = {"method": "adams", "atol": 1e-10, "rtol": 1e-10}
    batch = seq.build_batch(
        seq.vary("q.freq", [4.9, 5.1]),
        tlist=[0.0, 1.0], initial_state={"q": 1}, dissipation=False, options=options,
    )
    for result in batch.chip.solve_many(batch, progress=False):
        assert result.dissipation is False
        np.testing.assert_allclose(result.population("q", 1), 1.0, atol=1e-8)


def test_partitioned_simulation_propagates_dissipation_choice():
    """Independent component solves cannot accidentally re-enable requested exclusions."""
    devices = [DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label=label, T1=20.0) for label in ("a", "b")]
    seq = QuantumSequence(Chip(devices, frame="rotating"))
    result = seq.simulate(tlist=[0.0, 1.0], initial_state={"a": 1, "b": 1}, dissipation=False)
    assert result.dissipation is False
    for component in result.components:
        assert component.dissipation is False
    for device in devices:
        np.testing.assert_allclose(result.population(device, 1), 1.0, atol=1e-8)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_results_report_effective_numerical_settings(backend):
    """Report native defaults and user controls from the solver that actually ran."""
    if backend == "dynamiqs":
        dq = pytest.importorskip("dynamiqs")
        options = {"method": dq.method.Tsit5(atol=1e-10, rtol=1e-9)}
    else:
        options = {"atol": 1e-10, "rtol": 1e-9}
    result = _sequence(backend).simulate(tlist=[0.0, 1.0], options=options)
    effective = result.stats["options"]
    if backend == "dynamiqs":
        assert effective["method"].atol == 1e-10
        assert effective["method"].rtol == 1e-9
        assert effective["method"].max_steps > 0
    else:
        assert effective["method"] == "adams"
        assert effective["atol"] == 1e-10
        assert effective["rtol"] == 1e-9
        assert effective["nsteps"] > 0
