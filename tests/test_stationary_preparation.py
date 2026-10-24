"""Stationary queries share native preparation within one captured calculation."""


import numpy as np
import pytest

from quchip import Capacitive, Chip, Port, PortNetwork, Resonator, VNA


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_stationary_preparation_rejects_different_operating_points(backend):
    chip = Chip([Resonator(freq=6.0, levels=3, label="r", T1=20.0)], backend=backend, frame="rotating")
    engine = chip.resolve()
    prepared = chip.backend.prepare_stationary(engine)
    changed = chip.with_params({"r.T1": 30.0}).resolve()
    with pytest.raises(ValueError, match="different backend or operating point"):
        chip.backend.prepare_stationary(changed, prepared=prepared)
    other = type(chip.backend)()
    with pytest.raises(ValueError, match="different backend or operating point"):
        other.prepare_stationary(engine, prepared=prepared)


@pytest.mark.validation
@pytest.mark.parametrize("route", ["steady", "linear", "full"])
def test_conditioning_is_requested_and_uses_captured_inputs(route, monkeypatch):
    backend = "qutip"
    r = Resonator(freq=6.0, levels=3, label="r", T1=20.0)
    second = Resonator(freq=6.03, levels=2, label="b", T1=30.0)
    port = Port(r, rate=0.03, label="p")
    chip = Chip([r, second], [Capacitive(r, second, g=0.005)],
                port_network=PortNetwork.from_ports([port]), backend=backend, frame={"r": 6.0, "b": 6.0})
    xp = chip.backend.array_module
    cond = xp.linalg.cond
    evaluated = []

    def measured(matrix, *args, **kwargs):
        evaluated.append(matrix.shape)
        return cond(matrix, *args, **kwargs)

    frequencies = np.asarray([6.01, 6.02])

    def solve():
        if route == "steady":
            return chip.steadystate()
        return VNA(chip, ports=[port]).sweep(frequencies, options={} if route == "full" else None)

    def read(result):
        return result.condition_number if route == "steady" else result.diagnostics[0]["condition_number"]

    expected = read(solve())
    monkeypatch.setattr(xp.linalg, "cond", measured)
    result = solve()
    if route != "steady":
        assert result.diagnostics[0]["solver"] == ("stationary_resolvent" if route == "full" else "linear_response")
        assert "condition_number" in result.diagnostics[0]
        assert "condition_number" in result.diagnostics[0].keys()
    assert evaluated == []
    r.T1 = 40.0
    frequencies[:] = 9.0
    value = read(result)
    assert value == pytest.approx(expected)
    assert len(evaluated) == 1
    assert read(result) == pytest.approx(value)
    if route == "linear":
        assert np.isfinite(result.diagnostics[1]["condition_number"])
    assert len(evaluated) == 1


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_vna_positivity_is_only_computed_when_requested(backend, monkeypatch):
    r = Resonator(freq=6.0, levels=3, label="r", T1=20.0)
    port = Port(r, rate=0.03, label="p")
    chip = Chip([r], port_network=PortNetwork.from_ports([port]), backend=backend)
    eigvalsh = chip.backend.array_module.linalg.eigvalsh
    evaluated = []

    def measured(state):
        evaluated.append(state.shape)
        return eigvalsh(state)

    monkeypatch.setattr(chip.backend.array_module.linalg, "eigvalsh", measured)
    result = VNA(chip, ports=[port]).sweep([6.01], options={})
    assert np.all(np.isfinite(result.s11))
    assert result.diagnostics[0]["residual"] < 1e-12
    assert "positivity_error" in result.diagnostics[0]
    assert "positivity_error" in result.diagnostics[0].keys()
    assert evaluated == []
    assert result.diagnostics[0]["positivity_error"] < 1e-12
    assert evaluated == [(3, 3)]


@pytest.mark.validation
@pytest.mark.parametrize("route", ["steady", "linear", "full"])
def test_requested_conditioning_survives_jit_then_eager_access(route):
    import jax

    r = Resonator(freq=6.0, levels=3, label="r", T1=20.0)
    port = Port(r, rate=0.03, label="p")
    chip = Chip([r], port_network=PortNetwork.from_ports([port]), backend="dynamiqs", frame="rotating")
    result = chip.steadystate() if route == "steady" else VNA(chip, ports=[port]).sweep(
        [6.01], options={} if route == "full" else None,
    )

    def read():
        return result.condition_number if route == "steady" else result.diagnostics[0]["condition_number"]

    compiled = jax.jit(read)()
    assert read() == pytest.approx(float(compiled), rel=1e-10)
