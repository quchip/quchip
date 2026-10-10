"""Returned reductions retain matrices and channels beyond first-transition summaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import Capacitive, Chip, DuffingTransmon, Resonator, RWA, eliminate
from quchip.chip.sw import (
    bare_hamiltonian,
)


def _leaf(backend, g=0.05, thermal=False):
    q = DuffingTransmon(
        freq=5.0,
        anharmonicity=-0.3,
        levels=4,
        label="q",
        T1=300.0 if thermal else None,
        thermal_occupation=0.2 if thermal else None,
    )
    r = Resonator(freq=7.0, levels=4, label="r", T1=100.0, thermal_occupation=0.7 if thermal else None)
    return Chip([q, r], [Capacitive(q, r, g=g, label="qr")], backend=backend, approximation=RWA())


@pytest.mark.validation
@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_sw_bus_keeps_collective_dark_and_bright_decay(backend):
    a = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="a")
    b = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=2, label="b")
    bus = Resonator(freq=7.0, levels=3, T1=100.0, label="bus")
    full = Chip(
        [a, b, bus], [Capacitive(a, bus, g=0.05), Capacitive(b, bus, g=0.05)], backend=backend, approximation=RWA()
    )
    reduced = eliminate(full, "bus").chip
    jumps = [
        np.asarray(reduced.backend.to_array(op))
        for op in reduced.backend._collapse_operators(reduced.resolve(frame="lab"))
    ]
    dark = np.array([0, 1, -1, 0]) / np.sqrt(2)
    bright = np.array([0, 1, 1, 0]) / np.sqrt(2)
    assert sum(np.linalg.norm(op @ dark) ** 2 for op in jumps) == pytest.approx(0.0, abs=1e-20)
    expected_bright_rate = np.sin(np.sqrt(2) * 0.05 / 2.0) ** 2 / 100.0
    assert sum(np.linalg.norm(op @ bright) ** 2 for op in jumps) == pytest.approx(expected_bright_rate)


@pytest.mark.validation
def test_retained_high_level_shift_has_the_correct_jit_gradient():
    def value(g):
        reduced = eliminate(_leaf("dynamiqs", g=g), "r").chip
        return jnp.real(reduced.resolve(frame="lab").hamiltonian().matrix(backend=reduced.backend)[3, 3])

    g = 0.05
    actual, gradient = jax.jit(jax.value_and_grad(value))(g)
    assert actual == pytest.approx(3 * 5.0 - 0.3 * 3 + 3 * g**2 / (-2.0 - 2 * 0.3))
    assert gradient == pytest.approx(6 * g / (-2.0 - 2 * 0.3), rel=1e-10)


@pytest.mark.parametrize("method", ["sw", "exact"])
@pytest.mark.parametrize("bridge", [False, True])
def test_parallel_mode_legs_match_their_combined_interaction(method, bridge):
    """Parallel legs interfere in one mode; they do not create another survivor."""
    a = DuffingTransmon(freq=5., anharmonicity=-.3, levels=3, label="a")
    r = Resonator(freq=7., levels=3, label="r", T1=100.)
    devices = [a, r]
    parallel = [Capacitive(a, r, g=.03, label="leg0"), Capacitive(a, r, g=.02, label="leg1")]
    combined = [Capacitive(a, r, g=.05, label="combined")]
    if bridge:
        b = DuffingTransmon(freq=5.3, anharmonicity=-.25, levels=3, label="b")
        devices.append(b)
        for couplings in (parallel, combined):
            couplings.append(Capacitive(b, r, g=.04, label="leg_b"))
    result = eliminate(Chip(devices, parallel), "r", method=method)
    expected = eliminate(Chip(devices, combined), "r", method=method)
    np.testing.assert_allclose(bare_hamiltonian(result.chip)[0], bare_hamiltonian(expected.chip)[0], atol=1e-11)
    assert set(result.validity) == {coupling.label for coupling in parallel}
    # Each parallel leg reports the combined element, 0.05 GHz, over the 2 GHz detuning.
    assert [float(result.validity[label]["g_over_delta"]) for label in ("leg0", "leg1")] == pytest.approx(
        [.025, .025], rel=1e-12)
    if bridge:
        assert float(result.effective_params["exchange"]["dJ_domega_c"]) == pytest.approx(
            float(expected.effective_params["exchange"]["dJ_domega_c"]), rel=1e-12)
    else:
        assert "exchange" not in result.effective_params
        assert float(result.effective_params["a"]["chi"]) == pytest.approx(
            float(expected.effective_params["a"]["chi"]), abs=1e-12)


@pytest.mark.validation
@pytest.mark.parametrize("method", ["sw", "exact"])
@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
def test_removed_coupling_preserves_its_projected_collective_channel(method, backend):
    from quchip.extensions import CollectiveDecayCoupling

    q = DuffingTransmon(freq=5., anharmonicity=-.3, levels=3, label="q")
    r = Resonator(freq=7., levels=3, label="r")
    edge = CollectiveDecayCoupling(q, r, exchange_strength=.05, decay_rate=.01, label="qr")
    full = Chip([q, r], [edge], backend=backend)
    (source_jump,) = full.backend._collapse_operators(full.resolve(frame="lab"))
    source_jump = np.asarray(full.backend.to_array(source_jump))
    result = eliminate(full, "r", method=method)
    expected = full.backend.to_array(result.mapping.project_operator(source_jump))
    reduced = result.chip
    (actual,) = reduced.backend._collapse_operators(reduced.resolve(frame="lab"))
    actual = np.asarray(reduced.backend.to_array(actual))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_traced_reduced_chip_keeps_the_concrete_drive_bands():
    """Tracing a survivor coupling of an exactly reduced chip reproduces the concrete drive terms."""
    from quchip import ChargeDrive, ControlEquipment, Gaussian, QuantumSequence

    a = DuffingTransmon(freq=5.0, anharmonicity=-0.3, levels=3, label="a")
    b = DuffingTransmon(freq=5.2, anharmonicity=-0.3, levels=3, label="b")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    couplings = [Capacitive(a, bus, g=0.05), Capacitive(b, bus, g=0.05), Capacitive(a, b, g=0.002, label="ab")]
    full = Chip([a, b, bus], couplings, control_equipment=ControlEquipment([ChargeDrive(a, label="da")]),
                frame=5.1, approximation=RWA(), backend="dynamiqs")
    reduced = eliminate(full, "bus", method="exact").chip

    def drive_operators(g):
        sequence = QuantumSequence(reduced.with_params({"ab.g": g}))
        sequence.schedule("da", envelope=Gaussian(duration=20.0, amplitude=0.02), freq=5.0)
        result = sequence.build_problem(tlist=np.array([0.0, 20.0]), dissipation=False).engine_result
        return jnp.stack([jnp.asarray(term.operator.to_dense()) for term in result.dynamic_terms])

    concrete = np.asarray(drive_operators(0.002))
    traced = np.asarray(jax.jit(drive_operators)(0.002))
    assert traced.shape == concrete.shape
    np.testing.assert_allclose(traced, concrete, atol=1e-12)
