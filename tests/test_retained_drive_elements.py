"""Dressed drive queries follow the same retained coordinates as scheduled pulses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import (
    Capacitive, ChargeDrive, Chip, DuffingTransmon, Exact, QuantumSequence, Resonator, RWA, Square, eliminate,
)


def _pair(*, backend="qutip"):
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.1, label="qr")], backend=backend)
    chip.wire(ChargeDrive(q, label="d"))
    return chip


def _driven(chip):
    sequence = QuantumSequence(chip)
    sequence.schedule("d", envelope=Square(duration=1.0, amplitude=1.0), freq=None)
    return sequence


def _pulse_operator(chip):
    """Read the unit-amplitude baseband drive in ordinary GHz from the public resolved Hamiltonian."""
    driven = _driven(chip).resolve(frame="lab").hamiltonian().matrix(t=0.5)
    idle = chip.resolve(frame="lab").hamiltonian().matrix(t=0.5)
    return np.asarray(driven) - np.asarray(idle)


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
@pytest.mark.parametrize("reduction", ["sw", "exact", "active_patch"])
def test_reduced_drive_query_matches_the_pulse_operator_and_mixing_angle(backend, reduction):
    """Reduced drive queries and pulses share the SW or exact two-mode mixing amplitude."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    source = _pair(backend=backend)
    if reduction == "active_patch":
        chip = _driven(source).active_patch(hops=0, local=True).chip
    else:
        chip = eliminate(source, "r", method=reduction).chip

    element = complex(chip.drive_matrix_elements("q")["d"])
    assert element == pytest.approx(_pulse_operator(chip)[1, 0], abs=1e-12)
    # The SW map rotates by g/Delta; exact reduction uses the two-mode mixing angle.
    angle = 0.5 * np.arctan(2 * 0.1 / 2.0) if reduction == "exact" else 0.1 / 2.0
    assert element == pytest.approx(-1j * np.cos(angle), abs=1e-12)


@pytest.mark.parametrize("approximation", [RWA(), Exact()])
@pytest.mark.parametrize("basis", ["native", "eigen"])
def test_retained_drive_can_act_on_separated_survivors_with_a_conditioned_spectator(approximation, basis):
    """Retained drive support preserves spectator conditioning in sector and full-space evaluations."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    spectator = Resonator(freq=5.4, levels=2, label="s")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    r = Resonator(freq=5.8, levels=3, label="r")
    source = Chip([q, spectator, bus, r],
                  [Capacitive(q, bus, g=0.1), Capacitive(r, bus, g=0.07)],
                  approximation=approximation, basis=basis, frame="lab")
    source.wire(ChargeDrive(q, label="d"))
    reduced = eliminate(source, "bus", local=True)
    chip = reduced.chip
    initial_label, final_label = {"s": 1}, {"q": 1, "s": 1}

    def phase_fixed_state(label):
        state = np.asarray(chip.backend.to_array(chip.state(label))).ravel()
        index = np.ravel_multi_index(tuple(label.get(device.label, 0) for device in chip.devices), chip.dims)
        return state * state[index].conj() / abs(state[index])

    initial, final = phase_fixed_state(initial_label), phase_fixed_state(final_label)
    query = chip.drive_matrix_elements((initial_label, final_label))["d"]
    assert query == pytest.approx(np.vdot(final, _pulse_operator(chip) @ initial), abs=1e-12)
    # Independent coordinate-map reference, including the untouched spectator between q and r.
    a = np.diag(np.sqrt([1.0, 2.0]), 1)
    authored = np.kron(1j * (a - a.T), np.eye(18))
    embedding = np.asarray(reduced.mapping.embedding)
    retained = embedding.conj().T @ authored @ embedding
    assert query == pytest.approx(np.vdot(final, retained @ initial), abs=1e-12)


def test_local_retained_drive_query_avoids_the_large_product_space():
    """A local reduction's drive amplitude remains accessible without the full 24-device product space."""
    devices = [DuffingTransmon(freq=5.0 + 0.03 * i, anharmonicity=-0.25, levels=3, label=f"q{i}")
               for i in range(24)]
    bus = Resonator(freq=7.0, levels=3, label="bus")
    source = Chip(devices + [bus], [Capacitive(devices[0], bus, g=0.1)])
    source.wire(ChargeDrive(devices[0], label="d"))
    chip = eliminate(source, "bus", local=True).chip
    assert chip.total_dim == 3**24
    assert chip.drive_matrix_elements("q0")["d"] == pytest.approx(-1j * np.cos(0.1 / 2.0), abs=1e-12)


@pytest.mark.optional_backend
@pytest.mark.validation
@pytest.mark.parametrize("local", [False, True])
def test_retained_drive_value_and_gradient_follow_the_sw_rotation(local):
    """JIT drive values and coupling derivatives agree with the analytic SW rotation."""
    pytest.importorskip("dynamiqs")
    source = _pair(backend="dynamiqs")

    def element(g):
        reduced = eliminate(source.with_params({"qr.g": g}), "r", local=local).chip
        return jnp.abs(reduced.drive_matrix_elements("q")["d"])

    evaluate = jax.jit(jax.value_and_grad(element))
    for g in (0.0, 0.07, 0.1):
        value, gradient = evaluate(g)
        assert value == pytest.approx(np.cos(g / 2.0), abs=1e-12)
        assert gradient == pytest.approx(-np.sin(g / 2.0) / 2.0, abs=1e-11)
