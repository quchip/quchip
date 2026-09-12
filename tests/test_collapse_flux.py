"""Per-channel jump rates read back from stored states."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import Chip, PortNetwork, QuantumSequence, Resonator
from quchip.control.envelopes import Square
from quchip.control.field import CoherentInput
from quchip.devices.transmon.duffing import DuffingTransmon


def _decaying_qubit():
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=20.0)
    chip = Chip([qubit], frame="rotating")
    times = np.linspace(0.0, 60.0, 121)
    result = QuantumSequence(chip).simulate(
        times, initial_state=chip.bare_state(q=1), partition=False
    )
    return chip, times, result


def test_jump_rate_closes_the_excitation_balance() -> None:
    """The T1 jump rate integrates to the excitation the qubit lost."""
    _, times, result = _decaying_qubit()
    (key,) = result.collapse_channels
    assert key == "hidden.q.thermal_emission"
    flux = np.asarray(result.jump_rate(key))
    excited = np.asarray(result.population("q", 1))
    np.testing.assert_allclose(flux, excited / 20.0, rtol=1e-6)
    lost = np.asarray(result.collapse_integral(key))
    assert lost.shape == times.shape and lost[0] == 0.0
    np.testing.assert_allclose(lost[-1] + excited[-1], 1.0, atol=1e-4)
    assert result.jump_rate("q.thermal_emission") is result.jump_rate(key)


def test_jump_rate_requires_stored_states_and_known_keys() -> None:
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=20.0)
    chip = Chip([qubit], frame="rotating")
    result = QuantumSequence(chip).simulate(
        np.linspace(0.0, 10.0, 11),
        initial_state=chip.bare_state(q=1),
        states="final",
        partition=False,
        )
    with pytest.raises(RuntimeError, match='states="all"'):
        result.jump_rate(result.collapse_channels[0])
    with pytest.raises(KeyError, match="matches no channel"):
        result.jump_rate("nothing")


def _port_chip() -> tuple[Chip, object]:
    resonator = Resonator(freq=6.0, levels=4, label="r", internal_quality_factor=2.0e4)
    network = PortNetwork(label="line")
    port = network.port("readout", target=resonator, rate=0.05)
    plane = network.expose("line", at=port)
    return Chip([resonator], port_network=network, frame=6.0), plane


def test_port_flux_matches_the_raw_output_flux_for_vacuum_input_only() -> None:
    """An exposed plane's raw photon flux is <L^dag L> with vacuum input and differs once driven."""
    chip, plane = _port_chip()
    times = np.linspace(0.0, 40.0, 41)
    e_ops = {plane: plane.output}
    quiet = QuantumSequence(chip).simulate(
        times, e_ops=e_ops, initial_state=chip.bare_state(r=1), partition=False
    )
    np.testing.assert_allclose(quiet.jump_rate("line"), quiet.output("line").raw_photon_flux, atol=1e-10)
    assert quiet.jump_rate(plane) is quiet.jump_rate("line")
    internal = np.asarray(quiet.jump_rate("hidden.r.internal_photon_loss"))
    np.testing.assert_allclose(internal, 2 * np.pi * 6.0 / 2.0e4 * np.asarray(quiet.population("r", 1)), rtol=1e-6)

    driven = QuantumSequence(chip)
    driven.schedule(CoherentInput("line"), envelope=Square(duration=40.0, amplitude=0.05), freq=6.0)
    result = driven.simulate(times, e_ops=e_ops, partition=False)
    residual = np.asarray(result.output("line").raw_photon_flux) - np.asarray(result.jump_rate("line"))
    assert np.max(np.abs(residual)) > 1e-4  # |S beta|^2 and the interference term, absent for vacuum input


def test_batch_jump_rate_stacks_per_point_traces() -> None:
    """Batch results reshape per-point jump rates like every other trace."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=20.0)
    chip = Chip([qubit], frame="rotating")
    sequence = QuantumSequence(chip)
    from quchip.control.drive import ChargeDrive
    from quchip.control.equipment import ControlEquipment

    drive = ChargeDrive(target=qubit)
    chip.connect(ControlEquipment(lines=[drive]))
    pulse = sequence.schedule(drive, envelope=Square(duration=20.0, amplitude=0.01), freq=5.0)
    axis = pulse.vary("amplitude", [0.01, 0.02], name="amp")
    times = np.linspace(0.0, 20.0, 21)
    batch = sequence.simulate_batch(axis, tlist=times, progress=False)
    key = batch[0].collapse_channels[0]
    stacked = np.asarray(batch.jump_rate(key))
    assert stacked.shape == (2, 21)
    assert stacked[1, -1] > stacked[0, -1]  # the stronger pulse leaves more excitation to decay
    assert np.asarray(batch.collapse_integral(key, reduce="last")).shape == (2,)


def test_later_eager_flux_query_does_not_reuse_a_traced_cache():
    import jax

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, T1=10.0, label="q")
    chip = Chip([q], backend="dynamiqs", frame="rotating")
    result = QuantumSequence(chip).simulate(tlist=[0.0, 1.0, 3.0], initial_state={"q": 1})
    key = result.collapse_channels[0]
    compiled = jax.jit(lambda: result.jump_rate(key))()
    np.testing.assert_allclose(result.jump_rate(key), compiled)
