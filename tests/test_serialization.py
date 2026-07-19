"""Serialization tests for devices, controls, and chip topology."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    Crosstalk,
    Delay,
    DuffingTransmon,
    FluxDrive,
    Gain,
    Gaussian,
    QuantumSequence,
    Resonator,
    SimulationResult,
    Square,
)
from quchip.chip.coupling_base import BaseCoupling
from quchip.control.envelopes import BaseEnvelope
from quchip.devices.base import BaseDevice
from quchip.utils.labeling import reset_label_counters


def _assert_json_primitives(value: Any) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _assert_json_primitives(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_json_primitives(item)
        return
    raise AssertionError(f"Non-JSON-safe value encountered: {value!r}")


@pytest.fixture(autouse=True)
def _reset_labels() -> None:
    reset_label_counters()
    yield
    reset_label_counters()


def _build_chip(frame: str | float | dict[str, float] = "rotating") -> Chip:
    q = DuffingTransmon(
        freq=5.0,
        anharmonicity=-0.25,
        levels=3,
        label="q",
        T1=20.0,
        T2=30.0,
    )
    r = Resonator(freq=7.0, levels=5, label="r", quality_factor=1e4)
    coupling = Capacitive(q, r, g=0.02, rwa=False)

    readout = ChargeDrive(
        target=r,
        label="readout",
    )
    flux = FluxDrive(target=q, label="flux")
    equipment = ControlEquipment(
        lines=[readout, flux],
        signal_chain=[
            Crosstalk(
                source=readout.label,
                victim=flux.label,
                beta=0.15, theta=0.3, delay=2.0,
            ),
        ],
    )

    chip = Chip(
        devices=[q, r],
        couplings=[coupling],
        label="demo",
        frame=frame,
    )
    chip.connect(equipment)
    return chip


def test_device_registry_is_populated() -> None:
    """Concrete device subclasses auto-register under their fully-qualified class name."""
    assert "quchip.devices.transmon.duffing.DuffingTransmon" in BaseDevice._registry
    assert "quchip.devices.resonator.Resonator" in BaseDevice._registry


def test_coupling_registry_is_populated() -> None:
    """Concrete coupling subclasses auto-register under their fully-qualified class name."""
    assert "quchip.chip.couplings.Capacitive" in BaseCoupling._registry


def test_device_round_trip_is_json_safe() -> None:
    """A device's to_dict/from_dict round trip is JSON-safe and preserves its parameters."""
    device = DuffingTransmon(
        freq=5.1,
        anharmonicity=-0.3,
        levels=4,
        label="q0",
        T1=12.0,
        T2=18.0,
        thermal_population=0.02,
    )

    payload = device.to_dict()
    restored = DuffingTransmon.from_dict(payload)

    json.dumps(payload)
    _assert_json_primitives(payload)
    assert restored.freq == pytest.approx(device.freq)
    assert restored.anharmonicity == pytest.approx(device.anharmonicity)
    assert restored.levels == device.levels
    assert restored.label == device.label
    assert restored.thermal_population == pytest.approx(device.thermal_population)


def test_resonator_round_trip_preserves_quality_factor() -> None:
    """A Resonator's to_dict/from_dict round trip preserves its quality factor."""
    resonator = Resonator(freq=7.2, levels=6, label="r0", quality_factor=2e4)
    payload = resonator.to_dict()
    restored = Resonator.from_dict(payload)

    json.dumps(payload)
    _assert_json_primitives(payload)
    assert restored.freq == pytest.approx(resonator.freq)
    assert restored.quality_factor == pytest.approx(resonator.quality_factor)
    assert restored.levels == resonator.levels


def test_drive_and_equipment_round_trip_are_json_safe() -> None:
    """ControlEquipment's to_dict/from_dict round trip preserves drives and the signal chain."""
    chip = _build_chip()
    equipment = chip.control_equipment
    assert equipment is not None

    payload = equipment.to_dict()
    fresh_devices = [BaseDevice.from_dict(dev.to_dict()) for dev in chip.devices]
    restored = ControlEquipment.from_dict(
        payload,
        {dev.label: dev for dev in fresh_devices},
    )

    json.dumps(payload)
    _assert_json_primitives(payload)
    assert restored.lines[0].device_label == "r"
    assert restored.signal_chain
    assert isinstance(restored.signal_chain[0], Crosstalk)
    assert restored.signal_chain[0].beta == pytest.approx(0.15)
    assert restored.signal_chain[0].theta == pytest.approx(0.3)
    assert restored.signal_chain[0].delay == pytest.approx(2.0)


def test_signal_chain_round_trip_is_json_safe() -> None:
    """A signal chain's to_dict/from_dict round trip preserves Delay and Gain parameters."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    drive = ChargeDrive(
        target=q,
        label="q_drive",
    )
    equipment = ControlEquipment(
        lines=[drive],
        signal_chain=[
            Delay(line="q_drive", delta_t=1.5),
            Gain(line="q_drive", factor=0.2 + 0.05j),
        ],
    )

    payload = equipment.to_dict()
    fresh_q = BaseDevice.from_dict(q.to_dict())
    restored = ControlEquipment.from_dict(payload, {"q": fresh_q})

    json.dumps(payload)
    _assert_json_primitives(payload)
    assert len(restored.signal_chain) == 2
    assert isinstance(restored.signal_chain[0], Delay)
    assert isinstance(restored.signal_chain[1], Gain)
    assert restored.signal_chain[0].delta_t == pytest.approx(1.5)
    assert restored.signal_chain[1].factor == pytest.approx(0.2 + 0.05j)


def test_envelope_round_trip_is_json_safe() -> None:
    """A Gaussian envelope's to_dict/from_dict round trip preserves its parameters."""
    envelope = Gaussian(duration=20.0, sigmas=4.0, amplitude=0.25)
    payload = envelope.to_dict()
    restored = BaseEnvelope.from_dict(payload)

    json.dumps(payload)
    _assert_json_primitives(payload)
    assert isinstance(restored, Gaussian)
    assert restored.duration == pytest.approx(20.0)
    assert restored.sigmas == pytest.approx(4.0)
    assert restored.amplitude == pytest.approx(0.25)


@pytest.mark.parametrize(
    "frame",
    ["lab", "rotating", 4.75, {"q": 4.9, "r": 6.8}],
)
def test_chip_round_trip_preserves_topology_and_frame(
    frame: str | float | dict[str, float],
) -> None:
    """A chip's to_dict/from_dict round trip preserves topology and every frame form."""
    chip = _build_chip(frame=frame)
    payload = chip.to_dict()
    restored = Chip.from_dict(payload)

    json.dumps(payload)
    _assert_json_primitives(payload)
    assert set(payload) >= {"label", "frame", "devices", "couplings"}
    assert "backend" not in payload
    assert "result" not in payload
    assert restored.label == chip.label
    assert len(restored.devices) == len(chip.devices)
    assert len(restored.couplings) == len(chip.couplings)
    assert restored.frame == chip.frame
    assert restored.devices[0].label == "q"
    assert restored.devices[1].label == "r"
    assert restored.couplings[0].g == pytest.approx(chip.couplings[0].g)


def test_chip_round_trip_preserves_control_equipment() -> None:
    """A chip's to_dict/from_dict round trip preserves control equipment and crosstalk."""
    chip = _build_chip()
    restored = Chip.from_dict(chip.to_dict())

    assert restored.control_equipment is not None
    assert len(restored.control_equipment.lines) == 2
    assert len(restored["r"]["readout"].local_channels(restored["r"])) == 1
    assert restored.control_equipment.signal_chain
    xt = restored.control_equipment.signal_chain[0]
    assert isinstance(xt, Crosstalk)
    assert xt.source == "readout"
    assert xt.victim == "flux"


def test_chip_round_trip_can_dress_and_simulate() -> None:
    """A deserialized chip can still be dressed and simulated end-to-end."""
    chip = _build_chip()
    restored = Chip.from_dict(chip.to_dict())

    seq = QuantumSequence(restored)
    seq.schedule(
        restored["r"]["readout"],
        envelope=Square(duration=10.0, amplitude=0.005),
        freq=restored["r"].drive_freq + 1.0,
    )
    result = seq.simulate(tlist=np.linspace(0.0, 10.0, 101))

    assert isinstance(result, SimulationResult)
    assert restored.is_dressed
    assert not hasattr(restored, "result")
    assert not hasattr(restored, "sequence")
    assert len(result.times) == 101
