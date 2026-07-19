"""The describe() surface — human-readable chip and sequence summaries.

Assertions target content (labels, units, wiring), not exact layout, so the
report can be reformatted without rewriting tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from quchip.chip.baths import Bath
from quchip.chip.chip import Chip
from quchip.chip.coupling_base import BaseCoupling
from quchip.chip.couplings import Capacitive
from quchip.control.drive import ChargeDrive
from quchip.control.envelopes import Gaussian
from quchip.control.sequence import QuantumSequence
from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.models import CouplingModel
from quchip.declarative.parameters import Scalar, parameter
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon


def _demo_chip() -> tuple[Chip, ChargeDrive, DuffingTransmon, Resonator]:
    q = DuffingTransmon(freq=5.24, anharmonicity=-0.26, levels=3,
                        T1=51_570.0, T2=23_800.0, label="q")
    r = Resonator(freq=6.65, levels=8, quality_factor=5598.0, label="r")
    chip = Chip(
        [q, r],
        couplings=[Capacitive(q, r, g=0.060)],
        baths=[Bath("thermal", [q], temperature=30.0)],
        frame="rotating",
        rwa=True,
    )
    drv = ChargeDrive(target=q)
    chip.wire(drv)
    return chip, drv, q, r


def test_chip_describe_reports_composition_with_units() -> None:
    """describe() lists every device/coupling/drive/bath under its label with declared units."""
    chip, drv, q, r = _demo_chip()
    text = chip.describe()

    assert "q — DuffingTransmon" in text
    assert "freq = 5.24 GHz" in text
    assert "anharmonicity = -0.26 GHz" in text
    assert "T1 = 51570 ns" in text
    assert "r — Resonator" in text
    assert "g = 0.06 GHz" in text
    assert "q ↔ r" in text
    assert f"{drv.label} — ChargeDrive → q" in text
    assert "thermal on q" in text
    assert "temperature = 30 mK" in text
    assert "rotating" in text
    assert "3 x 8 = 24 levels" in text


def test_sequence_describe_lists_pulses_and_other_entries() -> None:
    """describe() reports pulse count, timing windows (delay-shifted), and delay entries."""
    chip, drv, q, _ = _demo_chip()
    seq = QuantumSequence(chip)
    seq.schedule(drv, envelope=Gaussian(duration=80.0, sigmas=3, amplitude=0.015), freq=5.2)
    seq.delay(q, 10.0)
    seq.schedule(drv, envelope=Gaussian(duration=40.0, sigmas=3, amplitude=0.0075), freq=5.1)
    text = seq.describe()

    assert "2 pulses" in text
    assert "[0, 80]" in text
    assert "[90, 130]" in text  # delay shifts the second pulse
    assert f"{drv.label} → q" in text
    assert "Gaussian(" in text
    assert "5.2 GHz" in text
    assert "Delay" in text


def test_describe_never_concretizes_traced_parameters() -> None:
    """describe() renders a traced frequency as "<traced>" instead of forcing concretization."""
    def probe(freq):
        q = DuffingTransmon(freq=freq, anharmonicity=-0.26, levels=3, label="q")
        chip = Chip([q])
        text = chip.describe()
        assert "<traced>" in text
        return jnp.asarray(len(text), dtype=jnp.float64)

    jax.jit(probe)(jnp.asarray(5.0))


def test_chip_physics_notes_includes_chip_level_bath_and_edge_drive_entries() -> None:
    """physics_notes() carries a chip-level local-Lindblad entry, bath notes, and edge-target drive notes."""
    from quchip.chip.couplings import TunableCapacitive
    from quchip.control.drive import ParametricDrive
    from quchip.control.equipment import ControlEquipment

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.24, levels=3, label="q1")
    coupling = TunableCapacitive(q0, q1, g_0=0.01, label="tc")
    pump = ParametricDrive(coupling, label="pump")
    chip = Chip(
        [q0, q1],
        couplings=[coupling],
        control_equipment=ControlEquipment([pump]),
        baths=[Bath("thermal", temperature=20.0, rate=1e-3)],
    )

    notes = chip.physics_notes()

    assert "chip" in notes
    assert any("LOCAL" in note for note in notes["chip"])
    assert all(f"bath:{bath.label}" in notes for bath in chip.baths)
    assert "drive:pump" in notes


def test_chip_physics_notes_keys_are_collision_proof_across_kinds() -> None:
    """A device, drive, and bath sharing a label with each other (and 'chip') never overwrite each other's entry."""
    from quchip.control.drive import ChargeDrive
    from quchip.control.equipment import ControlEquipment

    device = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="chip")
    drive = ChargeDrive(device, label="chip")
    bath = Bath("thermal", temperature=20.0, rate=1e-3, label="chip")
    chip = Chip([device], control_equipment=ControlEquipment([drive]), baths=[bath])

    notes = chip.physics_notes()

    assert notes.keys() >= {"chip", "device:chip", "drive:chip", "bath:chip"}
    assert notes["device:chip"] != notes["drive:chip"]
    assert notes["drive:chip"] != notes["bath:chip"]
    # The synthetic chip-level entry is never shadowed by a component labeled "chip".
    assert any("LOCAL" in note for note in notes["chip"])


def test_coupling_model_default_repr_covers_extensions() -> None:
    """The default CouplingModel repr names the class, endpoints, and declared parameters."""
    class ExchangeXX(CouplingModel):
        _type_prefix = "xx"
        j: Scalar = parameter(unit="GHz")

        def interaction(self, a, b) -> PhysicsExpr:
            return self.j * (a.a @ b.adag + a.adag @ b.a)

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    text = repr(ExchangeXX(q, r, j=0.01))
    assert "ExchangeXX" in text
    assert "'q' <-> 'r'" in text
    assert "j=0.01" in text


def test_base_coupling_fallback_repr_names_endpoints() -> None:
    """A BaseCoupling subclass without a custom repr still names its class and endpoints."""
    class RawCoupling(BaseCoupling):
        _type_prefix = "raw"

        @property
        def coupling_strength(self):
            return 0.0

        def interaction_hamiltonian(self):  # pragma: no cover - not built
            raise NotImplementedError

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    text = repr(RawCoupling(q, r))
    assert "RawCoupling" in text
    assert "'q' <-> 'r'" in text
