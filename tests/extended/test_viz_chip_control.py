"""Visualization coverage for pulse-sequence, energy-level, and chip-graph plots."""

from __future__ import annotations


from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    Crosstalk,
    DuffingTransmon,
    FluxDrive,
    QuantumSequence,
    Resonator,
    Square,
    plot_energy_levels,
    plot_sequence,
)


def _build_control_chip() -> tuple[Chip, DuffingTransmon, Resonator, ChargeDrive, ChargeDrive, FluxDrive]:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=6.8, levels=4, label="r")
    q_charge = ChargeDrive(target=q, label="q_charge")
    r_charge = ChargeDrive(target=r, label="r_charge")
    q_flux = FluxDrive(target=q, label="q_flux")
    chip = Chip([q, r])
    return chip, q, r, q_charge, r_charge, q_flux


def _build_sequence() -> QuantumSequence:
    chip, q, r, q_charge, r_charge, q_flux = _build_control_chip()
    chip.connect(ControlEquipment(lines=[q_charge, r_charge, q_flux]))
    seq = QuantumSequence(chip)
    seq.charge(q, envelope=Square(duration=10.0, amplitude=0.02), freq=chip.freq(q))
    seq.charge(q, envelope=Square(duration=6.0, amplitude=0.01), freq=chip.freq(q), phase=0.2)
    seq.charge(r, envelope=Square(duration=8.0, amplitude=0.03), freq=chip.freq(r))
    seq.flux(q, envelope=Square(duration=4.0, amplitude=0.015))
    return seq


def _build_chip_with_control() -> Chip:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=6.8, levels=4, label="r")
    coupling = Capacitive(q, r, g=0.02)
    readout = ChargeDrive(target=r, label="readout")
    flux = FluxDrive(target=q, label="flux")
    equipment = ControlEquipment(
        lines=[readout, flux],
        signal_chain=[Crosstalk(source=readout.label, victim=flux.label, beta=0.15, theta=0.3, delay=1.0)],
    )

    chip = Chip([q, r], couplings=[coupling], label="demo", frame="lab")
    chip.connect(equipment)
    return chip


def test_plot_sequence_reuses_supplied_axes() -> None:
    """When an axes is supplied, plot_sequence draws into it and returns its owning figure."""
    fig, ax = plt.subplots()

    returned = plot_sequence(_build_sequence(), ax=ax)

    assert returned is fig
    plt.close(fig)


def test_plot_graph_full_includes_control_nodes(tmp_path: Path) -> None:
    """plot_graph with full=True renders control-line labels as nodes in the output HTML."""
    pytest.importorskip("pyvis")
    path = _build_chip_with_control().plot_graph(str(tmp_path / "full.html"), full=True)

    content = Path(path).read_text()
    assert "flux" in content and "readout" in content


def test_chip_plot_energy_levels_returns_figure_with_state_labels() -> None:
    """plot_energy_levels labels states as kets and anchors the ground-level segment at zero energy."""
    chip = _build_chip_with_control()

    fig = plot_energy_levels(chip)

    assert isinstance(fig, Figure)
    labels = {text.get_text() for text in fig.axes[0].texts}
    assert any(label.startswith("|") for label in labels)
    assert min(segment[0][1] for collection in fig.axes[0].collections for segment in collection.get_segments()) == 0.0

    plt.close(fig)
