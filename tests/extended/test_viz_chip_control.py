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
    seq = QuantumSequence(chip)
    seq.charge(q, envelope=Square(duration=10.0, amplitude=0.02))
    seq.charge(q, envelope=Square(duration=6.0, amplitude=0.01, phase=0.2))
    seq.charge(r, envelope=Square(duration=8.0, amplitude=0.03))
    seq.flux(q, envelope=Square(duration=4.0, amplitude=0.015))
    return seq


def _build_chip_with_control() -> Chip:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=6.8, levels=4, label="r")
    coupling = Capacitive(q, r, g=0.02, rwa=False)
    readout = ChargeDrive(target=r, label="readout")
    flux = FluxDrive(target=q, label="flux")
    equipment = ControlEquipment(
        lines=[readout, flux],
        signal_chain=[Crosstalk(source=readout.label, victim=flux.label, beta=0.15, theta=0.3, delay=1.0)],
    )

    chip = Chip([q, r], couplings=[coupling], label="demo", frame="lab")
    chip.connect(equipment)
    return chip


def _line_by_label(ax: plt.Axes, label: str) -> plt.Line2D:
    for line in ax.lines:
        if line.get_label() == label:
            return line
    raise AssertionError(f"Missing plotted line {label!r}")


def test_plot_sequence_returns_figure_and_expected_lane_labels() -> None:
    """plot_sequence returns a Figure whose y-axis lists one lane per device:drive pair."""
    fig = plot_sequence(_build_sequence())

    assert isinstance(fig, Figure)
    labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    assert labels == ["q:q_charge", "q:q_flux", "r:r_charge"]

    plt.close(fig)


def test_plot_sequence_bar_positions_match_schedule() -> None:
    """Each plotted bar's x-position and width equal the pulse's start time and duration."""
    seq = _build_sequence()

    fig = plot_sequence(seq)

    bars = {patch.get_label(): (patch.get_x(), patch.get_width()) for patch in fig.axes[0].patches}
    assert bars["q:q_charge:0"] == (0.0, 10.0)
    assert bars["q:q_charge:1"] == (10.0, 6.0)
    assert bars["q:q_flux:0"] == (0.0, 4.0)
    assert bars["r:r_charge:0"] == (0.0, 8.0)

    plt.close(fig)


def test_plot_sequence_reuses_supplied_axes() -> None:
    """When an axes is supplied, plot_sequence draws into it and returns its owning figure."""
    fig, ax = plt.subplots()

    returned = plot_sequence(_build_sequence(), ax=ax)

    assert returned is fig
    plt.close(fig)


def test_control_chip_exposes_crosstalk() -> None:
    """A connected chip exposes its control equipment and the crosstalk source/victim labels."""
    chip = _build_chip_with_control()
    equipment = chip.control_equipment
    assert equipment is not None
    crosstalks = chip.crosstalks
    assert crosstalks
    assert crosstalks[0].source == "readout"
    assert crosstalks[0].victim == "flux"


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
