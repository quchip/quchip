"""Physical microwave ports and internal/external loss ownership."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import (
    Capacitive,
    ChargeBasisTransmon,
    Chip,
    CollapseChannel,
    DuffingTransmon,
    Port,
    Resonator,
    eliminate,
)


def test_internal_loss_and_ports_are_distinct_collapse_channels() -> None:
    """A resonator's internal Q and each external port contribute separately owned loss."""
    resonator = Resonator(
        freq=6.8,
        levels=5,
        internal_quality_factor=200_000,
        label="r",
    )
    input_port = Port(resonator, external_quality_factor=15_000, label="in")
    output_port = Port(resonator, external_quality_factor=18_000, label="out")
    chip = Chip([resonator], ports=[input_port, output_port])

    terms = chip.resolve().collapse_terms
    port_terms = chip.resolve().port_terms
    rates = {term.source: float(term.rate) for term in terms}
    paths = {term.source: term.parameter_paths for term in terms}

    assert set(rates) == {"r", "in", "out"}
    np.testing.assert_allclose(rates["r"], 2 * np.pi * 6.8 / 200_000)
    np.testing.assert_allclose(rates["in"], 2 * np.pi * 6.8 / 15_000)
    np.testing.assert_allclose(rates["out"], 2 * np.pi * 6.8 / 18_000)
    assert paths["in"] == ("r.freq", "port.in.external_quality_factor")
    assert [term.label for term in port_terms] == ["in", "out"]
    assert port_terms[0] is next(term for term in terms if term.source == "in")


def test_loaded_quality_factor_name_is_removed() -> None:
    """The old ambiguous loaded-Q constructor name is not accepted as an alias."""
    with pytest.raises(TypeError, match="quality_factor"):
        Resonator(freq=6.8, levels=4, quality_factor=10_000)


def test_collective_port_round_trips_and_connects_partition() -> None:
    """A matrix-valued two-device port serializes and makes its shared bath one component."""
    first = Resonator(freq=6.0, levels=2, label="a")
    second = Resonator(freq=6.2, levels=2, label="b")
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    operator = np.kron(lowering, np.eye(2)) + 0.5j * np.kron(np.eye(2), lowering)
    port = Port([first, second], rate=0.02, operator=operator, phase=0.3, label="feedline")
    chip = Chip([first, second], ports=[port])

    restored = Chip.from_dict(chip.to_dict())

    assert chip.partition().is_trivial
    assert restored.port("feedline").resolve_targets(restored) == ("a", "b")
    np.testing.assert_allclose(restored.port("feedline").operator, operator)
    term = next(term for term in restored.resolve().collapse_terms if term.source == "feedline")
    np.testing.assert_allclose(term.operator.to_dense(), operator)


def test_port_parameters_rebind_without_mutating_source() -> None:
    """Port rate and phase use stable parameter paths and immutable chip rebinding."""
    resonator = Resonator(freq=6.8, levels=3, label="r")
    chip = Chip([resonator], ports=[Port(resonator, rate=0.01, phase=0.2, label="readout")])

    shifted = chip.with_params({"port.readout.rate": 0.03, "port.readout.phase": -0.4})

    assert chip.port("readout").rate == pytest.approx(0.01)
    assert chip.port("readout").phase == pytest.approx(0.2)
    assert shifted.port("readout").rate == pytest.approx(0.03)
    assert shifted.port("readout").phase == pytest.approx(-0.4)


def test_collective_port_can_span_more_than_two_devices() -> None:
    """A general port operator embeds on arbitrary declared support."""
    devices = [Resonator(freq=6.0 + 0.1 * index, levels=2, label=f"r{index}") for index in range(3)]
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    operator = (
        np.kron(np.kron(lowering, np.eye(2)), np.eye(2))
        + np.kron(np.kron(np.eye(2), lowering), np.eye(2))
        + np.kron(np.kron(np.eye(2), np.eye(2)), lowering)
    )
    chip = Chip(devices, ports=[Port(devices, rate=0.01, operator=operator, label="common")])

    resolved = chip.resolve(frame={device.label: 6.0 for device in devices})
    term = next(term for term in resolved.collapse_terms if term.source == "common")

    np.testing.assert_allclose(term.operator.to_dense(), operator)
    assert term.frame_frequency == pytest.approx(6.0)
    assert chip.partition().is_trivial


def test_elimination_refuses_to_drop_a_port_boundary() -> None:
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    resonator = Resonator(freq=6.0, levels=4, label="r")
    chip = Chip(
        [qubit, resonator],
        [Capacitive(qubit, resonator, g=0.04)],
        ports=[Port(resonator, rate=0.03, label="readout")],
    )

    with pytest.raises(NotImplementedError, match="effective input-output port"):
        eliminate(chip, resonator)


def test_port_coupling_kind_cannot_be_made_ambiguous_after_construction() -> None:
    resonator = Resonator(freq=6.0, levels=3, label="r")
    port = Port(resonator, rate=0.02, label="p")

    with pytest.raises(ValueError, match="exactly one"):
        port.external_quality_factor = 10_000
    with pytest.raises(ValueError, match="exactly one"):
        port.rate = None


def test_port_copy_does_not_share_a_mutable_matrix() -> None:
    resonator = Resonator(freq=6.0, levels=2, label="r")
    operator = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    port = Port(resonator, rate=0.02, operator=operator, label="p")

    copied = port.copy()
    copied.operator[0, 1] = 0.5

    assert port.operator[0, 1] == 1.0


def test_port_with_mixed_frame_bands_is_rejected_as_dynamic() -> None:
    resonator = Resonator(freq=6.0, levels=3, label="r")
    lowering = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2.0)], [0.0, 0.0, 0.0]],
        dtype=complex,
    )
    port = Port(resonator, rate=0.02, operator=lowering + lowering.T, label="p")
    chip = Chip([resonator], ports=[port])

    assert chip.resolve(frame="lab").port_terms[0].frame_frequency == 0.0
    with pytest.raises(ValueError, match="different phases"):
        chip.resolve(frame={"r": 6.0})

    nearly_lowering = Port(resonator, rate=0.02, operator=lowering + 1e-11 * lowering.T, label="p")
    near_chip = Chip([resonator], ports=[nearly_lowering])
    assert near_chip.resolve(frame={"r": 6.0}).port_terms[0].frame_frequency == pytest.approx(6.0)


def test_explicit_port_uses_the_resolved_energy_frame_for_charge_basis_devices() -> None:
    """An explicit energy-lowering port is stationary in a non-Fock device's energy frame."""
    transmon = ChargeBasisTransmon(
        E_C=0.2,
        E_J=20.0,
        n_g=0.0,
        levels=3,
        num_basis=31,
        basis="eigen",
        label="q",
    )
    vectors = np.asarray(transmon.eigenvectors())
    lowering = np.outer(vectors[:, 0], vectors[:, 1].conj())
    port = Port(transmon, rate=0.02, operator=lowering, label="drive")

    term = Chip([transmon], ports=[port]).resolve(frame={"q": transmon.freq}).port_terms[0]

    assert term.frame_frequency == pytest.approx(transmon.freq)


def test_in_place_port_operator_mutation_invalidates_resolution_cache() -> None:
    """Mutating a public dense port operator cannot return stale resolved physics."""
    resonator = Resonator(freq=6.0, levels=2, label="r")
    operator = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    chip = Chip([resonator], ports=[Port(resonator, rate=0.02, operator=operator, label="p")])
    first = chip.resolve(frame="lab")

    operator[0, 1] = 0.5
    second = chip.resolve(frame="lab")

    assert second is not first
    np.testing.assert_allclose(second.port_terms[0].operator.to_dense(), operator)


def test_port_collapse_ownership_does_not_depend_on_display_labels() -> None:
    """A same-labelled component channel cannot hide the port's own collapse term."""

    class CollidingResonator(Resonator):
        def dissipation(self, op, p):
            return super().dissipation(op, p) + (
                CollapseChannel(op.a, 0.01, "external_coupling"),
            )

    resonator = CollidingResonator(freq=6.0, levels=3, label="shared")
    port = Port(resonator, rate=0.02, label="shared")

    resolved = Chip([resonator], ports=[port]).resolve(frame="lab")

    assert len(resolved.collapse_terms) == 2
    assert resolved.port_terms[0].rate == pytest.approx(0.02)
