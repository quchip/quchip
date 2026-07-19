"""Physics sentinels for the local patch used by the CR-ring design study."""

from __future__ import annotations

import numpy as np

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    DuffingTransmon,
    Resonator,
    analyze_cr_susceptibility,
    effective_hamiltonian,
    eliminate,
)

# Nominal CR drive amplitude (GHz) used only to convert ZX_per_amplitude into a gate
# time estimate for the echo-residual bound below; not a physical drive being simulated.
_ECHO_GATE_DRIVE_AMPLITUDE = 0.02


def _five_mode_patch():
    spectator = DuffingTransmon(freq=4.88, anharmonicity=-0.30, levels=3, label="s")
    control = DuffingTransmon(freq=5.20, anharmonicity=-0.30, levels=3, label="c")
    target = DuffingTransmon(freq=5.00, anharmonicity=-0.30, levels=3, label="t")
    left_bus = Resonator(freq=6.40, levels=3, label="b_left")
    right_bus = Resonator(freq=6.80, levels=3, label="b_right")
    control_drive = ChargeDrive(control, label="d_c")
    spectator_drive = ChargeDrive(spectator, label="d_s")
    target_drive = ChargeDrive(target, label="d_t")
    legs = [
        Capacitive(spectator, left_bus, g=0.08, rwa=True, label="s_left"),
        Capacitive(control, left_bus, g=0.08, rwa=True, label="c_left"),
        Capacitive(control, right_bus, g=0.08, rwa=True, label="c_right"),
        Capacitive(target, right_bus, g=0.08, rwa=True, label="t_right"),
    ]
    chip = Chip(
        [spectator, control, target, left_bus, right_bus],
        legs,
        control_equipment=ControlEquipment([control_drive, spectator_drive, target_drive]),
        frame="rotating",
        rwa=True,
    )
    return chip, spectator, control, target, left_bus, right_bus, control_drive, legs


def test_cr_patch_is_three_qubits_joined_by_two_independent_buses() -> None:
    """Each bus couples the control to exactly one of its two neighbours."""
    chip, spectator, control, target, left_bus, right_bus, _, _ = _five_mode_patch()

    assert chip.dims == (3, 3, 3, 3, 3)
    assert {
        frozenset((coupling.device_a_label, coupling.device_b_label))
        for coupling in chip.couplings
    } == {
        frozenset((spectator.label, left_bus.label)),
        frozenset((control.label, left_bus.label)),
        frozenset((control.label, right_bus.label)),
        frozenset((target.label, right_bus.label)),
    }


def test_sequential_bus_elimination_leaves_valid_cr_qubit_patch() -> None:
    """SW reduction emits two mediated edges and preserves the control line."""
    chip, spectator, control, target, left_bus, right_bus, control_drive, legs = _five_mode_patch()

    left = eliminate(chip, left_bus, method="sw")
    reduced = eliminate(left.chip, right_bus, method="sw")

    assert {device.label for device in reduced.chip.devices} == {
        spectator.label,
        control.label,
        target.label,
    }
    assert len(reduced.chip.couplings) == 2
    for leg in legs[:2]:
        assert float(left.validity[leg]["g_over_delta"]) < 0.1
    for leg in legs[2:]:
        assert float(reduced.validity[leg]["g_over_delta"]) < 0.1

    response = analyze_cr_susceptibility(reduced.chip, control, target, drive=control_drive)
    assert abs(complex(response.ZX_per_amplitude)) > 1e-6


def test_unreduced_patch_validates_sw_cr_rate_and_next_nearest_exchange() -> None:
    """The exact five-mode patch checks both the CR rate and spectator-target exchange."""
    chip, spectator, control, target, left_bus, right_bus, control_drive, _ = _five_mode_patch()
    reduced = eliminate(eliminate(chip, left_bus, method="sw").chip, right_bus, method="sw").chip

    exact_cr = analyze_cr_susceptibility(chip, control, target, drive=control_drive)
    sw_cr = analyze_cr_susceptibility(reduced, control, target, drive=control_drive)
    # Leg g/Delta ranges 0.044-0.067 across the four legs, setting the perturbative scale
    # for the SW reduction. Measured relative difference between the SW-reduced and exact
    # CR susceptibility magnitudes is ~4.7%, the same order as that leg g/Delta; rtol=0.15
    # keeps a ~3x margin over the measured value.
    np.testing.assert_allclose(
        abs(complex(sw_cr.ZX_per_amplitude)),
        abs(complex(exact_cr.ZX_per_amplitude)),
        rtol=0.15,
    )

    effective = effective_hamiltonian(chip, [spectator, control, target])

    def single_excitation_index(device):
        state = tuple(int(label == device.label) for label in effective.device_labels)
        return effective.basis.index(state)

    spectator_index = single_excitation_index(spectator)
    control_index = single_excitation_index(control)
    target_index = single_excitation_index(target)
    spectator_target = abs(complex(effective.h_eff[spectator_index, target_index]))
    spectator_control = abs(complex(effective.h_eff[spectator_index, control_index]))

    assert spectator_target < 0.01 * spectator_control


def test_echo_model_keeps_only_uncontrolled_spectator_target_zz() -> None:
    """A control echo reverses adjacent ZZ while spectator-target ZZ remains."""
    chip, spectator, control, target, _, _, control_drive, _ = _five_mode_patch()
    response = analyze_cr_susceptibility(chip, control, target, drive=control_drive)
    gate_time = 1.0 / (4.0 * _ECHO_GATE_DRIVE_AMPLITUDE * abs(complex(response.ZX_per_amplitude)))
    adjacent_zz = max(
        abs(float(chip.dispersive_shift(spectator, control))),
        abs(float(chip.dispersive_shift(control, target))),
    )
    residual_zz = abs(float(chip.dispersive_shift(spectator, target)))
    residual_error = (2.0 * np.pi * residual_zz * gate_time) ** 2

    # Measured: residual_zz/adjacent_zz ~= 6.3e-5 (0.01 keeps a ~160x margin) and
    # residual_error ~= 6.7e-9 (1e-6 keeps a ~150x margin) — the spectator-target coupling
    # is two hops removed from both buses, so its dispersive shift is far below the
    # adjacent (one-hop) shifts.
    assert residual_zz < 0.01 * adjacent_zz
    assert residual_error < 1e-6
