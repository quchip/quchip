"""Thermal Bath relaxation: steady-state occupation follows n̄(ω,T); device and chip baths coexist."""
import numpy as np
import pytest

from quchip import Bath, Chip, Resonator, simulate
from quchip.utils.constants import k_B


def test_thermal_bath_drives_mode_to_bose_occupation():
    """A thermal bath relaxes a mode from vacuum to its steady-state Bose occupation n_bar(freq, T)."""
    freq, temp, rate = 5.0, 300.0, 0.05  # GHz, mK, 1/ns
    n_bar_expected = 1.0 / np.expm1(freq / (k_B * temp))

    mode = Resonator(freq=freq, levels=12, label="m")
    chip = Chip([mode], baths=[Bath("thermal", temperature=temp, rate=rate)])

    # Evolve from vacuum well past the bath equilibration time (~1/rate).
    tlist = np.linspace(0.0, 400.0, 200)
    result = simulate(
        chip,
        [],
        tlist,
        initial_state=chip.bare_state(),
        e_ops={mode: mode.number_operator()},
    )
    n_final = float(np.real(result.expect("m"))[-1])
    # The equilibration residual at t_max=400ns is exp(-rate*t_max) = exp(-20) ~= 2e-9,
    # negligible. The measured error is dominated by Hilbert-space truncation of the
    # n_bar~0.82 thermal tail at levels=12 (~1e-3 relative, converging to ~7e-8 at
    # levels=25); rel=0.05 keeps a ~50x margin over that truncation floor.
    assert n_final == pytest.approx(n_bar_expected, rel=0.05)


def test_chip_bath_and_device_temperature_sum():
    """A chip-level thermal Bath raises steady-state occupation above the device's own thermal floor."""
    # Chip-level Bath adds a second thermal channel on top of the device's own T1/thermal_population.
    freq, rate = 5.0, 0.05
    mode = Resonator(freq=freq, levels=12, label="m", T1=50.0, thermal_population=0.02)
    chip = Chip([mode], baths=[Bath("thermal", temperature=300.0, rate=rate)])
    tlist = np.linspace(0.0, 400.0, 200)
    result = simulate(
        chip,
        [],
        tlist,
        initial_state=chip.bare_state(),
        e_ops={mode: mode.number_operator()},
    )
    n_final = float(np.real(result.expect("m"))[-1])
    assert n_final > 0.02  # above the device-only thermal floor
