"""``BaseDevice.intrinsic_decay_rate()`` — the device-owned Purcell-fold decay hook."""

from __future__ import annotations

import numpy as np
import pytest

from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon


def test_base_device_intrinsic_decay_rate_is_1_over_t1():
    """The base hook reports 1/T1 when T1 is the device's only decay channel."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=20_000.0)
    assert q.intrinsic_decay_rate() == pytest.approx(1.0 / 20_000.0)


def test_base_device_intrinsic_decay_rate_is_none_without_t1():
    """The base hook reports no decay channel when T1 is unset."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    assert q.intrinsic_decay_rate() is None


def test_resonator_intrinsic_decay_rate_combines_q_and_t1():
    """Resonator sums its Q-derived photon loss and T1 rates rather than prioritizing one."""
    r = Resonator(freq=7.0, quality_factor=5000.0, levels=6, label="r", T1=15_000.0)
    kappa = 2 * np.pi * 7.0 / 5000.0
    t1_rate = 1.0 / 15_000.0
    assert r.intrinsic_decay_rate() == pytest.approx(kappa + t1_rate)


def test_resonator_intrinsic_decay_rate_is_none_without_q_or_t1():
    """A resonator with neither quality_factor nor T1 reports no decay channel."""
    r = Resonator(freq=7.0, levels=6, label="r")
    assert r.intrinsic_decay_rate() is None


def test_base_device_intrinsic_decay_rate_with_t1_and_thermal_population():
    """T1 plus a nonzero thermal_population boosts the downward rate to (n_bar+1)/T1."""
    q = DuffingTransmon(
        freq=5.0, anharmonicity=-0.25, levels=3, label="q", T1=20_000.0, thermal_population=0.05
    )
    assert q.intrinsic_decay_rate() == pytest.approx((0.05 + 1.0) / 20_000.0)


def test_base_device_intrinsic_decay_rate_thermal_population_only():
    """With T1 unset, thermal_population alone drives the gamma=1 channel's downward rate n_bar+1."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q", thermal_population=0.05)
    assert q.intrinsic_decay_rate() == pytest.approx(0.05 + 1.0)


def test_resonator_intrinsic_decay_rate_combines_q_t1_and_thermal_population():
    """Resonator's Q-derived kappa is unaffected by thermal_population; only the T1 channel is boosted."""
    r = Resonator(
        freq=7.0, quality_factor=5000.0, levels=6, label="r", T1=15_000.0, thermal_population=0.05
    )
    kappa = 2 * np.pi * 7.0 / 5000.0
    t1_rate = (0.05 + 1.0) / 15_000.0
    assert r.intrinsic_decay_rate() == pytest.approx(kappa + t1_rate)
