"""Durable observable-builder and decomposition behavior."""

from __future__ import annotations

import numpy as np
import pytest

from quchip.backend import get_default_backend
from quchip.chip.chip import Chip
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.engine.stage3_observables import decompose_eops


@pytest.fixture()
def chip_qr():
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=6.2, levels=5, label="r")
    return Chip([q, r]), q, r


def test_unknown_device_is_rejected_at_the_builder_boundary(chip_qr):
    chip, _, _ = chip_qr
    with pytest.raises((ValueError, KeyError)):
        chip.e_ops(nonexistent="X")


def test_named_observables_decompose_into_solver_bands(chip_qr):
    chip, _, _ = chip_qr
    flat, meta = decompose_eops(
        chip.e_ops(q=["X", "Y", "Z"], r=["n", "a"]),
        chip,
        get_default_backend(),
    )
    assert flat
    assert len(flat) == len(meta)
    assert {entry.key for entry in meta} == {"q", "r"}


def test_unknown_operator_is_rejected_during_resolution(chip_qr):
    chip, _, _ = chip_qr
    with pytest.raises(ValueError, match="Unknown operator"):
        decompose_eops(chip.e_ops(q="bogus"), chip, get_default_backend())


def test_raw_authored_operator_is_preserved_in_native_basis(chip_qr):
    chip, q, _ = chip_qr
    raw = np.asarray(q.local_space().matrix("n"))
    flat, _ = decompose_eops({q: raw}, chip, get_default_backend())
    reconstructed = sum(get_default_backend().to_array(op) for op in flat)
    expected = get_default_backend().to_array(chip.observable(q, raw))
    np.testing.assert_allclose(reconstructed, expected)


def test_correlator_specs_resolve_and_retain_the_pair_key(chip_qr):
    chip, q, r = chip_qr
    flat, meta = decompose_eops(
        chip.e_ops(correlators={(q, r): ("Z", "n")}),
        chip,
        get_default_backend(),
    )
    assert flat
    assert len(flat) == len(meta)
    assert {entry.key for entry in meta} == {("q", "r")}


def test_device_object_and_string_keys_can_be_mixed(chip_qr):
    chip, q, r = chip_qr
    flat, meta = decompose_eops(
        {q: q.local_space().matrix("n"), "r": r.local_space().matrix("n")},
        chip,
        get_default_backend(),
    )
    assert flat
    assert {entry.key for entry in meta} == {"q", "r"}
