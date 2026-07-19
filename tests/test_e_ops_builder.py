"""Tests for Chip.e_ops() builder method."""

from __future__ import annotations

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


class TestEOpsBuilder:
    """Chip.e_ops() returns local-space operators keyed by device label."""

    def test_single_string(self, chip_qr):
        """A single operator-name string resolves to a local-space operator."""
        chip, q, _ = chip_qr
        result = chip.e_ops(q="Z")
        assert "q" in result
        op = result["q"]
        assert op.shape == (q.levels, q.levels)

    def test_list_of_strings(self, chip_qr):
        """A list of operator names for one device resolves to a list of local operators."""
        chip, _, r = chip_qr
        result = chip.e_ops(r=["n", "a"])
        assert isinstance(result["r"], list)
        assert len(result["r"]) == 2
        for op in result["r"]:
            assert op.shape == (r.levels, r.levels)

    def test_mixed_devices(self, chip_qr):
        """Requests spanning multiple devices resolve independently per device."""
        chip, q, r = chip_qr
        result = chip.e_ops(q="X", r=["n", "a"])
        assert set(result.keys()) == {"q", "r"}
        assert result["q"].shape == (q.levels, q.levels)
        assert len(result["r"]) == 2

    def test_raw_operator_passthrough(self, chip_qr):
        """A raw backend operator is passed through unchanged."""
        chip, _, r = chip_qr
        raw = r.number_operator()
        result = chip.e_ops(r=raw)
        assert result["r"] is raw

    def test_mixed_list_string_and_raw(self, chip_qr):
        """A list mixing a raw operator and an operator name resolves each entry independently."""
        chip, _, r = chip_qr
        raw_n = r.number_operator()
        result = chip.e_ops(r=[raw_n, "a"])
        assert result["r"][0] is raw_n
        assert result["r"][1].shape == (r.levels, r.levels)

    def test_invalid_device_label(self, chip_qr):
        """An unknown device label raises ValueError or KeyError."""
        chip, _, _ = chip_qr
        with pytest.raises((ValueError, KeyError)):
            chip.e_ops(nonexistent="X")

    def test_invalid_operator_name(self, chip_qr):
        """An unrecognized operator name raises ValueError."""
        chip, _, _ = chip_qr
        with pytest.raises(ValueError, match="Unknown operator"):
            chip.e_ops(q="bogus")

    def test_all_operator_names(self, chip_qr):
        """Every built-in operator name resolves to a correctly shaped local operator."""
        chip, q, _ = chip_qr
        for name in ("X", "Y", "Z", "n", "a", "a_dag", "I"):
            result = chip.e_ops(q=name)
            assert result["q"].shape == (q.levels, q.levels)

    def test_integration_with_decompose_eops(self, chip_qr):
        """e_ops() output is consumable by decompose_eops() with matching flat-op/meta counts."""
        chip, _, _ = chip_qr
        backend = get_default_backend()
        eops = chip.e_ops(q="X", r=["n", "a"])
        flat_ops, meta = decompose_eops(eops, chip, backend)
        assert len(flat_ops) > 0
        assert len(flat_ops) == len(meta)

    def test_dimensions_are_local_not_embedded(self, chip_qr):
        """e_ops() returns local-space operators, not chip-embedded ones."""
        chip, q, r = chip_qr
        full_dim = q.levels * r.levels
        result = chip.e_ops(q="Z", r="n")
        assert result["q"].shape == (q.levels, q.levels)
        assert result["r"].shape == (r.levels, r.levels)
        assert result["q"].shape != (full_dim, full_dim)


class TestCorrelators:
    """Tests for the chip.e_ops(correlators=...) API."""

    def test_correlator_string_keys(self, chip_qr):
        """A correlator keyed by string labels resolves to a pair of local operators."""
        chip, q, r = chip_qr
        result = chip.e_ops(correlators={("q", "r"): ("Z", "n")})
        assert ("q", "r") in result
        op_a, op_b = result[("q", "r")]
        assert op_a.shape == (q.levels, q.levels)
        assert op_b.shape == (r.levels, r.levels)

    def test_correlator_device_object_keys(self, chip_qr):
        """A correlator keyed by device objects is normalized to string-label keys."""
        chip, q, r = chip_qr
        result = chip.e_ops(correlators={(q, r): ("Z", "n")})
        assert ("q", "r") in result

    def test_correlator_mixed_with_single_device(self, chip_qr):
        """Single-device and correlator requests coexist in one e_ops() call."""
        chip, q, r = chip_qr
        result = chip.e_ops(
            q="X",
            correlators={("q", "r"): ("Z", "n")},
        )
        assert "q" in result
        assert ("q", "r") in result

    def test_correlator_integration_with_decompose_eops(self, chip_qr):
        """decompose_eops() preserves tuple keys for correlator entries."""
        chip, _, _ = chip_qr
        backend = get_default_backend()
        eops = chip.e_ops(correlators={("q", "r"): ("Z", "n")})
        flat_ops, meta = decompose_eops(eops, chip, backend)
        assert len(flat_ops) > 0
        assert len(flat_ops) == len(meta)
        for m in meta:
            assert isinstance(m.key, tuple)

    def test_correlator_with_raw_operators(self, chip_qr):
        """Raw backend operators in a correlator are passed through unchanged."""
        chip, q, r = chip_qr
        z_op = q.sigma_z
        n_op = r.number_operator()
        result = chip.e_ops(correlators={("q", "r"): (z_op, n_op)})
        op_a, op_b = result[("q", "r")]
        assert op_a is z_op
        assert op_b is n_op


class TestEOpsDeviceObjectKeys:
    """decompose_eops() accepts device objects as dict keys."""

    def test_device_object_key_single(self, chip_qr):
        """A device-object dict key is normalized to its string label."""
        chip, q, _ = chip_qr
        backend = get_default_backend()
        e_ops = {q: q.sigma_z}
        flat, meta = decompose_eops(e_ops, chip, backend)
        assert len(flat) > 0
        assert len(flat) == len(meta)
        for m in meta:
            assert m.key == "q"

    def test_device_object_key_two_tuple(self, chip_qr):
        """A tuple-of-device-objects dict key is normalized to a tuple of string labels."""
        chip, q, r = chip_qr
        backend = get_default_backend()
        e_ops = {(q, r): (q.sigma_z, r.number_operator())}
        flat, meta = decompose_eops(e_ops, chip, backend)
        assert len(flat) > 0
        for m in meta:
            assert m.key == ("q", "r")

    def test_mixed_string_and_object_keys(self, chip_qr):
        """String and device-object keys can be mixed in the same e_ops dict."""
        chip, q, r = chip_qr
        backend = get_default_backend()
        e_ops = {q: q.sigma_z, "r": r.number_operator()}
        flat, meta = decompose_eops(e_ops, chip, backend)
        keys = {m.key for m in meta}
        assert "q" in keys
        assert "r" in keys
