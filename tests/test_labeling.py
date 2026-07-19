"""Tests for label resolution and label-keyed dict lookup."""

from __future__ import annotations

import pytest

from quchip.utils.labeling import (
    LabelKeyedDict,
    bare_label_from_mapping,
    merge_labeled_values,
)


class _FakeDevice:
    """Minimal label-bearing stand-in for a device/coupling/drive object."""

    def __init__(self, label: str) -> None:
        self.label = label


def test_label_keyed_dict_two_tuple_lookup_is_order_independent():
    """A 2-tuple key matches lookup and membership in either order."""
    d = LabelKeyedDict()
    d[("a", "b")] = 1.0
    assert d[("b", "a")] == 1.0
    assert ("b", "a") in d


def test_label_keyed_dict_three_tuple_lookup_is_order_sensitive():
    """A 3-tuple key does not match its reversal."""
    d = LabelKeyedDict()
    d[("a", "b", "c")] = 1.0
    assert ("c", "b", "a") not in d
    with pytest.raises(KeyError):
        d[("c", "b", "a")]


def test_merge_labeled_values_raises_on_duplicate_mapping_keys():
    """Two mapping keys resolving to the same label raise ValueError."""
    dev = _FakeDevice("q0")
    with pytest.raises(ValueError, match="q0"):
        merge_labeled_values({dev: 1, "q0": 2}, {})


def test_merge_labeled_values_raises_on_mapping_kwargs_collision():
    """A label present in both the mapping and kwargs raises ValueError."""
    with pytest.raises(ValueError, match="q0"):
        merge_labeled_values({"q0": 1}, {"q0": 2})


def test_bare_label_from_mapping_fills_unmentioned_devices_with_zero():
    """Devices absent from the spec default to Fock index 0."""
    result = bare_label_from_mapping(["q0", "q1", "q2"], {"q1": 3}, {})
    assert result == (0, 3, 0)


def test_bare_label_from_mapping_raises_on_unknown_label():
    """A label outside device_labels raises ValueError."""
    with pytest.raises(ValueError, match="qX"):
        bare_label_from_mapping(["q0", "q1"], None, {"qX": 1})
