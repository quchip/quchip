"""Tests for the backend-free dims-normalization helpers."""

from __future__ import annotations

import pytest

from quchip.backend._dims import normalize_dims_from_list


def test_none_dims_uses_fallback():
    """``None`` dims fall back to a single subsystem of size ``fallback``."""
    assert normalize_dims_from_list(None, fallback=3) == (3,)


def test_none_dims_without_fallback_raises():
    """``None`` dims with no ``fallback`` raise ValueError."""
    with pytest.raises(ValueError):
        normalize_dims_from_list(None, fallback=None)


def test_nested_two_part_dims():
    """QuTiP-style ``[[rows], [cols]]`` dims normalize to the shared flat dims tuple."""
    assert normalize_dims_from_list([[2, 3], [2, 3]]) == (2, 3)


def test_nested_single_part_dims():
    """A single nested ``[[dims]]`` list unwraps to its flat dims tuple."""
    assert normalize_dims_from_list([[2, 3]]) == (2, 3)


def test_flat_single_dim():
    """A flat single-element dims list normalizes to a 1-tuple."""
    assert normalize_dims_from_list([2]) == (2,)


def test_flat_two_dims():
    """A flat two-element dims list normalizes to a 2-tuple."""
    assert normalize_dims_from_list([2, 3]) == (2, 3)


def test_flat_three_dims():
    """A flat three-element dims list normalizes to a 3-tuple."""
    assert normalize_dims_from_list([2, 3, 4]) == (2, 3, 4)
