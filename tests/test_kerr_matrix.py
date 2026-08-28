"""Public dressed self-Kerr and cross-Kerr matrix API."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import Capacitive, Chip, CrossKerr, DuffingTransmon, KerrCavity, KerrMatrix, Resonator


def test_kerr_matrix_result_is_frozen_labeled_and_a_jax_pytree() -> None:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    values = jnp.asarray([[-0.25, -0.001], [-0.001, 0.0]])
    matrix = KerrMatrix(labels=("q", "r"), values=values)

    assert matrix.labels == ("q", "r")
    np.testing.assert_allclose(matrix.values, values)
    assert matrix[q, "r"] == pytest.approx(-0.001)
    assert matrix["r", q] == pytest.approx(-0.001)

    leaves, treedef = jax.tree_util.tree_flatten(matrix)
    assert len(leaves) == 1
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.labels == matrix.labels
    np.testing.assert_allclose(rebuilt.values, matrix.values)

    with pytest.raises(FrozenInstanceError):
        matrix.labels = ("other",)  # type: ignore[misc]


def test_kerr_matrix_lookup_names_available_labels() -> None:
    matrix = KerrMatrix(labels=("q", "r"), values=jnp.zeros((2, 2)))

    with pytest.raises(KeyError, match=r"missing.*q.*r"):
        _ = matrix["missing", "r"]


@pytest.mark.parametrize(
    ("labels", "values", "match"),
    [
        (("q", "q"), jnp.zeros((2, 2)), "unique"),
        (("q", "r"), jnp.zeros((2, 3)), "shape"),
        (("q",), jnp.asarray([[1.0j]]), "real"),
        (("q", "r"), jnp.asarray([[0.0, 1.0], [2.0, 0.0]]), "symmetric"),
    ],
)
def test_kerr_matrix_rejects_ambiguous_or_invalid_data(labels, values, match) -> None:
    with pytest.raises(ValueError, match=match):
        KerrMatrix(labels=labels, values=values)


def test_chip_kerr_matrix_matches_scalar_dressed_observables_in_chip_order() -> None:
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.24, levels=3, label="q1")
    bus = Resonator(freq=7.0, levels=3, label="bus")
    chip = Chip(
        [q0, q1, bus],
        [Capacitive(q0, bus, g=0.035), Capacitive(q1, bus, g=0.04)],
    )

    matrix = chip.kerr_matrix()

    assert matrix.labels == ("q0", "q1", "bus")
    assert matrix.values.shape == (3, 3)
    assert not jnp.iscomplexobj(matrix.values)
    np.testing.assert_allclose(matrix.values, matrix.values.T, atol=0.0)
    for index, device in enumerate(chip.devices):
        assert matrix.values[index, index] == pytest.approx(chip.dressed_anharmonicity(device), abs=1e-13)
    for row, device_a in enumerate(chip.devices):
        for column, device_b in enumerate(chip.devices[row + 1 :], start=row + 1):
            assert matrix.values[row, column] == pytest.approx(chip.dispersive_shift(device_a, device_b), abs=1e-13)


def test_kerr_matrix_uses_duffing_and_kerr_cavity_diagonal_conventions() -> None:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    cavity = KerrCavity(freq=6.0, kerr=0.012, levels=4, label="cavity")

    assert Chip([q]).kerr_matrix()[q, q] == pytest.approx(-0.25)
    assert Chip([cavity]).kerr_matrix()[cavity, cavity] == pytest.approx(-0.024)


def test_direct_cross_kerr_edge_appears_as_full_pull() -> None:
    left = Resonator(freq=5.0, levels=3, label="left")
    right = Resonator(freq=7.0, levels=3, label="right")
    chip = Chip([left, right], [CrossKerr(left, right, chi=-0.0012)])

    matrix = chip.kerr_matrix()

    assert matrix[left, right] == pytest.approx(-0.0012, abs=1e-13)
    assert matrix[right, left] == pytest.approx(-0.0012, abs=1e-13)


def test_two_level_device_has_nan_only_on_its_kerr_diagonal() -> None:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    chip = Chip([q, r], [CrossKerr(q, r, chi=-0.001)])

    matrix = chip.kerr_matrix()

    assert jnp.isnan(matrix[q, q])
    assert matrix[q, r] == pytest.approx(-0.001, abs=1e-13)
    assert matrix[r, r] == pytest.approx(0.0, abs=1e-13)


def test_kerr_matrix_computes_one_labeled_eigensystem(monkeypatch) -> None:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.05)])
    original = chip.analysis._compute_array_labeled
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(chip.analysis, "_compute_array_labeled", counted)

    _ = chip.kerr_matrix()

    assert calls == 1
