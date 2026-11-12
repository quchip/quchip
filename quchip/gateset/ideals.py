"""Ideal gate unitaries, gate-name aliases, and the built-in decomposition rules.

Matrices use the qubit order of their arguments, with the first qubit most
significant, as in QuTiP's tensor order. ``RZ(θ) = exp(-iθZ/2)``, and the
other rotations follow the same sign.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)
PAULIS = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _rotation(pauli: np.ndarray) -> Callable[[float], np.ndarray]:
    return lambda theta: np.cos(theta / 2) * _I - 1j * np.sin(theta / 2) * pauli


def _zx(theta: float) -> np.ndarray:
    generator = np.kron(_Z, _X)
    return np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * generator


#: Registered ideal unitaries. A callable takes the gate's parameters.
IDEALS: dict[str, np.ndarray | Callable[..., np.ndarray]] = {
    "I": _I,
    "X": _X,
    "Y": _Y,
    "Z": _Z,
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "S": np.diag([1, 1j]),
    "T": np.diag([1, np.exp(1j * np.pi / 4)]),
    "SX": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]),
    "RX": _rotation(_X),
    "RY": _rotation(_Y),
    "RZ": _rotation(_Z),
    "ZX90": _zx(np.pi / 2),
    "RZX": _zx,
    "CX": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
    "CZ": np.diag([1, 1, 1, -1]).astype(complex),
    "SWAP": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex),
    "ISWAP": np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]),
    "SQRTISWAP": np.array(
        [[1, 0, 0, 0], [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0], [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 0, 1]]
    ),
}

#: Alternative names, including qutip-qip's.
ALIASES = {"SQRTX": "SX", "SQRTNOT": "SX", "CNOT": "CX", "CSIGN": "CZ", "SNOT": "H", "PHASEGATE": "RZ"}

#: Instructions that take no pulse.
VIRTUAL = ("RZ", "IDLE", "GLOBALPHASE")

#: Built-in rules: name -> (qubit count, expansion). An expansion maps the
#: gate's parameters to a time-ordered list of instructions on qubit slots.
RULES: dict[str, tuple[int, Callable[..., list[tuple[Any, ...]]]]] = {
    "RX": (1, lambda theta: [("RZ", 0, -np.pi / 2), ("SX", 0), ("RZ", 0, np.pi - theta), ("SX", 0),
                             ("RZ", 0, -np.pi / 2)]),
    "RY": (1, lambda theta: [("RZ", 0, -np.pi / 2), ("RX", 0, theta), ("RZ", 0, np.pi / 2)]),
    "H": (1, lambda: [("RZ", 0, np.pi / 2), ("SX", 0), ("RZ", 0, np.pi / 2)]),
    "X": (1, lambda: [("SX", 0), ("SX", 0)]),
    "Y": (1, lambda: [("RZ", 0, -np.pi / 2), ("X", 0), ("RZ", 0, np.pi / 2)]),
    "Z": (1, lambda: [("RZ", 0, np.pi)]),
    "S": (1, lambda: [("RZ", 0, np.pi / 2)]),
    "T": (1, lambda: [("RZ", 0, np.pi / 4)]),
    "I": (1, lambda: []),
    "CX": (2, lambda: [("ZX90", (0, 1)), ("RZ", 0, -np.pi / 2), ("RX", 1, -np.pi / 2)]),
    "CZ": (2, lambda: [("H", 1), ("CX", (0, 1)), ("H", 1)]),
    "SWAP": (2, lambda: [("CX", (0, 1)), ("H", 0), ("H", 1), ("CX", (0, 1)), ("H", 0), ("H", 1), ("CX", (0, 1))]),
}

#: Built-in rules that apply when the rules above do not resolve a gate, such
#: as a CX against the direction of the gate set's ZX90.
FALLBACK_RULES: dict[str, tuple[int, Callable[..., list[tuple[Any, ...]]]]] = {
    "CX": (2, lambda: [("H", 0), ("H", 1), ("CX", (1, 0)), ("H", 0), ("H", 1)]),
}


def canonical_name(name: str) -> str:
    """Return the registered spelling of a gate name."""
    upper = str(name).upper()
    return ALIASES.get(upper, upper)


def ideal_unitary(ideal: Any, params: Sequence[float] = ()) -> np.ndarray:
    """Return an ideal unitary from a registered name or a matrix.

    Parameters
    ----------
    ideal : str or array_like
        Registered gate name or a ``2^n x 2^n`` unitary.
    params : sequence of float
        Parameters of a parameterized gate, such as an angle.
    """
    if isinstance(ideal, str):
        name = canonical_name(ideal)
        if name not in IDEALS:
            raise KeyError(f"No registered ideal for {ideal!r}. Registered: {sorted(IDEALS)}.")
        entry = IDEALS[name]
        matrix = entry(*params) if callable(entry) else entry
    else:
        matrix = ideal
    matrix = np.asarray(matrix, dtype=complex)
    dimension = matrix.shape[0]
    if matrix.ndim != 2 or matrix.shape[1] != dimension or dimension & (dimension - 1) or dimension < 2:
        raise ValueError(f"An ideal unitary must be 2^n x 2^n; got shape {matrix.shape}.")
    if np.abs(matrix.conj().T @ matrix - np.eye(dimension)).max() > 1e-9:
        raise ValueError("The ideal matrix is not unitary.")
    return matrix


def embed(matrix: np.ndarray, positions: Sequence[int], count: int) -> np.ndarray:
    """Embed a gate on qubit ``positions`` into ``count`` qubits, first most significant."""
    k = len(positions)
    gate = np.asarray(matrix).reshape((2,) * (2 * k))
    identity = np.eye(2 ** count, dtype=complex).reshape((2,) * (2 * count))
    out = np.tensordot(gate, identity, axes=(list(range(k, 2 * k)), list(positions)))
    return np.moveaxis(out, list(range(k)), list(positions)).reshape(2 ** count, 2 ** count)


def pauli_components(generator: np.ndarray) -> dict[str, float]:
    """Return the coefficients ``c_P = Tr(P H) / d`` of a Hermitian generator on Pauli strings."""
    count = int(round(np.log2(generator.shape[0])))
    out = {}
    for labels in itertools.product("IXYZ", repeat=count):
        name = "".join(labels)
        if set(name) == {"I"}:
            continue
        pauli = np.array([[1.0]])
        for label in labels:
            pauli = np.kron(pauli, PAULIS[label])
        out[name] = float(np.real(np.trace(pauli @ generator)) / generator.shape[0])
    return out


def rule_table(user_rules: Mapping[str, list[tuple[int, Callable[..., Any]]]]) -> dict[str, list[tuple[int, Any]]]:
    """Merge user rules ahead of the built-in ones and the built-in ones ahead of the fallbacks."""
    table: dict[str, list[tuple[int, Any]]] = {name: [rule] for name, rule in RULES.items()}
    for name, rule in FALLBACK_RULES.items():
        table.setdefault(name, []).append(rule)
    for name, rules in user_rules.items():
        table[name] = list(rules) + table.get(name, [])
    return table
