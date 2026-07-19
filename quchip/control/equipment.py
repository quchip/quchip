"""Control-equipment container: drive lines and signal chain.

Signal transforms are owned here, not by individual drives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quchip.control.drive import BaseDrive
from quchip.control.signal import Crosstalk, SignalMap, SignalTransform
from quchip.engine.ir import Add, PolarScale, Shift, SignalProgram
from quchip.utils.jax_utils import (
    is_jax_array as _is_traced,
    select_array_module as _select_array_module,
)


def _setitem(arr: Any, idx: tuple[int, int], value: Any) -> Any:
    """Functional setitem: uses ``.at[...].set`` for JAX, ``copy+assign`` for NumPy."""
    if hasattr(arr, "at"):
        return arr.at[idx].set(value)
    out = arr.copy()
    out[idx] = value
    return out


@dataclass(frozen=True)
class CrosstalkMatrix(SignalTransform):
    """Dense crosstalk transform and matrix view in control-line order.

    Attributes
    ----------
    labels : tuple[str, ...]
        Drive labels in wiring order (the order drives appear in
        :attr:`ControlEquipment.lines`). Row / column ``i`` corresponds
        to ``labels[i]``.
    beta : Any
        ``[n, n]`` amplitude matrix. ``beta[i, j]`` is the leakage
        amplitude from source ``labels[j]`` onto victim ``labels[i]``
        (column = source, row = victim). Diagonals represent
        self-coupling and are conventionally ``1.0``.
    theta : Any
        ``[n, n]`` phase matrix (radians), same indexing as ``beta``.
    delay : Any
        ``[n, n]`` delay matrix (ns), same indexing as ``beta``.

    Notes
    -----
    Every off-diagonal edge reads the same input signal map, so reciprocal
    entries form one linear mixing stage without recursively leaking one
    another's output. Matrix entries flow directly into the signal-program IR
    (``PolarScale``/``Shift``), preserving end-to-end JAX traceability.
    """

    labels: tuple[str, ...]
    beta: Any
    theta: Any
    delay: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        shape = (len(self.labels), len(self.labels))
        for name in ("beta", "theta", "delay"):
            matrix = getattr(self, name)
            if getattr(matrix, "shape", None) != shape:
                raise ValueError(f"{name} matrix shape {getattr(matrix, 'shape', None)} does not match {shape}")

    def apply(self, signals: SignalMap) -> SignalMap:
        """Apply all directed leakage edges to one shared input snapshot."""
        output = dict(signals)
        line_index = {label: index for index, label in enumerate(self.labels)}
        for key, signal in signals.items():
            source_index = line_index.get(key[0])
            if source_index is None:
                continue
            for victim_index, victim in enumerate(self.labels):
                if victim_index == source_index:
                    continue
                leaked: SignalProgram = PolarScale(
                    child=Shift(signal, delta_t=self.delay[victim_index, source_index]),
                    amplitude=self.beta[victim_index, source_index],
                    theta=self.theta[victim_index, source_index],
                )
                victim_key = (victim, key[1])
                existing = output.get(victim_key)
                output[victim_key] = leaked if existing is None else Add((existing, leaked))
        return output

    def referenced_lines(self) -> tuple[str, ...]:
        return self.labels

    def without_line(self, line: str) -> "CrosstalkMatrix | None":
        if line not in self.labels:
            return self
        keep = [index for index, label in enumerate(self.labels) if label != line]
        if len(keep) < 2:
            return None
        labels = tuple(self.labels[index] for index in keep)
        return type(self)(
            labels=labels,
            beta=self.beta[keep][:, keep],
            theta=self.theta[keep][:, keep],
            delay=self.delay[keep][:, keep],
        )

    def edges(self) -> list[Crosstalk]:
        """Return the directed-edge view used by topology and visualization."""
        return [
            Crosstalk(
                source=self.labels[source],
                victim=self.labels[victim],
                beta=self.beta[victim, source],
                theta=self.theta[victim, source],
                delay=self.delay[victim, source],
            )
            for victim in range(len(self.labels))
            for source in range(len(self.labels))
            if victim != source
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-safe dictionary."""
        data = super().to_dict()
        data["labels"] = list(self.labels)
        for name in ("beta", "theta", "delay"):
            matrix = getattr(self, name)
            data[name] = [
                [float(matrix[i, j]) for j in range(len(self.labels))]
                for i in range(len(self.labels))
            ]
        return data

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CrosstalkMatrix":
        return cls(
            labels=tuple(str(label) for label in d["labels"]),
            beta=np.asarray(d["beta"], dtype=float),
            theta=np.asarray(d["theta"], dtype=float),
            delay=np.asarray(d["delay"], dtype=float),
        )


class ControlEquipment:
    """Ordered drive lines plus a sequence of signal-chain transforms.

    Drives produce raw line signals; the equipment pipes them through
    ``signal_chain`` in order before the engine assembles Hamiltonian terms.
    """

    def __init__(
        self,
        lines: list[BaseDrive],
        *,
        signal_chain: list[SignalTransform] | None = None,
    ) -> None:
        self._lines = list(lines)
        self._signal_chain = list(signal_chain) if signal_chain else []

    @property
    def lines(self) -> list[BaseDrive]:
        """Ordered drive lines (defensive copy)."""
        return list(self._lines)

    @property
    def signal_chain(self) -> list[SignalTransform]:
        """Signal-chain transforms (defensive copy)."""
        return list(self._signal_chain)

    def apply_signal_chain(self, signals: SignalMap) -> SignalMap:
        """Apply every signal-chain transform to *signals*, in order.

        Each transform receives the previous transform's output, so
        transforms compose sequentially: reordering :attr:`signal_chain`
        changes the result (e.g. a :class:`Delay` applied before a
        :class:`Gain` sees the undelayed signal).

        Parameters
        ----------
        signals : SignalMap
            ``{(drive_label, drive_index): SignalProgram}`` map of raw
            line signals. ``drive_index`` distinguishes multiple ops on
            the same drive line and is assigned by the engine when it
            enumerates the chip's scheduled drive ops.

        Returns
        -------
        SignalMap
            Transformed signal map. May contain keys absent from
            *signals*: a :class:`Crosstalk` transform, for example,
            adds an entry under the victim drive's label for every
            source entry it leaks from.
        """
        built = dict(signals)
        for transform in self._signal_chain:
            built = transform.apply(built)
        return built

    @property
    def crosstalks(self) -> list[Crosstalk]:
        """Directed crosstalk edges represented by the signal chain."""
        edges: list[Crosstalk] = []
        for transform in self._signal_chain:
            if isinstance(transform, Crosstalk):
                edges.append(transform)
            elif isinstance(transform, CrosstalkMatrix):
                edges.extend(transform.edges())
        return edges

    def crosstalk_matrix(self) -> CrosstalkMatrix:
        """Return a dense matrix view of the :class:`Crosstalk` transforms.

        The matrix uses wiring order (``self.lines``) as the stable axis
        ordering. Column index = source drive, row index = victim drive.
        Diagonal entries are ``beta=1``, ``theta=0``, ``delay=0`` by
        convention (self-coupling). Off-diagonal entries aggregate every
        :class:`Crosstalk` transform present in the signal chain; lines
        with no corresponding transform contribute zeros.

        Non-:class:`Crosstalk` transforms (``Gain``, ``Delay``) are
        ignored here; this is strictly a view of the crosstalk edges.

        Returns
        -------
        CrosstalkMatrix
            ``labels`` (wiring order), ``beta``, ``theta``, ``delay``
            as ``[n, n]`` arrays. Arrays use ``jax.numpy`` when any
            stored entry is a JAX tracer or array, otherwise
            ``numpy``.
        """
        labels = tuple(line.label for line in self._lines)
        index = {label: i for i, label in enumerate(labels)}
        n = len(labels)

        entries: list[tuple[int, int, Any, Any, Any]] = []
        any_traced = False
        for t in self.crosstalks:
            if t.source not in index or t.victim not in index:
                continue
            i = index[t.victim]
            j = index[t.source]
            entries.append((i, j, t.beta, t.theta, t.delay))
            for val in (t.beta, t.theta, t.delay):
                if _is_traced(val):
                    any_traced = True

        xp = _select_array_module(any_traced)

        beta = xp.eye(n, dtype=float) if n else xp.zeros((0, 0), dtype=float)
        theta = xp.zeros((n, n), dtype=float)
        delay = xp.zeros((n, n), dtype=float)

        for i, j, b, th, dl in entries:
            beta = _setitem(beta, (i, j), xp.asarray(b, dtype=float))
            theta = _setitem(theta, (i, j), xp.asarray(th, dtype=float))
            delay = _setitem(delay, (i, j), xp.asarray(dl, dtype=float))

        return CrosstalkMatrix(labels=labels, beta=beta, theta=theta, delay=delay)

    def set_crosstalk_matrix(
        self,
        beta: Any,
        theta: Any | None = None,
        delay: Any | None = None,
        *,
        labels: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """Rehydrate the crosstalk edges from dense matrices.

        Removes every crosstalk transform currently in the signal chain and
        replaces them with one :class:`CrosstalkMatrix`. Other signal-chain
        transforms (``Gain``, ``Delay``, and user-defined subclasses) are
        preserved in order.

        Parameters
        ----------
        beta : Any
            ``[n, n]`` amplitude matrix. ``beta[i, j]`` is emitted as
            the leakage amplitude from source ``labels[j]`` onto victim
            ``labels[i]``. Diagonal entries are ignored (self-coupling
            belongs to the drive itself, not to a crosstalk edge).
        theta : Any, optional
            ``[n, n]`` phase matrix (radians). Defaults to zeros.
        delay : Any, optional
            ``[n, n]`` delay matrix (ns). Defaults to zeros.
        labels : tuple[str, ...] | list[str] | None, optional
            Axis ordering. Defaults to wiring order
            (``self.lines``). Must match ``beta.shape[0]``.

        Notes
        -----
        Traced JAX entries flow unchanged into :class:`CrosstalkMatrix` and
        therefore into the signal-program IR. No concretization occurs.
        """
        order = tuple(line.label for line in self._lines) if labels is None else tuple(labels)
        n = len(order)
        # beta is required, so it is always shape-checked; theta and delay are
        # optional and validated only when provided.
        for name, matrix in (("beta", beta), ("theta", theta), ("delay", delay)):
            if matrix is None and name != "beta":
                continue
            if getattr(matrix, "shape", None) != (n, n):
                raise ValueError(
                    f"{name} matrix shape {getattr(matrix, 'shape', None)} does not match {n} drive lines"
                )

        any_traced = any(_is_traced(matrix) for matrix in (beta, theta, delay) if matrix is not None)
        xp = _select_array_module(any_traced)
        matrix_beta = xp.asarray(beta, dtype=float)
        matrix_theta = xp.zeros((n, n), dtype=float) if theta is None else xp.asarray(theta, dtype=float)
        matrix_delay = xp.zeros((n, n), dtype=float) if delay is None else xp.asarray(delay, dtype=float)
        for i in range(n):
            matrix_beta = _setitem(matrix_beta, (i, i), xp.asarray(1.0, dtype=float))
            matrix_theta = _setitem(matrix_theta, (i, i), xp.asarray(0.0, dtype=float))
            matrix_delay = _setitem(matrix_delay, (i, i), xp.asarray(0.0, dtype=float))

        kept = [
            transform for transform in self._signal_chain
            if not isinstance(transform, (Crosstalk, CrosstalkMatrix))
        ]
        self._signal_chain = kept + [
            CrosstalkMatrix(order, matrix_beta, matrix_theta, matrix_delay)
        ]

    def copy(self, device_map: dict[str, Any], coupling_map: dict[str, Any] | None = None) -> "ControlEquipment":
        """Return a structural copy with drive lines rebound to *device_map* / *coupling_map*.

        Edge lines (``target_kind == "edge"``) rebind via *coupling_map*,
        keyed by coupling label; device lines rebind via *device_map* as
        before.
        """
        copied_lines = []
        for line in self._lines:
            if line.target_kind == "edge":
                if coupling_map is None or line.target_label not in coupling_map:
                    raise KeyError(
                        f"No coupling '{line.target_label}' in the target map for edge line '{line.label}'."
                    )
                copied_lines.append(line.copy(target=coupling_map[line.target_label]))
            else:
                copied_lines.append(
                    line.copy(target=None if line.device_label is None else device_map[line.device_label])
                )
        return type(self)(lines=copied_lines, signal_chain=list(self._signal_chain) or None)

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-safe dictionary."""
        data: dict[str, Any] = {"lines": [line.to_dict() for line in self._lines]}
        if self._signal_chain:
            data["signal_chain"] = [transform.to_dict() for transform in self._signal_chain]
        return data

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        dev_map: dict[str, Any],
        coupling_map: dict[str, Any] | None = None,
    ) -> "ControlEquipment":
        """Reconstruct from :meth:`to_dict` output, rebinding drives via *dev_map* / *coupling_map*.

        Each line's ``target_label`` is resolved against *dev_map* first,
        then *coupling_map* — device and coupling labels are disjoint by
        Chip construction, so at most one map holds the label.
        """
        lines: list[BaseDrive] = []
        for line_dict in d.get("lines", []):
            target_label = line_dict.get("target_label")
            if target_label is None:
                target = None
            elif target_label in dev_map:
                target = dev_map[target_label]
            elif coupling_map is not None and target_label in coupling_map:
                target = coupling_map[target_label]
            else:
                raise KeyError(f"No device or coupling named '{target_label}' in the target maps.")
            lines.append(BaseDrive.from_dict(line_dict, target=target))

        signal_chain = [SignalTransform.from_dict(td) for td in d.get("signal_chain", [])]
        return cls(lines=lines, signal_chain=signal_chain or None)
