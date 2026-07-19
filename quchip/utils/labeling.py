"""Shared auto-labeling and label resolution for quchip components.

quchip devices, couplings, and drives never require users to invent unique
identifiers. Each component class declares a short ``_type_prefix`` (e.g.
``"duffing"`` for :class:`~quchip.devices.transmon.duffing.DuffingTransmon`,
``"charge"`` for :class:`~quchip.control.drive.ChargeDrive`,
``"capacitive"`` for :class:`~quchip.chip.couplings.Capacitive`), and when
the user constructs one without passing an explicit ``label`` the component
calls :func:`auto_label` to obtain ``"{prefix}_{n}"`` — where ``n`` is a
per-prefix counter that increments on each call.

Labels are strings that uniquely identify a component inside a
:class:`~quchip.chip.chip.Chip`. They are used as:

- Keys in :class:`~quchip.control.sequence.QuantumSequence` drive schedules.
- Keys in :class:`~quchip.control.signal.Crosstalk` matrices.
- Keys in expectation-value dictionaries.
- The user-facing identifier in plots, ``__repr__``, and error messages.

Any public API that accepts a component reference accepts either the
string label *or* the component object itself. :func:`resolve_label` is
the single helper that normalizes both into a string — call it wherever
you'd otherwise write a manual ``isinstance(..., str)`` branch.

Examples
--------
>>> from quchip.utils.labeling import auto_label, reset_label_counters, resolve_label
>>> reset_label_counters()
>>> auto_label("duffing")
'duffing_0'
>>> auto_label("duffing")
'duffing_1'
>>> auto_label("capacitive")
'capacitive_0'
>>> class FakeDevice:
...     label = "duffing_0"
>>> resolve_label(FakeDevice())
'duffing_0'
>>> resolve_label("duffing_0")
'duffing_0'
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

_label_counters: dict[str, int] = {}


def auto_label(prefix: str) -> str:
    """Return the next ``"{prefix}_{n}"`` label and advance the per-prefix counter.

    Parameters
    ----------
    prefix
        The component's ``_type_prefix`` (e.g. ``"duffing"``, ``"charge"``).

    Returns
    -------
    str
        ``f"{prefix}_{n}"`` where ``n`` starts at ``0`` and increments on
        each call with the same prefix.
    """
    idx = _label_counters.get(prefix, 0)
    _label_counters[prefix] = idx + 1
    return f"{prefix}_{idx}"


def reset_label_counters() -> None:
    """Reset every auto-label counter to zero.

    Intended for test fixtures — tests rely on deterministic labels, and the
    module-level counter dict is otherwise process-global.
    """
    _label_counters.clear()


def resolve_label(obj: str | Any) -> str:
    """Return a label string from either a string or a labeled component.

    Used wherever quchip's public API accepts *"component or its label"*
    interchangeably.

    Parameters
    ----------
    obj
        Either a label string, or any object exposing a non-``None``
        ``.label`` attribute (devices, drives, couplings all qualify).

    Raises
    ------
    TypeError
        If *obj* is neither a string nor an object with a usable ``.label``.
    """
    if isinstance(obj, str):
        return obj
    label = getattr(obj, "label", None)
    if label is None:
        raise TypeError(
            f"Expected a label string or an object with .label, got {type(obj).__name__}: {obj!r}"
        )
    return str(label)


def merge_labeled_values(
    mapping: Mapping[Any, Any] | None,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a ``{device-or-label: value}`` mapping + kwargs into one ``{label: value}`` dict.

    Keys in *mapping* are resolved through :func:`resolve_label`, so device
    objects and label strings are interchangeable. Two mapping keys that
    resolve to the same label, or a label supplied through both *mapping*
    and *kwargs*, are duplicates and raise :class:`ValueError`. Values are
    placed verbatim (no ``int`` cast) — callers that need type or bound
    validation layer it on top.

    The single resolve-and-dedup step shared by the bare-tuple builder
    (:func:`bare_label_from_mapping`) and the chip state-factory normalizer
    (:func:`quchip.chip.states.normalize_device_state_mapping`), so the rule for
    what counts as a duplicate device specification can never drift between
    spectroscopy sweeps and single-point state construction. *mapping* must be a
    :class:`~collections.abc.Mapping` (or ``None``); the public state factory
    type-checks user input before delegating here.
    """
    merged: dict[str, Any] = {}
    if mapping is not None:
        for key, value in mapping.items():
            label = resolve_label(key)
            if label in merged:
                raise ValueError(f"Duplicate device specification for '{label}'")
            merged[label] = value
    for label, value in kwargs.items():
        if label in merged:
            raise ValueError(f"Duplicate device specification for '{label}'")
        merged[label] = value
    return merged


def bare_label_from_mapping(
    device_labels: Sequence[str],
    mapping: Mapping[Any, Any] | None,
    kwargs: Mapping[str, Any],
) -> tuple[int, ...]:
    """Merge a ``{device: Fock}`` spec into a full bare-state tuple.

    The single canonical definition of how a partial ``{device: Fock}``
    specification becomes a complete bare-state tuple in *device_labels*
    order: keys are resolved and de-duplicated via
    :func:`merge_labeled_values`, any label outside *device_labels* is
    unknown and raises, and devices not mentioned default to Fock index ``0``.

    Both :class:`~quchip.sweep.SpectrumSweepResult` and the chip-side
    dressed analysis (:mod:`quchip.chip.analysis`) route through here, so
    the spec-to-tuple mapping lives in exactly one place. Values are
    placed verbatim (no ``int`` cast): callers that need type or Fock-bound
    validation layer it on top of the returned tuple.
    """
    merged = merge_labeled_values(mapping, kwargs)

    unknown = sorted(set(merged) - set(device_labels))
    if unknown:
        raise ValueError(f"Unknown device labels {unknown}. Available labels: {list(device_labels)}")

    return tuple(merged.get(label, 0) for label in device_labels)


class LabelKeyedDict(dict):
    """Result mapping keyed by labels (or label tuples) that also accepts the objects themselves.

    Wherever a reduction or analysis result is keyed by a component's label,
    the component object is an equally good key. Tuple keys (survivor pairs)
    additionally match in
    either order — a pair is unordered physics; the stored order is
    bookkeeping. Iteration, ``items()``, and serialization see the plain
    stored keys; only lookup is widened.
    """

    @staticmethod
    def _canonical(key: Any) -> Any:
        """Resolve *key* (or each element of a tuple key) to its label form."""
        try:
            if isinstance(key, tuple):
                return tuple(resolve_label(part) for part in key)
            return resolve_label(key)
        except TypeError:
            return key  # keys that are neither labels nor labeled objects pass through

    def __getitem__(self, key: Any) -> Any:
        """Look up *key*, falling back to a 2-tuple key's reversal when the forward order misses."""
        resolved = self._canonical(key)
        if not super().__contains__(resolved) and isinstance(resolved, tuple) and len(resolved) == 2:
            reordered = resolved[::-1]
            if super().__contains__(reordered):
                resolved = reordered
        return super().__getitem__(resolved)

    def __contains__(self, key: Any) -> bool:
        """Report membership, matching a 2-tuple key's reversal when the forward order misses."""
        resolved = self._canonical(key)
        if super().__contains__(resolved):
            return True
        return isinstance(resolved, tuple) and len(resolved) == 2 and super().__contains__(resolved[::-1])

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent (via :meth:`__getitem__`)."""
        try:
            return self[key]
        except KeyError:
            return default


def top_components(
    eigenvector_matrix: Any,
    bare_labels: Sequence[tuple[int, ...]],
    dressed_idx: int,
    n: int,
) -> dict[tuple[int, ...], float]:
    """Return the leading bare-basis probabilities ``|⟨bare|dressed⟩|²`` of one dressed column.

    Squares the amplitudes of column *dressed_idx* of *eigenvector_matrix*
    (dressed eigenvectors expressed in the bare product basis), sorts them
    descending, and maps the top *n* to their bare labels. This is the
    squared-amplitude / argsort / label-map kernel shared by
    :meth:`quchip.chip.analysis.ChipAnalysis.state_components` and
    :meth:`quchip.sweep.SpectrumSweepResult.state_components_at`; each call
    site keeps only its own dressed-index resolution.

    Operates on a concrete eigenvector matrix — both call sites resolve the
    dressed index off an already-materialized (non-traced) dressing, so the
    NumPy reduction here never sits on a differentiable path.
    """
    probs = np.asarray(np.abs(eigenvector_matrix[:, dressed_idx]) ** 2, dtype=float)
    order = np.argsort(probs)[::-1][:n]
    return {bare_labels[idx]: float(probs[idx]) for idx in order}
