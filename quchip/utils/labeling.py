"""Component labels and lookup by object or label.

A component without an explicit label gets ``"{prefix}_{n}"``. Each prefix has
a process-wide counter starting at zero. Labels identify components in a chip.
:func:`resolve_label` accepts the component or its label. Use
:func:`reset_label_counters` to get deterministic labels in tests.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

_label_counters: dict[str, int] = {}


def auto_label(prefix: str) -> str:
    r"""Return the next ``"{prefix}_{n}"`` label, where each prefix starts at zero.

    Parameters
    ----------
    prefix : str
        Prefix of the component type. Each prefix has its own process-wide counter.
    """
    idx = _label_counters.get(prefix, 0)
    _label_counters[prefix] = idx + 1
    return f"{prefix}_{idx}"


def reset_label_counters() -> None:
    """Reset the process-wide label counters, usually in test fixtures."""
    _label_counters.clear()


def resolve_label(obj: str | Any) -> str:
    r"""Return a string label from a string or a labeled component.

    Raise ``TypeError`` if the object has no usable label.

    Parameters
    ----------
    obj : str or object
        A label string, returned unchanged, or an object with a non-None label attribute.
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
    """Merge object-or-label keys and keyword arguments into a dict with label keys.

    Duplicate labels raise ``ValueError``, including a label supplied through
    both inputs. Values pass through unchanged, and callers validate their types
    and bounds.
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
    """Build a bare-state tuple in ``device_labels`` order from a partial mapping.

    Keys can be devices or labels. Unspecified devices get the default Fock index
    zero. Unknown or duplicate labels raise ``ValueError``. Callers validate
    values.
    """
    merged = merge_labeled_values(mapping, kwargs)

    unknown = sorted(set(merged) - set(device_labels))
    if unknown:
        raise ValueError(f"Unknown device labels {unknown}. Available labels: {list(device_labels)}")

    return tuple(merged.get(label, 0) for label in device_labels)


class LabelKeyedDict(dict):
    """Result mapping with label keys that also accepts component objects.

    Tuple keys with two elements match in each order. Iteration and serialization show
    the stored keys.
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
        """Find *key*, and use the reversed 2-tuple key if the forward order does not match."""
        resolved = self._canonical(key)
        if not super().__contains__(resolved) and isinstance(resolved, tuple) and len(resolved) == 2:
            reordered = resolved[::-1]
            if super().__contains__(reordered):
                resolved = reordered
        return super().__getitem__(resolved)

    def __contains__(self, key: Any) -> bool:
        """Report membership, and match the reversed 2-tuple key if the forward order does not match."""
        resolved = self._canonical(key)
        if super().__contains__(resolved):
            return True
        return isinstance(resolved, tuple) and len(resolved) == 2 and super().__contains__(resolved[::-1])

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if *key* is absent (through :meth:`__getitem__`)."""
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
    """Return the top ``n`` bare-basis probabilities of a dressed eigenvector.

    It pairs the squared amplitudes from column ``dressed_idx`` with bare
    labels in descending order. It requires a concrete eigenvector matrix.
    """
    probs = np.asarray(np.abs(eigenvector_matrix[:, dressed_idx]) ** 2, dtype=float)
    order = np.argsort(probs)[::-1][:n]
    return {bare_labels[idx]: float(probs[idx]) for idx in order}
