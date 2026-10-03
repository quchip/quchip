"""Ownership and cache keys for authored numerical values."""

from __future__ import annotations

import copy
from typing import Any, Callable
from dataclasses import dataclass, field

import jax
import jax.tree_util as jtu
import numpy as np
from jax.core import Tracer


def copy_value(value: Any, *, readonly: bool = False) -> Any:
    """Copy mutable values without detaching native JAX arrays or tracers."""
    memo: dict[int, Any] = {}

    def capture(item: Any) -> Any:
        if id(item) in memo:
            return memo[id(item)]
        if isinstance(item, (jax.Array, Tracer)):
            memo[id(item)] = item
            return item
        if isinstance(item, np.ndarray):
            copied = copy.deepcopy(item, memo)
            copied.flags.writeable = not readonly
            return copied
        if isinstance(item, dict):
            copied_dict: dict[Any, Any] = {}
            memo[id(item)] = copied_dict
            copied_dict.update((key, capture(value)) for key, value in item.items())
            return copied_dict
        if isinstance(item, list):
            copied_list: list[Any] = []
            memo[id(item)] = copied_list
            copied_list.extend(capture(value) for value in item)
            return copied_list
        # Seed deepcopy's memo from registered trees, including immutable JAX
        # leaves inside native backend objects and extension payloads.
        for leaf in jtu.tree_leaves(item, is_leaf=lambda child: child is not item):
            if leaf is not item:
                memo[id(leaf)] = capture(leaf)
        return copy.deepcopy(item, memo)

    return capture(value)


@dataclass(frozen=True)
class TracedKey:
    """Cache-key stand-in for a JAX tracer: its identity within the trace that owns it."""

    ident: int


def value_fingerprint(value: Any, *, traced: bool = False) -> Any:
    """Hash supported value contents; reject opaque mutable payloads.

    A tracer is rejected unless ``traced`` is true, when it keys by identity
    (:class:`TracedKey`). Such keys are valid only inside the trace that owns
    the tracer, so their cache entries must be scoped with :func:`scoped_entry`.
    """
    if isinstance(value, Tracer):
        if traced:
            return TracedKey(id(value))
        raise ValueError("Traced values cannot key an eager calculation cache.")
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex, type)):
        return value
    if isinstance(value, (np.ndarray, np.generic, jax.Array)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("Object arrays cannot key a calculation cache.")
        return array.shape, array.dtype.str, array.tobytes()
    if isinstance(value, dict):
        return tuple((value_fingerprint(key, traced=traced), value_fingerprint(item, traced=traced))
                     for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return type(value), tuple(value_fingerprint(item, traced=traced) for item in value)
    leaves, tree = jtu.tree_flatten(value)
    if len(leaves) == 1 and leaves[0] is value:
        raise ValueError(f"Opaque {type(value).__name__} cannot key a calculation cache.")

    def structure(node: Any) -> Any:
        data = node.node_data()
        return value_fingerprint(data, traced=traced), tuple(structure(child) for child in node.children())

    return structure(tree), tuple(value_fingerprint(leaf, traced=traced) for leaf in leaves)


def scoped_entry(key: Any, value: Any, *, traced: bool) -> tuple[Any, Any, Any]:
    """A cache entry ``(key, scope, value)``.

    An entry whose value is traced, or whose key holds a :class:`TracedKey`,
    records the current JAX trace and is reused only inside it, never in a
    nested or later trace. Other entries are valid everywhere.
    """
    scoped = traced or any(isinstance(leaf, TracedKey) for leaf in jtu.tree_leaves(key))
    return key, jax.core.get_opaque_trace_state() if scoped else None, value


def scoped_hit(entry: tuple[Any, Any, Any] | None, key: Any) -> bool:
    """Whether *entry* holds *key* and, when scoped, belongs to the current trace."""
    return (entry is not None and entry[0] == key
            and (entry[1] is None or entry[1] == jax.core.get_opaque_trace_state()))


@dataclass(eq=False)
class DeferredValue:
    """Evaluate captured inputs on demand; cache only concrete values."""

    evaluate: Callable[[], Any] = field(repr=False)
    _value: Any = field(default=None, init=False, repr=False)

    def __call__(self) -> Any:
        from quchip.utils.jax_utils import contains_tracer

        if self._value is None:
            value = self.evaluate()
            if not contains_tracer(value):
                self._value = value
            return value
        return self._value
