"""Tracer-detection and array-namespace dispatch helpers shared across quchip.

JAX is a required dependency of quchip (see ``pyproject.toml``), so these
helpers import it unconditionally.
"""

from __future__ import annotations

from typing import Any

from jax.core import Tracer
from jax.tree_util import tree_leaves
import jax.numpy as jnp
import numpy as np


def contains_tracer(pytree: Any) -> bool:
    """Return ``True`` if *pytree* has any :class:`jax.core.Tracer` leaf.

    Uses :func:`jax.tree_util.tree_leaves` so nested tuples/lists/dicts
    and custom registered pytrees are traversed correctly.

    Covers all built-in JAX transforms: ``jit``, ``vmap``, ``grad``,
    ``linearize``, ``pmap``, and their composition all produce
    subclasses of :class:`jax.core.Tracer`.
    """
    for leaf in tree_leaves(pytree):
        if isinstance(leaf, Tracer):
            return True
    return False


def array_namespace(array: Any) -> Any:
    """Return the array's namespace (``jax.numpy`` or ``numpy``).

    Uses ``__array_namespace__`` when available, falling back to NumPy.
    Single source of truth so dispatch logic stays consistent across
    engine, control, and analysis layers.
    """
    namespace = getattr(array, "__array_namespace__", None)
    if callable(namespace):
        return namespace()
    return np


def is_jax_namespace(xp: Any) -> bool:
    """Return ``True`` if *xp* is ``jax.numpy`` (by module name)."""
    return getattr(xp, "__name__", "").startswith("jax")


def is_jax_array(array: Any) -> bool:
    """Return ``True`` if *array* uses ``jax.numpy`` as its namespace."""
    return is_jax_namespace(array_namespace(array))


def select_array_module(prefer_jax: bool) -> Any:
    """Return ``jax.numpy`` if *prefer_jax*, else NumPy."""
    return jnp if prefer_jax else np


def maybe_concrete_scalar(value: Any) -> float | None:
    """Return a Python ``float`` if *value* is a concrete, real-valued 0-d scalar, else ``None``.

    Used throughout device, drive, and envelope constructors to inspect a
    parameter that *might* be a JAX tracer. Returns ``None`` — not a
    ``float`` or a raised error — for a JAX tracer, a non-scalar, or a
    complex or otherwise non-float-convertible payload. Callers therefore
    treat complex-valued input the same as traced input: the concrete
    validation check is skipped rather than run against a lossy cast.
    """
    if value is None:
        return None
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.ndim != 0:
        return None
    try:
        return float(array)
    except (TypeError, ValueError):
        # 0-d object arrays (e.g. an envelope or other non-numeric payload
        # passed as a declared parameter) are not concrete scalars.
        return None
