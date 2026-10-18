"""Helpers for tracer detection and array-namespace dispatch that quchip modules share.

JAX is a required dependency of quchip (see ``pyproject.toml``).
"""
# Because JAX is a required dependency, these helpers import JAX
# unconditionally.

from __future__ import annotations

from typing import Any, Callable

import jax
from jax.core import Tracer
from jax.tree_util import tree_leaves
import jax.numpy as jnp
import numpy as np


def contains_tracer(pytree: Any) -> bool:
    """Return ``True`` if *pytree* has a :class:`jax.core.Tracer` leaf.

    It uses :func:`jax.tree_util.tree_leaves`, so it traverses nested tuples,
    lists, dicts, and custom registered pytrees correctly.

    It covers all built-in JAX transforms, because ``jit``, ``vmap``, ``grad``,
    ``linearize``, ``pmap``, and their composition all make subclasses of
    :class:`jax.core.Tracer`.
    """
    for leaf in tree_leaves(pytree):
        if isinstance(leaf, Tracer):
            return True
    return False


def raise_if(invalid: Any, error: Callable[..., Exception], *values: Any) -> None:
    """Raise ``error(*values)`` when ``invalid`` holds, also inside ``jax.jit``.

    A traced check runs through :func:`jax.experimental.io_callback`, which
    also survives callers that request only a gradient. The values reach
    ``error`` as concrete arrays there. Inside ``jit``, the exception reaches
    the caller as a JAX runtime error with the same message, so only an eager
    check raises the type that ``error`` returns.

    Parameters
    ----------
    invalid : bool or array_like
        Scalar condition, concrete or traced.
    error : callable
        Returns the exception for the concrete ``values``.
    *values : array_like
        Diagnostic values for the exception.
    """
    def check(flag: Any, *concrete: Any) -> None:
        if bool(flag):
            raise error(*concrete)

    if contains_tracer((invalid, values)):
        jax.experimental.io_callback(
            check, None, jax.lax.stop_gradient(invalid), *(jax.lax.stop_gradient(value) for value in values),
        )
    else:
        check(invalid, *values)


def array_namespace(array: Any) -> Any:
    """Return the namespace of the array (``jax.numpy`` or ``numpy``).

    It uses ``__array_namespace__`` when available and falls back to NumPy.
    """
    # As the single source of truth for array-namespace dispatch,
    # array_namespace keeps the dispatch logic the same in the engine, control,
    # and analysis layers.
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
    """Return ``jax.numpy`` if *prefer_jax* is true, else NumPy."""
    return jnp if prefer_jax else np


def concrete_array_module(*values: Any) -> Any:
    """Return NumPy when all values are concrete, and ``jax.numpy`` when a value is traced.

    Concrete arithmetic on the host compiles no XLA programs, whereas each
    eager JAX operation on a new shape compiles one.
    """
    return select_array_module(contains_tracer(values))


def maybe_concrete_scalar(value: Any) -> float | None:
    """Return a Python ``float`` if *value* is a concrete, real-valued 0-d scalar, else ``None``.

    It examines a parameter that can be a JAX tracer. For a tracer, a
    non-scalar, or a complex or other payload that cannot convert to float, it
    returns ``None`` instead of a ``float`` or an error. Callers therefore
    treat complex-valued input like traced input, and skip the concrete
    validation check rather than run it on a lossy cast.
    """
    # Device, drive, and envelope constructors use maybe_concrete_scalar to
    # examine parameters that can be JAX tracers.
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
