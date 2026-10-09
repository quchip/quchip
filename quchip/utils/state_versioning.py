"""Track public attribute assignments to invalidate device and coupling caches.

After construction, tracked assignments increment ``state_version``. Private
attributes and ``_untracked_names`` are excluded. The counter tracks writes,
not in-place edits to mutable attribute values.

The outermost ``__init__`` turns on tracking once.
"""
# Classes that synthesize an initializer after `__init_subclass__` must apply
# `_wrap_init_for_finish` themselves. JAX pytree unflattening bypasses
# initialization and installs the tracking state directly.

from __future__ import annotations

import functools
from typing import Any, ClassVar


def _wrap_init_for_finish(cls: type) -> None:
    """Wrap the class's own initializer to enable tracking after its outermost call.

    Does nothing if the initializer is absent or already wrapped. The identity
    check prevents nested ``super().__init__`` calls from finishing early.
    """
    init = cls.__dict__.get("__init__")
    if init is None or getattr(init, "_sv_wrapped", False):
        return

    @functools.wraps(init)
    def _wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        if type(self).__init__ is _wrapped:
            self._finish_init()

    _wrapped._sv_wrapped = True  # type: ignore[attr-defined]
    cls.__init__ = _wrapped  # type: ignore[misc]


class StateVersioned:
    """Mixin: a monotone ``state_version`` that increments on tracked public mutations."""

    #: Public attribute names that must NOT bump ``_state_version`` when set.
    #: Subclasses (BaseDevice, BaseCoupling) extend this with their structural /
    #: identity attributes. Private/dunder names are always excluded separately.
    _untracked_names: ClassVar[frozenset[str]] = frozenset()

    #: Engine-visible cache-invalidation counter. Class-level default ``0``; the
    #: bump installs a per-instance value via ``object.__setattr__``.
    _state_version: int = 0
    #: Whether mutation tracking is live. Flipped on by :meth:`_finish_init`
    #: after the outermost ``__init__`` returns.
    _tracking_enabled: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap *cls*'s own ``__init__`` so that :meth:`_finish_init` runs exactly once."""
        super().__init_subclass__(**kwargs)
        _wrap_init_for_finish(cls)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set *name*, then run the cache-invalidation hook and increment ``state_version`` for tracked writes.
        """
        # The increment runs in a `finally` block, so if the `_on_attr_set` hook
        # raises an error, the changed attribute cannot stay paired with a stale
        # `state_version`.
        # ``object.__setattr__`` is used for the bump so it doesn't itself
        # retrigger the hook.
        object.__setattr__(self, name, value)
        try:
            self._on_attr_set(name)
        finally:
            if (
                not name.startswith("_")
                and name not in type(self)._untracked_names
                and getattr(self, "_tracking_enabled", False)
            ):
                object.__setattr__(self, "_state_version", self._state_version + 1)

    def _on_attr_set(self, name: str) -> None:
        """Handle an attribute set; default no-op cache-invalidation hook."""

    def _finish_init(self) -> None:
        """Enable mutation tracking; fired automatically once after construction."""
        object.__setattr__(self, "_tracking_enabled", True)

    @property
    def state_version(self) -> int:
        """Return the monotone counter that increments on each tracked mutation of a public parameter."""
        return self._state_version
