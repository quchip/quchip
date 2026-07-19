"""StateVersioned — shared mutation-tracking / cache-invalidation mixin.

The engine caches assembled Hamiltonians (and dressed-state analyses) keyed
on a per-object ``state_version`` counter. After construction finishes,
every tracked public mutation bumps the counter so stale caches invalidate
deterministically. Both :class:`~quchip.devices.base.BaseDevice` and
:class:`~quchip.chip.coupling_base.BaseCoupling` need exactly this
machinery, so it lives here once instead of being copy-pasted into each
base.

:meth:`_finish_init` flips mutation tracking on and fires exactly once per
instance. :meth:`__init_subclass__` wraps each concrete subclass's own
``__init__``; the wrapper calls :meth:`_finish_init` only when
``type(self).__init__`` resolves to itself — the entry-point init for the
instance's actual type — so ``super().__init__`` chains and explicit
inner-init forwards never re-fire it.

Subclass-authored init machinery that synthesizes ``__init__`` after the
``__init_subclass__`` super-chain runs (the declarative ``DeviceModel``)
must call :func:`_wrap_init_for_finish` on the freshly-installed init
itself; the helper is idempotent, so a repeat call is harmless.

The JAX pytree ``_unflatten`` path bypasses ``__init__`` entirely — it
builds via ``cls.__new__`` and sets ``_tracking_enabled`` directly — and so
bypasses the wrapper as well.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar


def _wrap_init_for_finish(cls: type) -> None:
    """Wrap ``cls.__dict__['__init__']`` so :meth:`_finish_init` fires once.

    No-op when *cls* defines no own ``__init__`` or when its ``__init__`` is
    already wrapped (idempotent). The wrapper runs the original init, then calls
    ``_finish_init`` only when ``type(self).__init__`` resolves to this wrapper
    — the entry-point init for the instance's actual type — so nested
    ``super().__init__`` / explicit inner-init calls never re-fire it.
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
    """Mixin: monotone ``state_version`` bumped on tracked public mutations."""

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
        """Wrap *cls*'s own ``__init__`` so :meth:`_finish_init` fires exactly once."""
        super().__init_subclass__(**kwargs)
        _wrap_init_for_finish(cls)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set *name*, then run the cache-invalidation hook and bump ``state_version`` for tracked writes.

        The bump runs in a ``finally`` block so a raising :meth:`_on_attr_set`
        hook cannot leave the mutated attribute paired with a stale
        ``state_version``.
        """
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
        """Return the monotone counter bumped on every tracked public-parameter mutation."""
        return self._state_version
