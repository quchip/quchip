"""Shared subclass-registry mixin for serializable quchip components.

A :class:`Registrable` subtree maintains a single registry mapping each
concrete subclass's fully-qualified name to the class object, populated
automatically at subclass-definition time. The mixin owns the *shared*
serialization contract — the ``{"type": ...}`` stamp and the registry-root
``from_dict`` dispatch — so devices, couplings, drives, envelopes, and
signal transforms share one registration and dispatch rule instead of each
component family hand-rolling its own registry dict, ``__init_subclass__``
registration, and base-vs-leaf dispatch.

Declaring a registry root
-------------------------
A *registry root* is declared with the ``registry_root=True`` class
keyword::

    class BaseThing(Registrable, registry_root=True):
        ...

The root owns a fresh registry dict and is itself excluded from
registration. Every concrete subclass below it is registered under
``f"{cls.__module__}.{cls.__qualname__}"``. Abstract subclasses — those
that still carry unimplemented abstract methods — are skipped as well:
they can never be instantiated, hence never serialized, so registering
them would only add dead entries.

Serialization contract
-----------------------
* :meth:`to_dict` stamps ``{"type": cls._type_key()}``. Subclasses that
  carry extra payload override :meth:`to_dict`, call ``super().to_dict()``,
  and add their own fields — preserving each type's payload exactly.
* :meth:`from_dict`, when invoked on the registry root, looks the concrete
  class up by ``data["type"]`` and delegates to *its* ``from_dict``,
  forwarding any extra positional / keyword arguments (a coupling's two
  endpoints, a drive's target, …). On a concrete subclass it reconstructs
  via :meth:`_from_dict_payload`, whose default is the parameter-less
  ``cls()``. Subclasses needing real reconstruction either override
  ``from_dict`` (devices, couplings, envelopes — payload-carrying) or
  override ``_from_dict_payload`` (drives — shared target/label/rwa
  reconstruction). The parameter-less default covers the signal transforms
  that take no constructor arguments.
"""

from __future__ import annotations

from typing import Any, ClassVar


def _is_abstract(cls: type) -> bool:
    """Return whether *cls* still carries unimplemented abstract methods.

    :attr:`type.__abstractmethods__` is computed by ``ABCMeta.__new__`` only
    *after* ``__init_subclass__`` returns, so it is not yet available when the
    registry decides whether to register a freshly-defined subclass. This
    reproduces that computation directly: for every name marked abstract
    anywhere in the MRO, find its most-derived definition; if that definition
    is still abstract, the class cannot be instantiated.
    """
    abstract_names = {
        name
        for klass in cls.__mro__
        for name, value in vars(klass).items()
        if getattr(value, "__isabstractmethod__", False)
    }
    for name in abstract_names:
        for klass in cls.__mro__:
            if name in vars(klass):
                if getattr(vars(klass)[name], "__isabstractmethod__", False):
                    return True
                break
    return False


class Registrable:
    """Mixin owning a subclass registry and the shared (de)serialization contract."""

    #: Per-root mapping ``fully-qualified-name -> concrete subclass``. Installed
    #: fresh on each registry root; inherited (shared) by every subclass below it.
    _registry: ClassVar[dict[str, type["Registrable"]]]
    #: The class that declared the registry (``registry_root=True``). Used by
    #: :meth:`from_dict` to tell the dispatching root from a concrete leaf.
    _registry_root: ClassVar[type["Registrable"]]

    def __init_subclass__(cls, *, registry_root: bool = False, **kwargs: Any) -> None:
        """Register *cls*, or seed a fresh registry when *registry_root* is set.

        ``registry_root=True`` installs a fresh :attr:`_registry` on *cls*
        and excludes *cls* itself from it. Otherwise, concrete subclasses
        register under their fully-qualified name (:meth:`_type_key`);
        abstract subclasses are skipped.
        """
        super().__init_subclass__(**kwargs)
        if registry_root:
            cls._registry = {}
            cls._registry_root = cls
            return
        # Abstract subclasses can't be instantiated → never serialized → skip.
        if _is_abstract(cls):
            return
        cls._registry[cls._type_key()] = cls

    @classmethod
    def _type_key(cls) -> str:
        """Return the fully-qualified registry key for *cls* (its serialization ``type``)."""
        return f"{cls.__module__}.{cls.__qualname__}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the type tag; subclasses extend with their own fields."""
        return {"type": type(self)._type_key()}

    @classmethod
    def from_dict(cls, data: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Reconstruct from :meth:`to_dict` output.

        On the registry root, dispatch to the concrete subclass named by
        ``data["type"]`` (forwarding ``*args`` / ``**kwargs``). On a concrete
        subclass, defer to :meth:`_from_dict_payload`. Concrete subclasses
        that carry payload override this method directly.
        """
        if cls is cls._registry_root:
            type_key = str(data["type"])
            try:
                target_cls = cls._registry[type_key]
            except KeyError:
                raise ValueError(
                    f"Unknown {cls.__name__} type {type_key!r}. "
                    f"Registered types: {sorted(cls._registry)}"
                ) from None
            return target_cls.from_dict(data, *args, **kwargs)
        return cls._from_dict_payload(data, *args, **kwargs)

    @classmethod
    def _from_dict_payload(cls, data: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Reconstruct a concrete instance (default: parameter-less ``cls()``)."""
        return cls()
