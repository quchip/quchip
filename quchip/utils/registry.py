"""Subclass registration and serialization dispatch.

The declaration ``class BaseThing(Registrable, registry_root=True)`` makes a
registry for that family. Concrete subclasses register under their qualified
names. The registry excludes registry roots and abstract classes.

``to_dict`` supplies the ``type`` tag. A registry root's ``from_dict`` finds
the concrete class and passes it the payload and all extra arguments. Concrete
classes implement ``from_dict`` or ``_from_dict_payload``. The default payload
constructor is ``cls()``.
"""

from __future__ import annotations

from typing import Any, ClassVar


def _is_abstract(cls: type) -> bool:
    """Check for abstract methods before ``ABCMeta`` sets ``__abstractmethods__``.

    Registration runs inside ``__init_subclass__``, before that attribute exists.
    Inspect the most-derived definition of each abstract name in the MRO.
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
    """Mixin that owns a subclass registry and the shared (de)serialization contract."""

    #: Per-root mapping ``fully-qualified-name -> concrete subclass``. Installed
    #: fresh on each registry root; inherited (shared) by every subclass below it.
    _registry: ClassVar[dict[str, type["Registrable"]]]
    #: The class that declared the registry (``registry_root=True``). Used by
    #: :meth:`from_dict` to tell the dispatching root from a concrete leaf.
    _registry_root: ClassVar[type["Registrable"]]

    def __init_subclass__(cls, *, registry_root: bool = False, **kwargs: Any) -> None:
        """Register *cls*, or start a new registry when *registry_root* is set.

        ``registry_root=True`` installs a new :attr:`_registry` on *cls* and
        excludes *cls* from that registry. Otherwise, concrete subclasses
        register under their fully qualified name (:meth:`_type_key`). Abstract
        subclasses are skipped.
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
        """Serialize the type tag; subclasses add their own fields."""
        return {"type": type(self)._type_key()}

    @classmethod
    def from_dict(cls, data: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        r"""Reconstruct the object from the output of :meth:`to_dict`.

        On the registry root, dispatch to the concrete subclass that
        ``data["type"]`` names, and send ``*args`` / ``**kwargs`` to it. On a
        concrete subclass, use :meth:`_from_dict_payload`. Concrete subclasses
        that carry a payload override this method directly.

        Parameters
        ----------
        data : dict
            Serialized declaration that contains the fully qualified type key.
        *args : object
            Additional positional arguments for reconstruction, for example coupling endpoints.
        **kwargs : object
            Keyword arguments for reconstruction that the registered class accepts.
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
