"""Library-agnostic registry and dispatch for third-party model mappings.

A :class:`ModelMapping` subclass authors a one-directional or bidirectional
conversion between a third-party circuit-QED object and a quchip device.
Subclasses register at class-definition time, keyed by the third-party type
they import from, the ``(library, quchip type)`` pair they export to, or both.
Concrete mappings (e.g. for scqubits) live in sibling modules that import both
sides.
"""

from __future__ import annotations

from typing import Any, ClassVar

_IMPORT_REGISTRY: dict[str, type["ModelMapping"]] = {}
_EXPORT_REGISTRY: dict[tuple[str, type], type["ModelMapping"]] = {}

_AUTHORING_SKELETON = """
class MyMapping(ModelMapping):
    source = "some_library.SomeClass"  # import support
    target = MyQuchipDevice            # export support
    def import_model(self, obj, **opts): ...
    def export_model(self, device, **opts): ...
""".strip()


def source_key(tp: type) -> str:
    """Return the registry key for third-party type *tp*.

    The key is the type's top-level module name joined with its qualified name,
    e.g. ``"scqubits.Transmon"``. Only the top-level module is used, so a
    mapping registered against a package root matches classes re-exported from
    submodules.

    Parameters
    ----------
    tp : type
        Third-party class to identify.

    Returns
    -------
    str
        Top-level module and qualified class name separated by a dot.
    """
    return f"{tp.__module__.split('.')[0]}.{tp.__qualname__}"


class ModelMapping:
    """Base class for a single third-party <-> quchip device conversion.

    Attributes
    ----------
    source : str or None
        Registry key (see :func:`source_key`) of the third-party type this
        mapping imports from. ``None`` means the mapping supports export
        only.
    target : type or None
        quchip device type this mapping exports to. Required when
        :meth:`export_model` is overridden. ``None`` means import-only.
    library : str or None
        Name of the third-party library this mapping exports for. Defaults to
        ``source.split(".")[0]`` when ``source`` is set. Export-only mappings
        must set it explicitly.

    Subclassing registers the mapping automatically:

    * setting ``source`` registers it for :func:`import_object` under that
      key. A second subclass that reuses the same ``source`` raises
      :class:`TypeError` at class-definition time.
    * overriding :meth:`export_model` registers it for :func:`export_object`
      under ``(library, target)``. An override without ``target``, or without a
      resolvable ``library`` (no ``source`` to default it from), raises
      :class:`TypeError`. A second subclass that reuses the same
      ``(library, target)`` pair raises :class:`TypeError` naming both classes.

    The abstract base itself (``source is None`` and no ``export_model``
    override) registers nothing.
    """

    source: ClassVar[str | None] = None
    target: ClassVar[type | None] = None
    library: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls.library is None and cls.source is not None:
            cls.library = cls.source.split(".")[0]

        # Validate both registrations before mutating either registry: a
        # class that fails export validation (missing target/library, or a
        # duplicate export key) must not leave a stale import-registry entry
        # behind just because the import-side check ran first.
        if cls.source is not None:
            existing_import = _IMPORT_REGISTRY.get(cls.source)
            if existing_import is not None:
                raise TypeError(
                    f"Duplicate ModelMapping.source {cls.source!r}: "
                    f"already registered by {existing_import.__qualname__}, cannot register {cls.__qualname__}."
                )

        export_key: tuple[str, type] | None = None
        if "export_model" in cls.__dict__:
            if cls.target is None:
                raise TypeError(
                    f"{cls.__qualname__} overrides export_model but does not set 'target'. "
                    "Export mappings must declare the quchip device type they produce."
                )
            if cls.library is None:
                raise TypeError(
                    f"{cls.__qualname__} overrides export_model but has no 'library' to register under "
                    "(no 'source' to default it from). Set 'library' explicitly on the mapping."
                )
            export_key = (cls.library, cls.target)
            existing_export = _EXPORT_REGISTRY.get(export_key)
            if existing_export is not None:
                raise TypeError(
                    f"Duplicate ModelMapping export target {export_key!r}: "
                    f"already registered by {existing_export.__qualname__}, cannot register {cls.__qualname__}."
                )

        if cls.source is not None:
            _IMPORT_REGISTRY[cls.source] = cls
        if export_key is not None:
            _EXPORT_REGISTRY[export_key] = cls

    def import_model(self, obj: Any, **opts: Any) -> Any:
        """Convert third-party object *obj* into a quchip device.

        Override to support import. The base implementation raises
        :class:`NotImplementedError`.

        Parameters
        ----------
        obj : object
            Third-party instance to convert.
        **opts
            Mapping-specific keyword options. Subclasses must document supported keys.

        Returns
        -------
        object
            Converted quchip model, as defined by the subclass.
        """
        raise NotImplementedError(f"{type(self).__qualname__} does not support import.")

    def export_model(self, device: Any, **opts: Any) -> Any:
        """Convert quchip *device* into a third-party object.

        Override to support export, and also set ``target``. The base
        implementation raises :class:`NotImplementedError`.

        Parameters
        ----------
        device : object
            quchip device to convert.
        **opts
            Mapping-specific keyword options. Subclasses must document supported keys.

        Returns
        -------
        object
            Converted third-party model, as defined by the subclass.
        """
        raise NotImplementedError(f"{type(self).__qualname__} does not support export.")


def import_object(obj: Any, **opts: Any) -> Any:
    """Import third-party *obj* into a quchip device through a registered mapping.

    Walks ``type(obj).__mro__`` and dispatches to the first
    :class:`ModelMapping` registered under that class's :func:`source_key`.

    Raises
    ------
    LookupError
        No mapping is registered for ``type(obj)`` or any of its base classes.
        The message names the missing source key and shows a skeleton for
        authoring a new :class:`ModelMapping`.

    Parameters
    ----------
    obj : object
        Third-party object whose type selects the registered mapping.
    **opts
        Keywords forwarded unchanged to the mapping's conversion method, which
        documents supported keys and defaults.

    Returns
    -------
    object
        Converted quchip model.
    """
    for klass in type(obj).__mro__:
        mapping_cls = _IMPORT_REGISTRY.get(source_key(klass))
        if mapping_cls is not None:
            return mapping_cls().import_model(obj, **opts)
    raise LookupError(
        f"No ModelMapping registered for {source_key(type(obj))!r} (or any base class). "
        f"Author one:\n{_AUTHORING_SKELETON}"
    )


def export_object(device: Any, library: str, **opts: Any) -> Any:
    """Export quchip *device* to a third-party object of *library*.

    Walks ``type(device).__mro__`` and dispatches to the first
    :class:`ModelMapping` registered under ``(library, that class)``.

    Raises
    ------
    LookupError
        No mapping is registered for ``(library, type(device))`` or any base
        class. The message lists the device types *library* can export.

    Parameters
    ----------
    device : BaseDevice
        Device whose type selects the registered mapping.
    library : str
        Registered target library, such as ``"scqubits"``.
    **opts
        Keywords forwarded unchanged to the mapping's conversion method, which
        documents supported keys and defaults.

    Returns
    -------
    object
        Converted third-party object.
    """
    for klass in type(device).__mro__:
        mapping_cls = _EXPORT_REGISTRY.get((library, klass))
        if mapping_cls is not None:
            return mapping_cls().export_model(device, **opts)
    exportable = sorted(tp.__qualname__ for (lib, tp) in _EXPORT_REGISTRY if lib == library)
    raise LookupError(
        f"No ModelMapping registered to export {type(device).__qualname__!r} to library {library!r}. "
        f"Exportable types for {library!r}: {exportable}"
    )


def registered_mappings() -> dict[str, type[ModelMapping]]:
    """Return a copy of the import registry, keyed by source key.

    Changing the returned dict does not affect the registration state.
    """
    return dict(_IMPORT_REGISTRY)
