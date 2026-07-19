"""Tests for the Registrable subclass-registry mixin."""

from __future__ import annotations

import abc

import pytest

from quchip.utils.registry import Registrable


def test_registry_root_owns_a_fresh_registry_excluding_itself():
    """A root's own type key never appears in its own registry, even once populated."""

    class Root(Registrable, registry_root=True):
        pass

    class Concrete(Root):
        pass

    assert Root._type_key() not in Root._registry
    assert Concrete._type_key() in Root._registry


def test_abstract_intermediate_subclass_is_not_registered():
    """A subclass that still carries an unimplemented abstract method is skipped."""

    class Root(Registrable, registry_root=True):
        pass

    class AbstractMid(Root, abc.ABC):
        @abc.abstractmethod
        def value(self) -> int: ...

    assert AbstractMid._type_key() not in Root._registry


def test_concrete_subclass_registers_under_module_qualified_name():
    """A concrete subclass registers under ``f"{module}.{qualname}"``."""

    class Root(Registrable, registry_root=True):
        pass

    class Concrete(Root):
        pass

    assert Root._registry[Concrete._type_key()] is Concrete
    assert Concrete._type_key() == f"{Concrete.__module__}.{Concrete.__qualname__}"


def test_from_dict_on_root_dispatches_to_concrete_class():
    """``Root.from_dict`` looks up and reconstructs via the concrete class named by ``type``."""

    class Root(Registrable, registry_root=True):
        pass

    class Concrete(Root):
        pass

    instance = Root.from_dict({"type": Concrete._type_key()})
    assert isinstance(instance, Concrete)


def test_from_dict_on_root_raises_with_sorted_known_types_on_unknown_key():
    """An unknown ``type`` key raises ValueError listing the registered types in sorted order."""

    class Root(Registrable, registry_root=True):
        pass

    class Zeta(Root):
        pass

    class Alpha(Root):
        pass

    with pytest.raises(ValueError) as exc_info:
        Root.from_dict({"type": "nonexistent.Type"})

    message = str(exc_info.value)
    assert message.index(Alpha._type_key()) < message.index(Zeta._type_key())
