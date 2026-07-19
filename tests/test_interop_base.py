"""Registry and dispatch contract of the interop authoring surface."""
from __future__ import annotations

import pytest

from quchip.interop.base import ModelMapping, import_object, export_object, source_key


class _FakeSource:  # stands in for a third-party class; source_key -> "tests.<qualname>"
    pass


class _FakeSub(_FakeSource):
    pass


class _FakeDevice:
    pass


def test_source_key_uses_top_module_and_qualname():
    """source_key joins a type's top-level module with its qualified name."""
    assert source_key(_FakeSource).endswith("._FakeSource")
    assert source_key(_FakeSource).split(".")[0] == "tests"


def test_import_dispatch_walks_mro_and_no_mapping_error_names_authoring_path():
    """import_object dispatches through a base class's mapping and names the authoring skeleton when none matches."""
    class FakeMapping(ModelMapping):
        source = source_key(_FakeSource)

        def import_model(self, obj, **opts):
            return ("imported", type(obj).__name__)

    assert import_object(_FakeSource()) == ("imported", "_FakeSource")
    assert import_object(_FakeSub()) == ("imported", "_FakeSub")  # MRO walk
    with pytest.raises(LookupError, match="ModelMapping"):
        import_object(object())


def test_export_requires_target_and_rejects_duplicates():
    """export_model overrides require a target, and duplicate (library, target) pairs raise at definition."""
    class ExpMapping(ModelMapping):
        library = "fakelib"
        target = _FakeDevice

        def export_model(self, device, **opts):
            return "exported"

    assert export_object(_FakeDevice(), "fakelib") == "exported"
    with pytest.raises(LookupError):
        export_object(object(), "fakelib")
    with pytest.raises(TypeError):  # duplicate (library, target)
        class ExpMapping2(ModelMapping):
            library = "fakelib"
            target = _FakeDevice

            def export_model(self, device, **opts):
                return "boom"
    with pytest.raises(TypeError):  # export override without target
        class NoTarget(ModelMapping):
            library = "fakelib2"

            def export_model(self, device, **opts):
                return None


def test_export_override_without_library_raises():
    """export_model with a target but no resolvable library raises at definition."""
    with pytest.raises(TypeError, match="library"):
        class NoLibrary(ModelMapping):
            target = _FakeDevice  # target set, but no source to default library from

            def export_model(self, device, **opts):
                return None


def test_duplicate_source_raises():
    """A second subclass reusing the same source key raises at definition."""
    class FirstSource(ModelMapping):
        source = "fakemod.FakeUniqueSource"

        def import_model(self, obj, **opts):
            return "first"

    with pytest.raises(TypeError, match="source"):
        class SecondSource(ModelMapping):
            source = "fakemod.FakeUniqueSource"

            def import_model(self, obj, **opts):
                return "second"


def test_invalid_export_declaration_leaves_no_ghost_import_registration():
    """A class with a valid source but an invalid export declaration registers neither side."""
    from quchip.interop.base import registered_mappings

    ghost_source = "fakemod.GhostSource"
    with pytest.raises(TypeError, match="target"):
        # source alone would register on the import side if validated non-atomically.
        class GhostMapping(ModelMapping):
            source = "fakemod.GhostSource"

            def export_model(self, device, **opts):
                return None  # no 'target' set: export declaration is invalid

    assert ghost_source not in registered_mappings()
