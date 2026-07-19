"""Freshness gate for the generated device __init__ stubs (tools/gen_device_stubs.py)."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TOOL_PATH = PROJECT_ROOT / "tools" / "gen_device_stubs.py"


def _load_tool():
    """Import tools/gen_device_stubs.py by path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location("gen_device_stubs", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen_device_stubs = _load_tool()


def test_generated_stubs_are_current():
    """The committed __init__ stubs match what tools/gen_device_stubs.py would generate."""
    stale = gen_device_stubs.check_stubs_current()
    assert not stale, f"stale generated stub(s); run `python tools/gen_device_stubs.py`: {stale}"


def test_every_synthesized_device_class_carries_stub_markers():
    """Every DeviceModel subclass with a synthesized __init__ has a generated-stub marker region."""
    classes = gen_device_stubs._synthesized_device_classes()
    assert classes, "no synthesized-init DeviceModel subclasses found; enumeration is broken"
    for cls in classes:
        source = inspect.getsource(cls)
        assert gen_device_stubs.MARKER_START in source, f"{cls.__name__} is missing the generated stub marker"
        assert gen_device_stubs.MARKER_END in source, f"{cls.__name__} is missing the generated stub end marker"
