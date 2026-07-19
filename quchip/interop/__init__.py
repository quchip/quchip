"""Third-party model interoperability: ModelMapping authoring surface and registry.

This package is the library-agnostic foundation for converting circuit-QED
models to and from quchip devices. Concrete mappings for specific libraries
(e.g. scqubits) live in sibling modules and import both sides; this module
never does.
"""

from __future__ import annotations

from quchip.interop.base import ModelMapping, export_object, import_object, source_key
from quchip.interop.eigenbasis import EigenbasisDevice

__all__ = ["EigenbasisDevice", "ModelMapping", "export_object", "import_object", "source_key"]
