"""Backend selection coverage for lazy dynamiqs resolution and error handling."""

from __future__ import annotations

import importlib
import re
import types

import pytest

import quchip.backend as backend_module
from quchip.chip.chip import Chip
from quchip.devices.transmon.duffing import DuffingTransmon


INSTALL_HINT = "DynamiqsBackend requires dynamiqs and JAX. Install with: pip install quchip[dynamiqs]"


def _single_qubit_chip(*, backend: str | None = None) -> Chip:
    """Build a minimal chip for backend-resolution tests."""
    qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    return Chip(devices=[qubit], backend=backend)


def _install_fake_dynamiqs(monkeypatch: pytest.MonkeyPatch) -> type[backend_module.QuTiPBackend]:
    """Patch the lazy dynamiqs import with a concrete fake backend class."""
    real_import_module = importlib.import_module

    class FakeDynamiqsBackend(backend_module.QuTiPBackend):
        """Concrete stand-in used to verify lazy import resolution."""

    def fake_import_module(name: str, package: str | None = None):
        if name == "quchip.backend.dynamiqs":
            return types.SimpleNamespace(DynamiqsBackend=FakeDynamiqsBackend)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    return FakeDynamiqsBackend


def test_qutip_default_backend_remains_unchanged() -> None:
    """The default backend remains the eager QuTiP implementation."""
    assert isinstance(backend_module.get_default_backend(), backend_module.QuTiPBackend)


def test_unknown_backend_name_still_raises_value_error() -> None:
    """Unknown backend names raise ValueError."""
    with pytest.raises(ValueError, match="Unknown backend 'nope'"):
        backend_module.set_default_backend("nope")


def test_missing_dynamiqs_extra_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting dynamiqs without the extra installed raises the install hint."""
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "quchip.backend.dynamiqs":
            raise ImportError("dynamiqs unavailable")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=re.escape(INSTALL_HINT)):
        backend_module.set_default_backend("dynamiqs")


def test_selecting_dynamiqs_returns_dynamiqs_backend_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Module-level selection instantiates the dynamiqs backend lazily."""
    fake_backend_cls = _install_fake_dynamiqs(monkeypatch)
    backend_module.set_default_backend("dynamiqs")

    assert isinstance(backend_module.get_default_backend(), fake_backend_cls)


def test_chip_backend_string_uses_same_lazy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chip-level string backend selection shares the lazy dynamiqs resolver."""
    real_import_module = importlib.import_module
    dynamiqs_imports: list[str] = []
    fake_backend_cls = _install_fake_dynamiqs(monkeypatch)

    def tracking_import_module(name: str, package: str | None = None):
        if name == "quchip.backend.dynamiqs":
            dynamiqs_imports.append(name)
            return types.SimpleNamespace(DynamiqsBackend=fake_backend_cls)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", tracking_import_module)

    chip = _single_qubit_chip(backend="dynamiqs")
    assert isinstance(chip.backend, fake_backend_cls)
    assert dynamiqs_imports == ["quchip.backend.dynamiqs"]


def test_chip_backend_string_missing_dynamiqs_matches_module_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chip-level dynamiqs selection raises the same install hint."""
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "quchip.backend.dynamiqs":
            raise ImportError("dynamiqs unavailable")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=re.escape(INSTALL_HINT)):
        _single_qubit_chip(backend="dynamiqs")
