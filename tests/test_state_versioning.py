"""Tests for the StateVersioned mutation-tracking mixin."""

from __future__ import annotations

import pytest

from quchip.utils.state_versioning import StateVersioned


class _Simple(StateVersioned):
    """Tracks one public attribute; ``label`` is declared untracked."""

    _untracked_names = frozenset({"label"})

    def __init__(self, value: float, label: str = "x") -> None:
        self.value = value
        self.label = label
        self._private = 0


def test_no_bumps_during_init():
    """state_version stays 0 through construction."""
    obj = _Simple(1.0)
    assert obj.state_version == 0


def test_tracked_public_write_bumps_exactly_once():
    """Each tracked public-attribute write after construction bumps state_version by one."""
    obj = _Simple(1.0)
    obj.value = 2.0
    assert obj.state_version == 1
    obj.value = 3.0
    assert obj.state_version == 2


def test_untracked_and_private_writes_never_bump():
    """Writes to an _untracked_names attribute or an underscore-prefixed attribute never bump state_version."""
    obj = _Simple(1.0)
    obj.label = "y"
    obj._private = 5
    assert obj.state_version == 0


class _FinishCounter(StateVersioned):
    """Records every ``_finish_init`` call into an external list."""

    def __init__(self, a: float, calls: list[str]) -> None:
        self._finish_calls = calls
        self.a = a

    def _finish_init(self) -> None:
        self._finish_calls.append("finish")
        super()._finish_init()


class _FinishCounterChild(_FinishCounter):
    """Subclass whose __init__ chains super().__init__."""

    def __init__(self, a: float, b: float, calls: list[str]) -> None:
        super().__init__(a, calls)
        self.b = b


def test_finish_init_fires_exactly_once_through_super_init_chain():
    """A subclass __init__ chaining super().__init__ fires _finish_init exactly once, after construction."""
    calls: list[str] = []
    obj = _FinishCounterChild(1.0, 2.0, calls)
    assert calls == ["finish"]
    assert obj.state_version == 0
    obj.b = 3.0
    assert obj.state_version == 1


class _RaisingHook(StateVersioned):
    """Raises from ``_on_attr_set`` once construction has finished."""

    def __init__(self, value: float) -> None:
        self.value = value

    def _on_attr_set(self, name: str) -> None:
        if self._tracking_enabled:
            raise RuntimeError("hook failure")


def test_raising_hook_still_bumps_version():
    """A raising _on_attr_set hook leaves the attribute mutated and state_version bumped."""
    obj = _RaisingHook(1.0)
    with pytest.raises(RuntimeError):
        obj.value = 2.0
    assert obj.value == 2.0
    assert obj.state_version == 1


class _HookCounter(StateVersioned):
    """Logs every attribute name seen by ``_on_attr_set`` into an external list.

    The hook appends to *calls* rather than mutating a tracked attribute on
    ``self`` — mutating a tracked attribute from inside the hook would
    re-trigger ``__setattr__`` and recurse.
    """

    def __init__(self, value: float, calls: list[str]) -> None:
        self._calls = calls
        self.value = value

    def _on_attr_set(self, name: str) -> None:
        self._calls.append(name)


def test_on_attr_set_fires_on_every_set():
    """_on_attr_set fires once per attribute set, tracked or not, including during __init__."""
    calls: list[str] = []
    obj = _HookCounter(1.0, calls)
    assert calls == ["_calls", "value"]
    obj.value = 2.0
    assert calls == ["_calls", "value", "value"]
