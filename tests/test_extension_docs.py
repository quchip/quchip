"""Keep the extension guide aligned with the supported public API."""

from __future__ import annotations

from tests.extension_guide import guide_code


def test_extension_guide_python_examples_execute_together():
    code = guide_code()

    assert code
    exec(code, {})
