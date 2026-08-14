"""Keep the extension guide aligned with the supported public API."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_extension_guide_python_examples_execute_together():
    text = (ROOT / "docs" / "extensions.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)

    assert blocks
    exec("\n\n".join(blocks), {})
