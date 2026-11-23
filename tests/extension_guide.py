"""The reference classes of the extension guide, built from its code blocks.

The guide holds the only copy of these classes, so the tests that import them
here check the code that readers copy.
"""

from __future__ import annotations

import re
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "extensions.md"


def guide_code() -> str:
    """Return the guide's Python blocks, joined in page order."""
    return "\n\n".join(re.findall(r"```python\n(.*?)```", GUIDE.read_text(), flags=re.DOTALL))


exec(compile(guide_code(), str(GUIDE), "exec"), globals())
