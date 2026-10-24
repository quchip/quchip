"""Stack levels that make quchip warnings point at user code."""

import os
import sys
from types import FrameType


def user_stacklevel() -> int:
    """Return the ``warnings.warn`` stack level of the first caller outside quchip.

    The caller of this function calls ``warnings.warn`` directly.
    """
    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frame: FrameType | None = sys._getframe(2)
    level = 2
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(package + os.sep):
        frame, level = frame.f_back, level + 1
    return level
