"""Warnings for deprecated public names."""

import warnings


def warn_renamed(old: str, new: str, *, removal: str = "0.5") -> None:
    """Warn that quchip ``removal`` will remove a compatibility name."""
    warnings.warn(
        f"{old} is deprecated; use {new}. The old name will be removed in quchip {removal}.",
        DeprecationWarning,
        stacklevel=3,
    )
