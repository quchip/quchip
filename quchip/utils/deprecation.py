"""Warnings for public names that quchip 0.5 will remove."""

import warnings


def warn_renamed(old: str, new: str) -> None:
    """Warn that quchip 0.5 will remove a compatibility name."""
    warnings.warn(
        f"{old} is deprecated; use {new}. The old name will be removed in quchip 0.5.",
        DeprecationWarning,
        stacklevel=3,
    )
