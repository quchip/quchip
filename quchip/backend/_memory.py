"""Memory checks for solvers that form Liouville-space superoperators."""

from __future__ import annotations

from pathlib import Path


def available_memory_bytes() -> int | None:
    """Return the memory this process can still allocate, or ``None`` when unknown.

    Linux reports ``MemAvailable`` and any cgroup v2 limit, whichever is
    smaller; other platforms use ``psutil`` when it is installed.
    """
    limits: list[int] = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                limits.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    cgroup = Path("/sys/fs/cgroup")
    try:
        maximum = (cgroup / "memory.max").read_text().strip()
        if maximum != "max":
            limits.append(int(maximum) - int((cgroup / "memory.current").read_text()))
    except (OSError, ValueError):
        pass
    if not limits:
        try:
            import psutil
        except ImportError:
            return None
        limits.append(int(psutil.virtual_memory().available))
    return max(min(limits), 0)


def require_memory(required: float, *, task: str, remedy: str) -> None:
    """Raise :class:`MemoryError` before an allocation that cannot fit.

    Parameters
    ----------
    required : float
        Estimated peak allocation in bytes.
    task : str
        What would allocate it, for the error message.
    remedy : str
        How to avoid the allocation.
    """
    available = available_memory_bytes()
    if available is not None and required > available:
        raise MemoryError(
            f"{task} needs about {required / 1e9:.1f} GB, but only {available / 1e9:.1f} GB "
            f"is available. {remedy}"
        )
