"""Memory estimates, and checks against the memory available to this process.

Call sites name what they allocate, for example the eigensystem of a matrix of
dimension ``D``, and the estimators below turn that into bytes. Each backend
adds the cost of its own lowering and solvers. :func:`require_memory` refuses
an allocation that cannot fit before quchip makes it.
"""

from __future__ import annotations

from math import prod
from pathlib import Path

import psutil

# Bytes of one complex double and of one 64-bit index.
COMPLEX_BYTES = 16
INDEX_BYTES = 8
# A dense Hermitian eigensystem keeps the matrix, a working copy that becomes
# the eigenvectors, the returned eigenvectors, and the complex and real
# workspace of LAPACK's divide-and-conquer driver.
_EIGENSYSTEM_COPIES = 6
# The cgroup v2 files of this process's container, when it has one.
_CGROUP = Path("/sys/fs/cgroup")


def dense_bytes(*shape: int, copies: int = 1, itemsize: int = COMPLEX_BYTES) -> int:
    """Return the bytes of ``copies`` dense arrays of ``shape``.

    Parameters
    ----------
    *shape : int
        Shape of one array.
    copies : int, default=1
        Number of arrays that exist at the same time.
    itemsize : int, default=16
        Bytes of one entry. The default is a complex double.
    """
    return int(copies * itemsize * prod(shape))


def eigensystem_bytes(dimension: int) -> int:
    """Return the peak bytes of the dense Hermitian eigensystem of a square matrix.

    Parameters
    ----------
    dimension : int
        Number of rows of the matrix.
    """
    return dense_bytes(dimension, dimension, copies=_EIGENSYSTEM_COPIES)


def level_table_bytes(states: int, devices: int) -> int:
    """Return the bytes of a table that gives the level of each device in each product state.

    Parameters
    ----------
    states : int
        Number of product states.
    devices : int
        Number of devices.
    """
    return dense_bytes(states, devices, itemsize=INDEX_BYTES)


def krylov_bytes(size: int, vectors: int) -> int:
    """Return the bytes of a Krylov basis.

    Parameters
    ----------
    size : int
        Number of entries of one vector.
    vectors : int
        Number of vectors that the basis keeps.
    """
    return dense_bytes(size, vectors)


def available_memory_bytes() -> int | None:
    """Return the memory this process can still allocate, or ``None`` when unknown.

    The value is the available system memory that ``psutil`` reports. Under a
    cgroup v2 memory limit, as in many Linux containers, the value is the
    remaining limit when that is smaller.
    """
    try:
        limits = [int(psutil.virtual_memory().available)]
    except (OSError, psutil.Error):
        return None
    try:
        maximum = (_CGROUP / "memory.max").read_text().strip()
        if maximum != "max":
            limits.append(int(maximum) - int((_CGROUP / "memory.current").read_text()))
    except (OSError, ValueError):
        pass
    return max(min(limits), 0)


def fits_memory(required: float) -> bool:
    """Return whether an estimated allocation fits, and ``True`` when memory is unknown.

    Parameters
    ----------
    required : float
        Estimated peak allocation in bytes.
    """
    available = available_memory_bytes()
    return available is None or required <= available


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
