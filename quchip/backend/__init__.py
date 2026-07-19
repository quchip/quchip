"""Backend selection and concrete backend implementations.

quchip ships two backends:

* :class:`QuTiPBackend` — CPU, ``qutip.Qobj`` operators, process-parallel
  sweeps via loky. Default backend.
* :class:`DynamiqsBackend` — JAX/dynamiqs, fully differentiable, native
  ``vmap`` batched solves. Optional extra (``pip install quchip[dynamiqs]``).

The public entry points are :func:`get_default_backend` /
:func:`set_default_backend` / :func:`reset_default_backend`.
Engine internals use the thread-safe :func:`_backend_context` to scope a
backend for the duration of a single assembly pass (every backend call the
engine makes is reentrant because the override is a ``ContextVar``).

Example
-------
>>> from quchip.backend import get_default_backend, set_default_backend
>>> backend = get_default_backend()  # QuTiPBackend by default
>>> set_default_backend("dynamiqs")   # doctest: +SKIP
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from quchip.backend._dims import (
    compute_two_body_permutation,
    default_solver_steps,
    normalize_dims_from_list,
    validate_two_body_indices,
)
from quchip.backend.containers import (
    DeferredBatch,
    EagerBatch,
    EigensystemData,
    PreparedBatch,
    PreparedHamiltonian,
    SolverResult,
    VmappedBatch,
)
from quchip.backend.protocol import Backend, Operator, State

__all__ = [
    "Backend",
    "Operator",
    "State",
    "SolverResult",
    "PreparedHamiltonian",
    "PreparedBatch",
    "EagerBatch",
    "VmappedBatch",
    "DeferredBatch",
    "EigensystemData",
    "compute_two_body_permutation",
    "default_solver_steps",
    "normalize_dims_from_list",
    "validate_two_body_indices",
    "QuTiPBackend",
    "get_default_backend",
    "set_default_backend",
    "reset_default_backend",
]

# Name -> (module, class). Kept lazy so `import quchip` works without dynamiqs.
_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    "qutip": ("quchip.backend.qutip", "QuTiPBackend"),
    "dynamiqs": ("quchip.backend.dynamiqs", "DynamiqsBackend"),
}
_DYNAMIQS_INSTALL_HINT = (
    "DynamiqsBackend requires dynamiqs and JAX. Install with: pip install quchip[dynamiqs]"
)

_backend_override: ContextVar[Backend | None] = ContextVar("quchip_backend_override", default=None)
_default_backend: Backend | None = None


def _resolve_backend_class(name: str) -> type[Backend]:
    """Import and return the backend class registered under *name*.

    Emits a clear install hint when the optional dynamiqs/JAX stack is
    missing rather than surfacing the raw ``ImportError``.
    """
    module_spec = _LAZY_BACKENDS.get(name)
    if module_spec is None:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(_LAZY_BACKENDS)}")

    module_name, class_name = module_spec
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        if name == "dynamiqs":
            raise ImportError(_DYNAMIQS_INSTALL_HINT) from exc
        raise
    return getattr(module, class_name)


# One shared instance per backend *class*. Backend-side caches (e.g. the
# dynamiqs jitted-solve cache) live on the instance, so handing out a fresh
# instance per ``backend="dynamiqs"`` call would recompile on every solve of
# a gradient loop. Keying on the resolved class (not the name) keeps name
# resolution lazy — a re-registered or patched backend class resolves fresh
# — while repeated coercions of the same class share one instance,
# mirroring the sharing ``get_default_backend`` already does.
_class_instances: dict[type, Backend] = {}


def _coerce_backend(backend: str | Backend) -> Backend:
    """Resolve a backend name to its shared per-class instance, or pass an instance through."""
    if isinstance(backend, str):
        cls = _resolve_backend_class(backend)
        instance = _class_instances.get(cls)
        if instance is None:
            instance = cls()
            _class_instances[cls] = instance
        return instance
    if isinstance(backend, Backend):
        return backend
    raise TypeError(f"Expected str or Backend instance, got {type(backend).__name__}")


@contextmanager
def _backend_context(backend: Backend) -> Iterator[None]:
    """Scope *backend* as the active default for the duration of the ``with`` block.

    Thread/async-safe via :class:`contextvars.ContextVar`. Used by the engine
    to stamp an assembly pass with the chip's own backend without mutating
    the module-level default.
    """
    token = _backend_override.set(backend)
    try:
        yield
    finally:
        _backend_override.reset(token)


def get_default_backend() -> Backend:
    """Return the active default backend.

    Returns (in order) the nearest :func:`_backend_context` override, the
    user-set default from :func:`set_default_backend`, or a freshly
    instantiated :class:`QuTiPBackend` (cached for subsequent calls).
    """
    override = _backend_override.get()
    if override is not None:
        return override

    global _default_backend
    if _default_backend is None:
        _default_backend = _resolve_backend_class("qutip")()
    return _default_backend


def set_default_backend(backend: str | Backend) -> None:
    """Set the active default backend by name (``"qutip"``/``"dynamiqs"``) or instance."""
    global _default_backend
    _default_backend = _coerce_backend(backend)


def reset_default_backend() -> None:
    """Clear the cached default backend; the next :func:`get_default_backend` rebuilds it."""
    global _default_backend
    _default_backend = None


def __getattr__(name: str):
    if name == "QuTiPBackend":
        return _resolve_backend_class("qutip")
    raise AttributeError(f"module 'quchip.backend' has no attribute {name!r}")
