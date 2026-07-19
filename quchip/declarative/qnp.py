"""Trace-safe numeric namespace for declarative model authors.

Use this namespace inside model implementations (:meth:`value`,
:meth:`local_hamiltonian`, :meth:`interaction`) so expressions stay
JAX-traceable.
"""

from __future__ import annotations

import jax.numpy as _jnp

from jax.numpy import *  # noqa: F403

__all__ = [name for name in dir(_jnp) if not name.startswith("_")]
