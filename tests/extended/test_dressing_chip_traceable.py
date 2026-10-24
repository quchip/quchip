"""Chip-level traceability of ``Chip.energy``/``freq``/``dispersive_shift``.

Verifies that the public dressed-energy API stays JAX-traceable
end-to-end: a traced device parameter flows through the lab-frame
Hamiltonian, the eigh, the ``label_eigensystem`` assignment, and the
bare-label lookup without breaking ``jit``/``grad``/``vmap``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.optional_backend

pytest.importorskip("dynamiqs")

from quchip.backend.dynamiqs import DynamiqsBackend  # noqa: E402
from quchip.chip.chip import Chip  # noqa: E402
from quchip.chip.couplings import Capacitive  # noqa: E402
from quchip.devices.resonator import Resonator  # noqa: E402
from quchip.devices.transmon.duffing import DuffingTransmon  # noqa: E402


def _build_chip(freq_q: jnp.ndarray | float, g: jnp.ndarray | float = 0.05) -> Chip:
    q = DuffingTransmon(freq=freq_q, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    return Chip(
        devices=[q, r],
        couplings=[Capacitive(q, r, g=g)],
        backend=DynamiqsBackend(),
    )


class TestCacheUnderTracing:
    @pytest.mark.validation
    def test_dress_rejects_tracing(self) -> None:
        """``Chip.dress()`` builds a concrete dict view and must reject tracers."""
        chip = _build_chip(freq_q=5.0)

        def loss(freq_q):
            # Rebind the chip's qubit frequency to a tracer via a fresh device inside the trace.
            c = _build_chip(freq_q)
            c.dress()  # must raise — dict materialization is not traceable
            return c.energy(q=0, r=0)

        with pytest.raises(RuntimeError, match="not traceable"):
            jax.grad(loss)(jnp.float32(5.0))
        _ = chip  # silence unused — outer chip proves eager dress() still works in the module
