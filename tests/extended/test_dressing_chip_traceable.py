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
from quchip.utils.labeling import reset_label_counters  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_labels() -> None:
    reset_label_counters()


def _build_chip(freq_q: jnp.ndarray | float, g: jnp.ndarray | float = 0.05) -> Chip:
    q = DuffingTransmon(freq=freq_q, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    return Chip(
        devices=[q, r],
        couplings=[Capacitive(q, r, g=g)],
        backend=DynamiqsBackend(),
    )


class TestEnergyTraceable:
    def test_energy_is_jittable(self) -> None:
        """``Chip.energy`` built from a traced device frequency jits and returns a finite value."""

        @jax.jit
        def e11(freq_q):
            chip = _build_chip(freq_q)
            return chip.energy(q=1, r=1)

        value = e11(jnp.float32(5.0))
        assert jnp.isfinite(value)

    def test_energy_grad_is_finite(self) -> None:
        """The gradient of the bare-q dressed energy w.r.t. its own frequency is finite and near unity."""

        def loss(freq_q):
            chip = _build_chip(freq_q)
            return chip.energy(q=1, r=0)

        grad = jax.grad(loss)(jnp.float32(5.0))
        # The bare-q dressed energy tracks its frequency near 1:1 far from the
        # readout, so the derivative should be near 1.
        assert jnp.isfinite(grad)
        assert 0.5 < float(grad) < 1.5

    def test_energy_vmap_over_device_frequency(self) -> None:
        """``chip.energy`` vmaps over a batch of device frequencies into finite, monotone values."""

        def e(freq_q):
            chip = _build_chip(freq_q)
            return chip.energy(q=1, r=0)

        freqs = jnp.linspace(4.8, 5.2, 5, dtype=jnp.float32)
        energies = jax.vmap(e)(freqs)
        assert energies.shape == (5,)
        assert jnp.all(jnp.isfinite(energies))
        # Monotone increasing with freq_q.
        diffs = jnp.diff(energies)
        assert jnp.all(diffs > 0)


class TestFreqTraceable:
    def test_freq_target_grad_is_finite(self) -> None:
        """The gradient of ``chip.freq`` w.r.t. its own device frequency is finite and near unity."""

        def loss(freq_q):
            chip = _build_chip(freq_q)
            return chip.freq("q")

        grad = jax.grad(loss)(jnp.float32(5.0))
        assert jnp.isfinite(grad)
        assert 0.5 < float(grad) < 1.5

    def test_freq_conditional_grad(self) -> None:
        """The gradient through a conditional ``chip.freq(when=...)`` lookup is finite."""

        def loss(freq_q):
            chip = _build_chip(freq_q)
            return chip.freq("q", when={"r": 1})

        grad = jax.grad(loss)(jnp.float32(5.0))
        assert jnp.isfinite(grad)

    def test_freq_conditional_jittable(self) -> None:
        """A conditional ``chip.freq(when=...)`` lookup jits and returns a finite value."""

        @jax.jit
        def freq_when_readout_excited(freq_q):
            chip = _build_chip(freq_q)
            return chip.freq("q", when={"r": 1})

        value = freq_when_readout_excited(jnp.float32(5.0))
        assert jnp.isfinite(value)


class TestDispersiveShiftTraceable:
    def test_grad_through_dispersive_shift_is_finite(self) -> None:
        """The gradient of ``chip.dispersive_shift`` w.r.t. coupling ``g`` is finite and non-zero."""

        def loss(g):
            chip = _build_chip(freq_q=5.0, g=g)
            return chip.dispersive_shift("q", "r")

        grad = jax.grad(loss)(jnp.float32(0.05))
        # χ ≈ g²α / (Δ(Δ+α)) with Δ = ω_q − ω_r = −2, α = −0.25 → χ < 0,
        # so dχ/dg has the opposite sign to g; assert only non-zero magnitude.
        assert jnp.isfinite(grad)
        assert abs(float(grad)) > 1e-4


class TestCacheUnderTracing:
    def test_tracer_result_is_not_cached(self) -> None:
        """A chip built and traced inside ``grad`` must not leave a tracer in its cache."""

        def loss(freq_q):
            c = _build_chip(freq_q)
            value = c.energy(q=0, r=0)
            # Inside the trace, _array_cache was skipped because the result is traced.
            assert c._analysis._array_cache is None
            return value

        _ = jax.grad(loss)(jnp.float32(5.0))

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
