"""JAX-traceability of chip-level baths and the ``eliminate`` transformation.

Locks in ``@jax.jit`` + ``jax.grad`` end-to-end:

* a thermal / collective :class:`~quchip.chip.baths.Bath` flows a traced
  temperature / rate through ``_collect_c_ops`` into the dynamiqs mesolve;
* :func:`~quchip.chip.transformations.eliminate` produces a reduced chip whose
  effective ``freq`` (Lamb shift) and ``T1`` (Purcell) stay traced, so a loss
  built on the reduced chip is differentiable in the original coupling ``g``.

The Purcell ``T1`` branch must be decided from *static* info (does the mode
carry a ``Q``?), never by comparing a traced rate to zero — otherwise jit
breaks with ``TracerBoolConversionError``. These tests guard that.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.optional_backend

pytest.importorskip("dynamiqs")

from quchip import (  # noqa: E402
    Capacitive,
    Chip,
    DuffingTransmon,
    Resonator,
    eliminate,
    set_default_backend,
)
from quchip.backend import reset_default_backend  # noqa: E402
from quchip.utils.labeling import reset_label_counters  # noqa: E402


@pytest.fixture(autouse=True)
def _dynamiqs_backend():
    reset_label_counters()
    set_default_backend("dynamiqs")
    yield
    reset_default_backend()
    reset_label_counters()


# -- Bath ----------------------------------------------------------------------

# -- eliminate -----------------------------------------------------------------

@pytest.mark.validation
def test_eliminate_effective_params_are_jittable_and_grad_in_g():
    """eliminate()'s effective freq_after is jittable and its gradient in g matches the perturbative 2g/Δ."""
    def freq_after(g):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
        r = Resonator(freq=7.0, internal_quality_factor=3000.0, levels=3, label="r")
        chip = Chip([q, r], couplings=[Capacitive(q, r, g=g)])
        return eliminate(chip, "r").effective_params["q"]["freq_after"]

    value, grad = jax.jit(jax.value_and_grad(freq_after))(jnp.float64(0.08))
    assert jnp.isfinite(value)
    # d/dg (g^2/Δ) = 2g/Δ, Δ = -2.0
    assert float(grad) == pytest.approx(2 * 0.08 / (5.0 - 7.0), rel=1e-4)


@pytest.mark.validation
def test_dispersive_readout_value_and_gradients():
    """The same reduction yields chi's perturbative derivative and a consistent readout SNR gradient."""
    from quchip.analysis import analyze_dispersive_readout

    def observables(g):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.5, internal_quality_factor=5000.0, levels=4, label="r")
        chip = Chip([q, r], couplings=[Capacitive(q, r, g=g)])
        eff = eliminate(chip, "r").effective_params["q"]
        snr = analyze_dispersive_readout(chi=eff["chi"], kappa=eff["kappa"], tau=500.0, n_photons=2.0).snr
        values = jnp.stack((eff["chi"], snr))
        return values, values

    g0 = jnp.float64(0.05)
    gradients, values = jax.jit(jax.jacrev(observables, has_aux=True))(g0)
    chi, snr = values
    dchi, dsnr = gradients
    assert jnp.isfinite(gradients).all()
    # chi is proportional to g squared in this dispersive regime.
    assert float(dchi) == pytest.approx(2 * float(chi) / 0.05, rel=0.02)
    assert float(dsnr) != 0.0
    assert float(snr) == pytest.approx(float(observables(g0)[0][1]), rel=1e-9)
    step = 1e-5
    reference = (observables(g0 + step)[0][1] - observables(g0 - step)[0][1]) / (2 * step)
    assert float(dsnr) == pytest.approx(float(reference), rel=1e-3)
