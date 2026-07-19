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
    Bath,
    Capacitive,
    Chip,
    DuffingTransmon,
    Resonator,
    eliminate,
    set_default_backend,
    simulate,
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

def test_thermal_bath_solve_is_jittable_and_grad_in_temperature():
    """A thermal-bath solve is jittable and its gradient in temperature is finite and positive."""
    def final_n(temp):
        m = Resonator(freq=5.0, levels=6, label="m")
        chip = Chip([m], baths=[Bath("thermal", temperature=temp, rate=0.1)])
        tlist = jnp.linspace(0.0, 100.0, 50)
        res = simulate(
            chip, [], tlist,
            initial_state=chip.bare_state(),
            e_ops={m: m.number_operator()},
            check_truncation=False,
        )
        return jnp.real(res.expect("m")[-1])

    value = jax.jit(final_n)(jnp.float64(300.0))
    grad = jax.grad(final_n)(jnp.float64(300.0))
    assert jnp.isfinite(value) and value > 0.0
    # Hotter bath -> higher steady-state occupation.
    assert jnp.isfinite(grad) and float(grad) > 0.0


def test_collective_decay_bath_solve_is_jittable_and_grad_in_rate():
    """A collective-decay bath solve is jittable and its gradient in rate is finite and negative."""
    def final_excited(rate):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q0")
        q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q1")
        chip = Chip([q0, q1], baths=[Bath("collective_decay", rate=rate)])
        tlist = jnp.linspace(0.0, 50.0, 40)
        res = simulate(
            chip, [], tlist,
            initial_state=chip.bare_state({q0: 1}),
            e_ops={q0: q0.projector(1, 1)},
            check_truncation=False,
        )
        return jnp.real(res.expect("q0")[-1])

    value = jax.jit(final_excited)(jnp.float64(0.02))
    grad = jax.grad(final_excited)(jnp.float64(0.02))
    assert jnp.isfinite(value)
    # Faster collective decay -> less surviving excitation.
    assert jnp.isfinite(grad) and float(grad) < 0.0


# -- eliminate -----------------------------------------------------------------

def test_eliminate_effective_params_are_jittable_and_grad_in_g():
    """eliminate()'s effective freq_after is jittable and its gradient in g matches the perturbative 2g/Δ."""
    def freq_after(g):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
        r = Resonator(freq=7.0, quality_factor=3000.0, levels=3, label="r")
        chip = Chip([q, r], couplings=[Capacitive(q, r, g=g)])
        return eliminate(chip, "r").effective_params["q"]["freq_after"]

    value = jax.jit(freq_after)(jnp.float64(0.08))
    grad = jax.grad(freq_after)(jnp.float64(0.08))
    assert jnp.isfinite(value)
    # d/dg (g^2/Δ) = 2g/Δ, Δ = -2.0
    assert float(grad) == pytest.approx(2 * 0.08 / (5.0 - 7.0), rel=1e-4)


def test_chi_is_traced_and_grad_matches_leading_order():
    """eliminate()'s chi stays traced and dchi/dg matches the leading-order chi ∝ g² scaling."""
    def chi_of_g(g):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.5, quality_factor=5000.0, levels=4, label="r")
        chip = Chip([q, r], couplings=[Capacitive(q, r, g=g)])
        return eliminate(chip, "r").effective_params["q"]["chi"]

    chi, dchi_dg = jax.value_and_grad(chi_of_g)(jnp.float64(0.05))
    assert jnp.isfinite(dchi_dg)
    # χ ∝ g² at leading order, so dχ/dg ≈ 2χ/g.
    assert float(dchi_dg) == pytest.approx(2 * float(chi) / 0.05, rel=0.02)


def test_readout_snr_is_differentiable_through_the_full_pipeline():
    """The eliminate -> analyze_dispersive_readout pipeline is differentiable in g and jit-consistent."""
    # eliminate() -> analyze_dispersive_readout() composes into one traced
    # graph: d(SNR)/d(g) is finite and nonzero, and the pipeline jits.
    from quchip.analysis import analyze_dispersive_readout

    def snr_of_g(g):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=6.5, quality_factor=5000.0, levels=4, label="r")
        chip = Chip([q, r], couplings=[Capacitive(q, r, g=g)])
        eff = eliminate(chip, "r").effective_params["q"]
        ro = analyze_dispersive_readout(chi=eff["chi"], kappa=eff["kappa"], tau=500.0, n_photons=2.0)
        return ro.snr

    g0 = jnp.float64(0.05)
    grad = jax.grad(snr_of_g)(g0)
    assert jnp.isfinite(grad)
    assert float(grad) != 0.0
    jitted = float(jax.jit(snr_of_g)(g0))
    assert jitted == pytest.approx(float(snr_of_g(g0)), rel=1e-9)


def test_eliminated_chip_solve_is_jittable_and_grad_in_g():
    """A solve on the eliminated (reduced) chip is jittable and its gradient in g is finite and negative."""
    def final_excited(g):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
        r = Resonator(freq=7.0, quality_factor=2000.0, levels=3, label="r")
        full = Chip([q, r], couplings=[Capacitive(q, r, g=g)])
        reduced = eliminate(full, "r").chip
        rq = reduced["q"]
        tlist = jnp.linspace(0.0, 200.0, 60)
        res = simulate(
            reduced, [], tlist,
            initial_state=reduced.bare_state({rq: 1}),
            e_ops={rq: rq.projector(1, 1)},
            check_truncation=False,
        )
        return jnp.real(res.expect("q")[-1])

    value = jax.jit(final_excited)(jnp.float64(0.05))
    grad = jax.grad(final_excited)(jnp.float64(0.05))
    assert jnp.isfinite(value)
    # Larger g -> faster Purcell decay -> less surviving excitation.
    assert jnp.isfinite(grad) and float(grad) < 0.0
