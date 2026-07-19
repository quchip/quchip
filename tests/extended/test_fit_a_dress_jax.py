"""JAX residual and Jacobian coverage for ``fit_a_dress``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares as scipy_least_squares

pytestmark = pytest.mark.optional_backend

pytest.importorskip("dynamiqs")

from quchip import Capacitive, Chip, DuffingTransmon, Resonator, fit_a_dress  # noqa: E402
from quchip.backend.dynamiqs import DynamiqsBackend  # noqa: E402
from quchip.inverse_design import fit as fit_module  # noqa: E402


def test_fit_a_dress_passes_an_exact_jax_jacobian_to_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    """SciPy receives a JAX Jacobian matching finite differences of the residual."""
    checked = False

    def recording_least_squares(fun, *args, **kwargs):
        nonlocal checked
        x0 = np.asarray(kwargs["x0"], dtype=float)
        jacobian = np.asarray(kwargs["jac"](x0), dtype=float)
        step = 1e-6
        finite_difference = np.column_stack([
            (fun(x0 + step * np.eye(x0.size)[column]) - fun(x0 - step * np.eye(x0.size)[column]))
            / (2.0 * step)
            for column in range(x0.size)
        ])
        np.testing.assert_allclose(jacobian, finite_difference, rtol=2e-4, atol=2e-6)
        checked = True
        return scipy_least_squares(fun, *args, **kwargs)

    monkeypatch.setattr(fit_module, "least_squares", recording_least_squares)
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=3, label="r")
    coupling = Capacitive(q, r, g=0.02, label="qr")
    chip = Chip([q, r], [coupling], frame="rotating", backend=DynamiqsBackend())

    result = fit_a_dress(
        chip,
        coupling_targets={coupling: "g"},
        observable_targets={
            q: {"freq": 5.01, "anharmonicity": -0.25},
            r: {"freq": 7.01},
        },
    )

    assert checked
    assert result.solver_info["jacobian"] == "jax"
