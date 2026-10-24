"""Dynamiqs backend behavior and solver coverage."""

from __future__ import annotations


import numpy as np
import numpy.testing as npt
import pytest

pytestmark = pytest.mark.optional_backend

dynamiqs = pytest.importorskip("dynamiqs")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from quchip.backend.dynamiqs import DynamiqsBackend  # noqa: E402
from quchip.backend.protocol import Backend  # noqa: E402
from quchip.chip.chip import Chip  # noqa: E402
from quchip.devices.transmon.duffing import DuffingTransmon  # noqa: E402
from quchip.control import ChargeDrive  # noqa: E402
from quchip.control.sequence import QuantumSequence  # noqa: E402
from quchip.control.envelopes import Square  # noqa: E402
from quchip.engine.ir import (  # noqa: E402
    CanonicalOperator,
    Constant,
    DynamicTerm,
    EngineResult,
    ResolvedSLH,
    ScalarModulation,
    Window,
)


@pytest.fixture
def backend() -> DynamiqsBackend:
    """Return a fresh dynamiqs backend instance."""
    return DynamiqsBackend()


@pytest.mark.unit
def test_dynamiqs_backend_enables_float64(backend: DynamiqsBackend) -> None:
    """Dynamiqs backend forces JAX float64 mode for physics accuracy."""
    assert jax.config.read("jax_enable_x64") is True
    assert backend.array_module is jnp


@pytest.mark.unit
def test_irregular_csr_uses_dense_when_dia_storage_would_be_larger(
    backend: DynamiqsBackend,
) -> None:
    """CSR lowering chooses dense storage when its distinct diagonals outgrow the matrix."""
    canonical = CanonicalOperator.from_csr(
        values=np.arange(1, 8, dtype=complex),
        indices=np.array([0, 1, 2, 3, 0, 0, 0]),
        indptr=np.array([0, 4, 5, 6, 7]),
        shape=(4, 4),
        dims=(4,),
        basis="fock",
        subsystem_labels=("q",),
    )

    rebuilt = backend.from_canonical_operator(canonical)

    assert getattr(rebuilt, "layout", None) == dynamiqs.dense
    npt.assert_allclose(
        np.asarray(backend.to_array(rebuilt)),
        np.array(
            [[1, 2, 3, 4], [5, 0, 0, 0], [6, 0, 0, 0], [7, 0, 0, 0]],
            dtype=complex,
        ),
    )


@pytest.mark.unit
def test_prepare_windowed_scalar_modulation_supports_traced_stop_time() -> None:
    """A windowed scalar modulation's traced stop time works under jax.jit and evaluates within the window."""
    backend = DynamiqsBackend()
    op = CanonicalOperator.from_dense(
        np.eye(2, dtype=complex),
        dims=(2,),
        basis="fock",
        subsystem_labels=("q",),
    )

    @jax.jit
    def sample_rhs(stop):
        desc = EngineResult(
            slh=ResolvedSLH.from_terms(
                static_terms=(),
                dynamic_terms=(
                    DynamicTerm(
                        operator=op,
                        time_dependence=ScalarModulation(
                            signal=Window(
                                child=Constant(1.0 + 0.0j),
                                start=0.0,
                                stop=stop,
                            )
                        ),
                        origin="drive",
                    ),
                ),
                collapse_terms=(),
            ),
            dims=(2,),
            metadata={},
        )
        prepared = backend.prepare_hamiltonian(desc, np.linspace(0.0, 1.0, 5))
        return backend.to_array(prepared.rhs(0.5))

    sample = np.asarray(sample_rhs(jnp.asarray(1.0)))
    npt.assert_allclose(sample, np.eye(2, dtype=complex), atol=1e-12)


@pytest.mark.validation
def test_batched_sesolve_handles_heterogeneous_problems_sequentially(backend: DynamiqsBackend) -> None:
    """The inherited dict-form batched API solves each heterogeneous problem independently."""
    # The engine's native vmap path lives in solve_batch (typed SolveBatch, structurally
    # homogeneous by construction); batched_sesolve is the protocol's sequential default.
    H0 = backend.from_array(np.zeros((2, 2), dtype=complex))
    H1 = backend.from_array(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
    problems = [
        {"H": H0, "psi0": backend.basis(2, 0), "tlist": jnp.linspace(0.0, 1.0, 5), "e_ops": [backend.number(2)]},
        {"H": H1, "psi0": backend.basis(2, 1), "tlist": jnp.linspace(0.0, 2.0, 5), "e_ops": [backend.identity(2)]},
    ]

    batched = backend.batched_sesolve(problems, progress=False)
    separate = [backend.sesolve(**problem) for problem in problems]

    assert len(batched) == len(separate) == 2
    for batch_result, single_result in zip(batched, separate):
        npt.assert_allclose(
            np.asarray(backend.to_array(batch_result.final_state)),
            np.asarray(backend.to_array(single_result.final_state)),
            atol=1e-12,
        )
        npt.assert_allclose(np.asarray(batch_result.expect[0]), np.asarray(single_result.expect[0]), atol=1e-12)


@pytest.mark.validation
def test_batched_mesolve_matches_separate_solves(backend: DynamiqsBackend) -> None:
    """The inherited dict-form batched mesolve matches independent solves."""
    H = backend.from_array(np.zeros((2, 2), dtype=complex))
    tlist = jnp.linspace(0.0, 1.0, 4)
    problems = [
        {
            "H": H,
            "rho0": backend.state_to_dm(backend.basis(2, 0)),
            "tlist": tlist,
            "c_ops": [np.sqrt(0.1) * backend.destroy(2)],
            "e_ops": [backend.number(2)],
        },
        {
            "H": H,
            "rho0": backend.state_to_dm(backend.basis(2, 1)),
            "tlist": tlist,
            "c_ops": [np.sqrt(0.1) * backend.destroy(2)],
            "e_ops": [backend.number(2)],
        },
    ]

    batched = backend.batched_mesolve(problems, progress=False)
    separate = [backend.mesolve(**problem) for problem in problems]

    assert len(batched) == len(separate) == 2
    for batch_result, single_result in zip(batched, separate):
        npt.assert_allclose(
            np.asarray(backend.to_array(batch_result.final_state)),
            np.asarray(backend.to_array(single_result.final_state)),
            atol=1e-12,
        )
        npt.assert_allclose(np.asarray(batch_result.expect[0]), np.asarray(single_result.expect[0]), atol=1e-12)


# ---------------------------------------------------------------------------
# Cached jitted ``solve_problem`` (the dynamiqs single-solve artifact cache):
# parity vs the un-jitted protocol-default path, grad/vmap traceability, and
# adversarial cache-correctness (no structural collision, no stale-value reuse).


def _r5_problem(chip, qubit, amp, tlist):
    seq = QuantumSequence(chip)
    seq.charge(qubit, envelope=Square(duration=40.0, amplitude=jnp.asarray(amp)), freq=5.0)
    return seq.build_problem(tlist=tlist, initial_state=chip.state(q=0), e_ops=chip.e_ops(q="n"),
                             states="final")


@pytest.mark.validation
def test_cached_jit_solve_no_stale_value_reuse() -> None:
    """Same structure with different traced values shares one compiled artifact but binds its own physics."""
    backend = DynamiqsBackend()
    tlist = np.linspace(0.0, 40.0, 5)
    cache = backend._get_jit_solve_cache()
    cache.clear()

    def problem_for(amp):
        qubit = DuffingTransmon(freq=5.0, anharmonicity=-0.30, levels=3, label="q")
        drive = ChargeDrive(target=qubit, label="d")
        chip = Chip(devices=[qubit], frame="rotating", backend=backend, label="cv")
        chip.wire(drive)
        return _r5_problem(chip, qubit, amp, tlist)

    p_lo, p_hi = problem_for(0.02), problem_for(0.08)
    r_lo, r_hi = backend.solve_problem(p_lo), backend.solve_problem(p_hi)

    # identical structure -> a single shared compiled artifact
    assert len(cache) == 1
    # yet each matches its OWN uncached reference (the reused artifact binds the
    # fresh operators, not a stale closure)
    for p, r in ((p_lo, r_lo), (p_hi, r_hi)):
        ru = Backend.solve_problem(backend, p)
        assert float(np.max(np.abs(np.asarray(r.expect[0]) - np.asarray(ru.expect[0])))) < 1e-9
    # ... and the two genuinely differ (proves it did not silently return one)
    assert float(np.max(np.abs(np.asarray(r_lo.expect[0]) - np.asarray(r_hi.expect[0])))) > 1e-3
