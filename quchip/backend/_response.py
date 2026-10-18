"""Frequency-domain response algebra for NumPy and JAX backends."""

from functools import lru_cache, partial
from typing import Any

import numpy as np

from quchip.backend.containers import LinearResponseSolverResult
from quchip.utils.jax_utils import contains_tracer
from quchip.utils.values import DeferredValue

# Relative residual that each conditional-response block and JAX derivative solve must reach.
_RESIDUAL_LIMIT = 1e-7
_CONVERGENCE_MESSAGE = (
    "Conditional VNA response or derivative solve failed to converge. "
    "Inspect losses, frequencies and model conditioning."
)


def stationary_condition_number(liouvillian: Any, dimension: int, *, xp: Any) -> Any:
    """Measure conditioning after replacing one equation with unit trace."""
    trace_row = xp.eye(dimension, dtype=liouvillian.dtype).reshape(1, -1)
    return xp.linalg.cond(xp.concatenate((liouvillian[:-1], trace_row), axis=0))


def linear_response(problem: Any, *, xp: Any) -> LinearResponseSolverResult:
    """Solve passive-linear scattering and capture deferred conditioning."""
    hamiltonian = xp.asarray(problem.hamiltonian, dtype=complex)
    couplings = xp.asarray(problem.couplings, dtype=complex)
    scattering = xp.asarray(problem.scattering, dtype=complex)
    frequencies = xp.atleast_1d(xp.array(problem.frequencies, dtype=float, copy=True))
    drift = -1j * hamiltonian - 0.5 * couplings.conj().T @ couplings
    indices = xp.asarray(problem.plane_indices)
    sources = -couplings.conj().T @ scattering[:, indices]

    def systems() -> Any:
        return -1j * 2.0 * xp.pi * frequencies[:, None, None] * xp.eye(drift.shape[0])[None, :, :] - drift[None, :, :]

    matrices = systems()
    amplitudes = xp.linalg.solve(matrices, sources[None, :, :])
    responses = scattering[xp.ix_(indices, indices)][None, :, :] + couplings[indices] @ amplitudes
    residuals = xp.linalg.norm(matrices @ amplitudes - sources[None, :, :], axis=(-2, -1))
    return LinearResponseSolverResult(
        responses=responses,
        mode_amplitudes=amplitudes,
        residuals=residuals,
        _condition_numbers=DeferredValue(lambda: xp.linalg.cond(systems())),
        _mode_covariance=DeferredValue(lambda: _mode_covariance(problem, drift, couplings, scattering, xp)),
    )


def _mode_covariance(problem: Any, drift: Any, couplings: Any, scattering: Any, xp: Any) -> Any:
    """Solve the passive Markov Lyapunov equation A N + N A† + D = 0."""
    occupations = xp.asarray([0.0 if c.input_occupation is None else c.input_occupation
                              for c in problem.field_channels]
                             + [0.0] * (scattering.shape[0] - len(problem.field_channels)))
    sources = -couplings.conj().T @ scattering
    diffusion = (sources * occupations) @ sources.conj().T
    # A common rotating frame cancels from the covariance equation.
    count = drift.shape[0]
    drift = drift - 1j*xp.imag(xp.trace(drift))/count*xp.eye(count)
    if xp is np:
        from scipy.linalg import solve_continuous_lyapunov

        covariance = solve_continuous_lyapunov(drift, -diffusion)
    else:
        identity = xp.eye(count)
        generator = xp.kron(drift, identity) + xp.kron(identity, drift.conj())
        covariance = xp.linalg.solve(generator, -diffusion.reshape(-1)).reshape((count, count))
    return (covariance + covariance.conj().T)/2


# Small adjacent-sector blocks use direct solves. Large blocks stay matrix-free.
_COHERENCE_DENSE_LIMIT = 256


def coherence_response(problem: Any, *, xp: Any) -> Any:
    """Solve the Lindblad action on (N,N-1) from the highest sector downwards."""
    from quchip.backend.containers import CoherenceResponseSolverResult
    from quchip.utils.memory import dense_bytes, krylov_bytes, require_memory

    H = tuple(xp.asarray(h, dtype=complex) for h in problem.hamiltonians)
    lower = tuple(xp.asarray(channel, dtype=complex) for channel in problem.lowering)
    kept = tuple(xp.asarray(channel, dtype=complex) for channel in problem.conserving)
    sources = tuple(xp.asarray(b, dtype=complex) for b in problem.sources)
    sizes = tuple(h.shape[0] for h in H)
    counts = tuple(sizes[n] * sizes[n - 1] for n in range(1, len(H)))
    require_memory(sum(dense_bytes(d, d, copies=6) if d <= _COHERENCE_DENSE_LIMIT else krylov_bytes(d, 150)
                       for d in counts),
                   task="Conditional VNA coherence solve", remedy="Reduce the model or its local level counts.")
    drift = tuple(-1j * h - 0.5 * (
        xp.einsum("kab,kac->bc", channel.conj(), channel) + xp.einsum("kab,kac->bc", z.conj(), z)
    ) for h, channel, z in zip(H, lower, kept, strict=True))

    def dense(n: int) -> Any:
        return (xp.kron(drift[n], xp.eye(sizes[n - 1]))
                + xp.kron(xp.eye(sizes[n]), drift[n - 1].conj())
                + sum((xp.kron(a, b.conj()) for a, b in zip(kept[n], kept[n - 1], strict=True)),
                      start=xp.zeros((counts[n - 1],) * 2, dtype=complex)))

    active_sectors = tuple(n for n in range(1, len(H)) if counts[n - 1])
    direct_sectors = tuple(n for n in active_sectors if counts[n - 1] <= _COHERENCE_DENSE_LIMIT)
    generators = {n: dense(n) for n in direct_sectors}
    frequencies = xp.atleast_1d(xp.array(problem.frequencies, dtype=float, copy=True))
    indices = list(problem.plane_indices)
    plane = xp.asarray(indices)
    scattering = xp.asarray(problem.scattering)[xp.ix_(plane, plane)]
    operators = (drift, lower, kept, sources, generators)
    if xp is not np and contains_tracer((operators, frequencies, scattering)):
        # A traced sweep keeps one loop, so the program of the caller does not grow with the sweep.
        responses, residuals = _compiled_sweep_solve()(operators, frequencies, scattering, plane)
    else:
        # Eager sweeps of every length reuse one executable for each frequency.
        solve = partial(_solve_frequency, xp=np) if xp is np else _compiled_frequency_solve()
        solved = [solve(*operators, frequency, scattering, plane) for frequency in np.asarray(frequencies)]
        responses, residuals = (xp.stack(values) for values in zip(*solved, strict=True))

    def condition_numbers() -> Any:
        # Report the largest diagonal-block condition, not a full-generator condition.
        if len(direct_sectors) != len(active_sectors):
            return None
        rebuilt = {n: dense(n) for n in direct_sectors}
        return xp.stack([xp.max(xp.stack([xp.linalg.cond(-2j * xp.pi * f * xp.eye(counts[n - 1]) - a)
                                         for n, a in rebuilt.items()])) for f in frequencies])

    return CoherenceResponseSolverResult(responses, residuals, DeferredValue(condition_numbers))


@lru_cache(maxsize=1)
def _compiled_frequency_solve() -> Any:
    """Reuse one JAX executable for each frequency, with all model values passed as arguments."""
    import jax
    import jax.numpy as jnp

    return jax.jit(partial(_solve_frequency, xp=jnp))


@lru_cache(maxsize=1)
def _compiled_sweep_solve() -> Any:
    """Map the frequency solve over a traced sweep inside one compiled loop."""
    import jax
    from jax import lax

    solve = _compiled_frequency_solve()

    def sweep(operators: Any, frequencies: Any, scattering: Any, indices: Any) -> tuple[Any, Any]:
        # Map sequentially: batching every Krylov basis would scale workspace with the sweep.
        return lax.map(lambda frequency: solve(*operators, frequency, scattering, indices), frequencies)

    return jax.jit(sweep)


def _solve_frequency(
    drift: Any, lower: Any, kept: Any, sources: Any, generators: Any,
    frequency: Any, scattering: Any, indices: Any, *, xp: Any,
) -> tuple[Any, Any]:
    """Solve one frequency with bounded workspace, from the highest sector downwards."""
    sizes = tuple(a.shape[0] for a in drift)
    counts = tuple(sizes[n] * sizes[n - 1] for n in range(1, len(sizes)))
    active_sectors = tuple(n for n in range(1, len(sizes)) if counts[n - 1])

    def action(n: int, x: Any) -> Any:
        return (drift[n] @ x + x @ drift[n - 1].conj().T
                + sum((a @ x @ b.conj().T for a, b in zip(kept[n], kept[n - 1], strict=True)),
                      start=xp.zeros_like(x)))

    shift = -2j * xp.pi * frequency
    response = scattering
    above = None
    errors = []
    for n in reversed(active_sectors):
        rhs = sources[n]
        if above is not None:
            # Jump recycling from (N+1,N) into (N,N-1).
            rhs = rhs + xp.einsum("kab,bci,kdc->adi", lower[n + 1], above, lower[n].conj(), optimize=True)
        flat = rhs.reshape(counts[n - 1], len(indices))
        if n in generators:
            matrix = shift * xp.eye(counts[n - 1]) - generators[n]
            solved = xp.linalg.solve(matrix, flat).reshape(rhs.shape)
        else:
            diagonal = (shift - xp.diag(drift[n])[:, None] - xp.diag(drift[n - 1]).conj()[None, :]
                        - xp.einsum("ka,kb->ab", xp.diagonal(kept[n], axis1=-2, axis2=-1),
                                    xp.diagonal(kept[n - 1], axis1=-2, axis2=-1).conj()))
            diagonal = xp.where(xp.abs(diagonal) > 1e-14, diagonal, 1.)

            def matvec(value: Any) -> Any:
                x = value.reshape(sizes[n], sizes[n - 1])
                return (shift * x - action(n, x)).reshape(-1)

            def precondition(value: Any) -> Any:
                return value / diagonal.reshape(-1)

            columns = []
            for column in range(len(indices)):
                if xp is np:
                    from scipy.sparse.linalg import LinearOperator, gmres

                    shape = (counts[n - 1],) * 2
                    value, _ = gmres(LinearOperator(shape, matvec=matvec, dtype=complex), flat[:, column],
                                     M=LinearOperator(shape, matvec=precondition, dtype=complex),
                                     rtol=1e-11, atol=0., restart=64, maxiter=100)
                else:
                    import equinox as eqx
                    from jax import lax
                    from jax.scipy.sparse.linalg import gmres

                    def solve(operator: Any, source: Any) -> Any:
                        # Recompute GMRES's source-relative stopping scale for each
                        # primal, tangent and transpose solve. The incremental method
                        # checks it at each Krylov step, as SciPy does.
                        value, _ = gmres(operator, source, M=precondition, tol=1e-11, atol=0.,
                                         restart=64, maxiter=100, solve_method="incremental")
                        relative = (xp.linalg.norm(operator(value) - source)
                                    / xp.maximum(xp.linalg.norm(source), 1e-30))
                        return eqx.error_if(
                            value, ~xp.isfinite(relative) | (relative > _RESIDUAL_LIMIT), _CONVERGENCE_MESSAGE,
                        )

                    value = lax.custom_linear_solve(matvec, flat[:, column], solve=solve, transpose_solve=solve)
                columns.append(value)
            solved = xp.stack(columns, axis=-1).reshape(rhs.shape)
        error = xp.stack([shift * solved[:, :, i] - action(n, solved[:, :, i]) - rhs[:, :, i]
                          for i in range(len(indices))])
        norms = xp.linalg.norm(flat, axis=0)
        relative = xp.linalg.norm(error.reshape(len(indices), -1), axis=1) / xp.maximum(norms, 1e-30)
        maximum = xp.max(relative)
        invalid = ~xp.isfinite(maximum) | (maximum > _RESIDUAL_LIMIT)
        if xp is np:
            if invalid:
                raise ValueError(_CONVERGENCE_MESSAGE)
        else:
            import equinox as eqx

            solved = eqx.error_if(solved, invalid, _CONVERGENCE_MESSAGE)
        errors.append(maximum)
        response = response + xp.einsum("cab,bai->ci", lower[n][xp.asarray(indices)], solved)
        above = solved
    return response, xp.max(xp.stack(errors))
