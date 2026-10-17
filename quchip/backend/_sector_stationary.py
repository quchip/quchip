"""Matrix-free stationary and resolvent solves on excitation-sector blocks.

The states are the product energy labels with total excitation ``N <= K``,
ordered by ``N``. A density matrix ``X`` splits into blocks ``X_NM``. The
generator keeps the static Hamiltonian, which conserves ``N``, channels that
lower ``N`` by one or conserve it, and a coherent probe ``i(c* L - c L†)`` on
the lowering channels. Without the probe, the jumps move ``X_{N+1,M+1}`` into
``X_NM``, so the probe-free generator is block triangular. The solver uses its
inverse as the right preconditioner of a restarted GMRES iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import scipy.linalg
from scipy import sparse

from quchip.backend._memory import available_memory_bytes

# Relative residual at which GMRES stops, and the residual above which a solve fails.
_TOLERANCE = 1e-12
_RESIDUAL_LIMIT = 1e-8
# Krylov vectors kept between restarts, and the restart cycles allowed.
_RESTART = 64
_MINIMUM_RESTART = 8
_CYCLES = 40
# Sectors up to this size use the eigenvectors of their drift in the preconditioner;
# larger or ill-conditioned sectors use the eigenvectors of their Hamiltonian.
_EIG_LIMIT = 1024
_CONDITION_LIMIT = 1e8
# Damping added to every preconditioner denominator, relative to the largest decay rate.
_DAMPING = 1e-6

_CONVERGENCE_MESSAGE = (
    "The excitation-sector stationary solve did not converge (relative residual {residual:.1e}). "
    "Check that the model has one stationary state, or set the excitation cutoff explicitly."
)


class SectorConvergenceError(ValueError):
    """A sector solve that did not reach its residual limit."""


@dataclass(frozen=True)
class SectorSolve:
    """One converged sector solve and its diagnostics.

    Attributes
    ----------
    state : numpy.ndarray
        Solution in the sector-ordered basis.
    iterations : int
        GMRES iterations.
    residual : float
        Relative residual of the original, unpreconditioned equations.
    """

    state: np.ndarray
    iterations: int
    residual: float


class SectorGenerator:
    """Truncated Lindblad generator of a conserving model on the totals ``0..K``.

    Parameters
    ----------
    sizes : sequence of int
        Dimension of each sector ``N = 0..K``.
    hamiltonian : sparse matrix
        Static Hamiltonian in rad/ns, in the lab frame and relative to the
        vacuum energy, with states ordered by ``N``.
    lowering : sequence of sparse matrix
        Channel couplings in ``1/sqrt(ns)`` that lower ``N`` by one.
    conserving : sequence of sparse matrix
        Channel couplings in ``1/sqrt(ns)`` that conserve ``N``.
    """

    def __init__(self, sizes: Any, hamiltonian: Any, lowering: Any, conserving: Any) -> None:
        self.sizes = tuple(int(size) for size in sizes)
        self.cutoff = len(self.sizes) - 1
        self.offsets = np.concatenate(([0], np.cumsum(self.sizes))).astype(int)
        self.dimension = int(self.offsets[-1])
        dimension = self.dimension
        self.totals = np.repeat(np.arange(len(self.sizes)), self.sizes)
        self.lowering = tuple(sparse.csr_matrix(channel, dtype=complex) for channel in lowering)
        self.conserving = tuple(sparse.csr_matrix(channel, dtype=complex) for channel in conserving)
        damping = sparse.csr_matrix((dimension, dimension), dtype=complex)
        for channel in (*self.lowering, *self.conserving):
            damping = damping + channel.conj().T @ channel
        self._drift = (-1j * sparse.csr_matrix(hamiltonian, dtype=complex) - 0.5 * damping).tocsr()
        self._spans = tuple(slice(self.offsets[n], self.offsets[n + 1]) for n in range(len(self.sizes)))
        self._jumps = tuple(
            tuple(channel[self._spans[n - 1], self._spans[n]].tocsr() for n in range(1, len(self.sizes)))
            for channel in self.lowering
        )
        self._factors = tuple(self._factor(n) for n in range(len(self.sizes)))
        rates = [np.max(-values.real, initial=0.0) for values, _, _, _ in self._factors]
        self.damping = _DAMPING * max(max(rates), np.finfo(float).tiny)
        # A drift eigenvalue that decays no faster than the damping marks a possible dark state.
        self.dark = any(np.any(values.real > -self.damping) for values, _, _, _ in self._factors[1:])

    def _factor(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        """Return diagonal drift values, the basis, its inverse and conserving diagonals of one sector."""
        span = self._spans[n]
        drift = self._drift[span, span].toarray()
        size = drift.shape[0]
        basis = inverse = None
        if size <= _EIG_LIMIT:
            values, vectors = scipy.linalg.eig(drift)
            try:
                candidate = np.linalg.inv(vectors)
            except np.linalg.LinAlgError:
                candidate = None
            if (candidate is not None
                    and np.linalg.norm(vectors) * np.linalg.norm(candidate) <= _CONDITION_LIMIT * size):
                basis, inverse = vectors, candidate
        if basis is None or inverse is None:
            # The Hermitian part keeps a unitary basis, and the diagonal of the drift approximates it.
            _, basis = np.linalg.eigh(0.5j * (drift - drift.conj().T))
            inverse = basis.conj().T
            values = np.einsum("ij,ij->j", inverse.T, drift @ basis)
        diagonals = tuple(np.einsum("ij,ij->j", inverse.T, channel[span, span].toarray() @ basis)
                          for channel in self.conserving)
        return values, basis, inverse, diagonals

    def drift(self, frequency: float) -> Any:
        """Return ``-i H' - (1/2) sum L†L`` in the frame that rotates at ``frequency`` GHz per excitation."""
        return self._drift + sparse.diags(2j * np.pi * frequency * self.totals)

    def probe(self, drive: Any) -> Any:
        """Return ``W - W†`` with ``W = sum_k conj(c_k) L_k``, so the probe adds ``[W - W†, X]``."""
        dimension = self.dimension
        weighted = sparse.csr_matrix((dimension, dimension), dtype=complex)
        for coefficient, channel in zip(drive, self.lowering, strict=True):
            if coefficient != 0:
                weighted = weighted + np.conj(coefficient) * channel
        return (weighted - weighted.conj().T).tocsr()

    def operator(self, frequency: float, drive: Any, shift: complex = 0.0) -> Callable[[np.ndarray], np.ndarray]:
        """Return ``X -> (G + shift) X`` with the vacuum equation replaced by the trace.

        Parameters
        ----------
        frequency : float
            Frame frequency in GHz per excitation.
        drive : sequence of complex
            Probe coefficient ``c_k`` of each lowering channel, in ``1/sqrt(ns)``.
        shift : complex, default=0
            Value added to the generator, ``i 2π f`` for a resolvent at offset ``f``.
        """
        dimension = self.dimension
        drift = self.drift(frequency)
        probe = self.probe(drive)
        probe_dagger = probe.conj().T.tocsr()
        channels = (*self.lowering, *self.conserving)

        def apply(vector: np.ndarray) -> np.ndarray:
            state = vector.reshape(dimension, dimension)
            adjoint = state.conj().T
            out = drift @ state + (drift @ adjoint).conj().T
            for channel in channels:
                out += channel @ (channel @ adjoint).conj().T
            out += probe @ state - (probe_dagger @ adjoint).conj().T
            if shift:
                out += shift * state
            out[0, 0] = np.trace(state)
            return out.reshape(-1)

        return apply

    def preconditioner(self, frequency: float, shift: complex = 0.0) -> Callable[[np.ndarray], np.ndarray]:
        """Return the inverse of the damped probe-free generator.

        The blocks are solved from the highest ``N + M`` downwards, each in the
        basis of its sector drifts. The conserving jumps keep only their
        diagonals in that basis. The vacuum block comes last from the trace.
        """
        dimension, cutoff, spans = self.dimension, self.cutoff, self._spans
        denominators = {}
        for n in range(cutoff + 1):
            values_n, _, _, kept_n = self._factors[n]
            values_n = values_n + 2j * np.pi * frequency * n
            for m in range(cutoff + 1):
                if n == m == 0:
                    continue
                values_m, _, _, kept_m = self._factors[m]
                values_m = values_m + 2j * np.pi * frequency * m
                value = values_n[:, None] + values_m.conj()[None, :] + shift
                for left, right in zip(kept_n, kept_m, strict=True):
                    value = value + left[:, None] * right.conj()[None, :]
                # Keep every real part at or below -2 * damping, whatever the basis.
                denominators[n, m] = np.minimum(value.real, 0.0) - 2 * self.damping + 1j * value.imag

        def apply(vector: np.ndarray) -> np.ndarray:
            residual = vector.reshape(dimension, dimension)
            out = np.zeros_like(residual)
            for level in range(2 * cutoff, 0, -1):
                for n in range(max(0, level - cutoff), min(cutoff, level) + 1):
                    m = level - n
                    rhs = residual[spans[n], spans[m]]
                    if n < cutoff and m < cutoff:
                        above = out[spans[n + 1], spans[m + 1]]
                        for blocks in self._jumps:
                            rhs = rhs - blocks[n] @ (blocks[m] @ above.conj().T).conj().T
                    _, basis_n, inverse_n, _ = self._factors[n]
                    _, basis_m, inverse_m, _ = self._factors[m]
                    solved = (inverse_n @ rhs @ inverse_m.conj().T) / denominators[n, m]
                    out[spans[n], spans[m]] = basis_n @ solved @ basis_m.conj().T
            out[0, 0] = residual[0, 0] - sum(np.trace(out[spans[n], spans[n]]) for n in range(1, cutoff + 1))
            return out.reshape(-1)

        return apply

    def restart(self) -> int:
        """Return the Krylov restart length that fits in the available memory, or zero."""
        vector = 16 * self.dimension**2
        available = available_memory_bytes()
        if available is None:
            return _RESTART
        # Besides the basis, a solve keeps about eight state-sized work arrays.
        count = int(0.8 * available // vector) - 9
        return min(_RESTART, count) if count >= _MINIMUM_RESTART else 0

    def required_memory(self) -> float:
        """Return the bytes that a solve with the shortest restart needs."""
        return 16 * self.dimension**2 * (_MINIMUM_RESTART + 9)

    def stationary(self, frequency: float, drive: Any, guess: np.ndarray | None = None) -> SectorSolve:
        """Solve the stationary state at one frame frequency and probe.

        Parameters
        ----------
        frequency : float
            Frame frequency in GHz per excitation.
        drive : sequence of complex
            Probe coefficient of each lowering channel, in ``1/sqrt(ns)``.
        guess : numpy.ndarray or None, optional
            Density matrix to start from, for example the state at a nearby grid point.
        """
        dimension = self.dimension
        operator = self.operator(frequency, drive)
        precondition = self.preconditioner(frequency)
        vacuum = np.zeros((dimension, dimension), dtype=complex)
        vacuum[0, 0] = 1.0
        target = np.zeros(dimension**2, dtype=complex)
        target[0] = 1.0
        # The vacuum carries the zeroth order, so the iteration resolves the probe's correction.
        rhs = target - operator(vacuum.reshape(-1))
        start = None if guess is None else _useful(operator, rhs, guess.reshape(-1) - vacuum.reshape(-1))
        solved = _gmres(operator, precondition, rhs, start, self.restart())
        state = solved.state.reshape(dimension, dimension) + vacuum
        if self.dark:
            # A dark state can leave more than one stationary state; a second start would find another.
            # GMRES changes a start only along directions that the generator reaches, so this start is
            # used even though its residual is large.
            mixed = np.eye(dimension, dtype=complex) / dimension - vacuum
            other = _gmres(operator, precondition, rhs, mixed.reshape(-1), self.restart())
            difference = (float(np.linalg.norm(other.state - solved.state))
                          / max(float(np.linalg.norm(solved.state)), 1.0))
            if difference > 1e-6:
                raise ValueError(
                    "The truncated model has more than one stationary state: two starts converge to states "
                    f"that differ by {difference:.1e}. A dark state that the probe does not reach can cause "
                    "this. Raise the excitation cutoff or add the missing loss channel."
                )
        return SectorSolve(state, solved.iterations, solved.residual)

    def resolvent(self, frequency: float, drive: Any, source: np.ndarray, offsets: Any,
                  observables: Any) -> tuple[np.ndarray, int, float, float]:
        """Return ``Tr(O Y)`` for ``(G + i 2π f) Y = source`` with ``Tr Y = 0`` at each offset ``f``.

        Parameters
        ----------
        frequency : float
            Frame frequency in GHz per excitation.
        drive : sequence of complex
            Probe coefficient of each lowering channel.
        source : numpy.ndarray
            Traceless source in the sector-ordered basis.
        offsets : array_like
            Offsets in GHz.
        observables : sequence of sparse matrix
            Observables ``O``.

        Returns
        -------
        tuple
            Values with shape ``(len(offsets), len(observables))``, the largest
            iteration count, the largest relative residual and the largest
            ``|Tr Y|``.
        """
        rhs = np.array(source, dtype=complex).reshape(-1)
        rhs[0] = 0.0
        values = np.zeros((len(offsets), len(observables)), dtype=complex)
        iterations, residual, trace = 0, 0.0, 0.0
        previous = None
        restart = self.restart()
        for index, offset in enumerate(np.asarray(offsets, dtype=float)):
            shift = 2j * np.pi * offset
            operator = self.operator(frequency, drive, shift)
            start = None if previous is None else _useful(operator, rhs, previous)
            solved = _gmres(operator, self.preconditioner(frequency, shift), rhs, start, restart)
            response = solved.state.reshape(self.dimension, self.dimension)
            values[index] = [observable.multiply(response.T).sum() for observable in observables]
            iterations = max(iterations, solved.iterations)
            residual = max(residual, solved.residual)
            trace = max(trace, abs(np.trace(response)))
            previous = solved.state
        return values, iterations, residual, trace


def _useful(operator: Callable[[np.ndarray], np.ndarray], rhs: np.ndarray, guess: np.ndarray) -> np.ndarray | None:
    """Return ``guess`` if its residual is smaller than that of zero, else ``None``."""
    return guess if np.linalg.norm(rhs - operator(guess)) < np.linalg.norm(rhs) else None


def _gmres(operator: Callable[[np.ndarray], np.ndarray], precondition: Callable[[np.ndarray], np.ndarray],
           rhs: np.ndarray, guess: np.ndarray | None, restart: int) -> SectorSolve:
    """Solve ``operator(x) = rhs`` by restarted GMRES with right preconditioning.

    The residual that GMRES minimizes is the residual of the original
    equations, so the returned residual is the true one.
    """
    if restart <= 0:
        raise MemoryError("The excitation-sector solve does not fit in the available memory. "
                          "Lower the excitation cutoff or reduce the model.")
    scale = np.linalg.norm(rhs)
    if scale == 0.0:
        return SectorSolve(np.zeros_like(rhs), 0, 0.0)
    solution = np.zeros_like(rhs) if guess is None else guess.copy()
    iterations = 0
    residual = rhs - operator(solution)
    for _ in range(_CYCLES):
        norm = np.linalg.norm(residual)
        if norm <= _TOLERANCE * scale:
            break
        basis = np.empty((restart + 1, rhs.size), dtype=complex)
        hessenberg = np.zeros((restart + 1, restart), dtype=complex)
        basis[0] = residual / norm
        step = 0
        coefficients = np.zeros(0, dtype=complex)
        for step in range(restart):
            vector = operator(precondition(basis[step]))
            for _ in range(2):
                projection = basis[:step + 1].conj() @ vector
                vector -= projection @ basis[:step + 1]
                hessenberg[:step + 1, step] += projection
            hessenberg[step + 1, step] = np.linalg.norm(vector)
            target = np.zeros(step + 2, dtype=complex)
            target[0] = norm
            coefficients = np.linalg.lstsq(hessenberg[:step + 2, :step + 1], target, rcond=None)[0]
            estimate = np.linalg.norm(hessenberg[:step + 2, :step + 1] @ coefficients - target)
            iterations += 1
            if estimate <= _TOLERANCE * scale or hessenberg[step + 1, step] == 0.0:
                break
            basis[step + 1] = vector / hessenberg[step + 1, step]
        solution = solution + precondition(coefficients @ basis[:step + 1])
        residual = rhs - operator(solution)
    relative = float(np.linalg.norm(residual) / scale)
    if not np.isfinite(relative) or relative > _RESIDUAL_LIMIT:
        raise SectorConvergenceError(_CONVERGENCE_MESSAGE.format(residual=relative))
    return SectorSolve(solution, iterations, relative)
