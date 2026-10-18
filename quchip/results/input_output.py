"""Results for continuous-wave port scattering calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from quchip.utils.jax_utils import concrete_array_module, contains_tracer, is_jax_array, select_array_module

from quchip.utils.labeling import resolve_label


class _LazyDiagnostics(Mapping[str, Any]):
    """Read-only diagnostic values with explicitly deferred calculations."""

    def __init__(self, values: Mapping[str, Any]):
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> Any:
        value = self._values[key]
        return value() if callable(value) else value

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: object) -> bool:
        return key in self._values


@dataclass(frozen=True)
class SParameterResult:
    """Complete selected-port small-signal scattering over a sweep grid.

    Around a phase-sensitive operating point, the response is
    ``delta <b_out> = S delta beta + T conj(delta beta)``. ``matrix`` stores ``S``
    and ``conjugate_matrix`` stores ``T``. Both have shape
    ``(*shape, n_ports, n_ports)`` and use ``[..., output, input]`` indexing in
    ``ports`` order. ``s(output, input)`` and ``t(output, input)`` select
    individual entries.

    The passive-linear route reports zero for ``T``. ``numpy.asarray(result)``
    returns ``matrix``.

    Attributes
    ----------
    frequencies : scalar or array_like
        Probe frequencies in GHz.
    ports : tuple of str
        Port labels in output/input matrix order.
    axes : tuple
        Sweep-axis ``(name, values)`` pairs. ``shape`` is the sweep shape.
    diagnostics : tuple of mapping
        Per-point solver diagnostics.
    matrix, conjugate_matrix : array_like
        ``S`` and phase-conjugating ``T`` arrays with shape
        ``(*shape, n_ports, n_ports)``.
    shape : tuple of int
        Sweep-grid shape preceding the matrix axes.
    """
    # The stationary route computes both matrices from one shifted-Liouvillian
    # factorization.

    frequencies: Any
    ports: tuple[str, ...]
    axes: tuple[tuple[str, Any], ...]
    shape: tuple[int, ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    matrix: Any
    conjugate_matrix: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            tuple(item if isinstance(item, _LazyDiagnostics) else _LazyDiagnostics(item) for item in self.diagnostics),
        )

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Names of the result axes, in array order."""
        return tuple(name for name, _ in self.axes)

    def s(self, output: Any, input: Any) -> Any:
        """Return ``S(output, input)`` over the sweep grid.

        Parameters
        ----------
        output, input : port object or str
            Output row and input column.
        """
        return self.matrix[..., self._index(output), self._index(input)]

    def t(self, output: Any, input: Any) -> Any:
        """Return phase-conjugating ``T(output, input)`` over the sweep grid.

        Parameters
        ----------
        output, input : port object or str
            Output row and input column.
        """
        return self.conjugate_matrix[..., self._index(output), self._index(input)]

    @property
    def s11(self) -> Any:
        """Return reflection from the first selected plane back onto itself."""
        return self.matrix[..., 0, 0]

    @property
    def s21(self) -> Any:
        """Return transmission from the first selected plane to the second."""
        if len(self.ports) < 2:
            raise AttributeError("s21 requires at least two ports.")
        return self.matrix[..., 1, 0]

    def _index(self, plane: Any) -> int:
        label = resolve_label(plane)
        try:
            return self.ports.index(label)
        except ValueError:
            raise KeyError(f"Port {label!r} is not in this result. Available: {list(self.ports)}") from None

    def __array__(self) -> np.ndarray:
        return np.asarray(self.matrix)


# A port sees or excites a pole when their overlap exceeds this fraction of their norms.
_DARK = 1e-10


@dataclass(frozen=True)
class ResonanceResult:
    """Poles of the weak-probe response between selected ports.

    The weak-probe routes of :meth:`VNA.sweep` give the response
    ``S(f) = S_d + C (-i 2π f - A)^-1 (-C† S_d)`` with the drift
    ``A = -iH - C†C/2``. Each eigenvalue ``λ_p`` of ``A`` is a pole of the S
    entries where its residue does not vanish, at ``f_p = -Im λ_p / 2π`` with
    the response linewidth ``κ_p = -2 Re λ_p``. In the convention of
    :class:`SParameterResult`,

    ``S(f) = T(f) (D + Σ_p r_p / (f - z_p))``,  ``z_p = f_p + i κ_p / 4π``,

    where ``D`` is :attr:`direct`, the conjugate of ``S_d``, and ``T(f)``
    holds the transfers of the port reference planes.
    :meth:`scattering` evaluates this sum. The poles are in increasing
    frequency. Iteration, an index, or the label of a pole's dominant mode
    gives one :class:`Resonance`.

    ``κ_p`` is the full width at half maximum of ``|r_p / (f - z_p)|²`` in
    angular units, so it gives the width of ``|S - D|²`` near an isolated
    pole where the transfers vary slowly. It includes port loss, internal loss,
    Purcell decay and dephasing, so it can exceed the decay rate of the
    mode's population. ``f_p`` is the frequency of the open system, which can
    differ from the closed dressed frequency of :meth:`Chip.freq`.

    Attributes
    ----------
    frequencies : array_like, shape (poles,)
        Pole frequencies ``f_p`` in GHz, in the lab frame.
    linewidths : array_like, shape (poles,)
        Response linewidths ``κ_p`` in 1/ns.
    poles : array_like, shape (poles,)
        Complex poles ``z_p`` in GHz.
    mode_labels : tuple of str
        Modes of :attr:`composition`: the harmonic modes, or the devices whose
        first excited levels span the one-excitation states, then the poles
        that ``eliminate()`` retained.
    ports : tuple of str
        Port labels in the output and input order of :attr:`residues`.
    route : SolveRoute
        The weak-probe route and the reason for it.
    """

    frequencies: Any
    linewidths: Any
    poles: Any
    mode_labels: tuple[str, ...]
    ports: tuple[str, ...]
    route: Any
    _eigensystem: Any = field(repr=False)
    _problem: Any = field(repr=False)
    _transfers: Any = field(repr=False)

    @property
    def composition(self) -> Any:
        """Weight of each mode in each pole, shape ``(poles, modes)``.

        Row p is ``|v_pk|² / Σ_k |v_pk|²`` for the right eigenvector ``v_p`` of
        the drift, in :attr:`mode_labels` order. It describes the pole in the
        mode basis. It is not an occupation probability, because the drift is
        not normal in general. A JAX derivative of it raises, because JAX does
        not differentiate the eigenvectors of a general matrix.
        """
        vectors, _ = self._eigensystem.modes()
        xp = concrete_array_module(vectors)
        weights = xp.abs(vectors.T) ** 2
        return weights / xp.sum(weights, axis=1, keepdims=True)

    @property
    def dominant(self) -> tuple[str, ...]:
        """Mode label with the largest weight in each pole.

        The labels need concrete values, so inside ``jax.jit`` they raise.
        Select poles by index there.
        """
        vectors = self._eigensystem.vectors
        if contains_tracer(vectors):
            raise ValueError("The dominant labels need concrete values. Inside jax.jit, select poles by index.")
        weights = np.abs(np.asarray(vectors)) ** 2
        return tuple(self.mode_labels[int(k)] for k in np.argmax(weights, axis=0))

    @property
    def direct(self) -> Any:
        """Instantaneous scattering ``D`` between the ports, shape ``(ports, ports)``.

        It is the conjugate of ``S_d``, in the convention of :class:`SParameterResult`.
        """
        problem = self._problem
        indices = np.asarray(problem.plane_indices, dtype=int)
        scattering = problem.scattering
        xp = concrete_array_module(scattering)
        return xp.conj(xp.asarray(scattering, dtype=complex)[xp.ix_(indices, indices)])

    @property
    def residues(self) -> Any:
        """Residue ``r_p`` of each S entry at each pole, shape ``(poles, ports, ports)``.

        The entries are ``[pole, output, input]``, before the reference-plane
        transfers. A JAX derivative of them raises, as for :attr:`composition`.
        """
        return self._residues(*self._eigensystem.modes())

    @property
    def dark(self) -> Any:
        """Whether the selected ports do not see each pole, shape ``(poles,)``.

        The residue of pole p from input i to output j is the product of
        ``c_j v_p``, how output j sees the right eigenvector ``v_p``, and
        ``u_p† b_i``, how input i excites it. Here ``c_j`` is row j of ``C``
        and ``b_i`` is column i of ``B = -C† S_d``. The pole is dark when no
        output sees it, ``|c_j v_p| <= 1e-10 |c_j| |v_p|`` for every j, or no
        input excites it, ``|u_p† b_i| <= 1e-10 |u_p| |b_i|`` for every i. Its
        residues then vanish, and it stays an eigenvalue of the drift, for
        example of a mode that only a hidden loss channel reaches. Each port
        is compared with its own scale, so a weak port does not make a pole
        dark: a lossless resonator with a port rate of ``1e-12`` /ns reflects
        ``-1`` on resonance.
        """
        system = self._eigensystem
        problem = self._problem
        xp = concrete_array_module(system.vectors, problem.couplings, problem.scattering)
        indices = np.asarray(problem.plane_indices, dtype=int)
        couplings = xp.asarray(problem.couplings, dtype=complex)
        rows = couplings[indices]
        sources = -couplings.conj().T @ xp.asarray(problem.scattering, dtype=complex)[:, indices]
        norm = xp.linalg.norm
        seen = xp.abs(rows @ system.vectors) > _DARK * norm(rows, axis=1)[:, None] * norm(system.vectors, axis=0)
        excited = (xp.abs(system.inverse @ sources)
                   > _DARK * norm(system.inverse, axis=1)[:, None] * norm(sources, axis=0))
        return ~(xp.any(seen, axis=0) & xp.any(excited, axis=1))

    def scattering(self, frequencies: Any) -> Any:
        """Return ``S(f)`` from the poles, shape ``(*frequencies.shape, ports, ports)``.

        It includes the reference-plane transfers, so on the weak-probe route
        it equals :meth:`VNA.sweep` at the same frequencies. It uses the
        residues, so a JAX derivative of it raises.

        Parameters
        ----------
        frequencies : scalar or array_like
            Probe frequencies in GHz.
        """
        residues = self.residues
        xp = concrete_array_module(residues, frequencies, self.poles)
        values = xp.asarray(frequencies, dtype=float)
        flat = xp.reshape(values, (-1,))
        outbound, inbound = self._transfers(flat)
        terms = residues[None] / (flat[:, None, None, None] - self.poles[None, :, None, None])
        core = self.direct[None] + xp.sum(terms, axis=1)
        matrix = xp.conj(outbound[:, :, None] * inbound[:, None, :]) * core
        return xp.reshape(matrix, (*values.shape, len(self.ports), len(self.ports)))

    def _residues(self, vectors: Any, inverse: Any) -> Any:
        problem = self._problem
        xp = concrete_array_module(vectors, problem.couplings, problem.scattering)
        indices = np.asarray(problem.plane_indices, dtype=int)
        couplings = xp.asarray(problem.couplings, dtype=complex)
        sources = -couplings.conj().T @ xp.asarray(problem.scattering, dtype=complex)[:, indices]
        rows = (couplings[indices] @ vectors).T
        columns = inverse @ sources
        # 1/(-i 2π f - λ) = (i/2π)/(f - z); the public convention conjugates the residue.
        return xp.conj(1j / (2.0 * np.pi) * rows[:, :, None] * columns[:, None, :])

    def __len__(self) -> int:
        return len(self.mode_labels)

    def __iter__(self):
        return (Resonance(self, index) for index in range(len(self)))

    def __getitem__(self, key: Any) -> "Resonance":
        """Return the pole at an index, or the one pole that a mode label dominates."""
        if isinstance(key, (int, np.integer)):
            if not -len(self) <= key < len(self):
                raise IndexError(f"Pole index {key} is out of range for {len(self)} poles.")
            return Resonance(self, int(key) % len(self))
        label = resolve_label(key)
        matches = [index for index, name in enumerate(self.dominant) if name == label]
        if len(matches) != 1:
            raise KeyError(f"{len(matches)} poles have the dominant mode {label!r}. Dominant modes: "
                           f"{list(self.dominant)}. Select the pole by index.")
        return Resonance(self, matches[0])


@dataclass(frozen=True)
class Resonance:
    """One pole of a :class:`ResonanceResult`.

    Attributes
    ----------
    result : ResonanceResult
        The poles that hold this one.
    index : int
        Position of the pole in increasing frequency.
    """

    result: ResonanceResult = field(repr=False)
    index: int

    @property
    def frequency(self) -> Any:
        """Pole frequency in GHz."""
        return self.result.frequencies[self.index]

    @property
    def linewidth(self) -> Any:
        """Response linewidth κ in 1/ns."""
        return self.result.linewidths[self.index]

    @property
    def pole(self) -> Any:
        """Complex pole ``f + iκ/4π`` in GHz."""
        return self.result.poles[self.index]

    @property
    def dominant(self) -> str:
        """Mode label with the largest weight."""
        return self.result.dominant[self.index]

    @property
    def composition(self) -> dict[str, Any]:
        """Weight of each mode, keyed by mode label."""
        return dict(zip(self.result.mode_labels, self.result.composition[self.index]))

    @property
    def residues(self) -> Any:
        """Residue of each S entry, shape ``(ports, ports)``, as ``[output, input]``."""
        return self.result.residues[self.index]

    @property
    def dark(self) -> Any:
        """Whether the selected ports do not see this pole."""
        return self.result.dark[self.index]


@dataclass(frozen=True)
class MeanFieldResponseResult:
    """Stationary mean output fields from a finite coherent probe.

    ``values`` stores ``<b_out>`` at every selected plane with shape
    ``(*shape, n_ports)`` in ``ports`` order. ``incident`` stores the input
    amplitude ``beta`` broadcast to ``shape``. ``axes`` lists chip and pump sweep
    axes first, followed by ``"amplitude"`` and ``"frequency"`` when those
    arguments are arrays.

    If no fixed pump leaves a coherent mean at that plane and carrier,
    ``ratio(plane)`` approaches the corresponding small-signal S-parameter as
    ``beta`` goes to zero. ``ratio(plane)`` is ``NaN`` where ``beta`` is zero.
    The result contains one stationary mean-field branch. It does not encode
    sweep-rate hysteresis or metastable branches.

    Attributes
    ----------
    ports : tuple of str
        Selected output labels in the final array axis.
    input : str
        Probe input label.
    frequencies, amplitudes : scalar or array_like
        Probe values in GHz and ``1/sqrt(ns)``.
    axes : tuple
        Sweep-axis ``(name, values)`` pairs. ``shape`` is their array shape.
    diagnostics : tuple of mapping
        Per-point stationary-solver diagnostics.
    values : array_like
        Complex output means with shape ``(*shape, n_ports)``.
    incident : array_like
        Incident amplitude broadcast over ``shape``.
    shape : tuple of int
        Sweep-grid shape preceding the port axis.
    """

    ports: tuple[str, ...]
    input: str
    frequencies: Any
    amplitudes: Any
    axes: tuple[tuple[str, Any], ...]
    shape: tuple[int, ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    values: Any
    incident: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            tuple(item if isinstance(item, _LazyDiagnostics) else _LazyDiagnostics(item) for item in self.diagnostics),
        )

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Names of the result axes, in array order."""
        return tuple(name for name, _ in self.axes)

    def mean(self, plane: Any) -> Any:
        """Return stationary ``<b_out>`` at ``plane`` with shape ``shape``.

        Parameters
        ----------
        plane : port object or str
            Output reference plane.
        """
        label = resolve_label(plane)
        try:
            return self.values[..., self.ports.index(label)]
        except ValueError:
            raise KeyError(f"Port {label!r} is not in this result. Available: {list(self.ports)}") from None

    def ratio(self, plane: Any) -> Any:
        """Return the stationary ``<b_out>/beta`` at ``plane`` with shape ``shape``.

        The result is complex ``NaN`` where the incident ``beta`` is zero. Its
        zero-amplitude limit is the corresponding small-signal S-parameter when
        no fixed pump leaves a coherent mean at that plane and carrier.

        Parameters
        ----------
        plane : port object or str
            Output reference plane.
        """
        mean = self.mean(plane)
        xp = select_array_module(is_jax_array(mean))
        incident = xp.asarray(self.incident)
        zero = incident == 0
        return xp.where(zero, xp.nan + 0j, mean / xp.where(zero, 1.0, incident))


@dataclass(frozen=True)
class OutputSpectrumResult:
    """Stationary output-field fluctuation spectra and signal photon fluxes.

    ``signal_fluctuation_spectrum`` is device-generated spectral excess, with
    input-system interference for thermal fields. It can be negative.
    ``added_noise_spectrum`` is directly propagated thermal and amplifier noise,
    and ``total_fluctuation_spectrum`` is their sum. ``signal_photon_flux`` is
    the propagated device-field flux, split into ``signal_coherent_flux`` and
    ``signal_incoherent_flux``. These fluxes exclude added noise, because
    converting a spectral density to flux requires a detection bandwidth.

    Attributes
    ----------
    port : str
        Output port label.
    frequencies : array_like
        Offset frequencies in GHz.
    total_fluctuation_spectrum, signal_fluctuation_spectrum, added_noise_spectrum : array_like
        Normal-ordered spectral densities in frequency order.
    signal_photon_flux, signal_coherent_flux, signal_incoherent_flux : scalar
        Propagated flux terms in photons/ns.
    steady_state : SteadyStateResult
        Captured stationary state used for the spectra.
    fourier_convention : str
        Definition of the reported two-sided spectrum. The engineering
        correlation is the conjugate of the internal physics correlation.
    diagnostics : mapping
        The solve's route under ``"route"`` and its name under ``"solver"``,
        with the truncation entries of a solve on excitation sectors.
    """

    port: str
    frequencies: Any
    total_fluctuation_spectrum: Any
    signal_fluctuation_spectrum: Any
    added_noise_spectrum: Any
    signal_photon_flux: Any
    signal_coherent_flux: Any
    signal_incoherent_flux: Any
    steady_state: Any
    fourier_convention: str = "2 Re integral_0^inf d tau exp(-j 2 pi f tau) C_engineering(tau), j = i"
    diagnostics: Mapping[str, Any] = field(default_factory=lambda: _LazyDiagnostics({}), repr=False)


@dataclass(frozen=True)
class OutputCorrelationResult:
    """Normalized stationary output-field correlation versus delay.

    Attributes
    ----------
    order : {1, 2}
        Correlation order.
    input_port, output_port : str
        Correlation input and delayed output labels.
    delays : array_like
        Non-negative delays in ns.
    values, unnormalized : array_like
        Normalized and raw correlation values in delay order.
    input_intensity, output_intensity : scalar
        Mean photon fluxes used for normalization.
    steady_state : SteadyStateResult
        Captured stationary state.
    normalization : str
        Human-readable normalization convention.
    diagnostics : mapping
        The solve's route under ``"route"`` and its name under ``"solver"``,
        with the truncation entries of a solve on excitation sectors.
    """

    order: int
    input_port: str
    output_port: str
    delays: Any
    values: Any
    unnormalized: Any
    input_intensity: Any
    output_intensity: Any
    steady_state: Any
    normalization: str
    diagnostics: Mapping[str, Any] = field(default_factory=lambda: _LazyDiagnostics({}), repr=False)

    @property
    def port(self) -> str:
        """Delayed output port, kept for single-port result code."""
        return self.output_port

    @property
    def intensity(self) -> Any:
        """Delayed output intensity, kept for single-port result code."""
        return self.output_intensity
