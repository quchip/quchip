r"""Energy-participation-ratio (EPR) models of Josephson circuits.

An EPR analysis describes a circuit by its linear eigenmodes and by the
fraction of each mode's inductive energy stored in each Josephson junction.
Field solvers obtain these numbers from one eigenmode simulation in which every
junction is a linear inductor; pyEPR and Quantum Metal report them for Ansys
HFSS designs. :class:`EPRModel` stores them without importing a third-party
package and builds quchip chips from them.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from math import factorial, prod
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from quchip.utils.constants import _H_SI, TWO_PI, Phi_0
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.chip.coupling_base import BaseCoupling
    from quchip.devices.base import BaseDevice

#: Josephson energy in GHz of a 1 H junction inductance, ``(Phi_0 / 2 pi)**2 / h``.
_GHZ_HENRY = (Phi_0 / TWO_PI) ** 2 / _H_SI * 1e-9
#: Relative change of a mode's anharmonicity with two more Fock levels above which a cosine chip warns.
_CONVERGENCE_TOLERANCE = 1e-2
#: Modes whose anharmonicity is smaller than this in GHz are not checked for convergence.
_ANHARMONICITY_FLOOR = 1e-3

_FIELDS = ("freqs", "participations", "junction_energies", "junction_inductances", "signs", "labels",
           "quality_factors")


def _real_array(name: str, value: Any, ndim: int) -> Any:
    """Return a read-only float array, or a JAX array when *value* is traced."""
    array: Any
    if contains_tracer(value):
        array = jnp.asarray(value, dtype=jnp.float64)
    else:
        array = np.array(value, dtype=float)
        array.flags.writeable = False
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimension(s), got shape {array.shape}.")
    return array


def _concrete(array: Any) -> np.ndarray | None:
    """Return *array* for value checks, or ``None`` while it is traced."""
    return None if contains_tracer(array) else np.asarray(array)


def _first_order_chi(freqs: Any, participations: Any, junction_energies: Any) -> Any:
    """Return pyEPR's first-order matrix ``chi_mn`` in GHz, positive for a downward shift."""
    weighted = freqs[:, None] * participations
    return 0.25 * (weighted / junction_energies[None, :]) @ weighted.T


def _quadrature(dims: tuple[int, ...], mode: int) -> np.ndarray:
    """Return ``a + a†`` of one mode embedded in the mode-ordered product space."""
    lowering = np.diag(np.sqrt(np.arange(1.0, dims[mode])), 1)
    result = np.eye(1)
    for index, dim in enumerate(dims):
        result = np.kron(result, lowering + lowering.T if index == mode else np.eye(dim))
    return result


def _cosine_remainder(phase: Any, cos_trunc: int | None) -> Any:
    """Return ``cos(phase) - 1 + phase**2 / 2``, exactly or through order ``phase**(2 * cos_trunc)``."""
    square = phase @ phase
    if cos_trunc is not None:
        power = square
        remainder = 0.0 * square
        for order in range(2, cos_trunc + 1):
            power = power @ square
            remainder = remainder + (-1) ** order * power / factorial(2 * order)
        return remainder
    identity = np.eye(square.shape[0])
    if contains_tracer(phase):
        # The matrix exponential keeps derivatives finite when a mode does not
        # participate and the phase operator has degenerate eigenvalues.
        cosine = jnp.real(jax.scipy.linalg.expm(1j * phase))
    else:
        values, vectors = np.linalg.eigh(phase)
        cosine = (vectors * np.cos(values)) @ vectors.T
    return cosine - identity + 0.5 * square


def _junction_hamiltonian(phi_zpf: Any, junction_energies: Any, dims: tuple[int, ...],
                          cos_trunc: int | None) -> Any:
    """Return ``-sum_j E_J,j [cos(phi_j) - 1 + phi_j**2 / 2]`` in GHz on the mode product space."""
    quadratures = [_quadrature(dims, mode) for mode in range(len(dims))]
    hamiltonian: Any = np.zeros((prod(dims), prod(dims)))
    for junction in range(junction_energies.shape[0]):
        phase = sum(phi_zpf[mode, junction] * x for mode, x in enumerate(quadratures))
        hamiltonian = hamiltonian - junction_energies[junction] * _cosine_remainder(phase, cos_trunc)
    return 0.5 * (hamiltonian + hamiltonian.T)


def _mode_anharmonicity(freq: float, phases: np.ndarray, junction_energies: np.ndarray, levels: int,
                        cos_trunc: int | None) -> float:
    """Return one mode's anharmonicity in GHz, alone with the junction term on ``levels`` Fock states."""
    if levels < 3:
        return 0.0
    junction = _junction_hamiltonian(phases[None, :], junction_energies, (levels,), cos_trunc)
    energies = np.linalg.eigvalsh(np.diag(freq * np.arange(levels)) + junction)
    return float(energies[2] - 2 * energies[1] + energies[0])


class EPRModel:
    r"""Linear eigenmodes and junction participations of a Josephson circuit.

    Parameters
    ----------
    freqs : array_like, shape (M,)
        Linear eigenmode frequencies :math:`f_m` in GHz, computed with every
        junction replaced by its linear inductance.
    participations : array_like, shape (M, J)
        Inductive energy-participation ratios :math:`p_{mj}` in ``[0, 1]``:
        the fraction of mode ``m``'s inductive energy stored in junction ``j``.
    junction_energies : array_like, shape (J,), optional
        Josephson energies :math:`E_{J,j}` in GHz of the linear junction
        inductances. Give exactly one of ``junction_energies`` and
        ``junction_inductances``.
    junction_inductances : array_like, shape (J,), optional
        Linear junction inductances :math:`L_{J,j}` in H, converted with
        :math:`E_J = (\Phi_0/2\pi)^2/(h L_J)`.
    signs : array_like, shape (M, J), optional
        Sign :math:`s_{mj} = \pm 1` of junction ``j``'s phase in mode ``m``.
        ``None`` uses ``+1``. Only relative signs within a mode across two or
        more junctions change the spectrum.
    labels : sequence of str, optional
        Unique mode labels, used as device labels. ``None`` gives
        ``"mode_0"``, ``"mode_1"``, and so on.
    quality_factors : array_like, shape (M,), optional
        Total quality factors :math:`Q_m`, referenced to :math:`f_m`, giving
        energy-decay rates :math:`\kappa_m = 2\pi f_m/Q_m` in 1/ns. ``inf``
        marks a lossless mode. ``None`` omits dissipation.

    Attributes
    ----------
    freqs, participations, junction_energies, signs, labels, quality_factors
        Stored inputs; ``junction_energies`` is in GHz even when the model was
        built from inductances.
    phi_zpf : array, shape (M, J)
        Reduced zero-point phase fluctuations
        :math:`\varphi_{mj} = s_{mj}\sqrt{p_{mj} f_m / (2 E_{J,j})}`.

    Notes
    -----
    In the basis of linear modes, the circuit Hamiltonian is

    .. math::

        H = \sum_m f_m a_m^\dagger a_m
            - \sum_j E_{J,j}\left[\cos\hat\varphi_j - 1
            + \tfrac{1}{2}\hat\varphi_j^2\right],
        \qquad
        \hat\varphi_j = \sum_m \varphi_{mj}\,(a_m + a_m^\dagger).

    The quadratic part of each junction potential is already included in
    :math:`f_m`. First-order perturbation theory in the quartic term gives the
    diagonal Kerr Hamiltonian

    .. math::

        H_1 = \sum_m (f_m - \Delta_m)\,n_m
              + \sum_m \tfrac{A_m}{2}\,n_m(n_m - 1)
              + \sum_{m<n} K_{mn}\,n_m n_n,

    with :math:`\chi_{mn} = \tfrac{1}{4} f_m f_n \sum_j p_{mj} p_{nj}/E_{J,j}`,
    Lamb shift :math:`\Delta_m = \tfrac{1}{2}\sum_n \chi_{mn}`, anharmonicity
    :math:`A_m = -\chi_{mm}/2`, and full-pull cross-Kerr
    :math:`K_{mn} = -\chi_{mn}`. pyEPR reports the same first-order results in
    MHz with the opposite sign: its ``chi_O1`` diagonal is :math:`-A_m` and its
    off-diagonal is :math:`-K_{mn}`, while ``chip(nonlinearity="first_order")``
    returns the chip whose :meth:`~quchip.chip.chip.Chip.kerr_matrix` is
    :math:`A_m` and :math:`K_{mn}` in GHz. Its relative error grows as
    :math:`\varphi_{mj}^2`; at transmon values near 0.4 it is about 10% in
    :math:`A_m` and larger in :math:`K_{mn}`.

    The model is a snapshot of one field simulation. Changing ``freqs``,
    ``participations`` or ``junction_energies`` does not re-solve the linear
    network: in a real layout a new junction inductance also moves the mode
    frequencies and participations.

    References
    ----------
    Minev et al., *Energy-participation quantization of Josephson circuits*,
    npj Quantum Inf. 7, 131 (2021), https://doi.org/10.1038/s41534-021-00461-8.
    Nigg et al., *Black-box superconducting circuit quantization*,
    Phys. Rev. Lett. 108, 240502 (2012),
    https://doi.org/10.1103/PhysRevLett.108.240502.

    Examples
    --------
    >>> from quchip import EPRModel
    >>> epr = EPRModel(
    ...     freqs=[5.0, 7.0],
    ...     participations=[[0.95], [0.02]],
    ...     junction_inductances=[12e-9],
    ...     signs=[[1], [-1]],
    ...     labels=["q", "r"],
    ... )
    >>> chip = epr.chip(levels={"q": 12, "r": 4})
    >>> [device.label for device in chip.devices]
    ['q', 'r']
    """

    def __init__(
        self,
        freqs: Any,
        participations: Any,
        *,
        junction_energies: Any = None,
        junction_inductances: Any = None,
        signs: Any = None,
        labels: Sequence[str] | None = None,
        quality_factors: Any = None,
    ) -> None:
        freqs = _real_array("freqs", freqs, 1)
        participations = _real_array("participations", participations, 2)
        n_modes = freqs.shape[0]
        if n_modes == 0 or participations.shape[0] != n_modes or participations.shape[1] == 0:
            raise ValueError(
                f"participations must have shape (M, J) with M = len(freqs) = {n_modes} and J >= 1, "
                f"got {participations.shape}."
            )
        n_junctions = participations.shape[1]

        if (junction_energies is None) == (junction_inductances is None):
            raise ValueError("Give exactly one of junction_energies (GHz) and junction_inductances (H).")
        if junction_energies is None:
            inductances = _real_array("junction_inductances", junction_inductances, 1)
            concrete = _concrete(inductances)
            if concrete is not None and not np.all(np.isfinite(concrete) & (concrete > 0)):
                raise ValueError("junction_inductances must be positive and finite.")
            junction_energies = _GHZ_HENRY / inductances
        energies = _real_array("junction_energies", junction_energies, 1)
        if energies.shape != (n_junctions,):
            raise ValueError(f"Expected {n_junctions} junction values, got shape {energies.shape}.")

        signs = _real_array("signs", np.ones((n_modes, n_junctions)) if signs is None else signs, 2)
        if signs.shape != participations.shape:
            raise ValueError(f"signs must have shape {participations.shape}, got {signs.shape}.")

        labels = tuple(f"mode_{index}" for index in range(n_modes)) if labels is None else tuple(labels)
        if len(labels) != n_modes:
            raise ValueError(f"Expected {n_modes} labels, got {len(labels)}.")
        if not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != n_modes:
            raise ValueError(f"labels must be unique nonempty strings, got {labels!r}.")

        if quality_factors is not None:
            quality_factors = _real_array("quality_factors", quality_factors, 1)
            if quality_factors.shape != (n_modes,):
                raise ValueError(f"quality_factors must have shape ({n_modes},), got {quality_factors.shape}.")

        checks = (
            (freqs, lambda f: np.all(np.isfinite(f) & (f > 0)), "freqs must be positive and finite."),
            (participations, lambda p: np.all((p >= 0) & (p <= 1)), "participations must lie in [0, 1]."),
            (energies, lambda e: np.all(np.isfinite(e) & (e > 0)), "junction_energies must be positive and finite."),
            (signs, lambda s: np.all(np.abs(s) == 1), "signs must be +1 or -1."),
            (quality_factors, lambda q: np.all(q > 0), "quality_factors must be positive; use inf for no loss."),
        )
        for array, valid, message in checks:
            concrete = None if array is None else _concrete(array)
            if concrete is not None and not valid(concrete):
                raise ValueError(message)

        self._freqs = freqs
        self._participations = participations
        self._junction_energies = energies
        self._signs = signs
        self._labels: tuple[str, ...] = labels
        self._quality_factors = quality_factors

    @property
    def freqs(self) -> Any:
        """Linear eigenmode frequencies in GHz, shape ``(M,)``."""
        return self._freqs

    @property
    def participations(self) -> Any:
        """Inductive energy-participation ratios, shape ``(M, J)``."""
        return self._participations

    @property
    def junction_energies(self) -> Any:
        """Josephson energies of the linear junction inductances in GHz, shape ``(J,)``."""
        return self._junction_energies

    @property
    def signs(self) -> Any:
        """Junction phase signs in each mode, shape ``(M, J)``."""
        return self._signs

    @property
    def labels(self) -> tuple[str, ...]:
        """Mode labels in model order."""
        return self._labels

    @property
    def quality_factors(self) -> Any:
        """Total mode quality factors, shape ``(M,)``, or ``None`` without dissipation."""
        return self._quality_factors

    @property
    def phi_zpf(self) -> Any:
        """Reduced zero-point phase fluctuation of each junction in each mode, shape ``(M, J)``."""
        xp = jnp if contains_tracer((self._freqs, self._participations, self._junction_energies)) else np
        return self._signs * xp.sqrt(
            0.5 * self._freqs[:, None] * self._participations / self._junction_energies[None, :]
        )

    def replace(self, **changes: Any) -> "EPRModel":
        """Return a copy with selected constructor arguments replaced.

        Parameters
        ----------
        **changes
            Any of ``freqs``, ``participations``, ``junction_energies``,
            ``junction_inductances``, ``signs``, ``labels`` and
            ``quality_factors``, with the constructor's meaning. Passing
            ``junction_inductances`` replaces the stored junction energies,
            and ``quality_factors=None`` removes dissipation.

        Returns
        -------
        EPRModel
            New model; this one is unchanged.
        """
        unknown = set(changes) - set(_FIELDS)
        if unknown:
            raise TypeError(f"Unknown EPRModel fields: {sorted(unknown)}. Choose from {_FIELDS}.")
        arguments: dict[str, Any] = {
            "freqs": self._freqs,
            "participations": self._participations,
            "signs": self._signs,
            "labels": self._labels,
            "quality_factors": self._quality_factors,
        }
        if "junction_inductances" not in changes:
            arguments["junction_energies"] = self._junction_energies
        arguments.update(changes)
        return EPRModel(**arguments)

    def chip(
        self,
        *,
        levels: int | Sequence[int] | Mapping[str, int],
        nonlinearity: Literal["cosine", "first_order"] = "cosine",
        cos_trunc: int | None = None,
        **chip_options: Any,
    ) -> "Chip":
        r"""Build a chip with one device per mode.

        Parameters
        ----------
        levels : int, sequence of int, or mapping of str to int
            Fock levels of each mode: one value for all modes, one per mode in
            model order, or one per label. Each must be at least 2.
        nonlinearity : {"cosine", "first_order"}, default "cosine"
            ``"cosine"`` builds one :class:`~quchip.devices.resonator.Resonator`
            per mode at :math:`f_m` plus the junction term as
            :class:`~quchip.chip.effective.EffectiveTerms` labeled
            ``"junctions"``: the Hamiltonian pyEPR diagonalizes numerically.
            ``"first_order"`` builds the first-order Hamiltonian :math:`H_1`
            from one :class:`~quchip.devices.transmon.duffing.DuffingTransmon`
            per mode and one :class:`~quchip.chip.couplings.CrossKerr` per mode
            pair, labeled ``"<a>_<b>"``. It is diagonal in the Fock basis, so
            frames and approximations leave it unchanged; use it for chips too
            large to diagonalize and to compare with pyEPR's ``chi_O1``.
        cos_trunc : int or None, default None
            With ``"cosine"``, keep the Taylor series of the cosine through
            :math:`\hat\varphi^{2\,\mathrm{cos\_trunc}}`, as pyEPR's
            ``cos_trunc`` does. Must be at least 2. ``None`` uses the exact
            matrix cosine of the truncated phase operator.
        **chip_options
            Keywords forwarded to :class:`~quchip.chip.chip.Chip`, such as
            ``frame``, ``approximation``, ``backend``, ``basis`` and ``label``.

        Returns
        -------
        Chip
            New chip whose device labels are :attr:`labels`. With
            ``quality_factors``, each lossy mode decays at
            :math:`\kappa_m = 2\pi f_m/Q_m`: through ``internal_quality_factor``
            for ``"cosine"`` and ``T1 = 1/kappa_m`` for ``"first_order"``.

        Raises
        ------
        ValueError
            ``nonlinearity`` is unknown, ``cos_trunc`` is given for
            ``"first_order"`` or is below 2, or ``levels`` is invalid.

        Warns
        -----
        UserWarning
            With ``"cosine"``, a mode diagonalized alone with the junction term
            changes its anharmonicity by more than 1% between ``levels`` and
            ``levels + 2``. Modes whose anharmonicity is below 1 MHz are not
            checked. The check needs concrete inputs and does not test
            convergence of the couplings between modes.

        Notes
        -----
        The ``"cosine"`` junction term is a dense matrix of dimension
        ``prod(levels)``. It is retained without rotating-wave filtering under
        any chip approximation; in a rotating frame its off-diagonal parts
        become time-dependent terms. A transmon-like mode with
        :attr:`phi_zpf` near 0.4 needs about 8 levels for 1% accuracy in its
        anharmonicity and about 16 for :math:`10^{-6}`; with 4 levels the sign
        is wrong. Values derived from a traced JAX input stay traced, so
        dressed quantities of either chip can be differentiated with respect
        to the model inputs.
        """
        dims = self._mode_levels(levels)
        if nonlinearity == "first_order":
            if cos_trunc is not None:
                raise ValueError("cos_trunc applies only to nonlinearity='cosine'.")
            return self._first_order_chip(dims, chip_options)
        if nonlinearity == "cosine":
            if cos_trunc is not None and (not isinstance(cos_trunc, (int, np.integer)) or cos_trunc < 2):
                raise ValueError(f"cos_trunc must be an integer of at least 2 or None, got {cos_trunc!r}.")
            cos_trunc = None if cos_trunc is None else int(cos_trunc)
            self._warn_unconverged(dims, cos_trunc)
            return self._cosine_chip(dims, cos_trunc, chip_options)
        raise ValueError(f"nonlinearity must be 'cosine' or 'first_order', got {nonlinearity!r}.")

    def _warn_unconverged(self, dims: tuple[int, ...], cos_trunc: int | None) -> None:
        """Warn when a mode's own anharmonicity still changes with two more Fock levels."""
        if any(contains_tracer(value) for value in (self._freqs, self._participations, self._junction_energies)):
            return
        phases = np.asarray(self.phi_zpf)
        for mode, label in enumerate(self._labels):
            coarse, fine = (
                _mode_anharmonicity(float(self._freqs[mode]), phases[mode], np.asarray(self._junction_energies),
                                    levels, cos_trunc)
                for levels in (dims[mode], dims[mode] + 2)
            )
            if abs(fine) >= _ANHARMONICITY_FLOOR and abs(coarse - fine) > _CONVERGENCE_TOLERANCE * abs(fine):
                warnings.warn(
                    f"The cosine chip is not converged for mode {label!r} at {dims[mode]} levels: its "
                    f"anharmonicity changes by {abs(coarse / fine - 1):.1%} with two more levels. "
                    "Increase its levels.",
                    UserWarning,
                    stacklevel=3,
                )

    def _mode_levels(self, levels: int | Sequence[int] | Mapping[str, int]) -> tuple[int, ...]:
        if isinstance(levels, Mapping):
            if set(levels) != set(self._labels):
                raise ValueError(f"levels must give each mode once: expected {self._labels}, got {tuple(levels)}.")
            values = [levels[label] for label in self._labels]
        elif isinstance(levels, (int, np.integer)):
            values = [levels] * len(self._labels)
        else:
            values = list(levels)
            if len(values) != len(self._labels):
                raise ValueError(f"Expected {len(self._labels)} levels, got {len(values)}.")
        if not all(isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= 2
                   for value in values):
            raise ValueError(f"levels must be integers of at least 2, got {values}.")
        return tuple(int(value) for value in values)

    def _mode_quality(self, mode: int) -> Any | None:
        """Return mode ``mode``'s quality factor, or ``None`` when it is lossless."""
        if self._quality_factors is None:
            return None
        quality = self._quality_factors[mode]
        concrete = maybe_concrete_scalar(quality)
        return None if concrete is not None and not np.isfinite(concrete) else quality

    def _first_order_chip(self, dims: tuple[int, ...], chip_options: dict[str, Any]) -> "Chip":
        from quchip.chip.chip import Chip
        from quchip.chip.couplings import CrossKerr
        from quchip.devices.transmon.duffing import DuffingTransmon

        chi = _first_order_chi(self._freqs, self._participations, self._junction_energies)
        lamb_shift = 0.5 * chi.sum(axis=1)
        devices: list[BaseDevice] = []
        for mode, label in enumerate(self._labels):
            quality = self._mode_quality(mode)
            devices.append(DuffingTransmon(
                freq=self._freqs[mode] - lamb_shift[mode],
                anharmonicity=-0.5 * chi[mode, mode],
                levels=dims[mode],
                label=label,
                T1=None if quality is None else quality / (TWO_PI * self._freqs[mode]),
            ))
        couplings: list[BaseCoupling] = [
            CrossKerr(devices[a], devices[b], chi=-chi[a, b], label=f"{self._labels[a]}_{self._labels[b]}")
            for a in range(len(devices))
            for b in range(a + 1, len(devices))
        ]
        return Chip(devices, couplings, **chip_options)

    def _cosine_chip(self, dims: tuple[int, ...], cos_trunc: int | None, chip_options: dict[str, Any]) -> "Chip":
        from quchip.chip.chip import Chip
        from quchip.chip.effective import EffectiveTerms
        from quchip.devices.resonator import Resonator

        devices: list[BaseDevice] = [
            Resonator(freq=self._freqs[mode], levels=dims[mode], label=label,
                      internal_quality_factor=self._mode_quality(mode))
            for mode, label in enumerate(self._labels)
        ]
        junctions = EffectiveTerms(
            self._labels,
            dims,
            _junction_hamiltonian(self.phi_zpf, self._junction_energies, dims, cos_trunc),
            label="junctions",
        )
        return Chip(devices, effective_terms=[junctions], **chip_options)

    def __repr__(self) -> str:
        return f"EPRModel(labels={self._labels!r}, junctions={self._junction_energies.shape[0]})"
