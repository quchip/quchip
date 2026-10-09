"""Cross-resonance Hamiltonian tomography analysis.

Fits the target qubit's Bloch-vector trajectories from a CR pulse-duration
sweep. Extracts the six effective Hamiltonian coefficients {IX, IY, IZ, ZX, ZY,
ZZ} in GHz (the package's units contract).

The CR effective Hamiltonian in the two-qubit subspace is:

    H_eff = (I ⊗ A + Z ⊗ B) / 2,   A, B ∈ span{X, Y, Z}

which expands to six coefficients:

    ω_IX, ω_IY, ω_IZ  (control-state-independent)
    ω_ZX, ω_ZY, ω_ZZ  (control-state-dependent)

in units of ordinary frequency (GHz).

References
----------
Sheldon, Magesan, Chow, Gambetta, "Procedure for systematically tuning up
cross-talk in the cross resonance gate", PRA 93, 060302(R) (2016).

Unit convention: public ``durations`` are in **ns** and returned
coefficients are in **GHz**. The fit runs internally in Hz/seconds (the
conditioning heuristics are tuned for CR-tomography scales, following
Sheldon et al.); both conversions
happen exactly once at the public boundary. Multiply by 2π to get angular
frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from quchip.utils.labeling import resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.control.drive import BaseDrive
    from quchip.devices.base import BaseDevice


# ---------------------------------------------------------------------------
# Bloch-trajectory model
# ---------------------------------------------------------------------------


def bloch_model(
    t: np.ndarray,
    px: float, py: float, pz: float,
    td: float,
    bx: float, by: float, bz: float,
) -> np.ndarray:
    r"""Model the Bloch-vector evolution under a constant drive Hamiltonian with a single decay envelope.

    The model assumes a drive Hamiltonian

        H = π (px X + py Y + pz Z)

    so the Bloch vector precesses at frequency
    :math:`f = \sqrt{p_x^2 + p_y^2 + p_z^2}` Hz about the axis
    :math:`\hat{n} = (p_x, p_y, p_z)/f`. A single exponential envelope
    :math:`\exp(-t / t_d)` damps all three Bloch components equally, without
    separate :math:`T_1` and :math:`T_2` decay processes.

    Parameters
    ----------
    t : (N,) ndarray
        Times in seconds.
    px, py, pz : float
        Drive Hamiltonian coefficients in Hz (ordinary frequency).
    td : float
        Decay time constant in seconds.
    bx, by, bz : float
        Bloch vector offset (fixed point, reached asymptotically).

    Returns
    -------
    (3, N) ndarray
        Rows are ``[⟨X(t)⟩, ⟨Y(t)⟩, ⟨Z(t)⟩]``.
    """
    f = np.sqrt(px * px + py * py + pz * pz)
    env = np.exp(-t / max(td, 1e-30))
    if f < 1e-12:
        return np.vstack([
            bx * np.ones_like(t),
            by * np.ones_like(t),
            env + bz,
        ])
    w = 2 * np.pi * f
    nx, ny, nz = px / f, py / f, pz / f
    c, s = np.cos(w * t), np.sin(w * t)
    return np.vstack([
        (ny * s + nx * nz * (1 - c)) * env + bx,
        (-nx * s + ny * nz * (1 - c)) * env + by,
        (nz * nz + (1 - nz * nz) * c) * env + bz,
    ])


# ---------------------------------------------------------------------------
# Fitting internals
# ---------------------------------------------------------------------------


def _residuals(p, t, dx, dy, dz, wx, wy, wz):
    m = bloch_model(t, *p)
    return np.concatenate([(m[0] - dx) * wx, (m[1] - dy) * wy, (m[2] - dz) * wz])


def _estimate_frequency(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate oscillation frequency (Hz) from an irregularly-sampled series.

    Uses FFT on a uniformly re-sampled copy, with a Savitzky-Golay derivative
    fallback for low-frequency signals that are poorly resolved by the FFT.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    intervals = np.diff(x)
    dt = np.median(intervals)
    if dt <= 0:
        return 1e6

    if not np.allclose(intervals, dt, rtol=1e-3):
        x_uniform = np.arange(x[0], x[-1], dt)
        y = np.interp(x_uniform, x, y)
        x = x_uniform

    n = len(x)
    fft_vals = np.fft.rfft(y - np.mean(y))
    freqs = np.fft.rfftfreq(n, dt)

    if len(fft_vals) < 2:
        return 1e6

    magnitudes = np.abs(fft_vals[1:])
    freq_guess = freqs[1:][np.argmax(magnitudes)]

    freq_resolution = 1.0 / (dt * n)
    if freq_guess < 1.5 * freq_resolution and n >= 5:
        wlen = min(n, max(5, n // 4))
        if wlen % 2 == 0:
            wlen -= 1
        if wlen >= 3:
            y_smooth = savgol_filter(y, window_length=wlen, polyorder=min(2, wlen - 1))
            y_amp = np.max(np.abs(y_smooth - np.mean(y_smooth)))
            if y_amp > 1e-10:
                max_deriv = np.max(np.abs(np.diff(y_smooth) / dt))
                freq_guess = max_deriv / (y_amp * 2 * np.pi)

    return abs(freq_guess)


def _generate_guesses(t, dx, dy, dz):
    """Multi-start initial guesses: FFT phi-sweep + sign-flip octants."""
    guesses = []

    omega_list = []
    for y_data in [dx, dy, dz]:
        ymin, ymax = np.percentile(y_data, [10, 90])
        if ymax - ymin < 0.2:
            continue
        try:
            omega_list.append(_estimate_frequency(t, y_data))
        except Exception:
            continue

    fg = np.mean(omega_list) if omega_list else 1e6

    zmin, zmax = np.percentile(dz, [10, 90])
    z_amp = np.clip((zmax - zmin) / 2, 0, 1)
    theta = np.arccos(np.sqrt(z_amp))

    td_guess = t[-1] * 2
    bx_g, by_g = dx.mean(), dy.mean()
    bz_g = np.mean(dz[-max(1, len(dz) // 5):])

    dt_step = t[1] - t[0] if len(t) > 1 else 1.0
    df = 1.0 / (dt_step * len(t))

    for fg_shifted in [fg, fg - df / 2, fg + df / 2]:
        for phi in np.linspace(-np.pi, np.pi, 5):
            guesses.append([
                fg_shifted * np.cos(theta) * np.cos(phi),
                fg_shifted * np.cos(theta) * np.sin(phi),
                fg_shifted * np.sin(theta),
                td_guess, bx_g, by_g, bz_g,
            ])

    if fg < df:
        guesses.append([fg, fg, 0.0, td_guess, bx_g, by_g, bz_g])

    ax = (dx.max() - dx.min()) / 2
    ay = (dy.max() - dy.min()) / 2
    p0_base = [
        ay * fg * 0.5 or 0.1 * fg,
        ax * fg * 0.5 or 0.1 * fg,
        np.sqrt(max(0, 1 - (dz.max() - dz.min()) / 2)) * fg,
        td_guess, bx_g, by_g, bz_g,
    ]
    for s0, s1, s2 in [
        (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1),
        (-1, -1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1),
    ]:
        p = list(p0_base)
        p[0] *= s0
        p[1] *= s1
        p[2] *= s2
        guesses.append(p)

    return guesses


def _fit_bloch(t, dx, dy, dz, sx=None, sy=None, sz=None):
    """Fit one control-state Bloch trajectory and retain optimizer status.

    Parameters
    ----------
    t : (N,) ndarray
        Durations in seconds.
    dx, dy, dz : (N,) ndarrays
        Target Bloch components for one control state.
    sx, sy, sz : (N,) ndarrays or None
        Per-point uncertainties (inverse weights). ``None`` → unit weights.

    Returns
    -------
    tuple
        Native optimizer result and covariance, or ``None`` when uncertainty
        is unavailable. The fitted vector is [px, py, pz, td, bx, by, bz].
    """
    idx = np.argsort(t)
    t, dx, dy, dz = t[idx], dx[idx], dy[idx], dz[idx]
    wx = 1.0 / sx[idx] if sx is not None else np.ones_like(t)
    wy = 1.0 / sy[idx] if sy is not None else np.ones_like(t)
    wz = 1.0 / sz[idx] if sz is not None else np.ones_like(t)

    guesses = _generate_guesses(t, dx, dy, dz)

    lb = [-np.inf, -np.inf, -np.inf, 1e-12, -2, -2, -2]
    ub = [np.inf, np.inf, np.inf, np.inf, 2, 2, 2]

    best, best_cost = None, np.inf
    last_error = None
    for p0 in guesses:
        try:
            r = least_squares(
                _residuals, p0,
                args=(t, dx, dy, dz, wx, wy, wz),
                bounds=(lb, ub), method="trf", max_nfev=5000,
            )
            if r.cost < best_cost:
                best, best_cost = r, r.cost
        except (ValueError, FloatingPointError) as error:
            last_error = error

    if best is None:
        raise RuntimeError("No initial guess produced a finite CR fit candidate.") from last_error

    return best, _fit_covariance(best)


def _fit_covariance(result):
    """Local residual-scaled covariance, unavailable without rank and convergence."""
    jac = np.asarray(result.jac)
    dof = jac.shape[0] - jac.shape[1]
    if not result.success or dof <= 0 or not np.all(np.isfinite(jac)):
        return None
    scales = np.linalg.norm(jac, axis=0)
    if np.any(scales == 0):
        return None
    try:
        _, singular, vt = np.linalg.svd(jac / scales, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular[-1] <= np.finfo(float).eps * max(jac.shape) * singular[0]:
        return None
    inverse = vt.T / singular / scales[:, None]
    covariance = (inverse @ inverse.T) * (2 * result.cost / dof)
    return covariance if np.all(np.isfinite(covariance)) else None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CRHamiltonianResult:
    """Store CR Hamiltonian coefficients in GHz and local uncertainty estimates.

    The uncertainties use a local, residual-scaled covariance, even when
    per-point standard deviations supply weights. An unavailable estimate is
    ``None``. Convergence alone does not establish identifiability.

    All six quantities are *ordinary* frequency (not angular). Multiply by 2π
    to convert to rad/ns. The convention agrees with Sheldon et al. (PRA 93,
    060302(R), 2016):

        H_eff = (I ⊗ A + Z ⊗ B) / 2

    where A = ω_IX X + ω_IY Y + ω_IZ Z and B = ω_ZX X + ω_ZY Y + ω_ZZ Z.

    ``params_ctrl0`` / ``params_ctrl1`` hold the raw fit output
    ``[px, py, pz, td, bx, by, bz]`` for each control state. They are
    diagnostics only and stay in the fit's internal units (px/py/pz in Hz, td
    in seconds).

    Attributes
    ----------
    IX, IY, IZ, ZX, ZY, ZZ : float
        Fitted ordinary-frequency coefficients in GHz.
    IX_err, IY_err, IZ_err, ZX_err, ZY_err, ZZ_err : float or None
        Local residual-scaled uncertainty estimates in GHz.
    params_ctrl0, params_ctrl1 : ndarray or None
        Raw fit parameters for the two control states.
    cov_ctrl0, cov_ctrl1 : ndarray or None
        Raw local covariance matrices.
    cost_ctrl0, cost_ctrl1 : float or None
        Least-squares costs.
    converged : bool or None
        Fit convergence status.
    message : str
        Solver status message.
    """

    IX: float
    IY: float
    IZ: float
    ZX: float
    ZY: float
    ZZ: float
    IX_err: float | None
    IY_err: float | None
    IZ_err: float | None
    ZX_err: float | None
    ZY_err: float | None
    ZZ_err: float | None
    params_ctrl0: np.ndarray | None = None
    params_ctrl1: np.ndarray | None = None
    cov_ctrl0: np.ndarray | None = None
    cov_ctrl1: np.ndarray | None = None
    cost_ctrl0: float | None = None
    cost_ctrl1: float | None = None
    converged: bool | None = None
    message: str = ""

    def coeffs(self) -> dict[str, tuple[float, float | None]]:
        """Return ``{name: (value_ghz, err_ghz)}`` for all six coefficients."""
        return {k: (getattr(self, k), getattr(self, k + "_err"))
                for k in ("IX", "IY", "IZ", "ZX", "ZY", "ZZ")}

    def _format(self, header: str) -> str:
        """Build the per-coefficient MHz summary text under the given header."""
        lines = [f"{header} (MHz):"]
        for k, (v, e) in self.coeffs().items():
            uncertainty = "uncertainty unavailable" if e is None else f"+/- {e * 1e3:.4f}"
            lines.append(f"  {k} = {v * 1e3:+.4f} {uncertainty}")
        status = "unavailable" if self.converged is None else ("converged" if self.converged else "not converged")
        lines.append(f"Fit status: {status}. {self.message}".rstrip())
        return "\n".join(lines)

    def summary(self) -> str:
        """Print and return a formatted summary of all six coefficients."""
        text = self._format("CR Hamiltonian")
        print(text)
        return text

    def __repr__(self) -> str:
        return self._format("CRHamiltonianResult")


@dataclass(frozen=True)
class CRSusceptibilityResult:
    r"""Store weak-drive CR coefficients per unit of programmed drive amplitude.

    For control-state-conditioned target-transition matrix elements

    .. math::

        m_z = \langle\widetilde{z_c,1_t}|D_c|\widetilde{z_c,0_t}\rangle,

    the effective-Hamiltonian convention

    .. math::

        H_\mathrm{eff} = \tfrac{1}{2}(IX\,I\!X + ZX\,Z\!X)

    gives ``IX_per_amplitude = m_0 + m_1`` and
    ``ZX_per_amplitude = m_0 - m_1``. The values are backend-native complex
    scalars and stay JAX-traceable.

    Attributes
    ----------
    m_control_0, m_control_1
        Dressed target-transition matrix elements, conditioned on the control
        in ``|0>`` and in ``|1>``.
    IX_per_amplitude, ZX_per_amplitude
        Weak-drive Pauli coefficients per unit of signal amplitude.
    control, target, drive
        Resolved labels.
    """

    m_control_0: Any
    m_control_1: Any
    IX_per_amplitude: Any
    ZX_per_amplitude: Any
    control: str
    target: str
    drive: str


def analyze_cr_susceptibility(
    chip: "Chip",
    control: str | "BaseDevice",
    target: str | "BaseDevice",
    *,
    drive: str | "BaseDrive" | None = None,
) -> CRSusceptibilityResult:
    r"""Return the dressed weak-drive CR response of one directed edge.

    The analysis projects the physical control-line operator onto the target's
    dressed ``0 -> 1`` transition twice, with the control in ``|0>`` and in
    ``|1>``. It does no pulse, no rotating-frame solve, and no time evolution.

    Parameters
    ----------
    chip
        Coupled chip with attached control equipment.
    control, target
        Directed CR control and target, as device objects or labels.
    drive
        Control line to project. If omitted, the function selects the unique
        wired device-target line attached to ``control``.

    Returns
    -------
    CRSusceptibilityResult
        Conditional matrix elements and the related ``IX`` and ``ZX``
        coefficients per unit of programmed amplitude.

    Raises
    ------
    ValueError
        If the edge is a self-edge, control equipment is absent, implicit
        drive resolution is ambiguous, or the selected line does not target
        the control.

    References
    ----------
    Magesan and Gambetta, Phys. Rev. A 101, 052308 (2020).
    Malekakhlagh, Magesan, and McKay, Phys. Rev. A 102, 042605 (2020).
    """
    _, control_device = chip._resolve_device_index(control)
    _, target_device = chip._resolve_device_index(target)
    if control_device is target_device:
        raise ValueError("control and target must be different devices")

    equipment = chip.control_equipment
    if equipment is None:
        raise ValueError("analyze_cr_susceptibility requires attached control equipment")

    candidates = [
        line
        for line in equipment.lines
        if line.device_label == control_device.label
    ]
    if drive is None:
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one drive targeting '{control_device.label}', found {len(candidates)}"
            )
        selected = candidates[0]
    else:
        selected_label = resolve_label(drive)
        matches = [line for line in equipment.lines if line.label == selected_label]
        if len(matches) != 1:
            raise ValueError(f"Drive '{selected_label}' is absent or ambiguous")
        selected = matches[0]
        if selected not in candidates:
            raise ValueError(
                f"Drive '{selected.label}' does not target control '{control_device.label}'"
            )

    m0 = chip.drive_matrix_elements(({}, {target_device: 1}), drives=[selected])[selected]
    m1 = chip.drive_matrix_elements(
        ({control_device: 1}, {control_device: 1, target_device: 1}),
        drives=[selected],
    )[selected]
    return CRSusceptibilityResult(
        m_control_0=m0,
        m_control_1=m1,
        IX_per_amplitude=m0 + m1,
        ZX_per_amplitude=m0 - m1,
        control=control_device.label,
        target=target_device.label,
        drive=selected.label,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _as_xyz(data: dict[str, np.ndarray] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, y, z)`` Bloch-component arrays from a dict or 2D array.

    Accepts either a mapping with keys ``"x"``, ``"y"``, ``"z"`` or a 2D
    ndarray shaped ``(3, N)`` (rows are X, Y, Z) or ``(N, 3)`` (columns are
    X, Y, Z).
    """
    if isinstance(data, dict):
        try:
            return (np.asarray(data["x"], dtype=float),
                    np.asarray(data["y"], dtype=float),
                    np.asarray(data["z"], dtype=float))
        except KeyError as exc:
            raise ValueError(
                f"Bloch dict must have keys 'x', 'y', 'z'; got {sorted(data)}"
            ) from exc
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == 3:
        return arr[0], arr[1], arr[2]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr[:, 0], arr[:, 1], arr[:, 2]
    raise ValueError(
        "Bloch data must be a dict with keys 'x', 'y', 'z' or a 2D array "
        f"shaped (3, N) or (N, 3); got array of shape {arr.shape}"
    )


def analyze_cross_resonance(
    durations: np.ndarray,
    ctrl0: dict[str, np.ndarray] | np.ndarray,
    ctrl1: dict[str, np.ndarray] | np.ndarray,
    sigma_ctrl0: dict[str, np.ndarray] | None = None,
    sigma_ctrl1: dict[str, np.ndarray] | None = None,
    t_offset: float = 0.0,
) -> CRHamiltonianResult:
    """Extract CR effective Hamiltonian coefficients from Bloch tomography data.

    Fits the target qubit's Bloch trajectory under a CR pulse to the analytic
    model in :func:`bloch_model`, once with the control qubit in |0⟩ and once
    in |1⟩. Combining the two fits isolates the six coefficients {IX, IY, IZ,
    ZX, ZY, ZZ}.

    The decomposition is::

        p_ctrl0 = (I/2) part → ω_IX, ω_IY, ω_IZ
        p_ctrl1 = (Z/2) part → ω_ZX, ω_ZY, ω_ZZ

    Specifically:

        ω_ZQ = (p_ctrl0_Q - p_ctrl1_Q) / 2
        ω_IQ = (p_ctrl0_Q + p_ctrl1_Q) / 2

    for Q ∈ {X, Y, Z}.

    Parameters
    ----------
    durations : (N,) ndarray
        CR pulse durations in **ns** (the package convention). They must be
        monotone but need not be equally spaced. The function converts them to
        seconds once, at its boundary, for the fit.
    ctrl0 : dict or 2D ndarray
        Target qubit's Bloch trajectory with the control in |0⟩. Give a dict
        with keys ``"x"``, ``"y"``, ``"z"`` or a 2D array with shape ``(3, N)``
        or ``(N, 3)`` (see :func:`_as_xyz`).
    ctrl1 : dict or 2D ndarray
        Same, with the control in |1⟩.
    sigma_ctrl0 : dict with keys ``"x"``, ``"y"``, ``"z"`` (optional)
        Per-point standard deviations for the control-|0⟩ data, used as inverse
        weights in the least-squares fit.
    sigma_ctrl1 : dict with keys ``"x"``, ``"y"``, ``"z"`` (optional)
        Same for the control-|1⟩ data.
    t_offset : float
        Subtract this offset (in **ns**) from ``durations`` before the fit,
        e.g. to exclude a pulse-ramp transient.

    Returns
    -------
    CRHamiltonianResult
        Six coefficients in **GHz** (the package's units contract). The function converts
        the Hz-valued fit internals exactly once, at this boundary. The uncertainties are
        local, residual-scaled covariance estimates, even when per-point standard
        deviations supply weights. An unavailable one-sigma uncertainty is ``None``.
        ``converged`` and ``message`` report the optimizer outcome separately from the fit
        quality and the local uncertainty, so you can still examine nonconverged
        candidates.

    Raises
    ------
    RuntimeError
        If no initial guess gives a finite candidate for one of the control states.

    Examples
    --------
    >>> import numpy as np
    >>> from quchip import analyze_cross_resonance
    >>> durations = np.linspace(0, 400, 20)  # ns
    >>> ctrl0 = {"x": np.sin(0.02 * durations), "y": np.zeros_like(durations),
    ...          "z": np.cos(0.02 * durations)}
    >>> ctrl1 = {"x": np.sin(0.03 * durations), "y": np.zeros_like(durations),
    ...          "z": np.cos(0.03 * durations)}
    >>> result = analyze_cross_resonance(durations, ctrl0, ctrl1)
    >>> zx, zx_err = result.coeffs()["ZX"]  # GHz
    """
    # Boundary conversion: public durations/t_offset are in ns; the fit
    # heuristics (frequency guesses, td bounds) are Hz/seconds-valued, so
    # convert once here and leave every fit internal untouched.
    te = (np.asarray(durations, dtype=float) - t_offset) * 1e-9
    x0, y0, z0 = _as_xyz(ctrl0)
    x1, y1, z1 = _as_xyz(ctrl1)
    s0 = sigma_ctrl0 or {}
    s1 = sigma_ctrl1 or {}

    fit0, cov0 = _fit_bloch(te, x0, y0, z0, s0.get("x"), s0.get("y"), s0.get("z"))
    fit1, cov1 = _fit_bloch(te, x1, y1, z1, s1.get("x"), s1.get("y"), s1.get("z"))

    p0, p1 = fit0.x, fit1.x
    hz_to_ghz = 1e-9

    def _coeff(i):
        return (p0[i] - p1[i]) / 2, (p0[i] + p1[i]) / 2

    def _err(i):
        if cov0 is None or cov1 is None:
            return None
        return np.sqrt(cov0[i, i] + cov1[i, i]) / 2 * hz_to_ghz

    ZX, IX = _coeff(0)
    ZY, IY = _coeff(1)
    ZZ, IZ = _coeff(2)

    return CRHamiltonianResult(
        IX=IX * hz_to_ghz, IY=IY * hz_to_ghz, IZ=IZ * hz_to_ghz,
        ZX=ZX * hz_to_ghz, ZY=ZY * hz_to_ghz, ZZ=ZZ * hz_to_ghz,
        IX_err=_err(0), IY_err=_err(1), IZ_err=_err(2),
        ZX_err=_err(0), ZY_err=_err(1), ZZ_err=_err(2),
        params_ctrl0=p0,
        params_ctrl1=p1,
        cov_ctrl0=cov0,
        cov_ctrl1=cov1,
        cost_ctrl0=fit0.cost,
        cost_ctrl1=fit1.cost,
        converged=bool(fit0.success and fit1.success),
        message=f"Control 0: {fit0.message}; control 1: {fit1.message}",
    )
