"""Calculate steady-state dispersive pointer states and detected information from a chip.

The inputs come from the chip. The dressed spectrum gives the conditional
resonator frequencies ``f_r|j`` with the qubit in level ``j``. ``kappa`` is the
energy linewidth of the resonator pole of the weak-probe drift
``A = -iH - C†C/2``, in which ``C`` keeps only the energy-loss channels, so it
omits dephasing. ``kappa_detected`` is the part of ``kappa`` that leaves
through the selected output port, before detector efficiency.

The model is one linear resonator mode with vacuum inputs and a fixed qubit
state. For the drive detuning ``delta_j = 2pi*(delta_r + f_r|j - f_r|0)``,
``alpha_j = -i*eps/(kappa/2 + i*delta_j)``. The ground-state photon number is
``nbar_0 = |alpha_0|²``. The dephasing rate is
``Gamma_m = kappa*|alpha_1-alpha_0|²/2``. Heterodyne detection gives
``SNR = |alpha_1-alpha_0|*sqrt(2*efficiency*kappa_detected*tau)``.

``chi`` is the full pull ``f_r|1-f_r|0`` in GHz. With
``Z=|0><0|-|1><1|``, the angular coefficient of ``Z*n_r`` is ``-pi*chi``.
The small-pull limit of ``Gamma_m`` is ``8*(pi*chi)²*nbar_0/kappa`` divided by
``1+(4pi*delta_r/kappa)²``. It reduces to the usual expression on resonance.

The pulls are the zero-photon pulls. A nonlinearity diagnostic reports how
much the pulls change over the photon numbers that the pointer states occupy.

References
----------
Koch et al., PRA 76, 042319 (2007), section III B:
https://arxiv.org/abs/cond-mat/0703002.
Gambetta et al., PRA 74, 042318 (2006):
https://arxiv.org/abs/cond-mat/0602322.
Krantz et al., Applied Physics Reviews 6, 021318 (2019), section V:
https://arxiv.org/abs/1904.06560.
Blais et al., RMP 93, 025005 (2021), section III.3:
https://arxiv.org/abs/2005.12667.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import erfc as _scipy_erfc
from scipy.stats import poisson

from quchip.analysis.vna import _PortTransfers, _resolve_exposure
from quchip.devices.base import BaseDevice
from quchip.engine.linear_response import response_poles, try_build_weak_probe_problem
from quchip.engine.routes import weak_probe_reason
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import concrete_array_module, contains_tracer, is_jax_namespace, raise_if

if TYPE_CHECKING:
    from quchip.chip.chip import Chip

# Poisson probability left outside the photon numbers that the nonlinearity diagnostic samples.
_TAIL_TOLERANCE = 1e-3


def _erfc(x: Any, xp: Any) -> Any:
    """Complementary error function in the given array namespace."""
    if is_jax_namespace(xp):
        from jax.scipy.special import erfc

        return erfc(x)
    return _scipy_erfc(x)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispersiveReadoutResult:
    """Store steady-state dispersive-readout figures of merit and the chip inputs they use.

    The numeric fields stay differentiable under JAX on a fixed structure,
    except ``kappa_detected`` with an ``output`` (see
    :func:`analyze_dispersive_readout`).

    ``nonlinearity`` reports how the pulls change with the photon number. Its
    ``status`` is ``"checked"``, or ``"unchecked"`` with a ``reason``. A
    checked result also holds the ``tail_tolerance``, the sampled
    ``photon_ranges`` of each pointer state, ``frequency_drift``,
    ``drift_over_hwhm`` and ``pull_change``. ``frequency_drift`` is the largest
    ``|f_r|j(n) - f_r|j(0)|`` in GHz over the range of each pointer state, and
    ``drift_over_hwhm`` divides it by the half width ``kappa/4π``.
    ``pull_change`` is the largest ``|f_r|1(n) - f_r|0(n) - chi|/|chi|`` over
    the ranges of levels 0 and 1, or ``None`` when ``chi`` is zero. The
    diagnostic has no pass threshold and does not certify validity.

    Attributes
    ----------
    pointer_states
        Complex ``α_j`` in the IQ plane, shape ``(levels,)``.
    photon_numbers
        Steady-state photons ``|α_j|²``, shape ``(levels,)``.
    sigma
        Integrated vacuum-noise width in the cavity amplitude plane. It is
        ``1/sqrt(2*efficiency*kappa_detected*tau)``, or infinity without detection.
    snr
        Pointer separation divided by ``sigma``.
    assignment_error
        ``½·erfc(SNR/(2√2))``, the optimal linear discriminant between two
        equal Gaussians.
    dephasing_rate
        Measurement-induced dephasing ``Γ_m = κ·|α₁ − α₀|²/2`` in 1/ns.
    resonator_frequencies
        Dressed resonator frequencies ``f_r|j`` in GHz with the qubit in level
        ``j``, shape ``(levels,)``.
    chi
        Full pull ``f_r|1 − f_r|0`` in GHz.
    kappa
        Energy linewidth of the resonator pole in 1/ns, without dephasing.
    kappa_detected
        The part of ``kappa`` that leaves through the output port in 1/ns, or
        ``kappa`` without an output.
    nonlinearity
        See above.
    notes
        Assumptions of the closed form.
    """

    pointer_states: Any
    photon_numbers: Any
    sigma: Any
    snr: Any
    assignment_error: Any
    dephasing_rate: Any
    resonator_frequencies: Any
    chi: Any
    kappa: Any
    kappa_detected: Any
    nonlinearity: dict[str, Any]
    notes: list[str]

    def summary(self) -> str:
        """Print and return a formatted summary.

        Concrete values only. Call it outside ``jax.jit``/``grad`` regions.
        """
        alphas = np.asarray(self.pointer_states)
        photons = np.asarray(self.photon_numbers)
        lines = ["Dispersive readout (steady state):"]
        lines.append(f"  chi = {float(self.chi) * 1e3:+.4f} MHz, kappa = {float(self.kappa):.4e} 1/ns, "
                     f"kappa_detected = {float(self.kappa_detected):.4e} 1/ns")
        lines.append("  pointer states α_j = " + ", ".join(f"{a.real:+.4f}{a.imag:+.4f}j" for a in alphas))
        lines.append("  photons n̄_j = " + ", ".join(f"{n:.4f}" for n in photons))
        lines.append(f"  SNR = {float(self.snr):.4f}  (blob width σ = {float(self.sigma):.4f})")
        lines.append(f"  assignment error = {float(self.assignment_error):.3e}")
        lines.append(f"  dephasing Γ_m = {float(self.dephasing_rate):.3e} 1/ns")
        if self.nonlinearity["status"] == "checked":
            drift = ", ".join(f"{value:.3g}" for value in np.asarray(self.nonlinearity["drift_over_hwhm"]))
            lines.append(f"  pull drift over the pointer photons = {drift} half widths")
        else:
            lines.append(f"  nonlinearity unchecked: {self.nonlinearity['reason']}")
        text = "\n".join(lines)
        print(text)
        return text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_dispersive_readout(
    chip: "Chip",
    qubit: str | BaseDevice,
    resonator: str | BaseDevice,
    *,
    tau: Any,
    n_photons: Any = None,
    eps: Any = None,
    delta_r: Any = 0.0,
    output: Any = None,
    efficiency: Any = 1.0,
    levels: int = 2,
) -> DispersiveReadoutResult:
    """Calculate steady-state readout figures of merit from the chip's pulls and linewidth.

    Parameters
    ----------
    chip
        Chip with the qubit and its readout resonator. The linewidth needs a
        weak-probe route of :meth:`~quchip.analysis.vna.VNA.sweep`, so the
        static model must conserve the excitation number, as under ``RWA()``.
    qubit, resonator
        The qubit and the resonator, as devices or labels.
    tau
        Positive integration time in ns.
    n_photons
        Nonnegative ground-state steady photon number, or None. Give exactly
        one of ``n_photons`` and ``eps``. This sets
        ``eps=sqrt(n_photons*((kappa/2)**2+(2*pi*delta_r)**2))``.
    eps
        Complex drive rate in rad/ns, or None. Give exactly one of ``eps`` and
        ``n_photons``. It is the coefficient driving the amplitude of the
        resonator mode.
    delta_r
        Real ``f_r|0-f_drive`` in GHz. The default zero drives the ground-state
        resonator frequency. Positive values place the drive below it.
    output
        Exposed port of the chip's ``PortNetwork``, as a port or a label, or
        None. With a port, ``kappa_detected`` is the part of ``kappa`` that
        reaches it, ``|t_out|²|c_out v|²/|v|²`` for the right eigenvector
        ``v`` of the pole, the row ``c_out`` of the port's output field, and
        the outbound transfer ``t_out`` of its reference plane at the pole.
        A JAX derivative through it raises, because it needs the eigenvector.
        None gives ``kappa_detected = kappa``, complete collection.
    efficiency
        Detection efficiency between zero and one, default one. Zero gives
        zero SNR, infinite ``sigma``, and assignment error one half.
    levels
        Static Python integer of at least two, default two. It sets the number
        of pointer states, each from the dressed pull of its qubit level.

    Returns
    -------
    DispersiveReadoutResult

    Raises
    ------
    ValueError
        If drive arguments are ambiguous, ``levels`` is invalid, concrete
        numerical values are nonfinite or outside their stated domains, the
        chip has no weak-probe route, or the pole nearest ``f_r|0`` is not the
        resonator's. Numerical domain checks are skipped for tracers. Callers
        must keep traced inputs within the same physical domains.

    Examples
    --------
    >>> from quchip import RWA, Capacitive, Chip, DuffingTransmon, PortNetwork, Resonator
    >>> from quchip import analyze_dispersive_readout
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
    >>> r = Resonator(freq=6.5, levels=10, label="r")
    >>> line = PortNetwork(label="line")
    >>> port = line.port("r_port", target=r, rate=0.005)
    >>> probe = line.expose("p", input=port.input, output=port.output)
    >>> chip = Chip([q, r], [Capacitive(q, r, g=0.05)], port_network=line, approximation=RWA())
    >>> ro = analyze_dispersive_readout(chip, q, r, tau=500.0, n_photons=2.0, output=probe)
    >>> snr, p_err = ro.snr, ro.assignment_error
    """
    if (n_photons is None) == (eps is None):
        raise ValueError("Provide exactly one of n_photons (target photons) or eps (drive rate, rad/ns).")
    if not isinstance(levels, int) or isinstance(levels, bool) or levels < 2:
        raise ValueError("levels must be a Python integer of at least two.")
    for name, value, positive in (("tau", tau, True), ("n_photons", n_photons, False),
                                  ("efficiency", efficiency, False)):
        if value is None or contains_tracer(value):
            continue
        array = np.asarray(value)
        if not np.isrealobj(array) or np.any(~np.isfinite(array)) or np.any(array <= 0 if positive else array < 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
    if not contains_tracer(efficiency) and np.any(np.asarray(efficiency) > 1):
        raise ValueError("efficiency must not exceed one.")
    for name, value in (("delta_r", delta_r), ("eps", eps)):
        if value is not None and not contains_tracer(value) and np.any(~np.isfinite(np.asarray(value))):
            raise ValueError(f"{name} must be finite.")
    _, qubit_device = chip._resolve_device_index(qubit)
    _, resonator_device = chip._resolve_device_index(resonator)
    if qubit_device.label == resonator_device.label:
        raise ValueError("The qubit and the resonator must be different devices.")

    frequencies = [chip.transition_frequency(resonator_device, 0, 1, when={qubit_device: j}) for j in range(levels)]
    xp: Any = concrete_array_module(frequencies)
    frequencies = xp.stack([xp.asarray(value) for value in frequencies])
    kappa, kappa_detected = _resonator_linewidths(chip, resonator_device.label, frequencies[0], output)

    given = (frequencies, kappa, kappa_detected, tau, n_photons, eps, delta_r, efficiency)
    xp = concrete_array_module(*(value for value in given if value is not None))
    # The single GHz -> rad/ns boundary of this module (ordinary -> angular).
    delta_0 = TWO_PI * delta_r
    half_kappa = kappa / 2.0
    if eps is None:
        eps = xp.sqrt(n_photons * (half_kappa**2 + delta_0**2))
    delta_j = TWO_PI * (delta_r + frequencies - frequencies[0])
    alpha = -1j * eps / (half_kappa + 1j * delta_j)
    photons = xp.abs(alpha) ** 2

    separation = xp.abs(alpha[1] - alpha[0])
    collected = efficiency * kappa_detected
    snr = separation * xp.sqrt(2.0 * collected * tau)
    variance_rate = xp.where(collected > 0, 2.0 * collected * tau, 1.0)
    sigma = xp.where(collected > 0, 1.0 / xp.sqrt(variance_rate), xp.inf)
    notes = [
        "Steady-state coherent pointer states of one isolated resonator mode with vacuum inputs; "
        "ring-up transients are not modeled.",
        "The damping is weak compared with the mode spacings.",
        "kappa omits the channels that conserve the excitation number, such as dephasing, "
        "which is assumed negligible.",
        "One kappa, with the qubit in level 0, serves every pointer state, and the output coupling "
        "does not depend on the qubit state.",
        "The pulls are the zero-photon pulls of the chip's Hamiltonian approximation and finite basis.",
        "Ordinary, Purcell and measurement-induced qubit decay during the acquisition are not modeled.",
        "The SNR and the assignment error compare levels 0 and 1 only.",
    ]
    return DispersiveReadoutResult(
        pointer_states=alpha,
        photon_numbers=photons,
        sigma=sigma,
        snr=snr,
        assignment_error=0.5 * _erfc(snr / (2.0 * np.sqrt(2.0)), xp),
        dephasing_rate=kappa * separation**2 / 2.0,
        resonator_frequencies=frequencies,
        chi=frequencies[1] - frequencies[0],
        kappa=kappa,
        kappa_detected=kappa_detected,
        nonlinearity=_nonlinearity(chip, qubit_device.label, resonator_device.label, photons, kappa, levels),
        notes=notes,
    )


def _resonator_linewidths(chip: "Chip", resonator: str, frequency: Any, output: Any) -> tuple[Any, Any]:
    """Return the energy linewidth of the resonator pole and the part that reaches ``output``, in 1/ns.

    The pole is the eigenvalue of the dephasing-free weak-probe drift nearest
    ``frequency``, and the resonator must dominate its right eigenvector.
    """
    planes = () if output is None else (_resolve_exposure(chip, output),)
    built = try_build_weak_probe_problem(chip, np.zeros(1), plane_labels=planes, conserving=False)
    if built is None:
        raise ValueError("analyze_dispersive_readout() takes kappa from a weak-probe route of VNA.sweep(). "
                         f"{weak_probe_reason(chip, planes)}")
    problem = built[0]
    system = response_poles(problem)
    xp = concrete_array_module(system.values, frequency)
    pole_frequencies = -xp.imag(system.values) / TWO_PI
    index = xp.argmin(xp.abs(pole_frequencies - frequency))
    mode = problem.mode_labels.index(resonator)
    dominant = xp.argmax(xp.abs(system.vectors[:, index]) ** 2)
    raise_if(dominant != mode, partial(_shared_pole, resonator, problem.mode_labels), pole_frequencies[index], dominant)
    kappa = -2.0 * xp.real(system.values[index])
    if not contains_tracer(kappa) and not kappa > 0:
        raise ValueError(f"The pole of resonator {resonator!r} has no energy loss. Give the resonator a port "
                         "or an internal loss.")
    if output is None:
        return kappa, kappa
    vectors, _ = system.modes()
    vector = vectors[:, index]
    row = xp.asarray(problem.couplings, dtype=complex)[problem.plane_indices[0]]
    emitted = xp.abs(row @ vector) ** 2 / xp.sum(xp.abs(vector) ** 2)
    outbound, _ = _PortTransfers(chip, planes)(xp.reshape(pole_frequencies[index], (1,)))
    return kappa, xp.abs(outbound[0, 0]) ** 2 * emitted


def _shared_pole(resonator: str, labels: tuple[str, ...], frequency: Any, dominant: Any) -> ValueError:
    return ValueError(
        f"The weak-probe pole nearest the dressed frequency of {resonator!r}, at {float(frequency):.9g} GHz, is "
        f"dominated by {labels[int(dominant)]!r}. The readout needs an isolated resonator mode."
    )


def _nonlinearity(chip: "Chip", qubit: str, resonator: str, photons: Any, kappa: Any, levels: int) -> dict[str, Any]:
    """Compare the pulls over the photon numbers that the pointer states occupy with the zero-photon pulls."""
    if contains_tracer((photons, kappa)):
        return {"status": "unchecked", "reason": "The check needs concrete values. Run it outside jax.jit."}
    ranges = tuple((int(poisson.ppf(_TAIL_TOLERANCE / 2, mean)), int(poisson.ppf(1 - _TAIL_TOLERANCE / 2, mean)))
                   if mean > 0 else (0, 0) for mean in np.asarray(photons, dtype=float))
    top = max(high for _, high in ranges)
    index, _ = chip._resolve_device_index(resonator)
    dimension = chip.resolve().dims[index]
    # The transition from n = top keeps one resonator level above it.
    if top + 2 >= dimension:
        return {"status": "unchecked",
                "reason": f"The pointer states reach n = {top} photons, so the check needs resonator levels up to "
                          f"{top + 2}, and the resonator keeps {dimension} levels."}
    try:
        ladders = np.asarray([[float(chip.transition_frequency(resonator, n, n + 1, when={qubit: j}))
                               for n in range(top + 1)] for j in range(levels)])
    except ValueError as error:
        return {"status": "unchecked", "reason": str(error)}
    drift = np.asarray([np.max(np.abs(ladders[j, low:high + 1] - ladders[j, 0]))
                        for j, (low, high) in enumerate(ranges)])
    low, high = min(ranges[0][0], ranges[1][0]), max(ranges[0][1], ranges[1][1])
    chi = ladders[1, 0] - ladders[0, 0]
    pulls = ladders[1, low:high + 1] - ladders[0, low:high + 1]
    return {
        "status": "checked",
        "reason": None,
        "tail_tolerance": _TAIL_TOLERANCE,
        "photon_ranges": ranges,
        "frequency_drift": drift,
        "drift_over_hwhm": drift / (float(kappa) / (2 * TWO_PI)),
        "pull_change": None if chi == 0 else float(np.max(np.abs(pulls - chi)) / abs(chi)),
    }
