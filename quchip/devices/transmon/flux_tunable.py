"""Symmetric/asymmetric-SQUID flux-tunable transmon.

``freq`` and ``anharmonicity`` are the simulated physics: the calibrated
local ``0 -> 1`` transition frequency and anharmonicity of the device *at
the stored* ``flux_bias``. :meth:`FluxTunableTransmon.local_hamiltonian`
builds the Duffing Hamiltonian from ``freq`` and ``anharmonicity`` alone —
it does not reference ``flux_bias``. :meth:`FluxTunableTransmon.frequency_at`
and :meth:`FluxTunableTransmon.flux_for_frequency` answer counterfactual
questions — what frequency the SQUID would reach at a different bias — via
the SQUID dispersion relative to this calibrated anchor. Mutating
``flux_bias`` therefore records a new nominal operating point without
retuning the device: it does not change ``freq``, ``anharmonicity``, or the
Hamiltonian.

**Declared approximations**

* Transmon regime: E_J ≫ E_C (exponential charge-dispersion suppression).
* Duffing truncation: the cosine Josephson potential is expanded to quartic
  order; anharmonicity α ≈ −E_C.
* Adiabatic flux: the SQUID dispersion underlying :meth:`frequency_at` /
  :meth:`flux_for_frequency` is a static calibration-anchor relation with no
  Landau–Zener physics. Time-dependent flux tuning during gates is applied
  through an external :class:`~quchip.control.drive.FluxDrive` whose
  real-baseband envelope carries δω(t) in GHz.

**SQUID dispersion**

.. math::

    E_J(\\Phi) = E_{J,\\max}
        \\sqrt{\\cos^2(\\pi \\Phi/\\Phi_0) + d^2 \\sin^2(\\pi \\Phi/\\Phi_0)}

    \\omega(\\Phi) = \\sqrt{8\\, E_C\\, E_J(\\Phi)} - E_C

where ``d = (E_{J1} − E_{J2}) / (E_{J1} + E_{J2})`` is the junction asymmetry
and Φ/Φ₀ is the reduced flux.  The user supplies the calibrated local
``freq`` and ``anharmonicity``; the Josephson parameters are derived
internally:

    α = −E_C  →  E_C = |α|
    (ω + E_C)² = 8 E_C E_J(flux_bias)  →  E_J_max via SQUID inversion

The SQUID parameters :attr:`_E_C` / :attr:`_E_J_max` are *derived on read* from
the current ``freq`` / ``anharmonicity`` / ``flux_bias`` / ``asymmetry`` — they
carry no cached state, so :meth:`frequency_at` and :meth:`flux_for_frequency`
always reflect a mutated or swept parameter (no stale SQUID metadata).

References
----------
* Koch et al., PRA 76, 042319 (2007), §II.
* Krantz et al., APR 6, 021318 (2019), §II.B and §V.A.
* Renger et al., *A superconducting qubit-resonator quantum processor with
  effective all-to-all connectivity* — flux-tunable qubits and MOVE/CZ gates.

Examples
--------
>>> from quchip.devices.transmon import FluxTunableTransmon
>>> q = FluxTunableTransmon(freq=4.47, anharmonicity=-0.2006, levels=3)
>>> round(q.freq, 3)
4.47
>>> round(q.anharmonicity, 4)
-0.2006
>>> round(float(q.frequency_at(0.0)), 3)
4.47
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.models import DeviceModel
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import Scalar, parameter
from quchip.devices.transmon.duffing import duffing_expr
from quchip.utils.jax_utils import maybe_concrete_scalar


def _check_anharmonicity(value: Any) -> None:
    """Raise ``ValueError`` if *value* is a concrete non-negative scalar.

    ``anharmonicity`` must be strictly negative: ``E_C = |anharmonicity|``
    feeds a division in :attr:`FluxTunableTransmon._E_J_max`, so zero
    divides by zero, and a positive value is unphysical for a transmon
    (α ≈ −E_C < 0). Concrete scalars only; traced values pass unchecked.
    """
    concrete = maybe_concrete_scalar(value)
    if concrete is not None and concrete >= 0.0:
        raise ValueError(
            f"anharmonicity must be negative (E_C = |anharmonicity| feeds the SQUID "
            f"inversion and must be positive), got {value}"
        )


def _check_asymmetry(value: Any) -> None:
    """Raise ``ValueError`` if *value* is a concrete scalar outside ``[0, 1)``."""
    concrete = maybe_concrete_scalar(value)
    if concrete is not None and not (0.0 <= concrete < 1.0):
        raise ValueError(f"asymmetry must be in [0, 1), got {value}")


def _check_flux_bias_dispersion(flux_bias: Any, asymmetry: Any) -> None:
    """Raise ``ValueError`` if the SQUID dispersion factor is concretely zero.

    ``flux_bias`` itself is unrestricted — any real value is a valid
    calibration anchor. The dispersion factor
    ``sqrt(cos²(π·flux_bias) + asymmetry²·sin²(π·flux_bias))`` vanishes only
    at the joint condition ``asymmetry == 0`` and ``flux_bias`` a
    half-integer (a symmetric SQUID's effective E_J is exactly zero at half
    flux quantum), which makes :attr:`FluxTunableTransmon._E_J_max` diverge.
    Concrete scalars only; either input being traced skips the check.
    """
    fb = maybe_concrete_scalar(flux_bias)
    d = maybe_concrete_scalar(asymmetry)
    if fb is None or d is None:
        return
    dispersion_factor = math.sqrt(math.cos(math.pi * fb) ** 2 + d**2 * math.sin(math.pi * fb) ** 2)
    if dispersion_factor < 1e-9:
        raise ValueError(
            f"flux_bias={flux_bias} with asymmetry={asymmetry} drives the SQUID's effective "
            "E_J to zero; no finite E_J_max reproduces a positive freq there. Choose a "
            "different flux_bias or a nonzero asymmetry."
        )


class FluxTunableTransmon(DeviceModel):
    """SQUID-dispersion flux-tunable transmon.

    The constructor takes the calibrated local physical parameters; SQUID
    metadata is derived on read and is not part of the public interface.

    Parameters
    ----------
    freq : float
        Calibrated local ``0 -> 1`` transition frequency ω in GHz, at the
        stored ``flux_bias``. Must be positive. May be a JAX tracer.
    anharmonicity : float
        Calibrated local anharmonicity α in GHz, at the stored
        ``flux_bias``. Must be negative (α ≈ −E_C). May be a JAX tracer.
    flux_bias : float, default 0.0
        Calibration-anchor operating point Φ/Φ₀. Any real value; the SQUID
        inversion is undefined only at the symmetric-SQUID degenerate point
        (``asymmetry == 0`` and ``flux_bias`` a half-integer — see
        :meth:`validate`). The local Hamiltonian does not reference this
        value directly — ``freq`` and ``anharmonicity`` already carry it. A
        pytree leaf, so it is differentiable / sweepable like every other
        device parameter.
    asymmetry : float, default 0.0
        SQUID junction asymmetry d = (E_{J1}−E_{J2})/(E_{J1}+E_{J2}).
        Must be in [0, 1).
    levels : int, default 3
        Fock-space truncation.
    label : str | None, default None
        Auto-generated as ``fluxtunable_{idx}`` when omitted.
    **noise_kwargs
        Forwarded to :class:`~quchip.devices.base.BaseDevice` — ``T1``,
        ``T2``, ``thermal_population``.
    """

    _type_prefix: str = "fluxtunable"
    _default_levels: int = 3
    tunable_param_names = ("freq", "anharmonicity")
    computational = True
    approximation = (
        "Duffing-approximated SQUID transmon; adiabatic flux (calibration-anchor, "
        "no Landau-Zener)."
    )

    freq: Scalar = parameter(positive=True, unit="GHz")
    anharmonicity: Scalar = parameter(unit="GHz")
    flux_bias: Scalar = parameter(default=0.0, unit="Phi_0")
    asymmetry: Scalar = parameter(default=0.0)

    # --- generated __init__ stub (tools/gen_device_stubs.py); do not edit ---
    if TYPE_CHECKING:
        def __init__(
            self,
            freq: Scalar,
            anharmonicity: Scalar,
            flux_bias: Scalar = 0.0,
            asymmetry: Scalar = 0.0,
            *,
            levels: int = 3,
            label: str | None = None,
            T1: float | None = None,
            T2: float | None = None,
            thermal_population: float | None = None,
        ) -> None: ...
    # --- end generated stub ---

    def validate(self) -> None:
        """Range checks on concrete scalars only; traced values pass unchecked."""
        _check_anharmonicity(self.anharmonicity)
        _check_asymmetry(self.asymmetry)
        _check_flux_bias_dispersion(self.flux_bias, self.asymmetry)

    def _validate_param_write(self, name: str, value: Any) -> None:
        """Re-run the construction-time SQUID checks on relevant post-construction writes.

        Mirrors :meth:`validate`, substituting *value* for the field being
        written so e.g. ``q.anharmonicity = 0.0`` after construction fails
        exactly as it would at construction.
        """
        super()._validate_param_write(name, value)
        if name == "anharmonicity":
            _check_anharmonicity(value)
        elif name == "asymmetry":
            _check_asymmetry(value)
            _check_flux_bias_dispersion(self.flux_bias, value)
        elif name == "flux_bias":
            _check_flux_bias_dispersion(value, self.asymmetry)

    def local_hamiltonian(self, op: LocalOps) -> PhysicsExpr:
        """Return the Duffing Hamiltonian built from the calibrated freq and anharmonicity.

        ``H = ω n + (α/2) n(n − I)``. Does not reference ``flux_bias``.
        """
        return duffing_expr(op, self.freq, self.anharmonicity)

    # -- Derived-on-read SQUID parameters ----------------------------------

    @property
    def _E_C(self) -> Any:
        """Charging energy E_C = |α|, recomputed from the current anharmonicity."""
        return jnp.abs(jnp.asarray(self.anharmonicity))

    @property
    def _E_J_max(self) -> Any:
        """Maximum Josephson energy, inverted from ``freq`` at the current bias.

        ``E_C = |α|``, ``E_J(flux_bias) = (ω + E_C)² / (8 E_C)``, then
        ``E_J_max = E_J(flux_bias) / sqrt(cos²(πΦ) + d²sin²(πΦ))``.
        """
        E_C = self._E_C
        E_J_at_bias = (jnp.asarray(self.freq) + E_C) ** 2 / (8.0 * E_C)
        phi = jnp.pi * jnp.asarray(self.flux_bias)
        d = jnp.asarray(self.asymmetry)
        dispersion_factor = jnp.sqrt(jnp.cos(phi) ** 2 + d ** 2 * jnp.sin(phi) ** 2)
        return E_J_at_bias / dispersion_factor

    def frequency_at(self, flux: Any) -> Any:
        """SQUID dispersion ω(Φ/Φ₀) in GHz, using derived E_C and E_J_max.

        Parameters
        ----------
        flux : float
            Reduced flux Φ/Φ₀. JAX-traceable.
        """
        phi = jnp.pi * jnp.asarray(flux)
        d = jnp.asarray(self.asymmetry)
        E_J = self._E_J_max * jnp.sqrt(jnp.cos(phi) ** 2 + d ** 2 * jnp.sin(phi) ** 2)
        return jnp.sqrt(8.0 * self._E_C * E_J) - self._E_C

    def flux_for_frequency(self, target_freq: Any) -> Any:
        """Inverse SQUID dispersion on the monotonic lobe Φ/Φ₀ ∈ [0, 0.5).

        Derivation:
            ω(Φ) = sqrt(8 E_C E_J_max sqrt(cos²(πΦ) + d²sin²(πΦ))) − E_C
            → let S = (ω + E_C)² / (8 E_C E_J_max)
            → cos²(πΦ)(1 − d²) + d² = S²
            → cos²(πΦ) = (S² − d²) / (1 − d²)

        Raises
        ------
        ValueError
            If *target_freq* is concrete and lands outside the frequency
            range :meth:`frequency_at` reaches over Φ/Φ₀ ∈ [0, 0.5) at the
            current calibration anchor. A traced *target_freq* (or a traced
            anchor) skips this check; the returned flux clips to the lobe
            endpoint, so out-of-domain behavior is undefined for traced
            inputs.
        """
        # Validate the concrete target against the attainable endpoint span
        # BEFORE forming S: squaring (omega + E_C) would otherwise map a
        # below-minimum target onto an attainable S and silently invert to
        # the wrong frequency.
        target_concrete = maybe_concrete_scalar(target_freq)
        if target_concrete is not None:
            lo_c = maybe_concrete_scalar(self.frequency_at(0.5))
            hi_c = maybe_concrete_scalar(self.frequency_at(0.0))
            if lo_c is not None and hi_c is not None:
                lo, hi = sorted((lo_c, hi_c))
                if not (lo <= target_concrete <= hi):
                    raise ValueError(
                        f"target_freq={target_freq} GHz is unattainable at the current "
                        f"calibration anchor: frequency_at spans [{lo:.6g}, {hi:.6g}] GHz "
                        "over Φ/Φ₀ ∈ [0, 0.5]."
                    )
        omega = jnp.asarray(target_freq)
        S = (omega + self._E_C) ** 2 / (8.0 * self._E_C * self._E_J_max)
        d2 = jnp.asarray(self.asymmetry) ** 2
        cos2 = (S ** 2 - d2) / (1.0 - d2)
        return jnp.arccos(jnp.sqrt(jnp.clip(cos2, 0.0, 1.0))) / jnp.pi

    def physics_notes(self) -> list[str]:
        """Return declared SQUID-transmon calibration-anchor assumptions."""
        notes = super().physics_notes()
        notes.append("Time-varying flux enters via an external FluxDrive (δω n̂ modulation)")
        return notes
