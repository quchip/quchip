r"""Charge-basis transmon — circuit-level transmon in the integer charge basis.

Hamiltonian (native basis, Koch et al. 2007):

.. math:: H = 4 E_C (\hat n - n_g)^2 - E_J \cos\hat\varphi

where :math:`\hat n` is diagonal on the integer charge basis
:math:`\{|{-n_\mathrm{cut}}\rangle, \ldots, |n_\mathrm{cut}\rangle\}`
and :math:`\cos\hat\varphi = \tfrac{1}{2}(|n\rangle\langle n+1| + \mathrm{h.c.})`.

Unlike :class:`~quchip.devices.transmon.duffing.DuffingTransmon`, the
spectrum is computed by exact diagonalization of the truncated charge
basis — no Duffing expansion. Charge-dispersion with :math:`n_g` is
captured exactly, which the Duffing expansion misses outside the deep
transmon regime :math:`E_J/E_C \gtrsim 50`.

References
----------
* Koch, Yu, Gambetta, Houck, Schuster, Majer, Blais, Devoret, Girvin &
  Schoelkopf, *Charge-insensitive qubit design derived from the
  Cooper pair box*, PRA **76**, 042319 (2007), §II.A–§II.C.
* Krantz et al., *A quantum engineer's guide to superconducting qubits*,
  APR **6**, 021318 (2019), §III.

Example
-------
>>> from quchip.devices.transmon import ChargeBasisTransmon
>>> q = ChargeBasisTransmon(E_C=0.25, E_J=20.0, levels=3)
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import jax.numpy as jnp

from quchip.devices.circuit import CircuitDevice, _check_positive_energy
from quchip.utils.jax_utils import maybe_concrete_scalar


class ChargeBasisTransmon(CircuitDevice):
    """Transmon in the integer charge basis — exact diagonalization.

    Parameters
    ----------
    E_C : float
        Charging energy in GHz. Must be positive. May be a JAX tracer.
    E_J : float
        Josephson energy in GHz. Must be positive. May be a JAX tracer.
    n_g : float, default ``0.0``
        Offset charge (units of :math:`2e`). May be a JAX tracer.
    levels : int, default ``3``
        Truncated eigenbasis size.
    label : str or None
        Auto-generated as ``charge_basis_transmon_{n}`` if omitted.
    num_basis : int, default ``61``
        Charge-basis cutoff — must be odd, corresponding to
        :math:`n \\in [-n_\\mathrm{cut}, +n_\\mathrm{cut}]` with
        :math:`n_\\mathrm{cut} = (\\text{num\\_basis} - 1)/2`.
    collapse_model : ``"fermi_golden"`` or ``"ladder"``, default ``"fermi_golden"``
        See :class:`~quchip.devices.circuit.CircuitDevice`.
    collapse_rate_threshold : float, default ``1e-8``
        See :class:`~quchip.devices.circuit.CircuitDevice`.
    **noise_kwargs
        Forwarded to :class:`~quchip.devices.base.BaseDevice`
        (``T1``, ``T2``, ``thermal_population``).

    Notes
    -----
    The T1 collapse-operator normalization assumes T1 is set by
    charge-operator-coupled relaxation (Breuer & Petruccione §3.4).
    For phonon / quasiparticle / flux-dominated T1, override
    :meth:`collapse_operators` or pass ``collapse_model='ladder'``.
    """

    _type_prefix: str = "charge_basis_transmon"
    tunable_param_names = ("E_C", "E_J", "n_g")
    _ENERGY_PARAM_NAMES = ("E_C", "E_J")
    # No flux degree of freedom: the golden-rule bath code would otherwise
    # silently substitute the phase operator for a channel this device does
    # not physically have. See _validate_param_write / __init__ below.
    _ALLOWED_COUPLING_CHANNELS = ("charge",)
    approximation = (
        "Exact diagonalization in the truncated integer charge basis; "
        "accuracy governed by num_basis."
    )

    def __init__(
        self,
        E_C: float,
        E_J: float,
        n_g: float = 0.0,
        levels: int = 3,
        label: str | None = None,
        *,
        num_basis: int = 61,
        collapse_model: Literal["fermi_golden", "ladder"] = "fermi_golden",
        coupling_channel: Literal["charge", "flux"] | None = None,
        collapse_rate_threshold: float = 1e-8,
        **noise_kwargs: float | None,
    ) -> None:
        _check_positive_energy("E_C", E_C)
        _check_positive_energy("E_J", E_J)
        if num_basis < 3 or num_basis % 2 == 0:
            raise ValueError(f"num_basis must be an odd integer >= 3, got {num_basis}")
        if levels > num_basis:
            raise ValueError(f"levels ({levels}) cannot exceed num_basis ({num_basis})")
        if coupling_channel == "flux":
            raise ValueError(
                "ChargeBasisTransmon has no flux bath model; coupling_channel='flux' is not "
                "supported (the golden-rule bath code would silently substitute the phase "
                "operator). Use coupling_channel='charge' or leave it unset."
            )

        super().__init__(
            levels=levels,
            label=label,
            collapse_model=collapse_model,
            coupling_channel=coupling_channel,
            collapse_rate_threshold=collapse_rate_threshold,
            **noise_kwargs,
        )
        self.E_C = E_C
        self.E_J = E_J
        self.n_g = n_g
        self.num_basis = num_basis

    def _validate_param_write(self, name: str, value: Any) -> None:
        """Extend :meth:`CircuitDevice._validate_param_write` with the odd-``num_basis`` and no-flux-channel checks."""
        if name == "coupling_channel" and value == "flux":
            raise ValueError(
                "ChargeBasisTransmon has no flux bath model; coupling_channel='flux' is not "
                "supported. Use coupling_channel='charge' or leave it unset."
            )
        super()._validate_param_write(name, value)
        if name == "num_basis" and value % 2 == 0:
            raise ValueError(f"num_basis must be an odd integer >= 3, got {value}")

    def tunable_param_bounds(self, name: str, value: float) -> tuple[float, float]:
        """``n_g`` lives in ``[-0.5, 0.5]`` (one charge period); other params delegate."""
        if name == "n_g":
            return (-0.5, 0.5)
        return super().tunable_param_bounds(name, value)

    # ------------------------------------------------------------------
    # Native-basis construction (charge basis)
    # ------------------------------------------------------------------

    def _build_native_hamiltonian(self) -> Any:
        n_cut = (self.num_basis - 1) // 2
        n_grid = jnp.arange(-n_cut, n_cut + 1, dtype=jnp.float64)
        kinetic_diag = 4.0 * self.E_C * (n_grid - self.n_g) ** 2
        cos_phi = 0.5 * (jnp.eye(self.num_basis, k=1) + jnp.eye(self.num_basis, k=-1))
        return jnp.diag(kinetic_diag.astype(jnp.complex128)) - self.E_J * cos_phi.astype(
            jnp.complex128
        )

    def _native_charge_operator(self) -> Any:
        n_cut = (self.num_basis - 1) // 2
        n_grid = jnp.arange(-n_cut, n_cut + 1, dtype=jnp.float64)
        return jnp.diag(n_grid.astype(jnp.complex128))

    def _native_phase_operator(self) -> Any:
        r"""Return sin(φ̂) in the integer charge basis (φ̂ itself is not single-valued there).

        :math:`\sin\hat\varphi = \tfrac{1}{2i}(|n\rangle\langle n+1| - \mathrm{h.c.})`
        """
        plus = jnp.eye(self.num_basis, k=1).astype(jnp.complex128)
        minus = jnp.eye(self.num_basis, k=-1).astype(jnp.complex128)
        return (plus - minus) / (2j)

    def physics_notes(self) -> list[str]:
        """Return declared integer-charge-basis assumptions."""
        notes = super().physics_notes()
        notes.append(
            f"Integer charge basis: n ∈ [-{(self.num_basis - 1) // 2}, +{(self.num_basis - 1) // 2}]"
        )
        notes.append("phase_coupling_operator returns sin(φ̂) (φ̂ is not single-valued in the charge basis)")
        return notes

    # ------------------------------------------------------------------
    # Alternate constructor — Duffing-regime inversion (issue #47)
    # ------------------------------------------------------------------

    @classmethod
    def from_frequency(
        cls,
        freq: float,
        anharmonicity: float,
        n_g: float = 0.0,
        levels: int = 3,
        label: str | None = None,
        *,
        num_basis: int = 61,
        collapse_model: Literal["fermi_golden", "ladder"] = "fermi_golden",
        coupling_channel: Literal["charge", "flux"] | None = None,
        collapse_rate_threshold: float = 1e-8,
        **noise_kwargs: float | None,
    ) -> "ChargeBasisTransmon":
        r"""Construct from (freq, anharmonicity) using the Koch-regime inversion.

        Uses :math:`E_C = -\alpha` and
        :math:`E_J = (\omega + E_C)^2 / (8 E_C)`. Residual between the
        Duffing approximation and the exact diagonalized spectrum is
        typically <1% for :math:`E_J/E_C > 50`, growing below that.
        A concrete-scalar warning fires at :math:`E_J/E_C < 20`.
        """
        E_C = -anharmonicity
        E_J = (freq + E_C) ** 2 / (8.0 * E_C)

        ratio = maybe_concrete_scalar(E_J / E_C)
        if ratio is not None and ratio < 20.0:
            warnings.warn(
                f"from_frequency: E_J/E_C ≈ {ratio:.1f} below transmon regime "
                f"(≥20 typical). Diagonalized 0→1 frequency may differ from "
                f"{freq} GHz by >1%. Use explicit E_C/E_J for charge-qubit "
                f"regime devices.",
                stacklevel=2,
            )

        return cls(
            E_C=E_C,
            E_J=E_J,
            n_g=n_g,
            levels=levels,
            label=label,
            num_basis=num_basis,
            collapse_model=collapse_model,
            coupling_channel=coupling_channel,
            collapse_rate_threshold=collapse_rate_threshold,
            **noise_kwargs,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @property
    def computational(self) -> bool:
        """Charge-basis transmon is a computational qubit."""
        return True

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Extend :meth:`CircuitDevice.to_dict` with the charge-basis circuit parameters."""
        data = super().to_dict()
        data["E_C"] = float(self.E_C)
        data["E_J"] = float(self.E_J)
        data["n_g"] = float(self.n_g)
        data["num_basis"] = int(self.num_basis)
        return data

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChargeBasisTransmon":
        return cls(
            E_C=float(d["E_C"]),
            E_J=float(d["E_J"]),
            n_g=float(d.get("n_g", 0.0)),
            levels=int(d.get("levels", 3)),
            label=d.get("label"),
            num_basis=int(d.get("num_basis", 61)),
            collapse_model=d.get("collapse_model", "fermi_golden"),
            coupling_channel=d.get("coupling_channel"),
            collapse_rate_threshold=float(d.get("collapse_rate_threshold", 1e-8)),
            **cls._noise_kwargs_from_dict(d),
        )._restore_reference_freq(d)
