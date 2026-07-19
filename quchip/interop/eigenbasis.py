r"""EigenbasisDevice — exact-lane device wrapping constant eigenbasis data.

A third-party circuit-QED model (e.g. an scqubits ``ZeroPi`` or ``Fluxonium``)
computes its own eigenenergies and coupling-operator matrix elements in its
own native basis and diagonalization routine. :class:`EigenbasisDevice` takes
that already-diagonalized data — energies and, optionally, the charge and
phase operator matrices expressed *in the eigenbasis* — and treats it as a
:class:`~quchip.devices.circuit.CircuitDevice` whose native basis happens to
already be diagonal. :meth:`~quchip.devices.circuit.CircuitDevice._eigensys`
then diagonalizes a diagonal matrix (a no-op beyond sorting and truncation),
so every inherited mechanism — truncation, drive-operator projection,
Fermi-golden-rule collapse channels, serialization — applies unchanged, with
no native-basis machinery of its own to maintain.

The snapshot is exact at the parameter point it was computed at and frozen
from then on: :class:`EigenbasisDevice` stores plain numbers, not a
recipe for recomputing them from underlying circuit parameters, so it is
not differentiable with respect to whatever generated it. See
:meth:`EigenbasisDevice.physics_notes`.

Example
-------
>>> import numpy as np
>>> from quchip.interop.eigenbasis import EigenbasisDevice
>>> E = np.array([0.0, 5.1, 9.8])
>>> n = np.array([[0, 1.0, 0], [1.0, 0, 1.3], [0, 1.3, 0]], dtype=complex)
>>> q = EigenbasisDevice(E, charge_operator=n, source_type="scqubits.Fluxonium")
>>> float(q.freq)
5.1
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from quchip.devices.circuit import CircuitDevice


def _validate_operator_shape(name: str, op: Any, n_native: int) -> Any:
    """Return *op* as a ``(n_native, n_native)`` complex jnp array, or raise."""
    arr = jnp.asarray(op, dtype=jnp.complex128)
    if arr.shape != (n_native, n_native):
        raise ValueError(
            f"{name} must have shape ({n_native}, {n_native}) to match the "
            f"{n_native} supplied energies, got {tuple(arr.shape)}"
        )
    return arr


def _operator_to_json(op: Any | None) -> list[list[list[float]]] | None:
    """Serialize an eigenbasis operator matrix as nested ``[real, imag]`` lists."""
    if op is None:
        return None
    arr = np.asarray(op)
    return [arr.real.tolist(), arr.imag.tolist()]


def _operator_from_json(data: list[list[list[float]]] | None) -> np.ndarray | None:
    """Reconstruct an operator matrix from :func:`_operator_to_json` output."""
    if data is None:
        return None
    real, imag = data
    return np.asarray(real) + 1j * np.asarray(imag)


class EigenbasisDevice(CircuitDevice):
    r"""Device wrapping constant eigenenergies and eigenbasis operator matrices.

    Parameters
    ----------
    energies : array_like
        1-D array of eigenenergies, any offset (stored ground-shifted so
        :math:`E_0 = 0`). Length sets the native-basis dimension.
    charge_operator : array_like, optional
        Charge-like coupling operator :math:`\hat n`, expressed in the same
        eigenbasis as *energies*, shape ``(len(energies), len(energies))``.
        Omit if the source model has no charge-like operator to hand over.
    phase_operator : array_like, optional
        Phase-like coupling operator, expressed in the same eigenbasis as
        *energies*, same shape convention as *charge_operator*.
    levels : int, optional
        Truncated eigenbasis size. Defaults to ``len(energies)``; must not
        exceed it.
    label : str or None
    source_type : str or None
        Free-form identifier of the originating third-party model (e.g.
        ``"scqubits.ZeroPi"``), surfaced in :meth:`physics_notes` and
        preserved through serialization. Purely descriptive.
    collapse_model, coupling_channel, collapse_rate_threshold : see
        :class:`~quchip.devices.circuit.CircuitDevice`.
    **noise_kwargs
        Forwarded to :class:`~quchip.devices.base.BaseDevice`.

    Raises
    ------
    ValueError
        If ``levels`` exceeds ``len(energies)``, or if ``charge_operator`` /
        ``phase_operator`` is supplied with the wrong shape.

    Notes
    -----
    Since the eigenbasis *is* the native basis here, ``_native_charge_operator``
    and ``_native_phase_operator`` return the stored matrix directly (or raise
    if it was never supplied, at the point a drive first asks for it) rather
    than computing one from circuit parameters.
    """

    _type_prefix = "eigenbasis"
    # Intent pin, not a behavior change: CircuitDevice's inherited empty
    # default already excludes this device from inverse design. Fitting an
    # imported fixed spectrum is meaningless — there is no underlying circuit
    # parameter here to move.
    tunable_param_names = ()

    def __init__(
        self,
        energies: Any,
        *,
        charge_operator: Any | None = None,
        phase_operator: Any | None = None,
        levels: int | None = None,
        label: str | None = None,
        source_type: str | None = None,
        collapse_model: Literal["fermi_golden", "ladder"] = "fermi_golden",
        coupling_channel: Literal["charge", "flux"] | None = None,
        collapse_rate_threshold: float = 1e-8,
        **noise_kwargs: float | None,
    ) -> None:
        energies_arr = jnp.asarray(energies, dtype=jnp.float64)
        n_native = int(energies_arr.shape[0])
        if levels is None:
            levels = n_native
        elif levels > n_native:
            raise ValueError(
                f"levels ({levels}) cannot exceed the number of supplied energies ({n_native})"
            )

        super().__init__(
            levels=levels,
            label=label,
            collapse_model=collapse_model,
            coupling_channel=coupling_channel,
            collapse_rate_threshold=collapse_rate_threshold,
            **noise_kwargs,
        )
        # The underscore-prefixed eigen-data (energies, charge/phase matrices) is a
        # frozen-at-import snapshot, deliberately left out of state_version tracking —
        # it never mutates over the device's life. A future public setter that swaps any
        # of it must also invalidate the cached native Hamiltonian / eigensystem, since
        # nothing here bumps state_version to signal the change.
        self._energies = energies_arr - energies_arr[0]
        self._charge_operator = (
            None if charge_operator is None else _validate_operator_shape("charge_operator", charge_operator, n_native)
        )
        self._phase_operator = (
            None if phase_operator is None else _validate_operator_shape("phase_operator", phase_operator, n_native)
        )
        self.source_type = source_type

    # ------------------------------------------------------------------
    # Native-basis construction (the eigenbasis itself)
    # ------------------------------------------------------------------

    def _build_native_hamiltonian(self) -> Any:
        return jnp.diag(self._energies.astype(jnp.complex128))

    def _native_charge_operator(self) -> Any:
        if self._charge_operator is None:
            raise ValueError(
                "this mapping did not supply a charge-like operator; pass "
                "charge_operator= to EigenbasisDevice"
            )
        return self._charge_operator

    def _native_phase_operator(self) -> Any:
        if self._phase_operator is None:
            raise ValueError(
                "this mapping did not supply a phase-like operator; pass "
                "phase_operator= to EigenbasisDevice"
            )
        return self._phase_operator

    # ------------------------------------------------------------------
    # Declared approximations (Principle 12)
    # ------------------------------------------------------------------

    def physics_notes(self) -> list[str]:
        """Return declared assumptions, including the frozen-import snapshot note."""
        notes = super().physics_notes()
        suffix = f" of {self.source_type}" if self.source_type else ""
        notes.append(
            f"Imported eigenbasis snapshot{suffix}: parameters frozen at import — "
            "spectrum exact, not differentiable w.r.t. source parameters"
        )
        return notes

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Extend :meth:`CircuitDevice.to_dict` with the eigenbasis snapshot data."""
        data = super().to_dict()
        # Energies serialize as a plain real list (they are stored real), whereas the
        # operator matrices serialize as [real, imag] pairs (they are complex).
        data["energies"] = np.asarray(self._energies).tolist()
        data["charge_operator"] = _operator_to_json(self._charge_operator)
        data["phase_operator"] = _operator_to_json(self._phase_operator)
        data["source_type"] = self.source_type
        return data

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EigenbasisDevice":
        """Reconstruct an :class:`EigenbasisDevice` from :meth:`to_dict` output.

        Parameters
        ----------
        d
            Dict produced by :meth:`to_dict`.

        Returns
        -------
        EigenbasisDevice
        """
        return cls(
            d["energies"],
            charge_operator=_operator_from_json(d.get("charge_operator")),
            phase_operator=_operator_from_json(d.get("phase_operator")),
            levels=d.get("levels"),
            label=d.get("label"),
            source_type=d.get("source_type"),
            collapse_model=d.get("collapse_model", "fermi_golden"),
            coupling_channel=d.get("coupling_channel"),
            collapse_rate_threshold=float(d.get("collapse_rate_threshold", 1e-8)),
            **cls._noise_kwargs_from_dict(d),
        )._restore_reference_freq(d)
