"""An ideal two-level system with a declared transition frequency."""

from __future__ import annotations

from operator import index
from typing import Any

from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import UNBOUND, Scalar, parameter
from quchip.devices.fock import FockDevice


class Qubit(FockDevice):
    r"""Ideal two-level device with ``H/h = freq * |1><1|``.

    Parameters
    ----------
    freq : float
        Positive transition frequency in GHz; may be a JAX tracer.
    levels : int, default 2
        Must equal two. Use a transmon model to include higher levels.
    label : str or None, default None
        Device label; ``None`` selects an automatic label.
    T1 : float or None, default None
        Energy-relaxation time in ns; ``None`` disables relaxation.
    T2 : float or None, default None
        Total coherence time in ns; ``None`` disables pure dephasing.
        With both times set, ``T2 <= 2*T1``. At zero thermal occupation,
        the pure-dephasing rate is ``1/T2 - 1/(2*T1)``.
    thermal_occupation : float or None, default None
        Dimensionless bath occupation; ``None`` disables thermal absorption.

    Notes
    -----
    Relaxation and dephasing use the standard device channels. In two levels,
    ``n = |1><1|`` and ``a = |0><1|``. A frame at ``freq`` removes the free
    Hamiltonian. The model contains no leakage levels or circuit parameters.
    """

    _type_prefix = "qubit"
    computational = True
    approximation = "Ideal two-level system; higher levels and leakage are omitted."
    dressed_fit_target_fields = (("freq", "freq"),)
    dressed_fit_param_names = ("freq",)

    freq: Scalar = parameter(default=UNBOUND, positive=True, unit="GHz", symbol=r"\omega")

    def local_hamiltonian(self, op: LocalOps, p: Any) -> PhysicsExpr:
        """Return the two-level Hamiltonian in GHz.

        Parameters
        ----------
        op : LocalOps
            Local two-level operator namespace.
        p : ParameterNamespace
            Symbolic transition frequency.
        """
        return p.freq * op.n

    def validate(self) -> None:
        """Require exactly two levels at construction and grouped rebinding."""
        super().validate()
        if index(self.levels) != 2:
            raise ValueError("Qubit requires levels=2; use a transmon model for higher levels.")

    def _validate_param_write(self, name: str, value: Any) -> None:
        super()._validate_param_write(name, value)
        if name == "levels" and index(value) != 2:
            raise ValueError("Qubit requires levels=2; use a transmon model for higher levels.")

    def truncation_boundary(self) -> None:
        """Return no numerical cutoff for an intrinsically two-level system."""
        return None

    def _truncation_note(self) -> str:
        """Describe the complete two-level basis."""
        return "Two-level basis: |0>, |1>"
