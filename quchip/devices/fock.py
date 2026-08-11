"""Convenience base for devices authored in a truncated Fock space."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from quchip.declarative.expr import PhysicsExpr
from quchip.declarative.models import DeviceModel
from quchip.declarative.ops import LocalOps


class FockDevice(DeviceModel):
    """Device with explicit conventional oscillator coupling operators.

    Subclasses still own their Hamiltonian and approximation. This base only
    declares the standard Fock-space operators used by charge, phase, and
    frequency-modulating drives.
    """

    __init__ = DeviceModel.__init__

    def _local_ops(self) -> LocalOps:
        return LocalOps(label=self.label, space=self.local_space(), device=self)

    @abstractmethod
    def local_hamiltonian(self, op: LocalOps, p: Any) -> PhysicsExpr:
        """Declare this device's local Hamiltonian in its Fock space."""
        ...

    def charge_coupling_operator(self) -> PhysicsExpr:
        """Return the conventional charge quadrature ``i(a - a†)``."""
        op = self._local_ops()
        return 1j * (op.a - op.adag)

    def phase_coupling_operator(self) -> PhysicsExpr:
        """Return the conventional phase quadrature ``a + a†``."""
        op = self._local_ops()
        return op.a + op.adag

    def flux_coupling_operator(self) -> PhysicsExpr:
        """Return the number operator used for frequency modulation."""
        return self._local_ops().n
