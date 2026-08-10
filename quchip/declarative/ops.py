"""Declarative operator-handle namespaces for the physics DSL.

:class:`LocalOps` (aliased :class:`EndpointOps` at coupling call sites)
exposes one Hilbert-space endpoint's operators as composable
:class:`~quchip.declarative.expr.PhysicsExpr` nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

from quchip.declarative.expr import PhysicsExpr
from quchip.devices.spaces import ChargeSpace, FockSpace, LocalSpace, PhaseGridSpace


@dataclass(frozen=True)
class LocalOps:
    """Declarative operator namespace for one local Hilbert space endpoint.

    Passed as ``op`` to :meth:`DeviceModel.local_hamiltonian` and as the two
    endpoints ``a``, ``b`` to :meth:`CouplingModel.interaction`. Each property
    returns a :class:`~quchip.declarative.expr.PhysicsExpr` that composes with
    ``+``, ``-``, ``@`` (same endpoint), ``*`` (scalar or tensor product).

    Examples
    --------
    >>> from quchip.declarative import LocalOps
    >>> from quchip.devices.spaces import FockSpace
    >>> op = LocalOps(label="q", space=FockSpace(3))
    >>> H = 5.0 * op.n + 0.5 * (op.adag @ op.adag @ op.a @ op.a)
    >>> H.kind
    'add'
    """

    label: str
    space: LocalSpace

    def _op(self, name: str) -> PhysicsExpr:
        return PhysicsExpr(kind="op", args=(name, self.space), labels=(self.label,))

    def __getitem__(self, name: str) -> PhysicsExpr:
        """Return a named operator supplied by this local space."""
        return self._op(name)

    @property
    def a(self) -> PhysicsExpr:
        """Lowering operator for this endpoint."""
        return self._op("a")

    @property
    def adag(self) -> PhysicsExpr:
        """Raising operator for this endpoint."""
        return self._op("adag")

    @property
    def n(self) -> PhysicsExpr:
        """Number operator for this endpoint."""
        return self._op("n")

    @property
    def n2(self) -> PhysicsExpr:
        """Squared charge operator in a compatible authored local space."""
        return self._op("n2")

    @property
    def phi(self) -> PhysicsExpr:
        """Phase operator in a compatible authored local space."""
        return self._op("phi")

    @property
    def cos_phi(self) -> PhysicsExpr:
        """Cosine of phase in a compatible authored local space."""
        return self._op("cos_phi")

    @property
    def sin_phi(self) -> PhysicsExpr:
        """Sine of phase in a compatible authored local space."""
        return self._op("sin_phi")

    @property
    def I(self) -> PhysicsExpr:  # noqa: E743 - physics API uses I for identity.
        """Identity operator for this endpoint."""
        return self._op("I")

    @property
    def x(self) -> PhysicsExpr:
        """Unnormalized quadrature ``x = a + a†`` (no 1/sqrt(2) factor)."""
        return self.a + self.adag

    @property
    def charge(self) -> PhysicsExpr:
        """Physical charge-like drive operator for this local representation."""
        if isinstance(self.space, FockSpace):
            return self.x
        if isinstance(self.space, (ChargeSpace, PhaseGridSpace)):
            return self.n
        return self._op("charge")

    @property
    def sigma_x(self) -> PhysicsExpr:
        """``|0><1| + |1><0|`` on the computational ``|0>, |1>`` subspace of the truncated space."""
        return self._op("sigma_x")

    @property
    def sigma_y(self) -> PhysicsExpr:
        """``-i|0><1| + i|1><0|`` on the computational ``|0>, |1>`` subspace."""
        return self._op("sigma_y")

    @property
    def sigma_z(self) -> PhysicsExpr:
        """``|0><0| - |1><1|`` on the computational ``|0>, |1>`` subspace."""
        return self._op("sigma_z")

    @property
    def sigma_plus(self) -> PhysicsExpr:
        """Raising operator ``|1><0|`` on the computational subspace."""
        return self._op("sigma_plus")

    @property
    def sigma_minus(self) -> PhysicsExpr:
        """Lowering operator ``|0><1|`` on the computational subspace."""
        return self._op("sigma_minus")


# `EndpointOps` is an alias of `LocalOps`: coupling call sites spell it
# `EndpointOps` for the two endpoint namespaces; device call sites spell it
# `LocalOps` for the single local namespace.
EndpointOps = LocalOps
