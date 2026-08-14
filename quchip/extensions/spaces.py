"""Reference device authored on a custom local operator space."""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp

from quchip.declarative.models import DeviceModel
from quchip.declarative.ops import LocalOps
from quchip.declarative.parameters import UNBOUND, Scalar, parameter, setting
from quchip.devices.spaces import CustomSpace


def _spin_operators() -> dict[str, Any]:
    identity = jnp.eye(2, dtype=jnp.complex128)
    sigma_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    sigma_y = jnp.asarray([[0.0, -1j], [1j, 0.0]], dtype=jnp.complex128)
    sigma_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    sigma_plus = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.complex128)
    sigma_minus = sigma_plus.conj().T
    return {
        "I": identity,
        "a": sigma_minus,
        "adag": sigma_plus,
        "n": sigma_plus @ sigma_minus,
        "charge": sigma_x,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "sigma_z": sigma_z,
        "sigma_plus": sigma_plus,
        "sigma_minus": sigma_minus,
    }


class SpinHalf(DeviceModel):
    """Two-level spin with a user-defined local operator vocabulary."""

    _type_prefix = "spin_half"
    _default_levels = 2
    computational = True
    approximation = "Exact two-level spin Hamiltonian in a fixed custom basis."

    basis: Literal["native", "eigen"] | None = setting(default=None, kw_only=True)
    freq: Scalar = parameter(default=UNBOUND, positive=True, unit="GHz", symbol=r"\omega")

    def local_space(self) -> CustomSpace:
        return CustomSpace(2, _spin_operators())

    def validate(self) -> None:
        if self.levels != 2:
            raise ValueError(
                f"SpinHalf has a fixed two-level local space; levels must be 2, got {self.levels}."
            )

    def local_hamiltonian(self, op: LocalOps, p: Any) -> Any:
        return -0.5 * p.freq * op.sigma_z

    def charge_coupling_operator(self) -> Any:
        op = LocalOps(label=self.label, space=self.local_space(), device=self)
        return op["charge"]
