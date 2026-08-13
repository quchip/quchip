"""Public records for component-owned time-dependent Hamiltonian terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import jax.tree_util as jtu

from quchip.declarative import qnp
from quchip.declarative.parameters import (
    Parameter,
    UNBOUND,
    DeclarativeMeta,
    Scalar,
    build_declared_signature,
    parameter,
    parameter_fields,
    resolve_declared_params,
    serializable_value,
    validate_declared_fields,
)
from quchip.utils.constants import TWO_PI
from quchip.utils.registry import Registrable


def _synthesize_coefficient_init(
    cls: type,
    fields: dict[str, Parameter],
) -> Any:
    """Build a constructor from a coefficient's declared fields."""
    signature = build_declared_signature(fields)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self")
        for name, value in resolve_declared_params(
            cls,
            arguments,
            fields=fields,
        ).items():
            setattr(self, name, value)

    __init__.__signature__ = signature  # type: ignore[attr-defined]
    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    return __init__


class TimeCoefficient(Registrable, ABC, registry_root=True, metaclass=DeclarativeMeta):
    """Scalar coefficient of a component's time-dependent Hamiltonian term.

    Time is measured in ns. Implementations use :mod:`quchip.qnp` so values
    remain JAX-traceable without receiving an array namespace argument.
    """

    __quchip_param_fields__: ClassVar[dict[str, Parameter]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        validate_declared_fields(cls)
        fields = parameter_fields(cls)
        cls.__quchip_param_fields__ = fields
        names = tuple(fields)

        def _flatten(obj: Any) -> tuple[tuple[Any, ...], tuple[()]]:
            return tuple(getattr(obj, name) for name in names), ()

        def _unflatten(_aux: tuple[()], children: tuple[Any, ...]) -> Any:
            return cls(**dict(zip(names, children)))

        jtu.register_pytree_node(cls, _flatten, _unflatten)
        if "__init__" not in cls.__dict__:
            cls.__init__ = _synthesize_coefficient_init(  # type: ignore[method-assign]
                cls,
                fields,
            )

    @abstractmethod
    def value(self, t: Any) -> Any:
        """Evaluate the coefficient at time *t* in ns."""

    def _signal_program(self) -> Any:
        """Lower this coefficient to the engine's private signal program."""
        from quchip.engine.ir import CoefficientRef

        return CoefficientRef(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the concrete coefficient and its numerical fields."""
        data = super().to_dict()
        data.update(
            {
                name: serializable_value(getattr(self, name))
                for name in type(self).__quchip_param_fields__
            }
        )
        return data

    @classmethod
    def _from_dict_payload(cls, data: dict[str, Any]) -> TimeCoefficient:
        return cls(**{name: data[name] for name in cls.__quchip_param_fields__})


class CosineCoefficient(TimeCoefficient):
    """Cosine coefficient with amplitude in GHz and frequency in GHz."""

    amplitude: Scalar = parameter(default=UNBOUND, unit="GHz")
    frequency: Scalar = parameter(default=UNBOUND, positive=True, unit="GHz")
    phase: Scalar = parameter(default=0.0, unit="rad")

    def value(self, t: Any) -> Any:
        return self.amplitude * qnp.cos(
            TWO_PI * self.frequency * qnp.asarray(t) + self.phase
        )

    def _signal_program(self) -> Any:
        from quchip.engine.ir import Carrier, PolarScale, RealPart

        return RealPart(
            PolarScale(
                Carrier(freq=TWO_PI * self.frequency, sign=1),
                amplitude=self.amplitude,
                theta=self.phase,
            )
        )


def bind_time_coefficient(
    coefficient: TimeCoefficient,
    bindings: dict[str, Any],
) -> TimeCoefficient:
    """Bind symbolic coefficient fields to an owner's current values."""
    from quchip.declarative.expr import materialize_scalar

    return type(coefficient)(
        **{
            name: materialize_scalar(getattr(coefficient, name), bindings=bindings)
            for name in type(coefficient).__quchip_param_fields__
        }
    )


@dataclass(frozen=True)
class TimeDependentTerm:
    """An authored local operator and its time coefficient."""

    operator: Any
    coefficient: TimeCoefficient

    def __post_init__(self) -> None:
        if not isinstance(self.coefficient, TimeCoefficient):
            raise TypeError(
                "TimeDependentTerm coefficient must be a TimeCoefficient; "
                f"got {type(self.coefficient).__name__}."
            )
