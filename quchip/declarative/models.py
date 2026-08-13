"""Declarative base classes for device and coupling physics models.

:class:`DeviceModel` and :class:`CouplingModel` let a subclass declare its
physics parameters as annotated class attributes via
:func:`~quchip.declarative.parameters.parameter` and implement only the
Hamiltonian expression; both synthesize their own ``__init__``, JAX pytree
registration (for :class:`DeviceModel`), and post-construction sign
validation from the declared fields.
"""

from __future__ import annotations

import inspect
import weakref
from typing import Any, ClassVar, cast

import jax.tree_util as jtu

from quchip.chip.coupling_base import BaseCoupling
from quchip.declarative.expr import (
    ParameterNamespace,
    as_operator_expr,
)
from quchip.declarative.dissipation import CollapseChannel, normalize_dissipation
from quchip.declarative.dynamics import (
    TimeDependentTerm,
    bind_time_coefficient,
)
from quchip.declarative.parameters import (
    Parameter,
    UNBOUND,
    DeclarativeMeta,
    build_declared_signature,
    constructor_field,
    parameter_fields,
    resolve_declared_params,
    resolve_declared_settings,
    serializable_value,
    setting_fields,
    validate_sign,
    validate_declared_fields,
)
from quchip.devices.base import _NOISE_FIELDS, BaseDevice
from quchip.utils.state_versioning import _wrap_init_for_finish


def _serialize_declared_params(obj: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Write each serializable declared parameter of *obj* into *data*, in place."""
    for name, spec in type(obj).__quchip_param_fields__.items():
        value = getattr(obj, name)
        if spec.serialize and value is not UNBOUND:
            data[name] = serializable_value(value)
    for name, spec in setting_fields(type(obj)).items():
        if spec.serialize:
            data[name] = getattr(obj, name)
    return data


def _symbolic_parameters(obj: Any) -> ParameterNamespace:
    """Return symbolic leaves for an object's declared parameters."""
    return ParameterNamespace(obj.label, type(obj).__quchip_param_fields__)


def _parameter_bindings(obj: Any) -> dict[str, Any]:
    """Return the object's available declared-parameter values."""
    return {
        f"{obj.label}.{name}": value
        for name in type(obj).__quchip_param_fields__
        if (value := getattr(obj, name)) is not UNBOUND and value is not None
    }


def _normalize_time_term(
    term: Any,
    *,
    owner: Any,
    labels: tuple[str, ...],
    dims: tuple[int, ...],
) -> TimeDependentTerm:
    """Validate and bind one public time term on its authored support."""
    if not isinstance(term, TimeDependentTerm):
        raise TypeError(
            f"{type(owner).__name__}.time_terms() must return TimeDependentTerm values; "
            f"got {type(term).__name__}."
        )
    bindings = _parameter_bindings(owner)
    operator = as_operator_expr(
        term.operator,
        labels=labels,
        dims=dims,
        name=rf"\hat H_{{{owner.label}}}(t)",
        owner=owner,
        scope=owner.label,
        allowed=type(owner).__quchip_param_fields__,
    ).with_bindings(bindings)
    return TimeDependentTerm(
        operator=operator,
        coefficient=bind_time_coefficient(term.coefficient, bindings),
    )


_MISSING = object()


def _validate_explicit_tunable_param_names(
    cls: type[DeviceModel], names: Any, param_fields: dict[str, Parameter]
) -> None:
    """Validate an explicit ``tunable_param_names`` declaration at class-definition time.

    Must be a tuple of unique strings, each resolving to a declared
    :func:`~quchip.declarative.parameters.parameter` field or a genuine class
    attribute/property of *cls* — checked via :func:`inspect.getattr_static`
    so a property is never invoked. Derived defaults (see
    :func:`_resolve_tunable_param_names`) are valid by construction and never
    reach this function.
    """
    if not isinstance(names, tuple):
        raise TypeError(
            f"{cls.__name__}.tunable_param_names must be a tuple of parameter-name strings, got {names!r}."
        )
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"{cls.__name__}.tunable_param_names entries must be strings, got {name!r}.")
        if name in seen:
            raise ValueError(f"{cls.__name__}.tunable_param_names has a duplicate entry {name!r}.")
        seen.add(name)
        if name not in param_fields and inspect.getattr_static(cls, name, _MISSING) is _MISSING:
            raise ValueError(
                f"{cls.__name__}.tunable_param_names names {name!r}, which is not a declared parameter() "
                f"field or class attribute of {cls.__name__}. Available parameter() fields: "
                f"{sorted(param_fields)}."
            )


def _tunable_param_names_explicit_in_lineage(cls: type[DeviceModel]) -> bool:
    """Whether *cls* or an ancestor (up to, not including ``BaseDevice``) explicitly curated ``tunable_param_names``."""
    for base in cls.__mro__:
        if base is BaseDevice:
            break
        if base.__dict__.get("_tunable_param_names_explicit", False):
            return True
    return False


def _resolve_tunable_param_names(cls: type[DeviceModel], param_fields: dict[str, Parameter]) -> None:
    """Resolve ``cls.tunable_param_names``: keep an explicit declaration, else derive from declared fields.

    An explicit declaration on *cls* itself (present in ``cls.__dict__``
    before this hook touches it) is validated here, at class-definition
    time. An explicit declaration inherited from an ancestor — including an
    empty tuple — remains authoritative and is *not* re-derived, so a
    subclass of an explicitly-curated parent inherits that exact curation
    unless it redeclares. Otherwise ``tunable_param_names`` is set to every
    declared ``parameter()`` field, in declaration order.
    """
    if "tunable_param_names" in cls.__dict__:
        _validate_explicit_tunable_param_names(cls, cls.__dict__["tunable_param_names"], param_fields)
        cls._tunable_param_names_explicit = True
        return
    if _tunable_param_names_explicit_in_lineage(cls):
        return
    cls.tunable_param_names = tuple(param_fields.keys())


def _synthesize_device_init(cls: Any) -> Any:
    """Build a positional ``__init__`` for a declarative device subclass.

    Declared parameters become positional-or-keyword arguments in
    declaration order (with their declared defaults); ``levels`` (default
    from ``cls._default_levels``), ``label`` and the noise kwargs follow as
    keyword-only. The body forwards to :meth:`DeviceModel.__init__`,
    which resolves and validates the parameters, so authors get a clean
    signature without hand-writing one.
    """
    fields = cls.__quchip_param_fields__
    param_fields = {
        name: spec for name, spec in fields.items() if not spec.noise
    }
    noise_fields = {
        name: spec for name, spec in fields.items() if spec.noise
    }
    setting_parameters = tuple(
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=spec.default)
        for name, spec in setting_fields(cls).items()
    )
    trailing = setting_parameters + (
        inspect.Parameter("levels", inspect.Parameter.KEYWORD_ONLY, default=cls._default_levels),
        inspect.Parameter("label", inspect.Parameter.KEYWORD_ONLY, default=None),
        *(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if spec.required else spec.default,
            )
            for name, spec in noise_fields.items()
        ),
    )
    signature = build_declared_signature(param_fields, trailing)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self")
        levels = arguments.pop("levels")
        label = arguments.pop("label")
        DeviceModel.__init__(self, levels=levels, label=label, **arguments)

    __init__.__signature__ = signature  # type: ignore[attr-defined]
    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    __init__.__doc__ = f"Initialize {cls.__name__} from its declared parameters."
    return __init__


def _synthesize_coupling_init(cls: Any) -> Any:
    """Build a constructor from coupling endpoints and declared fields."""
    parameters = [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("device_a", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("device_b", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
    parameters.extend(
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if spec.required else spec.default,
        )
        for name, spec in parameter_fields(cls).items()
        if not spec.kw_only
    )
    parameters.extend(
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=spec.default,
        )
        for name, spec in setting_fields(cls).items()
    )
    parameters.extend(
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if spec.required else spec.default,
        )
        for name, spec in parameter_fields(cls).items()
        if spec.kw_only
    )
    parameters.append(inspect.Parameter("label", inspect.Parameter.KEYWORD_ONLY, default=None))
    signature = inspect.Signature(parameters)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self")
        device_a = arguments.pop("device_a")
        device_b = arguments.pop("device_b")
        label = arguments.pop("label")
        CouplingModel.__init__(
            self,
            device_a,
            device_b,
            label=label,
            **arguments,
        )

    __init__.__signature__ = signature  # type: ignore[attr-defined]
    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    __init__.__doc__ = f"Initialize {cls.__name__} from its endpoints and declared fields."
    return __init__


class DeviceModel(BaseDevice, metaclass=DeclarativeMeta):
    """Declarative base for physics device models.

    Subclasses declare their parameters as annotated class attributes using
    :func:`parameter` (e.g. ``freq: Scalar = parameter(positive=True)``) and
    implement :meth:`local_hamiltonian`. The declared parameters become
    positional-or-keyword ``__init__`` arguments and JAX pytree leaves so the
    full instance is traceable / differentiable / sweepable end-to-end.

    The :meth:`hamiltonian` adapter compiles the declarative expression
    returned by :meth:`local_hamiltonian` into an operator for the active
    default backend.

    Examples
    --------
    >>> from quchip.declarative import DeviceModel, parameter, Scalar
    >>> class DuffingOscillator(DeviceModel):
    ...     freq: Scalar = parameter(positive=True, unit="GHz")
    ...     anharmonicity: Scalar = parameter(unit="GHz")
    ...     def local_hamiltonian(self, op, p):
    ...         return p.freq * op.n + 0.5 * p.anharmonicity * op.n @ (op.n - op.I)
    >>> device = DuffingOscillator(freq=5.0, anharmonicity=-0.3, levels=4)
    >>> device.freq
    5.0
    """

    levels: int = constructor_field(default=2, kw_only=True)
    label: Any = constructor_field(default=None, kw_only=True)
    T1: Any = constructor_field(default=None, kw_only=True, runtime=BaseDevice.T1)
    T2: Any = constructor_field(default=None, kw_only=True, runtime=BaseDevice.T2)
    thermal_population: Any = constructor_field(
        default=None,
        kw_only=True,
        runtime=BaseDevice.thermal_population,
    )

    #: Declared approximation-regime statement surfaced by
    #: :meth:`physics_notes` — the mechanism that keeps a model's stated
    #: validity range attached to the class rather than buried in a
    #: docstring a caller may not read.
    approximation: ClassVar[str | None] = None

    #: Whether this device represents a computational qubit, as opposed to
    #: e.g. a bus resonator or a coupler element.
    computational: ClassVar[bool] = False

    # Per-class Fock-truncation default baked into the synthesized ``__init__``.
    # Subclasses override (e.g. ``Resonator`` → 10, ``KerrCavity`` → 30).
    _default_levels: ClassVar[int] = 2

    __quchip_param_fields__: ClassVar[dict[str, Parameter]] = {}
    structural_setting_names: ClassVar[tuple[str, ...]] = ()

    # Whether *this class's own body* declared ``tunable_param_names``
    # explicitly, as opposed to getting the derived default. The ancestor
    # check lives in ``_tunable_param_names_explicit_in_lineage``, which walks
    # the MRO reading this marker per class; never read directly by user code.
    _tunable_param_names_explicit: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        validate_declared_fields(cls)
        # Cache resolved parameter fields once per class. ``__init__`` and
        # the pytree closures both read this — no per-instance re-walk of
        # the MRO.
        param_fields: dict[str, Parameter] = parameter_fields(cls)
        cls.__quchip_param_fields__ = param_fields
        tunable_fields = {
            name: spec for name, spec in param_fields.items() if not spec.noise
        }
        _resolve_tunable_param_names(cls, tunable_fields)

        # Children carried through pytree round-trip. Order is stable.
        # Declared parameters first (in declaration order), then noise
        # params, then the reference-frequency override. All three groups
        # are JAX-traceable scalars (or, for the override, ``None``) — they
        # MUST be children, not aux data, for gradients to flow.
        param_names: tuple[str, ...] = tuple(param_fields.keys())
        children_names: tuple[str, ...] = param_names + ("_reference_freq_override",)
        setting_names = tuple(
            dict.fromkeys((*cls.structural_setting_names, *setting_fields(cls)))
        )

        def _flatten(obj: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
            # ``getattr`` defaults to ``None`` so an instance that never set
            # ``_reference_freq_override`` (it has a class-level default,
            # never guaranteed as an instance attribute) still flattens.
            children = tuple(getattr(obj, name, None) for name in children_names)
            # Aux data must be hashable (jit cache key). ``levels`` and
            # ``label`` are structural; ``children_names`` makes unflatten
            # unambiguous.
            settings = tuple(getattr(obj, name) for name in setting_names)
            aux = (int(obj.levels), obj.label, children_names, setting_names, settings)
            return children, aux

        def _unflatten(aux: tuple[Any, ...], children: tuple[Any, ...]) -> Any:
            levels, label, names, names_of_settings, settings = aux
            obj = cls.__new__(cls)
            # Install structural state via object.__setattr__ to bypass
            # the tracked-mutation hook in BaseDevice.__setattr__. Do not
            # call __init__ — validation may force concretization on a
            # traced value.
            object.__setattr__(obj, "_state_version", 0)
            object.__setattr__(obj, "_tracking_enabled", False)
            object.__setattr__(obj, "levels", levels)
            object.__setattr__(obj, "label", label)
            object.__setattr__(obj, "_owner_chips", weakref.WeakSet())
            object.__setattr__(obj, "_connected_drives", [])
            for name, value in zip(names_of_settings, settings):
                object.__setattr__(obj, name, value)
            for name, value in zip(names, children):
                object.__setattr__(obj, name, value)
            object.__setattr__(obj, "_tracking_enabled", True)
            return obj

        jtu.register_pytree_node(cls, _flatten, _unflatten)

        # Synthesize a positional ``__init__`` from the declared fields unless
        # the subclass hand-writes its own, keeping extension authors in full
        # control when they need it.
        if "__init__" not in cls.__dict__:
            cls.__init__ = _synthesize_device_init(cls)  # type: ignore[method-assign]
            # The synthesized init is installed *after* the __init_subclass__
            # super-chain ran (where StateVersioned wraps a hand-written
            # __init__), so wrap it here to auto-fire _finish_init exactly once
            # after construction. Idempotent if already wrapped.
            _wrap_init_for_finish(cls)

    def __init__(
        self,
        *,
        levels: int = 2,
        label: str | None = None,
        **params: Any,
    ) -> None:
        """Initialize the device from declared parameters and noise kwargs."""
        settings = resolve_declared_settings(type(self), params)
        values = resolve_declared_params(
            type(self), params, fields=type(self).__quchip_param_fields__
        )
        super().__init__(
            levels=levels,
            label=label,
            **{name: values[name] for name in _NOISE_FIELDS},
        )
        for name, value in settings.items():
            setattr(self, name, value)
        for name, value in values.items():
            if name in _NOISE_FIELDS:
                continue
            setattr(self, name, value)
        self.validate()
        # Mutation tracking is switched on automatically by the StateVersioned
        # init wrapper once the outermost __init__ returns.

    def validate(self) -> None:
        """Cross-field validation hook, run at the end of construction.

        Default is a no-op. Subclasses override to enforce constraints that
        span multiple declared parameters (e.g. ``2 * edge <= duration``).
        Checks must be gated on *concrete* scalars via
        :func:`quchip.utils.jax_utils.maybe_concrete_scalar` so traced
        parameters never force concretization.
        """

    def _validate_param_write(self, name: str, value: Any) -> None:
        """Extend the base noise-field checks with declared sign constraints.

        The same :func:`validate_sign` the constructor's resolver runs, so
        e.g. ``r.quality_factor = -5_000.0`` fails after the fact exactly as
        it would at construction (concrete scalars only; tracers pass).
        """
        super()._validate_param_write(name, value)
        spec = type(self).__quchip_param_fields__.get(name)
        if spec is not None:
            validate_sign(name, spec, value)

    def __repr__(self) -> str:
        """Return a constructor-like summary: label, declared params, levels."""
        parts = [f"label={self.label!r}"]
        parts += [f"{name}={getattr(self, name)}" for name in type(self).__quchip_param_fields__]
        parts.append(f"levels={self.levels}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def local_hamiltonian(self, op: Any, p: Any) -> Any:
        """Return this device's local Hamiltonian as a declarative expression.

        Parameters
        ----------
        op : LocalOps
            Operator namespace for this device's endpoint, exposing ``a``,
            ``adag``, ``n``, ``I`` and the Pauli handles as composable
            :class:`~quchip.declarative.expr.PhysicsExpr` nodes.
        p : ParameterNamespace
            Symbolic leaves for the parameters declared on this model.

        Returns
        -------
        PhysicsExpr
            The local Hamiltonian expression, in ordinary-frequency units (GHz).
        """
        raise NotImplementedError

    def time_terms(self, op: Any, p: Any) -> tuple[TimeDependentTerm, ...]:
        """Return local time-dependent Hamiltonian terms beyond the static model."""
        _ = (op, p)
        return ()

    def dissipation(self, op: Any, p: Any) -> tuple[CollapseChannel, ...]:
        """Return device-local Lindblad channels.

        The base channels implement T1, T2, and thermal occupation. Subclasses
        may append channels with ``super().dissipation(op, p)``.
        """
        return super().dissipation(op, p)

    def _time_terms(self) -> tuple[TimeDependentTerm, ...]:
        """Normalize authored time terms for chip assembly."""
        from quchip.declarative.ops import LocalOps

        space = self.local_space()
        operators = LocalOps(label=self.label, space=space, device=self)
        parameters = _symbolic_parameters(self)
        authored = self.time_terms(operators, parameters)
        if not isinstance(authored, tuple):
            raise TypeError(
                f"{type(self).__name__}.time_terms() must return TimeDependentTerm "
                f"values in a tuple; got {type(authored).__name__}."
            )
        return tuple(
            _normalize_time_term(
                term,
                owner=self,
                labels=(self.label,),
                dims=(space.dimension,),
            )
            for term in authored
        )

    def unresolved_hamiltonian(self) -> Any:
        """Return the authored symbolic local Hamiltonian."""
        from quchip.declarative.ops import LocalOps

        space = self.local_space()
        op = LocalOps(label=self.label, space=space)
        parameters = _symbolic_parameters(self)
        authored = self.local_hamiltonian(op, parameters)
        expression = as_operator_expr(
            authored,
            labels=(self.label,),
            dims=(space.dimension,),
            name=rf"\hat H_{{{self.label}}}",
            owner=self,
            scope=self.label,
            allowed=type(self).__quchip_param_fields__,
        )
        return expression.with_bindings(_parameter_bindings(self))

    def to_dict(self) -> dict[str, Any]:
        """Serialize common device state plus declared parameter values."""
        return _serialize_declared_params(self, super().to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeviceModel":
        """Reconstruct the device from :meth:`to_dict` output."""
        fields = cls.__quchip_param_fields__
        params = {
            name: d[name]
            for name, spec in fields.items()
            if spec.serialize and name in d
        }
        settings = {
            name: d[name]
            for name, spec in setting_fields(cls).items()
            if spec.serialize and name in d
        }
        return cls(
            levels=int(d.get("levels", 2)),
            label=d.get("label"),
            **params,
            **settings,
        )._restore_reference_freq(d)

    def physics_notes(self) -> list[str]:
        """Return base device notes plus the declared approximation, if any."""
        notes = super().physics_notes()
        if self.approximation:
            notes.append(self.approximation)
        return notes


class CouplingModel(BaseCoupling, metaclass=DeclarativeMeta):
    """Declarative two-body coupling base.

    Subclasses declare physics parameters via :func:`parameter` and
    implement :meth:`interaction` (returning a
    :class:`~quchip.declarative.expr.PhysicsExpr` over the two endpoint
    operators). The chip's approximation strategy is applied structurally
    by the engine after the authored interaction is assembled.
    Optional overrides:

    - :meth:`time_terms` — time-dependent Hamiltonian terms,
      each pairing a local operator with a public time coefficient.

    :attr:`coupling_strength` defaults to the *first declared parameter
    field* (suited for the common case of one ``g``-like scalar). Override
    the property in subclasses with a different convention.

    .. note::
       Coupling instances are not registered as JAX pytrees and cannot be
       passed as dynamic ``jax.jit`` / ``jax.vmap`` / ``jax.grad``
       arguments. Coupling parameters remain differentiable when the
       coupling (and the devices or chip it couples) is constructed from
       traced arguments inside the transformed function.

    Examples
    --------
    >>> from quchip.declarative import CouplingModel, parameter, Scalar
    >>> class ExchangeCoupling(CouplingModel):
    ...     g: Scalar = parameter(unit="GHz")
    ...     def interaction(self, a, b, p):
    ...         return p.g * (a.a * b.adag + a.adag * b.a)
    >>> c = ExchangeCoupling("q0", "q1", g=0.01)
    >>> c.coupling_strength
    0.01
    """

    device_a: BaseDevice | str = constructor_field()
    device_b: BaseDevice | str = constructor_field()
    label: Any = constructor_field(default=None, kw_only=True)

    __quchip_param_fields__: ClassVar[dict[str, Parameter]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        validate_declared_fields(cls)
        cls.__quchip_param_fields__ = parameter_fields(cls)
        if "__init__" not in cls.__dict__:
            cls.__init__ = _synthesize_coupling_init(cls)  # type: ignore[method-assign]
            _wrap_init_for_finish(cls)

    def __init__(
        self,
        device_a: Any,
        device_b: Any,
        *,
        label: str | None = None,
        **params: Any,
    ) -> None:
        """Initialize a declarative coupling between two devices or labels."""
        settings = resolve_declared_settings(type(self), params)
        values = resolve_declared_params(type(self), params)
        super().__init__(device_a, device_b, label=label)
        for name, value in settings.items():
            setattr(self, name, value)
        for name, value in values.items():
            setattr(self, name, value)
        # Tracking auto-enables via the StateVersioned init wrapper post-construction.

    def __setattr__(self, name: str, value: Any) -> None:
        """Give post-construction writes the same declared-sign validation as construction.

        Mirrors :meth:`~quchip.devices.base.BaseDevice.__setattr__`: once
        ``_tracking_enabled`` (from
        :class:`~quchip.utils.state_versioning.StateVersioned`) goes live
        after construction, every non-private write runs
        :func:`~quchip.declarative.parameters.validate_sign` against the
        declared field spec before the value lands. Concrete scalars only;
        tracers pass.
        """
        if getattr(self, "_tracking_enabled", False) and not name.startswith("_"):
            spec = type(self).__quchip_param_fields__.get(name)
            if spec is not None:
                validate_sign(name, spec, value)
        super().__setattr__(name, value)

    @property
    def coupling_strength(self) -> Any:
        """Primary scalar coupling strength, defaulting to the first parameter."""
        fields = type(self).__quchip_param_fields__
        first = next(iter(fields), None)
        return getattr(self, first) if first is not None else 0.0

    @property
    def coupling_strength_name(self) -> str:
        """Display name of :attr:`coupling_strength`, defaulting to the first parameter field."""
        fields = type(self).__quchip_param_fields__
        first = next(iter(fields), None)
        return first if first is not None else "g"

    def __repr__(self) -> str:
        """Return a constructor-like summary: endpoints, declared params, label.

        Default for extension authors; mirrors :meth:`DeviceModel.__repr__`.
        Built-ins with richer summaries override it.
        """
        parts = [f"'{self.device_a_label}' <-> '{self.device_b_label}'"]
        parts += [f"{name}={getattr(self, name)}" for name in type(self).__quchip_param_fields__]
        parts.append(f"label={self.label!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    # --- Physics overrides for subclasses ---

    def interaction(self, a: Any, b: Any, p: Any) -> Any:
        """Return the full two-body interaction expression.

        Parameters
        ----------
        a, b : EndpointOps
            Operator namespaces for the two coupled endpoints. Same-endpoint
            operators compose with ``@``; cross-endpoint operators combine
            with ``*`` (tensor product).

        Returns
        -------
        PhysicsExpr
            The interaction Hamiltonian expression, in ordinary-frequency
            units (GHz).
        """
        raise NotImplementedError

    def time_terms(self, a: Any, b: Any, p: Any) -> tuple[TimeDependentTerm, ...]:
        """Return time-dependent interaction terms.

        Parameters
        ----------
        a, b : EndpointOps
            Operator namespaces for the two coupled endpoints.

        Returns
        -------
        tuple of TimeDependentTerm
            Local operators and their scalar time coefficients. The empty
            tuple denotes a purely static coupling.
        """
        _ = (a, b, p)
        return ()

    def parametric_interaction(self, a: Any, b: Any, p: Any) -> Any:
        """Return the parametric interaction structure, or ``None`` when this coupling is not modulable.

        The coupling-side mirror of the device drive-dispatch protocols: a
        :class:`~quchip.control.drive.ParametricDrive` accepts any coupling
        whose hook returns a :class:`~quchip.declarative.expr.PhysicsExpr`.
        """
        _ = (a, b, p)
        return None

    def dissipation(self, a: Any, b: Any, p: Any) -> tuple[CollapseChannel, ...]:
        """Return authored two-endpoint Lindblad channels."""
        _ = (a, b, p)
        return ()

    def _bind_parametric_interaction(self) -> Any | None:
        """Validate and bind the coupling's authored parametric interaction."""
        a_ops, b_ops = self._endpoint_ops()
        p = _symbolic_parameters(self)
        authored = self.parametric_interaction(a_ops, b_ops, p)
        if authored is None:
            return None
        expr = as_operator_expr(
            authored,
            labels=(self.device_a_label, self.device_b_label),
            dims=(a_ops.space.dimension, b_ops.space.dimension),
            name=rf"\hat P_{{{self.label}}}",
            owner=self,
            scope=self.label,
            allowed=type(self).__quchip_param_fields__,
        )
        self._check_endpoint_order(expr, "parametric_interaction")
        return expr.with_bindings(_parameter_bindings(self))

    # --- Compilation ---

    @property
    def _resolved_a(self) -> BaseDevice:
        """``device_a`` narrowed to a concrete device.

        ``device_a``/``device_b`` are typed ``BaseDevice | str`` on the base
        class to accept a label at construction time, but every caller below
        only runs post-``clone_for_chip``, which replaces the label with the
        resolved device object before any physics compilation starts.
        """
        return cast(BaseDevice, self.device_a)

    @property
    def _resolved_b(self) -> BaseDevice:
        """``device_b`` narrowed to a concrete device (see :attr:`_resolved_a`)."""
        return cast(BaseDevice, self.device_b)

    def _endpoint_ops(self) -> tuple[Any, Any]:
        """Return the ``(a, b)`` operator namespaces for this coupling's resolved endpoints."""
        from quchip.declarative.ops import EndpointOps

        return (
            EndpointOps(
                label=self.device_a_label,
                space=self._resolved_a.local_space(),
                device=self._resolved_a,
            ),
            EndpointOps(
                label=self.device_b_label,
                space=self._resolved_b.local_space(),
                device=self._resolved_b,
            ),
        )

    def _check_endpoint_order(self, expr: Any, method_name: str) -> None:
        """Reject a two-endpoint expression whose labels are not in ``(a, b)`` order.

        :func:`~quchip.declarative.expr.materialize_expr`'s tensor branch
        preserves the expression tree's argument order into the backend
        ``tensor()`` call, and the chip and engine embed the compiled
        two-body operator positionally against
        ``(device_a_label, device_b_label)`` (``Chip.hamiltonian``,
        ``embed_two_body``, ``assembly``'s canonical-operator
        metadata) without reading ``expr.labels`` back out of the compiled
        backend operator. An expression authored as ``b.op * a.op``
        therefore compiles to a Hilbert-space-reversed operator that
        mis-embeds without any shape-mismatch error whenever the two
        endpoints share a dimension.
        """
        endpoint_order = (self.device_a_label, self.device_b_label)
        if set(expr.labels) == set(endpoint_order) and expr.labels != endpoint_order:
            raise TypeError(
                f"{type(self).__name__}.{method_name}() built its expression with endpoint order "
                f"{expr.labels!r}; expected {endpoint_order!r}. Compose cross-endpoint terms with "
                "device_a's operator first, e.g. `a.<op> * b.<op>`."
            )

    def interaction_hamiltonian(self) -> Any:
        """Return the authored symbolic interaction Hamiltonian."""

        a_ops, b_ops = self._endpoint_ops()
        parameters = _symbolic_parameters(self)
        authored = self.interaction(a_ops, b_ops, parameters)
        expr = as_operator_expr(
            authored,
            labels=(self.device_a_label, self.device_b_label),
            dims=(a_ops.space.dimension, b_ops.space.dimension),
            name=rf"\hat H_{{{self.label}}}",
            owner=self,
            scope=self.label,
            allowed=type(self).__quchip_param_fields__,
        )
        self._check_endpoint_order(expr, "interaction")
        return expr.with_bindings(_parameter_bindings(self))

    def _collapse_channels_with_paths(
        self,
    ) -> tuple[tuple[CollapseChannel, tuple[str, ...]], ...]:
        """Normalize authored coupling dissipation and infer dependencies."""
        a_ops, b_ops = self._endpoint_ops()
        parameters = _symbolic_parameters(self)
        authored = self.dissipation(a_ops, b_ops, parameters)
        bindings = _parameter_bindings(self)
        fields = type(self).__quchip_param_fields__
        normalized = normalize_dissipation(
            authored,
            labels=(self.device_a_label, self.device_b_label),
            dims=(a_ops.space.dimension, b_ops.space.dimension),
            owner=self,
            scope=self.label,
            allowed=fields,
            bindings=bindings,
        )
        for channel, _paths in normalized:
            self._check_endpoint_order(channel.operator, "dissipation")
        return normalized

    def collapse_channels(self) -> tuple[CollapseChannel, ...]:
        """Return normalized coupling collapse channels."""
        return tuple(channel for channel, _paths in self._collapse_channels_with_paths())

    @classmethod
    def from_dict(cls, d: dict[str, Any], device_a: Any, device_b: Any) -> "CouplingModel":
        """Reconstruct a coupling from :meth:`to_dict` output.

        Default implementation: forward declared parameters straight into
        ``__init__``. Subclasses with bespoke serialization (e.g. envelope
        modulations) override this.
        """
        fields = cls.__quchip_param_fields__
        params = {name: d[name] for name in fields if name in d}
        settings = {
            name: d[name]
            for name, spec in setting_fields(cls).items()
            if spec.serialize and name in d
        }
        return cls(
            device_a=device_a,
            device_b=device_b,
            label=d.get("label"),
            **params,
            **settings,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize common coupling state plus declared parameter values."""
        return _serialize_declared_params(self, super().to_dict())

    def _time_terms(self) -> tuple[TimeDependentTerm, ...]:
        """Normalize authored time-dependent interactions for chip assembly."""
        a_ops, b_ops = self._endpoint_ops()
        parameters = _symbolic_parameters(self)
        authored = self.time_terms(a_ops, b_ops, parameters)
        if not isinstance(authored, tuple):
            raise TypeError(
                f"{type(self).__name__}.time_terms() must return TimeDependentTerm "
                f"values in a tuple; got {type(authored).__name__}."
            )
        normalized = tuple(
            _normalize_time_term(
                term,
                owner=self,
                labels=(self.device_a_label, self.device_b_label),
                dims=(a_ops.space.dimension, b_ops.space.dimension),
            )
            for term in authored
        )
        for term in normalized:
            self._check_endpoint_order(term.operator, "time_terms")
        return normalized
