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
from typing import Any, cast, dataclass_transform

import jax.tree_util as jtu

from quchip.chip.coupling_base import BaseCoupling
from quchip.declarative.parameters import (
    Parameter,
    build_declared_signature,
    parameter_fields,
    resolve_declared_params,
    serializable_value,
    validate_sign,
)
# ``_NOISE_FIELDS`` is the single source of truth for the noise-parameter set
# (BaseDevice owns it). These are JAX-traceable scalars (or ``None``), so they
# must travel as pytree *children* rather than aux data — otherwise gradients
# can't flow through them.
from quchip.devices.base import _NOISE_FIELDS, BaseDevice
from quchip.utils.state_versioning import _wrap_init_for_finish


def _serialize_declared_params(obj: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Write each serializable declared parameter of *obj* into *data*, in place."""
    for name, spec in type(obj).__quchip_param_fields__.items():
        if spec.serialize:
            data[name] = serializable_value(getattr(obj, name))
    return data


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
    param_fields = cls.__quchip_param_fields__
    trailing = (
        inspect.Parameter("levels", inspect.Parameter.KEYWORD_ONLY, default=cls._default_levels),
        inspect.Parameter("label", inspect.Parameter.KEYWORD_ONLY, default=None),
        *(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None) for name in _NOISE_FIELDS),
    )
    signature = build_declared_signature(param_fields, trailing, owner=cls)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self")
        levels = arguments.pop("levels")
        label = arguments.pop("label")
        noise = {name: arguments.pop(name) for name in _NOISE_FIELDS}
        DeviceModel.__init__(self, levels=levels, label=label, **noise, **arguments)

    __init__.__signature__ = signature  # type: ignore[attr-defined]
    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    __init__.__doc__ = f"Initialize {cls.__name__} from its declared parameters."
    return __init__


# ``@dataclass_transform`` is a static-analyzer hint: it tells type checkers
# to treat ``freq: Scalar = parameter(...)`` field declarations like
# dataclass fields, so a synthesized ``__init__`` accepting ``freq`` as a
# ``Scalar`` argument type-checks even though ``parameter()`` returns a
# ``Parameter`` instance. ``__init_subclass__`` below does the runtime work:
# it synthesizes a positional ``__init__`` from the declared fields unless
# the subclass hand-writes its own (see ``_synthesize_device_init``).
# TODO: dissolve the generated stubs (tools/gen_device_stubs.py) by declaring
# levels/label/noise as keyword-only specifier fields so PEP 681 describes the
# full constructor natively; that also fixes user-authored models in their own
# IDEs, which stubs in this repo never can.
@dataclass_transform(field_specifiers=(Parameter,))
class DeviceModel(BaseDevice):
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
    ...     def local_hamiltonian(self, op):
    ...         return self.freq * op.n + 0.5 * self.anharmonicity * op.n @ (op.n - op.I)
    >>> device = DuffingOscillator(freq=5.0, anharmonicity=-0.3, levels=4)
    >>> device.freq
    5.0
    """

    #: Declared approximation-regime statement surfaced by
    #: :meth:`physics_notes` — the mechanism that keeps a model's stated
    #: validity range attached to the class rather than buried in a
    #: docstring a caller may not read.
    approximation: str | None = None

    #: Whether this device represents a computational qubit, as opposed to
    #: e.g. a bus resonator or a coupler element.
    computational: bool = False

    # Per-class Fock-truncation default baked into the synthesized ``__init__``.
    # Subclasses override (e.g. ``Resonator`` → 10, ``KerrCavity`` → 30).
    _default_levels: int = 2

    __quchip_param_fields__: dict[str, Parameter] = {}

    # Whether *this class's own body* declared ``tunable_param_names``
    # explicitly, as opposed to getting the derived default. The ancestor
    # check lives in ``_tunable_param_names_explicit_in_lineage``, which walks
    # the MRO reading this marker per class; never read directly by user code.
    _tunable_param_names_explicit: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Cache resolved parameter fields once per class. ``__init__`` and
        # the pytree closures both read this — no per-instance re-walk of
        # the MRO.
        param_fields: dict[str, Parameter] = parameter_fields(cls)
        cls.__quchip_param_fields__ = param_fields
        _resolve_tunable_param_names(cls, param_fields)

        # Children carried through pytree round-trip. Order is stable.
        # Declared parameters first (in declaration order), then noise
        # params, then the reference-frequency override. All three groups
        # are JAX-traceable scalars (or, for the override, ``None``) — they
        # MUST be children, not aux data, for gradients to flow.
        param_names: tuple[str, ...] = tuple(param_fields.keys())
        children_names: tuple[str, ...] = param_names + _NOISE_FIELDS + ("_reference_freq_override",)

        def _flatten(obj: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
            # ``getattr`` defaults to ``None`` so an instance that never set
            # ``_reference_freq_override`` (it has a class-level default,
            # never guaranteed as an instance attribute) still flattens.
            children = tuple(getattr(obj, name, None) for name in children_names)
            # Aux data must be hashable (jit cache key). ``levels`` and
            # ``label`` are structural; ``children_names`` makes unflatten
            # unambiguous.
            aux = (int(obj.levels), obj.label, children_names)
            return children, aux

        def _unflatten(aux: tuple[Any, ...], children: tuple[Any, ...]) -> Any:
            levels, label, names = aux
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
        T1: Any = None,
        T2: Any = None,
        thermal_population: Any = None,
        **params: Any,
    ) -> None:
        """Initialize the device from declared parameters and noise kwargs."""
        values = resolve_declared_params(type(self), params)
        super().__init__(
            levels=levels,
            label=label,
            T1=T1,
            T2=T2,
            thermal_population=thermal_population,
        )
        for name, value in values.items():
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

    def local_hamiltonian(self, op: Any) -> Any:
        """Return this device's local Hamiltonian as a declarative expression.

        Parameters
        ----------
        op : LocalOps
            Operator namespace for this device's endpoint, exposing ``a``,
            ``adag``, ``n``, ``I`` and the Pauli handles as composable
            :class:`~quchip.declarative.expr.PhysicsExpr` nodes.

        Returns
        -------
        PhysicsExpr
            The local Hamiltonian expression, in ordinary-frequency units (GHz).
        """
        raise NotImplementedError

    def hamiltonian(self) -> Any:
        """Compile :meth:`local_hamiltonian` for the active default backend."""
        from quchip.backend import get_default_backend
        from quchip.declarative.expr import compile_expr
        from quchip.declarative.ops import LocalOps

        backend = get_default_backend()
        op = LocalOps(label=self.label, levels=self.levels)
        expr = self.local_hamiltonian(op)
        return compile_expr(expr, self.declarative_ops(), backend)

    def to_dict(self) -> dict[str, Any]:
        """Serialize common device state plus declared parameter values."""
        return _serialize_declared_params(self, super().to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeviceModel":
        """Reconstruct the device from :meth:`to_dict` output."""
        fields = cls.__quchip_param_fields__
        params = {name: d[name] for name, spec in fields.items() if spec.serialize and name in d}
        return cls(
            levels=int(d.get("levels", 2)),
            label=d.get("label"),
            **cls._noise_kwargs_from_dict(d),
            **params,
        )._restore_reference_freq(d)

    def physics_notes(self) -> list[str]:
        """Return base device notes plus the declared approximation, if any."""
        notes = super().physics_notes()
        if self.approximation:
            notes.append(self.approximation)
        return notes


@dataclass_transform(field_specifiers=(Parameter,))
class CouplingModel(BaseCoupling):
    """Declarative two-body coupling base.

    Subclasses declare physics parameters via :func:`parameter` and
    implement :meth:`interaction` (returning a
    :class:`~quchip.declarative.expr.PhysicsExpr` over the two endpoint
    operators). The RWA is applied structurally by the chip and engine
    via :meth:`~quchip.chip.coupling_base.BaseCoupling.rwa_keeps_band`;
    only the *parametric* RWA structures remain author-declared, because
    pump sideband selection depends on frequency intent rather than
    operator structure: which sideband a pump activates (red: Δa+Δb=0
    exchange; blue: |Δa+Δb|=2 two-photon) depends on where the pump
    frequency sits relative to traced device frequencies, and a structural
    rule cannot infer it without branching on traced values.
    Optional overrides:

    - :meth:`time_dependent` — parametric (time-dependent) modulation, as
      a :class:`PhysicsExpr` carrying a single dynamic source.

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
    ...     def interaction(self, a, b):
    ...         return self.g * (a.a * b.adag + a.adag * b.a)
    >>> c = ExchangeCoupling("q0", "q1", g=0.01)
    >>> c.coupling_strength
    0.01
    """

    __quchip_param_fields__: dict[str, Parameter] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__quchip_param_fields__ = parameter_fields(cls)

    def __init__(
        self,
        device_a: Any,
        device_b: Any,
        *,
        label: str | None = None,
        rwa: bool | None = None,
        **params: Any,
    ) -> None:
        """Initialize a declarative coupling between two devices or labels."""
        values = resolve_declared_params(type(self), params)
        super().__init__(device_a, device_b, label=label)
        self._rwa = rwa
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

    def interaction(self, a: Any, b: Any) -> Any:
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

    def time_dependent(self, a: Any, b: Any) -> Any:
        """Return an optional time-dependent interaction expression.

        Parameters
        ----------
        a, b : EndpointOps
            Operator namespaces for the two coupled endpoints.

        Returns
        -------
        PhysicsExpr or None
            An expression carrying exactly one dynamic source (envelope or
            modulation), or ``None`` when the coupling is purely static.
        """
        return None

    def rwa_time_dependent(self, a: Any, b: Any) -> Any:
        """Return the RWA time-dependent expression, defaulting to full form.

        Mirrors the parametric RWA split for the dynamic term: subclasses
        whose parametric drive keeps only the co-rotating operators under
        RWA override this.

        Parameters
        ----------
        a, b : EndpointOps
            Operator namespaces for the two coupled endpoints.
        """
        return self.time_dependent(a, b)

    def parametric_interaction(self, a: Any, b: Any) -> Any:
        """Return the parametric interaction structure, or ``None`` when this coupling is not modulable.

        The coupling-side mirror of the device drive-dispatch protocols: a
        :class:`~quchip.control.drive.ParametricDrive` accepts any coupling
        whose hook returns a :class:`~quchip.declarative.expr.PhysicsExpr`.
        """
        _ = (a, b)
        return None

    def rwa_parametric_interaction(self, a: Any, b: Any) -> Any:
        """RWA-retained parametric structure; defaults to the full form."""
        return self.parametric_interaction(a, b)

    def parametric_operator(self, chip: Any) -> Any | None:
        """Compile the parametric structure for *chip*'s resolved RWA policy.

        Returns the backend-native operator on the local two-body space, or
        ``None`` when :meth:`parametric_interaction` declines. Valid only once
        the coupling is chip-resolved (same contract as
        :meth:`interaction_hamiltonian`).
        """
        from quchip.declarative.expr import compile_expr

        a_ops, b_ops = self._endpoint_ops()
        rwa = chip.resolve_rwa(self)
        expr = self.rwa_parametric_interaction(a_ops, b_ops) if rwa else self.parametric_interaction(a_ops, b_ops)
        if expr is None:
            return None
        self._check_endpoint_order(expr, "rwa_parametric_interaction" if rwa else "parametric_interaction")
        from quchip.backend import get_default_backend

        return compile_expr(expr, self._endpoint_lookup(), get_default_backend())

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

    def _endpoint_lookup(self) -> dict[tuple[str, str], Any]:
        """Return the merged ``(label, op-name) -> operator`` lookup for both resolved endpoints."""
        return {**self._resolved_a.declarative_ops(), **self._resolved_b.declarative_ops()}

    def _endpoint_ops(self) -> tuple[Any, Any]:
        """Return the ``(a, b)`` operator namespaces for this coupling's resolved endpoints."""
        from quchip.declarative.ops import EndpointOps

        return (
            EndpointOps(label=self.device_a_label, levels=self._resolved_a.levels),
            EndpointOps(label=self.device_b_label, levels=self._resolved_b.levels),
        )

    def _check_endpoint_order(self, expr: Any, method_name: str) -> None:
        """Reject a two-endpoint expression whose labels are not in ``(a, b)`` order.

        :func:`~quchip.declarative.expr.compile_expr`'s tensor branch
        preserves the expression tree's argument order into the backend
        ``tensor()`` call, and the chip and engine embed the compiled
        two-body operator positionally against
        ``(device_a_label, device_b_label)`` (``Chip.hamiltonian``,
        ``embed_two_body``, ``stage2_assembly``'s canonical-operator
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
        """Compile the full interaction expression for the default backend."""
        from quchip.backend import get_default_backend
        from quchip.declarative.expr import compile_expr

        backend = get_default_backend()
        a_ops, b_ops = self._endpoint_ops()
        expr = self.interaction(a_ops, b_ops)
        self._check_endpoint_order(expr, "interaction")
        return compile_expr(expr, self._endpoint_lookup(), backend)

    @classmethod
    def from_dict(cls, d: dict[str, Any], device_a: Any, device_b: Any) -> "CouplingModel":
        """Reconstruct a coupling from :meth:`to_dict` output.

        Default implementation: forward declared parameters straight into
        ``__init__``. Subclasses with bespoke serialization (e.g. envelope
        modulations) override this.
        """
        fields = cls.__quchip_param_fields__
        params = {name: d[name] for name in fields if name in d}
        return cls(
            device_a=device_a,
            device_b=device_b,
            label=d.get("label"),
            rwa=d.get("rwa"),
            **params,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize common coupling state plus declared parameter values."""
        data = _serialize_declared_params(self, super().to_dict())
        data["rwa"] = self._rwa
        return data

    def _validate_dynamic_expr(self, expr: Any, method_name: str) -> None:
        """Require exactly one dynamic source on a compiled time-dependent expression.

        Applied to whichever expression :meth:`dynamic_interaction_terms`
        actually compiles: the full :meth:`time_dependent` form, and again
        to :meth:`rwa_time_dependent`'s replacement when the chip resolves
        RWA, so a malformed RWA override cannot skip the check the full
        form passed.
        """
        from quchip.declarative.expr import PhysicsExpr

        if not isinstance(expr, PhysicsExpr) or len(expr.dynamic_sources) != 1:
            raise ValueError(
                f"{type(self).__name__}.{method_name}() must return an expression containing "
                "exactly one dynamic source."
            )

    def dynamic_interaction_terms(self, chip: Any) -> list[tuple[Any, Any]]:
        """Compile the optional single-source dynamic interaction term.

        Parameters
        ----------
        chip : Chip
            Owning chip, consulted via :meth:`Chip.resolve_rwa` to select the
            static or RWA operator structure of the dynamic term.

        Returns
        -------
        list of (operator, modulation)
            One ``(static_operator, scalar_modulation)`` pair when
            :meth:`time_dependent` returns an expression, else an empty list.
        """
        from quchip.declarative.expr import compile_expr

        a_ops, b_ops = self._endpoint_ops()
        expr = self.time_dependent(a_ops, b_ops)
        if expr is None:
            return []
        self._validate_dynamic_expr(expr, "time_dependent")
        method_name = "time_dependent"
        if chip.resolve_rwa(self):
            expr = self.rwa_time_dependent(a_ops, b_ops)
            self._validate_dynamic_expr(expr, "rwa_time_dependent")
            method_name = "rwa_time_dependent"
        self._check_endpoint_order(expr, method_name)
        from quchip.backend import get_default_backend
        from quchip.engine.ir import as_scalar_modulation

        source = expr.dynamic_sources[0].source
        mod_signal = as_scalar_modulation(source, owner=type(self).__name__)
        static_expr = expr.without_dynamic_sources()
        backend = get_default_backend()
        return [(compile_expr(static_expr, self._endpoint_lookup(), backend), mod_signal)]
