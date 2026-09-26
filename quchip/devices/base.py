"""Base class for finite local quantum devices.

Devices own local Hamiltonians on an authored :class:`LocalSpace`; couplings
and drives contribute separate terms. :meth:`unresolved_hamiltonian` returns
the authored operator, while :meth:`hamiltonian` applies local basis and frame
policies. Each model must state its approximations and physical references.

Drive channels use the device's physical lowering, raising, and number
operators. ``sigma_x``, ``sigma_y``, and ``sigma_z`` act on the two lowest
isolated energy states of the current local Hamiltonian.

Numerical physics parameters may be JAX tracers; structural dimensions and
settings stay fixed during tracing. Validation must avoid concretizing traced
values; use :func:`quchip.utils.jax_utils.maybe_concrete_scalar` for concrete
checks. :class:`~quchip.utils.state_versioning.StateVersioned` tracks attribute
writes, and :class:`~quchip.utils.registry.Registrable` dispatches deserialization.

Frequencies and energies ``E/h`` are in GHz, times in ns, and temperatures in
mK. Engine assembly converts Hamiltonians to angular frequency.
"""

from __future__ import annotations

from types import MemberDescriptorType
from operator import index
import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Mapping, Self, TypeVar

import jax.numpy as jnp

from quchip.backend import get_default_backend
from quchip.backend.protocol import Operator, State
from quchip.declarative.dissipation import CollapseChannel
from quchip.declarative.parameters import UNBOUND, parameter
from quchip.utils.deprecation import warn_renamed
from quchip.utils.jax_utils import maybe_concrete_scalar
from quchip.utils.labeling import auto_label
from quchip.utils.registry import Registrable
from quchip.utils.state_versioning import StateVersioned

if TYPE_CHECKING:
    from quchip.control.drive import BaseDrive
    from quchip.devices.spaces import LocalSpace, TruncationBoundary
    from quchip.engine.basis import BasisRecord
    from quchip.engine.ir import EngineResult, FrameSpec


# Common device noise fields. Declarative model constructors expose these as
# keyword-only arguments; plain BaseDevice subclasses use the same names in
# their hand-written constructors.
_NOISE_FIELDS: tuple[str, ...] = (
    "T1",
    "T2",
    "thermal_occupation",
)


def _validate_level_pair(lower: Any, upper: Any, dimension: int) -> None:
    """Validate an ordered pair of energy-level indices."""
    for name, value in (("lower", lower), ("upper", upper)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer level index, got {type(value).__name__}.")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}.")
        if value >= dimension:
            raise ValueError(
                f"{name} level {value} exceeds device dimension {dimension}."
            )
    if lower >= upper:
        raise ValueError(
            f"Transition levels must satisfy lower < upper, got {lower} and {upper}."
        )


def _energy_basis_for_noise(device: Any) -> "BasisRecord":
    from quchip.engine.basis import resolve_device_basis

    levels = device.projection_levels or device.local_space().dimension
    return resolve_device_basis(device, basis="eigen", levels=levels)


def _semantic_level_operator(basis: "BasisRecord", operator: Any) -> Any:
    """Express an energy-level operator in the authored local basis."""
    vectors = basis.energy_vectors[:, : basis.resolved_dim]
    return vectors @ operator @ vectors.conj().T


def _matrix_element_emission_channel(
    device: Any,
    p: Any,
) -> list[CollapseChannel]:
    """Matrix-element-weighted relaxation in the local energy ordering."""
    record = _energy_basis_for_noise(device)
    if device.collapse_model == "ladder":
        dimension = record.resolved_dim
        lower = jnp.diag(jnp.sqrt(jnp.arange(1, dimension)), 1).astype(jnp.complex128)
        authored_lower = _semantic_level_operator(record, lower)
        rate = 1.0 / p.T1 if device.T1 is not None else 1.0
        occupation = (
            p.thermal_occupation
            if device.thermal_occupation is not None
            else device.thermal_occupation
        )
        return BaseDevice._emission_channels(
            rate,
            occupation,
            authored_lower,
            authored_lower.conj().T,
            emission_name="matrix_element_emission",
            absorption_name="matrix_element_absorption",
        ) if device.T1 is not None or device.thermal_occupation is not None else []
    if device.T1 is None:
        return []

    physical = (
        device.phase_coupling_operator()
        if device.coupling_channel == "flux"
        else device.charge_coupling_operator()
    )
    matrix_elements = record.energy_vectors.conj().T @ physical @ record.energy_vectors
    normalization = jnp.abs(matrix_elements[0, 1]) ** 2
    norm_concrete = maybe_concrete_scalar(normalization)
    if norm_concrete is not None and norm_concrete < 1e-24:
        raise ValueError("The selected coupling_channel has a dark 0-to-1 transition.")

    terms: list[CollapseChannel] = []
    vectors = record.energy_vectors
    for upper in range(1, record.resolved_dim):
        for lower_index in range(upper):
            rate_ratio = jnp.abs(matrix_elements[lower_index, upper]) ** 2 / normalization
            ratio_concrete = maybe_concrete_scalar(rate_ratio)
            if ratio_concrete is not None and ratio_concrete < device.collapse_rate_threshold:
                continue
            down = jnp.outer(vectors[:, lower_index], vectors[:, upper].conj())
            terms.extend(
                BaseDevice._emission_channels(
                    rate_ratio / p.T1,
                    (
                        p.thermal_occupation
                        if device.thermal_occupation is not None
                        else device.thermal_occupation
                    ),
                    down,
                    down.conj().T,
                    emission_name="matrix_element_emission",
                    absorption_name="matrix_element_absorption",
                )
            )
    return terms


def _energy_dephasing_channel(
    device: Any,
    p: Any,
) -> list[CollapseChannel]:
    gamma_phi = BaseDevice._dephasing_rate(device.T1, device.T2)
    if gamma_phi is None:
        return []
    record = _energy_basis_for_noise(device)
    level_index = jnp.diag(jnp.arange(record.resolved_dim, dtype=jnp.complex128))
    symbolic_gamma = 1.0 / p.T2
    if device.T1 is not None:
        symbolic_gamma = symbolic_gamma - 1.0 / (2.0 * p.T1)
    return [
        CollapseChannel(
            _semantic_level_operator(record, level_index),
            2.0 * symbolic_gamma,
            "pure_dephasing",
        )
    ]

#: Self-type for fluent helpers (e.g. ``_restore_reference_freq``) so a
#: ``from_dict`` returning ``cls(...)._restore_reference_freq(d)`` keeps the
#: concrete subclass type.
_DeviceT = TypeVar("_DeviceT", bound="BaseDevice")


class BaseDevice(StateVersioned, Registrable, ABC, registry_root=True):
    """Abstract truncated-Hilbert-space quantum device.

    Concrete subclasses must:

    1. Set ``_type_prefix`` (used for auto-labeling).
    2. Expose a ``freq`` attribute — the bare ``0 -> 1`` transition
       frequency in GHz. Any JAX-traceable scalar is fine.
    3. Implement :meth:`unresolved_hamiltonian` on the authored local space.

    Noise parameters (all optional; ``None`` means the channel is absent):

    * ``T1`` — relaxation time (ns); emission channel at rate ``1/T1``.
    * ``T2`` — total 0-1 coherence time (ns, requires ``T2 <= 2*T1``);
      adds pure dephasing at ``gamma_phi = 1/T2 - 1/(2*T1)``.
    * ``thermal_occupation`` — unitless bath occupation ``n̄``; adds thermal
      absorption and enhances emission.

    They are ordinary attributes: set them at construction or at any time
    after. A newly built calculation uses the current noise parameters;
    existing calculations retain their captured values. Writes get the same
    validation as construction. Setting a parameter to ``None`` removes its
    channel from subsequent calculations.

    Mutation tracking is enabled automatically once construction finishes
    (see :class:`~quchip.utils.state_versioning.StateVersioned`); subclasses
    do not call ``_finish_init`` themselves.

    Optional overrides:

    * :meth:`dissipation` — append channels beyond ``T1``/``T2``.
    * :meth:`to_dict` / :meth:`from_dict` — for extra parameters.
    * :attr:`computational` — ``True`` if the device represents a
      computational qubit (default ``False``).

    See module docstring for the full contract.
    """

    _type_prefix: ClassVar[str] = "device"

    T1: Any = parameter(default=None, positive=True, unit="ns", noise=True, kw_only=True)
    T2: Any = parameter(default=None, positive=True, unit="ns", noise=True, kw_only=True)
    thermal_occupation: Any = parameter(
        default=None,
        nonnegative=True,
        noise=True,
        kw_only=True,
    )

    #: Parameters eligible for inverse design. DeviceModel derives these
    #: from declared fields unless the class or an ancestor supplies a tuple.
    #: Explicit tuples remain authoritative when inherited; an empty tuple
    #: excludes the device from inverse design. Plain BaseDevice subclasses
    #: default to an empty tuple without derivation.
    tunable_param_names: ClassVar[tuple[str, ...]] = ()

    #: ``(dressed_observable, declared_field)`` pairs used when this device
    #: appears in the desired-chip form of ``fit_a_dress``.  Empty means that
    #: the model makes no automatic dressed-target claim; circuit-level models
    #: can remain fixed until the user supplies explicit constraints.
    dressed_fit_target_fields: ClassVar[tuple[tuple[str, str], ...]] = ()

    #: Bare parameters normally varied to reproduce
    #: :attr:`dressed_fit_target_fields`. This remains separate from
    #: :attr:`tunable_param_names`: a model may expose parameters for sweeps
    #: without claiming that inverse design can identify all of them from its
    #: default dressed observables.
    dressed_fit_param_names: ClassVar[tuple[str, ...]] = ()

    # Labels are fixed identity metadata. Other public assignments are tracked.
    _untracked_names = frozenset({"label"})

    # Per-device readout/rotating-frame reference override; ``None`` inherits
    # the calculation's dressed reference. Class-level default remains safe on the
    # JAX-pytree ``_unflatten`` path (which bypasses ``__init__``).
    _reference_freq_override: Any = None
    basis: Literal["native", "eigen"] | None = None
    projection_levels: int | None = None
    requires_projection_levels: ClassVar[bool] = False

    def __init__(
        self,
        levels: int,
        label: str | None = None,
        *,
        T1: float | None = None,
        T2: float | None = None,
        thermal_occupation: float | None = None,
        thermal_population: Any = UNBOUND,
    ) -> None:
        if thermal_population is not UNBOUND:
            if thermal_occupation is not None:
                raise TypeError("Pass thermal_occupation only, not both thermal occupation names.")
            warn_renamed("thermal_population", "thermal_occupation")
            thermal_occupation = thermal_population
        if index(levels) < 2:
            raise ValueError(f"levels must be >= 2, got {levels}")
        self.levels = levels

        _validate_noise_params(T1, T2, thermal_occupation)
        self.T1 = T1
        self.T2 = T2
        self.thermal_occupation = thermal_occupation

        self.label = label if label is not None else auto_label(type(self)._type_prefix)

        self._connected_drives: list[BaseDrive] = []
        self._reference_freq_override: Any = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Give post-construction writes the same validation as the constructor.

        Construction validates jointly while mutation tracking is still off
        (``__init__`` / the declarative resolver); once tracking is live,
        every public write runs :meth:`_validate_param_write` *before* the
        attribute lands, so a rejected value never sticks. Checks apply to
        concrete scalars only — traced writes flow through unchecked.
        The JAX pytree ``_unflatten`` path uses
        ``object.__setattr__`` and bypasses this hook entirely.
        """
        if name == "thermal_population":
            warn_renamed(name, "thermal_occupation")
            name = "thermal_occupation"
        if getattr(self, "_tracking_enabled", False):
            if name == "label":
                raise AttributeError("Device label is fixed at construction; create a replacement device.")
            if not name.startswith("_"):
                self._validate_param_write(name, value)
        super().__setattr__(name, value)

    @property
    def thermal_population(self) -> Any:
        """Deprecated alias for the bath's mean thermal occupation."""
        warn_renamed("thermal_population", "thermal_occupation")
        return self.thermal_occupation

    @thermal_population.setter
    def thermal_population(self, value: Any) -> None:
        warn_renamed("thermal_population", "thermal_occupation")
        self.thermal_occupation = value

    @staticmethod
    def _normalize_parameter_names(values: Mapping[str, Any]) -> dict[str, Any]:
        """Read legacy thermal occupation inputs without duplicating parameter state."""
        values = dict(values)
        if "thermal_population" in values:
            if "thermal_occupation" in values:
                raise TypeError("Pass thermal_occupation only, not both thermal occupation names.")
            warn_renamed("thermal_population", "thermal_occupation")
            values["thermal_occupation"] = values.pop("thermal_population")
        return values

    def _validate_param_write(self, name: str, value: Any) -> None:
        """Constructor-grade validation for one post-construction write.

        The base class checks the noise fields jointly — the same
        :func:`_validate_noise_params` the constructor runs, with *value*
        substituted for the field being written. Without this, e.g.
        ``q.T2 = 3 * q.T1`` after construction would not raise but silently
        drop the pure-dephasing channel (its implied rate goes negative).
        Subclasses extend (``DeviceModel`` adds declared-parameter sign
        checks) and must call ``super()``.
        """
        if name == "levels" and index(value) < 2:
            raise ValueError(f"levels must be >= 2, got {value}")
        if name in ("basis", "projection_levels"):
            self._validate_basis_request(
                basis=value if name == "basis" else self.basis,
                levels=value if name == "projection_levels" else self.projection_levels,
                native_dimension=self.local_space().dimension,
            )
        if name in _NOISE_FIELDS:
            candidate = {field: getattr(self, field, None) for field in _NOISE_FIELDS}
            candidate[name] = value
            _validate_noise_params(**candidate)
        if name == "collapse_model" and value not in ("fermi_golden", "ladder"):
            raise ValueError(
                f"collapse_model must be 'fermi_golden' or 'ladder', got {value!r}"
            )
        if name in ("T1", "collapse_model", "coupling_channel") and hasattr(
            self, "collapse_model"
        ):
            model = value if name == "collapse_model" else self.collapse_model
            t1 = value if name == "T1" else self.T1
            channel = value if name == "coupling_channel" else getattr(
                self, "coupling_channel", None
            )
            if model == "fermi_golden" and t1 is not None and channel is None:
                raise ValueError(
                    "coupling_channel is required when T1 uses matrix-element relaxation."
                )

    # -- Bare-parameter introspection (inverse design / autodiff) ------------

    def tunable_params(self) -> dict[str, Any]:
        """Return ``{name: current_value}`` for every bare parameter the
        device exposes for fitting / sweeping.

        The default implementation walks :attr:`tunable_param_names` and
        reads each attribute. Subclasses with derived bare parameters
        (e.g. circuit-level devices whose ``freq`` is computed from
        ``E_C``/``E_J``/``E_L``) should override the class attribute
        rather than this method — overrides are the right hook only when
        the *list* itself is not static (e.g. flux-tunable devices that
        gain ``phi_ext`` only at certain operating points).
        """
        return {name: getattr(self, name) for name in self.tunable_param_names}

    def default_dressed_targets(self) -> dict[str, Any]:
        """Return declared numbers interpreted as dressed-fit targets.

        This hook reads component fields and never diagonalizes a chip. A
        device class opts in by declaring
        :attr:`dressed_fit_target_fields`.
        """
        return {
            observable: getattr(self, field)
            for observable, field in self.dressed_fit_target_fields
        }

    def default_fit_parameters(self) -> tuple[str, ...]:
        """Return the conservative bare-parameter selection for default targets."""
        return self.dressed_fit_param_names

    def set_tunable_param(self, name: str, value: Any) -> None:
        """Update a bare parameter named in :meth:`tunable_params`.

        Parameters
        ----------
        name : str
            Tunable field name.
        value : Any
            Replacement numerical value.

        Notes
        -----
        Default implementation uses :func:`setattr` so any direct
        attribute (``freq``, ``anharmonicity``, ``E_C``, …) works
        without ceremony. Subclasses with derived properties that need
        to back-propagate to private state should override this.
        """
        if name not in self.tunable_param_names:
            raise ValueError(
                f"{type(self).__name__} does not expose {name!r} as a tunable "
                f"parameter. Allowed: {list(self.tunable_param_names)}"
            )
        setattr(self, name, value)

    def tunable_param_bounds(self, name: str, value: float) -> tuple[float, float]:
        """Return ``(lower, upper)`` bounds for a tunable parameter at a seed value.

        Parameters
        ----------
        name : str
            Tunable parameter name.
        value : float
            Concrete seed used to construct bounds.

        Notes
        -----
        These bounds are consumed by the inverse-design optimizer to keep
        searches physical. The default uses well-named conventions that
        cover the common circuit-QED parameters:

        * ``freq``, ``E_C``, ``E_J``, ``E_L``: positive, ``[0.5·s, 1.5·s]``
          around a positive seed (``s``).
        * ``anharmonicity``: sign-preserving — negative seeds bound in
          ``(2·s, -ε)``, positive seeds in ``(ε, 2·s)``.
        * ``phi_ext``: in ``[-0.5, 0.5]`` (one full flux period symmetric
          around the integer-flux point).

        Subclasses override for parameters with other physical
        constraints. Raises :class:`ValueError` for unknown names rather
        than silently optimizing over an unbounded axis, and for a *value*
        that is not a concrete real scalar (bounds for a JAX tracer are
        undefined — the optimizer needs a concrete numeric seed).
        """
        seed = maybe_concrete_scalar(value)
        if seed is None:
            raise ValueError(
                f"tunable_param_bounds({name!r}, {value!r}) requires a concrete real scalar "
                "seed; optimizer bounds cannot be computed from a JAX tracer."
            )
        if name in {"freq", "E_C", "E_J", "E_L"}:
            if seed <= 0:
                raise ValueError(f"{name} seed must be positive, got {seed}")
            return (max(1e-6, 0.5 * seed), 1.5 * seed)
        if name == "anharmonicity":
            if seed < 0:
                return (2.0 * seed, -1e-6)
            return (1e-6, 2.0 * seed if seed > 0 else 1.0)
        if name == "phi_ext":
            return (-0.5, 0.5)
        raise ValueError(
            f"{type(self).__name__} has no bounds rule for tunable parameter {name!r}; "
            "override tunable_param_bounds()."
        )

    @property
    def connected_drives(self) -> list["BaseDrive"]:
        """Drives wired to this device, as a fresh list (mutation-safe copy)."""
        return list(self._connected_drives)

    def copy(self) -> Self:
        """Structural copy detached from drive wiring (used by sweep cloning)."""
        from quchip.declarative.parameters import copy_authored_fields

        cloned = copy.copy(self)
        copy_authored_fields(self, cloned)
        object.__setattr__(cloned, "_connected_drives", [])
        return cloned

    def parameter_values(self) -> dict[str, Any]:
        """Return declared bindable values, including inactive optional fields."""
        from quchip.declarative.parameters import parameter_fields

        values = dict(self.tunable_params())
        values.update((name, getattr(self, name)) for name in parameter_fields(type(self)))
        return values

    def set_parameter_value(self, name: str, value: Any) -> None:
        """Apply one validated local parameter value on an isolated device copy.

        Parameters
        ----------
        name : str
            Declared or tunable parameter name.
        value : Any
            Replacement value.
        """
        if name == "thermal_population":
            warn_renamed(name, "thermal_occupation")
            name = "thermal_occupation"
        tunable = self.tunable_params()
        if name in tunable:
            self.set_tunable_param(name, value)
            return
        from quchip.declarative.parameters import parameter_fields

        if name in parameter_fields(type(self)):
            setattr(self, name, value)
            return
        raise KeyError(name)

    def set_parameter_values(self, values: Mapping[str, Any]) -> None:
        """Validate a complete candidate before applying a local parameter group.

        Parameters
        ----------
        values : mapping[str, Any]
            Parameter names and replacement values, applied atomically.
        """
        values = self._normalize_parameter_names(values)
        if not values:
            return
        from quchip.utils.jax_utils import contains_tracer

        if contains_tracer(tuple(values.values())):
            changed = [
                name for name, value in values.items()
                if (getattr(self, name, None) is None) != (value is None)
            ]
            if changed:
                raise ValueError(
                    f"Activate or deactivate optional parameters {changed} with concrete values before tracing."
                )
        self._commit_parameter_candidate(self._parameter_candidate(values))

    def _parameter_candidate(self, values: Mapping[str, Any]) -> "BaseDevice":
        """Prepare validated state without changing this device or its ownership."""
        from quchip.declarative.parameters import copy_authored_fields

        candidate = copy.copy(self)
        copy_authored_fields(self, candidate)
        object.__setattr__(candidate, "_tracking_enabled", False)
        for name, value in values.items():
            candidate.set_parameter_value(name, value)
        for name in values:
            candidate._validate_param_write(name, getattr(candidate, name))
        candidate.validate()
        object.__setattr__(candidate, "_tracking_enabled", self._tracking_enabled)
        object.__setattr__(candidate, "_state_version", self.state_version + 1)
        return candidate

    def _commit_parameter_candidate(self, candidate: "BaseDevice") -> None:
        """Adopt validated attribute storage, including slots declared by extensions."""
        for cls in type(self).__mro__:
            for descriptor in vars(cls).values():
                if isinstance(descriptor, MemberDescriptorType) and hasattr(candidate, descriptor.__name__):
                    descriptor.__set__(self, descriptor.__get__(candidate))
        object.__setattr__(self, "__dict__", candidate.__dict__)

    def validate(self) -> None:
        """Validate cross-field constraints at construction and grouped rebinding.

        Subclasses implement their joint constraints here. Gate numerical
        checks on concrete scalars so traced parameters remain supported.
        """

    # -- Drive lookup -------------------------------------------------------

    def __getitem__(self, key: str) -> "BaseDrive":
        for drv in self._connected_drives:
            if drv.label == key:
                return drv
        available = [d.label for d in self._connected_drives]
        raise KeyError(f"No drive {key!r} on device {self.label!r}. Available drives: {available}") from None

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return any(d.label == item for d in self._connected_drives)
        return item in self._connected_drives

    @property
    def reference_freq(self) -> Any:
        """Return the authored readout/frame reference in GHz, or ``None``.

        ``None`` lets each chip calculation resolve its dressed reference.
        An explicit value fixes the readout reference independently of pulse
        carriers and is captured in each calculation's resolved frame.
        """
        return self._reference_freq_override

    @reference_freq.setter
    def reference_freq(self, value: Any | None) -> None:
        """Set the readout/rotating-frame reference (``None`` restores automatic resolution)."""
        self._reference_freq_override = value

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization; subclasses extend with their own parameters."""
        data = super().to_dict()
        data["levels"] = int(self.levels)
        data["label"] = self.label
        for attr in _NOISE_FIELDS:
            value = getattr(self, attr)
            if value is not None:
                data[attr] = float(value)
        # Persist an explicit reference_freq override (not the resolved
        # default). Skip a traced override — it has no concrete serializable
        # value, matching how the rest of to_dict emits concrete scalars only.
        override = self._reference_freq_override
        if override is not None:
            override_value = maybe_concrete_scalar(override)
            if override_value is not None:
                data["reference_freq"] = float(override_value)
        return data

    @staticmethod
    def _noise_kwargs_from_dict(d: dict[str, Any]) -> dict[str, Any]:
        """Pull noise kwargs out of a serialization dict (helper for subclass from_dict)."""
        return {field: d.get(field) for field in _NOISE_FIELDS}

    def _restore_reference_freq(self: _DeviceT, d: dict[str, Any]) -> _DeviceT:
        """Restore a serialized ``reference_freq`` override (helper for from_dict).

        Sets the override only when the key is present (an absent key keeps the
        automatic default). Returns ``self`` (typed as the concrete
        subclass) so ``from_dict`` can ``return cls(...)._restore_reference_freq(d)``.
        """
        if "reference_freq" in d:
            self.reference_freq = d["reference_freq"]
        return self

    # -- Hamiltonian -------------------------------------------------------

    @abstractmethod
    def unresolved_hamiltonian(self) -> Operator:
        """Return the authored local Hamiltonian before engine policies."""
        ...

    def hamiltonian(self) -> Any:
        """Return the local Hamiltonian after basis and frame policies."""
        return self.resolve().hamiltonian()

    def resolve(self, *, frame: FrameSpec | None = None) -> EngineResult:
        """Resolve isolated device physics with its own basis and explicit frame.

        Parameters
        ----------
        frame : FrameSpec or None, keyword-only
            Explicit rotating-frame specification; ``None`` uses the lab frame.

        The default frame is the lab frame. Chip membership does not affect
        this local calculation; use the chip for coupled-system queries.
        """
        from quchip.chip.chip import Chip

        return Chip([self.copy()]).resolve(frame=frame)

    # -- Declared approximations --------------------------------------------

    def truncation_boundary(self) -> TruncationBoundary | None:
        """Return the authored-space cutoff used by sampled truncation checks.

        Intrinsically finite models override this to return None. Custom spaces
        with a numerical cutoff return a TruncationBoundary describing its indices.
        """
        return self.local_space().truncation_boundary()

    def _truncation_note(self) -> str:
        """Return the Hilbert-truncation physics note.

        Default states the Fock-basis truncation. Models with another authored
        local space override this hook so the remaining physics notes stay
        shared.
        """
        return f"Hilbert truncation: {self.levels} Fock levels"

    def physics_notes(self) -> list[str]:
        """Return human-readable declarations of this device's approximations.

        Each entry names a non-obvious assumption, approximation, or
        truncation that a user of this device should be aware of — e.g.
        "Hilbert truncation: 3 levels", a model regime (Duffing), or a
        noise-channel selection (charge- vs flux-coupled T1).

        The baseline entry is :meth:`_truncation_note`, since every
        :class:`BaseDevice` has some form of Hilbert-space truncation. A
        pure-dephasing note is added when ``T2`` is set, since the
        number-operator dephasing model carries non-obvious assumptions.
        Subclasses ``super().physics_notes()`` and append their own
        model-specific notes; no registry / engine-side dispatch is needed.
        """
        notes = [self._truncation_note()]
        if self.T2 is not None:
            notes.append(
                "Pure dephasing couples to the level-number operator "
                "(rate scales as (m-n)^2 across levels); the input T2 equals "
                "the resulting 0-1 coherence time only when thermal_occupation "
                "is 0."
            )
        return notes

    def _time_terms(self) -> tuple[Any, ...]:
        """Return normalized time-dependent terms (default: none)."""
        return ()

    # -- Fock-space operator defaults --------------------------------------

    def local_space(self) -> "LocalSpace":
        """Return this device's authored local operator space."""
        from quchip.devices.spaces import FockSpace

        return FockSpace(self.levels)

    def resolved_basis(
        self,
        chip_basis: Literal["native", "eigen"] = "native",
    ) -> Literal["native", "eigen"]:
        """Return the device override or inherited chip basis policy.

        Parameters
        ----------
        chip_basis : {"native", "eigen"}, default "native"
            Fallback policy when the device has no explicit basis setting.
        """
        policy = self.basis if self.basis is not None else chip_basis
        if policy not in ("native", "eigen"):
            raise ValueError(f"basis must be 'native', 'eigen', or None, got {policy!r}.")
        return policy

    @classmethod
    def _validate_basis_request(
        cls,
        *,
        basis: Literal["native", "eigen"] | None,
        levels: int | None,
        native_dimension: int,
    ) -> None:
        """Validate one local-basis policy against its authored dimension."""
        if basis not in (None, "native", "eigen"):
            raise ValueError(f"basis must be 'native', 'eigen', or None, got {basis!r}")
        if basis == "native" and levels is not None:
            raise ValueError("levels is not valid when basis='native'.")
        if basis == "eigen" and cls.requires_projection_levels and levels is None:
            raise ValueError("levels is required when basis='eigen'.")
        if levels is not None and not 1 <= index(levels) <= native_dimension:
            raise ValueError(
                f"levels must be between 1 and {native_dimension}, got {levels}"
            )

    def resolved_dimension(
        self,
        chip_basis: Literal["native", "eigen"] = "native",
    ) -> int:
        """Return the local dimension delivered to the solver.

        Parameters
        ----------
        chip_basis : {"native", "eigen"}, default "native"
            Fallback basis policy.
        """
        policy = self.resolved_basis(chip_basis)
        if policy == "native":
            return self.local_space().dimension
        levels = self.projection_levels
        if levels is None:
            if self.requires_projection_levels:
                raise ValueError(
                    f"{type(self).__name__} {self.label!r} requires levels when basis='eigen'."
                )
            levels = self.local_space().dimension
        if levels < 1 or levels > self.local_space().dimension:
            raise ValueError(
                f"levels must be between 1 and {self.local_space().dimension}, got {levels}."
            )
        return levels

    def lowering_operator(self) -> Operator:
        """Bosonic lowering operator ``a`` on the truncated Fock basis."""
        return self.local_space().operator("a", get_default_backend())

    def raising_operator(self) -> Operator:
        """Bosonic raising operator ``a†`` on the truncated Fock basis."""
        return self.local_space().operator("adag", get_default_backend())

    def number_operator(self) -> Operator:
        """Number operator ``n̂ = a†a`` on the truncated Fock basis."""
        return self.local_space().operator("n", get_default_backend())

    def energy_level_operator(self) -> Operator:
        """Return the energy-level index expressed in the authored local basis."""
        from quchip.engine.basis import resolve_device_basis
        import jax

        with jax.ensure_compile_time_eval():
            return resolve_device_basis(self, basis="native").level_operator()

    def identity(self) -> Operator:
        """Identity operator on the truncated Fock basis."""
        return self.local_space().operator("I", get_default_backend())

    def local_operator(self, name: str) -> Operator:
        """Map an operator-name string to this device's own local operator.

        Parameters
        ----------
        name : str
            Operator vocabulary name such as ``"X"``, ``"n"``, ``"a"``,
            ``"charge"`` or ``"I"``.

        Notes
        -----
        Recognized names: ``"X"`` / ``"Y"`` / ``"Z"`` (Pauli projections on
        the computational ``|0>, |1>`` subspace), ``"n"`` (number), ``"a"``
        (lowering), ``"a_dag"`` (raising), ``"I"`` (identity). The device owns
        this vocabulary. ``"charge"``, ``"phase"`` and ``"flux"`` use
        the corresponding physical coupling operator when declared.
        Other names are looked up in :meth:`local_space`. A subclass may
        override this method to supply additional derived operators.
        """
        physical = {
            "charge": "charge_coupling_operator",
            "phase": "phase_coupling_operator",
            "flux": "flux_coupling_operator",
        }
        method = getattr(self, physical.get(name, ""), None)
        if method is not None:
            return method()
        if name == "X":
            return self.sigma_x
        if name == "Y":
            return self.sigma_y
        if name == "Z":
            return self.sigma_z
        if name == "n":
            return self.number_operator()
        if name == "a":
            return self.lowering_operator()
        if name == "a_dag":
            return self.raising_operator()
        if name == "I":
            return self.identity()
        try:
            return self.local_space().operator(name, get_default_backend())
        except ValueError as error:
            raise ValueError(f"Unknown operator {name!r} for device {self.label!r}: {error}") from error

    def basis_state(self, n: int) -> State:
        """Return Fock basis state ``|n>``.

        Parameters
        ----------
        n : int
            Non-negative basis index below ``levels``.
        """
        return get_default_backend().basis(self.levels, n)

    def coherent_state(self, alpha: complex) -> State:
        """Return coherent state ``|alpha>`` on the truncated Fock basis.

        Parameters
        ----------
        alpha : complex
            Dimensionless coherent-state amplitude.
        """
        return get_default_backend().coherent(self.levels, alpha)

    def plot_energy_levels(self, *, ax: Any = None, **kwargs: Any) -> Any:
        """Plot the device's bare energy-level ladder.

        Parameters
        ----------
        ax : matplotlib Axes or None, keyword-only
            Existing axes, or ``None`` to create one.
        **kwargs : Any
            Plot styling forwarded to the visualization helper.
        """
        from quchip.viz.device import plot_energy_levels
        return plot_energy_levels(self, ax=ax, **kwargs)

    def plot_wavefunction(self, n: int, *, ax: Any = None, **kwargs: Any) -> Any:
        """Plot an eigenstate wavefunction.

        Parameters
        ----------
        n : int
            Eigenstate index.
        ax : matplotlib Axes or None, keyword-only
            Existing axes, or ``None`` to create one.
        **kwargs : Any
            Plot styling forwarded to the visualization helper.
        """
        from quchip.viz.device import plot_wavefunction
        return plot_wavefunction(self, n, ax=ax, **kwargs)

    # -- Pauli operators on the two lowest isolated energy states -----------

    def _pauli_operator(self, name: str, *, basis: BasisRecord | None = None) -> Operator:
        from quchip.declarative.parameters import component_fingerprint
        from quchip.devices.spaces import FockSpace
        from quchip.engine.basis import resolve_device_basis
        from quchip.utils.jax_utils import contains_tracer

        matrices: dict[str, Any] = {}
        if basis is None:
            key = component_fingerprint(self)
            cached = getattr(self, "_pauli_cache", None)
            if cached is not None and cached[0] == key:
                _, basis, matrices = cached
            else:
                basis = resolve_device_basis(self, basis="native")
                if not contains_tracer(basis.energy_vectors):
                    self._pauli_cache = (key, basis, matrices)
        assert basis is not None
        matrix = matrices.get(name)
        if matrix is None:
            vectors = basis.energy_vectors[:, :2]
            matrix = vectors @ FockSpace(2).matrix(name) @ vectors.conj().T
            # Cache immutable arrays only; each caller owns its lowered operator.
            if not contains_tracer(matrix):
                matrices[name] = matrix
        dimension = self.local_space().dimension
        return get_default_backend().from_array(matrix, dims=[[dimension], [dimension]])

    @property
    def sigma_x(self) -> Operator:
        """Return ``|0><1| + |1><0|`` on isolated energy levels, zero elsewhere."""
        return self._pauli_operator("sigma_x")

    @property
    def sigma_y(self) -> Operator:
        """Return ``-i|0><1| + i|1><0|`` on isolated energy levels, zero elsewhere."""
        return self._pauli_operator("sigma_y")

    @property
    def sigma_z(self) -> Operator:
        """Return ``|0><0| - |1><1|`` on isolated energy levels, zero elsewhere."""
        return self._pauli_operator("sigma_z")

    @property
    def sigma_plus(self) -> Operator:
        """Return ``|1><0|`` between the lowest isolated energy levels."""
        return self._pauli_operator("sigma_plus")

    @property
    def sigma_minus(self) -> Operator:
        """Return ``|0><1|`` between the lowest isolated energy levels."""
        return self._pauli_operator("sigma_minus")

    def projector(self, i: int, j: int) -> Operator:
        """``|i><j|`` on the authored local basis.

        Parameters
        ----------
        i, j : int
            Ket and bra indices in the authored local basis.

        Use ``projector(i, i)`` for the population projector
        ``|i><i|`` and ``projector(i, j)`` for ``|i><j|``. No subspace
        approximation: the operator acts on the full authored local space.
        """
        backend = get_default_backend()
        ket_i = backend.basis(self.local_space().dimension, i)
        ket_j = backend.basis(self.local_space().dimension, j)
        return backend.matmul(ket_i, backend.dag(ket_j))

    def transition(self, lower: int, upper: int) -> Operator:
        """Hermitian transition between isolated energy states.

        Parameters
        ----------
        lower, upper : int
            Isolated energy-level indices with ``lower < upper``.

        The operator ``|lower><upper| + |upper><lower|`` is returned in the
        authored local basis.
        """
        from quchip.engine.basis import resolve_device_basis

        authored_dimension = self.local_space().dimension
        _validate_level_pair(
            lower,
            upper,
            self.resolved_dimension(self.basis or "native"),
        )
        vectors = resolve_device_basis(self, basis="native").energy_vectors
        off_diagonal = jnp.outer(vectors[:, lower], jnp.conj(vectors[:, upper]))
        matrix = off_diagonal + jnp.conj(off_diagonal.T)
        return get_default_backend().from_array(
            matrix,
            dims=[[authored_dimension], [authored_dimension]],
        )

    def transition_frequency(self, lower: int, upper: int) -> Any:
        """Return the isolated ``E_upper - E_lower`` transition in GHz.

        Parameters
        ----------
        lower, upper : int
            Isolated energy-level indices with ``lower < upper``.
        """
        from quchip.engine.basis import resolve_device_basis

        _validate_level_pair(
            lower,
            upper,
            self.resolved_dimension(self.basis or "native"),
        )
        energies = resolve_device_basis(self, basis="native").energies
        return energies[upper] - energies[lower]

    # -- Classification ----------------------------------------------------

    @property
    def computational(self) -> bool:
        """Whether this device is a computational qubit. Override in subclasses."""
        return False

    # -- Collapse operators ------------------------------------------------

    def dissipation(self, op: Any, p: Any) -> tuple[CollapseChannel, ...]:
        """Return T1/T2 channels using the device's lowering, raising and number hooks."""
        def operator(name: str, hook: str) -> Any:
            # Keep inherited operators symbolic for parameter discovery and
            # linear-response lowering; overridden hooks own their physics.
            if getattr(type(self), hook) is getattr(BaseDevice, hook):
                return op[name]
            return getattr(self, hook)()

        channels: list[CollapseChannel] = []
        if self.T1 is not None or self.thermal_occupation is not None:
            occupation = 0.0 if self.thermal_occupation is None else p.thermal_occupation
            base_rate = 1.0 / p.T1 if self.T1 is not None else 1.0
            channels.append(
                CollapseChannel(operator("a", "lowering_operator"),
                                base_rate * (occupation + 1.0), "thermal_emission")
            )
            occupation_value = maybe_concrete_scalar(
                0.0 if self.thermal_occupation is None else self.thermal_occupation
            )
            if occupation_value is None or occupation_value > 0:
                channels.append(
                    CollapseChannel(operator("adag", "raising_operator"),
                                    base_rate * occupation, "thermal_absorption")
                )
        gamma_phi = self._dephasing_rate(self.T1, self.T2)
        if gamma_phi is not None:
            symbolic_gamma = 1.0 / p.T2
            if self.T1 is not None:
                symbolic_gamma = symbolic_gamma - 1.0 / (2.0 * p.T1)
            channels.append(CollapseChannel(operator("n", "number_operator"),
                                            2.0 * symbolic_gamma, "pure_dephasing"))
        return tuple(channels)

    def _collapse_channels_with_paths(
        self,
        basis: "BasisRecord | None" = None,
    ) -> tuple[tuple[CollapseChannel, tuple[str, ...]], ...]:
        """Normalize authored local dissipation and infer parameter paths."""
        from quchip.declarative.dissipation import normalize_dissipation
        from quchip.declarative.expr import ParameterNamespace
        from quchip.declarative.ops import LocalOps
        from quchip.declarative.parameters import parameter_fields

        del basis
        space = self.local_space()
        op = LocalOps(label=self.label, space=space, device=self)
        fields = parameter_fields(type(self))
        p = ParameterNamespace(self.label, fields)
        bindings = {
            f"{self.label}.{name}": value
            for name in fields
            if (value := getattr(self, name)) is not None
        }
        return normalize_dissipation(
            self.dissipation(op, p),
            labels=(self.label,),
            dims=(space.dimension,),
            owner=self,
            scope=self.label,
            allowed=fields,
            bindings=bindings,
        )

    def collapse_operators(self) -> list[Operator]:
        """Materialize the device's authored Lindblad collapse operators.

        The built-in channels cover ``T1``, ``T2``, and
        ``thermal_occupation``; subclasses append channels in
        :meth:`dissipation`.

        References
        ----------
        Breuer & Petruccione, *Theory of Open Quantum Systems* (Oxford,
        2002), Ch. 3. For circuit-QED conventions see Krantz et al.,
        *Applied Physics Reviews* **6**, 021318 (2019), §V.
        """
        from quchip.declarative.expr import materialize_expr
        from quchip.engine.basis import resolve_device_basis

        backend = get_default_backend()
        policy = self.resolved_basis()
        levels = self.resolved_dimension() if policy == "eigen" else None
        basis = resolve_device_basis(self, basis=policy, levels=levels)
        dims = [[basis.resolved_dim], [basis.resolved_dim]]
        operators: list[Operator] = []
        for channel, _paths in self._collapse_channels_with_paths(basis):
            authored = materialize_expr(channel.operator, backend)
            if basis.kind == "native":
                native = authored
            else:
                projected = basis.transform_operator(backend.to_array(authored))
                native = backend.from_array(projected, dims=dims)
            rate = materialize_expr(channel.rate, backend)
            operators.append(jnp.sqrt(rate) * native)
        return operators

    def collapse_channels(
        self,
        basis: "BasisRecord | None" = None,
    ) -> tuple[CollapseChannel, ...]:
        """Return normalized local collapse channels.

        Parameters
        ----------
        basis : BasisRecord or None, optional
            Resolved basis for matrix-element channels; ``None`` resolves the
            device's current basis policy.
        """
        return tuple(channel for channel, _paths in self._collapse_channels_with_paths(basis))

    @classmethod
    def noise_parameter_names(cls) -> tuple[str, ...]:
        """Declared fields that :meth:`Chip.set_noise` may configure."""
        from quchip.declarative.parameters import parameter_fields

        return tuple(
            name for name, spec in parameter_fields(cls).items() if spec.noise
        )

    def intrinsic_decay_rate(self) -> Any | None:
        """Total lowering-channel (downward) Lindblad rate, in 1/ns, or ``None`` with no decay channel.

        Reports the actual sum of squared amplitudes of the lowering-operator
        collapse channel(s) :meth:`collapse_operators` builds from
        the common thermal-emission construction exactly
        rather than approximating it:

        * ``T1`` set (``thermal_occupation`` set or not): ``(n̄+1)/T1`` —
          the ``sqrt(gamma*(n̄+1))·a`` channel's rate, ``gamma = 1/T1``;
          ``n̄`` defaults to ``0`` when ``thermal_occupation`` is unset, so
          this reduces to plain ``1/T1``.
        * ``T1`` unset, ``thermal_occupation`` set: ``n̄+1`` — the same
          channel with ``gamma = 1`` (the unitless-bath-occupation branch).
        * Neither set: ``None`` — no lowering channel.

        Subclasses whose :meth:`collapse_operators` combine several
        lowering-operator channels (e.g. :class:`~quchip.devices.resonator.Resonator`'s
        Q-derived photon loss alongside ``T1``) override this to report the
        summed rate, so a caller reading a single scalar decay rate (e.g.
        :mod:`quchip.chip.transformations.eliminate_device`'s Purcell fold)
        does not have to special-case per-device channel structure.

        This is the *downward* rate only — the ``sqrt(gamma*n̄)·a†`` upward
        (thermal-absorption) channel is not represented; a caller that needs
        to know whether that channel is present reads ``thermal_occupation``
        directly. Whether a channel exists, and which formula applies, is a
        *static* decision (is ``T1``/``thermal_occupation`` set?), never a
        traced-zero comparison on the resulting rate, which would concretize
        a traced value and break differentiability.
        """
        n_bar = self.thermal_occupation
        if self.T1 is not None:
            n_bar_eff = 0.0 if n_bar is None else n_bar
            return (n_bar_eff + 1.0) / self.T1
        if n_bar is not None:
            return n_bar + 1.0
        return None

    # -- Shared Lindblad rate algebra ----------------------------------------

    @staticmethod
    def _emission_channels(
        rate: Any,
        n_bar: Any | None,
        lower_op: Any,
        raise_op: Any,
        *,
        emission_name: str,
        absorption_name: str,
    ) -> list[CollapseChannel]:
        """Emission and absorption channels for one transition.

        Returns ``(lower_op, rate * (n_bar + 1))`` (relaxation / stimulated
        emission) and, when the bath occupation is non-zero, additionally
        ``(raise_op, rate * n_bar)`` (thermal absorption).
        ``n_bar is None`` is treated as zero occupation, yielding the down
        channel only. The positivity gate reads a *concrete* scalar only, so
        a traced ``n_bar`` keeps both channels. The math is
        pure :mod:`jax.numpy`, so the result type follows the supplied
        operators.
        """
        n_bar_eff = 0.0 if n_bar is None else n_bar
        terms = [
            CollapseChannel(
                lower_op,
                rate * (n_bar_eff + 1.0),
                emission_name,
            )
        ]
        n_bar_value = maybe_concrete_scalar(n_bar_eff)
        if n_bar_value is None or n_bar_value > 0:
            terms.append(
                CollapseChannel(
                    raise_op,
                    rate * n_bar_eff,
                    absorption_name,
                )
            )
        return terms

    @staticmethod
    def _dephasing_rate(T1: Any | None, T2: Any | None) -> Any | None:
        """Clamped pure-dephasing rate ``gamma_phi``, or ``None`` when absent.

        ``gamma_phi = 1/T2 - 1/(2*T1)`` when ``T1`` is set, or ``1/T2`` when
        ``T1`` is ``None`` (with no T1 subtraction). Returns ``None`` when
        there is no dephasing channel: either
        ``T2`` is unset, or ``gamma_phi`` is a *concrete* non-positive scalar.
        A traced ``gamma_phi`` is kept and clamped via :func:`jax.numpy.maximum`.
        The construction constraint ``T2 <= 2*T1`` already
        guarantees ``gamma_phi >= 0`` for concrete inputs, so the clamp only
        guards traced values.
        """
        if T2 is None:
            return None
        gamma_phi = 1.0 / T2
        if T1 is not None:
            gamma_phi = gamma_phi - 1.0 / (2.0 * T1)
        gamma_phi_value = maybe_concrete_scalar(gamma_phi)
        if gamma_phi_value is not None and gamma_phi_value <= 0:
            return None
        return jnp.maximum(gamma_phi, 0.0)

    # -- Drive wiring ------------------------------------------------------

    def connect(self, drive: "BaseDrive") -> None:
        """Register a drive as connected: idempotent on identity, replace-on-relabel.

        Parameters
        ----------
        drive : BaseDrive
            Drive object to attach.

        A drive's label is its stable identity as a control line
        (``chip.wire`` already rejects duplicate labels within one
        equipment). Clone-and-rewire flows — ``chip.clone()``,
        ``eliminate()``'s equipment reattachment, ``chip.partition()`` —
        build a *fresh* drive object bound to the same label when
        re-wiring a device that already carries a connected drive, so a
        same-label, different-object entry marks a stale copy of the same
        line rather than a second physical line. That stale entry is
        replaced in place (position preserved); a drive with a distinct
        label is always appended as an independent line.
        """
        for i, existing in enumerate(self._connected_drives):
            if existing is drive:
                return
            if existing.label == drive.label:
                self._connected_drives[i] = drive
                return
        self._connected_drives.append(drive)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(label={self.label!r}, "
            f"freq={getattr(self, 'freq', None)!r}, levels={self.levels})"
        )


def _validate_noise_params(
    T1: float | None,
    T2: float | None,
    thermal_occupation: float | None,
) -> None:
    """Validate T1 / T2 / thermal_occupation on concrete scalars only (JAX-safe)."""
    T1_value = maybe_concrete_scalar(T1)
    T2_value = maybe_concrete_scalar(T2)
    thermal_value = maybe_concrete_scalar(thermal_occupation)

    if T1_value is not None and T1_value <= 0:
        raise ValueError(f"T1 must be positive, got {T1}")
    if T2 is not None:
        if T2_value is not None and T2_value <= 0:
            raise ValueError(f"T2 must be positive, got {T2}")
        if T1_value is not None and T2_value is not None and T2_value > 2 * T1_value:
            raise ValueError(
                f"T2 must satisfy T2 <= 2*T1; got T2={T2}, T1={T1} (implied gamma_phi would be negative)"
            )
    if thermal_value is not None and thermal_value < 0:
        raise ValueError(
            f"thermal_occupation must be non-negative, got {thermal_occupation}"
        )
