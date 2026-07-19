"""Base device model for quchip.

A device is a truncated-Hilbert-space quantum system owned by a chip.
Subclasses declare their local Hamiltonian on a Fock basis of size
``levels``.

Contract
--------
* **Hamiltonian ownership.** A device owns its *local* Hamiltonian only;
  couplings and drives own theirs. :meth:`hamiltonian`
  must return an operator acting on this device's truncated Hilbert
  space.
* **JAX traceability.** Every parameter passed to a subclass's
  ``__init__`` (frequency, anharmonicity, T1/T2, thermal population,
  …) may be a JAX tracer. Validation routines must never force
  concretization on a traced value; use
  :func:`quchip.utils.jax_utils.maybe_concrete_scalar` to peek at
  concrete scalars only.
* **Approximation transparency.** Each concrete model must document its
  approximation level and cite a reference. Examples: :class:`Resonator`
  states "non-interacting harmonic mode"; :class:`DuffingTransmon` states
  "Duffing expansion valid in the transmon regime ``E_J >> E_C``".

Channels offered to drives
--------------------------
Drives sit on top of a device and emit local Hamiltonians built from
standard bosonic / projection operators the device exposes:

* :meth:`lowering_operator` (``a``) and :meth:`raising_operator`
  (``a_dag``) — used by charge / coupling-type drives.
* :meth:`number_operator` (``n_hat = a_dag @ a``) — used by
  number-coupled (dispersive) drives.
* :attr:`sigma_x`, :attr:`sigma_y`, :attr:`sigma_z` — the qubit
  subspace projections onto ``|0>``, ``|1>`` (cached; invalidated
  automatically when ``levels`` changes).

State versioning
----------------
The engine caches assembled Hamiltonians keyed on :attr:`state_version`.
Once construction finishes, every public mutation (anything not prefixed
with ``_`` and not ``label``) increments ``_state_version`` so caches are
invalidated deterministically. This machinery — the seed, the
``__setattr__`` tracking hook, ``state_version``, and ``_finish_init`` —
is owned by the shared :class:`~quchip.utils.state_versioning.StateVersioned`
mixin; :class:`BaseDevice` only contributes its untracked-name set
(``label``) and the ``levels`` cache-invalidation hook
(:meth:`_on_attr_set`). Tracking is switched on automatically exactly once
after the outermost ``__init__`` returns, so subclasses no longer call
``_finish_init`` by hand.

Auto-labeling
-------------
Subclasses set ``_type_prefix`` (e.g. ``"duffing"``, ``"resonator"``)
and a shared counter in :mod:`quchip.utils.labeling` yields labels like
``"duffing_0"``, ``"resonator_0"``. Reset between tests via
:func:`quchip.utils.labeling.reset_label_counters`.

Serialization
-------------
:meth:`to_dict` writes a JSON-safe snapshot (type fully-qualified name,
``levels``, ``label``, concrete noise parameters). Deserialization
dispatch to the registered concrete subclass is owned by the shared
:class:`~quchip.utils.registry.Registrable` mixin — the registry is
populated automatically at subclass-definition time, with no manual
registration step.

Units (immutable)
-----------------
* Frequencies: GHz, ordinary (not angular).
* Times: ns.
* Temperature: mK.
* Energies: GHz (with ``hbar = 1``).

Example
-------
>>> from quchip.devices import DuffingTransmon, Resonator
>>> from quchip.chip import Chip
>>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
>>> r = Resonator(freq=7.0, levels=6, label="r")
>>> chip = Chip(devices=[q, r])
>>> float((q.freq * q.number_operator()).norm())  # doctest: +SKIP
"""

from __future__ import annotations

import copy
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypeVar

import jax.numpy as jnp

from quchip.backend import get_default_backend
from quchip.backend.protocol import Operator, State
from quchip.utils.jax_utils import maybe_concrete_scalar
from quchip.utils.labeling import auto_label
from quchip.utils.registry import Registrable
from quchip.utils.state_versioning import StateVersioned

if TYPE_CHECKING:
    from quchip.control.drive import BaseDrive
    from quchip.chip.chip import Chip


# The noise kwargs accepted by ``BaseDevice.__init__`` and forwarded between
# BaseDevice and concrete subclasses (constructor plumbing + serialization).
# Subclass `from_dict` implementations pull these from the dict so they don't
# duplicate the forwarding boilerplate. Channels beyond these built-ins use
# declared parameters plus a :class:`NoiseChannel` entry — see
# ``BaseDevice._noise_channels``.
_NOISE_FIELDS: tuple[str, ...] = (
    "T1",
    "T2",
    "thermal_population",
)


# -- Noise channels ------------------------------------------------------


@dataclass(frozen=True)
class NoiseChannel:
    """Declarative spec for one family of Lindblad collapse channels.

    A device class composes its dissipation from a tuple of these in
    ``_noise_channels``; :meth:`BaseDevice.collapse_operators` concatenates
    each channel's ``build(device)`` in declaration order. Adding a noise
    type to a device is therefore one declaration — a parameter field plus
    a channel entry — with no method override and no change outside the
    device's own class.

    Parameters
    ----------
    name : str
        Human-readable channel-family name (diagnostics only).
    params : tuple[str, ...]
        Device attribute names this channel consumes — declarative
        metadata; the build reads the attributes directly.
    build : callable
        ``build(device) -> list[Operator]``: the channel's collapse
        operators, empty when its parameters are unset.
    """

    name: str
    params: tuple[str, ...]
    build: Callable[[Any], list["Operator"]]


def _thermal_emission_channel(device: Any) -> list["Operator"]:
    """Relaxation / thermal-absorption channels from ``T1`` and ``thermal_population``.

    Emits ``sqrt(gamma*(n_bar+1))·a`` (relaxation / stimulated emission)
    and, for non-zero bath occupation, ``sqrt(gamma*n_bar)·a†`` (thermal
    absorption). Rate selection: when ``T1`` is set, ``gamma = 1/T1``; when
    only ``thermal_population`` is set (unitless bath occupation),
    ``gamma = 1`` so the user controls the absolute rate via the thermal
    amplitude — this avoids double-counting when both are specified.
    """
    n_bar = device.thermal_population
    if device.T1 is not None:
        rate: Any = 1.0 / device.T1
    elif n_bar is not None:
        rate = 1.0
    else:
        return []
    return BaseDevice._emission_pair(
        rate, n_bar, device.lowering_operator(), device.raising_operator()
    )


def _pure_dephasing_channel(device: Any) -> list["Operator"]:
    """Pure-dephasing channel ``sqrt(2*gamma_phi)·n̂`` from ``T2`` (and ``T1``).

    ``gamma_phi = 1/T2 - 1/(2*T1)`` when ``T1`` is set (``1/T2`` when
    ``T1`` is ``None``); omitted when non-positive. The ``2*``
    normalization makes the 0-1 coherence decay at exactly
    ``1/(2*T1) + gamma_phi = 1/T2``, so the input ``T2`` is the resulting
    coherence time when ``thermal_population == 0`` (see
    :meth:`BaseDevice._dephasing_op`). The construction constraint
    ``T2 <= 2*T1`` guarantees ``gamma_phi >= 0`` for concrete inputs.
    """
    gamma_phi = BaseDevice._dephasing_rate(device.T1, device.T2)
    if gamma_phi is None:
        return []
    return [BaseDevice._dephasing_op(gamma_phi, device.number_operator())]

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
    3. Implement :meth:`hamiltonian` returning an operator on the
       truncated Fock basis.

    Noise parameters (all optional; ``None`` means the channel is absent):

    * ``T1`` — relaxation time (ns); emission channel at rate ``1/T1``.
    * ``T2`` — total 0-1 coherence time (ns, requires ``T2 <= 2*T1``);
      adds pure dephasing at ``gamma_phi = 1/T2 - 1/(2*T1)``.
    * ``thermal_population`` — unitless bath occupation ``n̄``; adds thermal
      absorption and enhances emission.

    They are ordinary attributes: set them at construction or at any time
    after — the next ``simulate``/``solve`` rebuilds collapse operators from
    current values (no rebuild, no cache poking), and post-construction
    writes get the same validation as the constructor. Setting a parameter
    back to ``None`` removes its channel.

    Mutation tracking is enabled automatically once construction finishes
    (see :class:`~quchip.utils.state_versioning.StateVersioned`); subclasses
    do not call ``_finish_init`` themselves.

    Optional overrides:

    * :attr:`_noise_channels` — declare channels beyond ``T1``/``T2``
      (append a :class:`NoiseChannel`; :meth:`collapse_operators`
      composes the declared channels automatically).
    * :meth:`to_dict` / :meth:`from_dict` — for extra parameters.
    * :attr:`computational` — ``True`` if the device represents a
      computational qubit (default ``False``).

    See module docstring for the full contract.
    """

    _type_prefix: str = "device"

    #: Bare parameters this device exposes as differentiable / tunable
    #: scalars. ``fit_a_dress`` walks this tuple to discover what it is
    #: allowed to optimize on each device, decoupling the inverse-design
    #: surface from any specific device model. Three states, keyed on
    #: whether the value is explicitly declared:
    #:
    #: * **No explicit declaration anywhere in the**
    #:   :class:`~quchip.declarative.models.DeviceModel` **lineage** — the
    #:   default is *derived*: every declared
    #:   :func:`~quchip.declarative.parameters.parameter` field, in
    #:   declaration order (see ``DeviceModel.__init_subclass__``).
    #: * **Explicit tuple on the class or an ancestor** — exact curation,
    #:   validated at class-definition time; authoritative and inherited
    #:   until a subclass explicitly replaces it.
    #: * **Explicit empty tuple** — deliberately freezes the device (and its
    #:   subclasses, until one replaces it) out of inverse design.
    #:
    #: On a plain (non-``DeviceModel``) :class:`BaseDevice` subclass there is
    #: no derivation; the default stays empty unless the subclass declares
    #: its own tuple — e.g. :class:`~quchip.devices.fluxonium.Fluxonium` uses
    #: ``("E_C", "E_J", "E_L", "phi_ext")``.
    tunable_param_names: ClassVar[tuple[str, ...]] = ()

    # A device's ``label`` is identity metadata, not a physics parameter, so
    # rebinding it must not invalidate engine caches. Everything else public is
    # tracked. (``levels`` is tracked *and* triggers the cache hook below.)
    _untracked_names = frozenset({"label"})

    # Per-device readout/rotating-frame reference override; ``None`` inherits
    # :attr:`drive_freq`. Class-level default so the getter is safe even on the
    # JAX-pytree ``_unflatten`` path (which bypasses ``__init__``).
    _reference_freq_override: Any = None

    # Declared Lindblad channel families, composed in order by
    # :meth:`collapse_operators`. Subclasses extend (or replace) this tuple
    # to add or specialize dissipation — one declaration, no override.
    _noise_channels: ClassVar[tuple[NoiseChannel, ...]] = (
        NoiseChannel("thermal_emission", ("T1", "thermal_population"), _thermal_emission_channel),
        NoiseChannel("pure_dephasing", ("T2",), _pure_dephasing_channel),
    )

    def __init__(
        self,
        levels: int,
        label: str | None = None,
        *,
        T1: float | None = None,
        T2: float | None = None,
        thermal_population: float | None = None,
    ) -> None:
        if levels < 2:
            raise ValueError(f"levels must be >= 2, got {levels}")
        self.levels = levels

        _validate_noise_params(T1, T2, thermal_population)
        self.T1 = T1
        self.T2 = T2
        self.thermal_population = thermal_population

        self.label = label if label is not None else auto_label(type(self)._type_prefix)

        self._owner_chips: weakref.WeakSet["Chip"] = weakref.WeakSet()
        self._connected_drives: list[BaseDrive] = []
        self._reference_freq_override: Any = None

    def _on_attr_set(self, name: str) -> None:
        # Invalidate cached Pauli projections when the Fock truncation changes.
        # ``levels`` itself stays a tracked attribute (it bumps state_version
        # via the StateVersioned hook); this only drops the derived caches.
        if name == "levels":
            for cached in ("sigma_x", "sigma_y", "sigma_z", "sigma_plus", "sigma_minus"):
                self.__dict__.pop(cached, None)

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
        if getattr(self, "_tracking_enabled", False):
            if name == "_noise_channels":
                # collapse_operators() composes the CLASS tuple, and an
                # instance-level tuple could survive neither the JAX pytree
                # round-trip nor serialization — so instead of a silent
                # no-op, fail loudly and point at the idioms that work.
                raise TypeError(
                    "_noise_channels is class-level: extend it on the device "
                    "class (or declare a subclass with the extra channel); an "
                    "instance-level assignment would be silently ignored."
                )
            if not name.startswith("_"):
                self._validate_param_write(name, value)
        super().__setattr__(name, value)

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
        if name == "levels" and value < 2:
            raise ValueError(f"levels must be >= 2, got {value}")
        if name in _NOISE_FIELDS:
            candidate = {field: getattr(self, field, None) for field in _NOISE_FIELDS}
            candidate[name] = value
            _validate_noise_params(**candidate)

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

    def set_tunable_param(self, name: str, value: Any) -> None:
        """Update a bare parameter named in :meth:`tunable_params`.

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

    def copy(self) -> "BaseDevice":
        """Structural copy detached from drive wiring (used by sweep cloning)."""
        cloned = copy.copy(self)
        object.__setattr__(cloned, "_connected_drives", [])
        object.__setattr__(cloned, "_owner_chips", weakref.WeakSet())
        return cloned

    def _attach_chip(self, chip: "Chip") -> None:
        """Register *chip* as an owner for context-dependent device properties."""
        self._owner_chips.add(chip)

    def _detach_chip(self, chip: "Chip") -> None:
        """Remove *chip* from the owner registry (mirror of :meth:`_attach_chip`).

        Needed by transformations that build a scratch chip around a device on
        the way to the one the user receives: a ``Chip`` participates in a
        chip↔analysis reference cycle, so an abandoned scratch chip dies only
        when the *cyclic* GC runs — until then it would shadow the real owner
        in :meth:`_single_owner_chip`.
        """
        self._owner_chips.discard(chip)

    def _single_owner_chip(self) -> "Chip | None":
        owners = list(self._owner_chips)
        if not owners:
            return None
        if len(owners) > 1:
            labels = [owner.label for owner in owners]
            raise RuntimeError(
                f"Device {self.label!r} belongs to multiple live Chip instances "
                f"({labels}); use chip.freq(device) to choose the chip context explicitly."
            )
        return owners[0]

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

    # -- Dressed / drive frequency -----------------------------------------

    @property
    def dressed_freq(self) -> float | None:
        """Chip-derived dressed 0→1 transition frequency in GHz, or ``None`` without a chip context."""
        chip = self._single_owner_chip()
        if chip is None:
            return None
        return chip.freq(self)

    @property
    def drive_freq(self) -> float:
        """Operational 0→1 drive frequency in GHz.

        When the device belongs to exactly one chip this is the chip-derived
        dressed frequency. Standalone devices fall back to their bare ``freq``
        because no chip Hamiltonian exists to dress against. Returned values
        may be JAX tracers during traced / differentiated flows.
        """
        chip = self._single_owner_chip()
        if chip is not None:
            return chip.freq(self)
        try:
            return self.freq  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__name__!s} must expose a `freq` attribute "
                "(bare 0->1 transition frequency in GHz) for drive_freq to be defined."
            ) from exc

    @property
    def reference_freq(self) -> Any:
        """Readout / rotating-frame reference frequency in GHz — the device's LO.

        This is the frequency the default (``frame="rotating"``) frame
        co-rotates at *and* the reference the readout is reported in:
        ``result.expect`` is expressed in this frame in every integration
        frame, and in the default rotating frame ``result.states`` are too. So
        transverse observables (``<a>``, ``<sigma_x>``) come back as the slow
        demodulated envelope a lab readout produces — non-oscillatory when the
        device sits at its reference, and turning at ``omega - reference_freq``
        when detuned (idle Ramsey). Diagonal observables (populations, ``<n>``)
        are frame-invariant and unaffected either way.

        Defaults to :attr:`drive_freq` (the dressed 0->1 frequency), so an
        unset device co-rotates at its own transition — bit-identical to the
        prior behavior. Set it to model a control/LO reference that differs
        from the qubit frequency (a calibration detuning). It is a *frame /
        readout* reference only: it does **not** detune drives — the drive
        carrier is a separate choice, so a real LO error must also set the
        drive frequency. May be a JAX tracer in traced / differentiated /
        swept flows. Assign ``None`` to restore the default.
        """
        override = self._reference_freq_override
        return self.drive_freq if override is None else override

    @reference_freq.setter
    def reference_freq(self, value: Any | None) -> None:
        """Set the readout/rotating-frame reference (``None`` restores ``drive_freq``)."""
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
        # Persist an explicit reference_freq override (not the drive_freq
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
        ``drive_freq`` default). Returns ``self`` (typed as the concrete
        subclass) so ``from_dict`` can ``return cls(...)._restore_reference_freq(d)``.
        """
        if "reference_freq" in d:
            self.reference_freq = d["reference_freq"]
        return self

    # -- Hamiltonian (abstract) --------------------------------------------

    @abstractmethod
    def hamiltonian(self) -> Operator:
        """Return the device Hamiltonian on the truncated Hilbert space."""
        ...

    # -- Declared approximations --------------------------------------------

    def _truncation_note(self) -> str:
        """Return the Hilbert-truncation physics note.

        Default states the Fock-basis truncation. Subclasses whose
        truncated basis is not a Fock ladder (e.g. the diagonalized
        native-basis eigenstates of :class:`~quchip.devices.circuit.CircuitDevice`)
        override this single hook rather than :meth:`physics_notes` itself,
        so the rest of the base notes (T2 dephasing caveat, subclass
        additions) are not duplicated.
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
                "the resulting 0-1 coherence time only when thermal_population "
                "is 0."
            )
        return notes

    def dynamic_terms(self) -> list[tuple[Operator, Any]]:
        """Time-dependent device terms on the local Hilbert space (default: none).

        Subclasses modelling tunable / parametrically modulated devices
        (tunable transmons, flux-biased fluxonium, parametrically
        driven modes, …) override this to emit
        ``(local_operator, modulation)`` pairs. Each ``local_operator``
        acts on the *device's own* truncated Fock space and is in
        ordinary GHz; the engine embeds it, applies the single 2π
        factor, and attaches the modulation as a
        :class:`~quchip.engine.ir.DynamicTerm`. ``modulation`` must be
        a :class:`~quchip.engine.ir.ScalarModulation` wrapping a
        JAX-traceable signal program.
        """
        return []

    # -- Fock-space operator defaults --------------------------------------

    def lowering_operator(self) -> Operator:
        """Bosonic lowering operator ``a`` on the truncated Fock basis."""
        return get_default_backend().destroy(self.levels)

    def raising_operator(self) -> Operator:
        """Bosonic raising operator ``a†`` on the truncated Fock basis."""
        backend = get_default_backend()
        return backend.dag(backend.destroy(self.levels))

    def number_operator(self) -> Operator:
        """Number operator ``n̂ = a†a`` on the truncated Fock basis."""
        return get_default_backend().number(self.levels)

    def identity(self) -> Operator:
        """Identity operator on the truncated Fock basis."""
        return get_default_backend().identity(self.levels)

    # Operator-name vocabulary recognized by :meth:`local_operator`. Subclasses
    # that expose extra named operators extend this tuple (so the "unknown
    # operator" error lists them) and override :meth:`local_operator`.
    _LOCAL_OPERATOR_NAMES: tuple[str, ...] = ("X", "Y", "Z", "a", "a_dag", "n", "I")

    def local_operator(self, name: str) -> Operator:
        """Map an operator-name string to this device's own local operator.

        Recognized names: ``"X"`` / ``"Y"`` / ``"Z"`` (Pauli projections on
        the computational ``|0>, |1>`` subspace), ``"n"`` (number), ``"a"``
        (lowering), ``"a_dag"`` (raising), ``"I"`` (identity). The device owns
        this vocabulary, so a subclass exposing extra named
        operators overrides this method — extending
        :attr:`_LOCAL_OPERATOR_NAMES` and delegating to
        ``super().local_operator(name)`` for the base set — and the chip's
        observable surface (:meth:`Chip.observable`, :meth:`Chip.e_ops`) gains
        the operator without any engine or :class:`Chip` change.
        """
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
        raise ValueError(
            f"Unknown operator '{name}' for device '{self.label}'. "
            f"Available: {sorted(self._LOCAL_OPERATOR_NAMES)}"
        )

    def declarative_ops(self) -> dict[tuple[str, str], Operator]:
        """Map declarative ``(label, op-name)`` keys to backend operators.

        This is the single lookup the declarative layer consumes when
        compiling a :class:`~quchip.declarative.expr.PhysicsExpr`: a
        :class:`DeviceModel` uses it for its local Hamiltonian, and a
        coupling merges the maps of both endpoints. Keys mirror the
        operator handles exposed by :class:`~quchip.declarative.ops.LocalOps`
        (``a``, ``adag``, ``n``, ``I``, and the computational-subspace
        ``sigma_*`` family, so spin-like models are first-class citizens of
        the declarative surface).
        """
        return {
            (self.label, "a"): self.lowering_operator(),
            (self.label, "adag"): self.raising_operator(),
            (self.label, "n"): self.number_operator(),
            (self.label, "I"): self.identity(),
            (self.label, "sigma_x"): self.sigma_x,
            (self.label, "sigma_y"): self.sigma_y,
            (self.label, "sigma_z"): self.sigma_z,
            (self.label, "sigma_plus"): self.sigma_plus,
            (self.label, "sigma_minus"): self.sigma_minus,
        }

    def basis_state(self, n: int) -> State:
        """Fock basis state ``|n>`` on the truncated Hilbert space."""
        return get_default_backend().basis(self.levels, n)

    def coherent_state(self, alpha: complex) -> State:
        """Coherent state ``|alpha>`` on the truncated Fock basis."""
        return get_default_backend().coherent(self.levels, alpha)

    def plot_energy_levels(self, *, ax: Any = None, **kwargs: Any) -> Any:
        """Plot the device's bare energy-level ladder (delegates to :mod:`quchip.viz`)."""
        from quchip.viz.device import plot_energy_levels
        return plot_energy_levels(self, ax=ax, **kwargs)

    def plot_wavefunction(self, n: int, *, ax: Any = None, **kwargs: Any) -> Any:
        """Plot the ``n``-th eigenstate wavefunction (delegates to :mod:`quchip.viz`)."""
        from quchip.viz.device import plot_wavefunction
        return plot_wavefunction(self, n, ax=ax, **kwargs)

    # -- Pauli projections into |0>, |1> (cached, invalidated on `levels`) ---

    @cached_property
    def sigma_x(self) -> Operator:
        """``|0><1| + |1><0|`` on the computational ``|0>, |1>`` subspace."""
        return self.transition(0, 1)

    @cached_property
    def sigma_y(self) -> Operator:
        """``-i|0><1| + i|1><0|`` on the computational ``|0>, |1>`` subspace."""
        return -1j * self.projector(0, 1) + 1j * self.projector(1, 0)

    @cached_property
    def sigma_z(self) -> Operator:
        """``|0><0| - |1><1|`` on the computational ``|0>, |1>`` subspace."""
        return self.projector(0, 0) - self.projector(1, 1)

    @cached_property
    def sigma_plus(self) -> Operator:
        """Raising operator on the computational ``|0>, |1>`` subspace: ``|1><0|``."""
        return self.projector(1, 0)

    @cached_property
    def sigma_minus(self) -> Operator:
        """Lowering operator on the computational ``|0>, |1>`` subspace: ``|0><1|``."""
        return self.projector(0, 1)

    def projector(self, i: int, j: int) -> Operator:
        """``|i><j|`` on the Fock basis.

        Use ``projector(i, i)`` for the population projector
        ``|i><i|`` and ``projector(i, j)`` for ``|i><j|``. No subspace
        approximation — the operator acts on the full truncated Hilbert
        space.
        """
        backend = get_default_backend()
        ket_i = backend.basis(self.levels, i)
        ket_j = backend.basis(self.levels, j)
        return backend.matmul(ket_i, backend.dag(ket_j))

    def transition(self, i: int, j: int) -> Operator:
        """Transition operator ``|i><j| + |j><i|`` between Fock levels ``i`` and ``j``.

        Acts like ``sigma_x`` on the two-level subspace ``{|i>, |j>}``.
        Useful for qudit work and erasure-protected subspaces where the
        computational pair is not ``{|0>, |1>}``.
        """
        return self.projector(i, j) + self.projector(j, i)

    # -- Classification ----------------------------------------------------

    @property
    def computational(self) -> bool:
        """Whether this device is a computational qubit. Override in subclasses."""
        return False

    # -- Collapse operators ------------------------------------------------

    def collapse_operators(self) -> list[Operator]:
        """Lindblad collapse operators composed from the class's declared noise channels.

        Concatenates each :class:`NoiseChannel` in :attr:`_noise_channels`
        in declaration order. The built-in channels cover ``T1`` / ``T2`` /
        ``thermal_population`` (see :func:`_thermal_emission_channel` and
        :func:`_pure_dephasing_channel` for the physics and normalization
        conventions); subclasses add channels by extending the tuple.

        References
        ----------
        Breuer & Petruccione, *Theory of Open Quantum Systems* (Oxford,
        2002), Ch. 3. For circuit-QED conventions see Krantz et al.,
        *Applied Physics Reviews* **6**, 021318 (2019), §V.
        """
        return [op for channel in type(self)._noise_channels for op in channel.build(self)]

    @classmethod
    def noise_parameter_names(cls) -> tuple[str, ...]:
        """Names of this class's dissipation parameters, in declaration order.

        The deduplicated union of each declared :class:`NoiseChannel`'s
        ``params`` — exactly the attributes
        :meth:`~quchip.chip.chip.Chip.set_noise` may touch. Hamiltonian
        parameters are never included.
        """
        seen: dict[str, None] = {}
        for channel in cls._noise_channels:
            for name in channel.params:
                seen.setdefault(name)
        return tuple(seen)

    def intrinsic_decay_rate(self) -> Any | None:
        """Total lowering-channel (downward) Lindblad rate, in 1/ns, or ``None`` with no decay channel.

        Reports the actual sum of squared amplitudes of the lowering-operator
        collapse channel(s) :meth:`collapse_operators` builds from
        :func:`_thermal_emission_channel`, matching that construction exactly
        rather than approximating it:

        * ``T1`` set (``thermal_population`` set or not): ``(n̄+1)/T1`` —
          the ``sqrt(gamma*(n̄+1))·a`` channel's rate, ``gamma = 1/T1``;
          ``n̄`` defaults to ``0`` when ``thermal_population`` is unset, so
          this reduces to plain ``1/T1``.
        * ``T1`` unset, ``thermal_population`` set: ``n̄+1`` — the same
          channel with ``gamma = 1`` (:func:`_thermal_emission_channel`'s
          unitless-bath-occupation branch).
        * Neither set: ``None`` — no lowering channel.

        Subclasses whose :meth:`collapse_operators` combine several
        lowering-operator channels (e.g. :class:`~quchip.devices.resonator.Resonator`'s
        Q-derived photon loss alongside ``T1``) override this to report the
        summed rate, so a caller reading a single scalar decay rate (e.g.
        :mod:`quchip.chip.transformations.eliminate_device`'s Purcell fold)
        does not have to special-case per-device channel structure.

        This is the *downward* rate only — the ``sqrt(gamma*n̄)·a†`` upward
        (thermal-absorption) channel is not represented; a caller that needs
        to know whether that channel is present reads ``thermal_population``
        directly. Whether a channel exists, and which formula applies, is a
        *static* decision (is ``T1``/``thermal_population`` set?), never a
        traced-zero comparison on the resulting rate, which would concretize
        a traced value and break differentiability.
        """
        n_bar = self.thermal_population
        if self.T1 is not None:
            n_bar_eff = 0.0 if n_bar is None else n_bar
            return (n_bar_eff + 1.0) / self.T1
        if n_bar is not None:
            return n_bar + 1.0
        return None

    # -- Shared Lindblad rate algebra (reused by CircuitDevice) -------------

    @staticmethod
    def _emission_pair(
        rate: Any,
        n_bar: Any | None,
        lower_op: Operator,
        raise_op: Operator,
    ) -> list[Operator]:
        """Emission / absorption collapse-operator pair for one transition.

        Returns ``[sqrt(rate * (n_bar + 1)) * lower_op]`` (relaxation /
        stimulated emission) and, when the bath occupation is non-zero,
        additionally ``sqrt(rate * n_bar) * raise_op`` (thermal absorption).
        ``n_bar is None`` is treated as zero occupation, yielding the down
        channel only. The positivity gate reads a *concrete* scalar only, so
        a traced ``n_bar`` keeps both channels. The math is
        pure :mod:`jax.numpy`, so the result type follows the operators
        passed in (Fock operators for :class:`BaseDevice`, eigenbasis
        projectors for :class:`~quchip.devices.circuit.CircuitDevice`).
        """
        n_bar_eff = 0.0 if n_bar is None else n_bar
        ops: list[Operator] = [jnp.sqrt(rate * (n_bar_eff + 1.0)) * lower_op]
        n_bar_value = maybe_concrete_scalar(n_bar_eff)
        if n_bar_value is None or n_bar_value > 0:
            ops.append(jnp.sqrt(rate * n_bar_eff) * raise_op)
        return ops

    @staticmethod
    def _dephasing_rate(T1: Any | None, T2: Any | None) -> Any | None:
        """Clamped pure-dephasing rate ``gamma_phi``, or ``None`` when absent.

        ``gamma_phi = 1/T2 - 1/(2*T1)`` when ``T1`` is set, or ``1/T2`` when
        ``T1`` is ``None`` (no T1 subtraction — the
        :class:`~quchip.devices.circuit.CircuitDevice` T1-absent branch relies
        on this). Returns ``None`` when there is no dephasing channel: either
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

    @staticmethod
    def _dephasing_op(gamma_phi: Any, diag_op: Operator) -> Operator:
        """Pure-dephasing collapse operator ``sqrt(2*gamma_phi) * diag_op``.

        ``diag_op`` is a Hermitian diagonal generator: the number operator
        ``n_hat`` on a Fock :class:`BaseDevice`, or the level-index operator
        on a :class:`~quchip.devices.circuit.CircuitDevice`. For a Lindblad
        dissipator ``D[c]`` with ``c = sqrt(k) * diag_op`` the coherence
        ``rho_mn`` decays at ``(k/2) * (m - n)**2``; the computational 0-1
        coherence (``|m - n| = 1``) therefore decays at ``k/2``. Choosing
        ``k = 2*gamma_phi`` makes that rate exactly ``gamma_phi``, so the
        total transverse rate is ``1/(2*T1) + gamma_phi = 1/T2`` and the
        input ``T2`` *is* the resulting coherence time (when
        ``thermal_population == 0``). Higher coherences scale as
        ``(m - n)**2`` (e.g. 0-2 decays at ``4*gamma_phi``), the standard
        number-operator dephasing law.

        The factor lives here, in one shared helper, so the Fock and
        circuit call sites cannot drift apart. The math is pure
        :mod:`jax.numpy`, so the result type follows ``diag_op`` and a
        traced ``gamma_phi`` stays differentiable.
        """
        return jnp.sqrt(2.0 * gamma_phi) * diag_op

    # -- Drive wiring ------------------------------------------------------

    def connect(self, drive: "BaseDrive") -> None:
        """Register a drive as connected: idempotent on identity, replace-on-relabel.

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

    def _repr_dressed_freq(self) -> str:
        try:
            return repr(self.dressed_freq)
        except RuntimeError:
            return "<multiple chip contexts>"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(label={self.label!r}, "
            f"freq={getattr(self, 'freq', None)!r}, levels={self.levels}, "
            f"dressed_freq={self._repr_dressed_freq()})"
        )


def _validate_noise_params(
    T1: float | None,
    T2: float | None,
    thermal_population: float | None,
) -> None:
    """Validate T1 / T2 / thermal_population on concrete scalars only (JAX-safe)."""
    T1_value = maybe_concrete_scalar(T1)
    T2_value = maybe_concrete_scalar(T2)
    thermal_value = maybe_concrete_scalar(thermal_population)

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
        raise ValueError(f"thermal_population must be ≥ 0, got {thermal_population}")
