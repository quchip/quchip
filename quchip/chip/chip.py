"""Composite quantum system — :class:`Chip` bundles devices, couplings, and control.

A chip owns the devices (which own their local Hamiltonians), the
couplings between them (which own their two-body interaction
Hamiltonians), and, optionally, the :class:`ControlEquipment` wiring
classical control lines. The engine consumes a chip to produce a
solver-ready problem.

All public parameters — device frequencies, coupling strengths, drive
amplitudes, crosstalk coefficients — remain JAX-traceable and sweepable so a
single loss function can span any of them.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, overload

import numpy as np

from quchip.backend import _backend_context
from quchip.backend.protocol import Backend, Operator, State
from quchip.chip.analysis import ChipAnalysis, DressedResult
from quchip.chip.baths import Bath
from quchip.chip.coupling_base import BaseCoupling
from quchip.chip.rwa import apply_rwa_mask
from quchip.chip.states import _DEFAULT_LEVEL_SYMBOLS
from quchip.control.drive import BaseDrive
from quchip.control.equipment import ControlEquipment
from quchip.control.signal import Crosstalk, SignalTransform
from quchip.declarative.parameters import validate_sign
from quchip.devices.base import BaseDevice, _validate_noise_params
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar
from quchip.utils.labeling import LabelKeyedDict, resolve_label

if TYPE_CHECKING:
    from quchip.chip.partition import PartitionResult
    from quchip.engine.ir import FrameSpec, SolveProblem
    from quchip.results.results import SimulationBatchResult, SimulationResult


def _format_float(value: float | None) -> str:
    """Compact numeric formatter for status dashboards."""
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def _is_scalar_like(value: Any) -> bool:
    """Return True for 0-dim arrays and plain Python numbers (used in frame spec)."""
    return getattr(value, "shape", None) == () or isinstance(value, (int, float))


def _same_concrete_value(a: Any, b: Any) -> bool:
    """True iff both are ``None`` or both concretize to equal scalars.

    Non-concrete (traced) values always compare as different, so traced
    writes are always applied and never hit a Python ``==`` on a tracer.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    concrete_a = maybe_concrete_scalar(a)
    concrete_b = maybe_concrete_scalar(b)
    return concrete_a is not None and concrete_b is not None and concrete_a == concrete_b


class Chip:
    """Composite quantum system: devices + couplings + (optional) control.

    The chip is the primary user-facing object. It exposes:

    - structural access (:attr:`devices`, :attr:`couplings`, :attr:`dims`),
    - Hamiltonian construction (:meth:`hamiltonian`),
    - dressed-state analysis (:meth:`dress`, :meth:`energy`, :meth:`freq`, …)
      delegated to :class:`~quchip.chip.analysis.ChipAnalysis`,
    - state factories (:meth:`state`, :meth:`bare_state`),
    - observable construction (:meth:`observable`, :meth:`e_ops`),
    - solver surface (:meth:`solve`, :meth:`solve_many`),
    - control wiring (:meth:`wire`, :meth:`connect`),
    - serialization / cloning (:meth:`to_dict`, :meth:`from_dict`,
      :meth:`clone`, :meth:`updated`).

    Methods that accept a device reference take either a device object
    **or** its label string; object references are preferred in examples.

    Parameters
    ----------
    devices : list[BaseDevice]
        Ordered device list. Tensor-product position equals list index;
        labels must be unique.
    couplings : list[BaseCoupling], optional
        Two-body couplings. Each coupling's referenced devices must be
        in ``devices``.
    control_equipment : ControlEquipment, optional
        Aggregates drive lines and signal-chain transforms (crosstalk,
        delays, gains). Can also be attached later via :meth:`wire` or
        :meth:`connect`.
    label : str, optional
        Human-readable chip label.
    frame : FrameSpec
        Initial frame specification:

        - ``"lab"`` — all reference frequencies 0 GHz (default).
        - ``"rotating"`` — per-device rotating frame at dressed drive
          frequencies.
        - scalar-like — one shared reference frequency for all devices.
        - ``dict`` — per-device references keyed by label or device.
    rwa : bool
        Default RWA policy for couplings and drives. Per-component
        ``rwa`` overrides inherit this (``None`` means inherit).
    backend : str or Backend, optional
        Chip-specific backend. ``None`` uses the process default.

    Examples
    --------
    >>> from quchip import DuffingTransmon, Resonator, Capacitive, Chip
    >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    >>> r = Resonator(freq=7.0, levels=5, label="r")
    >>> chip = Chip([q, r], couplings=[Capacitive(q, r, g=0.05)])
    >>> chip.energy({q: 1})  # dressed |1,0⟩ eigenvalue, GHz  # doctest: +SKIP
    """

    def __init__(
        self,
        devices: list[BaseDevice],
        couplings: list[BaseCoupling] | None = None,
        control_equipment: ControlEquipment | None = None,
        label: str | None = None,
        frame: FrameSpec = "lab",
        rwa: bool = True,
        backend: str | Backend | None = None,
        baths: list[Bath] | None = None,
    ) -> None:
        duplicates = [lbl for lbl, count in Counter(d.label for d in devices).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate device labels: {duplicates}. All device labels must be unique.")

        self.label = label
        self._devices = tuple(devices)
        self._device_map: dict[str, BaseDevice] = {d.label: d for d in devices}
        self._label_to_index: dict[str, int] = {d.label: i for i, d in enumerate(devices)}
        self._couplings = tuple(couplings) if couplings else ()
        coupling_labels = [c.label for c in self._couplings]
        dup = [lbl for lbl, n in Counter(coupling_labels).items() if n > 1]
        if dup:
            raise ValueError(f"Duplicate coupling labels: {dup}. Labels must be unique.")
        collision = [lbl for lbl in coupling_labels if lbl in self._device_map]
        if collision:
            raise ValueError(
                f"Coupling labels collide with device labels: {collision}. "
                "Device and coupling labels share one namespace so control targets resolve unambiguously."
            )
        self._coupling_map: dict[str, BaseCoupling] = {c.label: c for c in self._couplings}
        bath_duplicates = [lbl for lbl, n in Counter(b.label for b in baths or ()).items() if n > 1]
        if bath_duplicates:
            raise ValueError(f"Duplicate bath labels: {bath_duplicates}. Each bath must have a unique label.")
        for bath in baths or ():
            self._validate_bath(bath)
        self._baths = tuple(baths) if baths else ()
        self._dims = tuple(d.levels for d in devices)
        self._frame_spec: FrameSpec = "lab"
        self._rwa = bool(rwa)
        if control_equipment is not None:
            drive_duplicates = [
                lbl for lbl, n in Counter(d.label for d in control_equipment.lines).items() if n > 1
            ]
            if drive_duplicates:
                raise ValueError(
                    f"Duplicate drive labels in equipment: {drive_duplicates}. "
                    "Each drive must have a unique label."
                )
        self._control_equipment = control_equipment
        self._analysis = ChipAnalysis(self)

        for device in self._devices:
            device._attach_chip(self)

        for coupling in self._couplings:
            coupling._resolve_devices(self._device_map)

        if backend is not None:
            from quchip.backend import _coerce_backend

            self._backend: Backend | None = _coerce_backend(backend)
        else:
            self._backend = None

        # String-state shorthand — populated by :meth:`set_state_order`.
        self._state_order: tuple[str, ...] | None = None
        self._level_symbols: dict[str, int] = dict(_DEFAULT_LEVEL_SYMBOLS)

        # Memo of the lab-frame Hamiltonian. It is built twice per engine pass
        # (once lab-frame for dressing, once for the engine's static-H0) and is
        # bit-identical both times, so the second build is pure waste. Keyed on
        # the backend kind + per-device/per-coupling ``state_version`` so any
        # parameter or level change invalidates it. Never cached under a JAX
        # trace (the ``contains_tracer`` guard), so differentiability is intact.
        self._hamiltonian_cache: tuple[Any, Operator] | None = None

        if frame != "lab":
            self.set_frame(frame)

    # ------------------------------------------------------------------
    # Hamiltonian
    # ------------------------------------------------------------------

    def hamiltonian(self) -> Operator:
        """Lab-frame static Hamiltonian.

        Embeds every device Hamiltonian into the full tensor space and
        adds each coupling's embedded interaction. Does **not** apply
        the rotating-frame transform or any drive envelopes.

        Under resolved RWA (:meth:`resolve_rwa`) each coupling
        contributes only the bands its
        :meth:`~quchip.chip.coupling_base.BaseCoupling.rwa_keeps_band`
        retains — the :func:`~quchip.chip.rwa.apply_rwa_mask` of its full
        interaction. The same bands drop from stage 2's decomposition, so
        the two views agree.

        This is the **lab-frame** Hamiltonian. At solve time, the engine
        applies a rotating-frame transformation and subtracts
        ``2π Σ_i ω_ref,i n̂_i`` for each device, where ``ω_ref,i`` is the
        frame reference resolved by
        :func:`quchip.engine.stage1_frames.resolve_frame`. The 2π boundary
        crossing and the subtraction both live in
        :func:`quchip.engine.stage2_assembly._build_static_h0`. Use
        :meth:`frame_info` to inspect which reference frequency each
        device will use.

        Returns
        -------
        Operator
            Backend-native operator on ``⨂_d H_d``.
        """
        backend = self.backend
        signature = (
            type(backend).__qualname__,
            self._rwa,
            tuple((d.label, d.state_version) for d in self._devices),
            tuple((c.label, c.state_version, self.resolve_rwa(c)) for c in self._couplings),
        )
        cache = self._hamiltonian_cache
        if cache is not None and cache[0] == signature and not contains_tracer(cache[1]):
            return cache[1]

        with _backend_context(backend):
            H: Operator | None = None
            for i, dev in enumerate(self._devices):
                h_emb = backend.embed(dev.hamiltonian(), i, self._dims)
                H = h_emb if H is None else H + h_emb

            for coupling in self._couplings:
                idx_a = self._label_to_index[coupling.device_a_label]
                idx_b = self._label_to_index[coupling.device_b_label]
                h_int = coupling.interaction_hamiltonian()
                if self.resolve_rwa(coupling):
                    rwa_dims = (self._devices[idx_a].levels, self._devices[idx_b].levels)
                    rwa_labels = (coupling.device_a_label, coupling.device_b_label)
                    masked = apply_rwa_mask(
                        h_int,
                        dims=rwa_dims,
                        labels=rwa_labels,
                        keeps_band=coupling.rwa_keeps_band,
                        backend=backend,
                    )
                    if masked is None:
                        # No band survived rwa_keeps_band. An interaction that was
                        # already exactly zero has no band to reject in the first
                        # place; rerun the same decomposition with an always-true
                        # predicate to tell that case (skip silently) apart from a
                        # nonzero interaction every one of whose populated bands the
                        # predicate rejected (a policy surprise worth a warning).
                        has_any_band = apply_rwa_mask(
                            h_int,
                            dims=rwa_dims,
                            labels=rwa_labels,
                            keeps_band=lambda *_: True,
                            backend=backend,
                        ) is not None
                        if has_any_band:
                            warnings.warn(
                                f"Coupling '{coupling.label}' vanishes entirely under the resolved RWA: "
                                f"none of its bands pass rwa_keeps_band. Set rwa=False on the coupling "
                                f"(or override rwa_keeps_band) to retain it.",
                                UserWarning,
                                stacklevel=2,
                            )
                        continue
                    h_int = masked
                H = H + backend.embed_two_body(h_int, idx_a, idx_b, self._dims)

        # Only memoize concrete operators: under jax.jit/grad/vmap the device
        # params are tracers, so H carries tracers and must not be cached.
        if not contains_tracer(H):
            self._hamiltonian_cache = (signature, H)
        return H

    # ------------------------------------------------------------------
    # Component physics enumeration (consumed by the engine)
    # ------------------------------------------------------------------
    #
    # The chip is the single place that knows which component families
    # exist (devices, couplings, baths, drive lines). These methods hand
    # the engine flat lists of local contributions tagged with their
    # *support* — the device indices each operator acts on — so the engine
    # embeds by arity and never enumerates families itself. Adding a new
    # physics-bearing component family is therefore a chip-side change; it
    # never requires modifying the engine.

    def dynamic_contributions(self) -> list[tuple[Operator, Any, tuple[int, ...], str, str]]:
        """Chip-owned time-dependent Hamiltonian contributions with their support.

        Returns ``(local_op, time_dependence, support, origin, tag)``
        tuples: one support index for a device-local operator, two for a
        coupling's two-body operator. Operators are lab-frame, ordinary
        GHz — the engine applies the 2π boundary. Drive terms are *not*
        included; they are schedule-owned and enter through stage 2's
        drive compilation.
        """
        backend = self.backend
        out: list[tuple[Operator, Any, tuple[int, ...], str, str]] = []
        with _backend_context(backend):
            for coupling in self._couplings:
                term_pairs = coupling.dynamic_interaction_terms(self)
                if not term_pairs:
                    continue
                idx_a = self._label_to_index[coupling.device_a_label]
                idx_b = self._label_to_index[coupling.device_b_label]
                out.extend(
                    (op, td, (idx_a, idx_b), "coupling", "coupling_dynamic")
                    for op, td in term_pairs
                )
            for i, dev in enumerate(self._devices):
                out.extend(
                    (op, td, (i,), "device", "device_dynamic")
                    for op, td in dev.dynamic_terms()
                )
        return out

    def collapse_contributions(self) -> list[tuple[Operator, tuple[int, ...]]]:
        """Every Lindblad collapse operator on the chip, with its support.

        Returns ``(operator, support)`` pairs: one device index for a
        device- or drive-line-local operator, two for a coupling's
        two-body operator, and an empty tuple for an operator already
        embedded in the full space (baths). Rates are in 1/ns,
        Lindblad-ready — each component owns its rate physics, including
        any intrinsic 2π (e.g. a resonator's κ = 2π·f/Q).

        Every returned operator is a component's LOCAL (bare-basis)
        operator; the engine combines them with the dressed interacting
        Hamiltonian at solve time. This is the standard local-Lindblad
        approximation (Breuer & Petruccione, *The Theory of Open Quantum
        Systems*, Oxford, 2002, Ch. 3) rather than a dressed-basis
        (polaron-frame) master equation, and applies chip-wide regardless of
        which components carry noise — see the ``"chip"`` entry of
        :meth:`physics_notes`.
        """
        backend = self.backend
        out: list[tuple[Operator, tuple[int, ...]]] = []
        with _backend_context(backend):
            for i, dev in enumerate(self._devices):
                out.extend((op, (i,)) for op in dev.collapse_operators())
            if self.control_equipment is not None:
                for line in self.control_equipment.lines:
                    if line.device_label is None:
                        continue
                    idx = self._label_to_index[line.device_label]
                    out.extend((op, (idx,)) for op in line.collapse_operators(self._devices[idx]))
            for coupling in self._couplings:
                idx_a = self._label_to_index[coupling.device_a_label]
                idx_b = self._label_to_index[coupling.device_b_label]
                out.extend((op, (idx_a, idx_b)) for op in coupling.collapse_operators(self))
            for bath in self.baths:
                out.extend((op, ()) for op in bath.collapse_operators(self))
        return out

    # ------------------------------------------------------------------
    # Structural properties
    # ------------------------------------------------------------------

    @property
    def device_map(self) -> dict[str, BaseDevice]:
        """Label → device mapping (the chip's own dict; do not mutate)."""
        return self._device_map

    @property
    def coupling_map(self) -> dict[str, "BaseCoupling"]:
        """Couplings by label, insertion-ordered (do not mutate)."""
        return self._coupling_map

    def coupling(self, coupling: "str | BaseCoupling") -> "BaseCoupling":
        """Return a coupling by label or object."""
        label = resolve_label(coupling)
        if label not in self._coupling_map:
            raise KeyError(
                f"No coupling labeled '{label}' on this chip. Available: {list(self._coupling_map.keys())}"
            )
        return self._coupling_map[label]

    @property
    def backend(self) -> Backend:
        """Active backend: per-call override > chip-specific > process default.

        The per-call override is the ``backend=`` argument of
        :meth:`~quchip.control.sequence.QuantumSequence.simulate` /
        ``simulate_batch`` (scoped through
        :func:`quchip.backend._backend_context`); it outranks even a
        chip-constructed backend so one chip can serve, e.g., QuTiP sweeps
        and dynamiqs gradient solves without global state flips.
        """
        from quchip.backend import _backend_override, get_default_backend

        override = _backend_override.get()
        if override is not None:
            return override
        if self._backend is not None:
            return self._backend
        return get_default_backend()

    @property
    def frame(self) -> FrameSpec:
        """Current frame specification (see :meth:`set_frame`)."""
        return self._frame_spec

    @property
    def rwa(self) -> bool:
        """Default rotating-wave approximation policy for couplings and drives."""
        return self._rwa

    @property
    def control_equipment(self) -> ControlEquipment | None:
        """Attached control equipment, if any."""
        return self._control_equipment

    @property
    def devices(self) -> tuple[BaseDevice, ...]:
        """Ordered tuple of devices; index matches tensor-product position."""
        return self._devices

    @property
    def couplings(self) -> tuple[BaseCoupling, ...]:
        """Tuple of couplings in insertion order."""
        return self._couplings

    @property
    def dims(self) -> tuple[int, ...]:
        """Per-device Hilbert-space dimensions (same order as :attr:`devices`)."""
        return self._dims

    @property
    def total_dim(self) -> int:
        """Total Hilbert-space dimension (product of per-device :attr:`dims`)."""
        return int(np.prod(self._dims, dtype=int))

    @property
    def baths(self) -> tuple[Bath, ...]:
        """Chip-level baths (shared/collective dissipation), insertion order."""
        return self._baths

    def add_bath(self, bath: Bath) -> Bath:
        """Attach a bath to this chip and return it (for fluent use).

        Baths may be added at any time after construction — the next
        simulate/solve collects the bath's collapse operators automatically,
        no rebuild needed. The bath is validated immediately: a non-``Bath``
        argument, a target label not on this chip, or a label colliding
        with an already-attached bath fails here rather than cryptically at
        solve time (or, for a label collision, silently overwriting an
        entry on the :meth:`physics_notes` audit surface).
        """
        self._validate_bath(bath)
        if bath.label in {b.label for b in self._baths}:
            raise ValueError(
                f"Duplicate bath label: '{bath.label}' is already attached to this chip. "
                "Each bath must have a unique label."
            )
        self._baths = (*self._baths, bath)
        return bath

    def _validate_bath(self, bath: Bath) -> None:
        """Reject non-Bath objects and unknown target labels at attach time.

        Devices are fixed at construction, so an unknown target can never
        become valid later — failing here beats an opaque error at solve
        time.
        """
        if not isinstance(bath, Bath):
            raise TypeError(f"Expected a Bath, got {type(bath).__name__}: {bath!r}")
        unknown = [lbl for lbl in bath.resolve_targets(self) if lbl not in self._device_map]
        if unknown:
            raise ValueError(
                f"Bath {bath.label!r} targets unknown device(s) {unknown}; "
                f"this chip has {sorted(self._device_map)}"
            )

    def set_noise(
        self,
        config: Mapping[str | BaseDevice, Mapping[str, Any]] | None = None,
        *,
        baths: list[Bath] | None = None,
    ) -> None:
        """Replace this chip's entire noise description in one call.

        The call is the chip's complete noise state: for every device, each
        noise parameter (see
        :meth:`~quchip.devices.base.BaseDevice.noise_parameter_names`) is set
        from *config* when given and reset to ``None`` otherwise, and the
        chip's baths become exactly *baths* (omitted → no baths).
        ``chip.set_noise()`` therefore clears all noise. Re-running the same
        call is a silent no-op, so notebook cells converge instead of
        stacking duplicate baths.

        Only dissipation knobs are reachable — a key that is not a noise
        parameter of that device (e.g. ``freq``) raises, so an optimized
        Hamiltonian cannot be perturbed by this call. The full target state
        is validated *before* anything is written (the same checks the
        constructors run); on error the chip is left exactly as it was.
        Applied changes are printed one per line; a no-op prints nothing.

        Custom dissipation is supported by declaring it on the device class
        beforehand — a ``parameter(...)`` rate field plus a
        :class:`~quchip.devices.base.NoiseChannel` entry (see the extension
        guide). The declared rate is then an ordinary noise parameter here:
        sweepable, differentiable, serializable. Runtime closure-style
        channels are deliberately not supported — their rates could be
        neither swept, nor differentiated, nor serialized.

        Parameters
        ----------
        config : mapping, optional
            ``{device_or_label: {noise_param: value}}``. Devices may appear
            as objects or labels, once each.
        baths : list[Bath], optional
            The chip's complete new bath list (validated like
            :meth:`add_bath`).
        """
        # Resolve config keys (objects or labels; each device at most once).
        resolved: dict[str, Mapping[str, Any]] = {}
        for key, params in (config or {}).items():
            label = resolve_label(key)
            if label not in self._device_map:
                raise ValueError(f"Unknown device {label!r}; this chip has {sorted(self._device_map)}")
            if label in resolved:
                raise ValueError(f"Device {label!r} appears more than once in the noise config")
            resolved[label] = params

        # Validate the complete target state before writing anything.
        new_baths = list(baths) if baths else []
        bath_duplicates = [lbl for lbl, n in Counter(b.label for b in new_baths).items() if n > 1]
        if bath_duplicates:
            raise ValueError(f"Duplicate bath labels: {bath_duplicates}. Each bath must have a unique label.")
        for bath in new_baths:
            self._validate_bath(bath)

        targets: dict[str, dict[str, Any]] = {}
        for label, device in self._device_map.items():
            names = type(device).noise_parameter_names()
            given = resolved.get(label, {})
            unknown = sorted(set(given) - set(names))
            if unknown:
                raise ValueError(
                    f"{unknown} are not noise parameters of {label!r} "
                    f"({type(device).__name__}); valid: {sorted(names)}"
                )
            targets[label] = {name: given.get(name) for name in names}

        for label, target in targets.items():
            device = self._device_map[label]
            _validate_noise_params(target.get("T1"), target.get("T2"), target.get("thermal_population"))
            fields = getattr(type(device), "__quchip_param_fields__", {})
            for name, value in target.items():
                spec = fields.get(name)
                if spec is not None:
                    validate_sign(name, spec, value)

        # Apply: ordinary tracked writes; only real changes touch a device,
        # so an identical call is a true no-op (no state_version bumps).
        changes: list[str] = []
        for label, target in targets.items():
            device = self._device_map[label]
            for name in sorted(target, key=lambda n: n == "T2"):  # T2 last: its validator reads the final T1
                old = getattr(device, name, None)
                new = target[name]
                if _same_concrete_value(old, new):
                    continue
                setattr(device, name, new)
                changes.append(f"{label}: {name} {old!r} → {new!r}")

        old_ids = {id(bath) for bath in self._baths}
        new_ids = {id(bath) for bath in new_baths}
        for bath in self._baths:
            if id(bath) not in new_ids:
                changes.append(f"baths: - {bath.label} ({bath.recipe})")
        for bath in new_baths:
            if id(bath) not in old_ids:
                extra = f", {bath.temperature} mK" if bath.temperature is not None else ""
                changes.append(f"baths: + {bath.label} ({bath.recipe}{extra})")
        self._baths = tuple(new_baths)

        for line in changes:
            print(line)

    @property
    def crosstalks(self) -> list[Crosstalk]:
        """Convenience view — :class:`Crosstalk` entries from the signal chain."""
        if self._control_equipment is None:
            return []
        return self._control_equipment.crosstalks

    def device_index(self, label: str | BaseDevice) -> int:
        """Tensor-product index of a device. Accepts a label string or object."""
        return self._resolve_device_index(label)[0]

    def _resolve_device_index(self, device: str | BaseDevice) -> tuple[int, BaseDevice]:
        """Return ``(index, device)`` for a label string or a device object."""
        label = resolve_label(device)
        idx = self._label_to_index.get(label)
        if idx is None:
            raise ValueError(f"Device '{label}' not found. Available: {list(self._device_map.keys())}")
        return idx, self._devices[idx]

    def __getitem__(self, label: str | BaseDevice) -> BaseDevice:
        try:
            return self._resolve_device_index(label)[1]
        except ValueError as err:
            raise KeyError(str(err)) from None

    # ------------------------------------------------------------------
    # Viz — lazy delegates
    # ------------------------------------------------------------------

    def plot_graph(
        self, path: str = "chip_topology.html", *, full: bool = True, exclude: set[str] | None = None, **kwargs: Any
    ) -> str:
        """Render chip topology — delegates to :mod:`quchip.viz.chip`."""
        from quchip.viz.chip import plot_graph

        return plot_graph(self, path, full=full, exclude=exclude, **kwargs)

    def plot_energy_levels(self, *, ax: Any = None, **kwargs: Any) -> Any:
        """Render dressed spectrum — delegates to :mod:`quchip.viz.chip`."""
        from quchip.viz.chip import plot_energy_levels

        return plot_energy_levels(self, ax=ax, **kwargs)

    # ------------------------------------------------------------------
    # Frame management
    # ------------------------------------------------------------------

    def set_frame(self, frame: FrameSpec) -> None:
        """Set the frame used when assembling simulation inputs.

        Supported values:

        - ``"lab"`` — all reference frequencies are 0.0 GHz.
        - ``"rotating"`` — per-device references use dressed drive frequencies.
        - scalar-like — shared reference frequency for all devices.
        - ``dict`` — per-device references keyed by label or device.

        Frame changes never alter dressed-state data — dressing is
        always computed from the lab-frame static Hamiltonian.
        """
        if isinstance(frame, str):
            if frame not in ("lab", "rotating"):
                raise ValueError(f"frame string must be one of 'lab' or 'rotating', got {frame!r}")
            self._frame_spec = frame
            return

        if _is_scalar_like(frame):
            self._frame_spec = frame
            return

        if isinstance(frame, dict):
            self._frame_spec = {resolve_label(key): value for key, value in frame.items()}
            return

        raise TypeError(
            f"frame must be 'lab', 'rotating', a scalar-like frequency, or "
            f"dict[str|BaseDevice, scalar-like], got {type(frame).__name__}"
        )

    # ------------------------------------------------------------------
    # Dressed-state analysis — delegates to ChipAnalysis
    # ------------------------------------------------------------------

    @property
    def analysis(self) -> ChipAnalysis:
        """Dressed-state analysis namespace — the chip's :class:`ChipAnalysis`.

        Canonical entry point for the full dressed-analysis surface
        (power users, less-common methods). The common quantities are
        also exposed as flat ``chip.*`` forwarders — :meth:`energy`,
        :meth:`freq`, :meth:`dress`, :meth:`dispersive_shift`, … — which
        delegate here; reach for ``chip.analysis`` for everything else.
        """
        return self._analysis

    def _canonical_bare_labels(self) -> tuple[tuple[int, ...], ...]:
        """Internal delegate used by sweep/result utilities."""
        return self._analysis._canonical_bare_labels()

    def dress(
        self,
        *,
        overlap_threshold: float = 0.5,
        force: bool = False,
        labeling: str = "DE",
    ) -> DressedResult:
        """Compute (or retrieve) the dressed-state decomposition. See :meth:`ChipAnalysis.dress`."""
        return self._analysis.dress(
            overlap_threshold=overlap_threshold,
            force=force,
            labeling=labeling,
        )

    def _ensure_dressed(self) -> DressedResult:
        return self._analysis._ensure_dressed()

    @property
    def is_dressed(self) -> bool:
        """Whether a valid dressed-state result is cached. See :attr:`ChipAnalysis.is_dressed`."""
        return self._analysis.is_dressed

    def energy(
        self,
        device_states: Mapping[str | BaseDevice, int] | None = None,
        /,
        **device_state_kwargs: int,
    ) -> float:
        """Dressed eigenenergy (GHz) for a bare-state label. See :meth:`ChipAnalysis.energy`."""
        return self._analysis.energy(device_states, **device_state_kwargs)

    def dressed_spectrum(self) -> Any:
        """Raw dressed eigenvalue array without Python scalar coercion. See :meth:`ChipAnalysis.dressed_spectrum`."""
        return self._analysis.dressed_spectrum()

    def dressed_index(
        self,
        device_states: Mapping[str | BaseDevice, int] | None = None,
        /,
        **device_state_kwargs: int,
    ) -> int | None:
        """Dressed-state index matching a bare-state label, or ``None``. See :meth:`ChipAnalysis.dressed_index`."""
        return self._analysis.dressed_index(device_states, **device_state_kwargs)

    def bare_label(self, dressed_index: int) -> tuple[int, ...]:
        """Bare-state label assigned to a dressed-state index. See :meth:`ChipAnalysis.bare_label`."""
        return self._analysis.bare_label(dressed_index)

    def operator_in_dressed_basis(
        self,
        device: str | BaseDevice,
        op: str | Any,
        *,
        truncate: int | None = None,
    ) -> Operator:
        """Embedded device operator transformed to the dressed eigenbasis.

        See :meth:`ChipAnalysis.operator_in_dressed_basis`.
        """
        return self._analysis.operator_in_dressed_basis(device, op, truncate=truncate)

    def drive_matrix_elements(
        self,
        transition: str | BaseDevice | tuple[Mapping[str | BaseDevice, int], Mapping[str | BaseDevice, int]],
        *,
        drives: Sequence[str | BaseDrive] | None = None,
    ) -> LabelKeyedDict:
        """Return ``<final~|D_j|initial~>`` for wired drive lines.

        The final dressed state is the row index and the initial dressed
        state is the column index. In the weak-drive projection these matrix
        elements set the effective driven-Hamiltonian coefficients. See
        E. Magesan and J. M. Gambetta, Phys. Rev. A 101, 052308 (2020),
        DOI 10.1103/PhysRevA.101.052308, and
        :meth:`ChipAnalysis.drive_matrix_elements` for the transition
        shorthand, parameters, return type, and error conditions.
        """
        return self._analysis.drive_matrix_elements(transition, drives=drives)

    def state_components(
        self,
        state: int | Mapping[str | BaseDevice, int] | None = None,
        /,
        *,
        n_components: int = 5,
        **device_state_kwargs: int,
    ) -> dict[tuple[int, ...], float]:
        """Leading bare-basis probabilities for a dressed eigenstate. See :meth:`ChipAnalysis.state_components`."""
        return self._analysis.state_components(
            state,
            n_components=n_components,
            **device_state_kwargs,
        )

    def dispersive_shift(self, device_a: str | BaseDevice, device_b: str | BaseDevice) -> float:
        """Dressed cross-Kerr shift (GHz): ``E(1,1) − E(1,0) − E(0,1) + E(0,0)``.

        See :meth:`ChipAnalysis.dispersive_shift`.
        """
        return self._analysis.dispersive_shift(device_a, device_b)

    # ``static_zz`` is the same physics under a different name: the static ZZ
    # interaction strength equals the dressed dispersive (cross-Kerr) shift.
    static_zz = dispersive_shift

    def dressed_anharmonicity(self, device: str | BaseDevice) -> float:
        """Dressed anharmonicity of one device with others grounded (GHz).

        See :meth:`ChipAnalysis.dressed_anharmonicity`.
        """
        return self._analysis.dressed_anharmonicity(device)

    def effective_subspace_hamiltonian(
        self,
        states: (
            list[Mapping[str | BaseDevice, int] | tuple[int, ...]]
            | tuple[Mapping[str | BaseDevice, int] | tuple[int, ...], ...]
        ),
    ) -> Any:
        """Dressed effective Hamiltonian in a labeled bare subspace.

        See :meth:`ChipAnalysis.effective_subspace_hamiltonian`.
        """
        return self._analysis.effective_subspace_hamiltonian(states)

    @overload
    def freq(
        self,
        target: None = ...,
        when: dict[str | BaseDevice, int] | None = ...,
    ) -> dict[str, float]: ...

    @overload
    def freq(
        self,
        target: str | BaseDevice,
        when: dict[str | BaseDevice, int] | None = ...,
    ) -> float: ...

    def freq(
        self,
        target: str | BaseDevice | None = None,
        when: dict[str | BaseDevice, int] | None = None,
    ) -> dict[str, float] | float:
        """All dressed 0→1 frequencies (GHz), or one conditional transition. See :meth:`ChipAnalysis.freq`.

        Overloaded: no ``target`` returns the full ``{label: freq}`` dict;
        a single ``target`` (label or device) returns one scalar 0→1
        frequency. The runtime body is unchanged — under ``jax.jit`` the
        scalar is a traced 0-d array, so the overload is type-only and
        does not alter traceability.
        """
        return self._analysis.freq(target, when=when)

    def frame_info(self) -> dict[str, Any]:
        """Per-device frame reference frequency ``ω_ref,i`` (GHz). See :meth:`ChipAnalysis.frame_info`."""
        return self._analysis.frame_info()

    def physics_notes(self) -> dict[str, list[str]]:
        """Aggregate :meth:`physics_notes` across every component.

        Returns a dict keyed ``"chip"`` for the chip-level entry, and
        ``"<kind>:<label>"`` — ``kind`` one of ``"device"``, ``"coupling"``,
        ``"drive"``, ``"bath"`` — for every component, mapping to that
        component's declared approximations: Hilbert truncation, model
        regime, RWA status, noise-channel selection, and any other
        non-obvious assumption the component explicitly declares. Keys are
        kind-qualified rather than bare labels because the label namespaces
        are *not* globally disjoint — a device, coupling, drive, and bath may
        share a label — and this is an audit surface, so one component's
        entry silently overwriting another's is unacceptable. Drives are
        enumerated from :attr:`control_equipment`'s wiring rather than
        per-device ``connected_drives``, so an edge-target
        :class:`~quchip.control.drive.ParametricDrive` (pumping a coupling,
        not a device) is included too. The chip-level entry keyed ``"chip"``
        states the local-Lindblad approximation every collapse operator this
        chip assembles is built under (see :meth:`collapse_contributions`) —
        present even with no baths, since it applies regardless. Intended for
        inspection/audit rather than for runtime dispatch.
        """
        notes: dict[str, list[str]] = {
            "chip": [
                "Collapse operators are each component's LOCAL (bare-basis) operator combined "
                "with the dressed interacting Hamiltonian — the standard local-Lindblad "
                "approximation, not a dressed-basis (polaron-frame) master equation."
            ]
        }
        for device in self._devices:
            notes[f"device:{device.label}"] = list(device.physics_notes())
        if self.control_equipment is not None:
            for line in self.control_equipment.lines:
                notes[f"drive:{line.label}"] = list(line.physics_notes())
        for coupling in self._couplings:
            notes[f"coupling:{coupling.label}"] = list(coupling.physics_notes())
        for bath in self.baths:
            notes[f"bath:{bath.label}"] = list(bath.physics_notes())
        return notes

    # ------------------------------------------------------------------
    # Serialization and cloning
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize chip topology into a JSON-safe dictionary."""
        from quchip.chip.serialization import serialize_chip

        return serialize_chip(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chip":
        """Reconstruct a chip from :meth:`to_dict` output."""
        from quchip.chip.serialization import deserialize_chip

        return deserialize_chip(d)

    def clone(self) -> "Chip":
        """Isolated structural clone suitable for sweep evaluation."""
        from quchip.chip.serialization import clone_chip

        return clone_chip(self)

    def partition(self) -> "PartitionResult":
        """Split into independent sub-chips along the independence graph.

        Couplings, non-separable bath target sets, and drive-crosstalk pairs
        all count as connections. Exact — the joint solve factorizes as the
        tensor product of the component solves. ``simulate``/``seq.simulate``
        consult this automatically; call it directly to orchestrate solves
        yourself.
        """
        from quchip.chip.partition import partition_chip

        return partition_chip(self)

    def updated(self, update_fn: Callable[["Chip"], None]) -> "Chip":
        """Return a cloned chip after applying one structural update callback.

        Convenience for sweeps::

            modified = chip.updated(lambda c: setattr(c["q"], "freq", 5.1))
        """
        cloned = self.clone()
        update_fn(cloned)
        return cloned

    def status(self) -> None:
        """Print a lightweight diagnostic dashboard for the chip."""
        label = self.label if self.label is not None else "(unlabeled)"
        print(f"Chip: {label}")
        print(f"- devices: {len(self._devices)}")
        print(f"- couplings: {len(self._couplings)}")
        print(f"- frame: {self._frame_spec!r}")
        print(f"- rwa: {self._rwa}")
        print(f"- dressed: {'yes' if self.is_dressed else 'no'}")
        print("- device list:")
        for dev in self._devices:
            bare_freq = getattr(dev, "freq", None)
            connected = sorted(d.label for d in getattr(dev, "_connected_drives", []))
            line_text = ", ".join(connected) if connected else "none"
            print(
                f"  - {dev.label}: {type(dev).__name__} "
                f"(freq={_format_float(bare_freq)} GHz, "
                f"dressed={_format_float(dev.dressed_freq)} GHz, "
                f"levels={dev.levels}, lines={line_text})"
            )
        print("- couplings:")
        if self._couplings:
            for coupling in self._couplings:
                strength = coupling.coupling_strength
                print(
                    f"  - {type(coupling).__name__}: "
                    f"{coupling.device_a_label} <-> {coupling.device_b_label} "
                    f"(g={_format_float(strength)} GHz)"
                )
        else:
            print("  - none")
        if self._control_equipment is not None:
            signal_chain = self._control_equipment.signal_chain
            print(
                "- control equipment: "
                f"{len(self._control_equipment.lines)} lines, "
                f"{len(signal_chain)} signal chain transforms"
            )
        else:
            print("- control equipment: none")

    # ------------------------------------------------------------------
    # Observable helpers — delegate to quchip.chip.observables
    # ------------------------------------------------------------------

    def from_array(self, data: Any, device: str | BaseDevice | None = None) -> Any:
        """Build a backend operator from a raw NumPy array.

        With *device*, the array is interpreted as a local operator on
        that device's subspace and embedded into the full tensor-product
        space. With ``device=None`` the array must already span the full
        chip Hilbert space.
        """
        from quchip.chip.observables import from_array

        return from_array(self, data, device)

    def observable(self, device: str | BaseDevice, op: str | Any) -> Any:
        """Embed a device operator onto the full chip Hilbert space.

        Accepts either an operator name (``"X"``, ``"Y"``, ``"Z"``,
        ``"n"``, ``"a"``, ``"a_dag"``, ``"I"``) or an already-built
        local-space operator, and returns it embedded on the chip's
        tensor-product space.

        This is for manual full-space operator construction and
        analysis; it is *not* a solver ``e_op``. For solver expectation
        values use :meth:`e_ops`, which keeps operators *local* so the
        demodulation pipeline can band-decompose and embed them
        correctly.
        """
        from quchip.chip.observables import observable

        return observable(self, device, op)

    def e_ops(
        self,
        *,
        correlators: dict[
            tuple[str | BaseDevice, str | BaseDevice],
            tuple[str | Any, str | Any],
        ] | None = None,
        **specs: str | list | Any,
    ) -> dict[str | tuple[str, str], Any]:
        """Build a dict-form ``e_ops`` mapping for the solver pipeline.

        Each keyword maps a device label to an operator specification:
        a name string, a list of names, a raw local-space operator, or
        a mixed list of strings and operators. Two-device correlators
        (e.g. ``⟨Z₁⊗Z₂⟩``) are specified via *correlators* as
        device-label pairs → operator pairs. Returns local-space
        operators (not embedded) — the demodulation pipeline embeds as
        needed.

        Examples
        --------
        >>> from quchip import DuffingTransmon, Capacitive, Chip
        >>> q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q1")
        >>> q2 = DuffingTransmon(freq=5.2, anharmonicity=-0.22, levels=3, label="q2")
        >>> chip = Chip([q1, q2], couplings=[Capacitive(q1, q2, g=0.02)])
        >>> e = chip.e_ops(
        ...     q1=["X", "Y", "Z"], q2=["X", "Y", "Z"],
        ...     correlators={("q1", "q2"): ("Z", "Z")},
        ... )
        """
        from quchip.chip.observables import e_ops

        return e_ops(self, correlators=correlators, **specs)

    # ------------------------------------------------------------------
    # State factories
    # ------------------------------------------------------------------

    def set_state_order(
        self,
        *devices: str | BaseDevice,
        levels: Mapping[str, int] | None = None,
    ) -> None:
        """Declare the device order used to parse string-state shorthands.

        After this is called, :meth:`bare_state`, :meth:`state`, and
        :meth:`superposition` accept single-string specifications where
        each character is one level per device in *devices* order.
        Level symbols default to ``g=0, e=1, f=2, h=3``; digits ``0..9``
        are always accepted as raw Fock indices.

        Every chip device must be named exactly once.

        Examples
        --------
        >>> from quchip import DuffingTransmon, Resonator, Chip
        >>> qb = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="qb")
        >>> tc = DuffingTransmon(freq=5.5, anharmonicity=-0.20, levels=3, label="tc")
        >>> cr = Resonator(freq=7.0, levels=4, label="cr")
        >>> chip = Chip([qb, tc, cr])
        >>> chip.set_state_order(qb, tc, cr)
        >>> _ = chip.bare_state("eg1")  # {qb: 1, tc: 0, cr: 1}
        """
        from quchip.chip.states import set_state_order

        set_state_order(self, *devices, levels=levels)

    def superposition(
        self,
        *components: Mapping[str | BaseDevice, int] | str | tuple[Any, Any],
    ) -> State:
        """Normalized bare-basis superposition of tensor-product states.

        Each component is either a bare-state spec (dict keyed by device
        or label, or a string when :meth:`set_state_order` has been
        called) or an ``(amplitude, spec)`` tuple for weighted mixing.
        Uniform weights by default; results are normalized to unit norm.

        Unlike :meth:`state`, this stays in the bare product basis — no
        dressed diagonalization — so the probe basis is explicit.

        Examples
        --------
        >>> import numpy as np
        >>> from quchip import DuffingTransmon, Resonator, Chip
        >>> qb = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="qb")
        >>> cr = Resonator(freq=7.0, levels=4, label="cr")
        >>> chip = Chip([qb, cr])
        >>> _ = chip.superposition({qb: 0}, {qb: 1})  # equal |00> + |10>
        >>> _ = chip.superposition(                   # weighted mix
        ...     (np.sqrt(0.3), {qb: 1, cr: 0}),
        ...     (np.sqrt(0.7), {qb: 1, cr: 1}),
        ... )
        """
        from quchip.chip.states import superposition

        return superposition(self, *components)

    def state(
        self,
        device_states: Mapping[str | BaseDevice, int] | str | None = None,
        /,
        **device_state_kwargs: int,
    ) -> State:
        """Dressed eigenstate for the given Fock-indexed bare-state labels.

        Accepts a string shorthand (e.g. ``"eg1"``) when
        :meth:`set_state_order` has been called.

        Safe inside ``jax.jit``/``grad``/``vmap``: under tracing the
        assigned eigenvector column is selected through the
        :func:`~quchip.chip.dressing.label_eigensystem` array kernel, so
        dressed initial states are differentiable end-to-end. The global
        phase is gauge-dependent (``eigh`` column convention) —
        populations and ``|overlap|`` figures of merit are unaffected.
        """
        from quchip.chip.states import state

        return state(self, device_states, **device_state_kwargs)

    def bare_state(
        self,
        device_states: Mapping[str | BaseDevice, int | State] | str | None = None,
        /,
        **device_state_kwargs: int | State,
    ) -> State:
        """Bare tensor-product state from per-device Fock indices or kets.

        Each device may be specified as either a Fock index (``int``) or
        a ket vector in that device's local space. Devices not mentioned
        default to the ground state (Fock index 0). Unlike :meth:`state`
        this does **not** diagonalize the coupled system.

        Accepts a string shorthand (e.g. ``"eg1"``) when
        :meth:`set_state_order` has been called.
        """
        from quchip.chip.states import bare_state

        return bare_state(self, device_states, **device_state_kwargs)

    # ------------------------------------------------------------------
    # Control equipment
    # ------------------------------------------------------------------

    def wire(
        self,
        *lines: BaseDrive,
        signal_chain: Sequence[SignalTransform] | None = None,
    ) -> ControlEquipment:
        """Attach or replace classical control wiring.

        Preferred user-facing API for attaching control to a chip. If
        *lines* is omitted, the existing connected lines are reused and
        only the signal chain is replaced.

        Examples
        --------
        >>> # chip.wire(drive_q, drive_r, signal_chain=[Crosstalk(drive_q, drive_r, c=0.05)])  # doctest: +SKIP
        """
        if lines:
            for line in lines:
                if not isinstance(line, BaseDrive):
                    raise TypeError(
                        f"chip.wire(...) expects drive objects. Got {type(line).__name__}."
                    )
            connected_lines = list(lines)
        else:
            if self._control_equipment is None:
                raise ValueError(
                    "chip.wire() requires at least one drive when no control "
                    "equipment is attached yet."
                )
            connected_lines = self._control_equipment.lines

        duplicates = [lbl for lbl, count in Counter(d.label for d in connected_lines).items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate drive labels in equipment: {duplicates}. "
                "Each drive must have a unique label."
            )

        if signal_chain:
            drive_labels = {d.label for d in connected_lines}
            for transform in signal_chain:
                for value in transform.referenced_lines():
                    if value not in drive_labels:
                        raise ValueError(
                            f"Signal chain references drive '{value}' not in equipment. "
                            f"Available: {sorted(drive_labels)}"
                        )

        equipment = ControlEquipment(lines=connected_lines, signal_chain=list(signal_chain) if signal_chain else None)
        self.connect(equipment)
        return equipment

    def unwire(self, line: BaseDrive | str) -> BaseDrive:
        """Remove one control line and every signal-chain transform referencing it.

        The inverse of :meth:`wire` for a single line. Accepts the drive
        object or its label. Returns the removed drive so it can be rewired
        later. Removing the last line detaches the equipment entirely
        (``control_equipment`` becomes ``None``).
        """
        if self._control_equipment is None:
            raise ValueError("chip.unwire(...) requires connected control equipment; nothing is wired.")
        label = resolve_label(line)
        lines = self._control_equipment.lines
        remaining = [d for d in lines if d.label != label]
        if len(remaining) == len(lines):
            available = [d.label for d in lines]
            raise ValueError(f"No control line labeled '{label}' in equipment. Available: {available}")
        removed = next(d for d in lines if d.label == label)
        kept_chain = []
        for transform in self._control_equipment.signal_chain:
            retained = transform.without_line(label)
            if retained is not None:
                kept_chain.append(retained)
        if remaining:
            self._control_equipment = ControlEquipment(lines=remaining, signal_chain=kept_chain or None)
        else:
            self._control_equipment = None
        return removed

    def connect(self, control_equipment: ControlEquipment) -> None:
        """Attach control equipment to this chip (low-level API).

        Validates every drive target and rejects duplicate drive labels,
        then reconnects each drive to this chip's canonical device
        instances. User-facing code should prefer :meth:`wire`.
        """
        drive_duplicates = [
            lbl for lbl, n in Counter(d.label for d in control_equipment.lines).items() if n > 1
        ]
        if drive_duplicates:
            raise ValueError(
                f"Duplicate drive labels in equipment: {drive_duplicates}. "
                "Each drive must have a unique label."
            )
        for drive in control_equipment.lines:
            if drive.target_kind == "edge":
                target_label = drive.target_label
                if target_label not in self._coupling_map:
                    raise ValueError(
                        f"Edge line '{drive.label}' targets coupling '{target_label}', which is not on "
                        f"this chip. Available couplings: {list(self._coupling_map.keys())}"
                    )
                continue
            if drive._target is None:
                raise ValueError(
                    "ControlEquipment contains a drive with no connected target "
                    f"({drive!r}). Connect drives to chip devices before "
                    "calling chip.connect(control_equipment)."
                )
            target_label = drive._target.label
            if target_label not in self._device_map:
                raise ValueError(
                    f"Drive target '{target_label}' is not on this chip. Available: {list(self._device_map.keys())}"
                )

        self._control_equipment = control_equipment

        for drive in control_equipment.lines:
            if drive.target_kind == "edge":
                assert drive.target_label is not None
                coupling = self._coupling_map[drive.target_label]
                if coupling.parametric_operator(self) is None:
                    raise TypeError(
                        f"{type(coupling).__name__} is not modulable: its parametric_interaction() hook "
                        "returns None. Implement parametric_interaction()/rwa_parametric_interaction() on "
                        "the coupling (see CouplingModel), or use a modulable coupling such as TunableCapacitive."
                    )
                drive.connect(coupling)  # type: ignore[arg-type]  # ParametricDrive.connect accepts a coupling
                continue
            assert drive._target is not None
            drive.connect(self._device_map[drive._target.label])

    def disconnect(self) -> ControlEquipment:
        """Detach control equipment entirely (low-level API).

        The inverse of :meth:`connect`/:meth:`wire`: removes all lines and
        the signal chain at once (``control_equipment`` becomes ``None``).
        Returns the detached equipment so it can be reconnected later.

        Returns
        -------
        ControlEquipment
            The equipment that was attached before detachment.
        """
        if self._control_equipment is None:
            raise ValueError("chip.disconnect() requires connected control equipment; nothing is wired.")
        equipment = self._control_equipment
        self._control_equipment = None
        return equipment

    # ------------------------------------------------------------------
    # Typed solve surface
    # ------------------------------------------------------------------

    def _check_problem(self, problem: Any, index: int | None = None) -> None:
        """Validate that *problem* is a SolveProblem built for *this* chip.

        Raises ``TypeError`` when *problem* does not duck-type as a
        :class:`SolveProblem`, or ``ValueError`` when it was built for a
        different chip. When *index* is given the message is phrased for the
        ``problems[index]`` list position (used by :meth:`solve_many`);
        otherwise it is phrased for a single problem (used by :meth:`solve`).
        """
        if not hasattr(problem, "hamiltonian") or not hasattr(problem, "chip"):
            type_name = type(problem).__name__
            if index is None:
                raise TypeError(f"Expected SolveProblem, got {type_name}")
            raise TypeError(f"problems[{index}]: expected SolveProblem, got {type_name}")
        if getattr(problem, "chip", None) is not self:
            if index is None:
                raise ValueError(
                    "SolveProblem was built for a different chip. Use the same chip instance that produced the problem."
                )
            raise ValueError(
                f"problems[{index}] was built for a different chip. All problems must share the same chip instance."
            )

    def solve(
        self,
        problem: "SolveProblem",
        *,
        check_truncation: bool = True,
        truncation_threshold: float = 1e-3,
    ) -> "SimulationResult":
        """Solve a typed :class:`SolveProblem` through this chip's backend.

        Routes through the common :func:`~quchip.engine.solve_problem`
        chokepoint, so the Hilbert-truncation safety net applies by default;
        pass ``check_truncation=False`` to opt out or retune
        ``truncation_threshold``.
        """
        from quchip.engine import solve_problem

        self._check_problem(problem)
        return solve_problem(
            problem,
            check_truncation=check_truncation,
            truncation_threshold=truncation_threshold,
        )

    def solve_many(self, batch_or_problems: Any, *, progress: bool = True) -> "SimulationBatchResult":
        """Solve a :class:`SolveBatch`, :class:`ProblemBatch`, or list of problems.

        Chip-level validation only enforces what needs ``self`` (every input
        was built for *this* chip); the input-shape dispatch and batching are
        delegated to :func:`quchip.engine.solve_many`, which owns the single
        ProblemBatch / SolveBatch / list ladder.
        """
        from quchip.control.batch import ProblemBatch
        from quchip.engine import solve_many
        from quchip.engine.ir import SolveBatch

        if isinstance(batch_or_problems, ProblemBatch):
            if batch_or_problems.batch.chip is not self:
                raise ValueError("ProblemBatch was built for a different chip.")
        elif isinstance(batch_or_problems, SolveBatch):
            if batch_or_problems.chip is not self:
                raise ValueError("SolveBatch was built for a different chip.")
        else:
            batch_or_problems = list(batch_or_problems)
            for i, problem in enumerate(batch_or_problems):
                self._check_problem(problem, index=i)

        return solve_many(batch_or_problems, progress=progress)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Sectioned plain-text report of everything on the chip.

        Devices with their declared parameters (units included), noise
        settings, couplings, control wiring, and baths — the "what did I
        just build?" view. Returns a string; ``print(chip.describe())``.
        Traced parameters render as ``<traced>`` and are never
        concretized.
        """
        from quchip.chip.describe import describe_chip

        return describe_chip(self)

    def __repr__(self) -> str:
        return (
            f"Chip(label={self.label!r}, devices={len(self._devices)}, "
            f"couplings={len(self._couplings)}, frame={self._frame_spec!r}, rwa={self._rwa}, "
            f"dressed={'yes' if self.is_dressed else 'no'})"
        )

    def resolve_rwa(self, term_owner: Any) -> bool:
        """Resolve a coupling's or drive's RWA flag against the chip default.

        Returns the chip's default when the owner's ``rwa`` is ``None``;
        otherwise returns the owner's explicit flag.
        """
        rwa = getattr(term_owner, "rwa", None)
        if rwa is None:
            return self._rwa
        return bool(rwa)
