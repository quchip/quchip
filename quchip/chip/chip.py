"""Composite quantum system: :class:`Chip` holds devices, couplings, and control.

A chip owns the devices and the couplings between them. Each device owns its
local Hamiltonian, and each coupling owns its two-body interaction Hamiltonian.
The chip can also own the :class:`ControlEquipment` that wires the classical
control lines. The engine turns a chip into a solver-ready problem.

All public parameters, including device frequencies, coupling strengths, drive
amplitudes, and crosstalk coefficients, stay JAX-traceable and sweepable, so a
single loss function can span any of them.
"""

from __future__ import annotations

from collections import Counter
from math import prod
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence, overload

import numpy as np
from jax.core import Tracer

from quchip.backend import _backend_context
from quchip.backend.protocol import Backend, Operator, State
from quchip.approximations import Approximation, RWA, require_approximation
from quchip.chip.analysis import ChipAnalysis, DressedResult, KerrMatrix
from quchip.chip.baths import Bath
from quchip.chip.effective import EffectiveTerms
from quchip.chip.port_network import PortNetwork
from quchip.chip.ports import Port
from quchip.chip.coupling_base import BaseCoupling
from quchip.chip.states import _DEFAULT_LEVEL_SYMBOLS
from quchip.control.drive import BaseDrive, CouplingDrive
from quchip.control.equipment import ControlEquipment
from quchip.control.signal import Crosstalk, SignalTransform
from quchip.declarative.expr import PhysicsExpr
from quchip.devices.base import BaseDevice
from quchip.engine.frames import FramePlan
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar
from quchip.utils.labeling import LabelKeyedDict, resolve_label
from quchip.utils.values import TracedKey, scoped_entry, scoped_hit

if TYPE_CHECKING:
    from quchip.chip.partition import PartitionResult
    from quchip.engine.ir import EngineResult, FrameSpec, SolveProblem
    from quchip.results.results import SimulationBatchResult, SimulationResult
    from quchip.results.steady_state import SteadyStateResult


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


def _as_physics_expr(
    authored: Any,
    backend: Backend,
    *,
    labels: tuple[str, ...],
    dims: tuple[int, ...],
    name: str,
) -> PhysicsExpr:
    """Return an authored expression or wrap a native matrix contribution."""
    if isinstance(authored, PhysicsExpr):
        return authored
    return PhysicsExpr.from_matrix(backend.to_array(authored), labels=labels, dims=dims, name=name)


def _concrete_cache_value(value: Any, *, traced: bool = False) -> Any:
    """Return one stable scalar cache value or raise for non-scalars.

    A tracer keys by identity when ``traced`` is true and raises otherwise.
    """
    if value is None:
        return None
    if traced and isinstance(value, Tracer):
        return TracedKey(id(value))
    concrete = maybe_concrete_scalar(value)
    if concrete is None:
        raise ValueError
    return concrete


def _identity_if_opaque(fingerprint: Callable[[], Any], owner: Any) -> Any:
    """Content key from *fingerprint*, or *owner*'s identity when its payload is traced or opaque."""
    try:
        return fingerprint()
    except ValueError:
        return TracedKey(id(owner))


def _operator_cache_value(value: Any) -> Any:
    """Return a content-based cache key for one concrete port operator."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, PhysicsExpr) and value.kind == "matrix":
        from quchip.utils.values import value_fingerprint

        # Key the stored payload; a traced payload raises ValueError and disables the cache.
        payload, dims, _name, changes = value.args
        return value_fingerprint((payload, dims, value.labels, None if changes is None else tuple(sorted(changes))))
    operator = value
    if hasattr(operator, "matrix"):
        operator = operator.matrix()
    elif hasattr(operator, "to_jax"):
        operator = operator.to_jax()
    elif hasattr(operator, "full"):
        operator = operator.full()
    if contains_tracer(operator):
        raise ValueError
    try:
        array = np.asarray(operator)
    except (TypeError, ValueError) as error:
        raise ValueError from error
    return array.shape, array.dtype.str, array.tobytes()


def _frame_cache_value(frame: Any, *, traced: bool = False) -> Any:
    """Return a stable cache key for one frame specification; see :func:`_concrete_cache_value`."""
    if isinstance(frame, str):
        return frame
    if isinstance(frame, FramePlan):
        # A traced plan raises ValueError here, which disables the resolve cache.
        return ("plan", frame.concrete_key())
    if isinstance(frame, Mapping):
        return tuple(
            sorted(
                (resolve_label(label), _concrete_cache_value(value, traced=traced))
                for label, value in frame.items()
            )
        )
    return _concrete_cache_value(frame, traced=traced)


class Chip:
    """Composite quantum system: devices + couplings + (optional) control.

    Device arguments accept an object or its label. The ordered device list
    fixes tensor-product positions.

    Parameters
    ----------
    devices : list[BaseDevice]
        The tensor-product position equals the list index. Labels must be
        unique.
    couplings : list[BaseCoupling], optional
        Two-body couplings. Every device a coupling refers to must be in
        ``devices``.
    control_equipment : ControlEquipment, optional
        Holds the drive lines and the signal-chain transforms (crosstalk,
        delays, gains). You can also attach it later with :meth:`wire` or
        :meth:`connect`.
    label : str, optional
        Human-readable chip label.
    frame : FrameSpec
        Initial frame specification:

        - ``"lab"``: all reference frequencies are 0 GHz (default).
        - ``"rotating"``: a per-device rotating frame at the dressed drive
          frequencies.
        - ``"auto"``: per-device frequencies that quchip chooses from retained
          couplings, cascade-generated network couplings, delivered drive
          tones, and scattering-scaled coherent-input tones.
        - scalar-like: one shared reference frequency for all devices.
        - ``dict``: per-device references keyed by label or device.
    approximation : Approximation
        Strategy for dressed analysis and solver assembly. The default is
        :class:`~quchip.approximations.RWA`. To keep every band, use ``Exact()``.
    basis : {"native", "eigen"}
        Chip-wide policy for the local solver basis. ``"native"`` keeps each
        device's authored coordinate basis. ``"eigen"`` transforms into each
        device's retained local energy subspace. A device-level ``basis``
        overrides this policy.
    backend : str or Backend, optional
        Chip-specific backend. ``None`` uses the process default.
    baths : list[Bath], optional
        Shared or collective unobserved environments.
    port_network : PortNetwork, optional
        Complete accessible field boundary. Attach at most one network.

    effective_terms : sequence[EffectiveTerms]
        Captured contributions that a reduction made. The engine keeps their
        retained operator bands when it applies the selected frame.

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
        approximation: Approximation = RWA(),
        basis: Literal["native", "eigen"] = "native",
        backend: str | Backend | None = None,
        baths: list[Bath] | None = None,
        port_network: PortNetwork | None = None,
        effective_terms: Sequence[EffectiveTerms] = (),
    ) -> None:
        duplicates = [lbl for lbl, count in Counter(d.label for d in devices).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate device labels: {duplicates}. All device labels must be unique.")

        self.label = label
        self._devices = tuple(devices)
        self._device_map: dict[str, BaseDevice] = {d.label: d for d in devices}
        self._label_to_index: dict[str, int] = {d.label: i for i, d in enumerate(devices)}
        self._effective_terms = tuple(effective_terms)
        for terms in self._effective_terms:
            if not isinstance(terms, EffectiveTerms):
                raise TypeError("effective_terms must contain EffectiveTerms values.")
            terms.validate_for(self)
        projected_labels: set[str] = set()
        for terms in self._effective_terms:
            if terms.projection is not None:
                if projected_labels.intersection(terms.labels):
                    raise ValueError("Retained operator projections cannot overlap; compose them first.")
                projected_labels.update(terms.labels)
        if len({terms.label for terms in self._effective_terms}) != len(self._effective_terms):
            raise ValueError("Effective contribution labels must be unique.")
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
        shadowed = [t.label for t in self._effective_terms if t.label in self._device_map or t.label in coupling_labels]
        if shadowed:
            raise ValueError(
                f"Effective-term labels collide with device or coupling labels: {shadowed}. "
                "They share one namespace so eliminate() resolves targets unambiguously."
            )
        self._coupling_map: dict[str, BaseCoupling] = {c.label: c for c in self._couplings}
        bath_duplicates = [lbl for lbl, n in Counter(b.label for b in baths or ()).items() if n > 1]
        if bath_duplicates:
            raise ValueError(f"Duplicate bath labels: {bath_duplicates}. Each bath must have a unique label.")
        for bath in baths or ():
            self._validate_bath(bath)
        self._baths = tuple(baths) if baths else ()
        if port_network is not None and not isinstance(port_network, PortNetwork):
            raise TypeError(
                f"Expected a PortNetwork, got {type(port_network).__name__}: {port_network!r}"
            )
        if port_network is not None:
            port_network.validate_for(self)
        self._port_network = port_network
        if basis not in ("native", "eigen"):
            raise ValueError(f"basis must be 'native' or 'eigen', got {basis!r}")
        self._basis = basis
        tuple(d.resolved_dimension(basis) for d in devices)
        self._frame_spec: FrameSpec = "lab"
        self._approximation = require_approximation(approximation)
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

        # Both snapshots are keyed by their complete structural inputs below;
        # values produced under a JAX trace are never retained.
        self._unresolved_hamiltonian_cache: tuple[Any, Any, PhysicsExpr] | None = None
        self._resolved_result_cache: tuple[Any, Any, EngineResult] | None = None

        if frame != "lab":
            self.set_frame(frame)

    # ------------------------------------------------------------------
    # Hamiltonian
    # ------------------------------------------------------------------

    def unresolved_hamiltonian(self) -> PhysicsExpr:
        """Return the authored lab-frame static Hamiltonian.

        This method embeds every device Hamiltonian into the full tensor space
        and adds each embedded coupling interaction. It does **not** apply the
        rotating-frame transform or any drive envelopes.

        The result is exact, because only the engine applies the RWA and the
        frame transformations. At solve time, the engine subtracts
        ``2π Σ_i ω_ref,i n̂_i`` for each device, where ``ω_ref,i`` is the frame
        reference resolved by :func:`quchip.engine.frames.resolve_frame`. To
        see each device's reference frequency, use :meth:`frame_info`.

        Returns
        -------
        PhysicsExpr
            Symbolic Hamiltonian on ``⨂_d H_d``. For a dense numerical view
            with the current bindings, call ``.matrix()``.
        """
        # The 2π boundary crossing and the subtraction are both in
        # `quchip.engine.assembly._build_static_h0`.
        from quchip.declarative.parameters import component_fingerprint

        backend = self.backend
        signature = (
            type(backend).__qualname__,
            tuple(component_fingerprint(d, traced=True) for d in self._devices),
            tuple(component_fingerprint(c, traced=True) for c in self._couplings),
            tuple(id(terms) for terms in self.effective_terms),
        )
        cache = self._unresolved_hamiltonian_cache
        if scoped_hit(cache, signature):
            return cache[2]

        with _backend_context(backend):
            labels = tuple(device.label for device in self._devices)
            H: PhysicsExpr | None = None
            for i, dev in enumerate(self._devices):
                h_local = _as_physics_expr(
                    dev.unresolved_hamiltonian(),
                    backend,
                    labels=(dev.label,),
                    dims=(dev.local_space().dimension,),
                    name=rf"\hat H_{{{dev.label}}}",
                )
                h_emb = h_local.embed(labels, self.authored_dims)
                H = h_emb if H is None else H + h_emb

            for coupling in self._couplings:
                idx_a = self._label_to_index[coupling.device_a_label]
                idx_b = self._label_to_index[coupling.device_b_label]
                local_labels = (coupling.device_a_label, coupling.device_b_label)
                local_dims = (
                    self._devices[idx_a].local_space().dimension,
                    self._devices[idx_b].local_space().dimension,
                )
                h_int = _as_physics_expr(
                    coupling.interaction_hamiltonian(),
                    backend,
                    labels=local_labels,
                    dims=local_dims,
                    name=rf"\hat H_{{{coupling.label}}}",
                )
                H = H + h_int.embed(labels, self.authored_dims)

        assert H is not None
        for terms in self.effective_terms:
            H = H + terms.expression().embed(labels, self.authored_dims)
        # Expressions whose values belong to a JAX trace are reused only inside that trace.
        self._unresolved_hamiltonian_cache = scoped_entry(signature, H, traced=contains_tracer(H.numeric_values()))
        return H

    def hamiltonian(self) -> PhysicsExpr:
        """Return the Hamiltonian after the chip's engine policies."""
        return self.resolve().hamiltonian()

    def resolve(
        self,
        *,
        frame: FrameSpec | None = None,
        approximation: Approximation | None = None,
    ) -> EngineResult:
        """Return a frozen engine contract without mutating chip intent.

        ``frame=None`` uses :attr:`frame`. An explicit frame resolves only this
        snapshot. ``approximation=None`` uses the chip default. Neither
        override mutates chip intent.

        Parameters
        ----------
        frame : FrameSpec or None, default=None
            Snapshot frame, or ``None`` to use :attr:`frame`.
        approximation : Approximation or None, default=None
            Snapshot approximation, or ``None`` to use :attr:`approximation`.
        """
        return self._resolve(frame=frame, approximation=approximation)

    def _resolve(
        self, *, frame: FrameSpec | None = None, approximation: Approximation | None = None,
        _resolution: Any = None,
    ) -> EngineResult:
        """Resolve with optional preparation supplied by frame planning."""
        from quchip.engine.assembly import (
            _prepare_engine_assembly,
            build_engine_result,
        )

        from quchip.declarative.parameters import component_fingerprint

        frame_spec = self.frame if frame is None else frame
        strategy = self.approximation if approximation is None else require_approximation(approximation)
        equipment = self.control_equipment
        control_lines = () if equipment is None else equipment.lines
        try:
            signature = (
                id(self.backend),
                strategy,
                self.basis,
                _frame_cache_value(frame_spec, traced=True),
                tuple(component_fingerprint(device, traced=True) for device in self.devices),
                tuple(
                    (device.label, _concrete_cache_value(device._reference_freq_override, traced=True))
                    for device in self.devices
                ),
                tuple(
                    (device.label, id(type(device).dissipation))
                    for device in self.devices
                ),
                tuple(component_fingerprint(coupling, traced=True) for coupling in self.couplings),
                tuple(_identity_if_opaque(terms.fingerprint, terms) for terms in self.effective_terms),
                tuple(
                    (
                        line.label,
                        type(line),
                        line.target_label,
                        tuple(
                            (name, _concrete_cache_value(getattr(line, name), traced=True))
                            for name in line.parameter_values()
                        ),
                        id(type(line).dissipation),
                    )
                    for line in control_lines
                ),
                tuple(
                    (
                        bath.label,
                        bath.recipe,
                        tuple(bath.resolve_targets(self)),
                        _concrete_cache_value(bath.temperature, traced=True),
                        _concrete_cache_value(bath.rate, traced=True),
                    )
                    for bath in self.baths
                ),
                tuple(
                    (
                        port.label,
                        tuple(port.resolve_targets(self)),
                        tuple(
                            (name, _concrete_cache_value(value, traced=True))
                            for name, value in port.parameter_values().items()
                        ),
                        _identity_if_opaque(lambda: _operator_cache_value(port.operator), port.operator),
                    )
                    for port in self.ports
                ),
                None if (network := self.port_network) is None else _identity_if_opaque(network.fingerprint, network),
            )
        except ValueError:
            signature = None

        cache = self._resolved_result_cache
        if signature is not None and scoped_hit(cache, signature):
            return cache[2]

        local_resolution, resolved_frame = _prepare_engine_assembly(self, frame_spec, strategy, resolution=_resolution)
        result = build_engine_result(
            self,
            [],
            resolved_frame=resolved_frame,
            approximation=strategy,
            _local_resolution=local_resolution,
        )
        if signature is not None:
            # A traced resolution is reused only inside the JAX trace that produced it.
            self._resolved_result_cache = scoped_entry(signature, result, traced=result._contains_tracer())
        return result

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

    def dynamic_contributions(self) -> list[tuple[Operator, Any, tuple[int, ...], Any, str, str]]:
        """Return the time-dependent Hamiltonian contributions that the chip owns, with their support.

        Returns ``(local_op, time_dependence, support, owner, origin, tag)``
        tuples. A device-local operator has one support index, and a coupling's
        two-body operator has two. Operators are in the lab frame and in
        ordinary GHz, and the engine applies the 2π boundary. Drive terms are
        *not* included, because the schedule owns them.
        """
        # Drive terms enter through the engine assembly's drive compilation.
        backend = self.backend
        out: list[tuple[Operator, Any, tuple[int, ...], Any, str, str]] = []
        with _backend_context(backend):
            for coupling in self._couplings:
                terms = coupling._time_terms()
                if not terms:
                    continue
                idx_a = self._label_to_index[coupling.device_a_label]
                idx_b = self._label_to_index[coupling.device_b_label]
                out.extend(
                    (
                        term.operator,
                        term.coefficient,
                        (idx_a, idx_b),
                        coupling,
                        "coupling",
                        "coupling_dynamic",
                    )
                    for term in terms
                )
            for i, dev in enumerate(self._devices):
                out.extend(
                    (term.operator, term.coefficient, (i,), dev, "device", "device_dynamic")
                    for term in dev._time_terms()
                )
        return out

    def collapse_contributions(
        self,
        bases: Mapping[str, Any] | None = None,
    ) -> list[tuple[Operator, Any, tuple[int, ...], str, str, tuple[str, ...]]]:
        """Return every Lindblad collapse operator on the chip, with its support.

        Returns ``(operator, rate, support, source, channel, parameter_paths)``
        tuples. An operator local to a device or a drive line has one device
        index, and a coupling's two-body operator has two. An operator already
        embedded in the full space (baths) has an empty tuple. Rates are in
        1/ns and are ready for the Lindblad equation. Each component owns its
        rate physics, including any intrinsic 2π (for example, κ = 2π·f/Q for a
        resonator).

        Every returned operator is authored locally, and the engine transforms
        it into the same fixed local solver bases as the Hamiltonian. This is
        the standard local-Lindblad approximation (Breuer & Petruccione, *The
        Theory of Open Quantum Systems*, Oxford, 2002, Ch. 3). It is not a
        dressed-basis (polaron-frame) master equation. It applies to the full
        chip, independent of which components have noise. See the ``"chip"``
        entry of :meth:`physics_notes`.

        Parameters
        ----------
        bases : mapping or None, default=None
            Captured basis records, or ``None`` to resolve them.
        """
        return [
            (operator, rate, support, source, channel, parameter_paths)
            for operator, rate, support, source, channel, parameter_paths, _owner in (
                self._collapse_contributions_with_owners(bases)
            )
        ]

    def _collapse_contributions_with_owners(
        self,
        bases: Mapping[str, Any] | None = None,
        owners: Sequence[Any] | None = None,
    ) -> list[tuple[Operator, Any, tuple[int, ...], str, str, tuple[str, ...], Any]]:
        """Return collapse contributions with exact component ownership for assembly.

        ``owners`` limits the result to the channels of those components.
        """
        backend = self.backend
        out: list[
            tuple[Operator, Any, tuple[int, ...], str, str, tuple[str, ...], Any]
        ] = []
        with _backend_context(backend):
            for i, dev in enumerate(self._devices):
                out.extend(
                    (
                        channel.operator,
                        channel.rate,
                        (i,),
                        dev.label,
                        channel.name,
                        parameter_paths,
                        dev,
                    )
                    for channel, parameter_paths in dev._collapse_channels_with_paths(
                        None if bases is None else bases[dev.label]
                    )
                )
            if self.control_equipment is not None:
                for line in self.control_equipment.lines:
                    if line.device_label is None:
                        continue
                    idx = self._label_to_index[line.device_label]
                    device = self._devices[idx]
                    for channel, parameter_paths in line._collapse_channels_with_paths(device):
                        out.append(
                            (
                                channel.operator,
                                channel.rate,
                                (idx,),
                                line.label,
                                channel.name,
                                parameter_paths,
                                line,
                            )
                        )
            for coupling in self._couplings:
                idx_a = self._label_to_index[coupling.device_a_label]
                idx_b = self._label_to_index[coupling.device_b_label]
                for channel, parameter_paths in coupling._collapse_channels_with_paths():
                    out.append(
                        (
                            channel.operator,
                            channel.rate,
                            (idx_a, idx_b),
                            coupling.label,
                            channel.name,
                            parameter_paths,
                            coupling,
                        )
                    )
            for bath in self.baths:
                for channel, parameter_paths in bath._collapse_channels_with_paths(self, bases):
                    out.append(
                        (
                            channel.operator,
                            channel.rate,
                            (),
                            bath.label,
                            channel.name,
                            parameter_paths,
                            bath,
                        )
                    )
            for terms in self.effective_terms:
                support = tuple(self._label_to_index[label] for label in terms.labels)
                for channel in terms.channels:
                    out.append((terms.channel_expression(channel), channel.rate, support,
                                terms.label, channel.name, (), terms))
            for port in self.ports:
                support = tuple(self._label_to_index[label] for label in port.resolve_targets(self))
                for channel, parameter_paths in port._collapse_channels_with_paths(self):
                    out.append(
                        (
                            channel.operator,
                            channel.rate,
                            support,
                            port.label,
                            channel.name,
                            parameter_paths,
                            port,
                        )
                    )
        if owners is not None:
            wanted = {id(owner) for owner in owners}
            out = [contribution for contribution in out if id(contribution[-1]) in wanted]
        if any(terms.projection is not None for terms in self.effective_terms):
            from quchip.chip.effective import retained_operator

            projected = []
            labels = tuple(device.label for device in self.devices)
            for operator, rate, support, source, channel_name, paths, owner in out:
                if not isinstance(owner, EffectiveTerms) and not (
                    isinstance(owner, Bath) and owner._retained is not None
                ):
                    operator_labels = tuple(labels[index] for index in support) if support else labels
                    owner_key = f"port:{owner.label}" if isinstance(owner, Port) else None
                    retained = retained_operator(self, operator, operator_labels, backend, owner_key,
                                                 bases=bases, local_bases=bases)
                    if retained is not None:
                        operator = retained
                        support = tuple(self._label_to_index[label] for label in retained.labels)
                projected.append((operator, rate, support, source, channel_name, paths, owner))
            return projected
        return out

    # ------------------------------------------------------------------
    # Structural properties
    # ------------------------------------------------------------------

    @property
    def effective_terms(self) -> tuple[EffectiveTerms, ...]:
        """Captured Hamiltonian and loss contributions that are retained."""
        for terms in self._effective_terms:
            terms.validate_for(self)
        return self._effective_terms

    @property
    def device_map(self) -> dict[str, BaseDevice]:
        """Label → device mapping (the chip's own dict, do not mutate it)."""
        return self._device_map

    @property
    def coupling_map(self) -> dict[str, "BaseCoupling"]:
        """Couplings by label, in insertion order (do not mutate them)."""
        return self._coupling_map

    def coupling(self, coupling: "str | BaseCoupling") -> "BaseCoupling":
        """Return a coupling by label or object.

        Parameters
        ----------
        coupling : str or BaseCoupling
            Coupling label or object.
        """
        label = resolve_label(coupling)
        if label not in self._coupling_map:
            raise KeyError(
                f"No coupling labeled '{label}' on this chip. Available: {list(self._coupling_map.keys())}"
            )
        return self._coupling_map[label]

    @property
    def backend(self) -> Backend:
        """Active backend: per-call override, then chip-specific backend, then process default.

        The per-call override is the ``backend=`` argument of
        :meth:`~quchip.control.sequence.QuantumSequence.simulate` /
        ``simulate_batch``. The override has priority over a backend that the
        chip constructor set. So one chip can run, e.g., QuTiP sweeps and
        dynamiqs gradient solves without changing global state.
        """
        # `quchip.backend._backend_context` sets the scope of the per-call
        # backend override.
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
    def approximation(self) -> Approximation:
        """Approximation for the dressed analysis and the default solver assembly."""
        return self._approximation

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
        return tuple(device.resolved_dimension(self._basis) for device in self._devices)

    @property
    def authored_dims(self) -> tuple[int, ...]:
        """Per-device dimensions of the exact authored local spaces."""
        return tuple(device.local_space().dimension for device in self._devices)

    @property
    def basis(self) -> Literal["native", "eigen"]:
        """Chip-wide policy for the local solver basis that devices inherit."""
        return self._basis

    def resolve_basis(self, device: str | BaseDevice) -> Literal["native", "eigen"]:
        """Resolve a device override against the chip basis policy.

        Parameters
        ----------
        device : str or BaseDevice
            Device label or object.
        """
        resolved = self[device]
        return resolved.resolved_basis(self._basis)

    @property
    def total_dim(self) -> int:
        """Total Hilbert-space dimension (product of per-device :attr:`dims`)."""
        return prod(self.dims)

    @property
    def baths(self) -> tuple[Bath, ...]:
        """Chip-level baths (shared or collective dissipation), in insertion order."""
        return self._baths

    @property
    def ports(self) -> tuple[Port, ...]:
        """Accessible Markovian input-output channels, in declaration order."""
        return () if self._port_network is None else self._port_network.ports

    @property
    def port_network(self) -> PortNetwork | None:
        """Return the attached accessible field boundary, if any."""
        return self._port_network

    def port(self, port: str | Port) -> Port:
        """Return one declared port by object or label.

        Parameters
        ----------
        port : str or Port
            Port label or object.
        """
        label = resolve_label(port)
        for candidate in self.ports:
            if candidate.label == label:
                return candidate
        raise KeyError(f"No port labeled '{label}'. Available: {[candidate.label for candidate in self.ports]}")

    def connect_network(self, network: PortNetwork) -> None:
        """Attach the chip's one complete accessible field boundary.

        Parameters
        ----------
        network : PortNetwork
            Network whose ports target this chip.
        """
        if not isinstance(network, PortNetwork):
            raise TypeError(f"Expected a PortNetwork, got {type(network).__name__}: {network!r}")
        if self._port_network is not None:
            raise ValueError(
                "A PortNetwork is already attached; call disconnect_network() before replacing it."
            )
        network.validate_for(self)
        self._port_network = network
        self._resolved_result_cache = None

    def disconnect_network(self) -> PortNetwork | None:
        """Detach and return the accessible field boundary."""
        network = self._port_network
        self._port_network = None
        self._resolved_result_cache = None
        return network

    def add_bath(self, bath: Bath) -> Bath:
        """Attach a bath to this chip and return it (for fluent use).

        Parameters
        ----------
        bath : Bath
            Shared or collective dissipation model that targets this chip.

        Notes
        -----
        You can add baths at any time after construction. The next simulate or
        solve call collects the bath's collapse operators without a chip
        rebuild. The bath is validated immediately, so bad input fails here,
        not at solve time with an unclear error. Bad input is a non-``Bath``
        argument, a target label not on this chip, or a label that collides
        with an attached bath.
        """
        # Without the immediate bath validation, a label collision would
        # silently overwrite an entry on the `physics_notes` audit surface.
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
        if bath._retained is not None:
            size = int(np.prod(self.authored_dims))
            if any(op.shape != (size, size) for _, *operators in bath._retained.values() for op in operators):
                raise ValueError("Retained bath operators do not match this chip's authored dimensions.")
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

        Omitted device noise fields reset to ``None``. An omitted ``baths``
        clears all baths. The complete target state is validated before any
        write. Declare custom noise rates with ``parameter(noise=True)`` and
        implement them in :meth:`~quchip.devices.base.BaseDevice.dissipation`,
        so they stay sweepable, differentiable, and serializable. It prints the
        applied changes, and a repeated identical call is a silent no-op.

        Parameters
        ----------
        config : mapping, optional
            ``{device_or_label: {noise_param: value}}``. Devices can appear as
            objects or labels, once each. Non-noise parameters are rejected.
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
            resolved[label] = self._device_map[label]._normalize_parameter_names(params)

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

        changes: list[str] = []
        candidates = {}
        for label, target in targets.items():
            device = self._device_map[label]
            updates = {}
            for name, new in target.items():
                old = getattr(device, name, None)
                if _same_concrete_value(old, new):
                    continue
                updates[name] = new
                changes.append(f"{label}: {name} {old!r} → {new!r}")
            if updates:
                candidates[label] = device._parameter_candidate(updates)

        for label, candidate in candidates.items():
            self._device_map[label]._commit_parameter_candidate(candidate)

        old_ids = {id(bath) for bath in self._baths}
        new_ids = {id(bath) for bath in new_baths}
        for bath in self._baths:
            if id(bath) not in new_ids:
                changes.append(f"baths: - {bath.label} (model={bath.recipe})")
        for bath in new_baths:
            if id(bath) not in old_ids:
                extra = f", {bath.temperature} mK" if bath.temperature is not None else ""
                changes.append(f"baths: + {bath.label} (model={bath.recipe}{extra})")
        self._baths = tuple(new_baths)

        for line in changes:
            print(line)

    @property
    def crosstalks(self) -> list[Crosstalk]:
        """Convenience view of the :class:`Crosstalk` entries from the signal chain."""
        if self._control_equipment is None:
            return []
        return self._control_equipment.crosstalks

    def device_index(self, label: str | BaseDevice) -> int:
        """Return a device's tensor-product index.

        Parameters
        ----------
        label : str or BaseDevice
            Device label or object.
        """
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
        self,
        path: str = "chip_topology.html",
        *,
        full: bool = True,
        exclude: set[str] | None = None,
        values: str = "bare",
        **kwargs: Any,
    ) -> str:
        """Render the chip topology through :mod:`quchip.viz.chip`.

        Parameters
        ----------
        path : str, default="chip_topology.html"
            Output HTML path.
        full : bool, default=True
            Include the complete topology view.
        exclude : set[str] or None, default=None
            Labels omitted from the rendering.
        values : str, default="bare"
            Parameter-value view forwarded to the renderer.
        **kwargs : Any
            Additional renderer options.
        """
        from quchip.viz.chip import plot_graph

        return plot_graph(
            self,
            path,
            full=full,
            exclude=exclude,
            values=values,
            **kwargs,
        )

    def plot_energy_levels(self, *, ax: Any = None, **kwargs: Any) -> Any:
        """Render the dressed spectrum through :mod:`quchip.viz.chip`.

        Parameters
        ----------
        ax : object or None, default=None
            Matplotlib axes. ``None`` creates axes.
        **kwargs : Any
            Additional renderer options.
        """
        from quchip.viz.chip import plot_energy_levels

        return plot_energy_levels(self, ax=ax, **kwargs)

    # ------------------------------------------------------------------
    # Frame management
    # ------------------------------------------------------------------

    def set_frame(self, frame: FrameSpec) -> None:
        """Set the frame that the simulation inputs use.

        Supported values:

        - ``"lab"``: all reference frequencies are 0.0 GHz.
        - ``"rotating"``: per-device references use the dressed drive frequencies.
        - ``"auto"``: quchip plans per-device frequencies from retained
          couplings, cascade-generated network couplings, delivered drive
          tones, and scattering-scaled coherent-input tones.
        - scalar-like: a shared reference frequency for all devices.
        - ``dict``: per-device references keyed by label or device.

        Frame changes never change dressed-state data, because quchip always
        calculates the dressing from the lab-frame static Hamiltonian.

        Parameters
        ----------
        frame : {"lab", "rotating", "auto"}, scalar-like, or mapping
            Frame specification. Frequencies are in GHz.
        """
        if isinstance(frame, str):
            if frame not in ("lab", "rotating", "auto"):
                raise ValueError(f"frame string must be one of 'lab', 'rotating', or 'auto', got {frame!r}")
            self._frame_spec = frame
            return

        if _is_scalar_like(frame):
            self._frame_spec = frame
            return

        if isinstance(frame, dict):
            self._frame_spec = {resolve_label(key): value for key, value in frame.items()}
            return

        raise TypeError(
            f"frame must be 'lab', 'rotating', 'auto', a scalar-like frequency, or "
            f"dict[str|BaseDevice, scalar-like], got {type(frame).__name__}"
        )

    # ------------------------------------------------------------------
    # Dressed-state analysis — delegates to ChipAnalysis
    # ------------------------------------------------------------------

    @property
    def analysis(self) -> ChipAnalysis:
        """Dressed-state analysis namespace, i.e. the chip's :class:`ChipAnalysis`.

        Canonical entry point for the full dressed-analysis surface (for
        advanced users and less common methods). Flat ``chip.*`` forwarders to
        this namespace give the common quantities, e.g. :meth:`energy`,
        :meth:`freq`, :meth:`dress`, :meth:`dispersive_shift`, … For all other
        quantities, use ``chip.analysis``.
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
        """Diagonalize the exact lab-frame Hamiltonian.

        Parameters
        ----------
        overlap_threshold : float, default=0.5
            Minimum bare-state overlap used for labels.
        force : bool, default=False
            Recalculate even when a valid result is cached.
        labeling : {"DE"}, default="DE"
            Confidence-ordered row-greedy overlap assignment.
        """
        return self._analysis.dress(
            overlap_threshold=overlap_threshold,
            force=force,
            labeling=labeling,
        )

    def _ensure_dressed(self) -> DressedResult:
        return self._analysis._ensure_dressed()

    @property
    def is_dressed(self) -> bool:
        """True if a valid dressed-state result is cached. See :attr:`ChipAnalysis.is_dressed`."""
        return self._analysis.is_dressed

    def energy(
        self,
        device_states: Mapping[str | BaseDevice, int] | None = None,
        /,
        **device_state_kwargs: int,
    ) -> float:
        """Return the dressed eigenenergy in GHz for a bare-state label.

        Parameters
        ----------
        device_states : mapping, optional
            Device labels or objects mapped to levels.
        **device_state_kwargs : int
            Device-label keyword levels.
        """
        return self._analysis.energy(device_states, **device_state_kwargs)

    def dressed_spectrum(self) -> Any:
        """Return raw dressed eigenvalues without Python scalar coercion. See :meth:`ChipAnalysis.dressed_spectrum`."""
        return self._analysis.dressed_spectrum()

    def dressed_index(
        self,
        device_states: Mapping[str | BaseDevice, int] | None = None,
        /,
        **device_state_kwargs: int,
    ) -> int | None:
        """Return the matching dressed index, or ``None``.

        Parameters
        ----------
        device_states : mapping, optional
            Device labels or objects mapped to levels.
        **device_state_kwargs : int
            Device-label keyword levels.
        """
        return self._analysis.dressed_index(device_states, **device_state_kwargs)

    def bare_label(self, dressed_index: int) -> tuple[int, ...]:
        """Return the bare-state label assigned to a dressed index.

        Parameters
        ----------
        dressed_index : int
            Dressed eigenstate index.
        """
        return self._analysis.bare_label(dressed_index)

    def operator_in_dressed_basis(
        self,
        device: str | BaseDevice,
        op: str | Any,
        *,
        truncate: int | None = None,
    ) -> Operator:
        """Return an embedded device operator transformed to the dressed eigenbasis.

        See :meth:`ChipAnalysis.operator_in_dressed_basis` for the owner contract.

        Parameters
        ----------
        device : str or BaseDevice
            Device that owns the local operator.
        op : str or array-like
            Named or explicit local operator.
        truncate : int, optional
            Keep only the lowest dressed states.
        """
        return self._analysis.operator_in_dressed_basis(device, op, truncate=truncate)

    def drive_matrix_elements(
        self,
        transition: str | BaseDevice | tuple[Mapping[str | BaseDevice, int], Mapping[str | BaseDevice, int]],
        *,
        drives: Sequence[str | BaseDrive] | None = None,
    ) -> LabelKeyedDict:
        """Return ``<final~|D_j|initial~>`` for wired drive lines.

        Rows index the final dressed state and columns the initial dressed
        state. In the weak-drive projection, these matrix elements set the
        coefficients of the effective driven Hamiltonian. See E. Magesan and J.
        M. Gambetta, Phys. Rev. A 101, 052308 (2020), DOI
        10.1103/PhysRevA.101.052308, and
        :meth:`ChipAnalysis.drive_matrix_elements` for the owner contract.

        Parameters
        ----------
        transition : str, BaseDevice, or pair of mappings
            Transition shorthand or explicit lower/upper bare labels.
        drives : sequence, optional
            Drive labels or objects to include. The default is all wired drives.

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
        """Return leading bare-basis probabilities for a dressed eigenstate.

        Parameters
        ----------
        state : int or mapping, optional
            Dressed index or bare-state label.
        n_components : int, default=5
            Number of largest components to return.
        **device_state_kwargs : int
            Device-label keyword levels.
        """
        return self._analysis.state_components(
            state,
            n_components=n_components,
            **device_state_kwargs,
        )

    def dispersive_shift(self, device_a: str | BaseDevice, device_b: str | BaseDevice) -> float:
        """Return the dressed cross-Kerr shift (GHz): ``E(1,1) − E(1,0) − E(0,1) + E(0,0)``.

        Parameters
        ----------
        device_a, device_b : str or BaseDevice
            Devices whose dressed cross-Kerr shift is requested.
        """
        return self._analysis.dispersive_shift(device_a, device_b)

    # ``static_zz`` is the same physics under a different name: the static ZZ
    # interaction strength equals the dressed dispersive (cross-Kerr) shift.
    static_zz = dispersive_shift
    zz = dispersive_shift

    def kerr_matrix(self) -> KerrMatrix:
        """Return the labeled dressed self-Kerr and cross-Kerr matrix in GHz.

        See :meth:`ChipAnalysis.kerr_matrix`.
        """
        return self._analysis.kerr_matrix()

    def dressed_anharmonicity(self, device: str | BaseDevice) -> float:
        """Return the dressed anharmonicity of one device with the other devices grounded (GHz).

        See :meth:`ChipAnalysis.dressed_anharmonicity`.

        Parameters
        ----------
        device : str or BaseDevice
            Device label or object.
        """
        return self._analysis.dressed_anharmonicity(device)

    def transition_frequency(
        self,
        target: str | BaseDevice,
        lower: int,
        upper: int,
        when: dict[str | BaseDevice, int] | None = None,
    ) -> Any:
        """Return a dressed transition in GHz with optional spectator levels.

        Parameters
        ----------
        target : str or BaseDevice
            Device whose transition is requested.
        lower, upper : int
            Lower and upper local levels.
        when : mapping, optional
            Spectator device levels.

        Unspecified spectators are in their ground state. ``target`` and
        entries in ``when`` accept either device objects or labels.
        """
        return self._analysis.transition_frequency(
            target,
            lower,
            upper,
            when=when,
        )

    def effective_subspace_hamiltonian(
        self,
        states: (
            list[Mapping[str | BaseDevice, int] | tuple[int, ...]]
            | tuple[Mapping[str | BaseDevice, int] | tuple[int, ...], ...]
        ),
    ) -> Any:
        """Return the dressed effective Hamiltonian in a labeled bare subspace.

        See :meth:`ChipAnalysis.effective_subspace_hamiltonian`.

        Parameters
        ----------
        states : sequence of mappings or level tuples
            Bare labels spanning the requested subspace.
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
        """Return the dressed 0→1 frequencies in GHz, optionally conditioned on spectators.

        Parameters
        ----------
        target : str or BaseDevice, optional
            Device label or object. If you omit it, the method returns all frequencies.
        when : mapping, optional
            Spectator device levels.

        Overloaded: no ``target`` returns the full ``{label: freq}`` dict;
        a single ``target`` (label or device) returns one scalar 0→1
        frequency. Under ``jax.jit`` the scalar is a traced 0-d array; the
        overload is type-only and preserves traceability.
        """
        return self._analysis.freq(target, when=when)

    def frame_info(self) -> dict[str, Any]:
        """Return the per-device frame reference frequency ``ω_ref,i`` (GHz). See :meth:`ChipAnalysis.frame_info`."""
        return self._analysis.frame_info()

    def physics_notes(self) -> dict[str, list[str]]:
        """Collect :meth:`physics_notes` from every component.

        Returns a dict with the key ``"chip"`` for the chip-level entry. Each
        component has the key ``"<kind>:<label>"``, where ``kind`` is one of
        ``"device"``, ``"coupling"``, ``"drive"``, ``"bath"``, ``"port"``,
        ``"network"``, ``"effective"``. Each key maps to that component's
        declared approximations: Hilbert truncation, model regime, RWA status,
        noise-channel selection, and any other non-obvious assumption it
        explicitly declares.

        Keys are kind-qualified, not bare labels, because the label namespaces
        are *not* globally disjoint. A device, coupling, drive, and bath can
        share a label, and on this audit surface one component's entry must not
        silently overwrite another's. Drives are enumerated from the
        :attr:`control_equipment` wiring, not from per-device
        ``connected_drives``, so an edge-target
        :class:`~quchip.control.drive.ParametricDrive` (that pumps a coupling,
        not a device) is included.

        The chip-level ``"chip"`` entry states the local-Lindblad
        approximation, under which this chip builds every collapse operator
        (see :meth:`collapse_contributions`). The entry is present even without
        baths, because it always applies. The dict is for inspection and audit,
        not for runtime dispatch.
        """
        notes: dict[str, list[str]] = {
            "chip": [
                "Collapse operators are authored per component, transformed into the fixed "
                "local solver bases, and combined with the interacting Hamiltonian — the "
                "local-Lindblad approximation, not a global dressed-basis master equation."
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
        for port in self.ports:
            notes[f"port:{port.label}"] = list(port.physics_notes())
        if self.port_network is not None:
            notes[f"network:{self.port_network.label}"] = list(self.port_network.physics_notes())
        for terms in self.effective_terms:
            notes[f"effective:{terms.label}"] = terms.physics_notes()
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
        """Reconstruct a chip from :meth:`to_dict` output.

        Parameters
        ----------
        d : dict[str, Any]
            Serialized chip payload.
        """
        from quchip.chip.serialization import deserialize_chip

        return deserialize_chip(d)

    def clone(self) -> "Chip":
        """Return an isolated structural clone for sweep evaluation."""
        from quchip.chip.serialization import clone_chip

        return clone_chip(self)

    def _parameter_targets(self) -> dict[str, tuple[str, int, str, Any]]:
        """Map public parameter paths to component-owned local values."""
        targets: dict[str, tuple[str, int, str, Any]] = {}

        def add(path: str, target: tuple[str, int, str, Any]) -> None:
            if path in targets:
                raise ValueError(
                    f"Parameter path {path!r} is ambiguous; give the colliding components distinct labels."
                )
            targets[path] = target

        for index, device in enumerate(self._devices):
            for name, value in device.parameter_values().items():
                add(f"{device.label}.{name}", ("device", index, name, value))
        for index, coupling in enumerate(self._couplings):
            for name, value in coupling.parameter_values().items():
                add(f"{coupling.label}.{name}", ("coupling", index, name, value))
        if self._control_equipment is not None:
            for index, line in enumerate(self._control_equipment.lines):
                for name, value in line.parameter_values().items():
                    add(f"drive.{line.label}.{name}", ("drive", index, name, value))
            for index, transform in enumerate(self._control_equipment.signal_chain):
                for name, value in transform.parameter_values().items():
                    add(f"control.{index}.{name}", ("control", index, name, value))
        for index, bath in enumerate(self._baths):
            for name, value in bath.parameter_values().items():
                add(f"bath.{bath.label}.{name}", ("bath", index, name, value))
        for index, port in enumerate(self.ports):
            for name, value in port.parameter_values().items():
                add(f"port.{port.label}.{name}", ("port", index, name, value))
        if self._port_network is not None:
            for name, value in self._port_network.parameters.items():
                add(f"network.{name}", ("network", 0, name, value))
        return targets

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Bindable numerical values keyed by stable component paths."""
        return MappingProxyType({path: target[3] for path, target in self._parameter_targets().items()})

    @property
    def settings(self) -> Mapping[str, Any]:
        """Read-only structural choices that are not numerical fit parameters."""
        equipment = self._control_equipment
        lines = () if equipment is None else equipment.lines
        signal_chain = () if equipment is None else equipment.signal_chain
        return MappingProxyType(
            {
                "devices": tuple(
                    (device.label, type(device).__name__, device.resolved_dimension(self._basis))
                    for device in self._devices
                ),
                "basis": self._basis,
                "authored_dims": self.authored_dims,
                "device_bases": tuple(
                    (device.label, self.resolve_basis(device)) for device in self._devices
                ),
                "couplings": tuple(
                    (
                        coupling.label,
                        type(coupling).__name__,
                        coupling.device_a_label,
                        coupling.device_b_label,
                    )
                    for coupling in self._couplings
                ),
                "drives": tuple(
                    (line.label, type(line).__name__, line.target_label)
                    for line in lines
                ),
                "signal_chain": tuple(
                    type(transform).__name__
                    for transform in signal_chain
                ),
                "baths": tuple(
                    (bath.label, bath.recipe, tuple(bath.resolve_targets(self)))
                    for bath in self._baths
                ),
                "ports": tuple(
                    (port.label, tuple(port.resolve_targets(self)))
                    for port in self.ports
                ),
                "port_network": (
                    None
                    if self._port_network is None
                    else (
                        self._port_network.label,
                        tuple(component.label for component in self._port_network.components),
                        tuple(exposure.label for exposure in self._port_network.external_ports),
                    )
                ),
                "frame": self._frame_spec,
                "approximation": type(self._approximation).__name__,
                "backend": type(self.backend).__name__,
            }
        )

    def with_params(self, bindings: Mapping[str, Any]) -> "Chip":
        """Return an isolated structural copy with values rebound.

        Parameters
        ----------
        bindings : mapping[str, Any]
            Fully qualified parameter paths and replacement values.
        """
        from quchip.utils.values import copy_value

        bindings = dict(bindings)
        for device in self._devices:
            prefix = f"{device.label}."
            legacy = prefix + "thermal_population"
            if legacy in bindings:
                canonical = prefix + "thermal_occupation"
                local = {"thermal_population": bindings.pop(legacy)}
                if canonical in bindings:
                    local["thermal_occupation"] = bindings[canonical]
                bindings[canonical] = device._normalize_parameter_names(local)["thermal_occupation"]
        targets = self._parameter_targets()
        unknown = set(bindings) - set(targets)
        if unknown:
            raise KeyError(
                f"Unknown Chip parameter paths: {sorted(unknown)}. "
                f"Available: {list(targets)}"
            )

        cloned = self.clone()
        device_bindings: dict[int, dict[str, Any]] = {}
        changed_transforms: dict[int, SignalTransform] = {}
        for path, value in copy_value(dict(bindings)).items():
            kind, index, name, _ = targets[path]
            if kind == "device":
                device_bindings.setdefault(index, {})[name] = value
            elif kind == "coupling":
                cloned._couplings[index].set_parameter_value(name, value)
            elif kind == "drive":
                assert cloned._control_equipment is not None
                cloned._control_equipment._lines[index].set_parameter_value(name, value)
            elif kind == "control":
                assert cloned._control_equipment is not None
                transform = cloned._control_equipment._signal_chain[index]
                setattr(transform, name, value)
                changed_transforms[index] = transform
            elif kind == "bath":
                cloned._baths[index].set_parameter_value(name, value)
            elif kind == "network":
                assert cloned._port_network is not None
                cloned._port_network.set_parameter_value(name, value)
            else:
                cloned.ports[index].set_parameter_value(name, value)
        for index, local_bindings in device_bindings.items():
            cloned._devices[index].set_parameter_values(local_bindings)
        for transform in changed_transforms.values():
            transform.validate()
        return cloned

    def partition(self) -> "PartitionResult":
        """Split the chip into independent sub-chips along the independence graph.

        Connectivity comes from multi-device operator support in the resolved
        Hamiltonian and Lindblad channels, including Hamiltonian terms
        generated by SLH composition. Passive field scattering alone does not
        connect subsystems. Drive-crosstalk pairs stay together, so component
        solves keep their signal transform.
        """
        from quchip.chip.partition import partition_chip

        return partition_chip(self)

    def status(self) -> None:
        """Print a short diagnostic dashboard for the chip."""
        label = self.label if self.label is not None else "(unlabeled)"
        print(f"Chip: {label}")
        print(f"- devices: {len(self._devices)}")
        print(f"- couplings: {len(self._couplings)}")
        print(f"- frame: {self._frame_spec!r}")
        print(f"- approximation: {type(self._approximation).__name__}")
        print(f"- dressed: {'yes' if self.is_dressed else 'no'}")
        print("- device list:")
        for dev, dim in zip(self._devices, self.dims):
            bare_freq = getattr(dev, "freq", None)
            connected = sorted(
                line.label for line in self.control_equipment.lines
                if not isinstance(line, CouplingDrive) and line.target_label == dev.label
            ) if self.control_equipment is not None else []
            line_text = ", ".join(connected) if connected else "none"
            print(
                f"  - {dev.label}: {type(dev).__name__} "
                f"(freq={_format_float(bare_freq)} GHz, "
                f"dressed={_format_float(self.freq(dev))} GHz, "
                f"levels={dim}, lines={line_text})"
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

        With *device*, the array is a local operator on that device's subspace
        and is embedded into the full tensor-product space. With
        ``device=None`` the array must already span the full chip Hilbert
        space.

        Parameters
        ----------
        data : array-like
            Local or full-space operator array.
        device : str, BaseDevice, or None, default=None
            Owner of the local operator. ``None`` treats ``data`` as full-space.
        """
        from quchip.chip.observables import from_array

        return from_array(self, data, device)

    def observable(self, device: str | BaseDevice, op: str | Any) -> Any:
        """Embed a device operator onto the full chip Hilbert space.

        Accepts a prebuilt local-space operator or one of the names ``"X"``,
        ``"Y"``, ``"Z"``, ``"n"``, ``"a"``, ``"a_dag"``, and ``"I"``. Returns
        the operator embedded on the chip's tensor-product space.

        Use it for manual construction and analysis of full-space operators,
        not as a solver ``e_op``. For solver expectation values, use
        :meth:`e_ops`, which keeps operators *local* so the demodulation
        pipeline can band-decompose and embed them correctly.

        Parameters
        ----------
        device : str or BaseDevice
            Device whose local space owns the operator.
        op : str or array-like
            Named or explicit local operator.
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

        Each keyword maps a device label to an operator specification: a name
        string, a list of names, a raw local-space operator, or a mixed list of
        strings and operators. Specify two-device correlators (for example
        ``⟨Z₁⊗Z₂⟩``) with *correlators* as device-label pairs → operator pairs.
        Returns local-space operators (not embedded), which the demodulation
        pipeline embeds as needed.

        Parameters
        ----------
        correlators : dict or None, default=None
            Device-pair keys mapped to pairs of local operator specifications.
        **specs : str, list, or array-like
            Device labels mapped to local operator specifications.

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

        After this call, :meth:`bare_state`, :meth:`state`, and
        :meth:`superposition` accept single-string specifications. Each
        character gives one device's level, in *devices* order. The default
        level symbols are ``g=0, e=1, f=2, h=3``. The methods always accept
        digits ``0..9`` as energy-level indices.

        Name every chip device exactly once.

        Parameters
        ----------
        *devices : str or BaseDevice
            All chip devices in shorthand order.
        levels : mapping[str, int] or None, default=None
            Extra one-character level symbols and their indices.

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
        """Return a normalized bare-basis superposition of tensor-product states.

        Each component is a bare-state spec or an ``(amplitude, spec)`` tuple
        for weighted mixing. A bare-state spec is a dict keyed by device or
        label, or a string if you called :meth:`set_state_order` before.
        Weights default to uniform, and results are normalized to unit norm.

        Unlike :meth:`state`, this method stays in the bare product basis and
        does no dressed diagonalization, so the probe basis is explicit.

        Parameters
        ----------
        *components : mapping, str, or tuple
            Bare-state specifications, optionally paired with amplitudes.

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
        """Return the dressed eigenstate assigned from the given product-state level labels.

        Accepts a string shorthand (for example ``"eg1"``) if you called
        :meth:`set_state_order` before.

        If the requested label's assignment overlap is low, this method warns
        and names :meth:`bare_state` as the product-state alternative.

        This method is safe in ``jax.jit``/``grad``/``vmap``. Under tracing,
        the :func:`~quchip.chip.dressing.label_eigensystem` array kernel
        selects the assigned eigenvector column, so dressed initial states are
        differentiable end-to-end. The global phase is gauge-dependent
        (``eigh`` column convention), which does not change populations or
        ``|overlap|`` figures of merit.

        Parameters
        ----------
        device_states : mapping, str, or None, default=None
            Product-state level labels used for dressed assignment.
        **device_state_kwargs : int
            Per-device energy levels keyed by label.
        """
        return self._analysis.state(device_states, **device_state_kwargs)

    def bare_state(
        self,
        device_states: Mapping[str | BaseDevice, int | State] | str | None = None,
        /,
        **device_state_kwargs: int | State,
    ) -> State:
        """Return a product state from per-device energy levels or authored local kets.

        Specify each device as an energy-level index (``int``) or as a ket vector in that
        device's authored local space. Unspecified devices use the ground state (level
        0). Unlike :meth:`state`, this method does **not** diagonalize the coupled
        system.

        Accepts a string shorthand (for example ``"eg1"``) if you called
        :meth:`set_state_order` before.

        Parameters
        ----------
        device_states : mapping, str, or None, default=None
            Per-device levels or local kets. Omitted devices use level zero.
        **device_state_kwargs : int or State
            Per-device levels or local kets keyed by label.
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

        Preferred user-facing API to attach control to a chip. If you omit
        *lines*, the method reuses the connected lines and replaces only the
        signal chain.

        Parameters
        ----------
        *lines : BaseDrive
            Classical control lines to attach.
        signal_chain : sequence of SignalTransform or None, default=None
            Ordered classical signal transforms.

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
        """Remove one control line and every signal-chain transform that refers to it.

        The inverse of :meth:`wire` for a single line. Accepts the drive object
        or its label. Returns the removed drive, so you can rewire it later.
        Removing the last line fully detaches the equipment
        (``control_equipment`` becomes ``None``).

        Parameters
        ----------
        line : BaseDrive or str
            Drive object or label to remove.
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

        Validates every drive target, rejects duplicate drive labels, and
        reconnects each drive to this chip's canonical device or coupling
        instance. User-facing code should use :meth:`wire`.

        Parameters
        ----------
        control_equipment : ControlEquipment
            Complete classical control wiring to attach.
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
            if isinstance(drive, CouplingDrive):
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
            target_label = drive.target_label
            assert target_label is not None
            if target_label not in self._device_map:
                raise ValueError(
                    f"Drive target '{target_label}' is not on this chip. Available: {list(self._device_map.keys())}"
                )

        self._control_equipment = control_equipment

        for drive in control_equipment.lines:
            if isinstance(drive, CouplingDrive):
                assert drive.target_label is not None
                coupling = self._coupling_map[drive.target_label]
                drive.connect(coupling)
                continue
            assert drive.target_label is not None
            drive.connect(self._device_map[drive.target_label])

    def disconnect(self) -> ControlEquipment:
        """Fully detach the control equipment (low-level API).

        The inverse of :meth:`connect`/:meth:`wire`. Removes all lines and the
        signal chain at once (``control_equipment`` becomes ``None``). Returns
        the detached equipment, so you can reconnect it later.

        Returns
        -------
        ControlEquipment
            The previously attached equipment.
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
        if not hasattr(problem, "engine_result") or not hasattr(problem, "chip"):
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

    def solve(self, problem: "SolveProblem") -> "SimulationResult":
        """Solve a typed :class:`SolveProblem` through this chip's backend.

        Call ``result.check_truncation()`` to inspect saved boundary samples.

        Parameters
        ----------
        problem : SolveProblem
            Prepared problem built for this chip.
        """
        from quchip.engine import solve_problem

        self._check_problem(problem)
        return solve_problem(problem)

    def solve_many(
        self, batch_or_problems: Any, *, progress: bool = True,
    ) -> "SimulationBatchResult":
        """Solve a :class:`SolveBatch` or list of problems.

        Chip-level validation checks only what uses ``self`` (every input was
        built for *this* chip).

        Parameters
        ----------
        batch_or_problems : SolveBatch or sequence of SolveProblem
            Batch or prepared problems built for this chip.
        progress : bool, default=True
            Show backend progress where supported.
        """
        # `quchip.engine.solve_many` owns the single SolveBatch / list
        # input-shape dispatch and the batching.
        from quchip.engine import solve_many
        from quchip.engine.ir import SolveBatch

        if isinstance(batch_or_problems, SolveBatch):
            if batch_or_problems.chip is not self:
                raise ValueError("SolveBatch was built for a different chip.")
        else:
            batch_or_problems = list(batch_or_problems)
            for i, problem in enumerate(batch_or_problems):
                self._check_problem(problem, index=i)

        return solve_many(batch_or_problems, progress=progress)

    def steadystate(
        self,
        *,
        e_ops: dict | None = None,
        options: dict | None = None,
        frame: FrameSpec | None = None,
        approximation: Approximation | None = None,
    ) -> "SteadyStateResult":
        """Solve this chip's unique static Lindblad steady state.

        Parameters
        ----------
        e_ops : dict or None, default=None
            Static expectation operators.
        options : dict or None, default=None
            Backend solver options.
        frame : FrameSpec or None, default=None
            Solve frame, or ``None`` to use the chip default.
        approximation : Approximation or None, default=None
            Solve approximation, or ``None`` to use the chip default.
        """
        from quchip.engine.steady_state import steadystate

        return steadystate(
            self,
            e_ops=e_ops,
            options=options,
            frame=frame,
            approximation=approximation,
        )

    def steadystate_batch(
        self,
        *axes: Any,
        e_ops: dict | None = None,
        options: dict | None = None,
        frame: FrameSpec | None = None,
        approximation: Approximation | None = None,
        progress: bool = True,
    ) -> Any:
        """Solve static Lindblad steady states over parameter sweep axes.

        Parameters
        ----------
        *axes : Any
            Parameter sweep axes accepted by the batch builder.
        e_ops : dict or None, default=None
            Static expectation operators.
        options : dict or None, default=None
            Backend solver options.
        frame : FrameSpec or None, default=None
            Solve frame, or ``None`` to use the chip default.
        approximation : Approximation or None, default=None
            Solve approximation, or ``None`` to use the chip default.
        progress : bool, default=True
            Show backend progress where supported.
        """
        from quchip.engine.steady_state import steadystate_batch

        return steadystate_batch(
            self,
            *axes,
            e_ops=e_ops,
            options=options,
            frame=frame,
            approximation=approximation,
            progress=progress,
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Return a plain-text report of everything on the chip, in sections.

        The report shows devices with their declared parameters (units
        included), noise settings, couplings, control wiring, and baths: the
        "what did I just build?" view. Returns a string, so use
        ``print(chip.describe())``. Traced parameters show as ``<traced>`` and
        are never concretized.
        """
        from quchip.chip.describe import describe_chip

        return describe_chip(self)

    def __repr__(self) -> str:
        return (
            f"Chip(label={self.label!r}, devices={len(self._devices)}, "
            f"couplings={len(self._couplings)}, frame={self._frame_spec!r}, "
            f"approximation={type(self._approximation).__name__}, "
            f"dressed={'yes' if self.is_dressed else 'no'})"
        )
