"""Stage 2: assemble a :class:`HamiltonianDescription` from chip, drive ops, and frame.

Responsibilities
----------------
This module owns the **2π boundary** of the engine: inputs are ordinary
GHz (ν), outputs are operators scaled by
ω = 2π·ν so backends can solve Schrödinger's equation with ``d|ψ⟩/dt =
-i H |ψ⟩`` in ns/rad units. The operator angular-scaling boundary lives
entirely here, in:

* :func:`_build_static_h0` — frame-subtracted bare Hamiltonian,
* :func:`_collect_coupling_terms` — full interaction band-decomposed,
  each band filtered by the coupling's RWA policy and folded into ``H₀``
  or carried, per band,
* :func:`_apply_2pi_canonical` — the single point that scales every
  embedded *dynamic* operator (drive, crosstalk, coupling-dynamic,
  device-dynamic).

The same ``2π`` convention also expresses signal-AST carrier and
rotating-frame-phase *frequencies* in rad/ns
(:func:`_single_tone_coefficient`, :func:`_direct_real_coefficient`) and
the stage-3 demodulation phase; those are frequencies inside the
time-dependence / observable bookkeeping, not a second Hamiltonian
boundary. :mod:`quchip.engine.solver_hints` divides by ``2π`` only to
report advisory hints back in ordinary GHz.

Physics
-------
Stage 2 performs three physically distinct operations on top of the 2π
scaling:

1. **Rotating-frame transformation.** Each device's number operator is
   shifted by its frame reference ω_ref so that the static Hamiltonian
   becomes ``H₀ − Σᵢ ω_ref,ᵢ nᵢ`` (see any standard cQED reference,
   e.g. Scully & Zubairy, *Quantum Optics*, CUP 1997, §5.1).

2. **Band decomposition / rotating-wave approximation (RWA).** Coupling
   and drive operators are split into excitation-change bands of weight
   ``w = col − row`` and attached to carriers ``exp(−i w·ω t)``. When
   ``rwa=True``, counter-rotating coupling bands are dropped structurally
   by the coupling's ``rwa_keeps_band`` predicate and counter-rotating
   drive bands by the modulation policies (Jaynes & Cummings,
   *Proc. IEEE* **51**, 89 (1963); Walls & Milburn, *Quantum Optics*,
   Springer 2008, §10.3; for dispersive/structured cases see
   Gambetta et al., *PRA* **74**, 042318 (2006), and the
   cross-resonance treatment in Magesan & Gambetta, *PRA* **101**,
   052308 (2020)).

3. **Signal-program construction.** Time dependence is emitted as a
   :class:`~quchip.engine.ir.SignalProgram` AST — a pure, JAX-traceable
   description that backends lower into their native coefficient form.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from quchip.backend import _backend_context
from quchip.backend.protocol import Backend, Operator
from quchip.control.drive import BaseDrive
from quchip.control.signal_spec import DriveSignalSpec, DriveModulation
from quchip.engine.ir import HamiltonianTemplate
from quchip.engine.ir import (
    CanonicalOperator,
    Carrier,
    Conjugate,
    DroppedTerm,
    DynamicTerm,
    EnvelopeRef,
    HamiltonianDescription,
    Multiply,
    RealPart,
    Scale,
    ScalarModulation,
    Shift,
    SignalProgram,
    StaticTerm,
    TermOrigin,
    Window,
)
from quchip.engine.ir import simplify_signal as _simplify_signal
from quchip.engine.solver_hints import _solver_hint_metadata, _static_diagonal_span
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import array_namespace, maybe_concrete_scalar
from quchip.engine.bands import (
    decompose_two_body_canonical_bands,
    embed_on_support,
    embed_single_mode_bands,
    prune_zero_diagonals,
)

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.devices.base import BaseDevice
    from quchip.engine.ir import ResolvedFrame, DriveOp


# -- Drive signal IR construction ---------------------------------------
#
# Drives produce frame-agnostic `DriveSignalSpec` objects (ordinary GHz,
# no IR nodes). Stage 2 is the single place where the spec is composed
# with the resolved frame and the modulation dispatch into the IR-level
# `SignalProgram` AST. The engine owns the IR and dispatches generically,
# with no per-drive-class branches.


def _phase_factor(angle: Any) -> Any:
    """Return ``exp(i * angle)`` in the array namespace of *angle*.

    Kept traceable via :func:`~quchip.utils.jax_utils.array_namespace`,
    which returns ``jax.numpy`` for a traced angle (so the gradient
    survives) and NumPy for a concrete one.
    """
    xp = array_namespace(angle)
    return xp.exp(1j * angle)


@dataclass(frozen=True)
class BandContext:
    """Per-Fourier-band context for the stage-2 modulation dispatch.

    Parameters
    ----------
    weight : int
        Fourier band index (e.g. ``-1``, ``0``, ``+1`` for a single-mode
        device; arbitrary integer for higher bands).
    device_frame_freq : float
        Device-frame oscillation frequency for this band, GHz.
    drive_freq : float | None
        Carrier frequency for microwave drives, GHz; ``None`` for
        baseband (flux-like) drives.
    rwa : bool
        Whether the engine is assembling in the rotating-wave
        approximation.
    """

    weight: int
    device_frame_freq: Any
    drive_freq: Any | None
    rwa: bool


def _spec_to_raw_signal(spec: DriveSignalSpec) -> SignalProgram:
    """Build the raw line signal ``Scale(Shift(Window(env, 0, duration), start), exp(i·phi))``.

    This is the frame-agnostic line signal before any signal-chain
    transforms (delays, gains, crosstalk) and before carrier/RWA
    modulation. It contains no frame information — stage 1's frame is
    applied later via :func:`_coefficient_from_modulation`.
    """
    windowed: SignalProgram = Window(
        child=EnvelopeRef(spec.envelope), start=0.0, stop=spec.duration
    )
    shifted: SignalProgram = Shift(child=windowed, delta_t=spec.start_time)
    return Scale(shifted, factor=_phase_factor(spec.phase_offset))


def _single_tone_coefficient(signal: SignalProgram, band: BandContext) -> SignalProgram:
    """Microwave IQ-style modulation coefficient for one Fourier band.

    Lab frame
        Mix ``signal`` with the carrier at the drive frequency, project
        to the real field, then attach the band's rotating-frame phase
        ``exp(−i w ω_ref t)`` (Krantz et al. 2019, Eq. 89).

    RWA
        Decompose into co- and counter-rotating components per band
        weight. This is the standard drive-frame decomposition used for
        charge and phase drives on transmons (Jaynes & Cummings 1963;
        Scully & Zubairy, *Quantum Optics*, §5). The weight-0 band has no
        frame rotation (``weight · device_frame_freq = 0``) to cancel
        either single-tone sideband, so both remain fast and it is
        dropped structurally before compilation reaches this function
        (:func:`_compile_drive_terms`); genuine baseband diagonal
        modulation belongs on a ``DIRECT_REAL`` channel instead.
    """
    if band.drive_freq is None:
        raise ValueError("drive_freq is required for SINGLE_TONE drive channels")
    phase: SignalProgram = Carrier(freq=TWO_PI * band.weight * band.device_frame_freq, sign=-1)

    if band.rwa:
        if band.weight > 0:
            return Scale(
                Multiply(
                    (Conjugate(signal), Carrier(freq=TWO_PI * band.drive_freq, sign=1), phase)
                ),
                factor=0.5,
            )
        if band.weight < 0:
            return Scale(
                Multiply(
                    (signal, Carrier(freq=TWO_PI * band.drive_freq, sign=-1), phase)
                ),
                factor=0.5,
            )
        raise ValueError(
            "SINGLE_TONE weight-0 bands under RWA are dropped structurally at compile time "
            "(_compile_drive_terms); reaching this branch means that drop was bypassed."
        )

    field = RealPart(Multiply((signal, Carrier(freq=TWO_PI * band.drive_freq, sign=-1))))
    return Multiply((field, phase))


def _is_dropped_weight_zero_single_tone(modulation: DriveModulation, weight: int, rwa: bool) -> bool:
    """True when a SINGLE_TONE band at weight 0 is structurally dropped under RWA.

    Applies uniformly regardless of the numeric drive frequency: the rule
    keys only on modulation kind, RWA policy, and band weight, so it never
    forces concretization of a possibly-traced ``drive_freq``.
    """
    return rwa and modulation is DriveModulation.SINGLE_TONE and weight == 0


def _weight_zero_dropped_term(*, source: str, device_label: str, drive_freq: Any) -> DroppedTerm:
    """Advisory record for a SINGLE_TONE weight-0 band dropped structurally under RWA.

    Raises
    ------
    ValueError
        *drive_freq* is ``None``. SINGLE_TONE channels require a carrier
        frequency (:func:`_single_tone_coefficient` enforces this for
        every other band); a weight-0 band reaching here with no
        ``drive_freq`` would mean that guard was bypassed for a future
        diagonal-only SINGLE_TONE extension, so it must not silently
        disappear into an audit record with no frequency.
    """
    if drive_freq is None:
        raise ValueError(
            f"Weight-0 SINGLE_TONE band on '{device_label}' (drive '{source}') has no drive_freq; "
            "SINGLE_TONE channels require a carrier frequency."
        )
    return DroppedTerm(
        source=source,
        operator=f"drive band w=+0 on {device_label}",
        reason="no frame rotation to cancel either single-tone sideband under RWA",
        band_weights=(0,),
        frequency=abs(drive_freq),
    )


def _direct_real_coefficient(signal: SignalProgram, band: BandContext) -> SignalProgram:
    """Real-valued baseband modulation (no carrier, no RWA), e.g. flux drive.

    ``BandContext.drive_freq`` and ``BandContext.rwa`` are ignored — a
    flux line couples via the real part of its signal times the usual
    device-frame band phase (Krantz et al. 2019, Sec. V on flux
    tunability).
    """
    phase: SignalProgram = Carrier(freq=TWO_PI * band.weight * band.device_frame_freq, sign=-1)
    return Multiply((RealPart(signal), phase))


def _edge_pump_coefficient(signal: SignalProgram, band: BandContext) -> SignalProgram:
    """Edge-pump modulation: real pump δ(t) times the band's frame carrier.

    ``band.drive_freq is None`` selects the *structural* baseband form
    ``δ(t) = Re s(t)``; with a carrier the pump is
    ``δ(t) = Re[s(t)·e^{-i·2π·ν_d·t}]`` with both sidebands kept —
    ``band.rwa`` is ignored because the coupling's parametric hook already
    chose the RWA-retained operator structure, and the tone itself is never
    split. For an edge band ``(Δa, Δb)`` the frame carrier frequency
    ``Δa·ω_a + Δb·ω_b`` is stored in ``band.device_frame_freq`` (weight 1);
    it may be zero or traced — no special case.
    """
    frame_phase: SignalProgram = Carrier(freq=TWO_PI * band.weight * band.device_frame_freq, sign=-1)
    if band.drive_freq is not None:
        field: SignalProgram = RealPart(Multiply((signal, Carrier(freq=TWO_PI * band.drive_freq, sign=-1))))
    else:
        field = RealPart(signal)
    return Multiply((field, frame_phase))


# Generic dispatch: one table, no `isinstance(drive, …)` branches.
# To add a new modulation kind, add the tag to :class:`DriveModulation` and
# register its IR builder here.
_MODULATION_DISPATCH: dict[DriveModulation, Any] = {
    DriveModulation.SINGLE_TONE: _single_tone_coefficient,
    DriveModulation.DIRECT_REAL: _direct_real_coefficient,
    DriveModulation.EDGE_PUMP: _edge_pump_coefficient,
}


def _coefficient_from_modulation(
    signal: SignalProgram,
    modulation: DriveModulation,
    band: BandContext,
) -> SignalProgram:
    """Compose a raw line signal with the band/frame via the modulation dispatch."""
    try:
        builder = _MODULATION_DISPATCH[modulation]
    except KeyError as exc:  # pragma: no cover - guarded by enum membership
        raise ValueError(f"Unknown drive modulation: {modulation!r}") from exc
    return builder(signal, band)

# -- Static Hamiltonian --------------------------------------------------

def _build_static_h0(chip: "Chip", resolved_frame: "ResolvedFrame", backend: Backend) -> Operator:
    """Build the frame-subtracted static Hamiltonian in angular units.

    .. math::
        H_0 \\;=\\; 2\\pi \\left( H_{\\text{bare}}
                 - \\sum_i \\omega^{\\text{ref}}_i \\, n_i \\right)

    where ``H_bare`` is the chip-level tensored sum of device and static
    coupling contributions (ordinary GHz), and ``ω^ref_i`` is the
    per-device frame reference from stage 1. This function is one of the
    four places in the engine where the 2π boundary is crossed.
    """
    h0 = TWO_PI * chip.hamiltonian()
    dims = chip.dims
    for idx, dev in enumerate(chip.devices):
        omega_ref = resolved_frame.frequencies.get(dev.label, 0.0)
        with _backend_context(backend):
            n_emb = backend.embed(dev.number_operator(), idx, dims)
        h0 = h0 - TWO_PI * omega_ref * n_emb
    return h0


# -- Dynamic-term assembly helpers ---------------------------------------
#
# Every dynamic operator that enters the solver-facing Hamiltonian passes
# through ``_apply_2pi_canonical``: it is the *single* place the ``2π``
# (angular-frequency) boundary is crossed for time-dependent terms.
# ``bands.py`` never sees ``2π``; lab-frame
# operators arrive here in ordinary GHz and leave canonicalized in angular
# units, tagged for diagnostics.


def _apply_2pi_canonical(backend: Backend, embedded: Operator, *, dims, labels, tag: str) -> CanonicalOperator:
    """Apply the ``2π`` boundary to an embedded lab-frame operator and canonicalize."""
    return backend.to_canonical_operator(TWO_PI * embedded).with_metadata(
        dims=dims,
        subsystem_labels=labels,
        tag=tag,
    )


def _dynamic_term(
    backend: Backend,
    embedded: Operator,
    *,
    dims,
    labels,
    tag: str,
    origin: TermOrigin,
    time_dependence: ScalarModulation,
) -> DynamicTerm:
    """Wrap an embedded lab-frame operator as a ``2π``-scaled :class:`DynamicTerm`."""
    return DynamicTerm(
        operator=_apply_2pi_canonical(backend, embedded, dims=dims, labels=labels, tag=tag),
        time_dependence=time_dependence,
        origin=origin,
    )


def _modulated_dynamic_term(
    operator: CanonicalOperator,
    signal: SignalProgram,
    modulation: "DriveModulation",
    *,
    weight: int,
    device_frame_freq: float,
    drive_freq: float | None,
    rwa: bool,
    origin: TermOrigin,
    tag: str | None = None,
) -> DynamicTerm:
    """Build the per-band :class:`BandContext`, attach the rotating-frame carrier, and wrap.

    *operator* is already ``2π``-scaled and canonicalized; this only adds
    the band-specific modulation coefficient (shared by the drive and
    crosstalk instantiation paths).
    """
    band = BandContext(
        weight=weight,
        device_frame_freq=device_frame_freq,
        drive_freq=drive_freq,
        rwa=rwa,
    )
    return DynamicTerm(
        operator=operator,
        time_dependence=ScalarModulation(signal=_coefficient_from_modulation(signal, modulation, band)),
        origin=origin,
        tag=tag,
    )


# -- Coupling terms ------------------------------------------------------

def _collect_coupling_terms(
    chip: "Chip",
    resolved_frame: "ResolvedFrame",
    backend: Backend,
) -> tuple[list[Operator], list[tuple[Operator, ScalarModulation]], list[DroppedTerm]]:
    """Band-resolve couplings into ``(static_folds, (td_operator, modulation), dropped)``.

    Each coupling's *full* interaction is decomposed into ``(Δa, Δb)``
    excitation-change bands. Under resolved RWA, bands the coupling's
    ``rwa_keeps_band`` rejects are elided and reported as advisory
    :class:`DroppedTerm` records — the same bands
    :meth:`Chip.hamiltonian`'s mask removed from ``H₀``, so nothing is
    double-counted. Retained bands whose frame carrier
    ``Δa·ω_a + Δb·ω_b`` is *concretely* zero are already static inside
    ``H₀`` and stay there; a non-RWA coupling whose two frame references
    are both concretely zero skips the decomposition entirely (every
    band folds, nothing is dropped). Every other retained band is
    subtracted from ``H₀`` and re-attached with its carrier

    .. math::
        \\exp\\!\\left(-i\\,(\\Delta_a \\omega_a + \\Delta_b \\omega_b)\\,t\\right),

    the standard rotating-frame interaction picture for a bilinear
    coupling (see e.g. Gambetta et al., *PRA* **74**, 042318 (2006)).
    Traced carrier frequencies stay dynamic — concreteness is probed
    with :func:`maybe_concrete_scalar`, never by branching on a tracer.
    All emitted operators already carry the 2π factor.
    """
    coupling_static: list[Operator] = []
    td_terms: list[tuple[Operator, ScalarModulation]] = []
    dropped: list[DroppedTerm] = []
    dims = chip.dims
    label_to_index = {dev.label: i for i, dev in enumerate(chip.devices)}

    for coupling in chip.couplings:
        pair = (coupling.device_a_label, coupling.device_b_label)
        idx_a = label_to_index[pair[0]]
        idx_b = label_to_index[pair[1]]
        rwa = chip.resolve_rwa(coupling)
        omega_a = resolved_frame.frequencies.get(pair[0], 0.0)
        omega_b = resolved_frame.frequencies.get(pair[1], 0.0)

        # Without RWA and with both frame references concretely zero (lab
        # mode, or a shared-zero dict frame), every band's carrier is
        # concretely zero and nothing is dropped: the full interaction is
        # already static inside H₀, so the decomposition would be discarded
        # band by band. Skip it outright.
        if not rwa:
            conc_a = maybe_concrete_scalar(omega_a)
            conc_b = maybe_concrete_scalar(omega_b)
            if conc_a is not None and conc_a == 0.0 and conc_b is not None and conc_b == 0.0:
                continue

        with _backend_context(backend):
            h_full = coupling.interaction_hamiltonian()

        d_a = chip.devices[idx_a].levels
        d_b = chip.devices[idx_b].levels
        canonical = backend.to_canonical_operator(h_full).with_metadata(
            dims=(d_a, d_b),
            subsystem_labels=pair,
            tag="coupling_local",
        )
        sub_bands = decompose_two_body_canonical_bands(canonical, [d_a, d_b])
        for (delta_a, delta_b), band_canonical in sub_bands.items():
            osc_freq = delta_a * omega_a + delta_b * omega_b
            if rwa and not coupling.rwa_keeps_band(delta_a, delta_b):
                # The advisory amplitude is the dropped band's own largest
                # matrix element — the worst-case numerator of the
                # Bloch-Siegert smallness ratio — not the coupling's scalar
                # strength, which can differ per band in a multi-term
                # interaction. Raw arithmetic; stays traced if the payload is.
                band_values = band_canonical.values
                xp = array_namespace(band_values)
                dropped.append(
                    DroppedTerm(
                        source=coupling.label,
                        operator=f"coupling band (Δa={delta_a:+d}, Δb={delta_b:+d}) on {pair[0]}·{pair[1]}",
                        reason="counter-rotating under RWA",
                        band_weights=(delta_a, delta_b),
                        amplitude=xp.max(xp.abs(band_values)),
                        frequency=abs(osc_freq),
                    )
                )
                continue
            concrete_osc = maybe_concrete_scalar(osc_freq)
            if concrete_osc is not None and concrete_osc == 0.0:
                continue
            band_op = backend.from_canonical_operator(band_canonical)
            embedded = backend.embed_two_body(band_op, idx_a, idx_b, dims)
            scaled = TWO_PI * embedded
            coupling_static.append(-scaled)
            td_terms.append((scaled, ScalarModulation(signal=Carrier(freq=TWO_PI * osc_freq, sign=-1))))
    return coupling_static, td_terms, dropped

# -- Drive resolution ----------------------------------------------------

def _resolve_drives(
    chip: "Chip",
    drive_ops: list["DriveOp"],
) -> list[tuple[BaseDrive, "DriveOp", Any]]:
    """Map each drive op to ``(drive, drive_op, target)`` using the chip's control equipment.

    The target is the chip's canonical device (device lines) or coupling
    (edge pump lines); the two label spaces are disjoint by Chip
    construction, so a plain two-map lookup is unambiguous.

    Cross-checks the resolved drive's own wiring against the ``DriveOp``:
    an unconnected drive, a target mismatch, or a ``target_kind`` that
    disagrees with which map the label resolved from all raise
    ``ValueError``. Sequence scheduling enforces the same invariant at
    schedule time (:meth:`~quchip.control.sequence.QuantumSequence._schedule_on_drive`);
    this is the matching guard for ``DriveOp`` lists built directly,
    bypassing a ``QuantumSequence``.
    """
    equipment = chip.control_equipment
    resolved: list[tuple[BaseDrive, "DriveOp", Any]] = []
    for drive_op in drive_ops:
        label = drive_op.target_label
        if label in chip.device_map:
            device: Any = chip.device_map[label]
            resolved_kind = "device"
        elif label in chip.coupling_map:
            device = chip.coupling_map[label]
            resolved_kind = "edge"
        else:
            raise ValueError(
                f"Target label '{label}' not found on chip (neither device nor coupling). "
                f"Devices: {list(chip.device_map.keys())}; couplings: {list(chip.coupling_map.keys())}."
            )

        if equipment is None:
            raise ValueError("Chip has no control equipment configured.")
        drive = next((d for d in equipment.lines if d.label == drive_op.drive_label), None)
        if drive is None:
            raise ValueError(
                f"No drive with label '{drive_op.drive_label}' found in equipment. "
                f"Available: {[d.label for d in equipment.lines]}"
            )

        if drive.target_label is None:
            raise ValueError(
                f"Drive '{drive.label}' is not connected to a target, but a DriveOp schedules it "
                f"onto '{drive_op.target_label}'. Connect it first with drive.connect(device)."
            )
        if drive.target_label != drive_op.target_label:
            raise ValueError(
                f"Drive '{drive.label}' is wired to target '{drive.target_label}', but its DriveOp "
                f"targets '{drive_op.target_label}'."
            )
        if drive.target_kind != resolved_kind:
            raise ValueError(
                f"Drive '{drive.label}' declares target_kind '{drive.target_kind}', but target "
                f"'{drive_op.target_label}' resolved as '{resolved_kind}' on the chip."
            )
        resolved.append((drive, drive_op, device))
    return resolved


class _ScheduledSignal:
    """Internal record pairing a resolved drive with its (possibly transformed) signal program."""

    __slots__ = ("drive", "drive_op", "device", "signal")

    def __init__(
        self,
        drive: BaseDrive,
        drive_op: "DriveOp",
        device: "BaseDevice",
        signal: SignalProgram,
    ) -> None:
        self.drive = drive
        self.drive_op = drive_op
        self.device = device
        self.signal = signal

# -- Drive term compilation ----------------------------------------------


@dataclass(frozen=True)
class CompiledDriveTerm:
    """One pre-embedded, 2π-scaled drive band plus its reinstantiation metadata.

    Template-internal (stored on
    :attr:`~quchip.engine.ir.HamiltonianTemplate.drive_terms`, never
    handed to backends). During sweep instantiation, stage 2 combines
    ``weight``, ``device_frame_freq``, the variant's drive frequency, and
    ``rwa`` into a :class:`BandContext`, then dispatches on the channel's
    :class:`~quchip.control.signal_spec.DriveModulation` tag to emit the
    variant-specific :class:`~quchip.engine.ir.ScalarModulation` for
    ``operator``. ``device_frame_freq`` is in GHz (may be a traced
    scalar; see :class:`~quchip.engine.ir.ResolvedFrame`).
    """

    operator: CanonicalOperator
    drive_index: int
    modulation: DriveModulation
    weight: int
    device_frame_freq: Any
    rwa: bool
    origin: TermOrigin = "drive"
    tag: str | None = None


@dataclass(frozen=True)
class _StructuralDrop:
    """Template-cached pointer to a SINGLE_TONE weight-0 band dropped under RWA.

    The drop decision (:func:`_is_dropped_weight_zero_single_tone`) needs no
    drive frequency, but resolving it into a
    :class:`~quchip.engine.ir.DroppedTerm` audit record does; this record
    stays a pointer into ``drive_ops`` until
    :func:`instantiate_hamiltonian_description` knows the variant's
    frequency.
    """

    drive_index: int
    device_label: str


def _compile_edge_pump_terms(
    chip: "Chip",
    drive: BaseDrive,
    coupling: Any,
    drive_index: int,
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    *,
    dims: tuple[int, ...],
    subsystem_labels: tuple[str, ...],
) -> list[CompiledDriveTerm]:
    """Band-decompose a pump target's parametric operator into pre-embedded terms.

    Mirrors the static-coupling band residue (:func:`_collect_coupling_terms`):
    each ``(Δa, Δb)`` excitation-change band oscillates at
    ``Δa·ω_a + Δb·ω_b`` in the rotating frame, stored as the term's frame
    frequency with weight 1. The chip's RWA policy acted earlier, inside
    ``parametric_operator`` — it selects the retained *operator structure*
    only; the scheduled pump tone is never split
    (:func:`_edge_pump_coefficient`).
    """
    with _backend_context(backend):
        local_op = coupling.parametric_operator(chip)
    if local_op is None:
        raise TypeError(
            f"{type(coupling).__name__} is not modulable: its parametric_interaction() hook "
            "returns None. Implement parametric_interaction()/rwa_parametric_interaction() on "
            "the coupling (see CouplingModel), or use a modulable coupling such as TunableCapacitive."
        )
    modulation = drive._modulation
    assert modulation is not None, "edge drives declare a DriveModulation for the pump coefficient"
    idx_a = chip.device_index(coupling.device_a_label)
    idx_b = chip.device_index(coupling.device_b_label)
    d_a = chip.devices[idx_a].levels
    d_b = chip.devices[idx_b].levels
    canonical = backend.to_canonical_operator(local_op).with_metadata(
        dims=(d_a, d_b),
        subsystem_labels=(coupling.device_a_label, coupling.device_b_label),
        tag="edge_pump_local",
    )
    omega_a = resolved_frame.frequencies.get(coupling.device_a_label, 0.0)
    omega_b = resolved_frame.frequencies.get(coupling.device_b_label, 0.0)
    compiled: list[CompiledDriveTerm] = []
    for (delta_a, delta_b), band_canonical in decompose_two_body_canonical_bands(canonical, [d_a, d_b]).items():
        osc_freq = delta_a * omega_a + delta_b * omega_b
        band_op = backend.from_canonical_operator(band_canonical)
        embedded = backend.embed_two_body(band_op, idx_a, idx_b, dims)
        compiled.append(
            CompiledDriveTerm(
                operator=_apply_2pi_canonical(backend, embedded, dims=dims, labels=subsystem_labels, tag="edge_pump"),
                drive_index=drive_index,
                modulation=modulation,
                weight=1,
                device_frame_freq=osc_freq,
                rwa=False,
                origin="coupling",
                tag="edge_pump",
            )
        )
    return compiled


def _compile_drive_terms(
    chip: "Chip",
    resolved_drives: list[tuple[BaseDrive, "DriveOp", Any]],
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    *,
    dims: tuple[int, ...],
    subsystem_labels: tuple[str, ...],
) -> tuple[tuple[CompiledDriveTerm, ...], tuple[_StructuralDrop, ...]]:
    """Band-decompose each drive channel into pre-embedded, 2π-scaled drive terms.

    For each local drive channel the operator is split into
    single-subsystem excitation-change bands of weight ``w``
    (cf. :mod:`quchip.engine.bands`). Each band gets the RWA policy from
    its owning drive, which owns its local Hamiltonian; the modulation
    later assigns a carrier of the form
    ``exp(−i w (ω_d − ω_ref) t)`` evaluated against the envelope, which
    is the standard rotating-wave form for a driven multi-level system
    (Jaynes & Cummings 1963; Scully & Zubairy, *Quantum Optics*, §5). A
    SINGLE_TONE band at weight 0 under RWA is dropped here instead of
    compiled (:func:`_is_dropped_weight_zero_single_tone`) — the decision
    needs no drive frequency, so it is made once, structurally, at
    template-compile time; a :class:`_StructuralDrop` pointer is returned
    alongside so the variant-specific audit record can be built later.

    Operates on the resolved ``(drive, drive_op, device)`` triples —
    signal programs are variant-specific and play no role here. The
    resulting :class:`CompiledDriveTerm` is template-cached so
    homogeneous sweeps only rebuild signal-program leaves, not operators.
    """
    compiled: list[CompiledDriveTerm] = []
    structural_drops: list[_StructuralDrop] = []
    for drive_index, (drive, _drive_op, target) in enumerate(resolved_drives):
        if drive.target_kind == "edge":
            compiled.extend(
                _compile_edge_pump_terms(
                    chip, drive, target, drive_index, resolved_frame, backend,
                    dims=dims, subsystem_labels=subsystem_labels,
                )
            )
            continue
        device = target
        idx = chip.device_index(device.label)
        device_frame_freq = resolved_frame.frequencies.get(device.label, 0.0)
        drive_rwa = chip.resolve_rwa(drive)
        with _backend_context(backend):
            channels = drive.local_channels(device)
        for ch in channels:
            for weight, embedded in embed_single_mode_bands(
                backend,
                ch.operator,
                device_index=idx,
                dim=device.levels,
                label=device.label,
                dims=dims,
            ):
                if _is_dropped_weight_zero_single_tone(ch.modulation, weight, drive_rwa):
                    structural_drops.append(
                        _StructuralDrop(drive_index=drive_index, device_label=device.label)
                    )
                    continue
                compiled.append(
                    CompiledDriveTerm(
                        operator=_apply_2pi_canonical(
                            backend, embedded, dims=dims, labels=subsystem_labels, tag="drive"
                        ),
                        drive_index=drive_index,
                        modulation=ch.modulation,
                        weight=weight,
                        device_frame_freq=device_frame_freq,
                        rwa=drive_rwa,
                    )
                )
    return tuple(compiled), tuple(structural_drops)


# -- Signal-chain-injected dynamic terms (built per instantiation) -------

def _compile_extra_signal_terms(
    chip: "Chip",
    extra_signals: dict[tuple[str, int], SignalProgram],
    drive_ops: list["DriveOp"],
    resolved_frame: "ResolvedFrame",
    backend: Backend,
    *,
    dims: tuple[int, ...],
    subsystem_labels: tuple[str, ...],
) -> tuple[list[DynamicTerm], list[DroppedTerm]]:
    """Build dynamic terms for signal-chain-injected signals (e.g. crosstalk victims).

    Each key is ``(victim_drive_label, source_drive_index)`` so the
    source carrier frequency is always available when the victim drive
    line reuses a foreign source (classical microwave crosstalk: the
    nominal drive on device A leaks onto device B's drive line with a
    possibly different envelope but the same RF carrier). The emitted
    operators already carry 2π. Also returns advisory
    :class:`DroppedTerm` records for the victim bands' RWA-elided fast
    partners, mirroring the primary drive path.
    """
    equipment = chip.control_equipment
    if equipment is None:
        return [], []

    dynamic_terms: list[DynamicTerm] = []
    dropped: list[DroppedTerm] = []
    line_to_drive = {line.label: line for line in equipment.lines}

    for (victim_key, source_idx), victim_signal in extra_signals.items():
        victim_drive = line_to_drive.get(victim_key)
        if victim_drive is None or victim_drive._target is None:
            continue
        if victim_drive.target_kind == "edge":
            # A leak onto a pump line pumps the coupling with the leaked
            # signal at the source's carrier (the Crosstalk contract: the
            # leak is the RF copy of the source line — signal.py). Baseband
            # sources leak as baseband pumps (freq None).
            victim_target = victim_drive.target_label
            coupling = chip.coupling_map.get(victim_target) if victim_target is not None else None
            if coupling is None:
                continue
            source_freq = drive_ops[source_idx].freq
            for centry in _compile_edge_pump_terms(
                chip, victim_drive, coupling, source_idx, resolved_frame, backend,
                dims=dims, subsystem_labels=subsystem_labels,
            ):
                dynamic_terms.append(
                    _modulated_dynamic_term(
                        centry.operator,
                        victim_signal,
                        centry.modulation,
                        weight=centry.weight,
                        device_frame_freq=centry.device_frame_freq,
                        drive_freq=source_freq,
                        rwa=centry.rwa,
                        origin="crosstalk",
                        tag="crosstalk",
                    )
                )
            continue
        victim_device = victim_drive._target
        victim_label = victim_drive.device_label
        if victim_label is None:
            continue
        victim_idx = chip.device_index(victim_label)
        victim_frame_freq = resolved_frame.frequencies.get(victim_label, 0.0)
        drive_rwa = chip.resolve_rwa(victim_drive)
        source_freq = drive_ops[source_idx].freq

        with _backend_context(backend):
            victim_channels = victim_drive.local_channels(victim_device)
        for ch in victim_channels:
            for weight, embedded in embed_single_mode_bands(
                backend,
                ch.operator,
                device_index=victim_idx,
                dim=victim_device.levels,
                label=victim_label,
                dims=dims,
            ):
                if _is_dropped_weight_zero_single_tone(ch.modulation, weight, drive_rwa):
                    dropped.append(
                        _weight_zero_dropped_term(
                            source=victim_key, device_label=victim_label, drive_freq=source_freq,
                        )
                    )
                    continue
                dynamic_terms.append(
                    _modulated_dynamic_term(
                        _apply_2pi_canonical(
                            backend, embedded, dims=dims, labels=subsystem_labels, tag="crosstalk"
                        ),
                        victim_signal,
                        ch.modulation,
                        weight=weight,
                        device_frame_freq=victim_frame_freq,
                        drive_freq=source_freq,
                        rwa=drive_rwa,
                        origin="crosstalk",
                    )
                )
                partner = _dropped_drive_partner(
                    source=victim_key,
                    device_label=victim_label,
                    modulation=ch.modulation,
                    weight=weight,
                    drive_freq=source_freq,
                    device_frame_freq=victim_frame_freq,
                    rwa=drive_rwa,
                    origin="crosstalk",
                )
                if partner is not None:
                    dropped.append(partner)
    return dynamic_terms, dropped


# -- Build raw signals and apply signal chain ----------------------------

def _build_scheduled_signals_and_extras(
    chip: "Chip",
    drive_ops: list["DriveOp"],
) -> tuple[list[_ScheduledSignal], dict[tuple[str, int], SignalProgram]]:
    """Build raw drive signals and apply the equipment signal chain.

    Signals are keyed by ``(drive_label, drive_index)`` so multiple ops on
    the same drive stay distinct through the chain. Drives are
    frame-agnostic — the resolved frame is applied later via the
    modulation dispatch — so this step needs no frame input. Returns
    ``(scheduled, extra_signals)`` where *extra_signals* collects new keys
    introduced by the chain (e.g. crosstalk victim drives).
    """
    scheduled: list[_ScheduledSignal] = []
    for drive, drive_op, device in _resolve_drives(chip, drive_ops):
        spec = drive.signal_spec(drive_op, device)
        signal = _spec_to_raw_signal(spec)
        scheduled.append(_ScheduledSignal(drive=drive, drive_op=drive_op, device=device, signal=signal))

    extra_signals: dict[tuple[str, int], SignalProgram] = {}
    equipment = chip.control_equipment
    if equipment is None or not equipment.signal_chain:
        return scheduled, extra_signals

    raw_signals = {(item.drive.label, i): item.signal for i, item in enumerate(scheduled)}
    transformed = equipment.apply_signal_chain(raw_signals)
    scheduled_keys = set(raw_signals.keys())
    for i, item in enumerate(scheduled):
        key = (item.drive.label, i)
        if key in transformed:
            item.signal = transformed[key]
    for key, signal in transformed.items():
        if key not in scheduled_keys:
            extra_signals[key] = signal
    return scheduled, extra_signals


# -- Dropped-term aggregation --------------------------------------------

def _collect_dropped_terms(chip: "Chip", resolved_frame: "ResolvedFrame") -> tuple[DroppedTerm, ...]:
    """Gather coupling-declared advisory records (non-RWA approximations).

    RWA band drops are generated inside :func:`_collect_coupling_terms`;
    this pass collects whatever a coupling's own model elides. Records
    that declare ``band_weights`` without a frequency get the band's
    frame oscillation resolved here as ``|Σ wᵢ·f_ref,i|`` — raw
    arithmetic on possibly-traced frame frequencies, never concretized.
    """
    gathered: list[DroppedTerm] = []
    for coupling in chip.couplings:
        endpoint_labels = (coupling.device_a_label, coupling.device_b_label)
        for record in coupling.dropped_terms():
            weights = record.band_weights
            if record.frequency is None and weights is not None and len(weights) == len(endpoint_labels):
                freq = sum(
                    w * resolved_frame.frequencies.get(lbl, 0.0)
                    for w, lbl in zip(weights, endpoint_labels)
                )
                record = replace(record, frequency=abs(freq))
            gathered.append(record)
    return tuple(gathered)


def _dropped_drive_partner(
    *,
    source: str,
    device_label: str,
    modulation: "DriveModulation",
    weight: int,
    drive_freq: Any,
    device_frame_freq: Any,
    rwa: bool,
    origin: str,
) -> DroppedTerm | None:
    """Advisory record for the fast partner :func:`_single_tone_coefficient` elides under RWA.

    For a single-tone band of weight ``w ≠ 0`` the RWA keeps the slow
    component near ``|f_d − |w|·f_ref|`` and drops its counter-rotating
    partner oscillating at ``f_d + |w|·f_ref`` (≈ ``2·f_d`` on
    resonance) — the term whose amplitude-to-frequency ratio sets the
    Bloch–Siegert scale. Returns ``None`` when nothing is dropped
    (``rwa=False``, baseband modulations, or the weight-0 band, which
    keeps the full real field). The record's amplitude is ``None``:
    drive prefactors are time-dependent envelopes, described by the
    schedule rather than a single number.
    """
    if not rwa or modulation is not DriveModulation.SINGLE_TONE or weight == 0 or drive_freq is None:
        return None
    return DroppedTerm(
        source=source,
        operator=f"{origin} band w={weight:+d} on {device_label} (fast partner)",
        reason="counter-rotating drive component under RWA",
        band_weights=(weight,),
        frequency=drive_freq + abs(weight) * device_frame_freq,
    )


# -- Template validation -------------------------------------------------

def _validate_variant_drive_ops(
    template: HamiltonianTemplate,
    drive_ops: list["DriveOp"],
    chip: "Chip",
) -> None:
    """Check that *drive_ops* match the template's drive/device/envelope/drive-type shape."""
    reference_ops = template.reference_drive_ops
    if len(drive_ops) != len(reference_ops):
        raise ValueError(
            "Homogeneous Hamiltonian template requires the same number of scheduled pulse entries. "
            f"Expected {len(reference_ops)}, got {len(drive_ops)}."
        )

    equipment = chip.control_equipment
    for index, (reference_op, variant_op) in enumerate(zip(reference_ops, drive_ops)):
        if variant_op.target_label != reference_op.target_label:
            raise ValueError(
                f"Variant drive op {index} targets '{variant_op.target_label}', "
                f"expected '{reference_op.target_label}'."
            )
        if variant_op.drive_label != reference_op.drive_label:
            raise ValueError(
                f"Variant drive op {index} uses drive '{variant_op.drive_label}', "
                f"expected '{reference_op.drive_label}'."
            )
        if type(variant_op.envelope) is not type(reference_op.envelope):
            raise ValueError(
                f"Variant drive op {index} uses envelope '{type(variant_op.envelope).__name__}', "
                f"expected '{type(reference_op.envelope).__name__}'."
            )
        if equipment is not None:
            drive = next((d for d in equipment.lines if d.label == variant_op.drive_label), None)
            reference_drive = next((d for d in equipment.lines if d.label == reference_op.drive_label), None)
            if drive is not None and reference_drive is not None and type(drive) is not type(reference_drive):
                raise ValueError(
                    f"Variant drive op {index} resolved to drive '{type(drive).__name__}', "
                    f"expected '{type(reference_drive).__name__}'."
                )

# -- Public API ----------------------------------------------------------

def compile_hamiltonian_template(
    chip: "Chip",
    drive_ops: list["DriveOp"],
    *,
    resolved_frame: "ResolvedFrame",
) -> HamiltonianTemplate:
    """Compile the invariant Hamiltonian skeleton (H₀, couplings, pre-embedded drive bands).

    Everything that does not change across a homogeneous sweep lives in
    the template: static Hamiltonian, static-coupling folds, invariant
    dynamic couplings, and band-decomposed drive operators pre-embedded
    and pre-scaled by 2π. Per-sweep instantiation
    (:func:`instantiate_hamiltonian_description`) rebuilds only the
    :class:`~quchip.engine.ir.SignalProgram` leaves, so envelope
    parameters, drive frequencies, phases, and frame scalars can sweep
    through JAX without retracing operator tensors.
    """
    backend = chip.backend
    dims = tuple(chip.dims)
    subsystem_labels = tuple(d.label for d in chip.devices)

    h0 = _build_static_h0(chip, resolved_frame, backend)
    coupling_static, coupling_td, coupling_dropped = _collect_coupling_terms(chip, resolved_frame, backend)
    for op in coupling_static:
        h0 = h0 + op

    # The coupling fold above cancels the lab-frame interaction out of H₀
    # exactly, leaving its diagonal offsets stored as explicit zeros; prune
    # them (tracer-guarded) so backends never integrate dead structure.
    static_terms = (
        StaticTerm(
            operator=prune_zero_diagonals(
                backend.to_canonical_operator(h0).with_metadata(
                    dims=dims,
                    subsystem_labels=subsystem_labels,
                    tag="H0",
                )
            ),
            coefficient=1.0,
            origin="device",
            metadata={"frame": str(resolved_frame)},
        ),
    )

    invariant_dynamic_terms: list[DynamicTerm] = []
    for op, td in coupling_td:
        invariant_dynamic_terms.append(
            DynamicTerm(
                operator=backend.to_canonical_operator(op).with_metadata(
                    dims=dims,
                    subsystem_labels=subsystem_labels,
                    tag="coupling",
                ),
                time_dependence=td,
                origin="coupling",
            )
        )

    # Component-owned dynamic terms (tunable couplings, tunable device
    # frequencies, parametric flux, etc.). The chip enumerates its component
    # families generically; this stage embeds each local operator by its
    # support arity and applies the 2π boundary uniformly.
    for local_op, time_dependence, support, origin, tag in chip.dynamic_contributions():
        embedded = embed_on_support(backend, local_op, support, dims)
        invariant_dynamic_terms.append(
            _dynamic_term(
                backend,
                embedded,
                dims=dims,
                labels=subsystem_labels,
                tag=tag,
                # Chip.dynamic_contributions() declares its 4th tuple element as plain `str`
                # (chip/chip.py, out of scope here); only the five TermOrigin literals are
                # ever produced at runtime.
                origin=cast(TermOrigin, origin),
                time_dependence=time_dependence,
            )
        )

    # Invariant signals never change across variants, so simplification
    # happens once here instead of on every instantiation.
    simplified_invariant = tuple(
        replace(term, time_dependence=ScalarModulation(signal=_simplify_signal(term.time_dependence.signal)))
        for term in invariant_dynamic_terms
    )

    # Static terms are invariant across a homogeneous sweep, so their
    # spectral-bound hint (a dense diagonal materialization) is computed
    # once here rather than on every instantiation. Stored in ordinary GHz.
    static_span = _static_diagonal_span(static_terms)
    static_spectral_bound_ghz = static_span / TWO_PI if static_span is not None else None

    drive_terms, weight_zero_drops = _compile_drive_terms(
        chip,
        _resolve_drives(chip, drive_ops),
        resolved_frame,
        backend,
        dims=dims,
        subsystem_labels=subsystem_labels,
    )

    return HamiltonianTemplate(
        resolved_frame=resolved_frame,
        dims=dims,
        static_terms=static_terms,
        invariant_dynamic_terms=simplified_invariant,
        drive_terms=drive_terms,
        reference_drive_ops=tuple(drive_ops),
        dropped_terms=_collect_dropped_terms(chip, resolved_frame) + tuple(coupling_dropped),
        weight_zero_drops=weight_zero_drops,
        static_spectral_bound_ghz=static_spectral_bound_ghz,
    )


def instantiate_hamiltonian_description(
    template: HamiltonianTemplate,
    drive_ops: list["DriveOp"],
    chip: "Chip",
) -> HamiltonianDescription:
    """Rebuild signal-program leaves from *drive_ops* and attach them to the template's operators."""
    _validate_variant_drive_ops(template, drive_ops, chip)
    backend = chip.backend
    dims = template.dims
    subsystem_labels = tuple(d.label for d in chip.devices)

    scheduled, extra_signals = _build_scheduled_signals_and_extras(chip, drive_ops)

    # Only the variant-specific terms built below need simplification;
    # the template's invariant terms were simplified at compile time.
    # Drive frequencies are per-op, so the fast partners the RWA drops
    # only become auditable here — their records join the template's.
    fresh_terms: list[DynamicTerm] = []
    fresh_dropped: list[DroppedTerm] = []
    for compiled in template.drive_terms:
        drive_op = drive_ops[compiled.drive_index]
        signal = scheduled[compiled.drive_index].signal
        fresh_terms.append(
            _modulated_dynamic_term(
                compiled.operator,
                signal,
                compiled.modulation,
                weight=compiled.weight,
                device_frame_freq=compiled.device_frame_freq,
                drive_freq=drive_op.freq,
                rwa=compiled.rwa,
                origin=compiled.origin,
                tag=compiled.tag,
            )
        )
        partner = _dropped_drive_partner(
            source=drive_op.drive_label,
            device_label=drive_op.target_label,
            modulation=compiled.modulation,
            weight=compiled.weight,
            drive_freq=drive_op.freq,
            device_frame_freq=compiled.device_frame_freq,
            rwa=compiled.rwa,
            origin="drive",
        )
        if partner is not None:
            fresh_dropped.append(partner)

    extra_terms, extra_dropped = _compile_extra_signal_terms(
        chip,
        extra_signals,
        drive_ops,
        template.resolved_frame,
        backend,
        dims=dims,
        subsystem_labels=subsystem_labels,
    )
    fresh_terms.extend(extra_terms)
    fresh_dropped.extend(extra_dropped)

    # Structural weight-0 SINGLE_TONE drops carry no frequency at compile
    # time; resolve each pointer against its variant's drive_op now.
    for drop in template.weight_zero_drops:
        drive_op = drive_ops[drop.drive_index]
        fresh_dropped.append(
            _weight_zero_dropped_term(
                source=drive_op.drive_label, device_label=drop.device_label, drive_freq=drive_op.freq,
            )
        )

    dynamic_terms = template.invariant_dynamic_terms + tuple(
        replace(
            term,
            time_dependence=ScalarModulation(signal=_simplify_signal(term.time_dependence.signal)),
        )
        for term in fresh_terms
    )

    metadata: dict[str, Any] = {"frame": str(template.resolved_frame)}
    metadata.update(_solver_hint_metadata(template.static_spectral_bound_ghz, dynamic_terms))

    return HamiltonianDescription(
        static_terms=template.static_terms,
        dynamic_terms=dynamic_terms,
        dims=dims,
        metadata=metadata,
        dropped_terms=template.dropped_terms + tuple(fresh_dropped),
    )


def build_hamiltonian_description(
    chip: "Chip",
    drive_ops: list["DriveOp"],
    *,
    resolved_frame: "ResolvedFrame",
) -> HamiltonianDescription:
    """One-shot stage 2: compile the template then instantiate a single variant.

    Equivalent to
    :func:`compile_hamiltonian_template` followed by
    :func:`instantiate_hamiltonian_description` with the same
    ``drive_ops``. Prefer the two-step form when solving many variants
    that share the same chip topology.

    Parameters
    ----------
    chip : Chip
        The chip whose device, coupling, and drive Hamiltonians are
        assembled (2π applied at this boundary).
    drive_ops : list of DriveOp
        Scheduled drive operations to embed as dynamic terms.
    resolved_frame : ResolvedFrame
        Stage-1 frame result carrying the per-device frame frequencies,
        demodulation frequencies, and frame mode.

    Returns
    -------
    HamiltonianDescription
        Static terms, dynamic terms, and dropped-term records for the
        single variant.
    """
    template = compile_hamiltonian_template(chip, drive_ops, resolved_frame=resolved_frame)
    return instantiate_hamiltonian_description(template, drive_ops, chip)
