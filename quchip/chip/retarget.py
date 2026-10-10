"""Retarget registry for control lines stranded by ``eliminate()``.

A control line's target has no image in the reduced model when its device was
eliminated or its coupling touched the eliminated mode. Without this registry,
the user must unwire such a line. A per-(drive type, target type, result kind)
registry lets a converter replace the line with equivalent lines wired to the
reduced chip. For example, a :class:`~quchip.control.drive.FluxDrive` on an
eliminated coupler becomes a :class:`~quchip.control.drive.ParametricDrive`
that pumps each emitted edge. Extensions of this registry never touch
:func:`~quchip.chip.transformations.eliminate` itself. A target with no
registered rule still raises the fail-fast unwire/keep error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quchip.control.drive import FluxDrive
from quchip.devices.base import BaseDevice


@dataclass(frozen=True)
class RetargetContext:
    """Everything a converter can consult, built by ``eliminate()`` after the fold.

    Attributes
    ----------
    chip
        The original chip (read-only; pre-elimination).
    reduced_chip
        The final reduced chip. Every emitted or upgraded edge already
        exists on it; control equipment is not yet attached.
    mode_label
        Label of the eliminated device (or coupling, for ``"crosskerr"``).
    result_kind
        Structure of what the reduction produced. The value is ``"edge"`` when
        the eliminated device mediated exchange between two or more survivors
        (one effective edge per survivor pair). The value is ``"leaf-fold"``
        for a single-survivor leaf and ``"crosskerr"`` for a coupling target.
    edges
        For ``"edge"`` and ``"crosskerr"``: the per-pair reduction entries,
        keyed ``(label_a, label_b)`` in emission order. Each entry has at least
        ``"coupling"`` (the edge's label on the reduced chip). For ``"edge"``,
        each entry also has the exchange bookkeeping (``"j_eff"``,
        ``"dJ_domega_c"``, ...). The entries are always pair-keyed for any
        number of pairs, so one entry is the two-survivor case, not a different
        shape. ``None`` for ``"leaf-fold"``.
    """

    chip: Any
    reduced_chip: Any
    mode_label: str
    result_kind: str
    edges: dict[tuple[str, str], dict[str, Any]] | None


@dataclass(frozen=True)
class RetargetResult:
    """A converter's replacement lines, extra signal-chain transforms, and fold note.

    Attributes
    ----------
    lines
        Replacement control lines. Exactly one of them must keep the original
        line's label, so existing ``Crosstalk``/``Delay`` entries keyed by that
        label, and replayed ``schedule()`` calls, stay valid. All other lines
        carry derived labels.
    transforms
        Signal-chain transforms to append, in application order. The equipment
        applies its chain front to back, so a transform that feeds a line must
        come before a transform that scales it.
    note
        One fold-report line, appended to :attr:`EliminationResult.notes`.
    """

    lines: tuple[Any, ...]
    transforms: tuple[Any, ...] = ()
    note: str = ""


_RETARGET_RULES: dict[tuple[type, type, str], Any] = {}


def register_retarget_rule(drive_type: type, target_type: type, result_kind: str, rule: Any) -> None:
    """Register a converter for ``(drive type, eliminated-target type, result kind)``.

    Lookup (:func:`lookup_retarget_rule`) walks both types' MROs, so a rule
    registered for a base type also covers its subclasses. ``result_kind``
    matches exactly. Use this extension point to teach
    :func:`~quchip.chip.transformations.eliminate` to carry a new kind of
    stranded control line without changing
    :func:`~quchip.chip.transformations.eliminate`.

    Parameters
    ----------
    drive_type : type
        Control-line class the rule handles.
    target_type : type
        Eliminated-target class the rule handles: a device type (the
        eliminated mode itself) for ``"edge"``/``"leaf-fold"``.
    result_kind : str
        ``"edge"``, ``"leaf-fold"``, or ``"crosskerr"``.
    rule : callable
        ``rule(line, ctx: RetargetContext) -> RetargetResult``.
    """
    _RETARGET_RULES[(drive_type, target_type, result_kind)] = rule


def lookup_retarget_rule(drive_type: type, target_type: type, result_kind: str) -> Any | None:
    """MRO-aware registry lookup: the most specific ``(drive, target)`` pair wins."""
    for dt in drive_type.__mro__:
        for tt in target_type.__mro__:
            rule = _RETARGET_RULES.get((dt, tt, result_kind))
            if rule is not None:
                return rule
    return None


def _flux_edge_pump_rule(line: Any, ctx: RetargetContext) -> RetargetResult:
    """FluxDrive on an eliminated coupler → one baseband ParametricDrive per emitted edge.

    The flux line was a knob on the eliminated mode's frequency; each edge
    the elimination emitted responds to that knob with its own linearized
    weight, ``δJ_ab(t) = (∂J_ab/∂ω_c) · δω_c(t)``. The conversion realizes
    exactly that: the first emitted pair's pump keeps the original label (a
    static, emission-order choice — never a comparison of traced weights),
    every further edge gets a pump labeled ``{label}_{a}_{b}`` fed a
    unit-amplitude ``Crosstalk`` copy of the scheduled signal, and every
    pump line carries its own ``Gain(∂J_ab/∂ω_c)``. One replayed
    ``schedule(label, ...)`` call therefore drives every edge at the correct
    relative weight, with each weight a plain traced factor (no ratios).
    All lines are baseband (the FluxDrive envelope carries δω_c(t) in GHz;
    each pump envelope carries δJ_ab(t)), so the replayed call needs no freq
    argument. Copies precede gains in the transform tuple: a ``Gain`` on a
    copy-fed line is a no-op until the copy has landed on it. Small-signal:
    exact to first order in δω_c, valid for δω_c ≪ Δ.
    """
    from quchip.control.drive import ParametricDrive
    from quchip.control.signal import Crosstalk, Gain

    assert ctx.edges  # guaranteed by result_kind == "edge"
    lines: list[Any] = []
    copies: list[Any] = []
    gains: list[Any] = []
    pump_labels: list[str] = []
    for position, ((label_a, label_b), entry) in enumerate(ctx.edges.items()):
        edge = ctx.reduced_chip.coupling(entry["coupling"])
        pump_label = line.label if position == 0 else f"{line.label}_{label_a}_{label_b}"
        if position > 0:
            copies.append(Crosstalk(line.label, pump_label, beta=1.0))
        lines.append(ParametricDrive(edge, label=pump_label))
        gains.append(Gain(pump_label, entry["dJ_domega_c"]))
        pump_labels.append(f"'{entry['coupling']}'")
    note = (
        f"drive '{line.label}': FluxDrive('{ctx.mode_label}') → ParametricDrive on "
        f"{', '.join(pump_labels)}, Gain ∂J/∂ω_c per edge (small-signal, δω_c ≪ Δ)"
    )
    return RetargetResult(lines=tuple(lines), transforms=tuple(copies + gains), note=note)


register_retarget_rule(FluxDrive, BaseDevice, "edge", _flux_edge_pump_rule)
