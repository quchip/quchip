"""Human-readable text summaries — the ``describe()`` surface.

This is a *view* layer: it renders what the user built (devices, couplings,
control wiring, scheduled pulses) — or what a transform derived, e.g. an
``eliminate()`` fold report — as a sectioned plain-text report. It never
computes physics beyond ordinary arithmetic on already-derived quantities,
and never concretizes a traced value for anything but display — tracers
render as ``<traced>`` even on this debugging surface.

Units come from the declarative :func:`~quchip.declarative.parameters.parameter`
metadata (``unit=``), so extension authors who declare units get correct
``describe()`` output for free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quchip.declarative.parameters import parameter_fields
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.chip.transformations import EliminationResult
    from quchip.control.sequence import QuantumSequence


def format_value(value: Any) -> str:
    """Render one parameter value compactly; traced values stay symbolic."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    concrete = maybe_concrete_scalar(value)
    if concrete is not None:
        return f"{concrete:.6g}"
    if contains_tracer(value):
        return "<traced>"
    return str(value)


def _param_text(name: str, value: Any, unit: str | None) -> str:
    text = f"{name} = {format_value(value)}"
    if unit is not None and value is not None:
        text += f" {unit}"
    return text


def _declared_param_line(obj: Any) -> str:
    """One indented line joining an object's declared parameters with units."""
    fields = parameter_fields(type(obj))
    parts = [
        _param_text(name, getattr(obj, name), spec.unit)
        for name, spec in fields.items()
    ]
    return "   ".join(parts)


def _section(title: str, rule: str = "─") -> list[str]:
    return [title, rule * len(title)]


def describe_chip(chip: "Chip") -> str:
    """Sectioned plain-text report of a chip's composition. See :meth:`Chip.describe`."""
    lines: list[str] = []

    title = "Chip" if chip.label is None else f"Chip {chip.label!r}"
    lines += _section(title, "═")
    frame = chip.frame if isinstance(chip.frame, str) else format_value(chip.frame)
    lines.append(f"Frame    : {frame} (rwa={chip.rwa})")
    lines.append(f"Dressed  : {'cached' if chip.is_dressed else 'not computed'}")
    dims = [d.levels for d in chip.devices]
    if dims:
        total = 1
        for n in dims:
            total *= n
        lines.append(
            f"Hilbert  : {' x '.join(str(n) for n in dims)} = {total} levels"
        )

    lines.append("")
    lines += _section(f"Devices ({len(chip.devices)})")
    for dev in chip.devices:
        lines.append(f"{dev.label} — {type(dev).__name__}")
        params = _declared_param_line(dev)
        detail = f"{params}   levels = {dev.levels}" if params else f"levels = {dev.levels}"
        lines.append(f"    {detail}")
        noise = [
            _param_text(name, getattr(dev, name), unit)
            for name, unit in (("T1", "ns"), ("T2", "ns"), ("thermal_population", None))
            if getattr(dev, name) is not None
        ]
        if noise:
            lines.append(f"    {'   '.join(noise)}")

    if chip.couplings:
        lines.append("")
        lines += _section(f"Couplings ({len(chip.couplings)})")
        for coupling in chip.couplings:
            pair = f"{coupling.device_a_label} ↔ {coupling.device_b_label}"
            lines.append(f"{coupling.label} : {pair}")
            detail = _declared_param_line(coupling)
            if coupling.rwa is not None:
                override = f"rwa = {coupling.rwa}"
                detail = f"{detail}   {override}" if detail else override
            if detail:
                lines.append(f"    {detail}")

    equipment = chip.control_equipment
    if equipment is not None and equipment.lines:
        lines.append("")
        lines += _section(f"Control ({len(equipment.lines)} line{'s' if len(equipment.lines) != 1 else ''})")
        for drive in equipment.lines:
            target = drive.device_label if drive.device_label is not None else "(unwired)"
            lines.append(f"{drive.label} — {type(drive).__name__} → {target}")
        chain = equipment.signal_chain
        if chain:
            counts: dict[str, int] = {}
            for transform in chain:
                name = type(transform).__name__
                counts[name] = counts.get(name, 0) + 1
            summary = ", ".join(
                f"{name} ×{n}" if n > 1 else name for name, n in sorted(counts.items())
            )
            lines.append(f"    signal chain: {summary}")

    if chip.baths:
        lines.append("")
        lines += _section(f"Baths ({len(chip.baths)})")
        for bath in chip.baths:
            targets = ", ".join(bath.resolve_targets(chip))
            detail = f"{bath.label} — {bath.recipe} on {targets}"
            extras = [
                _param_text(name, getattr(bath, name), unit)
                for name, unit in (("temperature", "mK"), ("rate", "1/ns"))
                if getattr(bath, name) is not None
            ]
            if extras:
                detail += f"   ({'   '.join(extras)})"
            lines.append(detail)

    return "\n".join(lines)


def _envelope_text(envelope: Any) -> str:
    """Compact envelope rendering: type name plus its declared parameters."""
    fields = parameter_fields(type(envelope))
    parts = []
    for name, spec in fields.items():
        text = f"{name}={format_value(getattr(envelope, name))}"
        if spec.unit is not None:
            text += f" {spec.unit}"
        parts.append(text)
    return f"{type(envelope).__name__}({', '.join(parts)})" if parts else type(envelope).__name__


def describe_sequence(seq: "QuantumSequence") -> str:
    """Timeline table of a sequence's pulses. See :meth:`QuantumSequence.describe`."""
    ops = seq.scheduled_ops
    header = (
        f"QuantumSequence — {len(ops)} pulse{'s' if len(ops) != 1 else ''}, "
        f"total duration {format_value(seq.total_duration)} ns"
    )
    lines = [header, "─" * len(header)]

    if ops:
        rows = [("window (ns)", "drive", "envelope", "freq")]
        for op in ops:
            start = maybe_concrete_scalar(op.start_time)
            duration = maybe_concrete_scalar(op.envelope.duration)
            if start is not None and duration is not None:
                window = f"[{start:.6g}, {start + duration:.6g}]"
            else:
                window = f"[{format_value(op.start_time)}, +{format_value(op.envelope.duration)}]"
            drive = f"{op.drive_label} → {op.target_label}"
            freq = "baseband" if op.freq is None else f"{format_value(op.freq)} GHz"
            rows.append((window, drive, _envelope_text(op.envelope), freq))
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        for row in rows:
            lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())

    other = [entry for entry in seq._entries if not hasattr(entry, "envelope")]
    if other:
        counts: dict[str, int] = {}
        for entry in other:
            name = type(entry).__name__.strip("_").removesuffix("Entry")
            counts[name] = counts.get(name, 0) + 1
        summary = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
        lines.append(f"(plus {summary})")

    return "\n".join(lines)


def _concrete(value: Any) -> float | None:
    """Best-effort float for report arithmetic and unit conversion; ``None`` when traced."""
    return maybe_concrete_scalar(value)


def _survivor_lines(label: str, entry: Any, reduced_device: Any) -> list[str]:
    """Before -> after freq (and T1, when either side carries one) for one survivor."""
    freq_after = _concrete(entry["freq_after"])
    lamb_shift = _concrete(entry["lamb_shift"])
    if freq_after is None or lamb_shift is None:
        lines = [f"{label}: freq <traced> → <traced> GHz"]
    else:
        freq_before = freq_after - lamb_shift
        sign = "+" if lamb_shift >= 0 else ""
        lines = [
            f"{label}: freq {freq_before:.6g} → {freq_after:.6g} GHz   "
            f"(Lamb shift {sign}{lamb_shift * 1e3:.3g} MHz)"
        ]

    t1_after_raw = getattr(reduced_device, "T1", None)
    if t1_after_raw is None:
        return lines
    t1_after = _concrete(t1_after_raw)
    purcell_rate = _concrete(entry.get("purcell_rate", 0.0))
    if t1_after is None or purcell_rate is None:
        lines.append("    T1   <traced> → <traced> µs")
        return lines
    rate_before = 1.0 / t1_after - purcell_rate
    t1_before = 1.0 / rate_before if rate_before > 1e-30 else float("inf")
    tag = f"   (Purcell, mediated rate 1/{(1.0 / purcell_rate) / 1e3:.4g} µs)" if purcell_rate > 0 else ""
    lines.append(f"    T1   {t1_before / 1e3:.4g} → {t1_after / 1e3:.4g} µs{tag}")
    return lines


def _exchange_lines(chip: "Chip", entry: Any) -> list[str]:
    """Emitted-edge + Yan-formula tag, and the ZZ line (placeholder under ``method='sw'``)."""
    lines: list[str] = []
    edge_label = entry.get("folded_into")
    if edge_label is not None:
        edge = chip.coupling_map.get(edge_label)
        strength = _concrete(edge.coupling_strength) if edge is not None else None
        strength_name = edge.coupling_strength_name if edge is not None else "g"
        edge_type = type(edge).__name__ if edge is not None else "?"
        strength_text = f"{strength * 1e3:.4g} MHz" if strength is not None else "<traced>"
        lines.append(
            f"edge '{edge_label}': {edge_type}({strength_name} = {strength_text})   "
            "[J = g_a g_b / 2 · (1/Δ_a + 1/Δ_b)]"
        )
    label_a, label_b = entry.get("between", ("?", "?"))
    zz = entry.get("zz")
    if zz is None:
        zz_text = '—   (available under method="exact")'
    else:
        zz_value = _concrete(zz)
        zz_text = f"{zz_value * 1e3:.4g} MHz" if zz_value is not None else "<traced>"
    lines.append(f"ZZ({label_a}, {label_b}) = {zz_text}")
    return lines


def _validity_line(validity: dict[str, Any]) -> str:
    """One ``g/Δ`` verdict per eliminated coupling, plus the shared min block gap."""
    parts = []
    min_gap = None
    for coupling_label, v in validity.items():
        g_over_delta = _concrete(v.get("g_over_delta"))
        is_valid = _concrete(v.get("is_valid"))
        if min_gap is None and "min_block_gap" in v:
            min_gap = _concrete(v["min_block_gap"])
        if g_over_delta is None or is_valid is None:
            parts.append(f"{coupling_label} g/Δ=<traced>")
        else:
            mark = "✓" if is_valid else "✗"
            parts.append(f"{coupling_label} g/Δ={g_over_delta:.2g} {mark} (< 0.1)")
    line = "validity: " + " · ".join(parts)
    if min_gap is not None:
        line += f" · min block gap {min_gap:.3g} GHz"
    return line


def _classify_notes(notes: list[str]) -> tuple[str | None, str | None, list[str], list[str]]:
    """Split fold notes into (method, dropped text, retarget lines, leftover lines).

    The method tag and the "Dropped: ..." summary are literal substrings
    ``eliminate()`` always emits for a device target; a retarget conversion
    note always opens with ``"drive '<label>': "`` (:func:`quchip.chip.retarget._flux_bridge_rule`
    and any rule following the same convention). Everything else — the
    coupling-elimination summary, a folded-direct-coupling note, the
    per-pair exchange-formula note — is not itemized further here and
    renders verbatim.
    """
    method: str | None = None
    dropped: str | None = None
    retarget: list[str] = []
    leftover: list[str] = []
    for note in notes:
        if "(method='sw')" in note:
            method = "sw"
        elif "(method='exact')" in note:
            method = "exact"
        elif note.startswith("Dropped: "):
            dropped = note[len("Dropped: ") :].rstrip(".")
        elif note.startswith("drive '"):
            retarget.append(note)
        else:
            leftover.append(note)
    return method, dropped, retarget, leftover


def describe_elimination(result: "EliminationResult") -> str:
    """Human-readable fold report for :func:`~quchip.chip.transformations.eliminate`.

    Every fold stated explicitly, before -> after: per-survivor freq (and T1
    when either side carries one), the emitted/upgraded exchange edge with
    its Yan-formula tag, the ZZ line (a placeholder under ``method="sw"``,
    the exact residual under ``method="exact"``), any control-line retarget,
    the per-coupling validity verdict, and the dropped-physics summary. See
    :meth:`~quchip.chip.transformations.EliminationResult.describe`.
    """
    method, dropped, retarget, leftover = _classify_notes(result.notes)
    lines = _section("Elimination fold report" + (f" (method='{method}')" if method else ""))

    for label, entry in result.effective_params.items():
        if label == "exchange":
            continue
        lines.extend(_survivor_lines(label, entry, result.chip[label]))

    exchange = result.effective_params.get("exchange")
    if exchange is not None:
        if "between" in exchange:
            lines.extend(_exchange_lines(result.chip, exchange))
        else:
            for pair in sorted(exchange):
                lines.extend(_exchange_lines(result.chip, exchange[pair]))

    lines.extend(retarget)

    if result.validity:
        lines.append(_validity_line(result.validity))

    if dropped:
        items = [item.strip() for item in dropped.split(",")]
        lines.append("dropped: " + " · ".join(items))

    if leftover:
        lines.append("notes:")
        lines.extend(f"  - {note}" for note in leftover)

    return "\n".join(lines)
