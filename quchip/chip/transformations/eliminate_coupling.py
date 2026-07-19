"""Coupling-target elimination: reduce an exchange edge to its dressed cross-Kerr.

The device path removes a mode; this path removes an *edge*. Both endpoint
devices survive (Lamb-shifted), and the exchange coupling between them is
replaced by a :class:`~quchip.chip.couplings.CrossKerr` carrying the full
dressed pull ``χ = E₁₁ − E₁₀ − E₀₁ + E₀₀``, read off the chip's exact dressed
spectrum. This is the effective-readout-chip flow — reduce a qubit-resonator
exchange edge to the diagonal interaction an ordinary charge line probes.

The handler registers a coupling-kind :class:`~quchip.chip.transformations.dispatch.EliminationTarget`
at import time, so :func:`~quchip.chip.transformations.dispatch.eliminate`
dispatches any coupling label here without importing this module directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from quchip.chip.couplings import CrossKerr
from quchip.chip.transformations.dispatch import EliminationTarget, register_elimination_target
from quchip.chip.transformations.plumbing import (
    StrandedLine,
    detach_intermediate_clone,
    plan_stranded_lines,
    reattach_equipment,
    rebuild_chip,
)
from quchip.chip.transformations.result import EliminationResult, _HasFreq
from quchip.utils.labeling import LabelKeyedDict, resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


def reduce_coupling(chip: "Chip", target: Any, method: str) -> EliminationResult:
    """Reduce an exchange edge to its dressed cross-Kerr shift; both endpoints survive.

    Unlike device elimination, no mode is removed — the edge itself is the
    target, so there is no P/Q block to partition and no perturbative
    "sw" route: both endpoints' dressed 0→1 frequencies (the other endpoint
    pinned to its ground state) come from the chip's exact dressed spectrum
    regardless of :func:`eliminate`'s ``method`` argument, via
    :meth:`~quchip.chip.chip.Chip.freq`. The coupling is replaced by a
    :class:`~quchip.chip.couplings.CrossKerr` carrying the full dressed pull
    ``χ = E₁₁ − E₁₀ − E₀₁ + E₀₀`` — one full-chip diagonalization via
    :meth:`Chip.dispersive_shift`. This is a *uniform-χ* approximation
    (:class:`~quchip.chip.couplings.CrossKerr`'s own declared approximation):
    per-level χ differences and dispersive breakdown near the
    critical photon number are not represented.

    ``chi`` is evaluated eagerly rather than deferred like the device path's
    lazy ``effective_params["chi"]`` entry — :class:`CrossKerr` multiplies its
    ``chi`` parameter directly into an operator expression, so a live value
    (not a callable) is what the coupling itself needs; eager evaluation here
    is more JAX-friendly than the alternative, not less.

    Both endpoint devices survive, so no device-targeting control line is
    stranded. A line pumping the eliminated coupling *itself* has no image in
    the reduced model unless a registered rule converts it (``result_kind =
    "crosskerr"``, :mod:`quchip.chip.retarget`) — none ships, so such a line
    always raises the same fail-fast error the device path raises for a
    doomed line, before any fold work runs.

    Parameters
    ----------
    chip
        Source chip (never mutated).
    target
        The coupling to eliminate — label string or object.
    method
        Accepted for signature uniformity with the device path and has no
        effect here: no mode is removed, so the reduction always reads the
        chip's exact dressed spectrum (:func:`eliminate` has already validated
        it).

    Returns
    -------
    EliminationResult

    Raises
    ------
    NotImplementedError
        ``target`` names a coupling whose class does not declare
        :attr:`~quchip.chip.coupling_base.BaseCoupling.reduces_to_crosskerr`
        (its physics is not exchange-like), or either endpoint device exposes
        no ``'freq'`` tunable for the Lamb shift to land on.
    """
    coupling_label = resolve_label(target)
    coupling = chip.coupling_map[coupling_label]
    if not coupling.reduces_to_crosskerr:
        raise NotImplementedError(
            f"Coupling-target elimination reduces an exchange-like edge to a dressed CrossKerr "
            f"shift; '{coupling_label}' is a {type(coupling).__name__}, which does not declare "
            "reduces_to_crosskerr. To support it, declare reduces_to_crosskerr = True on the "
            "coupling class, only when its interaction genuinely reduces to a uniform dispersive "
            "pull."
        )
    a_label, b_label = coupling.device_a_label, coupling.device_b_label
    for endpoint_label in (a_label, b_label):
        if "freq" not in chip[endpoint_label].tunable_params():
            raise NotImplementedError(
                f"Coupling elimination Lamb-shifts both endpoint frequencies; '{endpoint_label}' "
                f"({type(chip[endpoint_label]).__name__}) exposes no 'freq' tunable (circuit-level "
                "devices expose E_C/E_J/n_g instead). Coupling-target elimination has no "
                "reduced-parameter slot for this endpoint."
            )

    result_kind = "crosskerr"
    equipment = chip.control_equipment

    def classify(line: Any) -> StrandedLine | None:
        if not (line.target_kind == "edge" and line._target.label == coupling_label):
            return None
        return StrandedLine(
            rule_target=coupling,
            missing_rule_message=(
                f"Control line '{line.label}' ({type(line).__name__}) pumps the eliminated "
                f"coupling '{coupling_label}'. No retarget rule converts it "
                f"(register_retarget_rule({type(line).__name__}, {type(coupling).__name__}, "
                f"'{result_kind}', ...)); unwire the line first (chip.unwire('{line.label}')), "
                "or keep the coupling."
            ),
        )

    survivor_lines, retarget_plan = plan_stranded_lines(equipment, classify, result_kind)

    dressed_a = chip.freq(a_label, when={b_label: 0})
    dressed_b = chip.freq(b_label, when={a_label: 0})
    lamb_a = dressed_a - cast(_HasFreq, chip[a_label]).freq
    lamb_b = dressed_b - cast(_HasFreq, chip[b_label]).freq
    chi = chip.dispersive_shift(a_label, b_label)

    reduced = chip.clone()
    kept_couplings = [c for c in reduced.couplings if c.label != coupling_label]
    reduced[a_label].set_tunable_param("freq", dressed_a)
    reduced[b_label].set_tunable_param("freq", dressed_b)
    crosskerr = CrossKerr(reduced[a_label], reduced[b_label], chi=chi, label=f"elim_{coupling_label}")
    kept_couplings.append(crosskerr)

    effective_params: dict[str, Any] = LabelKeyedDict({
        a_label: {"lamb_shift": lamb_a, "freq_after": dressed_a},
        b_label: {"lamb_shift": lamb_b, "freq_after": dressed_b},
    })
    delta = cast(_HasFreq, chip[a_label]).freq - cast(_HasFreq, chip[b_label]).freq
    g_over_delta = abs(coupling.coupling_strength / delta)
    validity: dict[str, Any] = LabelKeyedDict({
        coupling_label: {"g_over_delta": g_over_delta, "is_valid": g_over_delta < 0.1},
    })
    notes = [
        f"Coupling elimination: '{coupling_label}' replaced by CrossKerr('{crosskerr.label}') carrying "
        "the dressed pull chi = E11 - E10 - E01 + E00 (uniform-chi approximation; per-level chi "
        "differences and dispersive breakdown are not represented).",
        "Both endpoint devices survive with Lamb-shifted freq; control equipment passes through "
        "unchanged (no device was removed).",
    ]

    final = rebuild_chip(chip, devices=reduced.devices, couplings=kept_couplings)
    reattach_equipment(
        chip,
        final,
        equipment,
        survivor_lines,
        retarget_plan,
        mode_label=coupling_label,
        result_kind=result_kind,
        edges={(a_label, b_label): {"folded_into": crosskerr.label}},
        notes=notes,
    )
    detach_intermediate_clone(reduced.devices, reduced)
    return EliminationResult(chip=final, effective_params=effective_params, validity=validity, notes=notes)


register_elimination_target(EliminationTarget(
    kind="coupling",
    claims=lambda chip, target: resolve_label(target) in chip.coupling_map,
    reduce=reduce_coupling,
))
