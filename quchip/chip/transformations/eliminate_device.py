"""Device-target elimination: adiabatic reduction of a far-detuned mode.

A mode touching **one** survivor (leaf) folds into a Lamb shift, plus a
Purcell channel when the mode dissipates. A mode touching **two or more**
survivors — bus / tunable-coupler (bridge) or several at once — additionally
induces a mediated exchange ``J = g_a g_b / 2 · (1/Δ_a + 1/Δ_b)`` between every
survivor pair, folded into the direct coupling between that pair when one
exists or added as its own edge otherwise. A fixed eliminated mode emits a
:class:`~quchip.chip.couplings.Capacitive`; a frequency-controlled mode (or an
already-modulable direct edge) emits a
:class:`~quchip.chip.couplings.TunableCapacitive`.

The reduction route (``method="sw"`` / ``method="exact"``) is a
:class:`~quchip.chip.transformations.methods.ReductionMethod` strategy; the
generic P/Q partitioning kernels live in :mod:`quchip.chip.sw`. This module
owns the fold — reading a route's reduced parameters into a rebuilt chip and
retargeting stranded control lines — and registers a device-kind
:class:`~quchip.chip.transformations.dispatch.EliminationTarget` at import time,
so :func:`~quchip.chip.transformations.dispatch.eliminate` dispatches any device
label here without importing this module directly.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp

from quchip.backend import _backend_context
from quchip.chip.couplings import Capacitive, TunableCapacitive
from quchip.chip.rwa import apply_rwa_mask
from quchip.chip.sw import (
    bare_hamiltonian,
    mode_blocks,
    purcell_rate_from,
    sylvester_generator,
)
from quchip.chip.transformations.dispatch import EliminationTarget, register_elimination_target
from quchip.chip.transformations.methods import DeviceReductionContext, lookup_reduction_method
from quchip.chip.transformations.plumbing import (
    StrandedLine,
    detach_intermediate_clone,
    plan_stranded_lines,
    reattach_equipment,
    rebuild_chip,
)
from quchip.chip.transformations.result import EliminationResult, LazyEffectiveParams, _HasFreq
from quchip.control.drive import FluxDrive
from quchip.devices.protocols import FrequencyControlled
from quchip.utils.labeling import LabelKeyedDict, resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


def mode_decay_rate(mode: Any) -> tuple[Any, bool]:
    """``(kappa, has_purcell)``: the eliminated mode's own decay rate, and whether it decays at all.

    Reads :meth:`~quchip.devices.base.BaseDevice.intrinsic_decay_rate`, which
    each device class owns — e.g. :class:`~quchip.devices.resonator.Resonator`
    combines its Q-derived photon loss with any ``T1``, matching its actual
    :meth:`~quchip.devices.resonator.Resonator.collapse_operators`. Whether a
    channel exists is a *static* decision (does the hook return ``None``?),
    never a traced-zero comparison on the resulting rate, which would
    concretize a traced value and break differentiability.
    """
    rate = mode.intrinsic_decay_rate()
    if rate is None:
        return 0.0, False
    return rate, True


def _exchange_matrix_element(coupling: Any, row_label: str, chip: "Chip") -> Any:
    """One-excitation exchange matrix element of *coupling* as :meth:`Chip.hamiltonian` assembles it.

    Reads ``<1_row,0_other|H_int|0_row,1_other>`` off the *same* interaction form
    :meth:`~quchip.chip.chip.Chip.hamiltonian` embeds for this coupling: the full
    :meth:`~quchip.chip.coupling_base.BaseCoupling.interaction_hamiltonian` when the chip does not
    resolve RWA for it, or the :func:`~quchip.chip.rwa.apply_rwa_mask`-filtered interaction
    (respecting the coupling's own
    :meth:`~quchip.chip.coupling_base.BaseCoupling.rwa_keeps_band`) when it does. A coupling whose
    resolved RWA rejects the exchange band therefore correctly reports zero here, matching what
    ``total_coupling`` — read off the assembled, already-RWA-resolved bare Hamiltonian — actually
    contains for it; reading the unconditional full interaction would silently subtract exchange
    that was never in ``total_coupling`` to begin with. Row/column convention matches
    :func:`~quchip.chip.sw.extract_pair_parameters`'s ``("J", a, b)`` P-block entry. This is the
    physically correct measure of a coupling's exchange contribution — not its declared
    :attr:`~quchip.chip.coupling_base.BaseCoupling.coupling_strength`, which need not equal this
    matrix element for a coupling that is not dipole-dipole exchange-compatible (e.g. a
    dispersive ``g · n_a n_b`` interaction has no off-diagonal exchange element at all).
    """
    backend = chip.backend
    with _backend_context(backend):
        h_int = coupling.interaction_hamiltonian()
        if chip.resolve_rwa(coupling):
            h_int = apply_rwa_mask(
                h_int,
                dims=(coupling.device_a.levels, coupling.device_b.levels),
                labels=(coupling.device_a_label, coupling.device_b_label),
                keeps_band=coupling.rwa_keeps_band,
                backend=backend,
            )
            if h_int is None:
                return jnp.asarray(0.0, dtype=complex)
        h_int_array = jnp.asarray(backend.to_array(h_int), dtype=complex)
    dim_b = coupling.device_b.levels
    element = h_int_array[dim_b, 1]  # <device_a=1,device_b=0|H_int|device_a=0,device_b=1>, C-order flat index
    return element if coupling.device_a_label == row_label else jnp.conj(element)


def reduce_device(chip: "Chip", target: Any, method: str) -> EliminationResult:
    """Adiabatically eliminate a far-detuned device, folding its effect into the survivors.

    The registry guarantees ``target`` names a device on ``chip``, and
    :func:`eliminate` has already validated ``method``. See
    :func:`~quchip.chip.transformations.dispatch.eliminate` for the full
    physics contract and the ``method`` semantics.

    Parameters
    ----------
    chip
        Source chip (never mutated).
    target
        The device to eliminate — label string or object.
    method
        The reduction route (``"sw"`` or ``"exact"``), already validated.

    Returns
    -------
    EliminationResult
    """
    mode_label = resolve_label(target)
    touching = [c for c in chip.couplings if mode_label in (c.device_a_label, c.device_b_label)]
    survivors = []
    for c in touching:
        other = c.device_b_label if c.device_a_label == mode_label else c.device_a_label
        survivors.append((other, c))
    is_multi = len(survivors) >= 2

    # A bath that *explicitly* targets the eliminated mode would dangle after
    # reduction and raise KeyError at solve time — fail fast and clearly here.
    # Baths with default (None) targets re-resolve against the survivors, so
    # they are fine.
    for bath in chip.baths:
        if bath._targets is not None and mode_label in {resolve_label(t) for t in bath._targets}:
            raise ValueError(
                f"Cannot eliminate '{mode_label}': bath '{bath.label}' explicitly targets it. "
                "Remove the eliminated mode from the bath's targets first."
            )

    # A control line wired to the eliminated mode (or to a coupling that
    # touches it) has no image in the reduced model unless a registered rule
    # converts it (chip/retarget.py). plan_stranded_lines() looks the rules up
    # *before* any fold work, so a missing rule fails fast — exactly like the
    # bath guard above.
    result_kind = "edge" if is_multi else "leaf-fold"
    equipment = chip.control_equipment
    mode_device: Any = chip[mode_label]

    def classify(line: Any) -> StrandedLine | None:
        rule_target: Any
        if line.target_kind == "edge":
            edge_coupling: Any = line._target
            doomed = mode_label in (edge_coupling.device_a_label, edge_coupling.device_b_label)
            rule_target = edge_coupling
            what = f"coupling '{edge_coupling.label}' (touches mode '{mode_label}')"
        else:
            doomed = line._target is not None and line._target.label == mode_label
            rule_target = mode_device
            what = f"mode '{mode_label}'"
        if not doomed:
            return None
        return StrandedLine(
            rule_target=rule_target,
            missing_rule_message=(
                f"Control line '{line.label}' ({type(line).__name__}) targets the "
                f"eliminated {what}. No retarget rule converts it "
                f"(register_retarget_rule({type(line).__name__}, {type(rule_target).__name__}, "
                f"'{result_kind}', ...)); unwire the line first (chip.unwire('{line.label}')), "
                "or keep the target."
            ),
        )

    survivor_lines, retarget_plan = plan_stranded_lines(equipment, classify, result_kind)

    reduction = lookup_reduction_method(method)
    assert reduction is not None  # method membership checked above; keeps mypy's Optional narrow

    effective_params: dict[str, Any] = LabelKeyedDict()
    validity: dict[str, Any] = LabelKeyedDict()
    # bare_hamiltonian() assembles chip.hamiltonian() at the chip's resolved
    # RWA policy, so counter-rotating terms are actually dropped only when at
    # least one touching coupling resolves RWA True — never claim the drop
    # unconditionally.
    dropped_items = ["ring-up transients"]
    if any(chip.resolve_rwa(c) for _, c in survivors):
        dropped_items.insert(0, "counter-rotating terms")
    notes = [
        f"Adiabatic elimination (method='{method}'): steady-state (vacuum) reduction.",
        "Dropped: " + ", ".join(dropped_items) + reduction.dropped_suffix(),
    ]

    # Build the reduced chip from cloned survivors (mode + its couplings removed).
    survivor_labels = [d.label for d in chip.devices if d.label != mode_label]
    reduced = chip.clone()
    kept_couplings = [c for c in reduced.couplings if mode_label not in (c.device_a_label, c.device_b_label)]
    reduced_devices = [reduced[lbl] for lbl in survivor_labels]

    mode = chip[mode_label]
    mode_is_frequency_controlled = isinstance(mode, FrequencyControlled) or any(
        isinstance(line, FluxDrive) for line, _ in retarget_plan
    )
    h, labels, dims = bare_hamiltonian(chip, chip.backend)
    # Survivor pairs are keyed in the chip's device order everywhere — the
    # pair extraction, the exact route, and the fold loop below — so the two
    # sides of every ("J", a, b) lookup agree no matter what order the legs
    # were scanned in. Coupling-scan order and device order genuinely differ
    # on real chips (a center mode declared before its outer neighbors).
    scanned = {lbl for lbl, _ in survivors}
    touching_labels = [lbl for lbl in labels if lbl in scanned]

    p_mask, _ = mode_blocks(dims, labels, mode_label)
    s, min_gap = sylvester_generator(h, p_mask)

    ctx = DeviceReductionContext(
        chip=chip,
        mode=mode,
        mode_label=mode_label,
        survivor_labels=touching_labels,
        labels=labels,
        dims=dims,
        h=h,
        s=s,
        p_mask=p_mask,
    )
    pair_params = reduction.pair_parameters(ctx)

    kappa, has_purcell = mode_decay_rate(mode)
    amplitudes = reduction.survivor_amplitudes(ctx) if has_purcell else {}
    if getattr(mode, "thermal_population", None) is not None:
        # The Purcell fold below carries only the mode's lowering-operator
        # (downward) rate into the survivor's T1; thermal_population's
        # upward/absorption channel has no representation in the reduced
        # chip's collapse operators — declared explicitly rather than
        # silently dropped.
        notes.append(
            f"Mode '{mode_label}' carries thermal_population (an upward absorption channel); "
            "the Purcell fold only carries its lowering-operator (downward) decay rate into the "
            "survivor's T1 — thermal absorption is not represented in the reduced chip."
        )

    for survivor_label, coupling in survivors:
        freq_after = jnp.real(pair_params[survivor_label]["freq_after"])
        lamb_shift = freq_after - cast(_HasFreq, chip[survivor_label]).freq
        purcell_rate = purcell_rate_from(amplitudes[survivor_label], kappa) if has_purcell else 0.0
        g = coupling.coupling_strength
        delta = cast(_HasFreq, chip[survivor_label]).freq - cast(_HasFreq, mode).freq
        g_over_delta = abs(g / delta)

        dev = reduced[survivor_label]
        if "freq" in dev.tunable_params():
            dev.set_tunable_param("freq", freq_after)
        else:
            # Circuit-level survivors (E_C/E_J/n_g tunables) expose no bare
            # frequency to absorb the Lamb shift into, and inverting it
            # through the circuit parameters would silently reshape the
            # anharmonicity. The reduced device keeps its bare spectrum; the
            # shifted frequency stays reported, not folded.
            warnings.warn(
                f"Survivor '{survivor_label}' ({type(dev).__name__}) exposes no 'freq' tunable; the Lamb "
                "shift from this elimination is reported in effective_params ('freq_after', 'lamb_shift') "
                "but not folded into the reduced device's bare parameters.",
                UserWarning,
                stacklevel=2,
            )
        # Whether a T1 channel exists is decided from *static* info only (an
        # intrinsic T1, or a Purcell channel from the mode's own decay) —
        # never by comparing the traced rate to zero, which would break
        # @jax.jit.
        t1 = dev.T1
        has_intrinsic_t1 = t1 is not None
        if has_purcell and dev.thermal_population is not None:
            # The thermal-emission NoiseChannel scales both the downward
            # ((n̄+1)/T1) and upward (n̄/T1) rates off the same T1
            # coefficient, but the inherited Purcell channel is pure
            # lowering (the eliminated mode's own decay, folded through a
            # vacuum-bath-like coupling — see mode_decay_rate()). Bumping
            # 1/T1 by purcell_rate would therefore inflate the downward rate
            # by (n̄+1)·purcell_rate *and* invent an upward absorption
            # channel of n̄·purcell_rate with no physical basis. No device
            # API declares an independent pure-lowering collapse channel
            # (BaseDevice._noise_channels is class-level; per-instance
            # extension is rejected — see BaseDevice.__setattr__) to carry
            # this rate separately, so fail fast rather than mis-model it.
            raise NotImplementedError(
                f"Purcell fold onto survivor '{survivor_label}' would scale the shared T1 "
                f"coefficient, but '{survivor_label}' also carries thermal_population: the pure "
                "lowering Purcell channel cannot be represented by scaling T1 without inventing "
                "thermal absorption that was never physically present. Unset thermal_population "
                f"on '{survivor_label}' before eliminating '{mode_label}', or keep the mode."
            )
        if has_intrinsic_t1 or has_purcell:
            rate_before = 1.0 / t1 if t1 is not None else 0.0
            dev.T1 = 1.0 / (rate_before + purcell_rate)

        chi_value: Any
        if is_multi:
            # A mode coupling several survivors is a bus/coupler, not a readout mode.
            chi_value = 0.0
        else:
            def _chi(mode_label: str = mode_label, survivor_label: str = survivor_label) -> Any:
                return chip.dispersive_shift(mode_label, survivor_label)

            chi_value = _chi
        effective_params[survivor_label] = LazyEffectiveParams({
            "lamb_shift": lamb_shift,
            "purcell_rate": purcell_rate,
            "freq_after": freq_after,
            "chi": chi_value,
            "kappa": kappa,
        })
        validity[coupling.label] = {
            "g_over_delta": g_over_delta,
            "is_valid": g_over_delta < 0.1,
            "min_block_gap": min_gap,
        }

    exchange_by_pair: dict[tuple[str, str], dict[str, Any]] = LabelKeyedDict()
    if is_multi:
        # Mediated survivor-survivor exchange, 2nd order in the leg strengths:
        # J = (g_a g_b / 2) (1/Δ_a + 1/Δ_b), Δ_i = ω_i − ω_mode (RWA exchange
        # form; the standard bus / tunable-coupler result — F. Yan et al.,
        # Phys. Rev. Applied 10, 054062 (2018)), read off the pair extraction
        # above. Every survivor pair gets its own mediated exchange, folded
        # into an existing direct coupling between that pair when one
        # exists, else added as its own edge, so the reduced chip's net
        # coupling is what the survivors actually feel. Tunability is kept
        # only when the eliminated mode declares frequency control (or has a
        # flux line being retargeted), or when the direct edge was already
        # modulable; a fixed resonator does not create a control knob.
        leg_g = {lbl: c.coupling_strength for lbl, c in survivors}
        mode_freq = cast(_HasFreq, mode).freq
        leg_delta = {lbl: cast(_HasFreq, chip[lbl]).freq - mode_freq for lbl in touching_labels}

        pairs = list(combinations(touching_labels, 2))
        single_pair = len(pairs) == 1
        for label_a, label_b in pairs:
            # ``pair_params[("J", a, b)]`` is read off H_eff = P(H + correction)P,
            # so it already reports the *total* post-reduction coupling: any
            # pre-existing direct edge(s) between this pair are baked into the
            # bare H term and carry straight through the projection, on top of
            # the mode-mediated correction. See the edge_strength/base_all
            # accounting below for how that total is split back out across the
            # reduced chip's edges without double-counting.
            total_coupling = jnp.real(pair_params[("J", label_a, label_b)])
            dj_domega_c = (
                leg_g[label_a] * leg_g[label_b] / 2.0
                * (1.0 / leg_delta[label_a] ** 2 + 1.0 / leg_delta[label_b] ** 2)
            )
            zz = reduction.residual_zz(ctx, pair_params, label_a, label_b)
            pathways = reduction.pathways(ctx, pair_params, label_a, label_b)

            fresh_label = f"elim_{mode_label}" if single_pair else f"elim_{mode_label}_{label_a}_{label_b}"
            pair = {label_a, label_b}
            pair_edges = [c for c in kept_couplings if {c.device_a_label, c.device_b_label} == pair]
            # folds_exchange (a prior fold may already be a TunableCapacitive,
            # or the user built one directly) is a physics capability, not a
            # duck-typed "g"/"g_0" attribute check — a coupling that carries a
            # scalar strength but is not dipole-dipole exchange-compatible
            # (e.g. a dispersive g·n_a·n_b interaction) must not silently
            # absorb a mediated exchange term.
            direct = next((c for c in pair_edges if c.folds_exchange), None)
            others = [c for c in pair_edges if c is not direct]

            # total_coupling already includes every pair_edges member's own
            # contribution (each is baked into the bare Hamiltonian the pair
            # extraction reads), at whatever RWA the chip actually resolves
            # for it — so every edge's contribution here is read the same
            # way, through the resolved-RWA interaction, not the raw full
            # one (a coupling whose resolved RWA rejects the exchange band
            # contributes zero to total_coupling and must contribute zero
            # here too). "others_total" is what every edge *other than* the
            # fold target already carries; the fold target (or, with no
            # fold target, the new parallel edge) must carry exactly
            # total_coupling minus that, so the reduced chip's edges sum to
            # total_coupling with nothing counted twice.
            others_total = jnp.asarray(0.0, dtype=complex)
            for edge in others:
                others_total = others_total + _exchange_matrix_element(edge, label_a, chip)
            others_total = jnp.real(others_total)
            direct_element = (
                jnp.real(_exchange_matrix_element(direct, label_a, chip)) if direct is not None else 0.0
            )
            base_all = others_total + direct_element
            edge_strength = total_coupling - others_total

            if direct is not None:
                already_tunable = isinstance(direct, TunableCapacitive)
                keep_tunable = mode_is_frequency_controlled or already_tunable
                replacement: Any
                if keep_tunable:
                    replacement = TunableCapacitive(
                        reduced[label_a], reduced[label_b],
                        g_0=edge_strength, rwa=direct.rwa, label=direct.label,
                    )
                else:
                    replacement = Capacitive(
                        reduced[label_a], reduced[label_b],
                        g=edge_strength, rwa=direct.rwa, label=direct.label,
                    )
                kept_couplings[kept_couplings.index(direct)] = replacement
                folded_into = replacement.label
                if already_tunable:
                    notes.append(
                        f"Direct coupling '{direct.label}' (TunableCapacitive): mediated exchange "
                        "folded into its existing g_0."
                    )
                elif mode_is_frequency_controlled:
                    notes.append(
                        f"Direct coupling '{direct.label}' ({type(direct).__name__}) upgraded to "
                        "TunableCapacitive (net g_0 folds the mediated exchange)."
                    )
                else:
                    notes.append(
                        f"Direct coupling '{direct.label}' ({type(direct).__name__}): mediated exchange "
                        "folded into its static g."
                    )
                if others:
                    other_labels = ", ".join(f"'{edge.label}'" for edge in others)
                    notes.append(
                        f"Other preserved coupling(s) {other_labels} between '{label_a}' and "
                        f"'{label_b}' also carry exchange for this pair; subtracted out of "
                        f"'{direct.label}''s replacement strength to avoid double-counting."
                    )
            else:
                mediated: Any
                if mode_is_frequency_controlled:
                    mediated = TunableCapacitive(
                        reduced[label_a], reduced[label_b], g_0=edge_strength, label=fresh_label
                    )
                else:
                    mediated = Capacitive(
                        reduced[label_a], reduced[label_b], g=edge_strength, label=fresh_label
                    )
                kept_couplings.append(mediated)
                folded_into = mediated.label
                if others:
                    other_labels = ", ".join(f"'{edge.label}'" for edge in others)
                    notes.append(
                        f"Preserved coupling(s) {other_labels} between '{label_a}' and "
                        f"'{label_b}' do not declare folds_exchange; kept unchanged, with the "
                        f"mediated exchange added as a parallel edge '{fresh_label}' carrying "
                        "only the differential (no double count)."
                    )

            exchange_by_pair[(label_a, label_b)] = {
                "j_eff": total_coupling - base_all,
                "dJ_domega_c": dj_domega_c,
                "between": (label_a, label_b),
                "folded_into": folded_into,
                "zz": zz,
                "pathways": pathways,
            }

        notes.append(
            "Mediated exchange J = g_a*g_b/2*(1/Δ_a + 1/Δ_b) per survivor pair; mediated terms "
            "beyond exchange (e.g. coupler-induced ZZ) are a higher-order correction under "
            "method='sw' (available exactly as 'zz' under method='exact')."
        )
        effective_params["exchange"] = (
            next(iter(exchange_by_pair.values())) if single_pair else exchange_by_pair
        )

    final = rebuild_chip(chip, devices=reduced_devices, couplings=kept_couplings)
    reattach_equipment(
        chip,
        final,
        equipment,
        survivor_lines,
        retarget_plan,
        mode_label=mode_label,
        result_kind=result_kind,
        edges=exchange_by_pair if is_multi else None,
        notes=notes,
    )
    detach_intermediate_clone(reduced_devices, reduced)
    return EliminationResult(chip=final, effective_params=effective_params, validity=validity, notes=notes)


register_elimination_target(EliminationTarget(
    kind="device",
    claims=lambda chip, target: resolve_label(target) in chip.device_map,
    reduce=reduce_device,
))
