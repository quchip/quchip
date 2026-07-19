"""Advisory solver-hint heuristics (post-assembly metadata only).

These helpers summarize an already-assembled
:class:`~quchip.engine.ir.HamiltonianDescription` into advisory hints
(``max_carrier_freq_ghz``, ``spectral_bound_ghz``) that backends may
consult to pick a conservative solver step. They live outside
``stage2_assembly`` because they assemble no terms and never cross the
2π boundary: the only :data:`~quchip.utils.constants.TWO_PI` here divides
an already-angular carrier back to ordinary GHz for the advisory dict.
None of these values participate in physics — they are pure metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quchip.engine.ir import (
    Carrier,
    DynamicTerm,
    ScalarModulation,
    SignalProgram,
    StaticTerm,
    Window,
    signal_children,
)
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar

# Above this Hilbert-space dimension, _static_diagonal_span skips its dense
# diagonal materialization: the advisory hint is not worth an O(dim^2) dense
# conversion.
_MAX_SPECTRAL_HINT_DIM = 2048


def _collect_carrier_freqs_from_signal(signal: SignalProgram) -> list[float]:
    """Recursively gather absolute carrier frequencies (rad/ns) from a signal AST.

    Walks the AST via :func:`~quchip.engine.ir.signal_children`, the single
    structural source of truth for node-shape. Traced carrier frequencies
    are skipped because this feeds advisory solver metadata only; forcing
    them through Python ``max(...)`` would break JAX tracing without
    changing physics.
    """
    if isinstance(signal, Carrier):
        freq = maybe_concrete_scalar(signal.freq)
        return [] if freq is None else [abs(freq)]
    return [f for child in signal_children(signal) for f in _collect_carrier_freqs_from_signal(child)]


def _max_abs_carrier_freq(dynamic_terms: tuple[DynamicTerm, ...]) -> float | None:
    """Maximum absolute carrier frequency (rad/ns) across *dynamic_terms*, or ``None`` if none.

    Used as an advisory hint so backends can pick a conservative solver
    step; never participates in physics.
    :func:`_collect_carrier_freqs_from_signal` covers why ``max(...)`` is
    safe on the gathered list.
    """
    all_freqs: list[float] = []
    for term in dynamic_terms:
        if isinstance(term.time_dependence, ScalarModulation):
            all_freqs.extend(_collect_carrier_freqs_from_signal(term.time_dependence.signal))
    return max(all_freqs) if all_freqs else None


# Backends may consult max_step_ns to bound the integrator step so a
# finite-support pulse cannot be silently skipped by an adaptive step that
# grows through an idle span; this follows QuTiP's guidance that max_step
# be at most roughly half the thinnest pulse width.
_MAX_STEP_WIDTH_FACTOR = 0.5


def _collect_window_widths_from_signal(signal: SignalProgram) -> tuple[list[float], bool]:
    """Recursively gather ``Window`` widths (ns) and whether any bound is traced.

    Walks the AST via :func:`~quchip.engine.ir.signal_children`, mirroring
    :func:`_collect_carrier_freqs_from_signal`. ``Window.start``/``stop``
    are local to the node -- an enclosing
    :class:`~quchip.engine.ir.Shift` only translates the pulse in time and
    never changes its width, so no shift correction is needed here.
    Returns ``(widths, traced)``; *traced* is ``True`` when any window
    bound anywhere in the subtree is a JAX tracer, in which case *widths*
    must be treated as incomplete by the caller.
    """
    widths: list[float] = []
    traced = False
    if isinstance(signal, Window):
        if contains_tracer((signal.start, signal.stop)):
            traced = True
        else:
            width = signal.stop - signal.start
            if width > 0:
                widths.append(width)
    for child in signal_children(signal):
        child_widths, child_traced = _collect_window_widths_from_signal(child)
        widths.extend(child_widths)
        traced = traced or child_traced
    return widths, traced


def _min_positive_window_width(dynamic_terms: tuple[DynamicTerm, ...]) -> float | None:
    """Narrowest positive concrete ``Window`` width (ns) across *dynamic_terms*, or ``None``.

    Returns ``None`` when no term carries a positive-width window, or when
    any window bound anywhere in the term set is traced -- an incomplete
    ceiling is worse than none for a solver step-size hint.
    """
    all_widths: list[float] = []
    for term in dynamic_terms:
        if not isinstance(term.time_dependence, ScalarModulation):
            continue
        widths, traced = _collect_window_widths_from_signal(term.time_dependence.signal)
        if traced:
            return None
        all_widths.extend(widths)
    return min(all_widths) if all_widths else None


def _static_diagonal_span(static_terms: tuple[StaticTerm, ...]) -> float | None:
    """Spectral-bound hint ``max(diag) - min(diag)`` for the combined static Hamiltonian.

    Returns ``None`` when empty, oversized, or not fully concrete (a traced
    coefficient must stay dynamic — no ``float()`` forced concretization).
    """
    if not static_terms:
        return None
    dim = static_terms[0].operator.shape[0]
    if dim > _MAX_SPECTRAL_HINT_DIM:
        return None
    combined = np.zeros(dim, dtype=float)
    for term in static_terms:
        coeff = maybe_concrete_scalar(term.coefficient)
        if coeff is None:
            return None
        operator_payload = (
            term.operator.values,
            term.operator.indices,
            term.operator.indptr,
            term.operator.offsets,
        )
        if contains_tracer(operator_payload):
            return None
        combined += np.diag(np.real(np.asarray(term.operator.to_dense(), dtype=complex))) * coeff
    span = maybe_concrete_scalar(np.max(combined) - np.min(combined))
    if span is None or span <= 0:
        return None
    return span


def _solver_hint_metadata(
    static_spectral_bound_ghz: float | None,
    dynamic_terms: tuple[DynamicTerm, ...],
) -> dict[str, Any]:
    """Advisory hints ``max_carrier_freq_ghz``, ``spectral_bound_ghz``, and ``max_step_ns``.

    ``max_carrier_freq_ghz`` / ``spectral_bound_ghz`` are ordinary GHz. The
    static spectral bound is template-invariant, so it is precomputed once
    at template compile (:func:`~quchip.engine.stage2_assembly.compile_hamiltonian_template`)
    and passed in here already in ordinary GHz — only the variant-specific
    carrier frequency is recomputed per instantiation. Step-budget
    heuristics take the larger of the two: in a rotating frame the static
    spectral span is near-zero while the inter-mode / drive carriers
    dominate the fastest oscillation.

    ``max_step_ns`` is advisory only and never participates in physics; a
    backend may consult it to cap its integrator step so a finite-support
    pulse narrower than the natural step size is not silently skipped.
    Omitted whenever no dynamic term carries a positive-width concrete
    ``Window`` (none present, or any window bound traced).
    """
    metadata: dict[str, Any] = {}

    max_carrier = _max_abs_carrier_freq(dynamic_terms)
    if max_carrier is not None:
        metadata["max_carrier_freq_ghz"] = max_carrier / TWO_PI

    if static_spectral_bound_ghz is not None:
        metadata["spectral_bound_ghz"] = static_spectral_bound_ghz

    min_width = _min_positive_window_width(dynamic_terms)
    if min_width is not None:
        metadata["max_step_ns"] = _MAX_STEP_WIDTH_FACTOR * min_width

    return metadata
