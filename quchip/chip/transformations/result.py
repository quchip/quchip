"""Result and typing layer for :mod:`quchip.chip.transformations`.

:class:`ChipTransform` is the structural protocol every transformation result
satisfies; :class:`EliminationResult` is :func:`~quchip.chip.transformations.dispatch.eliminate`'s
return type, and :class:`LazyEffectiveParams` is the deferred-``chi``
dict it stores its per-survivor entries in. This module holds no dispatch
logic and imports nothing else from the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from quchip.chip.chip import Chip


@runtime_checkable
class ChipTransform(Protocol):
    """Structural protocol for any transformation result that yields a chip."""

    @property
    def chip(self) -> "Chip": ...


class _HasFreq(Protocol):
    """Typing-only view of the ``freq`` attribute every concrete BaseDevice subclass
    declares (devices/base.py contract point 2), but BaseDevice itself does not."""

    freq: Any


class _HasG(Protocol):
    """Typing-only view of the scalar coupling strength every Capacitive-like
    coupling exposes; not declared on BaseCoupling itself."""

    g: Any


@dataclass(frozen=True)
class EliminationResult:
    """Store the result of :func:`eliminate`.

    Attributes
    ----------
    chip
        The reduced chip (eliminated mode removed; effect folded into ordinary
        surviving devices). The input chip is never mutated.
    effective_params
        Derived quantities per survivor:
        ``{label: {"lamb_shift", "purcell_rate", "freq_after", "chi", "kappa"}}``
        (GHz for frequencies, 1/ns for rates). ``chi`` is the dispersive pull
        ``χ_pull ≡ f_mode(survivor in |1⟩) − f_mode(survivor in |0⟩)`` — the
        *full* resonator pull per survivor excitation, i.e. **2×** the
        σ_z-convention χ of ``H_disp = (ω_r + χσ_z)a†a``. For a mode touching
        more than one survivor, the mode is a bus / coupler,
        not a readout mode, so ``chi`` is reported as ``0.0`` for every
        survivor. Otherwise ``chi`` is a *deferred* entry: computed from the
        input chip's dressed spectrum (one full-chip diagonalization) on the
        first ``["chi"]`` access and cached, so reductions that never read it
        stay pure algebra (:class:`LazyEffectiveParams`); gradients through
        it follow ``Chip.dispersive_shift``'s backend rule, while every other
        entry remains backend-independent algebra. ``kappa`` is the
        eliminated mode's own decay rate, from its
        :meth:`~quchip.devices.base.BaseDevice.intrinsic_decay_rate` (e.g. a
        resonator's combined ``2π·f_mode/Q`` photon loss plus ``1/T1``), or
        ``0.0`` when the mode declares no decay channel.

        A mode touching **two or more** survivors additionally emits an
        ``"exchange"`` entry describing the mediated survivor-survivor
        coupling(s) derived from the Sylvester/exact pair extraction
        (:mod:`quchip.chip.sw`). For exactly two survivors (the bridge case)
        this is a single dict: ``{"j_eff", "dJ_domega_c", "between",
        "folded_into", "zz", "pathways"}``. For three or more survivors,
        every survivor pair induces its own mediated exchange, so
        ``"exchange"`` is instead a ``dict`` keyed by the pair
        ``(label_a, label_b)``, each value the same per-pair schema. In both
        cases: ``j_eff`` is the mediated exchange ``J = g_a g_b / 2 · (1/Δ_a +
        1/Δ_b)`` (F. Yan et al., Phys. Rev. Applied 10, 054062 (2018)), read
        off the ``method="sw"``/``"exact"`` pair extraction; ``dJ_domega_c``
        is its analytic derivative w.r.t. the eliminated mode's frequency
        (used by the flux-drive retarget rule); ``folded_into`` is the label
        of the effective edge the exchange landed on (an existing direct
        coupling between the pair, or a freshly emitted
        :class:`~quchip.chip.couplings.Capacitive` or
        :class:`~quchip.chip.couplings.TunableCapacitive`); ``zz`` is the
        exact residual ZZ between the pair (``method="exact"`` only —
        ``None`` under ``"sw"``, where it is
        a higher-order correction not represented); ``pathways`` is the
        top virtual-state attribution of the exchange (``method="sw"`` only,
        from :func:`quchip.chip.sw.pathway_attribution` — ``None`` under
        ``"exact"``, which has no perturbative generator to attribute).
    validity
        Per eliminated coupling:
        ``{coupling_label: {"g_over_delta", "is_valid", "min_block_gap"}}``.
        ``min_block_gap`` is the smallest bare-energy gap the Sylvester
        generator crossed for this mode (:func:`quchip.chip.sw.sylvester_generator`),
        shared across every coupling touching the mode. When ``eliminate``
        runs under ``jax.jit``/``grad``, ``is_valid`` (``g_over_delta < 0.1``)
        is a *traced* boolean, not a Python ``bool`` — read it outside the
        traced region, or branch on it with ``jnp.where`` rather than ``if``.
    notes
        Explicitly dropped physics (counter-rotating, transients, higher order).

    Every mapping above is a :class:`~quchip.utils.labeling.LabelKeyedDict`:
    the device or coupling *object* is as good a key as its label, and pair
    keys match in either order —
    ``res.effective_params[q1]``, ``res.validity[leg]``, and
    ``exchange[(q1, q0)]`` all resolve.
    """

    chip: "Chip"
    effective_params: dict[str, Any]
    validity: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Plain-text fold report: every fold stated explicitly, before -> after.

        Per survivor: freq (and T1 when either side carries one), each
        Lamb-shifted/Purcell-folded value read back from :attr:`effective_params`
        rather than a stored "before" chip — ``freq_before = freq_after -
        lamb_shift`` and ``T1_before`` from ``purcell_rate`` are exact
        identities of how :func:`eliminate` derives ``freq_after``/``T1``, not
        an approximation. Multi-survivor targets add the emitted exchange edge
        (Yan-formula tag) and a ZZ line (placeholder under ``method="sw"``,
        the exact residual under ``method="exact"``); any control-line
        retarget and the per-coupling validity verdict follow. Traced
        parameters render as ``<traced>`` and are never concretized; for use
        outside ``jit``/``grad`` regions, like every other ``describe()`` in
        the package.
        """
        from quchip.chip.describe import describe_elimination

        return describe_elimination(self)


class LazyEffectiveParams(dict):
    """Per-survivor effective parameters with deferred entries.

    A value stored as a zero-argument callable (``chi``, which costs a
    full-chip diagonalization) is evaluated and cached on first access, so
    ``eliminate()`` itself stays pure algebra for callers that never read it,
    with no overhead on paths that don't need it. Plain values behave
    exactly like normal dict entries.
    """

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        if callable(value):
            value = value()
            super().__setitem__(key, value)
        return value

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def items(self) -> Any:
        return [(key, self[key]) for key in self]

    def values(self) -> Any:
        return [self[key] for key in self]

    def __repr__(self) -> str:
        shown = {key: ("<deferred>" if callable(value) else value) for key, value in dict.items(self)}
        return repr(shown)
