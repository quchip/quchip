"""Choose how stationary and static calculations solve, and record the reason.

A stationary calculation solves on one of three kinds of space. A passive
linear chip, or a conserving chip under an infinitesimal probe, keeps its mode
or one-excitation equations. A conserving chip under a finite drive keeps the
product labels with total excitation ``N <= K``. Every other calculation uses
the backend's stationary Liouvillian on the product space, or on the retained
labels of an excitation cutoff. Static queries diagonalize the blocks of a
conserving chip or of an excitation cutoff, and otherwise the full static
model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import prod
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from quchip.utils.jax_utils import contains_tracer
from quchip.utils.values import DeferredValue
from quchip.utils.stack import user_stacklevel
from quchip.utils.warning_categories import AutoChoiceWarning

# Above this many product states, a conserving chip without a cutoff solves on sectors.
AUTOMATIC_SECTOR_STATES = 64
# The automatic cutoff starts here and compares adjacent cutoffs.
FIRST_CUTOFF = 2
# Relative change allowed between K and K + 1, and the floor as a fraction of each quantity's largest value.
CUTOFF_TOLERANCE = 1e-6
CUTOFF_FLOOR = 1e-9

_LIOUVILLIAN = "stationary_resolvent"
_TRACED = "The values are traced, and the sector solve runs on host NumPy arrays."
WEAK_PROBE: Mapping[str, str] = MappingProxyType({
    "linear_response": "The chip is passive and harmonic, so its mode equations give the small-signal response.",
    "vacuum_response": ("The static model conserves N and every channel lowers or conserves it, so the vacuum is "
                        "stationary and a weak probe keeps the vacuum and the one-excitation states."),
})


@dataclass(frozen=True)
class SolveRoute:
    """The numerical route of one stationary or static calculation and the reason for it.

    Attributes
    ----------
    name : str
        ``"linear_response"`` for passive-linear mode equations,
        ``"vacuum_response"`` for the vacuum and one-excitation states under a
        weak probe, ``"conditional_response"`` for a frozen dressed reference,
        ``"sector_stationary"`` for the excitation sectors ``N <= K``, and
        ``"stationary_resolvent"`` for the backend's stationary Liouvillian on
        the product space or on the retained labels of an excitation cutoff.
        Static queries use ``"sector_eigensystem"`` or ``"full_eigensystem"``.
    reason : str
        Why the planner chose the route.
    dimension : int
        Number of modes or states that the route solves on.
    arrays : str
        Array library of the solve, ``"numpy"`` or ``"jax"``.
    peak_bytes : int or None
        Estimated peak memory of the solve, calculated on request. ``None``
        means that the route has no estimate.
    """

    name: str
    reason: str
    dimension: int
    arrays: str
    _peak_bytes: Any = field(default=None, repr=False, compare=False)

    @property
    def peak_bytes(self) -> int | None:
        """Estimated peak memory of the solve in bytes, or ``None``."""
        value = self._peak_bytes
        return value() if callable(value) else value


@dataclass(frozen=True)
class Workflow:
    """What one public workflow supports.

    Attributes
    ----------
    name : str
        Public name of the workflow in messages.
    cutoff : bool
        Whether the workflow runs under an :class:`~quchip.ExcitationCutoff`.
    sectors : bool
        Whether the planner can solve the workflow on excitation sectors.
        The sector route needs concrete values, because it solves on host
        NumPy arrays; traced values use the backend's Liouvillian.
    """

    name: str
    cutoff: bool
    sectors: bool


WORKFLOWS: Mapping[str, Workflow] = MappingProxyType({workflow.name: workflow for workflow in (
    Workflow("steadystate()", cutoff=True, sectors=True),
    Workflow("VNA.sweep()", cutoff=False, sectors=False),
    Workflow("VNA.resonances()", cutoff=False, sectors=False),
    Workflow("VNA.finite_power()", cutoff=True, sectors=True),
    Workflow("VNA.measure()", cutoff=True, sectors=True),
    Workflow("VNA.output_spectrum()", cutoff=True, sectors=True),
    Workflow("VNA.g1()", cutoff=True, sectors=True),
    Workflow("VNA.g2()", cutoff=True, sectors=True),
    Workflow("fit_a_dress(evaluator='local')", cutoff=False, sectors=False),
    Workflow("eliminate()", cutoff=False, sectors=False),
    Workflow("eliminate_spectators()", cutoff=False, sectors=False),
    Workflow("active_patch()", cutoff=False, sectors=False),
    Workflow("partition()", cutoff=False, sectors=False),
)})


def require_workflow(chip: Any, name: str) -> Workflow:
    """Return a workflow's capabilities, and raise when the chip's excitation cutoff is outside them.

    Parameters
    ----------
    chip : Chip
        Chip whose excitation cutoff is checked.
    name : str
        Key of the workflow in :data:`WORKFLOWS`.
    """
    from quchip.cutoff import active_cutoff

    workflow = WORKFLOWS[name]
    policy = active_cutoff(chip)
    if policy is not None and not workflow.cutoff:
        raise ValueError(
            f"{workflow.name} does not support an excitation cutoff, and this chip keeps only "
            f"N <= {policy.max_total}. Use chip.with_space(None) to run it on the full product space."
        )
    return workflow


def array_name(backend: Any) -> str:
    """Return the name of a backend's array library."""
    return "numpy" if backend.array_module is np else "jax"


def plan_sectors(
    chip: Any, workflow: Workflow, *, labels: tuple[str, ...], inputs: tuple[str, ...] = (),
    modes: tuple[str, ...] = (), frequencies: Sequence[Any] = (), options: Any = None, concrete: bool = True,
) -> tuple[Any, str]:
    """Return the sector model of a stationary workflow, or ``None`` for the backend's Liouvillian, and why.

    A chip with an excitation cutoff, or a chip with more than
    :data:`AUTOMATIC_SECTOR_STATES` product states, solves on sectors when its
    static model conserves the total excitation ``N`` and its channels and
    inputs keep that structure. Solver options select the backend's
    Liouvillian.

    Parameters
    ----------
    chip : Chip
        Chip at one parameter point.
    workflow : Workflow
        Calling workflow.
    labels : tuple of str
        Output ports that the workflow reads.
    inputs : tuple of str
        Ports with coherent inputs.
    modes : tuple of str
        Fock modes whose observables the workflow reads.
    frequencies : sequence
        Carrier frequency of each coherent input in GHz.
    options : dict or None
        Stationary solver options.
    concrete : bool
        Whether the frequencies and amplitudes of the inputs are concrete.
    """
    from quchip.cutoff import active_cutoff
    from quchip.engine.frames import same_frequency
    from quchip.engine.sector_stationary import sector_probe_model

    policy = active_cutoff(chip)
    space = "the product space" if policy is None else f"the labels with N <= {policy.max_total}"
    if not workflow.sectors:
        return None, f"{workflow.name} solves with the backend's stationary Liouvillian."
    if options is not None:
        return None, f"Solver options select the backend's stationary Liouvillian on {space}."
    states = prod(chip.dims)
    if policy is None and states <= AUTOMATIC_SECTOR_STATES:
        return None, (f"The chip has {states} product states, at most {AUTOMATIC_SECTOR_STATES}, "
                      "so the backend's stationary Liouvillian solves the product space.")
    if not concrete or contains_tracer(tuple(frequencies)):
        return None, f"{_TRACED} The backend's stationary Liouvillian solves {space}."
    if any(not same_frequency(frequencies[0], other) for other in frequencies[1:]):
        return None, ("Inputs at different frequencies leave no frame in which the sector model is static. "
                      f"The backend's stationary Liouvillian solves {space}.")
    model, reason = sector_probe_model(chip, labels, inputs, modes)
    if model is None:
        return None, f"{reason} The backend's stationary Liouvillian solves {space}."
    if policy is not None:
        return model, f"{reason} The excitation cutoff keeps the labels with N <= {policy.max_total}."
    return model, f"{reason} The chip has {states} product states, more than {AUTOMATIC_SECTOR_STATES}."


def weak_probe_reason(chip: Any, labels: tuple[str, ...]) -> str:
    """Return why neither the harmonic nor the vacuum response gives a chip's small-signal matrix."""
    from quchip.engine.linear_response import _UnsupportedLinearModel, _sector_response_model

    try:
        _sector_response_model(chip, labels)
    except _UnsupportedLinearModel as error:
        detail = str(error) or "The model does not keep the excitation structure of the vacuum response."
    else:
        detail = "The vacuum response does not support this port network."
    return f"The chip is not passive and harmonic. {detail}"


def cutoff_reason(cutoff: int, check: "CutoffCheck | None") -> str:
    """Return the sentence that explains an automatic cutoff ``K``."""
    if check is None:
        return f" The solve keeps N <= {cutoff}, which holds every label."
    if check.status == "passed":
        return (f" The solve keeps N <= {cutoff}, because it changed the results of N <= {check.compared_with} "
                f"by {check.change:.1e}, within the tolerance {CUTOFF_TOLERANCE:.0e}. This is a convergence "
                "check, not an error bound.")
    return (f" The solve keeps N <= {cutoff}, and the comparison with N <= {check.compared_with} did not run, "
            "because it does not fit in the available memory.")


def sector_route(model: Any, cutoff: int, reason: str, *, solved: bool = True) -> SolveRoute:
    """Return the record of a solve on the sectors ``N <= cutoff``.

    ``solved=False`` records a result that needed no stationary solve, which
    has no memory estimate.
    """
    if array_name(model.chip.backend) != "numpy":
        reason += " The sector solve runs with SciPy on the host, not with the backend's JAX arrays."
    return SolveRoute("sector_stationary", reason, model.size(cutoff), "numpy",
                      (lambda: model.peak_bytes(cutoff)) if solved else None)


def sector_diagnostics(model: Any, cutoff: int, solve: Any, reason: str, choice: str,
                       check: "CutoffCheck | None") -> dict[str, Any]:
    """Return the diagnostics entries of a solve on the sectors ``N <= cutoff``.

    Parameters
    ----------
    model : SectorProbeModel
        Sector model of the solve.
    cutoff : int
        Kept cutoff ``K``.
    solve : SectorSolve
        The stationary solve on ``N <= K``.
    reason : str
        Why the planner chose the sector route.
    choice : {"explicit", "automatic"}
        Whether ``K`` comes from the chip's excitation cutoff or from the
        comparison with ``K + 1``.
    check : CutoffCheck or None
        The comparison of an automatic ``K``.
    """
    state = solve.state
    route = sector_route(model, cutoff, reason + (cutoff_reason(cutoff, check) if choice == "automatic" else ""))
    values: dict[str, Any] = {
        "solver": route.name, "route": route, "excitation_cutoff": cutoff, "cutoff_choice": choice,
        "retained_dimension": model.size(cutoff), "excitation_weights": model.weights(cutoff, state),
        "iterations": solve.iterations, "residual": solve.residual,
        "trace_error": float(abs(np.trace(state) - 1.0)),
        "hermiticity_error": float(np.linalg.norm(state - state.conj().T)),
        "positivity_error": lambda: max(0.0, -float(np.min(np.linalg.eigvalsh(0.5 * (state + state.conj().T))))),
    }
    if choice == "automatic":
        values["cutoff_check"] = None if check is None else check.record()
    return values


def liouvillian_route(backend: Any, engine_result: Any, reason: str) -> SolveRoute:
    """Return the record of a solve with the backend's stationary Liouvillian."""
    return SolveRoute(_LIOUVILLIAN, reason, prod(engine_result.dims), array_name(backend),
                      DeferredValue(lambda: backend.stationary_bytes(engine_result)))


def static_route(chip: Any, structure: Any) -> SolveRoute:
    """Return how static dressed queries diagonalize a chip.

    Parameters
    ----------
    chip : Chip
        Chip whose static model is queried.
    structure : ExcitationStructure
        Excitation structure of that static model.
    """
    arrays = array_name(chip.backend)
    if structure.cutoff is not None:
        return SolveRoute("sector_eigensystem", f"The excitation cutoff keeps N <= {structure.cutoff}, so a dressed "
                          "query diagonalizes the class of totals that holds its label.",
                          structure.largest_block(structure.cutoff), arrays)
    if structure.conserves:
        return SolveRoute("sector_eigensystem", "The static model conserves N, so a dressed query diagonalizes "
                          "the sector of its label.", structure.largest_block(2), arrays)
    return SolveRoute("full_eigensystem", "The static model does not conserve N, so dressed queries diagonalize "
                      "the full static model. chip.excitation_structure() lists the terms that change N.",
                      prod(chip.dims), arrays)


@dataclass(frozen=True)
class CutoffCheck:
    """Comparison of two adjacent cutoffs over a whole grid, which settles an automatic cutoff.

    Attributes
    ----------
    status : {"passed", "not run"}
        ``"not run"`` means that the larger cutoff did not fit in memory.
    compared_with : int
        The other cutoff of the pair: one below the kept cutoff when the
        check passed and kept the larger one, else one above it.
    change : float or None
        Largest relative change, or ``None`` when the comparison did not run.
    states : int
        Number of labels at the larger cutoff of the pair.
    """

    status: str
    compared_with: int
    change: float | None
    states: int

    def record(self) -> dict[str, Any]:
        """Return the check as diagnostics entries."""
        return {"status": self.status, "compared_with": self.compared_with, "change": self.change,
                "tolerance": CUTOFF_TOLERANCE}


def automatic_cutoff(
    maximum: int, solve: Callable[[int, Any], Any], change: Callable[[Any, Any], float], *,
    fits: Callable[[int], bool], states: Callable[[int], int],
    complete: Callable[[int, Any], Any] | None = None, partial: Callable[[Any], bool] | None = None,
    start: int | None = None,
) -> tuple[int, Any, CutoffCheck | None]:
    """Raise ``K`` from ``start`` until ``N <= K + 1`` changes the results by at most the tolerance.

    The larger cutoff of the passing pair is kept, since its results are
    already solved. This is a convergence check, not an error bound.

    Parameters
    ----------
    maximum : int
        Largest total of the chip, where the sectors hold every label.
    solve : callable
        ``solve(K, lower)`` returns the results on ``N <= K``, starting from
        the results ``lower`` at a smaller cutoff, or from ``None``.
    change : callable
        ``change(lower, upper)`` returns the largest relative change.
    fits : callable
        ``fits(K)`` says whether a solve on ``N <= K`` fits in memory.
    states : callable
        ``states(K)`` returns the number of labels with ``N <= K``.
    complete : callable, optional
        ``complete(K, results)`` completes results that a comparison solved in
        part, when ``K`` becomes the next lower cutoff.
    partial : callable, optional
        ``partial(results)`` says whether a comparison solved the results in
        part. A passing pair then keeps its smaller cutoff.
    start : int or None, default None
        First cutoff of the ladder. ``None`` takes :data:`FIRST_CUTOFF`.

    Returns
    -------
    tuple
        The kept cutoff, its results, and the check, which is ``None`` when the
        kept cutoff holds every label.
    """
    cutoff = min(FIRST_CUTOFF if start is None else start, maximum)
    lower = solve(cutoff, None)
    while cutoff < maximum:
        upper = cutoff + 1
        if not fits(upper):
            return cutoff, lower, CutoffCheck("not run", upper, None, states(upper))
        try:
            candidate = solve(upper, lower)
        except MemoryError:
            return cutoff, lower, CutoffCheck("not run", upper, None, states(upper))
        value = change(lower, candidate)
        if value <= CUTOFF_TOLERANCE:
            if partial is not None and partial(candidate):
                return cutoff, lower, CutoffCheck("passed", upper, value, states(upper))
            # The largest total holds every label, so the kept solve is exact.
            return upper, candidate, None if upper == maximum else CutoffCheck("passed", cutoff, value, states(upper))
        cutoff, lower = upper, candidate if complete is None else complete(upper, candidate)
    return cutoff, lower, None


def largest_change(lower: Sequence[Any], upper: Sequence[Any]) -> float:
    """Return the largest change ``|a - b| / (max(|a|, |b|) + floor / tolerance)`` over pairs of values."""
    norms = [float(np.linalg.norm(value)) for value in (*lower, *upper)]
    floor = CUTOFF_FLOOR / CUTOFF_TOLERANCE * max(norms, default=0.0)
    change = 0.0
    for a, b in zip(lower, upper, strict=True):
        scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b))) + floor
        if scale > 0.0:
            change = max(change, float(np.linalg.norm(np.asarray(a) - np.asarray(b))) / scale)
    return change


def warn_cutoff(method: str, cutoff: int, check: CutoffCheck, quantities: str, *, note: str = "") -> None:
    """Warn which truncation an automatic route kept and what its check found."""
    if check.status == "passed":
        text = (f"{method} kept the automatic excitation cutoff N <= {cutoff}: N <= {check.compared_with} changes the "
                f"{quantities} by at most {CUTOFF_TOLERANCE:.0e}{note}.")
    else:
        text = (f"{method} kept the automatic excitation cutoff N <= {cutoff}, not verified: N <= "
                f"{check.compared_with} does not fit in memory.")
    text += " Use chip.with_space(ExcitationCutoff(max_total=K)) to set K."
    warnings.warn(text, AutoChoiceWarning, stacklevel=user_stacklevel())
