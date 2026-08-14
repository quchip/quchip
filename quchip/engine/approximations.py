"""Engine-side operations owned by explicit approximation strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from quchip.approximations import Approximation
from quchip.engine.ir import Add, Carrier, ImagPart, Multiply, RealPart, SignalProgram
from quchip.utils.constants import TWO_PI

if TYPE_CHECKING:
    from quchip.backend.protocol import Backend, Operator

BandPredicate = Callable[[int, int], bool]


def apply_operator_band_filter(
    local_hamiltonian: "Operator",
    *,
    dims: tuple[int, int],
    labels: tuple[str, str],
    keeps_band: BandPredicate,
    backend: "Backend",
) -> "Operator | None":
    """Return the populated two-body bands accepted by ``keeps_band``."""
    from quchip.engine.bands import decompose_two_body_canonical_bands

    first_dimension, second_dimension = dims
    canonical = backend.to_canonical_operator(local_hamiltonian).with_metadata(
        dims=dims,
        subsystem_labels=labels,
        tag="coupling_local",
    )
    filtered: "Operator | None" = None
    for weights, band in decompose_two_body_canonical_bands(
        canonical,
        [first_dimension, second_dimension],
    ).items():
        if not keeps_band(*weights):
            continue
        band_operator = backend.from_canonical_operator(band)
        filtered = band_operator if filtered is None else filtered + band_operator
    return filtered


def resolve_drive_program(
    approximation: Approximation,
    program: SignalProgram,
    *,
    weight: int,
    frame_frequency: Any,
    has_carrier: bool,
    filter_signal_bands: bool | None = None,
) -> SignalProgram:
    """Combine one authored signal with its operator-frame phase."""
    frame = Carrier(freq=TWO_PI * weight * frame_frequency, sign=-1)
    filters_signal = approximation.filters_terms if filter_signal_bands is None else filter_signal_bands
    if not filters_signal or not has_carrier:
        return Multiply((program, frame))
    if weight == 0:
        raise ValueError("Carrier-driven weight-zero bands must be eliminated before resolution.")

    if isinstance(program, (RealPart, ImagPart)):
        bands = program.bands()
        selected = bands[1::2] if weight > 0 else bands[0::2]
        children = tuple(Multiply((band.envelope, Carrier(freq=band.freq, sign=1), frame)) for band in selected)
        return children[0] if len(children) == 1 else Add(children)

    # Exact preserves nonlinear signal expressions. RWA cannot infer a
    # first-order carrier partner from an arbitrary nonlinear expression, so
    # it leaves the authored program intact.
    return Multiply((program, frame))
