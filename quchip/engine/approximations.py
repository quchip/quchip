"""Engine-side operations owned by explicit approximation strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from quchip.approximations import Approximation
from quchip.engine.ir import Add, Carrier, ImagPart, Multiply, RealPart, Scale, SignalProgram
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import maybe_concrete_scalar

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
    frame = Carrier(freq=TWO_PI * frame_frequency, sign=-1)
    filters_signal = approximation.filters_terms if filter_signal_bands is None else filter_signal_bands
    if not filters_signal or not has_carrier or weight == 0:
        return Multiply((program, frame))

    if isinstance(program, (RealPart, ImagPart)):
        # A band co-rotates with a weight-w operator band when w and its frequency have one sign, for either sign
        # of the carrier. The bands come in (+f, -f) halves; a zero carrier keeps the conventional half.
        bands = program.bands()
        conventional = [index % 2 == (1 if weight > 0 else 0) for index in range(len(bands))]
        frequencies = [maybe_concrete_scalar(band.freq) for band in bands]
        if all(value is not None for value in frequencies):
            kept = [band for band, value, half in zip(bands, frequencies, conventional)
                    if value is not None and (weight * value > 0.0 or (value == 0.0 and half))]
            children = tuple(Multiply((band.envelope, Carrier(freq=band.freq, sign=1), frame)) for band in kept)
        else:
            children = tuple(
                Multiply((Scale(band.envelope, _co_rotating(weight, band.freq, half)), Carrier(freq=band.freq, sign=1),
                          frame))
                for band, half in zip(bands, conventional)
            )
        return children[0] if len(children) == 1 else Add(children)

    # Exact preserves nonlinear signal expressions. RWA cannot infer a
    # first-order carrier partner from an arbitrary nonlinear expression, so
    # it leaves the authored program intact.
    return Multiply((program, frame))


def _co_rotating(weight: int, frequency: Any, conventional: bool) -> Any:
    """Return 1 for a traced band that :func:`resolve_drive_program` keeps, else 0."""
    import jax.numpy as jnp

    return jnp.where(weight * frequency > 0.0, 1.0, jnp.where(frequency == 0.0, float(conventional), 0.0))
