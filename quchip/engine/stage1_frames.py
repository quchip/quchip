"""Stage 1: resolve a :data:`FrameSpec` into a :class:`ResolvedFrame`.

This stage is purely combinatorial: it decides which rotating frame to
work in.

Supported specs
---------------
* ``"lab"`` — every reference frequency is zero; stage 2 emits the
  bare chip Hamiltonian unchanged.
* ``"rotating"`` — each device's reference is its
  :attr:`~quchip.devices.base.BaseDevice.reference_freq` (the device's
  readout/LO reference, defaulting to the dressed drive frequency
  ``ω_d``), so stage 2 builds

  .. math::
      H(t) \\;=\\; H_0 - \\sum_i \\omega_{\\text{ref},i} n_i
                 + V_{\\text{drive}}(t) + V_{\\text{coupling}}(t),

  the standard rotating-frame form used in cQED / driven multi-level
  systems (e.g. Gambetta et al., *PRA* **74**, 042318 (2006);
  Krantz et al., *Appl. Phys. Rev.* **6**, 021318 (2019)). Setting a
  device's ``reference_freq`` off its transition surfaces a residual
  detuning ``Δ = ω − ω_ref`` in ``H₀`` — idle Ramsey precession.
* Scalar — every device uses the same shared reference frequency.
* ``dict[str | BaseDevice, scalar]`` — per-device references; missing
  entries default to ``0.0``.

The demodulation frequencies are ``ω_ref − ω_frame`` per device: observables
are always reported co-rotating at ``reference_freq`` (the readout LO),
independent of which frame the solver integrated in. In the default
``"rotating"`` mode the integration frame *is* the reference frame, so the
demodulation is a no-op and the raw stored states already sit in the readout
frame (``result.states`` and ``result.expect`` agree). Transverse observables
(``<a>``, ``<σ_x>``) thus come back as the non-oscillatory demodulated
envelope; only an explicitly overridden non-reference integration frame
leaves ``result.states`` in that other frame.

Whether a coupling band folds into ``H₀`` is decided per band in stage
2, from the concreteness of its frame carrier ``Δa·ω_a + Δb·ω_b`` — not
here (see :func:`~quchip.engine.stage2_assembly._collect_coupling_terms`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quchip.engine.ir import ResolvedFrame, _is_scalar_like
from quchip.utils.labeling import resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.engine.ir import FrameSpec


def resolve_frame(chip: Chip, frame_spec: FrameSpec) -> ResolvedFrame:
    """Resolve *frame_spec* into a :class:`ResolvedFrame`.

    Dispatches on ``frame_spec`` shape (``str`` / scalar-like / dict),
    fills a per-device ``frequencies`` dict, and computes the
    demodulation frequencies ``reference_freq − ω_frame``. See the
    module docstring for the physical meaning of each mode.

    Dressing happens lazily: reading a device's
    :attr:`~quchip.devices.base.BaseDevice.reference_freq` diagonalizes
    the chip only when its default (the dressed drive frequency) is
    actually consulted. A chip whose devices all carry explicit
    ``reference_freq`` overrides resolves any frame spec without ever
    diagonalizing.
    """
    devices = chip.devices
    labels = [dev.label for dev in devices]

    mode: str
    frequencies: dict[str, Any]

    # Dispatch by spec shape. Keep the isinstance(str) check ahead of any
    # equality comparison so no JAX tracer is compared to a string
    # (which would yield a traced bool and force concretization).
    if isinstance(frame_spec, str):
        if frame_spec == "lab":
            mode = "lab"
            frequencies = {label: 0.0 for label in labels}
        elif frame_spec == "rotating":
            mode = "rotating"
            frequencies = {dev.label: dev.reference_freq for dev in devices}
        else:
            raise ValueError(f"Unknown frame string. Expected 'lab' or 'rotating', got {frame_spec!r}")
    elif _is_scalar_like(frame_spec):
        mode = "float"
        frequencies = {label: frame_spec for label in labels}
    elif isinstance(frame_spec, dict):
        mode = "dict"
        normalized = {resolve_label(key): value for key, value in frame_spec.items()}
        frequencies = {label: normalized.get(label, 0.0) for label in labels}
    else:
        raise TypeError(
            "frame_spec must be 'lab', 'rotating', a scalar-like frequency, or a "
            f"dict[str|BaseDevice, scalar-like], got {type(frame_spec).__name__}"
        )

    demod_freqs = {dev.label: dev.reference_freq - frequencies[dev.label] for dev in devices}

    return ResolvedFrame(
        frequencies=frequencies,
        demod_freqs=demod_freqs,
        mode=mode,
    )
