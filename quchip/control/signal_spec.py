"""Frame-agnostic drive signal specifications.

This module defines the data a drive emits to describe one scheduled
pulse *without* committing to any specific frame, carrier convention,
or engine IR representation. The spec carries ordinary-GHz frequencies
and envelope references only — no :mod:`quchip.engine.ir` nodes appear
here.

Stage 2 of the engine is the sole consumer that composes a
:class:`DriveSignalSpec` with the resolved rotating frame and the
drive channel's :class:`DriveModulation` kind to produce the IR-level
``SignalProgram`` AST. This keeps drives owning their local Hamiltonians
and the engine owning IR assembly, with no legacy fallback path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quchip.control.envelopes import BaseEnvelope


class DriveModulation(Enum):
    """How a drive channel's operator is time-modulated.

    * :attr:`SINGLE_TONE` — microwave IQ-style carrier mixing. In the
      lab frame the raw signal is mixed with ``exp(-i·2π·ν_d·t)`` and
      projected to its real part; under RWA the signal is decomposed
      into co- and counter-rotating components per Fourier band
      (Krantz et al. 2019, Eq. 89–90).
    * :attr:`DIRECT_REAL` — real-valued baseband coupling (no carrier,
      no RWA), appropriate for flux-like drives (Koch et al. 2007,
      Sec. II; Krantz et al. 2019, Sec. V.A on flux tunability).
    * :attr:`EDGE_PUMP` — real pump δ(t) on a modulable coupling's
      strength: baseband ``δ(t) = Re s(t)`` when the scheduled op has no
      carrier, single-tone ``δ(t) = Re[s(t)·e^{-i·2π·ν_d·t}]`` when it
      does. The tone is never RWA-split — the coupling's parametric hook
      already selected the retained operator structure (Didier et al.,
      PRA 97, 022330 (2018) on parametric gate activation).

    The engine dispatches on this tag in a single code path; adding a
    new modulation kind is an infrastructure change, not a per-drive
    special case.
    """

    SINGLE_TONE = "single_tone"
    DIRECT_REAL = "direct_real"
    EDGE_PUMP = "edge_pump"


@dataclass(frozen=True)
class DriveSignalSpec:
    """Frame-agnostic description of one scheduled drive pulse.

    The spec carries everything the engine needs to build the raw line
    signal and later compose it with the frame + carrier during
    modulation — expressed in ordinary GHz (frequencies) and ns (times),
    with no IR references.

    Parameters
    ----------
    envelope : BaseEnvelope
        Pulse envelope reference. Its ``waveform(t)`` is called at
        evaluation time; all differentiable pulse parameters live as
        leaves on the envelope (:mod:`quchip.control.envelopes`).
    start_time : float
        Onset of the pulse in ns.
    duration : float
        Envelope duration in ns (the pulse is zero outside
        ``[start_time, start_time + duration]``).
    phase_offset : float
        Phase rotation applied to the raw line signal, in radians.
    drive_freq : float | None
        Microwave carrier frequency in ordinary GHz for
        :attr:`DriveModulation.SINGLE_TONE` channels; ``None`` for baseband
        (:attr:`DriveModulation.DIRECT_REAL`) drives.
    """

    envelope: "BaseEnvelope"
    start_time: float
    duration: float
    phase_offset: float
    drive_freq: float | None
