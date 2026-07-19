"""TwoPhotonDrive -- parametric two-photon drive for Kerr-cat qubits.

Physical Hamiltonian (lab frame)::

    H_drive = eps2(t) * [a_dag^2 * exp(-i*2pi*omega_d*t) + a^2 * exp(+i*2pi*omega_d*t)]

At resonance (omega_d = 2*omega_f), the rotating-frame Hamiltonian is::

    H_rot = eps2(t) * (a_dag^2 + a^2)

The coupling operator is ``a^2 + a_dag^2`` and the drive is assembled by the
engine's ``DriveModulation.SINGLE_TONE`` dispatch -- the same carrier-mixing path
used for charge and phase drives.  Setting the carrier frequency to ``2*omega_f``
ensures that each operator band picks up the correct two-photon detuning.

SINGLE_TONE's real-field projection halves the scheduled amplitude: an
envelope of amplitude ``A(t)`` scheduled on this drive contributes
``A(t)/2 * (a_dag^2 + a^2)`` to ``H_rot``, not ``A(t) * (a_dag^2 + a^2)``.
Scheduling ``amplitude = 2*eps2(t)`` realizes the ``eps2(t)`` coefficient
shown in ``H_rot`` above.

References
----------
.. [1] Grimm et al., Nature 584, 205 (2020). arXiv:1907.12131.
.. [2] Hajr et al., PRX Quantum 5, 020347 (2024). arXiv:2404.16697.
"""

from __future__ import annotations

from quchip.control.drive import BaseDrive, DriveChannel
from quchip.control.signal_spec import DriveModulation
from quchip.devices.base import BaseDevice


class TwoPhotonDrive(BaseDrive):
    """Parametric two-photon drive for Kerr-cat qubit stabilisation.

    Coupling operator: ``a^2 + a_dag^2``

    The drive should be scheduled at twice the cavity frequency
    (``freq = 2 * cavity.freq``) so that in the rotating frame the
    interaction is static: ``eps2(t) * (a_dag^2 + a^2)``.  This combination
    of Kerr nonlinearity and two-photon drive creates and stabilises cat states.

    Uses the standard ``DriveModulation.SINGLE_TONE`` carrier dispatch -- no engine
    modifications required.  The engine band-decomposes ``a^2 + a_dag^2`` into
    bands of excitation weight Delta_n = +2 and Delta_n = -2, attaching the
    correct two-photon carrier automatically.

    SINGLE_TONE's real-field projection contributes only half the
    scheduled envelope amplitude to each band: the coefficient landing on
    ``a_dag^2 + a^2`` in the rotating frame is ``A(t)/2``, where ``A(t)``
    is the amplitude scheduled on this drive's envelope.  Schedule
    ``amplitude=2*eps2(t)`` to realize the target two-photon drive
    strength ``eps2(t)`` used above and in ``alpha^2 = eps2/K``.

    Parameters
    ----------
    target : BaseDevice | None
        Device to connect this drive to.  ``None`` means unconnected.
    label : str | None
        Optional explicit label; otherwise auto-generated.
    rwa : bool | None
        Per-drive RWA override.  ``None`` means follow the chip-level setting.

    References
    ----------
    .. [1] Grimm et al., Nature 584, 205 (2020). arXiv:1907.12131.
    .. [2] Hajr et al., PRX Quantum 5, 020347 (2024). arXiv:2404.16697.

    Examples
    --------
    >>> from quchip.devices.kerr_cavity import KerrCavity
    >>> from quchip.control.drives_two_photon import TwoPhotonDrive
    >>> cav = KerrCavity(freq=5.0, kerr=1.0, levels=10, label="cav")
    >>> d2 = TwoPhotonDrive(target=cav)
    >>> channels = d2.local_channels(cav)
    >>> len(channels)
    1
    """

    _type_prefix: str = "two_photon"
    # The two-photon carrier ``2 * omega_d`` has no natural default on the
    # target device, so callers must always supply ``freq`` explicitly. Marked
    # for the same sequence-level guard used by PhaseDrive — without it,
    # ``seq.schedule`` would create a pulse with ``freq=None`` that fails
    # later inside stage 2 with a generic SINGLE_TONE error.
    _carrier_required: bool = True

    def local_channels(self, device: BaseDevice) -> list[DriveChannel]:
        """Return the two-photon coupling channel ``a^2 + a_dag^2``.

        Parameters
        ----------
        device : BaseDevice
            The cavity device being driven.

        Returns
        -------
        list[DriveChannel]
            One channel with operator ``a^2 + a_dag^2`` and
            ``DriveModulation.SINGLE_TONE`` modulation.
        """
        a = device.lowering_operator()
        a_dag = device.raising_operator()
        return [DriveChannel(
            operator=a @ a + a_dag @ a_dag,
            modulation=DriveModulation.SINGLE_TONE,
        )]

    def physics_notes(self) -> list[str]:
        """Return the base drive notes plus the two-photon coupling declaration."""
        notes = super().physics_notes()
        notes.append(
            "Two-photon parametric drive: coupling operator a^2 + a_dag^2; "
            "schedule at freq=2*cavity.freq for resonant two-photon interaction."
        )
        return notes
