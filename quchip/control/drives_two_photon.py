"""TwoPhotonDrive -- parametric two-photon drive for Kerr-cat qubits.

Physical Hamiltonian (lab frame)::

    H_drive = eps2(t) * [a_dag^2 * exp(-i*2pi*omega_d*t) + a^2 * exp(+i*2pi*omega_d*t)]

At resonance (omega_d = 2*omega_f), the rotating-frame Hamiltonian is::

    H_rot = eps2(t) * (a_dag^2 + a^2)

The drive maps the delivered in-phase signal to ``a^2 + a_dag^2``. Setting the
carrier frequency to ``2*omega_f`` makes the weight-two operator bands
resonant in the corresponding rotating frame.

The real-field projection halves the scheduled amplitude: an
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

from typing import ClassVar

from quchip.control.drive import DeviceDrive
from quchip.control.signal import AnalyticSignal
from quchip.declarative.expr import as_operator_expr
from quchip.devices.base import BaseDevice


class TwoPhotonDrive(DeviceDrive):
    """Parametric two-photon drive for Kerr-cat qubit stabilisation.

    Coupling operator: ``a^2 + a_dag^2``

    The drive should be scheduled at twice the cavity frequency
    (``freq = 2 * cavity.freq``) so that in the rotating frame the
    interaction is static: ``eps2(t) * (a_dag^2 + a^2)``.  This combination
    of Kerr nonlinearity and two-photon drive creates and stabilises cat states.

    The engine band-decomposes ``a^2 + a_dag^2`` into excitation weights
    Delta_n = +2 and Delta_n = -2 and combines them with the delivered
    signal's carrier.

    The real-field projection contributes only half the
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
    >>> d2.target_label == cav.label
    True
    """

    _type_prefix: ClassVar[str] = "two_photon"

    def hamiltonian(self, device: BaseDevice, signal: AnalyticSignal):
        """Return the two-photon coupling channel ``a^2 + a_dag^2``.

        Parameters
        ----------
        device : BaseDevice
            The cavity device being driven.

        """
        a = device.lowering_operator()
        a_dag = device.raising_operator()
        operator = as_operator_expr(
            a @ a + a_dag @ a_dag,
            labels=(device.label,),
            dims=(device.local_space().dimension,),
            name=rf"\hat H_{{2\gamma,{device.label}}}",
        )
        return signal.i * operator

    def physics_notes(self) -> list[str]:
        """Return the base drive notes plus the two-photon coupling declaration."""
        notes = super().physics_notes()
        notes.append(
            "Two-photon parametric drive: coupling operator a^2 + a_dag^2; "
            "schedule at freq=2*cavity.freq for resonant two-photon interaction."
        )
        return notes
