"""Weak-drive cross-resonance response of one directed edge.

:func:`analyze_cr_susceptibility` projects a control line onto the target's
dressed ``0 -> 1`` transition, with the control in ``|0>`` and in ``|1>``. It
returns the ``IX`` and ``ZX`` coefficients per unit of drive amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quchip.utils.labeling import resolve_label

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.control.drive import BaseDrive
    from quchip.devices.base import BaseDevice


@dataclass(frozen=True)
class CRSusceptibilityResult:
    r"""Store weak-drive CR coefficients per unit of programmed drive amplitude.

    For control-state-conditioned target-transition matrix elements

    .. math::

        m_z = \langle\widetilde{z_c,1_t}|D_c|\widetilde{z_c,0_t}\rangle,

    the effective-Hamiltonian convention

    .. math::

        H_\mathrm{eff} = \tfrac{1}{2}[\mathrm{Re}(C_I)IX + \mathrm{Im}(C_I)IY
        + \mathrm{Re}(C_Z)ZX + \mathrm{Im}(C_Z)ZY]

    gives ``IX_per_amplitude = (m_0 + m_1)/2`` and
    ``ZX_per_amplitude = (m_0 - m_1)/2`` for a real carrier's peak amplitude.
    The raw matrix elements retain the full drive operator normalization.
    Complex values encode ``IX+i IY`` and ``ZX+i ZY`` in the raising convention.
    Their real and imaginary parts multiply Hermitian X and Y operators.

    Attributes
    ----------
    m_control_0, m_control_1 : complex or array_like
        Dressed target-transition matrix elements, conditioned on the control
        in ``|0>`` and in ``|1>``.
    IX_per_amplitude, ZX_per_amplitude : complex or array_like
        Complex weak-drive X/Y coefficients per unit of scheduled peak amplitude.
    control, target, drive : str
        Resolved labels.
    """

    m_control_0: Any
    m_control_1: Any
    IX_per_amplitude: Any
    ZX_per_amplitude: Any
    control: str
    target: str
    drive: str


def analyze_cr_susceptibility(
    chip: "Chip",
    control: str | "BaseDevice",
    target: str | "BaseDevice",
    *,
    drive: str | "BaseDrive" | None = None,
) -> CRSusceptibilityResult:
    r"""Return the dressed weak-drive CR response of one directed edge.

    The analysis projects the physical control-line operator onto the target's
    dressed ``0 -> 1`` transition twice, with the control in ``|0>`` and in
    ``|1>``. It does no pulse, no rotating-frame solve, and no time evolution.

    Parameters
    ----------
    chip
        Coupled chip with attached control equipment.
    control, target
        Directed CR control and target, as device objects or labels.
    drive
        Control line to project. If omitted, the function selects the unique
        wired device-target line attached to ``control``.

    Returns
    -------
    CRSusceptibilityResult
        Conditional matrix elements and the related ``IX`` and ``ZX``
        complex X/Y coefficients per unit of scheduled peak amplitude.
        Carrier selection contributes a factor of one half.

    Raises
    ------
    ValueError
        If the edge is a self-edge, control equipment is absent, implicit
        drive resolution is ambiguous, or the selected line does not target
        the control.

    References
    ----------
    Magesan and Gambetta, Phys. Rev. A 101, 052308 (2020).
    https://arxiv.org/abs/1804.04073.
    Malekakhlagh, Magesan, and McKay, Phys. Rev. A 102, 042605 (2020).
    https://arxiv.org/abs/2005.00133.
    """
    _, control_device = chip._resolve_device_index(control)
    _, target_device = chip._resolve_device_index(target)
    if control_device is target_device:
        raise ValueError("control and target must be different devices")

    equipment = chip.control_equipment
    if equipment is None:
        raise ValueError("analyze_cr_susceptibility requires attached control equipment")

    candidates = [
        line
        for line in equipment.lines
        if line.device_label == control_device.label
    ]
    if drive is None:
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one drive targeting '{control_device.label}', found {len(candidates)}"
            )
        selected = candidates[0]
    else:
        selected_label = resolve_label(drive)
        matches = [line for line in equipment.lines if line.label == selected_label]
        if len(matches) != 1:
            raise ValueError(f"Drive '{selected_label}' is absent or ambiguous")
        selected = matches[0]
        if selected not in candidates:
            raise ValueError(
                f"Drive '{selected.label}' does not target control '{control_device.label}'"
            )

    m0 = chip.drive_matrix_elements(({}, {target_device: 1}), drives=[selected])[selected]
    m1 = chip.drive_matrix_elements(
        ({control_device: 1}, {control_device: 1, target_device: 1}),
        drives=[selected],
    )[selected]
    return CRSusceptibilityResult(
        m_control_0=m0,
        m_control_1=m1,
        IX_per_amplitude=(m0 + m1) / 2,
        ZX_per_amplitude=(m0 - m1) / 2,
        control=control_device.label,
        target=target_device.label,
        drive=selected.label,
    )
