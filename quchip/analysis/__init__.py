"""quchip.analysis — physics analysis tools.

Sub-modules
-----------
cross_resonance
    :func:`analyze_cross_resonance` — extract the six CR effective Hamiltonian
    coefficients {IX, IY, IZ, ZX, ZY, ZZ} from Bloch-vector tomography data.
dispersive_readout
    :func:`analyze_dispersive_readout` — closed-form steady-state readout
    figures of merit (pointer states, SNR, assignment error, measurement
    dephasing) from the χ/κ that :func:`~quchip.chip.transformations.eliminate`
    reports.
effective_hamiltonian
    :func:`analyze_static_zz` — exact static ZZ between two devices plus its
    2nd-order SW virtual-pathway attribution. :func:`effective_hamiltonian` —
    the des-Cloizeaux effective Hamiltonian on a labeled computational
    subspace, with eigenvalues exactly the labeled dressed energies.
    :func:`effective_hamiltonian_between_states` — the same construction
    specialized to exactly two explicit bare states (e.g. a static exchange
    rate between two single-excitation states).
"""

from quchip.analysis.cross_resonance import (
    CRHamiltonianResult,
    CRSusceptibilityResult,
    analyze_cross_resonance,
    analyze_cr_susceptibility,
)
from quchip.analysis.dispersive_readout import (
    DispersiveReadoutResult,
    analyze_dispersive_readout,
)
from quchip.analysis.effective_hamiltonian import (
    EffectiveHamiltonianResult,
    StaticZZResult,
    analyze_static_zz,
    effective_hamiltonian,
    effective_hamiltonian_between_states,
)

__all__ = [
    "CRHamiltonianResult",
    "CRSusceptibilityResult",
    "DispersiveReadoutResult",
    "EffectiveHamiltonianResult",
    "StaticZZResult",
    "analyze_cross_resonance",
    "analyze_cr_susceptibility",
    "analyze_dispersive_readout",
    "analyze_static_zz",
    "effective_hamiltonian",
    "effective_hamiltonian_between_states",
]
