"""Physical constants in the quchip ``GHz / ns / mK`` unit convention.

quchip does not use an ``hbar = 1`` convention. Domain Hamiltonians express
energies as ``E / h`` in ordinary (not angular) frequency, GHz; the single
``2 * pi`` conversion to angular frequency happens later, at the engine
assembly boundary. Times are expressed in ns and temperatures in mK.

Constants below are provided for user-facing physics expressions (thermal
populations, flux quanta, etc.) and derived from SI defining values so the
GHz/ns/mK-convention constants and their SI counterparts never drift apart.

All numerical values are taken from the **CODATA 2018** recommended
constants as published by NIST (https://physics.nist.gov/cuu/Constants/)
and rescaled into the quchip unit system where applicable.
"""

from __future__ import annotations

import math

#: ``2 * pi``.
TWO_PI: float = 2.0 * math.pi

#: Planck constant, SI units ``J*s``. Exact since the 2019 SI redefinition
#: (CODATA 2018).
_H_SI: float = 6.62607015e-34

#: Elementary charge, SI units ``C``. Exact since the 2019 SI redefinition
#: (CODATA 2018).
_E_SI: float = 1.602176634e-19

#: Boltzmann constant, SI units ``J/K``. Exact since the 2019 SI
#: redefinition (CODATA 2018).
_K_B_SI: float = 1.380649e-23

#: Boltzmann constant in ``GHz / mK``.
#:
#: Derived as ``k_B / h`` (both exact since the 2019 SI redefinition),
#: rescaled from Hz/K to GHz/mK: approximately ``2.0836619123e10 Hz/K``,
#: i.e. ``2.0836619123e-2 GHz/mK``.
k_B: float = (_K_B_SI / _H_SI) * 1e-12

#: Reduced Planck constant in SI units, ``J*s``.
#:
#: Derived as ``h / (2 * pi)`` with ``h`` exact (CODATA 2018 / 2019 SI
#: redefinition). quchip's own Hamiltonians work in ordinary frequency, so
#: this value is only needed when cross-checking against SI expressions.
hbar: float = _H_SI / TWO_PI

#: Superconducting magnetic flux quantum ``Phi_0 = h / (2 * e)`` in Weber.
#:
#: Derived from ``h`` and ``e``, both exact since the 2019 SI redefinition
#: (CODATA 2018): approximately ``2.067833848e-15 Wb``.
Phi_0: float = _H_SI / (2 * _E_SI)
