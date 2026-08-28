# Kerr matrix API design

## Purpose

Add one labeled, differentiable view of the chip's dressed self-Kerr and
cross-Kerr coefficients. The API consolidates the energy arithmetic already
exposed by `dressed_anharmonicity()` and `dispersive_shift()` without removing
their physics-specific names.

## Public API

`Chip.kerr_matrix()` returns an immutable `KerrMatrix` in `chip.devices` order:

```python
kerr = chip.kerr_matrix()

kerr.labels
# ("q0", "q1", "bus")

kerr.values
# [[alpha_q0, chi_q0_q1, chi_q0_bus],
#  [chi_q0_q1, alpha_q1, chi_q1_bus],
#  [chi_q0_bus, chi_q1_bus, alpha_bus]]

kerr[q0, "bus"]
# dressed cross-Kerr in GHz
```

`KerrMatrix` is a frozen public result type with:

- `labels: tuple[str, ...]`, the device labels in chip order;
- `values`, a real, symmetric, JAX-compatible square array in ordinary GHz;
- two-axis lookup accepting device objects or label strings.

Unknown labels raise `KeyError` naming the available labels. The first version
always includes every device and has no filtering or alternate-convention
arguments.

The existing scalar API remains public:

- `dressed_anharmonicity(device)`;
- `dispersive_shift(a, b)`;
- `static_zz(a, b)`;
- `zz(a, b)`.

`static_zz` and `zz` remain aliases of `dispersive_shift` for compatibility.
`cross_kerr_matrix()` is not added: `kerr_matrix()` is the accurate name for a
matrix whose diagonal contains self-Kerr terms.

## Physics convention

Let `E(...)` be the dressed eigenenergy assigned to the indicated bare product
state, with all unmentioned devices in level zero.

For distinct devices `i` and `j`,

```text
K[i, j] = E(1_i, 1_j) - E(1_i) - E(1_j) + E(0).
```

This is the same full-pull convention as `dispersive_shift(i, j)`. It is also
the static-ZZ coefficient for two qubits. The matrix entry is derived from the
complete dressed chip and can therefore differ from a declared
`CrossKerr.chi` when other modes or interactions contribute.

On the diagonal,

```text
K[i, i] = E(2_i) - 2 E(1_i) + E(0).
```

This is exactly `dressed_anharmonicity(i)`. It is level curvature, not a promise
to reproduce a component parameter with a similar name. In particular, the
current `KerrCavity` convention is `H = omega*n - K*n*(n-1)`, so an isolated
cavity has diagonal entry `-2*K`.

If a device has fewer than three resolved levels, its diagonal entry is `NaN`;
defined off-diagonal entries remain available. Device spaces must still support
levels zero and one for a cross-Kerr entry, as required by the existing scalar
API.

## Implementation structure

`Chip.kerr_matrix()` delegates to `ChipAnalysis`. A single private energy-stencil
helper owns both cases:

```python
_kerr_entry(i, j, *, eigenvalues, labeling)
```

The helper uses the existing bare-label construction and labeled dressed-energy
gather. For `i == j` it evaluates the diagonal stencil; otherwise it evaluates
the pair stencil. `ChipAnalysis.kerr_matrix()` computes the labeled eigensystem
once, fills the upper triangle, and mirrors it. Scalar methods call the same
helper for only their requested entry, so they do not construct an entire
matrix.

This preserves the existing analysis signature cache: one matrix call performs
at most one static-Hamiltonian assembly, diagonalization, and dressed labeling
for a fixed chip state. The result does not inspect only declared `CrossKerr`
edges.

`KerrMatrix` is registered as a JAX pytree with `values` as dynamic data and
`labels` as static metadata. Matrix construction uses JAX-compatible stacking
and indexed updates, without converting traced values to Python or NumPy
scalars. Both `kerr_matrix().values` and scalar lookup remain differentiable
away from dressed-label assignment discontinuities, matching the existing
`dispersive_shift()` contract.

## Ownership and mutation

The matrix is a read-only analysis result. There is no `set_kerr_matrix()`:
setting a dressed matrix would require an underdetermined inverse problem and
would blur the boundary between authored model parameters and derived dressed
observables. Users author `CrossKerr` couplings, other physical couplings, or
use inverse-design tools to fit those parameters.

## Verification

Focused tests will cover:

1. labels, shape, chip ordering, real dtype, and symmetry;
2. every off-diagonal entry matching `dispersive_shift()`;
3. every defined diagonal matching `dressed_anharmonicity()`;
4. an isolated Duffing device's diagonal convention;
5. the current `KerrCavity` convention, including the `-2*K` diagonal;
6. a direct `CrossKerr` edge reproducing its `chi` off diagonal;
7. a two-level device producing `NaN` only on its diagonal;
8. lookup by device and label, plus useful unknown-label errors;
9. reuse of one dressed eigensystem per matrix call;
10. JAX `jit` and gradient flow through matrix values and scalar lookup.

The public docs will describe the energy formulas, GHz units, dressed-versus-
authored distinction, two-level `NaN`, and `KerrCavity` sign/factor convention.

## Non-goals

- Setting or fitting a matrix directly.
- Returning a pandas object.
- Selecting a device subset in the first version.
- Per-level cross-Kerr tensors beyond the `0/1` full-pull convention.
- Changing the existing `CrossKerr` coupling model or scalar API names.
