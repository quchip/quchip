# Dressed-chip fitting design

## Purpose

`fit_a_dress(desired)` treats `desired` as a numerical specification of the
dressed chip the user wants. It does not dress or otherwise simulate that chip
to discover its targets. Component classes declare how their input numbers are
interpreted by inverse design.

The returned `fit.chip` is a separate, runnable chip containing the bare
parameters that reproduce those dressed constraints. The desired chip is never
mutated.

## Component-owned defaults

Each supported device declares its default dressed observables and the bare
parameters normally varied to reproduce them. For example:

- `DuffingTransmon.freq` targets dressed `freq`;
- `DuffingTransmon.anharmonicity` targets dressed `anharmonicity`;
- `Resonator.freq` targets dressed `freq`.

Each coupling declares one default inverse-design observable. Its declared
scalar is interpreted as the numerical target for that observable:

- `Capacitive` targets `cross_kerr`;
- `CrossKerr` targets `cross_kerr`;
- `TunableCapacitive` targets `cross_kerr`;
- an unknown coupling defaults to its literal `coupling_strength`.

`cross_kerr` has one definition for every device pair:

\[
K_{ab}=E_{11}-E_{10}-E_{01}+E_{00}.
\]

It may be described as a conditional readout pull for a qubit-resonator pair
or static ZZ for two qubits, but fitting uses one evaluator and one numerical
convention. The half-pull convention is not silently introduced.

## Additional constraints

`constraints=` adds numerical observables on top of component defaults. A
constraint may locate a component or any pair of devices, including devices
without a direct edge:

```python
fit = fit_a_dress(
    desired,
    constraints={
        (q0, q1): {"exchange_rate": -0.0022},
        (q0, q2): {"cross_kerr": 0.00015},
    },
)
```

An explicit constraint with the same locator and observable replaces the
default. A value of `None` removes that default. Canonical result names are
`freq`, `anharmonicity`, `cross_kerr`, `exchange_rate`, and
`coupling_strength`; familiar legacy spellings may be normalized at the API
boundary but must not create a second physical convention.

## Parameters, starting points, and ambiguity

The default free parameters come from the same component policies that produce
the default targets. Additional constraints do not silently free unrelated
parameters. `vary=` is the complete manual allowlist when the user wants to
choose which bare quantities may move.

The desired chip is not an optimizer seed. `start=` supplies manual starting
values. Otherwise quchip uses declared device values and coupling-specific seed
estimation. Candidate chips may be evaluated while seeding and solving; the
desired specification itself is never evaluated.

A cross-Kerr target alone generally does not determine the sign of a
capacitive coupling. quchip preserves an explicitly supplied starting sign or
uses a documented positive convention and records that choice.

## Result and errors

The result records:

- every target, its source (`component default` or `explicit`), and its final
  residual;
- every varied bare parameter, starting value, bound, and final value;
- automatic seeding and sign choices;
- evaluator, Jacobian, convergence, and identifiability information;
- the fitted chip and objective history.

Invalid or underdetermined automatic plans fail before returning an arbitrary
chip. Overdetermined plans are allowed and reported as compromises. Legacy
keywords remain temporarily available with their existing semantics; they must
not be silently reinterpreted as the new desired-chip contract.

## Implementation boundaries

- Component classes own default target and free-parameter declarations.
- `inverse_design/observables.py` normalizes and compiles constraints without
  evaluating the desired chip.
- `inverse_design/fit.py` constructs starting values and solves the compiled
  problem.
- `inverse_design/types.py` owns structured plans and results.
