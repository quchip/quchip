---
orphan: true
---

# SQA 2026 examples

The talk showed five short calculations. Each one is now part of a page that
runs it in full, with its output and checks.

- Your first chip: declare a transmon and a resonator and read their
  Hamiltonian, in {doc}`your first chip <../get-started/first-chip>`.
- Calculate and sweep the spectrum: change one bare frequency and follow the
  dressed transitions, in {doc}`spectra and parameter sweeps <spectra-and-sweeps>`.
- Simulate one pulse: drive a transmon with a Gaussian pulse, in
  {doc}`your first simulation <../get-started/first-simulation>` and
  {doc}`pulses, leakage, and readout <pulses-and-readout>`.
- Transform a chip: remove a resonator with `eliminate()` and check its
  validity, in {doc}`model reduction <model-reduction>`.
- Differentiate a spectral fitting objective: the loss gradient and the
  Jacobian of $f_{01}$ and $\chi_Z$, in
  {doc}`gradients and parameter fitting <gradients-and-fitting>`.

quchip uses GHz for frequencies and couplings, ns for time, and mK for
temperature.
