# Guide redesign

Status: implemented in the topic-guide redesign. The published static
reproduction remains pending.

## Structure

The documentation will have four SQA-independent guides:

1. Statics and parameter studies
2. Dynamics, pulses, observables, and readout
3. Chip transformations
4. Differentiability

Each guide will be fully readable on `docs.quchip.org` and backed by an
executed Markdown/notebook pair. The SQA page may show a small runnable
snippet for each topic, followed by one `docs.quchip.org` link per topic.
It will not contain a second version of the guides.

Reader-facing guide links will point to `docs.quchip.org`, not GitHub. The
current source/notebook/download link clusters will be removed. An optional
hosted notebook link may appear once at the end of a guide.

## 1. Statics and parameter studies

This guide contains no `QuantumSequence` or time evolution.

It will progress from declaring a chip and reading bare and dressed quantities
to changing one parameter with `with_params`, sweeping one frequency, and
interpreting an avoided crossing. It will then cover multiple independent and
zipped static parameters, dressed-state assignments and overlaps, extracting
observables from sweep results, and numerical convergence.

Static quantities should include frequencies, anharmonicities, dispersive
shifts, static ZZ, state assignments, and matrix elements where appropriate.

A published reproduction will be added after a suitable paper and target
figure have been agreed. This part is pending.

## 2. Dynamics, pulses, observables, and readout

The guide will begin with one pulse and one population trace. It will then add
pulse amplitude, width, carrier frequency, phase, detuning, DRAG, delays,
barriers, virtual Z operations, and composed sequences.

It will show how to inspect populations, expectation values, overlaps, states,
reduced states, leakage, and truncation. It will cover pulse-parameter batches,
relaxation and dephasing, coupled qubit-resonator dynamics, static dispersive
readout quantities, driven resonator readout, and IQ separation.

The text must distinguish static readout analysis from simulated resonator
dynamics and state the limits of the represented measurement model.

## 3. Chip transformations

The guide will cover parameter rebinding, frames and approximations, dressed
representations, eliminating devices and couplings, Schrieffer-Wolff versus
exact elimination, active patches, control-line retargeting, replaying an
experiment, partitioning, fitting dressed parameters, serialization, and
supported interoperability.

Every worked structural transformation will show the model before and after,
what was retained or folded into the result, its validity information, and a
relevant observable comparison against the original model.

## 4. Differentiability

The guide will have three stages.

### Losses through statics

Begin with a scalar loss and vector residual built from public static
observables. Show the gradient, Jacobian, named parameters, array shapes,
parameter scaling, and a finite-difference check.

### Losses through simple dynamics

Use one pulse on one qubit. Define losses from final population and leakage,
differentiate them with respect to chip and pulse parameters, check the
derivatives independently, and use them in a small calibration step.

### Losses through multi-sequence analysis

Define several experiments that share the same physical parameters. Possible
experiments include Rabi, Ramsey, spectroscopy-like evolution, and readout.
Collect their residuals into one vector and form a weighted joint loss. Show
both the Jacobian separated by experiment and the gradient of the combined
loss with respect to shared chip and control parameters.

## Writing and examples

Each section will pose a concrete physics question, show the smallest runnable
code that answers it, present the executed result, explain the result, and then
extend the example by one idea. Examples will use public quchip APIs and will
not replace the declared model with hand-written solver operators.

The prose should be direct and specific. Avoid repeated template headings,
generic transitions, summary paragraphs that repeat the section, and claims
that the code has not demonstrated.

## Verification

- Execute every canonical notebook from a clean kernel.
- Keep Jupytext Markdown and notebook code cells identical.
- Test numerical receipts and SQA snippets.
- Check plots visually and label their quantities and units.
- Build the Sphinx documentation and inspect the rendered guide path.
- Check that instructional links resolve under `docs.quchip.org`.
- Run the focused example tests, project lint, and the final prose audit.
