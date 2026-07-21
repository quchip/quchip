# Cookbook

This cookbook collects the conventions that make quchip recipes physically legible and reproducible, followed by task-oriented recipes that link to complete executed notebooks.

## Core conventions

### Declare the physics first

Put parameters at the top, then construct devices, couplings, the chip, control lines, and a sequence in that order. Use quchip device and control abstractions instead of raw QuTiP operators, and do not hand-write an effective Hamiltonian in place of the physical model. Prefer object references over string labels.

quchip uses GHz for ordinary frequencies, ns for time, and mK for temperature. Set the frame and chip RWA policy explicitly, and state the model approximation and Hilbert-space truncation that matter for the result. A drive constructed with its default `rwa=None` inherits the chip policy when the Hamiltonian template is resolved.

### Prepare the intended state

Use `chip.state()` by default. It returns the dressed eigenstate assigned to the requested bare labels and is safe inside `jax.jit`, `grad`, and `vmap`. Use `chip.bare_state()` only when the question intentionally concerns a bare product state, such as a swap or transfer experiment; explain that choice where it appears.

### Let analysis dress automatically

Methods such as `chip.freq()` and dispersive analyses dress the chip when needed. Do not call `chip.dress()` explicitly before them. A resonant control sequence should normally use the dressed transition returned by the chip, not the device's bare declaration.

### Let the model choose the solver

QuTiP is the default backend. When no solver is specified, quchip selects `mesolve` if the declared model contributes collapse operators and `sesolve` otherwise. Noise parameters such as a resonator quality factor therefore change the equation being solved without changing the sequence.

### Sweep through the batch API

For a sweep, vary a scheduled pulse handle or the `initial_state` and call `sequence.simulate_batch()` with batch axes. Zip fields that must change together, such as a pulse's duration and amplitude. This keeps the parameter structure attached to the sequence and avoids manual simulation loops.

### Read stored results directly

Solves store the state trajectory and final state by default, including when `e_ops` are requested. A `SimulationResult` can therefore provide `population()` and `plot_populations()` from the same run; a `SimulationBatchResult` stacks `population()` and `expect()` over its sweep axes. Set `options={"store_states": False}` only when the trajectory is unnecessary.

### Derive checks and report receipts

Choose tolerances from solver accuracy, Hilbert truncation, approximation validity, spectral bandwidth, or a simple physical estimate. Do not turn a previous output into the expected answer. Report compact, machine-readable JSON receipts so a reader can judge the actual run.

### Keep outputs focused

Explain the flow in prose rather than print narration. Make the smallest figure set that answers the question, label axes with units, and make any traced-out subsystem explicit.

## Compare Gaussian drive leakage

### Purpose

Drive a multilevel transmon at its dressed $0\rightarrow1$ transition and show how pulse bandwidth controls population of $|2\rangle$.

### Assumptions

The transmon uses the Duffing approximation and remains capacitively coupled to the declared lossy resonator. The chip sets the rotating frame and RWA; a `ChargeDrive` with no explicit `rwa` argument inherits that policy. Retain enough transmon levels for the broader pulse's leakage to be physical.

### Minimal usage

Read `f01 = chip.freq(qubit)` and `f12 = chip.freq(qubit, when={qubit: 1})`. For a Gaussian with temporal standard deviation $\sigma_t$, compare a short pulse whose spectral width $1/(2\pi\sigma_t)$ approaches $|f_{12}-f_{01}|$ with a longer selective pulse of the same shape. Integrate each unit-amplitude waveform and set its amplitude so $2\pi\int E(t)\,dt=\pi$; this is a nominal-pi area prescription, not a simulated calibration.

### Expected receipt

Report both dressed frequencies, both durations, final $P_1$, and peak $P_2$. The long pulse should end near $|1\rangle$ with materially less leakage than the short pulse.

### Common mistake

Do not choose durations by matching an earlier output or optimize the amplitude until one simulation reaches a desired population. Derive duration from the dressed adjacent-line separation and amplitude from the waveform integral, then let the multilevel solve reveal the leakage.

### Full notebook

Read the full {doc}`Hello, drive and readout example <examples/hello-chip>` or {download}`download the executed notebook <../examples/00_hello_chip.ipynb>`.

## Read out one dressed qubit

### Purpose

Perform dispersive readout by resolving the resonator response conditioned on the dressed qubit state using the same declared chip.

### Assumptions

The readout is a truncated lossy linear resonator, and the transmon-resonator interaction uses the chip's declared RWA. Its finite quality factor contributes photon loss, so the automatic solver selects `mesolve` on the default QuTiP backend.

### Minimal usage

Read both conditional frequencies with `chip.freq(readout, when={qubit: level})`, drive their midpoint with one Gaussian-edge flat-top pulse, vary dressed $|0,0\rangle$ and $|1,0\rangle$ through `initial_state`, and call `simulate_batch` once. Derive the duration as the larger of five resonator lifetimes and half the inverse conditional pull. Record local $a$ to recover the two IQ paths. Advanced readout statistics will be treated in a later analysis example.

### Expected receipt

Report the conditional frequencies, midpoint carrier, rounded duration, final IQ separation, and solver. Plot both IQ paths with emphasized final points and equal aspect ratio.

### Common mistake

Driving at the bare `readout.freq` ignores qubit-state-dependent dressing. Do not replace the pulse-level model with a hand-written effective Hamiltonian, use a steady-state shortcut, or add synthetic acquisition clouds.

### Full notebook

Read the full {doc}`Hello, drive and readout example <examples/hello-chip>` or {download}`download the executed notebook <../examples/00_hello_chip.ipynb>`.
