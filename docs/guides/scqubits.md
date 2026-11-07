# Import circuits from scqubits

[scqubits](https://scqubits.readthedocs.io) computes the spectra of
superconducting circuits from their circuit energies. `from_scqubits()` turns
a scqubits device or `HilbertSpace` into a quchip device or chip, and
`to_scqubits()` exports a quchip model back. Install the optional dependency
with `pip install quchip[scqubits]`. Energies are in GHz. Run the cells in
order.

## Import a transmon

A scqubits `Transmon` becomes a `ChargeBasisTransmon`, which diagonalizes the
same Cooper-pair-box Hamiltonian in the charge basis. `truncated_dim` sets the
number of energy levels that the device keeps.

```python
import scqubits as scq
from quchip import Chip, DuffingTransmon, from_scqubits, to_scqubits

transmon = scq.Transmon(EJ=30.0, EC=0.2, ng=0.25, ncut=31, truncated_dim=5, id_str="q")
q = from_scqubits(transmon)
single = Chip([q])

f01 = float(single.freq(q))
alpha = float(single.transition_frequency(q, 1, 2)) - f01
print(f"{type(q).__name__} {q.label!r}: f01 = {f01:.9f} GHz, alpha = {alpha:.9f} GHz")
print(f"scqubits: f01 = {transmon.E01():.9f} GHz, alpha = {transmon.anharmonicity():.9f} GHz")
```

Output:

```text
ChargeBasisTransmon 'q': f01 = 6.721939754 GHz, alpha = -0.215016213 GHz
scqubits: f01 = 6.721939754 GHz, alpha = -0.215016213 GHz
```

The labels come from `id_str`. Pass `label=`, `levels=` or noise options such
as `T1` to `from_scqubits()` to set them on the quchip device.

## Import a coupled system

A `HilbertSpace` becomes a chip. Each subsystem becomes a frozen snapshot of
its eigenbasis, and each interaction a coupling between the snapshots. The
chip uses `Exact()`, so it keeps the counter-rotating terms of the
interaction.

```python
oscillator = scq.Oscillator(E_osc=6.0, truncated_dim=6, id_str="r")
space = scq.HilbertSpace([transmon, oscillator])
space.add_interaction(g=0.05, op1=transmon.n_operator, op2=oscillator.creation_operator, add_hc=True)
chip = from_scqubits(space)

space.generate_lookup()
energy = space.energy_by_bare_index
reference = (energy((1, 1)) - energy((1, 0))) - (energy((0, 1)) - energy((0, 0)))
print(f"quchip chi: {float(chip.dispersive_shift('q', 'r')) * 1e3:.6f} MHz")
print(f"scqubits chi: {reference * 1e3:.6f} MHz")
```

Output:

```text
quchip chi: -5.181478 MHz
scqubits chi: -5.181478 MHz
```

Both values are the full dispersive pull `f_r|1 - f_r|0` of the dressed
spectrum. The chip is now an ordinary quchip chip: add ports, drives and
reductions as for any other chip. A snapshot keeps its matrix elements at the
imported circuit energies, so quchip cannot differentiate through `EJ` or
`EC`. Import again after you change them in scqubits.

## Export a model

`to_scqubits()` exports a device or a chip. A `DuffingTransmon` becomes a
`Transmon` whose `EJ` and `EC` reproduce its frequency and anharmonicity to
within 1 kHz in the charge basis, with `ncut=30` by default.

```python
duffing = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="d")
exported = to_scqubits(duffing)
print(f"EJ = {exported.EJ:.6f} GHz, EC = {exported.EC:.6f} GHz")
print(f"f01 = {exported.E01():.6f} GHz, alpha = {exported.anharmonicity():.6f} GHz")
```

Output:

```text
EJ = 15.347593 GHz, EC = 0.223110 GHz
f01 = 5.000000 GHz, alpha = -0.249999 GHz
```

A chip exports to a `HilbertSpace` with its supported couplings. The export
omits drives and other control equipment with a warning, and it does not
support port networks or effective terms. The API reference lists the
[device mappings](../api.md) in `quchip.interop.scqubits.devices`.
