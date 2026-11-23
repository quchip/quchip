# Studies

Each study answers one physical question from start to end. It states the
model assumptions, compares the result with published data or a known limit,
and checks the numerics. The notebook studies take longer to run than the
guides.

- {doc}`Purcell filtering and T1 <purcell-filter>`: how much can a Purcell
  filter suppress qubit decay and keep the readout bandwidth?
- {doc}`Fluxonium spectroscopy <fluxonium-spectroscopy>`: does one model
  reproduce published fluxonium and readout data, and does a gradient fit
  recover the circuit energies?
- {doc}`Resonator depletion with CLEAR <resonator-depletion>`: how much faster
  does a CLEAR pulse empty a filtered readout resonator than waiting?
- {doc}`What happens during a qubit measurement? <continuous-measurement>`:
  follow single measurement trajectories, map their density, and condition on
  an endpoint.

For the main quchip workflows, see the {doc}`guides <../guides/index>`.

```{toctree}
:hidden:
:maxdepth: 1

Purcell filtering and T1 <purcell-filter>
Fluxonium spectroscopy <fluxonium-spectroscopy>
Resonator depletion with CLEAR <resonator-depletion>
What happens during a qubit measurement? <continuous-measurement>
```
