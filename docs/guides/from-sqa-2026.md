# From the SQA 2026 talk

Start with one question and one chip. Each page below begins with a short
public-API example, then links to the executed notebook behind the corresponding
talk figure.

quchip uses GHz for frequencies and ns for time. Install the base package for
the first three examples. The gradient example uses the JAX-native dynamiqs
backend.

```bash
python -m pip install quchip
python -m pip install 'quchip[dynamiqs]'
```

| Question | Small example | Then reproduce |
| --- | --- | --- |
| What are this chip's dressed frequencies? | {doc}`Resolve and sweep <../examples/resolve-and-sweep>` | The bus-mediated avoided crossing in Fig. 1 |
| What does one pulse do? | {doc}`Drive and read out <../examples/hello-chip>` | Leakage and conditional readout paths |
| Can I run the same schedule on a smaller model? | {doc}`Reduce and replay <../examples/reduce-and-replay>` | The active-patch comparison |
| How does an objective change with the design parameters? | {doc}`Gradient and Jacobian <../examples/differentiate-a-driven-chip>` | The checked pulse gradient in Fig. 6 |

The examples declare devices and couplings directly so the physical assumptions
stay in view. A `Chip` can also come from fitted dressed observables with
`fit_a_dress()` or from a scqubits model with `from_scqubits()`.

## A useful order

Read a dressed quantity with `chip.freq()` before sweeping anything. Run one
sequence before batching it. Call `active_patch()` only after the full schedule
exists. Differentiate a small vector of static observables before tracing a
time-domain solve.

The downloaded notebooks add the slide parameters, plots, approximation
records, and numerical checks without changing those public-API patterns.
