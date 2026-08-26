```{include} ../../examples/03_differentiate_a_driven_chip.md
:start-after: <!-- reader-content -->
:end-before: <!-- simple-example-end -->
```

The inputs are three design knobs and the outputs are two dressed observables,
so the gradient has shape `(3,)` and the Jacobian has shape `(2, 3)`.

## Continue with the full example

The full version replaces the static observables with a final qubit population.
It differentiates that population with respect to pulse amplitude, Gaussian
shape, and detuning, then checks all three derivatives with central differences.

```{figure} ../images/differentiate_a_driven_chip.png
:width: 720px
:alt: Final driven-qubit population with a local JAX tangent and finite-difference convergence below

The upper panel compares the predicted population changes with central finite
differences. The lower panel shows convergence as the finite-difference step
shrinks.
```

[View the full example source on GitHub](https://github.com/quchip/quchip/blob/main/examples/03_differentiate_a_driven_chip.md)
or [view the executed notebook](https://github.com/quchip/quchip/blob/main/examples/03_differentiate_a_driven_chip.ipynb).
You can also {download}`download the executed notebook <../../examples/03_differentiate_a_driven_chip.ipynb>`.
