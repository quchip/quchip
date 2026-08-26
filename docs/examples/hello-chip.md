```{include} ../../examples/00_hello_chip.md
:start-after: <!-- reader-content -->
:end-before: <!-- simple-example-end -->
```

## Continue with the full example

The full version compares two pulse widths, then adds a resonator drive and
conditional readout paths using the same chip.

```{figure} ../images/hello_qubit_drive_leakage.png
:width: 760px
:alt: Short and long Gaussian pulses with multilevel qubit populations

Two nominal-pi Gaussians with different bandwidths expose multilevel leakage.
```

```{figure} ../images/hello_dispersive_readout_iq.png
:width: 560px
:alt: Conditional resonator IQ paths with emphasized final points

The resonator follows different IQ paths for prepared dressed qubit states
$|0\rangle$ and $|1\rangle$.
```

[View the full example source on GitHub](https://github.com/quchip/quchip/blob/main/examples/00_hello_chip.md)
or [view the executed notebook](https://github.com/quchip/quchip/blob/main/examples/00_hello_chip.ipynb).
You can also {download}`download the executed notebook <../../examples/00_hello_chip.ipynb>`.
