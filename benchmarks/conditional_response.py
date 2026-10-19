"""Measure conditional VNA sweeps and derivatives in one selected source checkout."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


def measure(call: Callable[[], Any], repeat: int) -> dict[str, Any]:
    """Record cold execution, warm-up, and synchronized repeated samples."""
    import jax
    import numpy as np

    def timed() -> float:
        start = time.perf_counter()
        jax.block_until_ready(call())
        return time.perf_counter() - start

    cold = timed()
    timed()
    samples = [timed() for _ in range(repeat)]
    return {"cold_seconds": cold, "warm_seconds": samples, "median_seconds": float(np.median(samples))}


def readout_chip(count: int, backend: str) -> Any:
    """Declare the shared-feedline transmon/readout ladder reported in issue 147."""
    import numpy as np
    from quchip import RWA, Capacitive, Chip, DuffingTransmon, PortNetwork, Resonator

    qubits = [DuffingTransmon(freq=5. + .1*i, anharmonicity=-.25, levels=3, label=f"q{i}", T1=50e3)
              for i in range(count)]
    readouts = [Resonator(freq=6.5 + .1*i, levels=3, label=f"r{i}", internal_quality_factor=2e5)
                for i in range(count)]
    couplings = [Capacitive(q, r, g=.04, label=f"q{i}_r{i}")
                 for i, (q, r) in enumerate(zip(qubits, readouts, strict=True))]
    couplings += [Capacitive(qubits[i], qubits[i + 1], g=.005, label=f"q{i}_q{i + 1}")
                  for i in range(count - 1)]
    line = PortNetwork(label="line")
    ports = [line.port(f"r{i}_port", target=r, rate=2*np.pi*.001) for i, r in enumerate(readouts)]
    line.cascade(*ports)
    line.expose("feed", input=ports[0].input, output=ports[-1].output)
    return Chip(qubits + readouts, couplings, port_network=line, frame=5., approximation=RWA(), backend=backend)


def run(source: Path, repeat: int) -> dict[str, Any]:
    """Compare public sweeps and backend coherence solves on the same CPU."""
    # Both revisions run this same harness, each in a fresh process.
    sys.path.insert(0, str(source.resolve()))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import quchip
    from quchip import VNA
    from quchip.engine.conditional_response import build_conditional_response

    if not Path(quchip.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError(f"Imported quchip from {quchip.__file__}, not from {source}.")

    jax.config.update("jax_enable_x64", True)
    report: dict[str, Any] = {
        "commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "dynamiqs", "qutip")},
        "devices": [str(device) for device in jax.devices()],
        "repeat": repeat,
        "cases": [],
    }
    for count in (2, 4):
        reference_chip = readout_chip(count, "qutip")
        center = float(reference_chip.freq("r0", when={"q0": 1}))
        for frequency_count in (5, 21):
            frequencies = center + np.linspace(-.002, .002, frequency_count)
            reference = np.asarray(VNA(reference_chip).sweep(frequencies, when={"q0": 1}).matrix)
            for backend in ("qutip", "dynamiqs"):
                chip = reference_chip if backend == "qutip" else readout_chip(count, backend)
                vna = VNA(chip)

                def sweep() -> Any:
                    return vna.sweep(frequencies, when={"q0": 1}).matrix

                sweep_times = measure(sweep, repeat)
                actual = np.asarray(sweep())
                np.testing.assert_allclose(actual, reference, rtol=1e-8, atol=1e-9)
                problem, metadata = build_conditional_response(
                    chip, frequencies, ("feed",), (1,) + (0,) * (2 * count - 1))
                solve_times = measure(lambda: chip.backend.coherence_response(problem).responses, repeat)
                case = {"pairs": count, "frequencies": frequency_count, "backend": backend,
                        "coherences": metadata["coherence_count"], "sweep": sweep_times, "solve": solve_times,
                        "max_reference_error": float(np.max(np.abs(actual - reference)))}
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
        frequencies = center + np.linspace(-.002, .002, 5)
        chip = readout_chip(count, "dynamiqs")

        def signal(frequency: Any) -> Any:
            value = VNA(chip.with_params({"q0.freq": frequency})).sweep(frequencies, when={"q0": 1}).matrix
            return jnp.stack((value.real, value.imag))

        step = 1e-5
        plus = VNA(reference_chip.with_params({"q0.freq": 5. + step})).sweep(frequencies, when={"q0": 1}).matrix
        minus = VNA(reference_chip.with_params({"q0.freq": 5. - step})).sweep(frequencies, when={"q0": 1}).matrix
        difference = (np.asarray(plus) - np.asarray(minus)) / (2 * step)
        expected = np.stack((difference.real, difference.imag))
        scalar_expected = (np.sum(np.abs(plus)**2) - np.sum(np.abs(minus)**2)) / (2 * step)
        compiled_signal = jax.jit(signal)
        case = {"pairs": count, "frequencies": 5, "backend": "dynamiqs",
                "compiled_sweep": measure(lambda: compiled_signal(5.), repeat)}
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
        derivatives = {
            "jacrev": jax.jit(jax.jacrev(signal)),
            "jacfwd": jax.jit(jax.jacfwd(signal)),
            "jvp": jax.jit(lambda f: jax.jvp(signal, (f,), (1.,))[1]),
            "scalar_reverse": jax.jit(jax.grad(lambda f: jnp.sum(signal(f)**2))),
        }
        for mode, derivative in derivatives.items():
            timing = measure(lambda: derivative(5.), repeat)
            actual = np.asarray(derivative(5.))
            reference = scalar_expected if mode == "scalar_reverse" else expected
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=1e-7)
            case = {"pairs": count, "frequencies": 5, "backend": "dynamiqs", "derivative_mode": mode,
                    "derivative": timing, "max_derivative_error": float(np.max(np.abs(actual - reference)))}
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(run(args.source, args.repeat), indent=2) + "\n")
