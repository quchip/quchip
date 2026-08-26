"""The short guide examples remain copy-paste runnable."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
CODE_BLOCK_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


def _run_guide(path: str) -> dict[str, object]:
    source_path = ROOT / path
    source = source_path.read_text(encoding="utf-8")
    simple_source, marker, _ = source.partition("<!-- simple-example-end -->")
    assert marker, f"No simple-example marker found in {path}"
    blocks = CODE_BLOCK_RE.findall(simple_source)
    assert blocks, f"No Python examples found in {path}"

    namespace: dict[str, object] = {"__name__": "__guide_example__"}
    exec(compile("\n\n".join(blocks), str(source_path), "exec"), namespace)
    return namespace


@pytest.mark.examples
def test_resolve_and_sweep_starts_with_a_small_runnable_example() -> None:
    """The short statics guide executes and resolves the talk-scale splitting."""
    example = _run_guide("examples/01_resolve_and_sweep.md")

    minimum_splitting = float(np.min(example["splitting"]))
    assert 4.3e-3 < minimum_splitting < 4.5e-3


@pytest.mark.examples
def test_drive_guide_starts_with_one_pulse_and_one_solve() -> None:
    """The short drive guide returns one finite population trajectory."""
    example = _run_guide("examples/00_hello_chip.md")

    result = example["result"]
    population = np.asarray(result.population("q", level=1))
    assert population.shape == (161,)
    assert np.isfinite(population).all()


@pytest.mark.examples
def test_reduction_guide_replays_the_same_schedule() -> None:
    """The short reduction guide keeps the neighbourhood and replays its schedule."""
    example = _run_guide("examples/02_reduce_and_replay.md")

    assert example["patch"].active_labels == ("q0", "q1")
    assert example["patch"].eliminated_labels == ("q2",)
    residual = np.max(np.abs(example["p_full"] - example["p_small"]))
    assert residual < 1.0e-5


@pytest.mark.examples
@pytest.mark.optional_backend
def test_gradient_guide_has_the_promised_gradient_and_jacobian_shapes() -> None:
    """The short JAX guide returns the documented gradient and Jacobian shapes."""
    pytest.importorskip("dynamiqs")
    example = _run_guide("examples/03_differentiate_a_driven_chip.md")

    jax = example["jax"]
    theta = example["theta"]
    assert jax.grad(example["loss"])(theta).shape == (3,)
    assert jax.jacrev(example["residual"])(theta).shape == (2, 3)
