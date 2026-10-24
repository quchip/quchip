"""Tests for the declarative parameter sweep framework."""

from __future__ import annotations

import numpy as np
import pytest

from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.control.sequence import QuantumSequence
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.results.results import SimulationBatchResult
from quchip.sweep import SpectrumSweep, Sweep


class TestZippedSweep:

    def test_zip_mismatched_raises(self):
        """Sweep.zip() of mismatched-length sweeps raises ValueError."""
        a = Sweep([1, 2], name="a")
        b = Sweep([3, 4, 5], name="b")
        with pytest.raises(ValueError, match="equal lengths"):
            Sweep.zip(a, b)

    def test_zip_single_sweep_raises(self):
        """Sweep.zip() of a single sweep raises ValueError; zipping requires at least two axes."""
        a = Sweep([1, 2], name="a")
        with pytest.raises(ValueError, match="at least two"):
            Sweep.zip(a)


class TestSimulationBatchResultAxes:
    def test_sweep_coordinates_capture_the_built_model_values(self):
        """Source edits and returned metadata cannot relabel solved points."""
        from quchip.engine import solve_batch

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")
        sequence = QuantumSequence(Chip([q], frame="rotating"))
        frequencies = np.array([5.0, 6.0])
        batch = sequence.build_batch(sequence.vary("q.freq", frequencies), tlist=[0.0, 1.0])
        frequencies[:] = 9.0
        result = solve_batch(batch, progress=False)
        np.testing.assert_array_equal(batch.axes[0][1], [5.0, 6.0])
        np.testing.assert_array_equal(result.axes[0][1], [point.chip.freq("q") for point in batch])
        for index in (1.9, "1"):
            with pytest.raises(TypeError):
                batch[index]

        axes = (("a/b", ({"a": [1.0], "b": 2.0}, {"a": [3.0], "b": 4.0})),)
        annotated = result.with_sweep_metadata(shape=(2,), axes=axes)
        axes[0][1][0]["a"][0] = 7.0
        annotated.axes[0][1][0]["a"][0] = 8.0
        assert annotated.axes[0][1][0]["a"] == [1.0]

    def test_population_and_expect_are_reshaped_to_sweep_axes(self):
        """population() and expect() reshape per-element traces to the batch's sweep-axis shape."""
        class _Backend:
            array_module = np

        class _Result:
            _backend = _Backend()
            times = np.asarray([0.0, 1.0, 2.0])

            def __init__(self, offset: float) -> None:
                self.offset = offset

            def population(self, device, level=0):
                return np.asarray([self.offset, self.offset + 1.0, self.offset + 2.0])

            def expect(self, key, index=None):
                return np.asarray([self.offset + 10.0, self.offset + 20.0, self.offset + 30.0])

        results = [_Result(float(idx)) for idx in range(6)]
        batch = SimulationBatchResult(
            results,
            shape=(2, 3),
            axes=(("amp", [0.1, 0.2]), ("freq", [4.9, 5.0, 5.1])),
        )

        pops = batch.population("q", level=1)
        expected = np.asarray(
            [
                [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]],
            ]
        )

        assert batch.shape == (2, 3)
        assert batch.axes == (("amp", [0.1, 0.2]), ("freq", [4.9, 5.0, 5.1]))
        np.testing.assert_allclose(pops, expected)
        np.testing.assert_allclose(batch.population("q", level=1, reduce="last"), expected[..., -1])
        np.testing.assert_allclose(batch.expect("n", reduce="mean"), expected[..., 0] + 20.0)
        assert batch[{"amp": np.int64(1), "freq": 2}] is results[5]
        for index in (1.9, "1"):
            with pytest.raises(TypeError):
                batch[{"amp": index, "freq": 2}]


    def test_zipped_axis_can_be_indexed_by_constituent_names(self):
        """A zipped axis can be indexed by either constituent name, and inconsistent indices raise."""
        class _Backend:
            array_module = np

        class _Result:
            _backend = _Backend()

        results = [_Result() for _ in range(2)]
        batch = SimulationBatchResult(
            results,
            shape=(2,),
            axes=(("amp/freq", ({"amp": 0.01, "freq": 4.9}, {"amp": 0.02, "freq": 5.1})),),
        )

        assert batch[{"amp": 1}] is results[1]
        assert batch[{"freq": 1}] is results[1]
        assert batch[{"amp": 1, "freq": 1}] is results[1]
        with pytest.raises(ValueError, match="same index"):
            batch[{"amp": 0, "freq": 1}]


class TestSpectrumSweep:
    def test_chip_sweep_basic_shape_and_lookup(self):
        """SpectrumSweep produces eigenvalues shaped (n_points, evals_count) with dressed lookups."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=4, label="r")
        chip = Chip([q, r], [Capacitive(q, r, g=0.05)])

        freq_axis = Sweep([6.8, 7.0], name="r.freq")

        result = SpectrumSweep(
            chip,
            [freq_axis],
            evals_count=5,
            store_eigenstates=True,
        ).run(progress=False)

        assert result.eigenvalues.shape == (2, 5)
        assert result.dressed_index(q=1, r=0).shape == (2,)
        assert result.energy_by_bare_label(q=0, r=0).shape == (2,)
        components = result.state_components_at(0, {"q": 0, "r": 0}, n_components=3)
        assert components
        assert 0.0 < sum(components.values()) <= 1.0 + 1e-12

    def test_chip_sweep_does_not_mutate_original_chip(self):
        """SpectrumSweep leaves the original chip's parameters unchanged after running."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=4, label="r")
        chip = Chip([q, r], [Capacitive(q, r, g=0.05)])
        original_freq = chip["r"].freq

        SpectrumSweep(
            chip,
            [Sweep([6.8, 7.1], name="r.freq")],
        ).run(progress=False)

        assert chip["r"].freq == original_freq

    def test_chip_sweep_without_eigenstate_storage_rejects_component_query(self):
        """state_components_at() raises when eigenstates were not stored during the sweep."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=4, label="r")
        chip = Chip([q, r], [Capacitive(q, r, g=0.05)])

        sweep_result = SpectrumSweep(
            chip,
            [Sweep([7.0], name="r.freq")],
            evals_count=4,
            store_eigenstates=False,
        ).run(progress=False)

        with pytest.raises(ValueError, match="not stored"):
            sweep_result.state_components_at(0, {"q": 0, "r": 0})

    def test_chip_sweep_marks_low_overlap_labels_as_nan(self):
        """SpectrumSweep marks a bare-label trajectory NaN once its dressed overlap drops too low."""
        r_a = Resonator(freq=6.0, levels=4, label="r_a")
        r_b = Resonator(freq=6.0, levels=4, label="r_b")
        coupling = Capacitive(r_a, r_b, g=0.0, label="rr")
        chip = Chip([r_a, r_b], [coupling])

        g_axis = Sweep([0.0, 0.05], name="rr.g")

        result = SpectrumSweep(chip, [g_axis], evals_count=8).run(progress=False)

        trajectory = result.energy_by_bare_label(r_a=0, r_b=3)
        indices = result.dressed_index(r_a=0, r_b=3)

        assert np.isfinite(trajectory[0])
        assert np.isnan(trajectory[1])
        assert np.isfinite(indices[0])
        assert np.isnan(indices[1])

class TestSpectrumSweepValidation:
    def _chip(self):
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=4, label="r")
        return Chip([q, r], [Capacitive(q, r, g=0.05)])  # total_dim = 12

    def test_zero_length_axis_raises(self):
        """A zero-length sweep axis raises ValueError before SpectrumSweep.run() iterates."""
        chip = self._chip()
        sweep = SpectrumSweep(
            chip,
            [Sweep([], name="r.freq")],
        )
        with pytest.raises(ValueError, match=r"r\.freq"):
            sweep.run(progress=False)

    @pytest.mark.parametrize("bad_evals_count", [0, -1, 2.5, 100, True])
    def test_invalid_evals_count_raises(self, bad_evals_count):
        """evals_count outside [1, chip.total_dim], non-integral, or bool raises ValueError."""
        chip = self._chip()
        sweep = SpectrumSweep(
            chip,
            [Sweep([7.0], name="r.freq")],
            evals_count=bad_evals_count,
        )
        with pytest.raises(ValueError, match="evals_count"):
            sweep.run(progress=False)
