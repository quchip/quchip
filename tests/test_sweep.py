"""Tests for the declarative parameter sweep framework."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import ChargeDrive, ControlEquipment
from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.control.sequence import QuantumSequence
from quchip.control.envelopes import Square
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.results.results import SimulationBatchResult
from quchip.sweep import SpectrumSweep, Sweep, ZippedSweep


class TestZippedSweep:
    def test_zip_equal_size(self):
        """Sweep.zip() of equal-length sweeps produces a ZippedSweep of that length."""
        a = Sweep([1, 2], name="a")
        b = Sweep([3, 4], name="b")
        z = Sweep.zip(a, b)
        assert isinstance(z, ZippedSweep)
        assert z.size == 2

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


class TestExpand:
    def test_cartesian_product(self):
        """Sweep.expand() of independent sweeps produces their Cartesian product."""
        x = Sweep([1, 2], name="x")
        y = Sweep([3, 4, 5], name="y")
        combos = Sweep.expand([x, y])
        assert len(combos) == 6  # 2 * 3
        for c in combos:
            assert "x" in c
            assert "y" in c

    def test_zipped_expansion(self):
        """Sweep.expand() of a ZippedSweep pairs constituent values index-wise, not as a product."""
        a = Sweep([1, 2], name="a")
        b = Sweep([3, 4], name="b")
        z = Sweep.zip(a, b)
        combos = Sweep.expand([z])
        assert len(combos) == 2
        assert combos[0] == {"a": 1, "b": 3}
        assert combos[1] == {"a": 2, "b": 4}

    def test_zipped_with_independent(self):
        """ZippedSweep combined with independent Sweep produces Cartesian product across groups."""
        a = Sweep([1, 2], name="a")
        b = Sweep([3, 4], name="b")
        z = Sweep.zip(a, b)
        c = Sweep([10, 20, 30], name="c")
        combos = Sweep.expand([z, c])
        assert len(combos) == 6  # 2 zipped * 3 independent


class TestDuplicateAxisNames:
    def test_duplicate_names_across_independent_axes_raise(self):
        """Two independent axes sharing a name raise ValueError instead of silently overwriting."""
        x1 = Sweep([1, 2], name="x")
        x2 = Sweep([3, 4], name="x")
        with pytest.raises(ValueError, match="x"):
            Sweep.expand([x1, x2])

    def test_duplicate_names_within_a_zip_raise(self):
        """Two members of the same ZippedSweep sharing a name raise ValueError."""
        a = Sweep([1, 2], name="x")
        b = Sweep([3, 4], name="x")
        with pytest.raises(ValueError, match="x"):
            Sweep.expand([Sweep.zip(a, b)])

    def test_zip_member_colliding_with_independent_axis_raises(self):
        """A zipped member sharing a name with an independent axis raises ValueError."""
        a = Sweep([1, 2], name="x")
        b = Sweep([3, 4], name="y")
        c = Sweep([5, 6], name="x")
        with pytest.raises(ValueError, match="x"):
            Sweep.expand([Sweep.zip(a, b), c])


@pytest.mark.parametrize("target", ["pulse", "device", "state", "delay"])
@pytest.mark.parametrize("grouping", ["independent", "zipped", "mixed"])
def test_batch_rejects_duplicate_targets_with_different_names(target, grouping):
    chip = Chip([DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="q")])
    drive = ChargeDrive(chip["q"], label="xy")
    chip.wire(drive)
    seq = QuantumSequence(chip)
    pulse = seq.schedule(drive, envelope=Square(duration=1.0, amplitude=0.1))
    if target == "pulse":
        first = pulse.vary("amplitude", [0.1, 0.2], name="first")
        second = seq.vary("pulse.0.amplitude", [0.3, 0.4], name="second")
    elif target == "device":
        first = seq.vary("q.freq", [5.0, 5.1], name="first")
        second = seq.vary("q.freq", [5.2, 5.3], name="second")
    elif target == "state":
        first = seq.vary("initial_state", [{"q": 0}, {"q": 1}], name="first")
        second = seq.vary("initial_state", [{"q": 1}, {"q": 0}], name="second")
    else:
        delay = seq.delay("q", 1.0)
        first = delay.vary("duration", [1.0, 2.0], name="first")
        second = delay.vary("duration", [3.0, 4.0], name="second")
    if grouping == "zipped":
        axes = (seq.zip(first, second),)
    elif grouping == "mixed":
        axes = (seq.zip(first, pulse.vary("phase", [0.0, 0.5], name="phase")), second)
    else:
        axes = (first, second)
    with pytest.raises(ValueError, match="same parameter"):
        seq.build_batch(*axes, tlist=np.linspace(0, 6, 7))


class TestQuTiPBatchedIntegration:
    def test_qutip_parallel_sweep_matches_sequential(self, monkeypatch):
        """The reusable-loky parallel batch path matches the in-process sequential one exactly."""
        from quchip.backend.qutip import QuTiPBackend

        amps = [0.005 * (k + 1) for k in range(12)]  # >= _PARALLEL_MIN_BATCH, exercises loky dispatch

        def run(force_sequential: bool):
            q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=4, label="q")
            drive = ChargeDrive(target=q)
            backend = QuTiPBackend()
            chip = Chip([q], backend=backend)
            chip.connect(ControlEquipment(lines=[drive]))
            if force_sequential:
                def _sequential_map(*, task, items, n_jobs, progress, desc):
                    return [task(item) for item in items]

                monkeypatch.setattr(backend, "_parallel_map", _sequential_map)
            seq = QuantumSequence(chip)
            pulse = seq.schedule(drive, envelope=Square(duration=10.0, amplitude=0.02), freq=5.0)
            amp = pulse.vary("amplitude", amps, name="amp")
            return seq.simulate_batch(
                amp,
                tlist=np.linspace(0.0, 10.0, 21),
                initial_state=chip.bare_state(q=0),
                progress=False,
            )

        parallel = run(force_sequential=False)
        sequential = run(force_sequential=True)

        assert len(parallel) == len(amps) == len(sequential)
        for element in range(len(amps)):
            for level in range(4):
                np.testing.assert_array_equal(
                    parallel[element].population("q", level),
                    sequential[element].population("q", level),
                )

    def test_qutip_warmup_is_safe_and_pool_works_after(self):
        """warmup() is idempotent and non-raising, and a parallel sweep still works after it."""
        from quchip.backend.qutip import QuTiPBackend

        backend = QuTiPBackend()
        backend.warmup()  # idempotent + non-raising
        backend.warmup()

        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q], backend=backend)
        chip.connect(ControlEquipment(lines=[drive]))
        seq = QuantumSequence(chip)
        pulse = seq.schedule(drive, envelope=Square(duration=10.0, amplitude=0.02), freq=5.0)
        # >= _PARALLEL_MIN_BATCH so the warmed pool is actually exercised.
        amp = pulse.vary("amplitude", [0.01 * (k + 1) for k in range(12)], name="amp")
        batch = seq.simulate_batch(
            amp,
            tlist=np.linspace(0.0, 10.0, 11),
            initial_state=chip.bare_state(q=0),
            progress=False,
        )
        assert len(batch) == 12
        for element in batch:
            assert element.population("q", 0).shape == (11,)


class TestQuantumSequenceBatch:
    def test_build_batch_supports_zipped_axes(self):
        """build_batch() with a zipped axis produces one batch element per index pair."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q])
        chip.connect(ControlEquipment(lines=[drive]))
        seq = QuantumSequence(chip)
        pulse = seq.schedule(
            drive,
            envelope=Square(duration=10.0, amplitude=0.02),
            freq=5.0,
        )

        amp = pulse.vary("amplitude", [0.01, 0.02], name="amp")
        freq = pulse.vary("freq", [4.9, 5.1], name="freq")
        batch = seq.build_batch(
            seq.zip(amp, freq),
            tlist=np.asarray([0.0, 10.0]),
            initial_state=chip.bare_state(q=0),
        )

        assert len(batch) == 2
        assert batch.shape == (2,)
        assert batch.params_at(1) == {"amp": 0.02, "freq": 5.1}

    def test_build_batch_supports_initial_state_axis(self):
        """build_batch() accepts a swept initial_state as a batch axis."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
        chip = Chip([q], frame="rotating")
        seq = QuantumSequence(chip)
        state = seq.vary("initial_state", [{"q": 0}, {"q": 1}], name="state")

        batch = seq.build_batch(
            state,
            tlist=np.linspace(0.0, 10.0, 11),
            e_ops=chip.e_ops(q="Z"),
        )

        assert len(batch) == 2
        assert batch.shape == (2,)

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

    def test_slice_returns_flat_batch_with_batch_axis(self):
        """Slicing a shaped batch returns a flat batch with a default "batch" axis."""
        class _Backend:
            array_module = np

        class _Result:
            _backend = _Backend()

        results = [_Result() for _ in range(6)]
        batch = SimulationBatchResult(
            results,
            shape=(2, 3),
            axes=(("amp", [0.1, 0.2]), ("freq", [4.9, 5.0, 5.1])),
        )

        sliced = batch[::2]

        assert sliced.shape == (3,)
        assert sliced.axes == (("batch", (0, 2, 4)),)
        assert sliced[{"batch": 1}] is results[2]

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

    def test_int_point_on_multi_dimensional_grid_raises(self):
        """An int point index against a multi-D sweep grid raises ValueError from _normalize_point."""
        chip = self._chip()

        result = SpectrumSweep(
            chip,
            [
                Sweep([6.8, 7.0], name="r.freq"),
                Sweep([0.01, 0.02], name=f"{chip.couplings[0].label}.g"),
            ],
            evals_count=5,
            store_eigenstates=True,
        ).run(progress=False)

        with pytest.raises(ValueError, match="coordinate"):
            result.state_components_at(0, {"q": 0, "r": 0})
