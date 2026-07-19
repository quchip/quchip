"""Tests for the declarative parameter sweep framework."""

from __future__ import annotations

import numpy as np
import pytest

from quchip import ChargeDrive, ControlEquipment, Crosstalk, FluxDrive
from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.control.sequence import QuantumSequence
from quchip.control.envelopes import Square
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.results.results import SimulationBatchResult
from quchip.sweep import SpectrumSweep, Sweep, ZippedSweep


# ---------------------------------------------------------------------------
# Sweep construction
# ---------------------------------------------------------------------------


class TestSweepConstruction:
    def test_sweep_basic(self):
        """A Sweep exposes its name, size, and values."""
        s = Sweep(np.linspace(5.0, 6.0, 3), name="freq")
        assert s.name == "freq"
        assert s.size == 3
        assert len(s.values) == 3
        np.testing.assert_allclose(s.values, [5.0, 5.5, 6.0])

    def test_sweep_from_list(self):
        """An unnamed Sweep defaults its name to "unnamed"."""
        s = Sweep([1, 2, 3])
        assert s.name == "unnamed"
        assert s.size == 3

    def test_sweep_repr(self):
        """repr(sweep) contains the class name, sweep name, and size."""
        s = Sweep([1, 2, 3], name="g")
        r = repr(s)
        assert "Sweep" in r
        assert "g" in r
        assert "3" in r


# ---------------------------------------------------------------------------
# ZippedSweep
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Backend batched_sesolve default implementation
# ---------------------------------------------------------------------------


class TestBatchedProtocol:
    def test_default_batched_sesolve_loops(self):
        """Backend.batched_sesolve default impl calls sesolve for each problem."""
        from unittest.mock import MagicMock

        mock_backend = MagicMock()
        from quchip.backend.protocol import Backend

        problems = [
            {"H": "h1", "psi0": "p1", "tlist": [0, 1]},
            {"H": "h2", "psi0": "p2", "tlist": [0, 1]},
        ]
        Backend.batched_sesolve(mock_backend, problems)
        assert mock_backend.sesolve.call_count == 2


# ---------------------------------------------------------------------------
# Integration: QuTiPBackend.batched_sesolve
# ---------------------------------------------------------------------------


class TestQuTiPBatchedIntegration:
    def test_qutip_batched_sesolve_integration(self):
        """Integration: QuTiPBackend.batched_sesolve produces valid SolverResults."""
        from quchip.backend.qutip import QuTiPBackend
        from quchip.backend import SolverResult

        backend = QuTiPBackend()
        # Simple 2-level system: H = number op (diagonal), psi0 = |0>
        H = backend.number(2)
        psi0 = backend.basis(2, 0)
        tlist = [0.0, 1.0, 2.0]

        problems = [
            {"H": H, "psi0": psi0, "tlist": tlist},
            {"H": H, "psi0": psi0, "tlist": tlist},
        ]
        results = backend.batched_sesolve(problems, progress=False, n_jobs=1)

        assert len(results) == 2
        for r in results:
            assert isinstance(r, SolverResult)
            assert r.states is not None or r.expect is not None
            assert r.solver == "sesolve"

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
                    parallel[element].population_array("q", level),
                    sequential[element].population_array("q", level),
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
            assert element.population_array("q", 0).shape == (11,)


# ---------------------------------------------------------------------------
# QuantumSequence.build_batch()
# ---------------------------------------------------------------------------


class TestQuantumSequenceBatch:
    def test_build_batch_returns_structured_problem_batch(self):
        """build_batch() over two independent axes returns a batch shaped as their product."""
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
            amp,
            freq,
            tlist=np.asarray([0.0, 10.0]),
            initial_state=chip.bare_state(q=0),
        )

        assert len(batch) == 4
        assert batch.shape == (2, 2)
        assert batch.params_at((1, 0)) == {"amp": 0.02, "freq": 4.9}

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

    def test_simulate_batch_uses_chip_solve_many(self, monkeypatch):
        """simulate_batch() dispatches every batch problem through chip.solve_many()."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        drive = ChargeDrive(target=q)
        chip = Chip([q])
        chip.connect(ControlEquipment(lines=[drive]))
        seq = QuantumSequence(chip)
        pulse = seq.schedule(drive, envelope=Square(duration=10.0, amplitude=0.02), freq=5.0)
        amp = pulse.vary("amplitude", [0.01, 0.02, 0.03], name="amp")

        captured: dict[str, object] = {}

        def fake_solve_many(problems, *, progress):
            captured["count"] = len(problems)
            return [f"batched_{idx}" for idx in range(len(problems))]

        monkeypatch.setattr(chip, "solve_many", fake_solve_many)
        # check_truncation=False isolates the dispatch wiring under test: the
        # fake returns a bare list, not a batch of SimulationResults to screen.
        result = seq.simulate_batch(
            amp,
            tlist=np.asarray([0.0, 10.0]),
            initial_state=chip.bare_state(q=0),
            progress=False,
            check_truncation=False,
        )

        assert captured["count"] == 3
        assert result == ["batched_0", "batched_1", "batched_2"]


class TestSimulationBatchResultAxes:
    def test_flat_batch_gets_named_batch_axis_and_repr(self):
        """A flat (unshaped) batch gets a default "batch" axis and a matching repr."""
        class _Backend:
            array_module = np

        class _Result:
            _backend = _Backend()

        results = [_Result() for _ in range(4)]
        batch = SimulationBatchResult(results)

        assert batch.shape == (4,)
        assert batch.axes == (("batch", (0, 1, 2, 3)),)
        assert batch[{"batch": 2}] is results[2]
        assert repr(batch) == "SimulationBatchResult(n=4, shape=(4,), axes=['batch'])"

    def test_population_and_expect_are_reshaped_to_sweep_axes(self):
        """population() and expect() reshape per-element traces to the batch's sweep-axis shape."""
        class _Backend:
            array_module = np

        class _Result:
            _backend = _Backend()

            def __init__(self, offset: float) -> None:
                self.offset = offset

            def population_array(self, device, level=0):
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
        assert batch[{"amp": 1, "freq": 2}] is results[5]

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
    def test_chip_dress_recomputes_without_serializing_model_state(self, monkeypatch):
        """dressed_spectrum() recomputes after a parameter change without serializing model state."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=4, label="r")
        coupling = Capacitive(q, r, g=0.05)
        chip = Chip([q, r], [coupling])

        def fail(*args, **kwargs):
            raise AssertionError("dress() should not serialize model state")

        monkeypatch.setattr(q, "to_dict", fail)
        monkeypatch.setattr(r, "to_dict", fail)
        monkeypatch.setattr(coupling, "to_dict", fail)

        original = np.asarray(chip.dressed_spectrum(), dtype=float)
        chip["r"].freq = 6.75
        updated = np.asarray(chip.dressed_spectrum(), dtype=float)

        assert not np.allclose(updated, original)

    def test_chip_sweep_basic_shape_and_lookup(self):
        """SpectrumSweep produces eigenvalues shaped (n_points, evals_count) with dressed lookups."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=4, label="r")
        chip = Chip([q, r], [Capacitive(q, r, g=0.05)])

        freq_axis = Sweep([6.8, 7.0], name="r_freq")

        def update_fn(chip: Chip, params: dict[str, float]) -> None:
            chip["r"].freq = params["r_freq"]

        result = SpectrumSweep(
            chip,
            [freq_axis],
            update_fn=update_fn,
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

        def update_fn(chip_point: Chip, params: dict[str, float]) -> None:
            chip_point["r"].freq = params["r_freq"]

        SpectrumSweep(
            chip,
            [Sweep([6.8, 7.1], name="r_freq")],
            update_fn=update_fn,
        ).run(progress=False)

        assert chip["r"].freq == original_freq

    def test_chip_sweep_without_eigenstate_storage_rejects_component_query(self):
        """state_components_at() raises when eigenstates were not stored during the sweep."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=4, label="r")
        chip = Chip([q, r], [Capacitive(q, r, g=0.05)])

        sweep_result = SpectrumSweep(
            chip,
            [Sweep([7.0], name="r_freq")],
            update_fn=lambda chip, params: setattr(chip["r"], "freq", params["r_freq"]),
            evals_count=4,
            store_eigenstates=False,
        ).run(progress=False)

        with pytest.raises(ValueError, match="not stored"):
            sweep_result.state_components_at(0, {"q": 0, "r": 0})

    def test_chip_sweep_marks_low_overlap_labels_as_nan(self):
        """SpectrumSweep marks a bare-label trajectory NaN once its dressed overlap drops too low."""
        r_a = Resonator(freq=6.0, levels=4, label="r_a")
        r_b = Resonator(freq=6.0, levels=4, label="r_b")
        coupling = Capacitive(r_a, r_b, g=0.0)
        chip = Chip([r_a, r_b], [coupling])

        g_axis = Sweep([0.0, 0.05], name="g")

        def update_fn(chip: Chip, params: dict[str, float]) -> None:
            chip.couplings[0].g = params["g"]

        result = SpectrumSweep(chip, [g_axis], update_fn=update_fn, evals_count=8).run(progress=False)

        trajectory = result.energy_by_bare_label(r_a=0, r_b=3)
        indices = result.dressed_index(r_a=0, r_b=3)

        assert np.isfinite(trajectory[0])
        assert np.isnan(trajectory[1])
        assert np.isfinite(indices[0])
        assert np.isnan(indices[1])

    def test_chip_sweep_uses_structural_clone_not_serialization(self, monkeypatch):
        """SpectrumSweep clones chip points structurally, without serializing chip or equipment."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=4, label="r")
        coupling = Capacitive(q, r, g=0.05)
        chip = Chip([q, r], [coupling])

        readout = ChargeDrive(target=r, label="readout")
        flux = FluxDrive(target=q, label="flux")
        equipment = ControlEquipment(
            lines=[readout, flux],
            signal_chain=[Crosstalk(source=readout.label, victim=flux.label, beta=0.15, theta=0.3, delay=2.0)],
        )
        chip.connect(equipment)

        def fail(*args, **kwargs):
            raise AssertionError("sweep() should not serialize topology to clone chip points")

        monkeypatch.setattr(chip, "to_dict", fail)
        monkeypatch.setattr(q, "to_dict", fail)
        monkeypatch.setattr(r, "to_dict", fail)
        monkeypatch.setattr(coupling, "to_dict", fail)
        monkeypatch.setattr(equipment, "to_dict", fail)
        monkeypatch.setattr(readout, "to_dict", fail)
        monkeypatch.setattr(flux, "to_dict", fail)
        monkeypatch.setattr(Crosstalk, "to_dict", fail)

        result = SpectrumSweep(
            chip,
            [Sweep([6.8, 7.1], name="r_freq")],
            update_fn=lambda chip_point, params: setattr(chip_point["r"], "freq", params["r_freq"]),
            evals_count=4,
        ).run(progress=False)

        assert result.eigenvalues.shape == (2, 4)
        assert chip["r"].freq == pytest.approx(7.0)


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
            [Sweep([], name="r_freq")],
            update_fn=lambda c, p: setattr(c["r"], "freq", p["r_freq"]),
        )
        with pytest.raises(ValueError, match="r_freq"):
            sweep.run(progress=False)

    @pytest.mark.parametrize("bad_evals_count", [0, -1, 2.5, 100, True])
    def test_invalid_evals_count_raises(self, bad_evals_count):
        """evals_count outside [1, chip.total_dim], non-integral, or bool raises ValueError."""
        chip = self._chip()
        sweep = SpectrumSweep(
            chip,
            [Sweep([7.0], name="r_freq")],
            update_fn=lambda c, p: setattr(c["r"], "freq", p["r_freq"]),
            evals_count=bad_evals_count,
        )
        with pytest.raises(ValueError, match="evals_count"):
            sweep.run(progress=False)

    def test_evals_count_equal_to_total_dim_passes(self):
        """evals_count exactly equal to chip.total_dim is accepted."""
        chip = self._chip()
        sweep = SpectrumSweep(
            chip,
            [Sweep([7.0], name="r_freq")],
            update_fn=lambda c, p: setattr(c["r"], "freq", p["r_freq"]),
            evals_count=chip.total_dim,
        )
        result = sweep.run(progress=False)
        assert result.eigenvalues.shape == (1, chip.total_dim)

    def test_update_fn_changing_device_levels_raises(self):
        """update_fn changing a device's levels mid-sweep raises ValueError naming the grid point."""
        chip = self._chip()

        def update_fn(chip_point, params):
            chip_point["r"].freq = params["r_freq"]
            if params["r_freq"] == 7.1:
                chip_point["r"].levels = 5

        sweep = SpectrumSweep(chip, [Sweep([7.0, 7.1], name="r_freq")], update_fn=update_fn)
        with pytest.raises(ValueError, match="topology"):
            sweep.run(progress=False)

    def test_int_point_on_multi_dimensional_grid_raises(self):
        """An int point index against a multi-D sweep grid raises ValueError from _normalize_point."""
        chip = self._chip()

        def update_fn(chip_point, params):
            chip_point["r"].freq = params["r_freq"]
            chip_point.couplings[0].g = params["g"]

        result = SpectrumSweep(
            chip,
            [Sweep([6.8, 7.0], name="r_freq"), Sweep([0.01, 0.02], name="g")],
            update_fn=update_fn,
            evals_count=5,
            store_eigenstates=True,
        ).run(progress=False)

        with pytest.raises(ValueError, match="coordinate"):
            result.state_components_at(0, {"q": 0, "r": 0})
