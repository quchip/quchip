"""Contract tests for the Hamiltonian description IR types."""

from __future__ import annotations

import numpy as np
import pytest

import quchip.engine.ir as ir
from quchip.engine.ir import (
    CanonicalOperator,
    Carrier,
    DynamicTerm,
    HamiltonianDescription,
    ScalarModulation,
    SolveProblem,
    StaticTerm,
)


# ── Template shape contract ──────────────────────────────────────────


def test_compiled_template_tracks_drive_terms_only() -> None:
    """HamiltonianTemplate must expose drive_terms but not crosstalk_term_templates."""
    from quchip.chip.chip import Chip
    from quchip.control.drive import ChargeDrive
    from quchip.control.envelopes import Square
    from quchip.control.equipment import ControlEquipment
    from quchip.devices.transmon.duffing import DuffingTransmon
    from quchip.engine.ir import DriveOp
    from quchip.engine.stage1_frames import resolve_frame
    from quchip.engine.stage2_assembly import compile_hamiltonian_template

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q_tmpl")
    d = ChargeDrive(target=q)
    chip = Chip([q], frame="rotating", control_equipment=ControlEquipment(lines=[d]))
    drive_op = DriveOp(
        target_label="q_tmpl",
        envelope=Square(amplitude=0.02, duration=50),
        freq=5.0,
        start_time=0.0,
        drive_label=d.label,
    )

    template = compile_hamiltonian_template(
        chip,
        [drive_op],
        resolved_frame=resolve_frame(chip, chip.frame),
    )

    assert hasattr(template, "drive_terms")
    assert not hasattr(template, "crosstalk_term_templates")
    assert not hasattr(template, "chip")


def test_ir_exports_scalar_modulation_not_time_dependence() -> None:
    """``ir`` exports ScalarModulation, not the removed TimeDependence."""
    assert hasattr(ir, "ScalarModulation")
    assert not hasattr(ir, "TimeDependence")


def test_carrier_default_sign_matches_rotating_frame_convention() -> None:
    """Carrier defaults to sign=-1, matching the rotating-frame convention."""
    carrier = ir.Carrier(freq=5.0)
    assert carrier.sign == -1


# ── Carrier-band decomposition ───────────────────────────────────────
#
# Correctness-critical: backends rely on decompose_carrier_bands to keep
# fast carriers analytic. A carrier-algebra error would silently corrupt
# every QuTiP simulation, so these assert the reconstruction is exact and
# that each band envelope is genuinely carrier-free.


def _reconstruct_bands(signal: ir.SignalProgram, t: np.ndarray) -> np.ndarray:
    """Sum the decomposed bands ``Σ_k env_k(t)·exp(i·freq_k·t)`` over *t*."""
    total = np.zeros_like(np.asarray(t, dtype=complex))
    for band in ir.decompose_carrier_bands(signal):
        env = np.asarray(ir.evaluate_signal_program(band.envelope, t, xp=np), dtype=complex)
        total = total + np.broadcast_to(env.ravel(), (len(t),)) * np.exp(1j * band.freq * t)
    return total


def _contains_carrier(signal: ir.SignalProgram) -> bool:
    if isinstance(signal, ir.Carrier):
        return True
    children = getattr(signal, "children", None)
    if children is not None:
        return any(_contains_carrier(c) for c in children)
    child = getattr(signal, "child", None)
    return _contains_carrier(child) if child is not None else False


class TestCarrierBandDecomposition:
    """``decompose_carrier_bands`` must be exact and yield carrier-free envelopes."""

    def _signals(self) -> dict[str, ir.SignalProgram]:
        from quchip.control.envelopes import Gaussian, Square

        wd = 2 * np.pi * 5.0
        env = ir.EnvelopeRef(Square(duration=30.0, amplitude=0.02))
        # The real lab-frame charge-drive coefficient shape emitted by stage 2:
        # Multiply(( RealPart(Multiply((line, Carrier(-w_d)))), phase=Carrier(0) )).
        line = ir.Scale(ir.Shift(ir.Window(env, 0.0, 30.0), 0.0), factor=np.exp(1j * 0.3))
        return {
            "carrier": ir.Carrier(freq=wd, sign=-1),
            "real_field": ir.RealPart(ir.Multiply((line, ir.Carrier(freq=wd, sign=-1)))),
            "conjugate": ir.Conjugate(ir.Multiply((line, ir.Carrier(freq=wd, sign=-1)))),
            "lab_full": ir.Multiply(
                (ir.RealPart(ir.Multiply((line, ir.Carrier(freq=wd, sign=-1)))), ir.Carrier(freq=0.0, sign=-1))
            ),
            "polar": ir.PolarScale(ir.Multiply((line, ir.Carrier(freq=wd, sign=-1))), 2.0, 0.7),
            "shifted_carrier": ir.Shift(
                ir.Multiply((ir.EnvelopeRef(Gaussian(duration=50.0, amplitude=0.005, sigmas=4)),
                             ir.Carrier(freq=1.3, sign=-1))),
                4.0,
            ),
            "nested": ir.RealPart(
                ir.Multiply((ir.Add((line, ir.Conjugate(line))),
                             ir.Carrier(freq=wd, sign=-1), ir.Carrier(freq=0.2, sign=1)))
            ),
        }

    def test_reconstruction_is_exact(self) -> None:
        """Summing decomposed bands reproduces the original signal exactly."""
        t = np.linspace(0.0, 30.0, 257)
        for name, signal in self._signals().items():
            reference = np.asarray(ir.evaluate_signal_program(signal, t, xp=np), dtype=complex)
            np.testing.assert_allclose(
                _reconstruct_bands(signal, t), reference, rtol=0, atol=1e-12,
                err_msg=f"band reconstruction differs for {name!r}",
            )

    def test_band_envelopes_are_carrier_free(self) -> None:
        """Every decomposed band envelope holds no residual carrier."""
        for name, signal in self._signals().items():
            for band in ir.decompose_carrier_bands(signal):
                assert not _contains_carrier(band.envelope), f"band envelope still holds a carrier for {name!r}"


class TestCanonicalOperator:
    def test_valid_construction(self):
        """CanonicalOperator.from_dense stores shape, dims, basis, and layout."""
        data = np.eye(6, dtype=complex)
        op = CanonicalOperator.from_dense(
            data,
            dims=(2, 3),
            basis="fock",
            subsystem_labels=("q0", "r0"),
        )
        assert op.shape == (6, 6)
        assert op.dims == (2, 3)
        assert op.basis == "fock"
        assert op.layout == "dense"

    def test_rejects_non_square(self):
        """A non-square operator raises ValueError."""
        with pytest.raises(ValueError, match="square"):
            CanonicalOperator(
                layout="dense",
                values=np.ones((2, 3), dtype=complex),
                shape=(2, 3),
                dims=(2,),
                basis="fock",
                subsystem_labels=("q",),
            )

    def test_rejects_dims_mismatch(self):
        """dims whose product mismatches the operator shape raises ValueError."""
        with pytest.raises(ValueError, match="Product of dims"):
            CanonicalOperator(
                layout="dense",
                values=np.eye(4, dtype=complex),
                shape=(4, 4),
                dims=(2, 3),
                basis="fock",
                subsystem_labels=("a", "b"),
            )

    def test_rejects_labels_mismatch(self):
        """subsystem_labels length mismatching dims raises ValueError."""
        with pytest.raises(ValueError, match="subsystem_labels length"):
            CanonicalOperator(
                layout="dense",
                values=np.eye(4, dtype=complex),
                shape=(4, 4),
                dims=(2, 2),
                basis="fock",
                subsystem_labels=("a",),
            )


class TestTermTypes:
    def test_static_term_stores_canonical(self):
        """StaticTerm stores the CanonicalOperator and defaults coefficient to 1.0."""
        op = CanonicalOperator.from_dense(
            np.eye(3, dtype=complex),
            dims=(3,),
            basis="fock",
            subsystem_labels=("q",),
        )
        term = StaticTerm(operator=op, origin="device")
        assert isinstance(term.operator, CanonicalOperator)
        assert term.coefficient == 1.0

    def test_dynamic_term_scalar_modulation_carrier(self):
        """DynamicTerm carries a ScalarModulation wrapping a Carrier signal."""
        op = CanonicalOperator.from_dense(
            np.eye(3, dtype=complex),
            dims=(3,),
            basis="fock",
            subsystem_labels=("q",),
        )
        term = DynamicTerm(
            operator=op,
            time_dependence=ScalarModulation(signal=Carrier(freq=5.0, sign=-1)),
            origin="coupling",
        )
        assert isinstance(term.time_dependence, ScalarModulation)
        assert isinstance(term.time_dependence.signal, Carrier)
        assert term.time_dependence.signal.freq == pytest.approx(5.0)

    def test_hamiltonian_description_assembly(self):
        """HamiltonianDescription assembles static/dynamic terms with dims and metadata."""
        op = CanonicalOperator.from_dense(
            np.eye(3, dtype=complex),
            dims=(3,),
            basis="fock",
            subsystem_labels=("q",),
        )
        static = StaticTerm(operator=op, origin="device")
        dynamic = DynamicTerm(
            operator=op,
            time_dependence=ScalarModulation(signal=Carrier(freq=1.0, sign=-1)),
            origin="coupling",
        )
        desc = HamiltonianDescription(
            static_terms=(static,),
            dynamic_terms=(dynamic,),
            dims=(3,),
            metadata={"frame_mode": "rotating"},
        )
        assert len(desc.static_terms) == 1
        assert len(desc.dynamic_terms) == 1
        assert desc.dims == (3,)
        assert desc.metadata["frame_mode"] == "rotating"


class TestPolarScale:
    def test_polar_scale_evaluates_amplitude_times_exp_theta(self):
        """PolarScale evaluates to amplitude * exp(i*theta) times the child signal."""
        from quchip.engine.ir import PolarScale, Constant, evaluate_signal_program

        signal = PolarScale(child=Constant(1.0 + 0j), amplitude=0.5, theta=0.0)
        result = evaluate_signal_program(signal, np.array([0.0]))
        np.testing.assert_allclose(result, [0.5 + 0j])

    def test_polar_scale_with_nonzero_theta(self):
        """Nonzero theta rotates PolarScale's output onto the imaginary axis."""
        from quchip.engine.ir import PolarScale, Constant, evaluate_signal_program

        signal = PolarScale(child=Constant(1.0 + 0j), amplitude=0.1, theta=np.pi / 2)
        result = evaluate_signal_program(signal, np.array([0.0]))
        np.testing.assert_allclose(result, [0.1j], atol=1e-15)

    def test_crosstalk_apply_uses_polar_scale_not_numpy(self):
        """Crosstalk.apply builds a PolarScale node instead of an eagerly-computed numpy factor."""
        from quchip.engine.ir import PolarScale, Constant
        from quchip.control.signal import Crosstalk

        edge = Crosstalk(source="charge_0", victim="charge_1", beta=0.1, theta=0.3)
        signals = {("charge_0", 0): Constant(1.0 + 0j)}
        result = edge.apply(signals)
        victim_signal = result[("charge_1", 0)]
        assert isinstance(victim_signal, PolarScale)


class TestSolveProblem:
    def test_rejects_backend_in_options(self):
        """SolveProblem.options containing 'backend' raises ValueError."""
        with pytest.raises(ValueError, match="must not contain 'backend'"):
            SolveProblem(
                chip=None,
                hamiltonian=None,
                initial_state=None,
                tlist=None,
                options={"backend": "something"},
            )

    def test_accepts_valid_options(self):
        """SolveProblem stores arbitrary non-'backend' options."""
        problem = SolveProblem(
            chip=None,
            hamiltonian=None,
            initial_state=None,
            tlist=np.linspace(0, 100, 200),
            options={"nsteps": 5000},
        )
        assert problem.options == {"nsteps": 5000}


class TestDroppedTerms:
    """Surface RWA-dropped terms on HamiltonianDescription (issue #59)."""

    def test_capacitive_rwa_reports_counter_rotating_drops(self):
        """RWA on a Capacitive coupling records the dropped counter-rotating terms."""
        from quchip.chip.chip import Chip
        from quchip.chip.couplings import Capacitive
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.engine.ir import DroppedTerm
        from quchip.engine.stage1_frames import resolve_frame
        from quchip.engine.stage2_assembly import build_hamiltonian_description

        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.25, levels=3, label="q1")
        cap = Capacitive(q0, q1, g=0.01, rwa=True, label="cap_q0_q1")
        chip = Chip([q0, q1], couplings=[cap], frame="rotating")

        description = build_hamiltonian_description(
            chip, [], resolved_frame=resolve_frame(chip, chip.frame)
        )

        assert all(isinstance(dt, DroppedTerm) for dt in description.dropped_terms)
        operators = {dt.operator for dt in description.dropped_terms}
        assert operators == {
            "coupling band (Δa=+1, Δb=+1) on q0·q1",
            "coupling band (Δa=-1, Δb=-1) on q0·q1",
        }
        band_weights = {dt.band_weights for dt in description.dropped_terms}
        assert band_weights == {(-1, -1), (1, 1)}
        for dt in description.dropped_terms:
            assert dt.source == "cap_q0_q1"
            assert "counter-rotating" in dt.reason.lower()
            # amplitude = the dropped band's largest matrix element — for the
            # a†b† / ab bands of g·(a+a†)(b+b†) on 3-level ladders that is
            # g·√2·√2 = 2g; frequency = the band's rotating-frame oscillation
            # |Δa·f_a + Δb·f_b| = f_a + f_b (dressed refs here, so approximate
            # to the hybridization shift).
            assert dt.amplitude == pytest.approx(0.02, rel=1e-12)
            assert dt.frequency == pytest.approx(10.2, rel=1e-3)

        summary = description.dropped_terms_summary()
        assert "cap_q0_q1" in summary
        assert "on q0·q1" in summary
        assert "amp 0.02 GHz" in summary
        assert "freq 10.2" in summary

    def test_capacitive_without_rwa_reports_nothing(self):
        """Capacitive coupling without RWA drops no terms."""
        from quchip.chip.chip import Chip
        from quchip.chip.couplings import Capacitive
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.engine.stage1_frames import resolve_frame
        from quchip.engine.stage2_assembly import build_hamiltonian_description

        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="qa")
        q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.25, levels=3, label="qb")
        chip = Chip(
            [q0, q1],
            couplings=[Capacitive(q0, q1, g=0.01, rwa=False)],
            frame="rotating",
            rwa=False,
        )
        description = build_hamiltonian_description(
            chip, [], resolved_frame=resolve_frame(chip, chip.frame)
        )
        assert description.dropped_terms == ()
        assert description.dropped_terms_summary() == "No dropped terms."

    def test_drive_rwa_reports_fast_partners(self):
        """Each nonzero-weight single-tone band drops one counter-rotating partner at f_d + |w|·f_ref."""
        import numpy as np

        from quchip.chip.chip import Chip
        from quchip.control.drive import ChargeDrive
        from quchip.control.envelopes import Gaussian
        from quchip.control.equipment import ControlEquipment
        from quchip.control.sequence import QuantumSequence
        from quchip.devices.transmon.duffing import DuffingTransmon

        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        drive = ChargeDrive(target=q0, label="d0")
        chip = Chip(
            [q0],
            control_equipment=ControlEquipment(lines=[drive]),
            frame={q0: 5.0},
            rwa=True,
        )
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=3), freq=5.0)
        problem = sequence.build_problem(
            tlist=np.linspace(0.0, 20.0, 21), initial_state=chip.bare_state(q0=0)
        )

        records = problem.hamiltonian.dropped_terms
        assert {dt.band_weights for dt in records} == {(-1,), (1,)}
        for dt in records:
            assert dt.source == "d0"
            assert dt.amplitude is None  # drive prefactors are envelopes, not scalars
            assert dt.frequency == pytest.approx(10.0)  # f_d + |w|·f_ref = 5 + 5

    def test_drive_without_rwa_reports_nothing(self):
        """rwa=False keeps both drive components — nothing to audit."""
        import numpy as np

        from quchip.chip.chip import Chip
        from quchip.control.drive import ChargeDrive
        from quchip.control.envelopes import Gaussian
        from quchip.control.equipment import ControlEquipment
        from quchip.control.sequence import QuantumSequence
        from quchip.devices.transmon.duffing import DuffingTransmon

        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        drive = ChargeDrive(target=q0, label="d0")
        chip = Chip(
            [q0],
            control_equipment=ControlEquipment(lines=[drive]),
            frame={q0: 5.0},
            rwa=False,
        )
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=3), freq=5.0)
        problem = sequence.build_problem(
            tlist=np.linspace(0.0, 20.0, 21), initial_state=chip.bare_state(q0=0)
        )
        assert problem.hamiltonian.dropped_terms == ()

    def test_summary_prints_traced_values_as_placeholder(self):
        """Traced amplitudes format as 'traced' — the summary never concretizes them."""
        import jax
        import jax.numpy as jnp

        from quchip.engine.ir import DroppedTerm

        seen: dict[str, str] = {}

        @jax.jit
        def build(g):
            record = DroppedTerm(
                source="c", operator="a·b", reason="counter-rotating under RWA",
                band_weights=(-1, -1), amplitude=g, frequency=10.2,
            )
            description = HamiltonianDescription(
                static_terms=(), dynamic_terms=(), dropped_terms=(record,)
            )
            seen["summary"] = description.dropped_terms_summary()
            return record.amplitude * 2.0  # raw value stays live for autodiff

        out = build(jnp.asarray(0.01))
        assert "amp traced" in seen["summary"]
        assert "freq 10.2 GHz" in seen["summary"]
        assert float(out) == pytest.approx(0.02)
