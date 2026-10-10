"""Contract tests for the Hamiltonian description IR types."""

from __future__ import annotations

from quchip.approximations import RWA, Exact

import numpy as np
import pytest

import quchip.engine.ir as ir
from quchip.engine.ir import (
    CanonicalOperator,
    Carrier,
    DynamicTerm,
    EngineResult,
    ResolvedSLH,
    ScalarModulation,
    SolveProblem,
    StaticTerm,
)


@pytest.mark.parametrize("backend", ["qutip", "dynamiqs"])
@pytest.mark.parametrize("change", ["reorder", "add", "remove"])
def test_template_signal_routes_preserve_operator_identity(backend, change):
    from quchip import Chip, ChargeDrive, ControlEquipment, DuffingTransmon, QuantumSequence, Square
    from quchip.control.signal import SignalTransform
    from quchip.engine.assembly import compile_hamiltonian_template, instantiate_engine_result

    class Routing(SignalTransform):
        change = None

        def apply(self, signals):
            delivered = dict(signals)
            if self.change == "reorder":
                return dict(reversed(list(delivered.items())))
            if self.change == "add":
                delivered[("db", 0)] = signals[("da", 0)]
            if self.change == "remove":
                del delivered[("da", 0)]
            return delivered

    first = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="a")
    second = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=2, label="b")
    drives = [ChargeDrive(first, label="da"), ChargeDrive(second, label="db")]
    equipment = ControlEquipment(lines=drives, signal_chain=[Routing()])
    chip = Chip([first, second], frame="rotating", backend=backend, control_equipment=equipment)
    sequence = QuantumSequence(chip)
    sequence.schedule(drives[0], envelope=Square(duration=1.0, amplitude=0.02), freq=5.0)
    sequence.schedule(drives[1], envelope=Square(duration=1.0, amplitude=0.06), freq=5.0)
    operations = sequence._materialize_drive_ops()
    resolved = chip.resolve()

    def compile_template():
        return compile_hamiltonian_template(
            chip, operations, resolved_frame=resolved.resolved_frame,
            _base_result=resolved,
        )

    template = compile_template()
    before = instantiate_engine_result(template, operations, chip).hamiltonian().matrix(backend=chip.backend, t=0.2)
    chip.control_equipment.signal_chain[0].change = change
    if change == "reorder":
        after = instantiate_engine_result(template, operations, chip).hamiltonian().matrix(backend=chip.backend, t=0.2)
        np.testing.assert_allclose(after, before, atol=1e-12)
    else:
        with pytest.raises(ValueError, match="Signal routes changed"):
            instantiate_engine_result(template, operations, chip)
        after = instantiate_engine_result(compile_template(), operations, chip).hamiltonian().matrix(
            backend=chip.backend, t=0.2,
        )
        assert np.linalg.norm(after - before) > 0.005
    assert np.linalg.norm(before) > 0.01


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
        # The real lab-frame charge-drive coefficient shape emitted by assembly:
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
                ir.Multiply(
                    (ir.EnvelopeRef(Gaussian(duration=50.0, amplitude=0.005, sigmas=4)), ir.Carrier(freq=1.3, sign=-1))
                ),
                4.0,
            ),
            "nested": ir.RealPart(
                ir.Multiply(
                    (ir.Add((line, ir.Conjugate(line))), ir.Carrier(freq=wd, sign=-1), ir.Carrier(freq=0.2, sign=1))
                )
            ),
        }

    def test_reconstruction_is_exact(self) -> None:
        """Summing decomposed bands reproduces the original signal exactly."""
        t = np.linspace(0.0, 30.0, 257)
        for name, signal in self._signals().items():
            reference = np.asarray(ir.evaluate_signal_program(signal, t, xp=np), dtype=complex)
            np.testing.assert_allclose(
                _reconstruct_bands(signal, t),
                reference,
                rtol=0,
                atol=1e-12,
                err_msg=f"band reconstruction differs for {name!r}",
            )

    def test_band_envelopes_are_carrier_free(self) -> None:
        """Every decomposed band envelope holds no residual carrier."""
        for name, signal in self._signals().items():
            for band in ir.decompose_carrier_bands(signal):
                assert not _contains_carrier(band.envelope), f"band envelope still holds a carrier for {name!r}"


class TestCanonicalOperator:
    @pytest.mark.parametrize("layout", ["dense", "csr", "dia"])
    def test_diagonal_reads_each_layout_without_dense_materialization(self, layout, monkeypatch):
        """Canonical diagonal access is layout-native for dense, CSR, and DIA payloads."""
        matrix = np.array(
            [[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]],
            dtype=complex,
        )
        if layout == "dense":
            op = CanonicalOperator.from_dense(matrix, dims=(3,), basis="fock", subsystem_labels=("q",))
        elif layout == "csr":
            op = CanonicalOperator.from_csr(
                values=np.array([1, 2, 3, 4, 5, 6, 7], dtype=complex),
                indices=np.array([0, 1, 0, 1, 2, 1, 2]),
                indptr=np.array([0, 2, 5, 7]),
                shape=(3, 3),
                dims=(3,),
                basis="fock",
                subsystem_labels=("q",),
            )
        else:
            op = CanonicalOperator.from_dia(
                values=np.array(
                    [[0, 2, 5], [1, 4, 7], [3, 6, 0]],
                    dtype=complex,
                ),
                offsets=np.array([1, 0, -1]),
                shape=(3, 3),
                dims=(3,),
                basis="fock",
                subsystem_labels=("q",),
            )

        monkeypatch.setattr(CanonicalOperator, "to_dense", lambda self: pytest.fail("densified"))

        np.testing.assert_allclose(op.diagonal(), np.diag(matrix))
    @pytest.mark.parametrize(
        "shape,dims,labels,message",
        [
            ((2, 3), (2,), ("q",), "square"),
            ((4, 4), (2, 3), ("a", "b"), "Product of dims"),
            ((4, 4), (2, 2), ("a",), "subsystem_labels length"),
        ],
    )
    def test_rejects_inconsistent_shape_metadata(self, shape, dims, labels, message):
        """Shape, tensor dimensions, and subsystem labels must agree."""
        with pytest.raises(ValueError, match=message):
            CanonicalOperator(
                layout="dense", values=np.ones(shape, dtype=complex), shape=shape,
                dims=dims, basis="fock", subsystem_labels=labels,
            )


class TestTermTypes:
    def test_engine_result_hamiltonian_matrix_is_public_ghz_time_slice(self):
        """Hamiltonian inspection converts canonical angular terms back to public GHz."""
        op = CanonicalOperator.from_dense(
            np.eye(2, dtype=complex),
            dims=(2,),
            basis="fock",
            subsystem_labels=("q",),
        )
        result = EngineResult(
            slh=ResolvedSLH.from_terms(
                static_terms=(StaticTerm(operator=op, coefficient=2.0),),
                dynamic_terms=(
                    DynamicTerm(
                        operator=op,
                        time_dependence=ScalarModulation(signal=Carrier(freq=1.0)),
                    ),
                ),
                collapse_terms=(),
            ),
            dims=(2,),
        )
        with pytest.raises(ValueError, match="t is required"):
            result.hamiltonian().matrix()
        np.testing.assert_allclose(result.hamiltonian().matrix(t=0.0), 3.0 / (2.0 * np.pi) * np.eye(2))


class TestPolarScale:
    def test_polar_scale_with_nonzero_theta(self):
        """Nonzero theta rotates PolarScale's output onto the imaginary axis."""
        from quchip.engine.ir import PolarScale, Constant, evaluate_signal_program

        signal = PolarScale(child=Constant(1.0 + 0j), amplitude=0.1, theta=np.pi / 2)
        result = evaluate_signal_program(signal, np.array([0.0]))
        np.testing.assert_allclose(result, [0.1j], atol=1e-15)

class TestSolveProblem:
    def test_rejects_backend_in_options(self):
        """SolveProblem.options containing 'backend' raises ValueError."""
        with pytest.raises(ValueError, match="must not contain 'backend'"):
            SolveProblem(
                chip=None,
                engine_result=None,
                initial_state=None,
                tlist=None,
                options={"backend": "something"},
            )

class TestDroppedTerms:
    """Surface RWA-dropped terms on EngineResult (issue #59)."""

    @pytest.mark.parametrize("approximation", [RWA(), Exact()])
    def test_capacitive_reports_drops_for_selected_approximation(self, approximation):
        """RWA on a Capacitive coupling records the dropped counter-rotating terms."""
        from quchip.chip.chip import Chip
        from quchip.chip.couplings import Capacitive
        from quchip.devices.transmon.duffing import DuffingTransmon
        from quchip.engine.ir import DroppedTerm
        from quchip.engine.frames import resolve_frame
        from quchip.engine.assembly import build_engine_result

        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.25, levels=3, label="q1")
        cap = Capacitive(q0, q1, g=0.01, label="cap_q0_q1")
        chip = Chip([q0, q1], couplings=[cap], frame="rotating", approximation=approximation)

        description = build_engine_result(chip, [], resolved_frame=resolve_frame(chip, chip.frame))

        if isinstance(approximation, Exact):
            assert description.dropped_terms == ()
            assert description.dropped_terms_summary() == "No dropped terms."
            return

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
            # a†b† / ab bands of g·i(a-a†)·i(b-b†) on 3-level ladders that is
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

    @pytest.mark.parametrize("approximation", [RWA(), Exact()])
    def test_drive_reports_drops_for_selected_approximation(self, approximation):
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
            approximation=approximation,
        )
        sequence = QuantumSequence(chip)
        sequence.schedule(drive, envelope=Gaussian(duration=20.0, amplitude=0.02, sigmas=3), freq=5.0)
        problem = sequence.build_problem(tlist=np.linspace(0.0, 20.0, 21), initial_state=chip.bare_state(q0=0))

        records = problem.engine_result.dropped_terms
        if isinstance(approximation, Exact):
            assert records == ()
            return
        assert {dt.band_weights for dt in records} == {(-1,), (1,)}
        for dt in records:
            assert dt.source == "d0"
            assert dt.amplitude is None  # drive prefactors are envelopes, not scalars
            assert dt.frequency == pytest.approx(10.0)  # f_d + |w|·f_ref = 5 + 5

    def test_summary_prints_traced_values_as_placeholder(self):
        """Traced amplitudes format as 'traced' — the summary never concretizes them."""
        import jax
        import jax.numpy as jnp

        from quchip.engine.ir import DroppedTerm

        seen: dict[str, str] = {}

        @jax.jit
        def build(g):
            record = DroppedTerm(
                source="c",
                operator="a·b",
                reason="counter-rotating under RWA",
                band_weights=(-1, -1),
                amplitude=g,
                frequency=10.2,
            )
            description = EngineResult(
                slh=ResolvedSLH.from_terms(
                    static_terms=(),
                    dynamic_terms=(),
                    collapse_terms=(),
                ),
                dropped_terms=(record,),
            )
            seen["summary"] = description.dropped_terms_summary()
            return record.amplitude * 2.0  # raw value stays live for autodiff

        out = build(jnp.asarray(0.01))
        assert "amp traced" in seen["summary"]
        assert "freq 10.2 GHz" in seen["summary"]
        assert float(out) == pytest.approx(0.02)
