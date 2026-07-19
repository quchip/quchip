"""Unit tests for single-mode band decomposition and demod helpers."""

from __future__ import annotations

import numpy as np

from quchip.chip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.control.drive import ChargeDrive
from quchip.control.envelopes import Gaussian
from quchip.control.equipment import ControlEquipment
from quchip.control.sequence import QuantumSequence
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.results import ObservableTrace
from quchip.engine.bands import decompose_bands
from quchip.engine.stage3_observables import BandMeta, decompose_eops, recombine_expect


def _annihilation(d: int) -> np.ndarray:
    """Analytical lowering operator for a d-level oscillator."""
    a = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        a[n - 1, n] = np.sqrt(n)
    return a


def _build_coupled_sequence(frame: str) -> tuple[QuantumSequence, Chip, np.ndarray]:
    """Build a transmon+resonator sequence in a numerically stable regime."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=2, label="q")
    r = Resonator(freq=6.2, levels=8, label="r")
    drive_q = ChargeDrive(target=q, label=f"Dq-{frame}")
    chip = Chip(
        [q, r],
        couplings=[Capacitive(q, r, g=1e-6, rwa=False)],
        control_equipment=ControlEquipment(lines=[drive_q]),
        frame=frame,
    )

    seq = QuantumSequence(chip)
    seq.schedule(
        drive_q,
        envelope=Gaussian(duration=10.0, amplitude=1e-4, sigmas=3),
        freq=q.freq,
    )

    tlist = np.linspace(0.0, 10.0, 501)
    return seq, chip, tlist


_STRICT_SOLVER_OPTS = {
    "nsteps": 500000,
    "atol": 1e-12,
    "rtol": 1e-12,
    "method": "bdf",
}


def test_decompose_bands_lowering() -> None:
    """3-level a has only weight +1 band."""
    a = _annihilation(3)
    bands = decompose_bands(a, 3)
    assert set(bands.keys()) == {1}
    np.testing.assert_allclose(bands[1], a, atol=1e-14)


def test_decompose_bands_raising() -> None:
    """3-level a† has only weight -1 band."""
    adag = _annihilation(3).conj().T
    bands = decompose_bands(adag, 3)
    assert set(bands.keys()) == {-1}
    np.testing.assert_allclose(bands[-1], adag, atol=1e-14)


def test_decompose_bands_number() -> None:
    """3-level n̂ has only weight 0 band."""
    n_hat = np.diag([0, 1, 2]).astype(complex)
    bands = decompose_bands(n_hat, 3)
    assert set(bands.keys()) == {0}
    np.testing.assert_allclose(bands[0], n_hat, atol=1e-14)


def test_decompose_bands_sigma_x_like() -> None:
    """a + a† decomposes into weights {-1, +1}."""
    a = _annihilation(3)
    x = a + a.conj().T
    bands = decompose_bands(x, 3)

    assert set(bands.keys()) == {-1, 1}
    np.testing.assert_allclose(sum(bands.values()), x, atol=1e-14)


def test_decompose_bands_identity() -> None:
    """Identity decomposes to a single weight-0 band."""
    eye = np.eye(4, dtype=complex)
    bands = decompose_bands(eye, 4)

    assert set(bands.keys()) == {0}
    np.testing.assert_allclose(bands[0], eye, atol=1e-14)


def test_decompose_bands_completeness() -> None:
    """Sum of all bands reconstructs the original matrix."""
    rng = np.random.default_rng(123)
    mat = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))

    bands = decompose_bands(mat, 5)
    reconstructed = sum(bands.values())
    np.testing.assert_allclose(reconstructed, mat, atol=1e-14)


def test_decompose_bands_supports_jitted_jax_inputs() -> None:
    """Single-mode decomposition should remain usable under ``jax.jit``."""
    import jax
    import jax.numpy as jnp

    matrix = jnp.array(
        [
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [2.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j],
            [0.0 + 0.0j, 4.0 + 0.0j, 0.0 + 0.0j],
        ]
    )

    bands = jax.jit(lambda op: decompose_bands(op, 3))(matrix)
    active_weights = {weight for weight, band in bands.items() if np.linalg.norm(np.asarray(band)) > 1e-15}

    assert active_weights == {-1, 1}
    np.testing.assert_allclose(np.asarray(bands[-1]), np.asarray(np.tril(matrix, k=-1)), atol=1e-14)
    np.testing.assert_allclose(np.asarray(bands[1]), np.asarray(np.triu(matrix, k=1)), atol=1e-14)


def test_decompose_recombine_roundtrip() -> None:
    """With zero frame frequencies, recombined = raw sum per key."""
    tlist = np.array([0.0, 0.2, 0.5])

    s1 = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])
    s2 = np.array([0.5j, 1.0j, 1.5j])
    s3 = np.array([2.0 + 1.0j, 2.0 + 1.0j, 2.0 + 1.0j])
    s4 = np.array([-1.0j, -2.0j, -3.0j])

    flat_expect = [s1, s2, s3, s4]
    meta = [
        BandMeta(key="q", weight=1, device_labels="q"),
        BandMeta(key="q", weight=-1, device_labels="q"),
        BandMeta(key=("q", "r"), weight=(1, -1), device_labels=("q", "r")),
        BandMeta(key=("q", "r"), weight=(-1, 1), device_labels=("q", "r")),
    ]

    expect, raw = recombine_expect(
        flat_expect=flat_expect,
        meta_list=meta,
        tlist=tlist,
        frame_freqs={"q": 0.0, "r": 0.0},
    )

    np.testing.assert_allclose(expect["q"], s1 + s2, atol=1e-14)
    np.testing.assert_allclose(raw["q"], s1 + s2, atol=1e-14)
    np.testing.assert_allclose(expect[("q", "r")], s3 + s4, atol=1e-14)
    np.testing.assert_allclose(raw[("q", "r")], s3 + s4, atol=1e-14)


def test_recombine_remodulate() -> None:
    """Remodulate direction applies exp(-i·ω·w·t) per band (rotating→lab)."""
    tlist = np.array([0.0, 0.125, 0.25])
    freqs = {"q": 5.0, "r": 7.0}

    s_q_p = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])
    s_q_m = np.array([0.0 + 2.0j, 0.0 + 2.0j, 0.0 + 2.0j])
    s_qr_1 = np.array([0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j])
    s_qr_2 = np.array([0.0 + 0.25j, 0.0 + 0.25j, 0.0 + 0.25j])

    flat_expect = [s_q_p, s_q_m, s_qr_1, s_qr_2]
    meta = [
        BandMeta(key="q", weight=1, device_labels="q"),
        BandMeta(key="q", weight=-1, device_labels="q"),
        BandMeta(key=("q", "r"), weight=(1, -1), device_labels=("q", "r")),
        BandMeta(key=("q", "r"), weight=(-1, 1), device_labels=("q", "r")),
    ]

    band_sum, phase_corrected = recombine_expect(
        flat_expect,
        meta,
        tlist,
        freqs,
        direction="remodulate",
    )

    # Remodulate: exp(-i·ω·w·t)
    phase_q_p = np.exp(-1j * 2 * np.pi * freqs["q"] * (+1) * tlist)
    phase_q_m = np.exp(-1j * 2 * np.pi * freqs["q"] * (-1) * tlist)
    phase_qr_1 = np.exp(-1j * 2 * np.pi * (freqs["q"] * (+1) + freqs["r"] * (-1)) * tlist)
    phase_qr_2 = np.exp(-1j * 2 * np.pi * (freqs["q"] * (-1) + freqs["r"] * (+1)) * tlist)

    expected_q = phase_q_p * s_q_p + phase_q_m * s_q_m
    expected_qr = phase_qr_1 * s_qr_1 + phase_qr_2 * s_qr_2

    # band_sum = direct accumulation
    np.testing.assert_allclose(band_sum["q"], s_q_p + s_q_m, atol=1e-14)
    np.testing.assert_allclose(band_sum[("q", "r")], s_qr_1 + s_qr_2, atol=1e-14)

    # phase_corrected = remodulated (exp(-i·ω·w·t) applied)
    np.testing.assert_allclose(phase_corrected["q"], expected_q, atol=1e-14)
    np.testing.assert_allclose(phase_corrected[("q", "r")], expected_qr, atol=1e-14)


def test_recombine_demodulate() -> None:
    """Demodulate direction applies exp(+i·ω·w·t) per band (lab→slowly varying)."""
    tlist = np.array([0.0, 0.125, 0.25])
    freqs = {"q": 5.0, "r": 7.0}

    s_q_p = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])
    s_q_m = np.array([0.0 + 2.0j, 0.0 + 2.0j, 0.0 + 2.0j])

    flat_expect = [s_q_p, s_q_m]
    meta = [
        BandMeta(key="q", weight=1, device_labels="q"),
        BandMeta(key="q", weight=-1, device_labels="q"),
    ]

    band_sum, phase_corrected = recombine_expect(
        flat_expect,
        meta,
        tlist,
        freqs,
        direction="demodulate",
    )

    # Demodulate: exp(+i·ω·w·t) — conjugate of remodulate
    phase_q_p = np.exp(+1j * 2 * np.pi * freqs["q"] * (+1) * tlist)
    phase_q_m = np.exp(+1j * 2 * np.pi * freqs["q"] * (-1) * tlist)

    expected_q = phase_q_p * s_q_p + phase_q_m * s_q_m

    np.testing.assert_allclose(band_sum["q"], s_q_p + s_q_m, atol=1e-14)
    np.testing.assert_allclose(phase_corrected["q"], expected_q, atol=1e-14)


def test_decompose_eops_handles_string_and_tuple_keys(backend) -> None:
    """decompose_eops expands both single-device and tuple-key entries."""
    q = Resonator(freq=5.0, levels=3, label="q")
    r = Resonator(freq=6.0, levels=3, label="r")
    chip = Chip([q, r])

    a = backend.destroy(3)
    x = a + backend.dag(a)  # weights {-1, +1}

    flat_ops, meta = decompose_eops(
        {
            "q": x,
            ("q", "r"): (x, x),
        },
        chip,
        backend,
    )

    assert len(flat_ops) == len(meta) == 6

    single_weights = {m.weight for m in meta if m.key == "q"}
    assert single_weights == {-1, 1}

    cross_weights = {m.weight for m in meta if m.key == ("q", "r")}
    assert cross_weights == {
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    }


def test_single_device_demod_matches_lab(backend) -> None:
    """Rotating-frame dict e_ops use identity demodulation (expect == expect_raw)."""
    seq_lab, chip_lab, tlist = _build_coupled_sequence(frame="lab")
    seq_rot, chip_rot, _ = _build_coupled_sequence(frame="rotating")

    a_r_lab = chip_lab.device_map["r"].lowering_operator()
    a_r_rot = chip_rot.device_map["r"].lowering_operator()

    init_lab = chip_lab.bare_state(
        q=0,
        r=backend.coherent(chip_lab.device_map["r"].levels, 0.3),
    )
    init_rot = chip_rot.bare_state(
        q=0,
        r=backend.coherent(chip_rot.device_map["r"].levels, 0.3),
    )

    result_lab = seq_lab.simulate(
        tlist=tlist,
        e_ops={"r": a_r_lab},
        initial_state=init_lab,
        options=_STRICT_SOLVER_OPTS,
    )
    result_rot = seq_rot.simulate(
        tlist=tlist,
        e_ops={"r": a_r_rot},
        initial_state=init_rot,
        options=_STRICT_SOLVER_OPTS,
    )

    assert np.max(np.abs(np.asarray(result_lab._expect_data["r"].values))) > 1e-2
    # Rotating frame has demod_freq=0, so demodulation is identity.
    np.testing.assert_allclose(
        np.asarray(result_rot._expect_data["r"].values),
        np.asarray(result_rot._expect_data["r"].raw),
        atol=1e-8,
    )


def test_cross_device_demod(backend) -> None:
    """Tuple-key rotating-frame dict e_ops use identity demodulation."""
    seq_lab, chip_lab, tlist = _build_coupled_sequence(frame="lab")
    seq_rot, chip_rot, _ = _build_coupled_sequence(frame="rotating")

    a_q_lab = chip_lab.device_map["q"].lowering_operator()
    a_r_lab = chip_lab.device_map["r"].lowering_operator()

    a_q_rot = chip_rot.device_map["q"].lowering_operator()
    a_r_rot = chip_rot.device_map["r"].lowering_operator()

    init_lab = chip_lab.bare_state(
        q=backend.coherent(chip_lab.device_map["q"].levels, 0.2),
        r=backend.coherent(chip_lab.device_map["r"].levels, 0.2),
    )
    init_rot = chip_rot.bare_state(
        q=backend.coherent(chip_rot.device_map["q"].levels, 0.2),
        r=backend.coherent(chip_rot.device_map["r"].levels, 0.2),
    )

    result_lab = seq_lab.simulate(
        tlist=tlist,
        e_ops={("q", "r"): (a_q_lab, backend.dag(a_r_lab))},
        initial_state=init_lab,
        options=_STRICT_SOLVER_OPTS,
    )
    result_rot = seq_rot.simulate(
        tlist=tlist,
        e_ops={("q", "r"): (a_q_rot, backend.dag(a_r_rot))},
        initial_state=init_rot,
        options=_STRICT_SOLVER_OPTS,
    )

    key = ("q", "r")
    assert key in result_rot._expect_data
    assert isinstance(result_rot._expect_data[key], ObservableTrace)
    assert np.iscomplexobj(np.asarray(result_rot._expect_data[key].values))
    assert np.max(np.abs(np.asarray(result_lab._expect_data[key].values))) > 1e-2

    # Rotating frame has demod_freq=0, so demodulation is identity.
    np.testing.assert_allclose(
        np.asarray(result_rot._expect_data[key].values),
        np.asarray(result_rot._expect_data[key].raw),
        atol=1e-8,
    )


def test_lab_frame_dict_eops(backend) -> None:
    """Lab-frame dict e_ops should expose named traces with raw components."""
    seq_lab, chip_lab, tlist = _build_coupled_sequence(frame="lab")

    a_q = chip_lab.device_map["q"].lowering_operator()
    a_r = chip_lab.device_map["r"].lowering_operator()

    dict_e_ops = {
        "r": a_r,
        ("q", "r"): (a_q, backend.dag(a_r)),
    }

    init_state = chip_lab.bare_state(
        q=backend.coherent(chip_lab.device_map["q"].levels, 0.2),
        r=backend.coherent(chip_lab.device_map["r"].levels, 0.2),
    )

    problem = seq_lab.build_problem(
        tlist=tlist,
        e_ops=dict_e_ops,
        initial_state=init_state,
        options=_STRICT_SOLVER_OPTS,
    )
    backend_result = seq_lab._chip.backend.solve_problem(problem)
    flat_expect = (
        list(backend_result.expect.values()) if isinstance(backend_result.expect, dict) else backend_result.expect
    )
    band_sum, phase_corrected = recombine_expect(
        flat_expect=flat_expect,
        meta_list=problem.e_ops_meta or [],
        tlist=problem.tlist,
        frame_freqs=problem.resolved_frame.demod_freqs,
        direction="demodulate",
    )

    result_dict = seq_lab.simulate(
        tlist=tlist,
        e_ops=dict_e_ops,
        initial_state=init_state,
        options=_STRICT_SOLVER_OPTS,
    )

    assert isinstance(result_dict._expect_data, dict)
    assert set(result_dict._expect_data.keys()) == {"r", ("q", "r")}
    assert isinstance(result_dict._expect_data["r"], ObservableTrace)
    assert isinstance(result_dict._expect_data[("q", "r")], ObservableTrace)
    np.testing.assert_allclose(result_dict._expect_data["r"].raw, band_sum["r"], atol=1e-6)
    np.testing.assert_allclose(result_dict._expect_data["r"].values, phase_corrected["r"], atol=1e-6)
    np.testing.assert_allclose(
        result_dict._expect_data[("q", "r")].raw,
        band_sum[("q", "r")],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result_dict._expect_data[("q", "r")].values,
        phase_corrected[("q", "r")],
        atol=1e-6,
    )


def test_dict_eops_result_shape(backend) -> None:
    """Result expect/expect_raw shape semantics for dict e_ops in both frames."""
    seq_rot, chip_rot, tlist = _build_coupled_sequence(frame="rotating")
    a_r_rot = chip_rot.device_map["r"].lowering_operator()

    init_rot = chip_rot.bare_state(
        q=0,
        r=backend.coherent(chip_rot.device_map["r"].levels, 0.2),
    )

    result_rot = seq_rot.simulate(
        tlist=tlist,
        e_ops={"r": a_r_rot},
        initial_state=init_rot,
        options=_STRICT_SOLVER_OPTS,
    )

    assert isinstance(result_rot._expect_data, dict)
    assert isinstance(result_rot._expect_data["r"], ObservableTrace)
    assert result_rot._expect_data["r"].values.shape == tlist.shape
    assert result_rot._expect_data["r"].raw.shape == tlist.shape
    assert "expect=dict(" in repr(result_rot)
    assert "expect_raw" not in repr(result_rot)

    seq_lab, chip_lab, _ = _build_coupled_sequence(frame="lab")
    a_r_lab = chip_lab.device_map["r"].lowering_operator()
    init_lab = chip_lab.bare_state(
        q=0,
        r=backend.coherent(chip_lab.device_map["r"].levels, 0.2),
    )
    result_lab = seq_lab.simulate(
        tlist=tlist,
        e_ops={"r": a_r_lab},
        initial_state=init_lab,
        options=_STRICT_SOLVER_OPTS,
    )

    assert isinstance(result_lab._expect_data, dict)
    assert isinstance(result_lab._expect_data["r"], ObservableTrace)
    assert "expect=dict(" in repr(result_lab)
    assert "expect_raw" not in repr(result_lab)


# ---------------------------------------------------------------------------
# List-valued dict e_ops
# ---------------------------------------------------------------------------


def test_list_value_lab_frame(backend) -> None:
    """Lab-frame dict e_ops with list value returns list of arrays."""
    seq_lab, chip_lab, tlist = _build_coupled_sequence(frame="lab")

    a_r = chip_lab.device_map["r"].lowering_operator()
    n_r = chip_lab.device_map["r"].number_operator()

    init_state = chip_lab.bare_state(
        r=backend.coherent(chip_lab.device_map["r"].levels, 0.2),
    )

    result = seq_lab.simulate(
        tlist=tlist,
        e_ops={"r": [n_r, a_r]},
        initial_state=init_state,
        options=_STRICT_SOLVER_OPTS,
    )

    assert isinstance(result._expect_data, dict)
    assert "r" in result._expect_data
    # List input → list output
    assert isinstance(result._expect_data["r"], list)
    assert len(result._expect_data["r"]) == 2
    assert all(isinstance(item, ObservableTrace) for item in result._expect_data["r"])
    assert result._expect_data["r"][0].values.shape == tlist.shape
    assert result._expect_data["r"][1].values.shape == tlist.shape
    assert result._expect_data["r"][0].raw.shape == tlist.shape
    assert result._expect_data["r"][1].raw.shape == tlist.shape


def test_list_value_rotating_frame(backend) -> None:
    """Rotating-frame dict e_ops with list value returns list of arrays."""
    seq_rot, chip_rot, tlist = _build_coupled_sequence(frame="rotating")

    a_r = chip_rot.device_map["r"].lowering_operator()
    n_r = chip_rot.device_map["r"].number_operator()

    init_state = chip_rot.bare_state(
        r=backend.coherent(chip_rot.device_map["r"].levels, 0.2),
    )

    result = seq_rot.simulate(
        tlist=tlist,
        e_ops={"r": [n_r, a_r]},
        initial_state=init_state,
        options=_STRICT_SOLVER_OPTS,
    )

    assert isinstance(result._expect_data, dict)
    assert "r" in result._expect_data
    assert isinstance(result._expect_data["r"], list)
    assert len(result._expect_data["r"]) == 2
    assert all(isinstance(item, ObservableTrace) for item in result._expect_data["r"])
    assert result._expect_data["r"][0].values.shape == tlist.shape
    assert result._expect_data["r"][1].values.shape == tlist.shape


def test_list_value_mixed_with_scalar(backend) -> None:
    """Dict e_ops mixing list and scalar values works correctly."""
    seq_lab, chip_lab, tlist = _build_coupled_sequence(frame="lab")

    a_q = chip_lab.device_map["q"].lowering_operator()
    a_r = chip_lab.device_map["r"].lowering_operator()
    n_r = chip_lab.device_map["r"].number_operator()

    init_state = chip_lab.bare_state(
        q=backend.coherent(chip_lab.device_map["q"].levels, 0.1),
        r=backend.coherent(chip_lab.device_map["r"].levels, 0.2),
    )

    result = seq_lab.simulate(
        tlist=tlist,
        e_ops={"r": [n_r, a_r], "q": a_q},
        initial_state=init_state,
        options=_STRICT_SOLVER_OPTS,
    )

    # "r" was list → list output
    assert isinstance(result._expect_data["r"], list)
    assert len(result._expect_data["r"]) == 2

    # "q" was scalar → array output (backwards compat)
    assert not isinstance(result._expect_data["q"], list)
    assert isinstance(result._expect_data["q"], ObservableTrace)
    assert result._expect_data["q"].values.shape == tlist.shape


def test_list_value_backwards_compat(backend) -> None:
    """Single-operator dict values still produce single arrays, not lists."""
    seq_lab, chip_lab, tlist = _build_coupled_sequence(frame="lab")

    a_r = chip_lab.device_map["r"].lowering_operator()

    init_state = chip_lab.bare_state(
        r=backend.coherent(chip_lab.device_map["r"].levels, 0.2),
    )

    result = seq_lab.simulate(
        tlist=tlist,
        e_ops={"r": a_r},
        initial_state=init_state,
        options=_STRICT_SOLVER_OPTS,
    )

    # Scalar value → array, NOT wrapped in list
    assert not isinstance(result._expect_data["r"], list)
    assert isinstance(result._expect_data["r"], ObservableTrace)
    assert result._expect_data["r"].values.shape == tlist.shape
