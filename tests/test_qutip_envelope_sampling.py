"""QuTiP samples windowed envelopes at the rate that the solver's rtol needs."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import quchip.backend.qutip as qutip_backend
from quchip import ChargeDrive, Chip, DuffingTransmon, Gaussian, QuantumSequence, Square
from quchip.engine.ir import decompose_carrier_bands

EXACT = {"method": "vern9", "rtol": 1e-12, "atol": 1e-14, "nsteps": 10**6}


def _pi_pulse(duration: float, start: float = 0.0, envelope: type = Gaussian):
    """Return a 3-level transmon and a sequence with one π pulse of *duration* ns starting at *start*."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating", backend="qutip")
    xy = ChargeDrive(q, label="xy")
    chip.wire(xy)
    shape = envelope(duration=duration, sigmas=4.0) if envelope is Gaussian else envelope(duration=duration)
    amplitude = float(chip.rotation_amplitude(q, xy, shape))
    sequence = QuantumSequence(chip)
    sequence.schedule(xy, envelope=shape.with_params({"amplitude": amplitude}), freq=chip.freq(q), start_time=start)
    return q, sequence


def _excited_population(sequence, q, span, options, **kwargs):
    return float(sequence.simulate(tlist=np.array([0.0, span]), options=options, **kwargs).population(q, 1)[-1])


def _exact_population(monkeypatch, sequence, q, span):
    """Solve with the exact envelope callable at tight tolerances, the reference for every grid."""
    with monkeypatch.context() as patch, warnings.catch_warnings():
        patch.setattr(qutip_backend, "_MAX_WINDOW_SAMPLES", 0)
        warnings.filterwarnings("ignore", message="At rtol=")
        return _excited_population(sequence, q, span, EXACT)


def _window_envelope(duration: float):
    q, sequence = _pi_pulse(duration)
    signal = sequence.build_problem(np.array([0.0, duration])).engine_result.dynamic_terms[0].time_dependence.signal
    return decompose_carrier_bands(signal)[0].envelope


@pytest.mark.unit
@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
def test_window_samples_follow_the_linear_interpolation_bound(rtol):
    """A Gaussian has max|f''| / max|f| = 1/σ², so the rule gives about T / (σ sqrt(8 rtol)) samples."""
    duration = 12.0
    sigma = duration / 8.0
    count = qutip_backend._window_interior_samples(_window_envelope(duration), 0.0, duration, rtol)
    assert count == pytest.approx(duration / (sigma * np.sqrt(8 * rtol)), rel=0.02)


@pytest.mark.unit
def test_a_flat_window_keeps_the_base_rate():
    """A square envelope has no curvature, so its window keeps 40 points/ns."""
    q, sequence = _pi_pulse(10.0, envelope=Square)
    signal = sequence.build_problem(np.array([0.0, 10.0])).engine_result.dynamic_terms[0].time_dependence.signal
    envelope = decompose_carrier_bands(signal)[0].envelope
    assert qutip_backend._window_interior_samples(envelope, 0.0, 10.0, 1e-10) == 401


@pytest.mark.parametrize(("duration", "start", "span"), [(4.0, 0.0, 4.0), (12.0, 0.0, 12.0), (12.0, 5.0, 27.0)])
@pytest.mark.parametrize(("options", "bound"), [(None, 3e-6), ({"rtol": 1e-8, "atol": 1e-10}, 1e-7)])
def test_a_pi_pulse_reaches_the_accuracy_of_its_rtol(monkeypatch, duration, start, span, options, bound):
    """The QuTiP P1 error follows rtol, with and without idle time around the pulse.

    With 40 points/ns at any rtol, the error was 8.6e-5 for the 4 ns pulse and 4.1e-6 for the 12 ns one.
    """
    q, sequence = _pi_pulse(duration, start)
    reference = _exact_population(monkeypatch, sequence, q, span)
    assert reference > 0.3  # the 4 ns pulse leaks to |2>
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        population = _excited_population(sequence, q, span, options)
    assert abs(population - reference) < bound


def test_a_window_over_the_sample_cap_uses_the_exact_envelope_and_warns(monkeypatch):
    """Above the cap, one warning names the window, and the callable keeps the accuracy."""
    q, sequence = _pi_pulse(12.0, start=5.0)
    reference = _exact_population(monkeypatch, sequence, q, 27.0)
    monkeypatch.setattr(qutip_backend, "_MAX_WINDOW_SAMPLES", 1000)
    with pytest.warns(UserWarning) as record:
        population = _excited_population(sequence, q, 27.0, {"rtol": 1e-8, "atol": 1e-10})
    messages = [str(item.message) for item in record]
    assert len(messages) == 1 and record[0].filename == __file__
    assert "linear samples for the pulse window from 5 to 17 ns, more than 1000" in messages[0]
    assert abs(population - reference) < 1e-7


def test_a_batch_warns_once_for_its_windows_over_the_cap(monkeypatch):
    """The main process warns once for a QuTiP batch, whose workers do not report warnings."""
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    chip = Chip([q], frame="rotating", backend="qutip")
    chip.wire(ChargeDrive(q, label="xy"))
    sequence = QuantumSequence(chip)
    pulse = sequence.schedule("xy", envelope=Gaussian(duration=12.0, sigmas=4.0, amplitude=0.02), freq=5.0)
    monkeypatch.setattr(qutip_backend, "_MAX_WINDOW_SAMPLES", 1000)
    with pytest.warns(UserWarning) as record:
        sequence.simulate_batch(pulse.vary("amplitude", [0.01, 0.02]), tlist=np.array([0.0, 12.0]),
                                options={"rtol": 1e-8}, max_workers=1, progress=False)
    messages = [str(item.message) for item in record if "linear samples" in str(item.message)]
    assert len(messages) == 1 and "more windows, uses the exact envelope" in messages[0]
