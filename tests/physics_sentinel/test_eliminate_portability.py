"""Same flux schedule replayed on the full chip and its eliminate()-reduced chip agrees (spec Sec. 9, rung 1)."""

from __future__ import annotations

import warnings

import numpy as np

from quchip import (
    Capacitive,
    Chip,
    ControlEquipment,
    DuffingTransmon,
    FluxDrive,
    FluxTunableTransmon,
    QuantumSequence,
    Square,
)
from quchip.chip.transformations import eliminate
from quchip.declarative import EnvelopeShape, Scalar, parameter, qnp

_LEG_G = 0.08
_BRIDGE_FREQ = 6.3
_PUMP_AMPLITUDE = 0.05  # GHz; baseband delta-omega_c excursion


class CosineTone(EnvelopeShape):
    """Baseband cosine tone: the envelope itself carries the oscillation.

    Both the full chip's ``FluxDrive`` and the reduced chip's retargeted
    ``ParametricDrive`` ignore ``freq`` (both are baseband-only channels), so
    a schedule call that needs an oscillating term at a chosen frequency must
    carry it inside the envelope. One ``CosineTone`` type serves both chips
    through the identical ``schedule("cflux", envelope=...)`` call.
    """

    duration: Scalar = parameter(positive=True, unit="ns")
    frequency: Scalar = parameter(unit="GHz")
    amplitude: Scalar = parameter(default=1.0)

    def value(self, t: object) -> object:
        """Evaluate ``amplitude * cos(2*pi*frequency*t)``."""
        return qnp.asarray(self.amplitude * qnp.cos(2 * qnp.pi * self.frequency * t), dtype=complex)


def _bridge_chip(q0_freq: float, q1_freq: float) -> Chip:
    q0 = DuffingTransmon(freq=q0_freq, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=q1_freq, anharmonicity=-0.24, levels=3, label="q1")
    fc = FluxTunableTransmon(freq=_BRIDGE_FREQ, anharmonicity=-0.2, levels=3, label="fc")
    couplings = [
        Capacitive(q0, fc, g=_LEG_G, label="leg0"),
        Capacitive(q1, fc, g=_LEG_G, label="leg1"),
    ]
    flux = FluxDrive(fc, label="cflux")
    return Chip(
        [q0, q1, fc],
        couplings=couplings,
        control_equipment=ControlEquipment([flux]),
        frame="rotating",
        rwa=True,
    )


def _replay_and_measure(full: Chip, reduced: Chip, duration: float) -> tuple[float, float]:
    """Schedule the flux pulse by its drive label on both chips; return final q1 populations."""
    tlist = np.linspace(0.0, duration, 11)
    with warnings.catch_warnings():
        # Exactly-degenerate survivors (the companion test below) have no bare
        # eigenbasis of their own once any exchange is present; the dressed-state
        # labeling pass this triggers (in bare_state and again while solving) warns
        # about the near-degenerate assignment, but still returns/uses the exact
        # bare product state.
        warnings.simplefilter("ignore", UserWarning)
        init_full = full.bare_state({"q0": 1, "q1": 0, "fc": 0})
        init_reduced = reduced.bare_state({"q0": 1, "q1": 0})

        seq_full = QuantumSequence(full)
        seq_full.schedule("cflux", envelope=Square(duration=duration, amplitude=_PUMP_AMPLITUDE))
        result_full = seq_full.simulate(tlist=tlist, initial_state=init_full)

        seq_reduced = QuantumSequence(reduced)
        seq_reduced.schedule("cflux", envelope=Square(duration=duration, amplitude=_PUMP_AMPLITUDE))
        result_reduced = seq_reduced.simulate(tlist=tlist, initial_state=init_reduced)

    pop_full = float(np.real(result_full.population(full["q1"], 1))[-1])
    pop_reduced = float(np.real(result_reduced.population(reduced["q1"], 1))[-1])
    return pop_full, pop_reduced


def test_rung1_full_vs_reduced_agree_at_pump_derived_exchange_period():
    """Ladder rung 1 fixture (q0=5.0, q1=5.2, fc=6.3 idle): full and reduced chips agree."""
    full = _bridge_chip(5.0, 5.2)
    res = eliminate(full, "fc")
    reduced = res.chip
    exchange = res.effective_params["exchange"]
    # delta_J is the pump-only contribution to the mediated exchange; g/Delta ~=
    # 0.062-0.073 and delta_J/Delta ~= 0.03-0.04 on the two legs, both
    # second-order-small (eliminate()'s declared validity). The 0.2 GHz detuning
    # dwarfs delta_J, so the transferred population stays near zero on both
    # chips; the check is that the two representations agree, not that a swap
    # completes.
    delta_j = float(exchange["dJ_domega_c"]) * _PUMP_AMPLITUDE
    duration = 1.0 / (4.0 * delta_j)

    pop_full, pop_reduced = _replay_and_measure(full, reduced, duration)

    # Measured: pop_full = 0.0015109, pop_reduced = 0.0020303, |diff| = 0.00052.
    assert abs(pop_full - pop_reduced) < 0.05


def test_rung1_degenerate_companion_validates_exchange_rate_formula():
    """Degenerate companion (q0=q1=5.0): reduced-chip exchange rate matches dJ_domega_c * delta_omega."""
    full = _bridge_chip(5.0, 5.0)
    res = eliminate(full, "fc")
    reduced = res.chip
    exchange = res.effective_params["exchange"]
    # With degenerate survivors the leg-mediated static exchange j_eff and the
    # pump contribution dJ_domega_c * pump_amplitude add to one rate J_tot; a
    # resonant pair completes a full swap in a quarter period 1/(4*|J_tot|),
    # which makes a checkable prediction for the reduced chip's own exchange
    # rate. The pump-only period 1/(4*dJ_domega_c*pump_amplitude) would instead
    # integrate several periods of the much larger static exchange and compare
    # full vs reduced at an accumulated-phase-drift instant rather than at a
    # swap extremum.
    j_total = float(exchange["j_eff"]) + float(exchange["dJ_domega_c"]) * _PUMP_AMPLITUDE
    duration = 1.0 / (4.0 * abs(j_total))

    pop_full, pop_reduced = _replay_and_measure(full, reduced, duration)

    # Measured: pop_reduced = 0.999999999968 (J_tot's swap-period prediction
    # holds on the reduced chip alone); pop_full = 0.991251, |diff| = 0.00875.
    assert pop_reduced > 0.999
    assert abs(pop_full - pop_reduced) < 0.05


def test_three_survivor_replay_full_vs_reduced():
    """Three degenerate survivors on one coupler: full and reduced chips agree, symmetrically, on both receivers."""
    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q1")
    q2 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q2")
    fc = FluxTunableTransmon(freq=_BRIDGE_FREQ, anharmonicity=-0.2, levels=3, label="fc")
    couplings = [Capacitive(q, fc, g=_LEG_G, label=f"leg{i}") for i, q in enumerate((q0, q1, q2))]
    full = Chip(
        [q0, q1, q2, fc],
        couplings=couplings,
        control_equipment=ControlEquipment([FluxDrive(fc, label="cflux")]),
        frame="rotating",
        rwa=True,
    )
    res = eliminate(full, "fc")
    reduced = res.chip
    exchange = res.effective_params["exchange"]
    assert set(exchange) == {("q0", "q1"), ("q0", "q2"), ("q1", "q2")}

    pair = exchange[("q0", "q1")]
    j_total = float(pair["j_eff"]) + float(pair["dJ_domega_c"]) * _PUMP_AMPLITUDE
    duration = 1.0 / (4.0 * abs(j_total))
    tlist = np.linspace(0.0, duration, 11)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        init_full = full.bare_state({"q0": 1, "q1": 0, "q2": 0, "fc": 0})
        init_reduced = reduced.bare_state({"q0": 1, "q1": 0, "q2": 0})
        seq_full = QuantumSequence(full)
        seq_full.schedule("cflux", envelope=Square(duration=duration, amplitude=_PUMP_AMPLITUDE))
        result_full = seq_full.simulate(tlist=tlist, initial_state=init_full)
        seq_reduced = QuantumSequence(reduced)
        seq_reduced.schedule("cflux", envelope=Square(duration=duration, amplitude=_PUMP_AMPLITUDE))
        result_reduced = seq_reduced.simulate(tlist=tlist, initial_state=init_reduced)

    # Measured: pop_full = 0.22743, pop_reduced = 0.22222, |diff| = 0.0052 for
    # both receiving survivors — identical on the crosstalk-fed pump and the
    # directly scheduled one.
    # g/Delta = 0.062, delta_omega/Delta = 0.038 (both second-order-small, eliminate()'s declared validity).
    for target in ("q1", "q2"):
        pop_full = float(np.real(result_full.population(full[target], 1))[-1])
        pop_reduced = float(np.real(result_reduced.population(reduced[target], 1))[-1])
        assert abs(pop_full - pop_reduced) < 0.05, (target, pop_full, pop_reduced)

    # The two receiving survivors are fully symmetric on both chips. The model
    # symmetry is exact; the realized asymmetry is adaptive-solver noise and
    # varies with platform BLAS (observed 1.9e-6 on Linux/OpenBLAS runners),
    # so the bound is a solver-accuracy floor, not a physics tolerance.
    pop_q1 = float(np.real(result_reduced.population(reduced["q1"], 1))[-1])
    pop_q2 = float(np.real(result_reduced.population(reduced["q2"], 1))[-1])
    assert abs(pop_q1 - pop_q2) < 1e-5


def test_tone_form_parametric_resonance_full_vs_reduced():
    """Parametric drive at the dressed |10>,|01> splitting swaps population; full and reduced chips agree."""
    full = _bridge_chip(5.0, 5.2)
    res = eliminate(full, "fc")
    reduced = res.chip
    exchange = res.effective_params["exchange"]
    j_eff = float(exchange["j_eff"])
    delta_bare = abs(
        float(res.effective_params["q1"]["freq_after"]) - float(res.effective_params["q0"]["freq_after"])
    )
    # Dressed |10>,|01> splitting including the static residual exchange's level repulsion.
    dressed_splitting = float(np.sqrt(delta_bare**2 + (2.0 * j_eff) ** 2))
    delta_j = float(exchange["dJ_domega_c"]) * _PUMP_AMPLITUDE  # A_J: peak delta_J(t) excursion, GHz
    rabi_rate = delta_j / 2.0  # RWA halving: cos(2*pi*nu*t) resonant term contributes half its amplitude
    duration = 1.0 / (4.0 * rabi_rate)  # quarter-period full-transfer prediction, same convention as rung 1

    n_points = 6001  # ~13.6 samples per carrier period (1/dressed_splitting ~= 5.02 ns); converged (checked to n=30001)
    tlist = np.linspace(0.0, duration, n_points)
    envelope = CosineTone(duration=duration, amplitude=_PUMP_AMPLITUDE, frequency=dressed_splitting)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        init_full = full.bare_state({"q0": 1, "q1": 0, "fc": 0})
        init_reduced = reduced.bare_state({"q0": 1, "q1": 0})

        seq_full = QuantumSequence(full)
        seq_full.schedule("cflux", envelope=envelope)
        result_full = seq_full.simulate(tlist=tlist, initial_state=init_full)

        seq_reduced = QuantumSequence(reduced)
        seq_reduced.schedule("cflux", envelope=envelope)
        result_reduced = seq_reduced.simulate(tlist=tlist, initial_state=init_reduced)

    pop_full = float(np.real(result_full.population(full["q1"], 1))[-1])
    pop_reduced = float(np.real(result_reduced.population(reduced["q1"], 1))[-1])

    # g/Delta = 0.0615 (leg0), 0.0727 (leg1); delta_omega/Delta = 0.2/1.1 = 0.1818 (the
    # larger, leg1-referenced ratio) -- all second-order-small (eliminate()'s declared
    # validity). Tolerance = max(g/Delta, delta_omega/Delta)**2 = 0.1818**2 = 0.0330.
    # Measured: pop_reduced = 0.999479, pop_full = 0.977404, |diff| = 0.022076.
    assert pop_reduced > 0.8
    assert abs(pop_full - pop_reduced) < 0.1818**2
