"""A removed readout retains both directions of its hanger response."""
from __future__ import annotations

import json

import numpy as np
import pytest

from quchip import (
    Capacitive, Chip, DuffingTransmon, EffectiveTerms, Exact, PhysicsExpr, PortNetwork, QuantumSequence, Resonator,
    RWA, Square, VNA, eliminate,
)


def _hanger(count=2, *, g=0.04, backend="qutip", approximation=None, effective_terms=(), probe=False):
    qubits = [DuffingTransmon(freq=5.32 - .07*i, anharmonicity=-.26, levels=3 if count == 3 else 2, label=f"q{i}")
              for i in range(count)]
    readouts = [Resonator(freq=6.56 + .05*i, levels=2, internal_quality_factor=2e5, label=f"r{i}")
                for i in range(count)]
    network = PortNetwork(label="feedline")
    right = [network.port(f"r{i}_right", target=r, rate=2*np.pi*.001*(1 + .2*i), phase=.17*i)
             for i, r in enumerate(readouts)]
    left = [network.port(f"r{i}_left", target=readouts[i], rate=2*np.pi*.001*(1 - .1*i), phase=-.13*i)
            for i in reversed(range(count))]
    if count > 1:
        network.cascade(*right)
        network.cascade(*left)
    network.expose("right", input=right[0].input, output=right[-1].output)
    network.expose("left", input=left[0].input, output=left[-1].output)
    if probe:
        # A second line on r0 alone, which no retained response reaches.
        port = network.port("r0_probe", target=readouts[0], rate=2*np.pi*.0005)
        network.expose("probe", input=port.input, output=port.output)
    return Chip(qubits + readouts, [Capacitive(q, r, g=g) for q, r in zip(qubits, readouts)],
                port_network=network, frame=5.2, approximation=RWA() if approximation is None else approximation,
                backend=backend, effective_terms=effective_terms)


def _directional_rates(chip):
    resolved = chip.resolve(frame="lab")
    dressed = resolved.dress(overlap_threshold=0.)
    labels = tuple(d.label for d in chip.devices)

    def vector(label):
        return np.asarray(chip.backend.to_array(dressed.eigenstates[dressed.state_map[label]])).reshape(-1)

    ground = vector((0,) * len(labels))
    return np.asarray([
        [abs(ground.conj() @ np.asarray(channel.coupling.to_dense())
             @ vector(tuple(int(label == qubit) for label in labels)))**2
         for channel in resolved.slh.external_channels]
        for qubit in labels if qubit.startswith("q")
    ])


@pytest.mark.parametrize("count", [2, pytest.param(3, marks=pytest.mark.validation)])
@pytest.mark.parametrize(("method", "local"), [("sw", False), ("sw", True), ("exact", False)])
def test_hanger_elimination_keeps_every_dip_and_directional_decay(count, method, local):
    """Both field directions, dressed frequencies and Purcell rates retain their dispersive accuracy."""
    chip = _hanger(count)
    modes = ("r1", "r0")
    poles = [float(chip.freq(f"r{i}")) for i in range(count)]
    frequencies = np.concatenate([pole + np.linspace(-.003, .003, 31) for pole in poles])
    full = np.asarray(VNA(chip).sweep(frequencies).matrix)
    original_rates = _directional_rates(chip)
    detuning = min(abs(chip[mode].freq - chip[f"q{mode[1:]}"].freq) for mode in modes)
    kappa = max(sum(float(p.rate_value(chip)) for p in chip.ports if p.resolve_targets(chip) == (mode,))
                for mode in modes)
    mixing = (.04 / detuning)**2
    # Existing dispersive boundary error, plus SW's fourth-order shift converted
    # to scattering near resonance. Sum the two removed modes' error scales.
    scattering_bound = 2 * mixing * kappa / (2*np.pi*detuning)
    if method == "sw":
        scattering_bound += 2 * 8*np.pi*.04**4 / (kappa*detuning**3)
    reduced = chip
    for mode in modes:
        reduced = eliminate(reduced, mode, method=method, local=local).chip
        reduced = Chip.from_dict(json.loads(json.dumps(reduced.to_dict())))
        actual = np.asarray(VNA(reduced).sweep(frequencies).matrix)
        np.testing.assert_allclose(actual, full, atol=scattering_bound, rtol=0)
        # Markov channels omit the removed mode's subsequent scattering at each
        # qubit frequency. SW also leaves the O((g/Delta)^2) relative rate error.
        rate_bound = 2*kappa/(2*np.pi*detuning) + (3*mixing if method == "sw" else 0)
        np.testing.assert_allclose(_directional_rates(reduced), original_rates, rtol=rate_bound, atol=1e-12)
        # The retained Hamiltonian omits second-order shifts from directional
        # network couplings, bounded by |J| <= kappa/(2*pi).
        frequency_bound = 2*(kappa/(2*np.pi))**2/.05 + (2*.04**4/detuning**3 if method == "sw" else 0)
        for device in reduced.devices:
            assert abs(reduced.freq(device.label) - chip.freq(device.label)) < frequency_bound


@pytest.mark.unit
def test_multiport_pole_matches_rank_one_scattering_and_rebinding():
    """Unequal port rates and phases give one coupled pole, which follows network parameter rebinding."""
    chip = _hanger(1, g=0.)
    chip = chip.with_params({"port.r0_right.rate": .007, "port.r0_left.rate": .011,
                             "port.r0_right.phase": .21, "port.r0_left.phase": -.37})
    reduced = eliminate(chip, "r0").chip
    shifted = reduced.with_params({"network.response.r0.freq": 6.6})
    frequencies = np.array([6.58, 6.6, 6.62])
    v = np.sqrt([.007, .011]) * np.exp(1j*np.array([.21, -.37]))
    loss = 2*np.pi*chip["r0"].freq / 2e5
    # The VNA convention conjugates the internal exp(-i*omega*t) solution.
    expected = (np.eye(2) - v[None, :, None] * v.conj()[None, None, :]
                / ((.018 + loss)/2 - 2j*np.pi*(frequencies - 6.6))[:, None, None]).conj()
    # With g = 0 the reduction is exact. The probes sit on the pole or far from it,
    # where round-off stays below 1e-14.
    np.testing.assert_allclose(VNA(shifted).sweep(frequencies).matrix, expected, atol=1e-12, rtol=0)
    assert reduced.parameters["network.response.r0.freq"] == pytest.approx(chip["r0"].freq)
    symmetric = eliminate(_hanger(1, g=0.), "r0").chip
    s = np.asarray(VNA(symmetric).sweep(frequencies).matrix)
    # Equal port amplitudes make S00 - 1 equal S10 up to round-off.
    np.testing.assert_allclose(s[:, 0, 0], 1 + s[:, 1, 0], atol=1e-12, rtol=0)


@pytest.mark.parametrize("operation", ["finite", "conditional", "options", "transient"])
def test_multiport_response_rejects_unsupported_field_routes(operation):
    """Unsupported field calculations cannot silently discard the removed pole."""
    reduced = eliminate(_hanger(), "r1").chip
    with pytest.raises(NotImplementedError, match="multiport mode responses"):
        vna = VNA(reduced)
        if operation == "finite":
            vna.finite_power([6.6], [.01], input="right")
        elif operation == "conditional":
            vna.sweep([6.6], when={"q1": 1})
        elif operation == "options":
            vna.sweep([6.6], options={})
        else:
            sequence = QuantumSequence(reduced)
            sequence.schedule(reduced.port_network.external_port("left").input,
                              envelope=Square(duration=2., amplitude=.01), freq=6.61)
            sequence.build_problem(np.linspace(0., 2., 5))


def test_multiport_response_rejects_time_domain_fields_and_records_on_its_line():
    """Output fields and monitored records that carry the pole's emission raise; other observables run."""
    from dataclasses import replace

    from quchip import with_monitoring
    from quchip.engine import solve_problem
    from quchip.observables import OutputField

    reduced = eliminate(_hanger(probe=True), "r1").chip
    sequence = QuantumSequence(reduced)
    tlist = np.linspace(0., 2., 3)
    for plane in ("right", "left"):
        with pytest.raises(NotImplementedError, match=rf"Output field '{plane}' .*\['r1'\]"):
            sequence.build_problem(tlist, e_ops={plane: OutputField(plane)})
    sequence.build_problem(tlist, e_ops={"probe": OutputField("probe")})
    base = sequence.build_problem(tlist, e_ops=reduced.e_ops(r0="n"), initial_state={"r0": 1})
    options = {"dt": 0.01, "progress_bar": ""}
    sse = replace(base, solver="ssesolve", run_args={"ntraj": 2, "seeds": 1}, options=options)
    sme = replace(sse, solver="smesolve")
    # Pure-state SSE monitors every channel, the affected lines included.
    for problem in (sse, with_monitoring(sme, {"right": .5})):
        with pytest.raises(NotImplementedError, match=r"Monitored channel 'right' .*\['r1'\]"):
            solve_problem(problem)
    for selection in ({"probe": 1.}, {"right": 0., "probe": 1.}):
        result = solve_problem(with_monitoring(sme, selection))
        assert np.all(np.isfinite(np.real(result.expect("r0"))))


def test_multiport_elimination_order_and_vacuum_reference_agree():
    """Repeated elimination preserves the same coupled response in either order and for an explicit vacuum."""
    chip = _hanger()
    frequencies = np.array([6.559, 6.562, 6.609, 6.612])
    responses = []
    for order in (("r0", "r1"), ("r1", "r0")):
        reduced = chip
        for mode in order:
            reduced = eliminate(reduced, mode, method="exact").chip
        responses.append(np.asarray(VNA(reduced).sweep(frequencies, when={}).matrix))
        # The explicit vacuum and the default sweep solve the same problem.
        np.testing.assert_allclose(responses[-1], VNA(reduced).sweep(frequencies).matrix, atol=1e-12, rtol=0)
    # One ulp of a pole's angular frequency moves S by 4*eps*omega/kappa, about 3e-12
    # at resonance, so 1e-11 allows a few ulps between the two orders.
    np.testing.assert_allclose(responses[0], responses[1], atol=1e-11, rtol=0)


@pytest.mark.parametrize(("method", "local"), [("sw", False), ("sw", True), ("exact", False)])
@pytest.mark.parametrize("upstream", [True, False])
def test_mixed_port_elimination_preserves_complex_matrix_order(method, local, upstream):
    """A scalar pole stays on its original side of a two-port pole in either removal order."""
    qa = DuffingTransmon(freq=4.8, anharmonicity=-.2, levels=2, T1=1000, label="qa")
    qb = DuffingTransmon(freq=5.1, anharmonicity=-.2, levels=2, T1=1200, label="qb")
    a = Resonator(freq=6.4, levels=2, label="a")
    b = Resonator(freq=6.5, levels=2, label="b")
    network = PortNetwork()
    pa = network.port("a_port", target=a, rate=.017)
    br = network.port("b_right", target=b, rate=.013, phase=.19)
    bl = network.port("b_left", target=b, rate=.013, phase=-.23)
    line = (pa, br) if upstream else (br, pa)
    network.cascade(*line)
    network.expose("right", input=line[0].input, output=line[-1].output)
    network.expose("left", at=bl)
    chip = Chip([qa, qb, a, b], [Capacitive(qa, a, g=0.), Capacitive(qb, b, g=0.)],
                port_network=network, frame="lab", approximation=RWA())
    frequencies = np.array([6.399, 6.4, 6.401, 6.499, 6.5, 6.501])
    scalar = 1 - .017 / (.017/2 - 2j*np.pi*(frequencies - a.freq))
    diagonal = np.broadcast_to(np.eye(2, dtype=complex), (len(frequencies), 2, 2)).copy()
    diagonal[:, 0, 0] = scalar
    v = np.sqrt(.013) * np.exp(1j*np.array([.19, -.23]))
    hanger = (np.eye(2) - v[None, :, None] * v.conj()[None, None, :]
              / (.013 - 2j*np.pi*(frequencies - b.freq))[:, None, None])
    expected = (hanger @ diagonal if upstream else diagonal @ hanger).conj()
    # Zero couplings make the reduction exact. One ulp of the lab-frame pole frequency
    # moves S by up to 4*eps*omega/kappa = 2e-12.
    np.testing.assert_allclose(VNA(chip).sweep(frequencies).matrix, expected, atol=2e-12, rtol=0)
    for order in (("a", "b"), ("b", "a")):
        reduced = chip
        for mode in order:
            reduced = eliminate(reduced, mode, method=method, local=local).chip
            np.testing.assert_allclose(VNA(reduced).sweep(frequencies).matrix, expected, atol=2e-12, rtol=0)


@pytest.mark.unit
def test_retained_pole_materializes_scalar_expressions():
    """Symbolic pole frequency, weight and loss use the same scalar lowering as Hamiltonian coefficients."""
    reduced = eliminate(_hanger(1, g=0.), "r0").chip
    values = {"network.response.r0.freq": 6.6, "network.response.r0.weight": .7,
              "network.response.r0.internal_rate": .003}
    numeric = reduced.with_params(values)
    symbolic = reduced.with_params({name: PhysicsExpr.literal(value) for name, value in values.items()})
    frequencies = np.array([6.598, 6.6, 6.602])
    # Both chips lower the same float values, so the solves agree to round-off.
    np.testing.assert_allclose(VNA(symbolic).sweep(frequencies).matrix,
                               VNA(numeric).sweep(frequencies).matrix, atol=1e-13, rtol=0)


@pytest.mark.optional_backend
@pytest.mark.validation
def test_multiport_pole_jit_gradient_matches_finite_difference():
    """A retained pole's phase and frequency affect JIT VNA derivatives without changing topology."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    reduced = eliminate(_hanger(backend="dynamiqs"), "r1").chip

    def response(values):
        rebound = reduced.with_params({"network.response.r1.freq": values[0], "port.r1_left.phase": values[1]})
        return jnp.real(VNA(rebound).sweep(jnp.array([6.6105])).matrix[0, 1, 0])

    values = jnp.array([6.611, .3])
    actual = jax.jit(jax.grad(response))(values)
    steps = np.array([1e-7, 1e-5])
    expected = [(float(response(values + jnp.eye(2)[i]*step))
                 - float(response(values - jnp.eye(2)[i]*step)))/(2*step) for i, step in enumerate(steps)]
    # Central differences at these steps are within 1e-8 relative of the exact derivative.
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.optional_backend
@pytest.mark.validation
def test_lossless_pole_gradient_through_elimination_matches_finite_difference():
    """A coupling derivative through eliminate() of a lossless readout matches finite differences."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    def reflection(g):
        chip = _hanger(1, g=g, backend="dynamiqs").with_params({"r0.internal_quality_factor": np.inf})
        return jnp.real(VNA(eliminate(chip, "r0").chip).sweep(jnp.array([6.5618])).matrix[0, 1, 0])

    step = 1e-6
    expected = (float(reflection(.04 + step)) - float(reflection(.04 - step)))/(2*step)
    # The central difference converges as step**2 and is within 1e-8 relative at this step.
    np.testing.assert_allclose(jax.jit(jax.grad(reflection))(.04), expected, rtol=1e-6)


@pytest.mark.validation
def test_lossless_hanger_response_converges_to_the_full_linear_limit():
    """Without discarded internal-loss correlations, exact reduction preserves the full single-excitation transfer."""
    chip = _hanger().with_params({"r0.internal_quality_factor": np.inf, "r1.internal_quality_factor": np.inf})
    frequencies = np.array([6.559, 6.561, 6.609, 6.611, 6.613])
    full = VNA(chip).sweep(frequencies).matrix
    reduced = eliminate(eliminate(chip, "r0", method="exact").chip, "r1", method="exact").chip
    # Only round-off remains. One ulp of a pole's angular frequency moves S by
    # 4*eps*omega/kappa, about 3e-12 at resonance.
    np.testing.assert_allclose(VNA(reduced).sweep(frequencies).matrix, full, atol=2e-11, rtol=0)


@pytest.mark.unit
def test_multiport_elimination_rejects_two_ports_on_one_line():
    """Two ports of one mode on the same directional line raise instead of forming a pole."""
    qubit = DuffingTransmon(freq=5.3, anharmonicity=-.26, levels=2, label="q")
    readout = Resonator(freq=6.56, levels=2, label="r")
    network = PortNetwork(label="line")
    first = network.port("r_first", target=readout, rate=.006)
    second = network.port("r_second", target=readout, rate=.004)
    network.cascade(first, second)
    network.expose("line", input=first.input, output=second.output)
    chip = Chip([qubit, readout], [Capacitive(qubit, readout, g=.04)], port_network=network, approximation=RWA())
    with pytest.raises(NotImplementedError, match="share a directional line"):
        eliminate(chip, "r")


@pytest.mark.unit
@pytest.mark.parametrize("local", [False, True])
def test_multiport_elimination_requires_an_excitation_conserving_chip(local):
    """A nonconserving model raises, also when the nonconserving term lies outside the local patch."""
    if local:
        drive = EffectiveTerms(("r0",), (2,), np.array([[0., 1e-3], [1e-3, 0.]]), label="drive")
        chip = _hanger(effective_terms=(drive,))
    else:
        chip = _hanger(approximation=Exact())
    with pytest.raises(NotImplementedError, match="excitation-conserving"):
        eliminate(chip, "r1", local=local)


@pytest.mark.unit
def test_partition_and_restrict_keep_a_retained_pole_whole():
    """Partition keeps a retained pole with its survivor, and restrict refuses to split its ports."""
    qubit = DuffingTransmon(freq=5.3, anharmonicity=-.26, levels=2, label="q")
    readout = Resonator(freq=6.56, levels=2, label="r")
    spare = Resonator(freq=7.1, levels=2, label="s")
    network = PortNetwork(label="lines")
    network.expose("right", at=network.port("r_right", target=readout, rate=.006, phase=.2))
    network.expose("left", at=network.port("r_left", target=readout, rate=.004))
    network.expose("spare", at=network.port("s_port", target=spare, rate=.01))
    chip = Chip([qubit, readout, spare], [Capacitive(qubit, readout, g=0.)], port_network=network,
                approximation=RWA())
    reduced = eliminate(chip, "r", local=True).chip
    with pytest.raises(ValueError, match="needs all of its ports"):
        reduced.port_network.restrict(["r_right"])
    part = reduced.partition()
    assert len(part) == 2
    frequencies = np.array([6.558, 6.5599, 6.56, 6.5601, 6.562])
    v = np.sqrt([.006, .004]) * np.exp(1j*np.array([.2, 0.]))
    expected = (np.eye(2) - v[None, :, None] * v.conj()[None, None, :]
                / (.005 - 2j*np.pi*(frequencies - 6.56))[:, None, None]).conj()
    # With g = 0 the reduction is exact. One ulp of the pole's angular frequency
    # moves S by 4*eps*omega/kappa, about 4e-12 at resonance.
    np.testing.assert_allclose(VNA(part.components[part.owner_of("q")].chip).sweep(frequencies).matrix, expected,
                               atol=1e-11, rtol=0)


@pytest.mark.unit
def test_deserialized_pole_on_a_linear_chip_enters_the_vacuum_sweep():
    """A pole loaded onto a harmonic chip adds its response, and its ports must exist in the network."""
    readout = Resonator(freq=6.5, levels=2, label="r")
    network = PortNetwork(label="hanger")
    network.expose("right", at=network.port("r_right", target=readout, rate=.006))
    network.expose("left", at=network.port("r_left", target=readout, rate=.004, phase=.3))
    data = network.to_dict()
    data["mode_responses"] = [{"label": "m", "ports": ["r_right", "missing"], "freq": 6.502, "weight": .8,
                               "internal_rate": .001}]
    with pytest.raises(ValueError, match="distinct ports"):
        PortNetwork.from_dict(data)
    data["mode_responses"][0]["ports"] = ["r_right", "r_left"]
    chip = Chip([readout], port_network=PortNetwork.from_dict(data))
    frequencies = np.array([6.499, 6.5, 6.501, 6.502, 6.503])
    # The pole is a second linear mode with amplitudes sqrt(weight) times the port amplitudes.
    port = np.sqrt([.006, .004]) * np.exp(1j*np.array([0., .3]))
    couplings = np.stack([port, np.sqrt(.8)*port], axis=1)
    damping = couplings.conj().T @ couplings/2 + np.diag([0., .001])/2
    expected = np.stack([
        np.eye(2) - couplings @ np.linalg.solve(damping + 2j*np.pi*np.diag([6.5 - f, 6.502 - f]), couplings.conj().T)
        for f in frequencies]).conj()
    # One ulp of a pole's angular frequency moves S by 4*eps*omega/kappa, about 4e-12 at resonance.
    np.testing.assert_allclose(VNA(chip).sweep(frequencies).matrix, expected, atol=1e-11, rtol=0)
