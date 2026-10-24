"""Local device elimination reads only a patch and agrees with the full-chip reduction."""

from math import prod

import numpy as np
import pytest

from quchip import (
    Bath,
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    DuffingTransmon,
    Gaussian,
    PortNetwork,
    QuantumSequence,
    Resonator,
    RWA,
    Square,
    VNA,
    eliminate,
)
from quchip.chip.effective import OperatorProjection
from quchip.declarative.expr import as_operator_expr
from quchip.declarative.models import CouplingModel
from quchip.declarative.parameters import Scalar, parameter


def _ring(n, *, noise=True, backend=None, ports=False):
    """Transmons in a ring with a coupling resonator between neighbours and one readout each."""
    qs = [DuffingTransmon(freq=5.32 - 0.07 * i, anharmonicity=-0.26, levels=3, label=f"q{i}",
                          T1=40e3 + 1e3 * i if noise else None) for i in range(n)]
    cs = [Resonator(freq=6.30 + 0.02 * i, levels=2, label=f"c{i}", T1=20e3 if noise else None) for i in range(n)]
    rs = [Resonator(freq=6.56 + 0.05 * i, levels=2, label=f"r{i}", T1=1e3 + 10 * i if noise else None)
          for i in range(n)]
    couplings = []
    for i in range(n):
        j = (i + 1) % n
        couplings += [Capacitive(qs[i], cs[i], g=0.03, label=f"q{i}_c{i}"),
                      Capacitive(qs[j], cs[i], g=0.03, label=f"q{j}_c{i}"),
                      Capacitive(qs[i], rs[i], g=0.04, label=f"q{i}_r{i}")]
    network = None
    if ports:
        network = PortNetwork(label="lines")
        for i in range(n):
            network.port(f"p{i}", target=rs[i], rate=0.002 + 0.001 * i)
    return Chip(qs + cs + rs, couplings, frame=5.2, approximation=RWA(), port_network=network,
                **({"backend": backend} if backend else {}))


def _model(chip):
    """Return the lab-frame Hamiltonian, the summed rate-weighted L†L and the jump term on |psi><phi|.

    psi and phi are random states.
    """
    resolved = chip.resolve(frame="lab")
    hamiltonian = np.asarray(resolved.hamiltonian().matrix())
    rng = np.random.default_rng(3)
    psi, phi = rng.normal(size=(2, len(hamiltonian))) + 1j * rng.normal(size=(2, len(hamiltonian)))
    loss = np.zeros_like(hamiltonian)
    jumps = np.zeros_like(hamiltonian)
    for term in resolved.collapse_terms:
        jump = np.asarray(term.operator.to_dense())
        loss = loss + complex(term.rate) * jump.conj().T @ jump
        jumps = jumps + complex(term.rate) * np.outer(jump @ psi, (jump @ phi).conj())
    return hamiltonian, loss, jumps


def _wire_readouts(chip, directions):
    """Put the readouts on one or two counter-propagating lines with a reference delay."""
    network = PortNetwork(label="feedline")
    readouts = [device for device in chip.devices if device.label.startswith("r")]
    for direction in range(directions):
        ordered = readouts if direction == 0 else readouts[::-1]
        ports = [network.port(f"p{direction}_{readout.label}", target=readout,
                              rate=0.012 / directions, phase=0.2 * index)
                 for index, readout in enumerate(ordered)]
        cable = network.phase_shift(f"cable{direction}", phase=0.17)
        delay = network.delay(f"delay{direction}", duration=0.125)
        network.cascade(*ports, cable, delay.input_terminal("1"))
        network.expose(f"feed{direction}", input=ports[0].input, output=delay.output_terminal("2"))
    return Chip(chip.devices, chip.couplings, port_network=network, frame=chip.frame,
                approximation=chip.approximation)


@pytest.mark.parametrize("directions", [1, 2])
def test_shared_feedline_does_not_expand_the_generator_patch(directions):
    """Line-only spectators cannot enter a generator computed without network Hamiltonians."""
    reduced = eliminate(_wire_readouts(_ring(8, noise=False), directions), "q0", local=True).chip
    retained = next(terms for terms in reduced.effective_terms if terms.label == "retained_q0")
    # With independent ports, q0's patch holds its coupling neighbours only.
    assert retained.projection.source_labels == ("q0", "c0", "c7", "r0")


def test_local_boundary_section_on_a_large_rwa_ring_keeps_the_pair_transition():
    """A reference section on a 24-mode ring uses its local RWA pole and weight without a full-space matrix."""
    chip = _wire_readouts(_ring(8, noise=False), 1)
    reduced = eliminate(chip, "r0", local=True).chip
    detuning, g = 6.56 - 5.32, 0.04
    prefix = "network.component.r0_transmission."
    # The section evaluates the same two-state SW formulas, so only round-off differs.
    # 1e-13 GHz is about 100 double-precision spacings at 6.56 GHz.
    np.testing.assert_allclose(reduced.parameters[prefix + "freq"], 6.56 + g**2 / detuning, atol=1e-13)
    np.testing.assert_allclose(reduced.parameters[prefix + "external_rate"],
                               0.012 * np.cos(g / detuning)**2, rtol=1e-12)
    projection = next(terms.projection for terms in reduced.effective_terms if terms.projection is not None)
    assert projection.source_labels == ("q0", "r0")


def test_shared_feedline_local_reduction_matches_full_dynamics():
    """Local maps retain the full Hamiltonian, dissipator and complex scattering on a two-way shared line."""
    sites, directions = 2, 2
    independent = _ring(sites, ports=True)
    source = full = local = _wire_readouts(independent, directions)
    topology = source.port_network.to_dict()
    order = ([f"q{i}" for i in range(0, sites, 2)] + [f"c{i}" for i in range(sites)]
             + [f"q{i}" for i in range(1, sites, 2)])
    for target in order:
        independent = eliminate(independent, target, local=True).chip
        full = eliminate(full, target).chip
        local = eliminate(local, target, local=True).chip
        reference = next(terms for terms in independent.effective_terms if terms.label == f"retained_{target}")
        actual = next(terms for terms in local.effective_terms if terms.label == f"retained_{target}")
        assert actual.projection.source_labels == reference.projection.source_labels
        # Both routes use the same second-order SW generator, so only product-space
        # round-off differs. Each step also compares recycling on an off-diagonal state.
        _assert_same_model(full, local)
    np.testing.assert_allclose(_jumps(local), _jumps(full), rtol=0, atol=1e-12)
    assert local.port_network.to_dict() == topology
    frequencies = np.concatenate([float(full.freq(f"r{i}")) + np.linspace(-0.005, 0.005, 11)
                                  for i in range(sites)])
    # A 1e-12 GHz rounding error changes a resonant field by about 8π δf/κ,
    # or 2e-9 for the readouts' 0.012/ns external linewidth.
    np.testing.assert_allclose(VNA(local).sweep(frequencies).matrix, VNA(full).sweep(frequencies).matrix,
                               rtol=0, atol=2e-9)


def _assert_same_model(expected, actual, atol=1e-12):
    assert [d.label for d in actual.devices] == [d.label for d in expected.devices]
    for want, got in zip(_model(expected), _model(actual)):
        np.testing.assert_allclose(got, want, rtol=0, atol=atol)


def _assert_same_reduction(full, local):
    _assert_same_model(full.chip, local.chip)
    assert local.mapping.source_labels == full.mapping.source_labels
    assert local.mapping.target_labels == full.mapping.target_labels
    np.testing.assert_allclose(local.mapping.embedding, full.mapping.embedding, rtol=0, atol=1e-12)
    for label, params in full.effective_params.items():
        if label != "exchange":
            assert float(local.effective_params[label]["lamb_shift"]) == pytest.approx(
                float(params["lamb_shift"]), abs=1e-12)


def _on(operator, labels, order, dims):
    """Embed an operator on ``labels`` into the product space ``order`` with identities elsewhere."""
    rest = [label for label in order if label not in labels]
    full = np.kron(operator, np.eye(prod(dims[label] for label in rest)))
    current = list(labels) + rest
    shape = [dims[label] for label in current]
    axes = [current.index(label) for label in order]
    full = full.reshape(shape + shape).transpose(axes + [len(order) + axis for axis in axes])
    size = prod(dims[label] for label in order)
    return full.reshape(size, size)


def _isometry(rng, rows, columns):
    matrix = rng.normal(size=(rows, columns)) + 1j * rng.normal(size=(rows, columns))
    return np.linalg.qr(matrix)[0]


@pytest.mark.unit
def test_chained_projection_applies_its_parent_first_and_keeps_spectators():
    rng = np.random.default_rng(7)
    dims = {"a": 3, "m": 2, "b": 2, "n": 2, "s": 2}
    parent = OperatorProjection(("a", "m"), (3, 2), ("a",), (3,), _isometry(rng, 6, 3))
    child = OperatorProjection(("a", "b", "n"), (3, 2, 2), ("a", "b"), (3, 2), _isometry(rng, 12, 6),
                               parents=(parent,))
    operator = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))

    matrix, labels, out_dims = child.transport(operator, ("a", "s"), (3, 2))

    # The chain lifts (a, b, s) through the child and then the parent.
    lift = np.kron(parent.embedding, np.eye(8)) @ np.kron(child.embedding, np.eye(2))
    source = _on(operator, ("a", "s"), ("a", "m", "b", "n", "s"), dims)
    assert labels == ("a", "b", "s") and out_dims == (3, 2, 2)
    np.testing.assert_allclose(matrix, lift.conj().T @ source @ lift, rtol=0, atol=1e-12)
    restored = OperatorProjection.from_dict(child.to_dict())
    np.testing.assert_allclose(restored.transport(operator, ("a", "s"), (3, 2))[0], matrix, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="Pass dims"):
        child.apply(operator, ("a", "s"))


@pytest.mark.unit
def test_projection_lineage_applies_overlapping_maps_in_order():
    """The second map reads m from the first map's targets, and a, which it does not read, is a spectator."""
    rng = np.random.default_rng(3)
    dims = {"a": 3, "m": 2, "b": 2}
    first = OperatorProjection(("a", "m"), (3, 2), ("a", "m"), (3, 2), _isometry(rng, 6, 6))
    second = OperatorProjection(("m", "b"), (2, 2), ("m",), (2,), _isometry(rng, 4, 2), parents=(first,))
    operator = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    source = _on(operator, ("a",), ("a", "m", "b"), dims)

    def expected(lift):
        # The lift maps (a, m) after both maps into (a, m, b). The result lists the targets of the second map first.
        matrix = (lift.conj().T @ source @ lift).reshape(3, 2, 3, 2)
        return matrix.transpose(1, 0, 3, 2).reshape(6, 6)

    matrix, labels, out_dims = second.transport(operator, ("a",), (3,))
    assert labels == ("m", "a") and out_dims == (2, 3)
    second_lift = np.kron(np.eye(3), second.embedding)
    np.testing.assert_allclose(matrix, expected(np.kron(first.embedding, np.eye(2)) @ second_lift), atol=1e-12)
    # An owner that the first map already holds in retained coordinates passes through the second map only.
    marked = OperatorProjection(("m", "b"), (2, 2), ("m",), (2,), second.embedding,
                                parents=(first.with_current_operators(("port:p",)),))
    np.testing.assert_allclose(marked.transport(operator, ("a",), (3,), "port:p")[0], expected(second_lift),
                               atol=1e-12)
    with pytest.raises(TypeError, match="parents"):
        OperatorProjection(("a", "b"), (3, 2), ("a",), (3,), _isometry(rng, 6, 3), parents=("a",))


@pytest.mark.parametrize("target", ["r0", "c0", "q1"])
def test_local_reduction_equals_the_full_reduction(target):
    chip = _ring(2)
    local = eliminate(chip, target, local=True)
    _assert_same_reduction(eliminate(chip, target), local)
    # Only the survivors in the patch around the target carry the new retained terms.
    patch = {"r0": ("q0",), "c0": ("q0", "q1"), "q1": ("c0", "c1", "r1")}[target]
    assert local.chip.effective_terms[-1].labels == patch


def test_local_steps_reduce_a_ring_whose_full_space_does_not_fit_in_memory():
    chip = _ring(8, noise=False)
    assert chip.total_dim == 429_981_696
    order = [f"r{i}" for i in range(2, 8)] + ["c1"] + [f"{kind}{i}" for i in range(2, 8) for kind in "qc"]
    for target in order:
        chip = eliminate(chip, target, local=True).chip
    assert len(order) == 19
    assert [device.label for device in chip.devices] == ["q0", "q1", "c0", "r0", "r1"]
    assert chip.total_dim == 72


class _LevelShift(CouplingModel):
    """A rescaled diagonal edge tests patch selection at tiny operator values and zero coupling."""

    x: Scalar = parameter(unit="GHz")
    y: Scalar = parameter(unit="GHz")

    def interaction(self, a, b, p):
        dims = (a.space.dimension, b.space.dimension)
        numbers = np.kron(np.diag(np.arange(dims[0])), np.diag(np.arange(dims[1])))
        operator = as_operator_expr(1e-13 * numbers, labels=(a.label, b.label), dims=dims, name="n_a n_b")
        return 1e13 * (p.x - p.y) * operator


def _jumps(chip):
    """Return the jump part of the dissipator, the sum of rate * L ⊗ conj(L)."""
    jumps = 0.0
    for term in chip.resolve(frame="lab").collapse_terms:
        jump = np.asarray(term.operator.to_dense())
        jumps = jumps + complex(term.rate) * np.kron(jump, jump.conj())
    return jumps


@pytest.mark.parametrize("order", [("c0", "c1", "q0", "q1"), ("q0", "q1", "r0")])
def test_local_steps_equal_the_full_reduction_in_sequence(order):
    """Later patches leave part of earlier loss and maps outside. After q0 and q1, the port of r0 widens r0's patch."""
    full = local = _ring(2, ports=True)
    for target in order:
        full, local = eliminate(full, target).chip, eliminate(local, target, local=True).chip
        _assert_same_model(full, local)
    np.testing.assert_allclose(_jumps(local), _jumps(full), rtol=0, atol=1e-12)
    expected, actual = full.resolve().slh, local.resolve().slh
    assert [c.key for c in actual.external_channels] == [c.key for c in expected.external_channels]
    for want, got in zip(expected.external_channels, actual.external_channels):
        np.testing.assert_allclose(np.asarray(got.coupling.to_dense()), np.asarray(want.coupling.to_dense()),
                                   atol=1e-12)


@pytest.mark.parametrize("x", [0.05, 0.03], ids=["nonzero", "cancelled"])
def test_local_patch_holds_each_term_that_shifts_a_neighbour(x):
    """The diagonal a-d edge enters the generator, and the network regenerates the line's a-b-c term."""
    m = Resonator(freq=7.0, levels=2, label="m")
    a = DuffingTransmon(freq=6.0, anharmonicity=-0.25, levels=3, label="a")
    b, c, d, e = (Resonator(freq=freq, levels=2, label=label) for freq, label in zip((5.0, 5.5, 4.5, 4.0), "bcde"))
    network = PortNetwork(label="line")
    joint = network.port("ab", target=(a, b), rate=0.01, operator=np.kron(np.diag([0., 1., 2.]), np.diag([0., 1.])))
    single = network.port("c", target=c, rate=0.01, operator=np.diag([0., 1.]))
    # With the π/2 shift, the series Hamiltonian is proportional to n_a n_b n_c.
    network.cascade(joint, network.phase_shift("cable", phase=np.pi / 2), single)
    network.expose("feed", input=joint.input, output=single.output)
    couplings = [Capacitive(m, a, g=0.03, label="m_a"), _LevelShift(a, d, x=x, y=0.03, label="a_d"),
                 Capacitive(d, e, g=0.02, label="d_e")]
    chip = Chip([m, a, b, c, d, e], couplings, port_network=network)
    local = eliminate(chip, "m", local=True)
    _assert_same_reduction(eliminate(chip, "m"), local)
    assert local.chip.effective_terms[-1].labels == ("a", "d")


@pytest.mark.parametrize("order", ["abumr", "uabmr"])
def test_device_without_a_neighbour_reduces_alone_in_any_device_order(order):
    """m has no neighbour. Its patch holds neither the port of a, cascaded into b, nor the retained terms of u."""
    devices = {label: Resonator(freq=freq, levels=2, label=label, T1=1e3)
               for label, freq in zip("abumr", (5.0, 5.5, 6.0, 7.0, 6.5))}
    network = PortNetwork(label="line")
    first = network.port("a", target=devices["a"], rate=0.01)
    second = network.port("b", target=devices["b"], rate=0.01)
    network.cascade(first, second)
    network.expose("feed", input=first.input, output=second.output)
    chip = Chip([devices[label] for label in order], [Capacitive(devices["u"], devices["r"], g=0.05, label="u_r")],
                port_network=network)
    chip = eliminate(chip, "r", local=True).chip
    _assert_same_reduction(eliminate(chip, "m"), eliminate(chip, "m", local=True))


def test_local_reduction_transforms_independent_readout_ports():
    chip = _ring(2, ports=True)
    full, local = eliminate(chip, "r0"), eliminate(chip, "r0", local=True)
    _assert_same_reduction(full, local)
    expected, actual = full.chip.resolve().slh, local.chip.resolve().slh
    np.testing.assert_allclose(np.asarray(actual.S), np.asarray(expected.S), rtol=0, atol=1e-12)
    assert [c.key for c in actual.external_channels] == [c.key for c in expected.external_channels]
    for want, got in zip(expected.external_channels, actual.external_channels):
        np.testing.assert_allclose(np.asarray(got.coupling.to_dense()), np.asarray(want.coupling.to_dense()),
                                   rtol=0, atol=1e-12)
    assert ([c.label for c in local.chip.port_network.components]
            == [c.label for c in full.chip.port_network.components])


def test_partition_keeps_each_map_with_the_devices_it_acts_on():
    """n dresses c. The map of m mixes a and b, although their mediated exchange cancels."""
    a, m, b, n = (Resonator(freq=freq, levels=2, label=label) for freq, label in zip((5.0, 6.0, 7.0, 6.5), "ambn"))
    c = Resonator(freq=5.5, levels=2, label="c", T1=1e3)
    chip = Chip([a, m, b, c, n], [Capacitive(a, m, g=0.05, label="a_m"), Capacitive(b, m, g=0.05, label="b_m"),
                                  Capacitive(c, n, g=0.05, label="c_n")])
    for target in ("n", "m"):
        chip = eliminate(chip, target, local=True).chip
    parts = chip.partition().components
    assert [part.labels for part in parts] == [("a", "b"), ("c",)]
    (h_ab, loss_ab, _), (h_c, loss_c, _) = (_model(part.chip) for part in parts)
    hamiltonian, loss, _ = _model(chip)
    np.testing.assert_allclose(np.kron(h_ab, np.eye(2)) + np.kron(np.eye(4), h_c), hamiltonian, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.kron(loss_ab, np.eye(2)) + np.kron(np.eye(4), loss_c), loss, rtol=0, atol=1e-12)


def test_local_reduction_raises_outside_its_scope():
    chip = _ring(2)
    with pytest.raises(NotImplementedError, match="method='sw' only"):
        eliminate(chip, "r0", method="exact", local=True)
    with pytest.raises(NotImplementedError, match="coupling targets"):
        eliminate(chip, "q0_r0", local=True)
    bathed = _ring(2, noise=False)
    bathed.add_bath(Bath("collective_decay", targets=["q0", "q1"], rate=1e-4))
    with pytest.raises(NotImplementedError, match="baths"):
        eliminate(bathed, "r0", local=True)


def _final_state(chip, pump=False):
    sequence = QuantumSequence(chip)
    sequence.schedule("d0", envelope=Gaussian(duration=40.0, amplitude=0.02), freq=float(chip.freq("q0")))
    if pump:
        sequence.pump("q0_c1", envelope=Square(duration=40.0, amplitude=0.004), freq=0.98)
    result = sequence.simulate(tlist=np.linspace(0.0, 40.0, 41), dissipation=pump)
    return np.asarray(chip.backend.to_array(result.final_state))


def test_drive_on_a_dressed_survivor_matches_the_full_reduction():
    chip = _ring(2, noise=False)
    chip.connect(ControlEquipment([ChargeDrive(chip["q0"], label="d0")]))
    full = local = chip
    for target in ("r0", "c0", "r1"):
        full = eliminate(full, target).chip
        local = eliminate(local, target, local=True).chip
    np.testing.assert_allclose(_final_state(local), _final_state(full), rtol=0, atol=1e-8)


def test_local_reduction_keeps_small_resolved_lamb_shifts():
    """A small but resolved shift must not disappear under an energy-scale cutoff."""
    q = Resonator(freq=5.0, levels=2, label="q")
    r = Resonator(freq=6.0, levels=2, label="r")
    g = 1e-6
    chip = Chip([q, r], [Capacitive(q, r, g=g)])
    expected = g**2 / (q.freq - r.freq)
    for local in (False, True):
        frequency = float(eliminate(chip, "r", local=local).chip.freq("q"))
        np.testing.assert_allclose(frequency - q.freq, expected, rtol=0, atol=np.spacing(q.freq))
