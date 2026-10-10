"""Local device elimination reads only a patch and agrees with the full-chip reduction."""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quchip import (
    Bath,
    Capacitive,
    ChargeDrive,
    Chip,
    ControlEquipment,
    CrossKerr,
    DuffingTransmon,
    Gaussian,
    ParametricDrive,
    PortNetwork,
    QuantumSequence,
    Resonator,
    RWA,
    Square,
    TunableCapacitive,
    eliminate,
)
from quchip.chip.effective import OperatorProjection


def _ring(n, *, noise=True, backend=None, cross_kerr=False, ports=False, port_on_q0=False):
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
    if cross_kerr:
        couplings.append(CrossKerr(qs[0], rs[1], chi=0.01, label="zz"))
    network = None
    if ports or port_on_q0:
        network = PortNetwork(label="lines")
        readouts = [network.port(f"p{i}", target=rs[i], rate=0.002 + 0.001 * i) for i in range(n)]
        if port_on_q0:
            # A line on q0 feeds the readout port of r1, so one field subgraph spans both.
            line = network.port("line", target=qs[0], rate=0.001)
            network.cascade(line, readouts[1])
            network.expose("feed", input=line.input, output=readouts[1].output)
    return Chip(qs + cs + rs, couplings, frame=5.2, approximation=RWA(), port_network=network,
                **({"backend": backend} if backend else {}))


def _model(chip):
    """Return the lab-frame Hamiltonian and the summed rate-weighted L†L."""
    resolved = chip.resolve(frame="lab")
    hamiltonian = np.asarray(resolved.hamiltonian().matrix())
    loss = np.zeros_like(hamiltonian)
    for term in resolved.collapse_terms:
        jump = np.asarray(term.operator.to_dense())
        loss = loss + complex(term.rate) * jump.conj().T @ jump
    return hamiltonian, loss


def _assert_same_model(expected, actual, atol=1e-12):
    assert [d.label for d in actual.devices] == [d.label for d in expected.devices]
    for want, got in zip(_model(expected), _model(actual)):
        np.testing.assert_allclose(got, want, atol=atol)


def _assert_same_reduction(full, local):
    _assert_same_model(full.chip, local.chip)
    assert local.mapping.source_labels == full.mapping.source_labels
    assert local.mapping.target_labels == full.mapping.target_labels
    np.testing.assert_allclose(local.mapping.embedding, full.mapping.embedding, atol=1e-12)
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
    np.testing.assert_allclose(matrix, lift.conj().T @ source @ lift, atol=1e-12)
    restored = OperatorProjection.from_dict(child.to_dict())
    np.testing.assert_allclose(restored.transport(operator, ("a", "s"), (3, 2))[0], matrix, atol=1e-12)
    with pytest.raises(ValueError, match="Pass dims"):
        child.apply(operator, ("a", "s"))


@pytest.mark.unit
def test_projection_parents_must_map_disjoint_source_devices():
    rng = np.random.default_rng(3)
    parent = OperatorProjection(("a", "m"), (3, 2), ("a",), (3,), _isometry(rng, 6, 3))
    with pytest.raises(ValueError, match="disjoint"):
        OperatorProjection(("a", "b"), (3, 2), ("a",), (3,), _isometry(rng, 6, 3), parents=(parent, parent))
    with pytest.raises(ValueError, match="matching dimensions"):
        OperatorProjection(("a", "b"), (2, 2), ("a",), (2,), _isometry(rng, 4, 2), parents=(parent,))
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


def test_local_patch_keeps_the_far_device_of_a_cross_kerr_edge():
    """The cross-Kerr edge shifts the mode's neighbour by the level of r1, so r1 joins the patch."""
    chip = _ring(2, cross_kerr=True)
    local = eliminate(chip, "r0", local=True)
    _assert_same_reduction(eliminate(chip, "r0"), local)
    assert local.chip.effective_terms[-1].labels == ("q0", "r1")


def test_traced_coupling_strength_keeps_the_concrete_patch():
    """A traced strength leaves the capacitive diagonal zero, so c0 stays outside the patch of r0."""
    base = _ring(2, noise=False, backend="dynamiqs")
    patches = []

    def level(g):
        reduced = eliminate(base.with_params({"q0_c0.g": g}), "r0", local=True).chip
        patches.append(reduced.effective_terms[-1].labels)
        return reduced.freq("q0")

    level(0.03)
    jax.make_jaxpr(level)(0.03)
    assert patches == [("q0",), ("q0",)]


def test_local_reduction_transforms_independent_readout_ports():
    chip = _ring(2, ports=True)
    full, local = eliminate(chip, "r0"), eliminate(chip, "r0", local=True)
    _assert_same_reduction(full, local)
    expected, actual = full.chip.resolve().slh, local.chip.resolve().slh
    np.testing.assert_allclose(np.asarray(actual.S), np.asarray(expected.S), atol=1e-12)
    assert [c.key for c in actual.external_channels] == [c.key for c in expected.external_channels]
    for want, got in zip(expected.external_channels, actual.external_channels):
        np.testing.assert_allclose(np.asarray(got.coupling.to_dense()), np.asarray(want.coupling.to_dense()),
                                   atol=1e-12)
    assert ([c.label for c in local.chip.port_network.components]
            == [c.label for c in full.chip.port_network.components])


def test_local_and_full_reductions_chain_in_either_order():
    chip = _ring(2)
    reference = eliminate(eliminate(chip, "r0").chip, "c0")
    _assert_same_model(reference.chip, eliminate(eliminate(chip, "r0", local=True).chip, "c0").chip)
    _assert_same_model(reference.chip, eliminate(eliminate(chip, "r0").chip, "c0", local=True).chip)
    chained = eliminate(eliminate(chip, "r0", local=True).chip, "c0", local=True).chip
    _assert_same_model(reference.chip, chained)
    # The second map keeps the first as its parent, and both survive serialization.
    restored = Chip.from_dict(chained.to_dict())
    assert len(restored.effective_terms[-1].projection.parents) == 1
    _assert_same_model(chained, restored, atol=0.0)


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
    shared = _ring(2, port_on_q0=True)
    with pytest.raises(NotImplementedError, match="port network"):
        eliminate(shared, "r0", local=True)
    assert [d.label for d in eliminate(shared, "r0").chip.devices] == ["q0", "q1", "c0", "c1", "r1"]


def test_active_patch_forwards_local():
    chip = _ring(2)
    chip.connect(ControlEquipment([ChargeDrive(chip["q0"], label="d0")]))
    sequence = QuantumSequence(chip)
    sequence.schedule("d0", envelope=Gaussian(duration=40.0, amplitude=0.02), freq=5.32)
    full, local = sequence.active_patch(hops=1), sequence.active_patch(hops=1, local=True)
    assert local.eliminated_labels == full.eliminated_labels == ("r1", "q1")
    _assert_same_model(full.chip, local.chip)
    np.testing.assert_allclose(local.mapping.embedding, full.mapping.embedding, atol=1e-12)


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
    np.testing.assert_allclose(_final_state(local), _final_state(full), atol=1e-8)


@pytest.mark.validation
def test_local_reduction_equals_the_full_reduction_on_a_three_transmon_ring():
    chip = _ring(3)
    _assert_same_reduction(eliminate(chip, "c0"), eliminate(chip, "c0", local=True))


@pytest.mark.validation
def test_pump_on_an_edge_that_leaves_the_patch_matches_the_full_reduction():
    """The q0-c1 pump spans the patch boundary, so its operator joins the retained coordinates of q0."""
    qs = [DuffingTransmon(freq=5.32 - 0.07 * i, anharmonicity=-0.26, levels=3, label=f"q{i}", T1=40e3)
          for i in range(2)]
    cs = [Resonator(freq=6.30 + 0.02 * i, levels=2, label=f"c{i}", T1=20e3) for i in range(2)]
    rs = [Resonator(freq=6.56 + 0.05 * i, levels=2, label=f"r{i}", T1=2e3) for i in range(2)]
    couplings = [Capacitive(qs[0], cs[0], g=0.03, label="q0_c0"), Capacitive(qs[1], cs[0], g=0.03, label="q1_c0"),
                 Capacitive(qs[1], cs[1], g=0.03, label="q1_c1"),
                 TunableCapacitive(qs[0], cs[1], g_0=0.03, label="q0_c1"),
                 Capacitive(qs[0], rs[0], g=0.04, label="q0_r0"), Capacitive(qs[1], rs[1], g=0.04, label="q1_r1")]
    chip = Chip(qs + cs + rs, couplings, frame=5.2, approximation=RWA())
    chip.connect(ControlEquipment([ChargeDrive(chip["q0"], label="d0"),
                                   ParametricDrive(chip.coupling("q0_c1"), label="pump")]))
    local = eliminate(chip, "r0", local=True).chip
    assert local.effective_terms[-1].labels == ("q0",)
    np.testing.assert_allclose(_final_state(local, pump=True), _final_state(eliminate(chip, "r0").chip, pump=True),
                               atol=1e-8)


@pytest.mark.validation
def test_local_reduction_gradient_matches_the_full_reduction():
    """q0_r0 couples inside the first patch, and q0_c0 crosses its boundary."""
    base = _ring(2, noise=False, backend="dynamiqs")

    def level(g, local):
        chip = base.with_params({"q0_r0.g": g[0], "q0_c0.g": g[1]})
        for target in ("r0", "c0"):
            chip = eliminate(chip, target, local=local).chip
        return chip.freq("q0")

    g = jnp.array([0.04, 0.03])
    expected = jax.jit(jax.grad(lambda g: level(g, False)))(g)
    actual = jax.jit(jax.grad(lambda g: level(g, True)))(g)
    assert np.all(np.abs(expected) > 1e-2)
    np.testing.assert_allclose(actual, expected, rtol=1e-9)
