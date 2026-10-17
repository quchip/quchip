"""The default order of a chain of eliminations keeps transported operators small and its folds accurate."""

import re
import warnings
from math import prod

import jax
import numpy as np
import pytest

from quchip import (
    Capacitive,
    ChargeDrive,
    Chip,
    CrossKerr,
    DuffingTransmon,
    QuantumSequence,
    Resonator,
    RWA,
    Square,
    eliminate,
    eliminate_spectators,
)
from quchip.chip.effective import retained_support, transport_retained_operator
from quchip.chip.transformations.elimination_order import EliminationOrder


def _ring(n, keep=(), g=0.03, fc=6.30):
    """n transmons in a ring, a coupling resonator between neighbours, a drive line on each kept transmon."""
    qs = [DuffingTransmon(freq=5.32 - 0.07 * i, anharmonicity=-0.26, levels=4, label=f"q{i}") for i in range(n)]
    cs = [Resonator(freq=fc + 0.02 * i, levels=3, label=f"c{i}") for i in range(n)]
    couplings = []
    for i in range(n):
        j = (i + 1) % n
        couplings += [Capacitive(qs[i], cs[i], g=g, label=f"q{i}_c{i}"),
                      Capacitive(qs[j], cs[i], g=g, label=f"q{j}_c{i}")]
    chip = Chip(qs + cs, couplings, frame=5.2, approximation=RWA())
    if keep:
        chip.wire(*[ChargeDrive(target=qs[k], label=f"d{k}") for k in keep])
    return chip


def _walked(n, keep):
    """Each arc between kept transmons from one end, its end couplers included; the couplers between neighbours last."""
    seq = [f"{kind}{i}" for i in range(n) for kind in "qc"]
    arcs = [[seq[(2 * k + d) % (2 * n)] for d in range(1, 2 * ((keep[(m + 1) % len(keep)] - k) % n))]
            for m, k in enumerate(keep)]
    return [x for arc in arcs if len(arc) > 1 for x in arc] + [arc[0] for arc in arcs if len(arc) == 1]


def _widest(steps, labels):
    """The largest product dimension of the support of an operator on ``labels`` after any step."""
    most = 0
    for step in steps:
        dims = dict(zip((d.label for d in step.chip.devices), step.chip.authored_dims))
        most = max(most, *(prod(dims[x] for x in retained_support(step.chip, (label,))) for label in labels))
    return most


def _largest_fold(result):
    return max(abs(float(entry["g_over_delta"])) for step in result.steps for entry in step.validity.values()
               if "g_over_delta" in entry)


def _kuv(f_v=5.1, scale=1.0, **options):
    """A kept transmon k, a far-detuned mediator u and a spectator v near k."""
    k = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="k")
    u = Resonator(freq=7.0, levels=2, label="u")
    v = DuffingTransmon(freq=f_v, anharmonicity=-0.25, levels=3, label="v")
    return Chip([k, u, v], [Capacitive(k, u, g=0.05 * scale, label="ku"), Capacitive(u, v, g=0.05 * scale, label="uv")],
                approximation=RWA(), **options)


@pytest.mark.parametrize("n, keep", [(6, (1, 2, 4, 5)), (8, (2, 3, 5, 6))])
def test_default_order_keeps_every_kept_operator_of_a_ring_within_the_walked_order(n, keep):
    """Removing every coupler first spreads a drive over all transmons; both entry points stay within 256 states."""
    chip = _ring(n, keep)
    labels = [f"q{k}" for k in keep]
    result = eliminate_spectators(chip, labels, local=True)
    sequence = QuantumSequence(chip)
    for k in keep:
        sequence.schedule(f"d{k}", envelope=Square(duration=20.0, amplitude=0.01), freq=float(chip.freq(f"q{k}")))
    patch = sequence.active_patch(hops=0, local=True)

    assert patch.eliminated_labels == result.eliminated_labels
    assert [d.label for d in result.chip.devices] == labels
    assert result.kept_labels == tuple(labels)
    assert _widest(result.steps, labels) <= 256
    assert _widest(eliminate_spectators(chip, labels, order=_walked(n, keep), local=True).steps, labels) == 256
    assert not any(note.startswith("Eliminated ") for note in result.notes)


def test_a_local_step_reports_its_widest_operator_and_warns_when_the_dense_matrix_is_large():
    """With the couplers removed first, an operator on q0 spreads over the whole ring of six transmons."""
    chip = _ring(6)
    with pytest.warns(UserWarning, match=r"'q0' now acts on 6 devices .*\(4,096 states, 256 MiB as a dense matrix\)"):
        for target in ("c0", "c1", "c2", "c3", "c4"):
            step = eliminate(chip, target, local=True)
            chip = step.chip
    reports = [note for note in step.notes if note.endswith("the widest of any device.")]
    assert reports == ["An operator on 'q0' acts on ['q0', 'q1', 'q2', 'q3', 'q4', 'q5'] in retained coordinates "
                       "(4096 states), the widest of any device."]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        small = eliminate(_ring(6), "c0", local=True)
    assert any(re.fullmatch(r"An operator on 'q0' acts on \['q0', 'q1'\] .*\(16 states\), the widest of any device\.",
                            note) for note in small.notes)


def test_an_explicit_order_gives_the_chain_of_single_eliminations():
    chip = _ring(3)
    order = ["q2", "c0", "c1", "c2"]
    result = eliminate_spectators(chip, ["q0", "q1"], order=order, local=True)
    manual = chip
    for target in order:
        manual = eliminate(manual, target, local=True).chip
    assert result.eliminated_labels == tuple(order)
    np.testing.assert_allclose(np.asarray(result.chip.resolve(frame="lab").hamiltonian().matrix()),
                               np.asarray(manual.resolve(frame="lab").hamiltonian().matrix()), rtol=0, atol=1e-12)


def test_local_and_full_routes_take_the_same_order_and_transport_the_same_operators():
    chip = _ring(3)
    local = eliminate_spectators(chip, ["q0", "q1"], local=True)
    full = eliminate_spectators(chip, ["q0", "q1"])
    assert local.eliminated_labels == full.eliminated_labels
    np.testing.assert_allclose(np.asarray(local.chip.resolve(frame="lab").hamiltonian().matrix()),
                               np.asarray(full.chip.resolve(frame="lab").hamiltonian().matrix()), rtol=0, atol=1e-10)
    rng = np.random.default_rng(5)
    for label in ("q0", "q1"):
        operator = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        operators = []
        for reduced in (local.chip, full.chip):
            matrix, labels, dims = transport_retained_operator(reduced, operator, (label,), (4,))
            assert labels == ("q0", "q1")
            operators.append(np.asarray(matrix))
        np.testing.assert_allclose(operators[0], operators[1], rtol=0, atol=1e-10)


def test_a_spectator_waits_for_the_mediator_to_a_kept_device_near_it():
    """Removing u first makes the k-v exchange explicit, so the chain keeps the shift J²/Δ_kv of k."""
    chip = _kuv()
    result = eliminate_spectators(chip, ["k"], local=True)
    other = eliminate_spectators(chip, ["k"], order=["v", "u"], local=True)
    exchange = 0.05**2 / 2 * (1 / (5.0 - 7.0) + 1 / (5.1 - 7.0))
    shift = exchange**2 / (5.0 - 5.1)

    assert result.eliminated_labels == ("u", "v")
    # The rest is fourth order without the near-resonant enhancement, of order |Δ_kv/Δ_ku| times the shift.
    assert abs(float(result.chip.freq("k")) - float(chip.freq("k"))) < 0.2 * abs(shift)
    assert float(other.chip.freq("k")) - float(result.chip.freq("k")) == pytest.approx(-shift, rel=0.1)


@pytest.mark.parametrize("f_c2, g_left, g_right", [(7.22, 0.075, 0.075), (7.22, 0.19, 0.01), (7.20, 0.075, 0.075)])
def test_a_spectator_waits_while_its_step_would_couple_two_near_degenerate_spectators(f_c2, g_left, g_right):
    """Removing q first gives c1 and c2 an exchange that is large against their gap after the step.

    With legs of 0.19 and 0.01 GHz, the step also moves c1 18 MHz towards c2. With both couplers at
    7.20 GHz, the later fold of the pair is undefined.
    """
    k1 = DuffingTransmon(freq=4.0, anharmonicity=-0.25, levels=3, label="k1")
    c1 = Resonator(freq=7.20, levels=2, label="c1")
    q = DuffingTransmon(freq=5.2, anharmonicity=-0.25, levels=3, label="q")
    c2 = Resonator(freq=f_c2, levels=2, label="c2")
    k2 = DuffingTransmon(freq=6.4, anharmonicity=-0.25, levels=3, label="k2")
    chip = Chip([k1, c1, q, c2, k2], [Capacitive(k1, c1, g=0.075), Capacitive(c1, q, g=g_left),
                                      Capacitive(q, c2, g=g_right), Capacitive(c2, k2, g=0.04)],
                frame=5.2, approximation=RWA())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = eliminate_spectators(chip, ["k1", "k2"], local=True)
    assert result.eliminated_labels.index("q") > min(result.eliminated_labels.index("c1"),
                                                     result.eliminated_labels.index("c2"))
    assert _largest_fold(result) < 0.1
    if f_c2 == 7.20:
        with pytest.raises(ValueError, match="degenerate levels with nonzero coupling"):
            eliminate_spectators(chip, ["k1", "k2"], order=["q", "c1", "c2"], local=True)
        return
    # The second-order exchange and shifts that the rule compares predict the fold of the pair.
    exchange = g_left * g_right * (1 / (7.20 - 5.2) + 1 / (f_c2 - 5.2)) / 2
    gap = 7.20 + g_left**2 / (7.20 - 5.2) - f_c2 - g_right**2 / (f_c2 - 5.2)
    with pytest.warns(UserWarning, match="poor Schrieffer-Wolff validity"):
        other = eliminate_spectators(chip, ["k1", "k2"], order=["q", "c1", "c2"], local=True)
    assert _largest_fold(other) == pytest.approx(abs(exchange / gap), rel=1e-2)
    assert _largest_fold(other) > 0.1


def test_the_waits_read_the_exchange_that_the_approximation_keeps():
    """Without the exchange band, removing q gives c1 and c2 no exchange to screen, so q can go first."""
    k1 = DuffingTransmon(freq=4.0, anharmonicity=-0.25, levels=3, label="k1")
    c1 = Resonator(freq=7.20, levels=2, label="c1")
    q = DuffingTransmon(freq=5.2, anharmonicity=-0.25, levels=3, label="q")
    c2 = Resonator(freq=7.22, levels=2, label="c2")
    k2 = DuffingTransmon(freq=6.4, anharmonicity=-0.25, levels=3, label="k2")
    couplings = [Capacitive(k1, c1, g=0.075), Capacitive(c1, q, g=0.075), Capacitive(q, c2, g=0.075),
                 Capacitive(c2, k2, g=0.04)]
    orders = {}
    for name, approximation in (("rwa", RWA()), ("counter-rotating", RWA(keep_bands={(1, 1), (-1, -1)}))):
        chip = Chip([k1, c1, q, c2, k2], couplings, frame=5.2, approximation=approximation)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="RWA\\(\\) drops band")
            orders[name] = eliminate_spectators(chip, ["k1", "k2"], local=True).eliminated_labels
    assert orders == {"rwa": ("c1", "q", "c2"), "counter-rotating": ("q", "c1", "c2")}


@pytest.mark.parametrize("case", ["cross-Kerr chain", "six-transmon ring"])
def test_the_order_model_follows_the_supports_of_the_local_chain(case):
    """Removing v makes the shift of a depend on the level of b, so the later step of x reaches b."""
    if case == "six-transmon ring":
        chip, keep = _ring(6, (1, 2, 4, 5)), ("q1", "q2", "q4", "q5")
    else:
        k = DuffingTransmon(freq=5.1, anharmonicity=-0.25, levels=3, label="k")
        x = Resonator(freq=8.0, levels=2, label="x")
        a = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="a")
        v = Resonator(freq=7.0, levels=2, label="v")
        b = Resonator(freq=4.0, levels=2, label="b")
        chip = Chip([k, x, a, v, b], [Capacitive(k, x, g=0.02), Capacitive(x, a, g=0.02), Capacitive(a, v, g=0.05),
                                      CrossKerr(v, b, chi=0.01)], approximation=RWA())
        keep = ("k", "b")
    result = eliminate_spectators(chip, keep, local=True)
    model = EliminationOrder(chip, keep)
    for target, step in zip(result.eliminated_labels, result.steps):
        model.remove(target)
        assert model.supports() == {label: frozenset(retained_support(step.chip, (label,))) for label in keep}
    if case == "cross-Kerr chain":
        assert result.eliminated_labels == ("v", "x", "a")
        assert set(retained_support(result.steps[1].chip, ("k",))) == {"k", "a", "b"}


def test_a_step_that_breaks_a_wait_is_recorded():
    """Each spectator links the other to a kept device near it, so one of them must go first."""
    k1 = DuffingTransmon(freq=7.95, anharmonicity=-0.25, levels=3, label="k1")
    v1 = DuffingTransmon(freq=6.0, anharmonicity=-0.25, levels=3, label="v1")
    v2 = DuffingTransmon(freq=8.0, anharmonicity=-0.25, levels=3, label="v2")
    k2 = DuffingTransmon(freq=6.05, anharmonicity=-0.25, levels=3, label="k2")
    chip = Chip([k1, v1, v2, k2], [Capacitive(k1, v1, g=0.05), Capacitive(v1, v2, g=0.05),
                                   Capacitive(v2, k2, g=0.05)], approximation=RWA())
    result = eliminate_spectators(chip, ["k1", "k2"], local=True)
    assert result.eliminated_labels == ("v1", "v2")
    assert [note for note in result.notes if note.startswith("Eliminated ")] == [
        "Eliminated 'v1' while 'v2' still linked it to the kept 'k2', because every remaining spectator had to "
        "wait. The chain omits the fourth-order 'k2'-'v1' exchange shift through 'v2'."]


def _reduced_frequency(base, f_v, order=None):
    return eliminate_spectators(base.with_params({"v.freq": f_v}), ["k"], order=order, local=True).chip.freq("k")


@pytest.mark.optional_backend
def test_the_default_order_needs_concrete_values():
    pytest.importorskip("dynamiqs")
    with pytest.raises(ValueError, match=r"compares transition frequencies .* are traced\. Pass order="):
        jax.grad(lambda f_v: _reduced_frequency(_kuv(backend="dynamiqs"), f_v))(5.1)


@pytest.mark.optional_backend
@pytest.mark.validation
def test_an_explicit_order_differentiates_like_the_default_order():
    pytest.importorskip("dynamiqs")
    base = _kuv(backend="dynamiqs")
    order = eliminate_spectators(base, ["k"], local=True).eliminated_labels
    gradient = jax.grad(lambda f_v: _reduced_frequency(base, f_v, order))(5.1)
    step = 1e-4
    expected = (float(_reduced_frequency(base, 5.1 + step)) - float(_reduced_frequency(base, 5.1 - step))) / (2 * step)
    assert abs(expected) > 1e-4
    assert float(gradient) == pytest.approx(expected, rel=1e-5)


def test_eliminate_spectators_checks_keep_and_order():
    chip = _kuv()
    with pytest.raises(KeyError, match="keep names no device"):
        eliminate_spectators(chip, ["x"])
    with pytest.raises(ValueError, match="keep names no device"):
        eliminate_spectators(chip, [])
    with pytest.raises(ValueError, match="order names kept devices"):
        eliminate_spectators(chip, "k", order=["k", "u"])
    with pytest.raises(ValueError, match="more than once"):
        eliminate_spectators(chip, "k", order=["u", "u"])
    partial = eliminate_spectators(chip, chip["k"], order=[chip["u"]])
    assert [d.label for d in partial.chip.devices] == ["k", "v"]
    assert "spectators ['v'] are not in order; left in place" in partial.notes


@pytest.mark.validation
def test_the_order_gap_is_fourth_order_in_the_legs_and_follows_the_sign_of_the_detuning():
    """The two orders of the k-u-v chain differ by the k-v exchange shift, which scales as g⁴ and as 1/Δ_kv."""
    gaps = {}
    for f_v in (5.1, 4.9):
        for scale in (1.0, 0.5):
            chip = _kuv(f_v, scale)
            default = eliminate_spectators(chip, ["k"], local=True)
            assert default.eliminated_labels == ("u", "v")
            other = eliminate_spectators(chip, ["k"], order=["v", "u"], local=True)
            gaps[f_v, scale] = float(other.chip.freq("k")) - float(default.chip.freq("k"))
    for f_v in (5.1, 4.9):
        assert gaps[f_v, 1.0] / gaps[f_v, 0.5] == pytest.approx(16.0, rel=0.02)
    assert gaps[5.1, 1.0] > 0 > gaps[4.9, 1.0]


@pytest.mark.validation
@pytest.mark.parametrize("n, keep, g, fc", [
    (6, (1, 2, 4, 5), 0.03, 6.30),
    (8, (2, 3, 5, 6), 0.03, 6.30),
    (8, (2, 3, 5, 6), 0.075, 7.20),
    (8, (2, 3, 5, 6), 0.05, 6.00),
])
def test_the_default_order_is_at_least_as_accurate_as_the_walked_order(n, keep, g, fc):
    """f01 against the full ring, and the ZZ of neighbouring kept transmons, which needs two excitations."""
    full = _ring(n, g=g, fc=fc)
    labels = [f"q{k}" for k in keep]
    pairs = [(f"q{a}", f"q{b}") for a, b in zip(keep, keep[1:]) if b - a == 1]

    def errors(reduced):
        f01 = max(abs(float(reduced.freq(q)) - float(full.freq(q))) for q in labels)
        zz = max(abs(float(reduced.freq(a, when={b: 1})) - float(reduced.freq(a))
                     - float(full.freq(a, when={b: 1})) + float(full.freq(a))) for a, b in pairs)
        return f01, zz

    default = eliminate_spectators(full, labels, local=True)
    walked = eliminate_spectators(full, labels, order=_walked(n, keep), local=True)
    (f01, zz), (walked_f01, walked_zz) = errors(default.chip), errors(walked.chip)
    assert f01 <= walked_f01
    assert zz <= 1.05 * walked_zz
    assert _largest_fold(default) < 0.1
