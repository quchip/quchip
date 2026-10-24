"""Physics-verifying dressed-state tests for Chip.dress/energy/transition_freq.

The dispersive model below supplies the reference values.

System under test (matches TestDispersiveShift in tests/test_physics.py):
    - Transmon:  w_q = 5.0 GHz, alpha = -0.25 GHz, levels = 4
    - Resonator: w_r = 7.0 GHz, levels = 10
    - Coupling:  g = 0.05 GHz

Perturbative dispersive formula:
    chi_pert = g^2 * alpha / [Delta * (Delta + alpha)] with Delta = w_q - w_r

For these parameters:
    Delta = -2.0
    chi_pert ~ -1.389e-4 GHz

In this codebase, the observable number-splitting between
n=1 and n=0 manifolds is 2*chi_pert.  This is computed via
energy arithmetic: E(q=1,r=1) - E(q=1,r=0) - E(q=0,r=1) + E(q=0,r=0).
"""

from __future__ import annotations

import numpy as np
import pytest
import warnings

from quchip.chip import Chip
from quchip.chip.couplings import Capacitive
from quchip.control import ChargeDrive, ControlEquipment
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.devices.transmon.charge_basis import ChargeBasisTransmon


OMEGA_Q = 5.0
OMEGA_R = 7.0
ALPHA = -0.25
G = 0.05
Q_LEVELS = 4
R_LEVELS = 10


@pytest.mark.parametrize("backend", ["qutip", pytest.param("dynamiqs", marks=pytest.mark.optional_backend)])
@pytest.mark.parametrize("basis", ["native", "eigen"])
def test_dressed_analysis_uses_local_energy_labels(backend, basis) -> None:
    """Dressed observables and components use energy labels in either solver basis."""
    if backend == "dynamiqs":
        pytest.importorskip("dynamiqs")
    q = ChargeBasisTransmon(
        E_C=0.25, E_J=12.0, n_g=0.13, num_basis=7, basis=basis, label="q",
        **({"levels": 4} if basis == "eigen" else {}),
    )
    drive = ChargeDrive(q, label="xy")
    chip = Chip([q], backend=backend, control_equipment=ControlEquipment([drive]))
    # Independent isolated-device eigenbasis: no coupling can mix energy labels.
    energies, vectors = np.linalg.eigh(np.asarray(q.unresolved_hamiltonian().matrix()))
    vectors = vectors[:, :chip.total_dim]
    reference = np.asarray(chip._analysis.engine_result().bases[q.label].energy_vectors)
    phases = np.sum(vectors.conj() * reference, axis=0)
    vectors = vectors * (phases / np.abs(phases))
    expected_n = vectors.conj().T @ np.asarray(q.charge_coupling_operator()) @ vectors
    dressed_n = chip.backend.to_array(chip.operator_in_dressed_basis(q, "n"))
    np.testing.assert_allclose(dressed_n, expected_n, atol=1e-12)
    explicit = chip.operator_in_dressed_basis(q, chip.observable(q, "n"))
    np.testing.assert_allclose(chip.backend.to_array(explicit), expected_n, atol=1e-12)
    element = chip.drive_matrix_elements(({q: 2}, {q: 3}), drives=[drive])[drive]
    assert element == pytest.approx(expected_n[3, 2], abs=1e-12)
    assert chip.state_components({q: 1})[(1,)] == pytest.approx(1.0, abs=1e-12)
    effective = chip.effective_subspace_hamiltonian(({q: 0}, {q: 1}))
    np.testing.assert_allclose(effective, np.diag(energies[:2]), atol=1e-12)
    assert isinstance(effective, np.ndarray)
    assert effective.dtype == np.complex128
    with pytest.raises(ValueError):
        chip.effective_subspace_hamiltonian(({q: 0}, {q: 0}))
    with pytest.raises(ValueError, match="at least one state"):
        chip.effective_subspace_hamiltonian(())
    if backend == "dynamiqs":
        import jax

        state = jax.jit(lambda ej: chip.with_params({"q.E_J": ej}).state(q=1))(12.0)
        np.testing.assert_allclose(
            chip.backend.to_array(state), chip.backend.to_array(chip.state(q=1)), atol=1e-12,
        )


@pytest.mark.optional_backend
def test_dressed_states_follow_one_phase_convention_on_every_backend() -> None:
    """Dressed vectors have a real, positive assigned bare amplitude, so the backends agree in sign."""
    pytest.importorskip("dynamiqs")
    import jax

    def build(backend):
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        q1 = DuffingTransmon(freq=5.3, anharmonicity=-0.25, levels=3, label="q1")
        return Chip([q0, q1], [Capacitive(q0, q1, g=0.02, label="c")], backend=backend)

    def vector(chip, state):
        return np.ravel(np.asarray(chip.backend.to_array(state)))

    chips = {backend: build(backend) for backend in ("qutip", "dynamiqs")}
    dressed = {backend: chip.dress() for backend, chip in chips.items()}
    for backend, result in dressed.items():
        matrix = np.asarray(result.eigenvector_matrix)
        kets = np.column_stack([vector(chips[backend], ket) for ket in result.eigenstates])
        np.testing.assert_allclose(kets, matrix, atol=1e-12)
        anchors = [matrix[np.ravel_multi_index(label, (3, 3)), result.state_map[label]] for label in result.bare_labels]
        np.testing.assert_allclose(np.imag(anchors), 0.0, atol=1e-12)
        assert min(np.real(anchors)) > 0.9
    np.testing.assert_allclose(
        np.asarray(dressed["dynamiqs"].eigenvector_matrix), np.asarray(dressed["qutip"].eigenvector_matrix),
        atol=1e-10,
    )
    with warnings.catch_warnings():
        # The 2-excitation states hybridize; the comparison still covers them.
        warnings.filterwarnings("ignore", message="Dressed state label")
        for label in dressed["qutip"].bare_labels:
            states = [vector(chip, chip.state(q0=label[0], q1=label[1])) for chip in chips.values()]
            np.testing.assert_allclose(states[1], states[0], atol=1e-10)
    number = [chip.backend.to_array(chip.operator_in_dressed_basis("q0", "n")) for chip in chips.values()]
    np.testing.assert_allclose(number[1], number[0], atol=1e-10)
    traced = jax.jit(lambda g: chips["dynamiqs"].with_params({"c.g": g}).state(q0=1))(0.02)
    np.testing.assert_allclose(
        vector(chips["dynamiqs"], traced), vector(chips["qutip"], chips["qutip"].state(q0=1)), atol=1e-10,
    )


@pytest.mark.optional_backend
def test_traced_default_ground_state_keeps_the_phase_of_a_fallback_anchor() -> None:
    """The traced default ground state has the derivative of its fallback anchor's phase."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    from quchip import EffectiveTerms, Exact

    root3 = np.sqrt(3.0)
    # The all-ground label takes the column of energy 10, which has no amplitude on it.
    U = np.array([[root3, root3, root3, 0.0], [-2, 1, 1, root3], [1, -2, 1, root3], [1, 1, -2, root3]]) / 3
    bare = np.diag([0.0, 0.3, 0.2, 0.5])

    def ground(phi):
        rotation = jnp.diag(jnp.exp(1j * phi * jnp.array([0.0, 1.0, 0.0, 0.0])))
        H = rotation @ (U @ np.diag([1.0, 2.0, 3.0, 10.0]) @ U.T) @ rotation.conj().T
        a = DuffingTransmon(freq=0.2, anharmonicity=0.0, levels=2, label="a")
        b = DuffingTransmon(freq=0.3, anharmonicity=0.0, levels=2, label="b")
        chip = Chip([a, b], effective_terms=[EffectiveTerms(("a", "b"), (2, 2), H - bare, label="mix")],
                    approximation=Exact(), backend="dynamiqs")
        return jnp.ravel(jnp.asarray(chip.analysis._ground_ket(Exact())))

    eager = ground(0.3)
    assert abs(eager[0]) < 1e-12
    assert float(jnp.real(eager[1])) > 0.5
    assert abs(float(jnp.imag(eager[1]))) < 1e-12
    _, tangent = jax.jvp(ground, (0.3,), (1.0,))
    central = (ground(0.3 + 1e-6) - ground(0.3 - 1e-6)) / 2e-6
    np.testing.assert_allclose(tangent, central, atol=1e-8)


@pytest.fixture
def dispersive_system():
    """Build the shared transmon-resonator test system."""
    qubit = DuffingTransmon(
        freq=OMEGA_Q,
        anharmonicity=ALPHA,
        levels=Q_LEVELS,
        label="q",
    )
    resonator = Resonator(freq=OMEGA_R, levels=R_LEVELS, label="r")
    coupling = Capacitive(qubit, resonator, g=G)
    chip = Chip(devices=[qubit, resonator], couplings=[coupling])
    return chip, qubit, resonator


class TestDressedStates:
    """Verify dressed-state computation and state assignment surfaces."""


    def test_resonant_system_exposes_hybridization_diagnostics(self) -> None:
        """Resonant coupling produces hybridized labels, overlap fractions, and one warning."""
        r_a = Resonator(freq=6.0, levels=4, label="r_a")
        r_b = Resonator(freq=6.0, levels=4, label="r_b")
        chip = Chip([r_a, r_b], [Capacitive(r_a, r_b, g=0.05)])

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = chip.dress()

        assert result.hybridized_labels
        assert (0, 3) in result.hybridized_labels
        assert (3, 0) in result.hybridized_labels
        assert 0.35 < result.assignment_overlaps[(0, 3)] < 0.4
        assert 0.35 < result.assignment_overlaps[(3, 0)] < 0.4

        hybridization_warnings = [
            warning for warning in captured if "keep less than overlap_threshold" in str(warning.message)
        ]
        assert len(hybridization_warnings) == 1
        assert "hybridized_labels" in str(hybridization_warnings[0].message)

    def test_overlap_threshold_controls_hybridization_flags(self) -> None:
        """Relaxing overlap_threshold drops labels from hybridized_labels."""
        r_a = Resonator(freq=6.0, levels=4, label="r_a")
        r_b = Resonator(freq=6.0, levels=4, label="r_b")
        chip = Chip([r_a, r_b], [Capacitive(r_a, r_b, g=0.05)])

        default = chip.dress()
        relaxed = chip.dress(overlap_threshold=0.2, force=True)

        assert (0, 3) in default.hybridized_labels
        assert (3, 0) in default.hybridized_labels
        assert (0, 3) not in relaxed.hybridized_labels
        assert (3, 0) not in relaxed.hybridized_labels

    def test_vectorized_assignment_matches_bruteforce_greedy_overlap(self) -> None:
        """state_map matches a brute-force greedy overlap assignment computed independently."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        r = Resonator(freq=7.0, levels=3, label="r")
        chip = Chip([q, r], [Capacitive(q, r, g=0.05)])
        result = chip.dress()

        backend = chip.backend
        bare_labels = list(result.bare_labels)
        overlaps: dict[tuple[tuple[int, ...], int], float] = {}
        for bare_label in bare_labels:
            basis_states = [backend.basis(dev.levels, k) for dev, k in zip(chip.devices, bare_label)]
            bare_state = backend.tensor_states(*basis_states)
            for dressed_idx, estate in enumerate(result.eigenstates):
                overlaps[(bare_label, dressed_idx)] = float(abs(backend.overlap(bare_state, estate)) ** 2)

        assigned_bare: set[tuple[int, ...]] = set()
        assigned_dressed: set[int] = set()
        reference_map: dict[tuple[int, ...], int] = {}
        for (bare_label, dressed_idx), overlap in sorted(overlaps.items(), key=lambda item: item[1], reverse=True):
            if bare_label in assigned_bare or dressed_idx in assigned_dressed:
                continue
            reference_map[bare_label] = dressed_idx
            assigned_bare.add(bare_label)
            assigned_dressed.add(dressed_idx)

        assert result.state_map == reference_map


class TestLookupHelpers:

    def test_state_components_are_normalized(self, dispersive_system) -> None:
        """state_components() weights sum to 1."""
        chip, _, _ = dispersive_system
        chip.dress()

        components = chip.state_components(0, n_components=len(chip._ensure_dressed().bare_labels))

        assert components
        assert sum(components.values()) == pytest.approx(1.0, abs=1e-12)

    def test_operator_in_dressed_basis_preserves_hermiticity(self, dispersive_system) -> None:
        """operator_in_dressed_basis() of a Hermitian operator stays Hermitian after truncation."""
        chip, qubit, _ = dispersive_system
        dressed_op = chip.operator_in_dressed_basis(qubit, "Z", truncate=4)
        dense = np.asarray(chip.backend.to_array(dressed_op), dtype=complex)

        assert dense.shape == (4, 4)
        np.testing.assert_allclose(dense, dense.conj().T, atol=1e-12)


class TestDriveMatrixElements:
    """Verify dressed drive operators against an explicit basis transform."""

    @staticmethod
    def _driven_pair():
        q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q1")
        q2 = DuffingTransmon(freq=5.3, anharmonicity=-0.24, levels=3, label="q2")
        d1 = ChargeDrive(q1, label="d1")
        d2 = ChargeDrive(q2, label="d2")
        chip = Chip(
            [q1, q2],
            [Capacitive(q1, q2, g=0.03)],
            control_equipment=ControlEquipment([d1, d2]),
        )
        return chip, q1, q2, d1, d2


    def test_explicit_transition_supports_arbitrary_bare_state_mappings(self) -> None:
        """Explicit ``(initial, final)`` mappings select arbitrary dressed transitions."""
        chip, q1, q2, d1, _ = self._driven_pair()
        transition = ({q1: 0, q2: 1}, {q1: 1, q2: 1})

        element = chip.drive_matrix_elements(transition, drives=[d1])[d1]

        initial = chip.dressed_index(transition[0])
        final = chip.dressed_index(transition[1])
        assert initial is not None and final is not None
        from quchip.control.signal import AnalyticSignal
        from quchip.declarative.expr import split_dynamic_hamiltonian
        from quchip.engine.ir import Constant

        authored = d1.hamiltonian(q1, AnalyticSignal(Constant(1.0)))
        operator = split_dynamic_hamiltonian(authored)[0][1]
        dressed = chip.operator_in_dressed_basis(q1, operator)
        explicit = np.asarray(chip.backend.to_array(dressed), dtype=complex)[final, initial]
        assert element == pytest.approx(explicit, abs=1e-12)


    def test_missing_equipment_and_unknown_drive_report_available_lines(self) -> None:
        """Drive resolution failures identify missing equipment and available labels."""
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        unwired = Chip([q])
        with pytest.raises(ValueError, match="control equipment"):
            unwired.drive_matrix_elements(q)

        chip, q1, _, _, _ = self._driven_pair()
        with pytest.raises(KeyError, match=r"missing.*Available.*d1.*d2"):
            chip.drive_matrix_elements(q1, drives=["missing"])

class TestCacheInvalidation:
    """Verify frame changes do NOT invalidate dressed-state cache."""

    def test_static_parameter_change_invalidates_dressed_cache(self, dispersive_system) -> None:
        """Mutating a device parameter invalidates the cached dressed result."""
        chip, qubit, _ = dispersive_system

        original = chip.dress()
        qubit.freq += 0.1

        refreshed = chip._ensure_dressed()

        assert refreshed is not original
        assert refreshed.eigenvalues[1] != pytest.approx(original.eigenvalues[1])

    def test_hamiltonian_resolves_frame_without_changing_dressed_data(self, dispersive_system) -> None:
        """Frame policy changes resolved inspection but not dressed frequencies."""
        chip, qubit, resonator = dispersive_system

        frequencies = (chip.freq(qubit), chip.freq(resonator))
        H_lab = chip.hamiltonian().matrix(t=0.0)
        chip.set_frame("rotating")
        H_rotating = chip.hamiltonian().matrix(t=0.0)

        assert not np.allclose(H_lab, H_rotating)
        np.testing.assert_allclose(
            (chip.freq(qubit), chip.freq(resonator)),
            frequencies,
            atol=1e-12,
        )


def test_cached_qutip_kerr_values_remain_accessible_inside_jit():
    import jax

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
    r = Resonator(freq=7.0, levels=4, label="r")
    chip = Chip([q, r], [Capacitive(q, r, g=0.08)], backend="qutip")
    expected = float(chip.dispersive_shift("q", "r"))
    assert jax.jit(lambda scale: chip.dispersive_shift("q", "r") * scale)(2.0) == pytest.approx(2 * expected)


def _readout_pair(readout_levels: int, readout_type=Resonator) -> tuple[Chip, DuffingTransmon, Resonator]:
    q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
    r = readout_type(freq=7.0, levels=readout_levels, label="r")
    return Chip([q, r], [Capacitive(q, r, g=0.105)]), q, r


def _two_level_readout_pull() -> float:
    """Return the cross-Kerr of a three-level transmon and a two-level readout under RWA."""
    one = np.linalg.eigvalsh([[5.0, 0.105], [0.105, 7.0]])
    two = np.linalg.eigvalsh([[9.8, np.sqrt(2) * 0.105], [np.sqrt(2) * 0.105, 12.0]])
    # |1,0> is the lower one-excitation state and |1,1> is the upper two-excitation state.
    return two[1] - one[0] - one[1]


@pytest.mark.unit
def test_two_level_readout_warns_on_two_excitation_queries_and_keeps_the_truncated_value() -> None:
    """A two-excitation query on a two-level readout warns once and returns the truncated-model value."""
    import jax

    chip, q, r = _readout_pair(2)
    with pytest.warns(UserWarning, match=r"second order.*'r' keeps 2 levels.*levels=3 or more on 'r'") as record:
        shift = chip.dispersive_shift(q, r)
    assert len([w for w in record if "keeps 2 levels" in str(w.message)]) == 1
    assert float(shift) == pytest.approx(_two_level_readout_pull(), abs=1e-12)
    with pytest.warns(UserWarning, match=r"'r' keeps 2 levels"):
        pulled = chip.freq(r, when={q: 1})
    assert float(pulled - chip.freq(r)) == pytest.approx(_two_level_readout_pull(), abs=1e-12)
    with pytest.warns(UserWarning, match=r"'r' keeps 2 levels") as record:
        matrix = chip.kerr_matrix()
    assert len([w for w in record if "keeps 2 levels" in str(w.message)]) == 1
    assert float(matrix[q, r]) == pytest.approx(_two_level_readout_pull(), abs=1e-12)
    with pytest.warns(UserWarning, match=r"'r' keeps 2 levels"):
        traced = jax.jit(lambda scale: chip.dispersive_shift(q, r) * scale)(2.0)
    assert float(traced) == pytest.approx(2 * _two_level_readout_pull(), abs=1e-12)
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q1")
    q2 = DuffingTransmon(freq=5.3, anharmonicity=-0.2, levels=3, label="q2")
    bus = Resonator(freq=6.5, levels=2, label="bus")
    bus_chip = Chip([q1, q2, bus], [Capacitive(q1, bus, g=0.05), Capacitive(q2, bus, g=0.05)])
    with pytest.warns(UserWarning, match=r"level 2 of 'bus'.*fourth order.*'bus' keeps 2 levels"):
        bus_chip.static_zz(q1, q2)
    q2 = DuffingTransmon(freq=5.3, anharmonicity=-0.2, levels=2, label="q2")
    bus = Resonator(freq=6.5, levels=3, label="bus")
    bus_chip = Chip([q1, q2, bus], [Capacitive(q1, bus, g=0.05), Capacitive(q2, bus, g=0.05)])
    with pytest.warns(UserWarning, match=r"fourth order.*'q2' keeps 2 levels.*levels=3 or more on 'q2'"):
        bus_chip.static_zz(q1, q2)


def _coupler_chain(qubit_levels: int) -> tuple[Chip, DuffingTransmon, DuffingTransmon]:
    """Return two transmons that interact through two couplers in series, so their ZZ starts at sixth order."""
    q1 = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=qubit_levels, label="q1")
    c1 = Resonator(freq=6.4, levels=3, label="c1")
    c2 = Resonator(freq=6.8, levels=3, label="c2")
    q2 = DuffingTransmon(freq=5.3, anharmonicity=-0.2, levels=qubit_levels, label="q2")
    couplings = [Capacitive(q1, c1, g=0.1), Capacitive(c1, c2, g=0.1), Capacitive(c2, q2, g=0.1)]
    return Chip([q1, c1, c2, q2], couplings), q1, q2


@pytest.mark.unit
def test_truncation_warning_follows_counter_rotating_steps_and_long_interaction_paths() -> None:
    """Exact() reaches a missing level by pair creation, and distant excitations widen the checked orders."""
    from quchip import Exact

    def exact_pair(qubit_levels: int) -> Chip:
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=qubit_levels, label="q")
        r = Resonator(freq=7.0, levels=5, label="r")
        return Chip([q, r], [Capacitive(q, r, g=0.105)], approximation=Exact())

    with pytest.warns(UserWarning, match=r"second order.*'q' keeps 2 levels.*levels=3 or more on 'q'"):
        truncated = exact_pair(2).freq("q")
    # Pair creation couples |1,0> directly to the missing |2,1>, which moves f01 by about 1.9 MHz.
    assert float(truncated - exact_pair(3).freq("q")) == pytest.approx(1.865e-3, abs=1e-5)
    # The ZZ of two couplers in series starts at sixth order, the order at which the missing level 2 of a
    # two-level transmon enters. That level changes the ZZ by a factor of two.
    chain, q1, q2 = _coupler_chain(2)
    with pytest.warns(UserWarning, match=r"sixth order.*'q1' keeps 2 levels.*levels=3 or more on 'q1'"):
        truncated = chain.static_zz(q1, q2)
    full, q1, q2 = _coupler_chain(3)
    assert float(truncated) / float(full.static_zz(q1, q2)) == pytest.approx(0.465, abs=0.01)


@pytest.mark.unit
def test_truncation_warning_reads_the_kept_bands_of_couplings_and_port_pairs() -> None:
    """A coupling that RWA removes links nothing, and a port pair that creates excitations counts."""
    from quchip import Coupling, Exact, PortNetwork, RWA

    def pairs(a, b, backend):
        lower = backend.tensor(a.lowering_operator(), b.lowering_operator())
        return lower + backend.tensor(backend.dag(a.lowering_operator()), backend.dag(b.lowering_operator()))

    def pair_chip(approximation: RWA | Exact) -> Chip:
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        r = Resonator(freq=7.0, levels=2, label="r")
        return Chip([q, r], [Coupling(q, r, g=0.05, interaction=pairs)], approximation=approximation)

    with pytest.warns(UserWarning, match="vanishes entirely under RWA") as record:
        frequency = pair_chip(RWA()).freq("q", when={"r": 1})
    assert float(frequency) == pytest.approx(5.0, abs=1e-12)
    assert not [w for w in record if "keeps 2 levels" in str(w.message)]
    with pytest.warns(UserWarning, match=r"second order.*'r' keeps 2 levels"):
        pair_chip(Exact()).freq("q", when={"r": 1})

    def cascade(levels: int) -> Chip:
        a = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=levels, label="a")
        b = Resonator(freq=7.0, levels=3, label="b")
        line = PortNetwork(label="line")
        # The generated pair L_b^dagger L_a = 0.01 b^dagger a^dagger creates an excitation on each device.
        up = line.port("a_out", target=a, rate=0.01, operator="a_dag")
        down = line.port("b_in", target=b, rate=0.01)
        line.cascade(up, down)
        line.expose("probe", input=up, output=down)
        return Chip([a, b], port_network=line, approximation=Exact())

    with pytest.warns(UserWarning, match=r"second order.*'a' keeps 2 levels.*levels=3 or more on 'a'"):
        truncated = cascade(2).freq("a")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        full = cascade(3).freq("a")
    # The pair term has strength g = rate / (4 pi) in GHz, and the missing |2,1> lies f_a + f_b + alpha_a
    # = 11.8 GHz above |1,0>, so the two-level transition is too high by 2 g^2 / 11.8 at second order.
    assert float(truncated - full) == pytest.approx(2 * (0.01 / (4 * np.pi)) ** 2 / 11.8, rel=1e-3)


@pytest.mark.unit
def test_converged_or_single_excitation_queries_do_not_warn_about_truncation() -> None:
    """Queries that reach no missing level of a truncated device within two coupling steps stay silent."""
    from quchip import CrossKerr, Exact, Qubit

    def filtered_readout(filter_levels: int) -> tuple[Chip, DuffingTransmon, Resonator]:
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        r = Resonator(freq=7.0, levels=4, label="r")
        f = Resonator(freq=7.02, levels=filter_levels, label="f")
        return Chip([q, r, f], [Capacitive(q, r, g=0.105), Capacitive(r, f, g=0.0077)]), q, r

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chip, q, r = _readout_pair(3)
        chip.dispersive_shift(q, r)
        chip.freq(r, when={q: 1})
        two_level, q, r = _readout_pair(2)
        two_level.freq(q)
        two_level.freq()
        qubit = Qubit(freq=5.0, label="q")
        r = Resonator(freq=7.0, levels=3, label="r")
        Chip([qubit, r], [Capacitive(qubit, r, g=0.105)]).dispersive_shift(qubit, r)
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        r = Resonator(freq=7.0, levels=3, label="r")
        spectator = Resonator(freq=8.0, levels=2, label="spectator")
        Chip([q, r, spectator], [Capacitive(q, r, g=0.105)]).dispersive_shift(q, r)
        # A two-level filter behind the readout misses only states three steps away, at sixth order.
        two_level, q, r = filtered_readout(2)
        truncated = two_level.freq(r, when={q: 1})
        # A diagonal coupling moves no excitation, so the missing level 2 of r never mixes in.
        q = DuffingTransmon(freq=5.0, anharmonicity=-0.2, levels=3, label="q")
        r = Resonator(freq=7.0, levels=2, label="r")
        diagonal = Chip([q, r], [CrossKerr(q, r, chi=-0.001)])
        assert float(diagonal.freq(q, when={r: 1})) == pytest.approx(4.999, abs=1e-12)
        # Under Exact(), a missing level 3 of a three-level transmon needs pair creation and one more step.
        q1 = DuffingTransmon(freq=5.326, anharmonicity=-0.261, levels=3, label="q1")
        q2 = DuffingTransmon(freq=5.192, anharmonicity=-0.263, levels=3, label="q2")
        bus = Resonator(freq=6.3, levels=3, label="bus")
        exact = Chip([q1, q2, bus], [Capacitive(q1, bus, g=0.03), Capacitive(q2, bus, g=0.03)],
                     approximation=Exact())
        exact.freq(q1, when={q2: 1})
    three_level, q, r = filtered_readout(3)
    assert float(truncated) == pytest.approx(float(three_level.freq(r, when={q: 1})), abs=1e-7)
