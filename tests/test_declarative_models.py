from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from quchip import CouplingModel, DeviceModel, EnvelopeShape, Scalar, qnp, parameter
from quchip.control.envelopes import BaseEnvelope
from quchip.devices.base import BaseDevice


class CosineEnvelope(EnvelopeShape):
    duration: Scalar = parameter(positive=True)
    amplitude: Scalar = parameter(default=1.0)

    def value(self, t):
        return self.amplitude * (1.0 - qnp.cos(qnp.pi * t / self.duration))


def test_custom_envelope_samples_without_xp_argument():
    """A custom EnvelopeShape subclass samples correctly through the base pipeline without an explicit xp argument."""
    env = CosineEnvelope(duration=10.0, amplitude=2.0)
    samples = env.sample(np.asarray([0.0, 5.0, 10.0]))
    npt.assert_allclose(np.asarray(samples), np.asarray([0.0, 2.0, 4.0]), atol=1e-7)


class HarmonicMode(DeviceModel):
    freq: Scalar = parameter(positive=True)
    approximation = None

    def local_hamiltonian(self, op):
        return self.freq * op.n


def test_custom_device_hamiltonian_compiles_without_backend_calls():
    """A custom DeviceModel subclass compiles its local Hamiltonian to an operator sized to its Fock truncation."""
    mode = HarmonicMode(freq=7.0, levels=4, label="m")
    h = mode.hamiltonian()
    assert h.shape == (4, 4)


def test_device_physics_notes_include_approximation_when_declared():
    """DeviceModel.physics_notes() reports Hilbert-space truncation even when the subclass declares no approximation."""
    mode = HarmonicMode(freq=7.0, levels=4, label="m")
    notes = mode.physics_notes()
    assert any("Hilbert truncation" in note for note in notes)


def test_tunable_param_names_derived_default_covers_all_declared_fields():
    """A DeviceModel subclass with no explicit tunable_param_names exposes exactly its declared parameter() fields."""
    class _DerivedTunables(DeviceModel):
        freq: Scalar = parameter(positive=True)
        anharm: Scalar = parameter()

        def local_hamiltonian(self, op):
            return self.freq * op.n

    dev = _DerivedTunables(freq=5.0, anharm=-0.2, levels=3)
    assert dev.tunable_param_names == ("freq", "anharm")
    assert set(dev.tunable_params()) == {"freq", "anharm"}


def test_tunable_param_names_explicit_curation_is_exact():
    """An explicit tunable_param_names tuple exposes exactly those names, excluding other declared fields."""
    class _CuratedTunables(DeviceModel):
        freq: Scalar = parameter(positive=True)
        quality_factor: Scalar = parameter(default=None)
        tunable_param_names = ("freq",)

        def local_hamiltonian(self, op):
            return self.freq * op.n

    dev = _CuratedTunables(freq=5.0, levels=3)
    assert set(dev.tunable_params()) == {"freq"}


def test_tunable_param_names_explicit_empty_freezes_device():
    """An explicit empty tunable_param_names tuple exposes no tunable parameters (deliberate inverse-design freeze)."""
    class _FrozenTunables(DeviceModel):
        freq: Scalar = parameter(positive=True)
        tunable_param_names = ()

        def local_hamiltonian(self, op):
            return self.freq * op.n

    dev = _FrozenTunables(freq=5.0, levels=3)
    assert dev.tunable_params() == {}


def test_tunable_param_names_inherited_explicit_curation_is_not_re_derived():
    """A subclass of an explicitly-curated DeviceModel inherits that exact curation instead of re-deriving."""
    class _CuratedParent(DeviceModel):
        freq: Scalar = parameter(positive=True)
        quality_factor: Scalar = parameter(default=None)
        tunable_param_names = ()

        def local_hamiltonian(self, op):
            return self.freq * op.n

    class _CuratedChild(_CuratedParent):
        extra: Scalar = parameter(default=1.0)

    dev = _CuratedChild(freq=5.0, levels=3)
    assert dev.tunable_param_names == ()
    assert dev.tunable_params() == {}


def test_tunable_param_names_derived_lineage_re_derives_with_new_fields():
    """A subclass of a purely derived-default DeviceModel re-derives to include its own new declared fields."""
    class _DerivedParent(DeviceModel):
        a: Scalar = parameter(positive=True)
        b: Scalar = parameter(default=0.0)

        def local_hamiltonian(self, op):
            return self.a * op.n

    class _DerivedChild(_DerivedParent):
        c: Scalar = parameter(default=0.0)

    assert _DerivedParent.tunable_param_names == ("a", "b")
    assert _DerivedChild.tunable_param_names == ("a", "b", "c")


def test_tunable_param_names_accepts_a_plain_class_attribute():
    """An explicit tunable_param_names entry may name a genuine class attribute, not only a parameter() field."""
    class _WithClassAttr(DeviceModel):
        freq: Scalar = parameter(positive=True)
        derived_freq = 0.0
        tunable_param_names = ("freq", "derived_freq")

        def local_hamiltonian(self, op):
            return self.freq * op.n

    dev = _WithClassAttr(freq=5.0, levels=3)
    assert set(dev.tunable_param_names) == {"freq", "derived_freq"}


def test_tunable_param_names_unresolved_name_raises_at_class_definition():
    """An unresolvable explicit tunable_param_names entry raises ValueError at class definition."""
    with pytest.raises(ValueError, match="not a declared parameter"):
        class _BadTunables(DeviceModel):
            freq: Scalar = parameter(positive=True)
            tunable_param_names = ("not_a_field",)

            def local_hamiltonian(self, op):
                return self.freq * op.n


def test_tunable_param_names_duplicate_entry_raises_at_class_definition():
    """A duplicate name in an explicit tunable_param_names tuple raises ValueError at class definition."""
    with pytest.raises(ValueError, match="duplicate"):
        class _BadTunables(DeviceModel):
            freq: Scalar = parameter(positive=True)
            tunable_param_names = ("freq", "freq")

            def local_hamiltonian(self, op):
                return self.freq * op.n


def test_tunable_param_names_bare_string_raises_at_class_definition():
    """A bare string tunable_param_names value (instead of a tuple) raises TypeError at class definition."""
    with pytest.raises(TypeError, match="must be a tuple"):
        class _BadTunables(DeviceModel):
            freq: Scalar = parameter(positive=True)
            tunable_param_names = "freq"

            def local_hamiltonian(self, op):
                return self.freq * op.n


def test_tunable_param_names_non_string_entry_raises_at_class_definition():
    """A non-string entry in an explicit tunable_param_names tuple raises TypeError at class definition."""
    with pytest.raises(TypeError, match="must be strings"):
        class _BadTunables(DeviceModel):
            freq: Scalar = parameter(positive=True)
            tunable_param_names = (1,)

            def local_hamiltonian(self, op):
                return self.freq * op.n


class NumberNumber(CouplingModel):
    chi: Scalar = parameter()

    def interaction(self, a, b):
        return self.chi * a.n * b.n


def test_custom_coupling_compiles_without_backend_tensor_calls():
    """A custom CouplingModel subclass compiles its interaction to an operator on the joint two-device Hilbert space."""
    a = HarmonicMode(freq=5.0, levels=3, label="a")
    b = HarmonicMode(freq=6.0, levels=4, label="b")
    coupling = NumberNumber(a, b, chi=0.01)
    h = coupling.interaction_hamiltonian()
    assert h.shape == (12, 12)


def test_time_dependent_without_dynamic_source_errors():
    """dynamic_interaction_terms() rejects a time_dependent override whose expression carries no dynamic source."""
    class BadDynamic(CouplingModel):
        g: Scalar = parameter()

        def interaction(self, a, b):
            return self.g * a.x * b.x

        def time_dependent(self, a, b):
            return self.g * a.x * b.x

    a = HarmonicMode(freq=5.0, levels=3, label="a")
    b = HarmonicMode(freq=6.0, levels=4, label="b")
    coupling = BadDynamic(a, b, g=0.01)
    with pytest.raises(ValueError, match="exactly one dynamic source"):
        coupling.dynamic_interaction_terms(None)


def _arr(op):
    """Dense numpy view of a backend operator (Qobj or array)."""
    return op.full() if hasattr(op, "full") else np.asarray(op)


def test_tunable_capacitive_parametric_operator_follows_chip_rwa():
    """The pump-multiplied operator structure re-selects under chip RWA."""
    from quchip import Chip, DuffingTransmon, TunableCapacitive

    def _coupler():
        q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        q1 = DuffingTransmon(freq=5.05, anharmonicity=-0.25, levels=3, label="q1")
        return q0, q1, TunableCapacitive(q0, q1, g_0=0.0)

    q0r, q1r, c_rwa = _coupler()
    chip_rwa = Chip([q0r, q1r], [c_rwa], frame="rotating", rwa=True)
    op_rwa = c_rwa.parametric_operator(chip_rwa)
    q0f, q1f, c_full = _coupler()
    chip_full = Chip([q0f, q1f], [c_full], frame="rotating", rwa=False)
    op_full = c_full.parametric_operator(chip_full)

    # RWA keeps a†b + a b†; full keeps (a + a†)(b + b†).
    assert not np.allclose(_arr(op_rwa), _arr(op_full))


def test_tunable_capacitive_without_modulation_has_no_dynamic_term():
    """A purely static TunableCapacitive emits no dynamic interaction term."""
    from quchip import Chip, DuffingTransmon, TunableCapacitive

    q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
    q1 = DuffingTransmon(freq=5.05, anharmonicity=-0.25, levels=3, label="q1")
    c = TunableCapacitive(q0, q1, g_0=0.02)
    assert c.dynamic_interaction_terms(Chip([q0, q1], [c], frame="rotating", rwa=True)) == []


def test_declarative_device_to_dict_contains_declared_parameters():
    """DeviceModel.to_dict() serializes declared parameter values alongside the base device fields."""
    mode = HarmonicMode(freq=7.0, levels=4, label="m")
    payload = mode.to_dict()
    assert payload["freq"] == 7.0
    assert payload["label"] == "m"


def test_declarative_device_round_trip_uses_declared_parameters():
    """A DeviceModel round-trips through to_dict()/from_dict() with its type and declared parameter values preserved."""
    mode = HarmonicMode(freq=7.0, levels=4, label="m")
    restored = BaseDevice.from_dict(mode.to_dict())
    assert isinstance(restored, HarmonicMode)
    assert restored.freq == 7.0
    assert restored.levels == 4
    assert restored.label == "m"


def test_declarative_parameter_mutation_bumps_state_version():
    """Mutating a declared parameter increments the device's state_version by exactly one."""
    mode = HarmonicMode(freq=7.0, levels=4, label="m")
    before = mode.state_version
    mode.freq = 7.1
    assert mode.state_version == before + 1


def test_declarative_physics_notes_include_approximation():
    """DeviceModel.physics_notes() includes a subclass's declared approximation string."""
    class ApproxDevice(DeviceModel):
        freq: Scalar = parameter(positive=True)
        approximation = "Toy expansion."

        def local_hamiltonian(self, op):
            return self.freq * op.n

    notes = ApproxDevice(freq=5.0).physics_notes()
    assert "Toy expansion." in notes


def test_declarative_envelope_round_trip_uses_declared_parameters():
    """An EnvelopeShape round-trips through to_dict()/from_dict() with type and declared parameters preserved."""
    env = CosineEnvelope(duration=10.0, amplitude=2.0)
    restored = BaseEnvelope.from_dict(env.to_dict())
    assert isinstance(restored, CosineEnvelope)
    assert restored.duration == 10.0
    assert restored.amplitude == 2.0


def test_declarative_coupling_to_dict_contains_declared_parameters_and_endpoints():
    """CouplingModel.to_dict() serializes declared parameter values together with both endpoint device labels."""
    a = HarmonicMode(freq=5.0, levels=3, label="a")
    b = HarmonicMode(freq=6.0, levels=4, label="b")
    coupling = NumberNumber(a, b, chi=0.01, label="zz")
    payload = coupling.to_dict()
    assert payload["device_a_label"] == "a"
    assert payload["device_b_label"] == "b"
    assert payload["label"] == "zz"
    assert payload["chi"] == 0.01


def test_duffing_transmon_is_declarative_and_keeps_hamiltonian_shape():
    """DuffingTransmon compiles to its Fock truncation and discloses the Duffing quartic term in physics notes."""
    from quchip import DuffingTransmon

    q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3)
    assert isinstance(q, DeviceModel)
    assert q.hamiltonian().shape == (3, 3)
    assert any("Duffing" in note or "quartic" in note.lower() for note in q.physics_notes())


def test_gaussian_shape_is_declarative_without_xp():
    """Gaussian's value() returns a scalar for scalar time input without an explicit xp argument."""
    from quchip import Gaussian

    g = Gaussian(duration=20.0, amplitude=0.5, sigmas=3.0)
    assert isinstance(g, EnvelopeShape)
    value = g.value(qnp.asarray(10.0))
    assert value.shape == ()
