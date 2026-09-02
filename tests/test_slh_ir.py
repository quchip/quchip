"""Resolved SLH internal-normal-form invariants."""

from __future__ import annotations

import numpy as np
import pytest

from quchip.engine.ir import (
    CanonicalOperator,
    CollapseTerm,
    HamiltonianProgram,
    ResolvedSLH,
    SLHChannel,
)


def _collapse(
    source: str,
    channel: str,
    *,
    frame_frequency: float | None,
    operator: CanonicalOperator | None = None,
    rate: float = 0.25,
    phase: float | None = None,
) -> CollapseTerm:
    if operator is None:
        operator = CanonicalOperator.from_dense(
            np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex),
            dims=(2,),
            basis="native",
            subsystem_labels=(source,),
            tag=f"collapse:{source}:{channel}",
        )
    return CollapseTerm(
        operator=operator,
        rate=rate,
        source=source,
        channel=channel,
        phase=(0.0 if frame_frequency is not None else None) if phase is None else phase,
        frame_frequency=frame_frequency,
    )


def test_closed_resolved_slh_has_zero_channel_identity() -> None:
    """A closed model resolves to the unique zero-channel SLH identity."""
    slh = ResolvedSLH.from_terms(static_terms=(), dynamic_terms=(), collapse_terms=())

    assert slh.S.shape == (0, 0)
    assert slh.L == ()
    assert slh.channels == ()


def test_resolved_slh_orders_exposed_before_hidden() -> None:
    """Accessible channels precede hidden baths while preserving group order."""
    hidden = _collapse("q", "relaxation", frame_frequency=None)
    exposed = _collapse("readout", "external_coupling", frame_frequency=6.0)

    slh = ResolvedSLH.from_terms(
        static_terms=(),
        dynamic_terms=(),
        collapse_terms=(hidden, exposed),
    )

    assert [channel.key for channel in slh.channels] == [
        "external.readout.external_coupling",
        "hidden.q.relaxation",
    ]
    np.testing.assert_allclose(slh.S, np.eye(2))


def test_from_terms_disambiguates_repeated_hidden_channel_names() -> None:
    """Distinct authored jumps keep distinct stable addresses even when names repeat."""
    first = _collapse("q", "loss", frame_frequency=None)
    second = _collapse("q", "loss", frame_frequency=None)

    slh = ResolvedSLH.from_terms(
        static_terms=(),
        dynamic_terms=(),
        collapse_terms=(first, second),
    )

    assert [channel.key for channel in slh.channels] == [
        "hidden.q.loss",
        "hidden.q.loss#2",
    ]


def test_concrete_scattering_payload_is_read_only() -> None:
    """Frozen SLH structure cannot be mutated through its concrete S payload."""
    channel = SLHChannel(
        key="hidden.q.relaxation",
        accessibility="hidden",
        collapse=_collapse("q", "relaxation", frame_frequency=None),
    )
    slh = ResolvedSLH(
        scattering=np.eye(1, dtype=complex),
        hamiltonian=HamiltonianProgram(),
        channels=(channel,),
    )

    with pytest.raises(ValueError, match="read-only"):
        slh.S[0, 0] = 0.0


def test_channel_coupling_applies_rate_and_phase_without_densifying() -> None:
    """The physical L keeps sparse layout while applying exp(i phase) sqrt(rate)."""
    lowering = CanonicalOperator.from_csr(
        values=np.array([1.0 + 0.0j]),
        indices=np.array([1]),
        indptr=np.array([0, 1, 1]),
        shape=(2, 2),
        dims=(2,),
        basis="native",
        subsystem_labels=("readout",),
        tag="lowering",
    )
    collapse = _collapse(
        "readout",
        "external_coupling",
        frame_frequency=6.0,
        operator=lowering,
        rate=0.25,
        phase=np.pi / 2,
    )

    coupling = ResolvedSLH.from_terms(
        static_terms=(),
        dynamic_terms=(),
        collapse_terms=(collapse,),
    ).L[0]

    assert coupling.layout == "csr"
    assert coupling.tag == "slh:external.readout.external_coupling"
    np.testing.assert_allclose(coupling.values, np.array([0.5j]))
    np.testing.assert_allclose(coupling.to_dense(), 0.5j * lowering.to_dense())


def test_resolved_slh_rejects_scattering_shape_mismatch() -> None:
    """Scattering has one scalar input and output axis per resolved channel."""
    channel = SLHChannel(
        key="hidden.q.relaxation",
        accessibility="hidden",
        collapse=_collapse("q", "relaxation", frame_frequency=None),
    )

    with pytest.raises(ValueError, match="one row and column per channel"):
        ResolvedSLH(
            scattering=np.eye(2),
            hamiltonian=HamiltonianProgram(),
            channels=(channel,),
        )


def test_resolved_slh_rejects_nonunitary_concrete_scattering() -> None:
    """Concrete scalar scattering must preserve all modeled channel flux."""
    channel = SLHChannel(
        key="hidden.q.relaxation",
        accessibility="hidden",
        collapse=_collapse("q", "relaxation", frame_frequency=None),
    )

    with pytest.raises(ValueError, match="unitary"):
        ResolvedSLH(
            scattering=np.array([[0.5]], dtype=complex),
            hamiltonian=HamiltonianProgram(),
            channels=(channel,),
        )


def test_resolved_slh_rejects_duplicate_channel_keys() -> None:
    """Every resolved channel has one stable address."""
    collapse = _collapse("q", "relaxation", frame_frequency=None)
    channel = SLHChannel(
        key="hidden.q.relaxation",
        accessibility="hidden",
        collapse=collapse,
    )

    with pytest.raises(ValueError, match="unique"):
        ResolvedSLH(
            scattering=np.eye(2),
            hamiltonian=HamiltonianProgram(),
            channels=(channel, channel),
        )


def test_resolved_slh_rejects_hidden_channel_before_exposed_channel() -> None:
    """The complete channel vector keeps exposed entries before hidden baths."""
    hidden = SLHChannel(
        key="hidden.q.relaxation",
        accessibility="hidden",
        collapse=_collapse("q", "relaxation", frame_frequency=None),
    )
    exposed = SLHChannel(
        key="external.readout.external_coupling",
        accessibility="exposed",
        collapse=_collapse("readout", "external_coupling", frame_frequency=6.0),
    )

    with pytest.raises(ValueError, match="before hidden"):
        ResolvedSLH(
            scattering=np.eye(2),
            hamiltonian=HamiltonianProgram(),
            channels=(hidden, exposed),
        )


def test_engine_exports_resolved_slh_contract() -> None:
    """Advanced users can inspect the resolved SLH types from quchip.engine."""
    from quchip.engine import HamiltonianProgram as ExportedHamiltonianProgram
    from quchip.engine import ResolvedSLH as ExportedResolvedSLH
    from quchip.engine import SLHChannel as ExportedSLHChannel

    assert ExportedHamiltonianProgram is HamiltonianProgram
    assert ExportedResolvedSLH is ResolvedSLH
    assert ExportedSLHChannel is SLHChannel


def test_chip_resolve_exposes_input_free_slh_core() -> None:
    """Ordinary chip resolution exposes baths through the internal SLH view."""
    from quchip import Chip, DuffingTransmon

    qubit = DuffingTransmon(
        freq=5.0,
        anharmonicity=-0.2,
        levels=3,
        T1=100.0,
        label="q",
    )
    result = Chip([qubit]).resolve()

    assert isinstance(result.slh, ResolvedSLH)
    assert result.slh.external_channels == ()
    assert len(result.slh.hidden_channels) == 1


def test_hidden_relaxation_backend_operator_matches_declared_t1() -> None:
    """SLH packaging leaves backend Lindblad lowering and rate unchanged."""
    from quchip import Chip, DuffingTransmon

    lifetime = 40.0
    qubit = DuffingTransmon(
        freq=5.0,
        anharmonicity=-0.2,
        levels=3,
        T1=lifetime,
        label="q",
    )
    chip = Chip([qubit], backend="qutip")
    result = chip.resolve()

    backend_operator = chip.backend._collapse_operators(result)[0].full()
    lowering = np.diag(np.sqrt(np.arange(1, qubit.levels)), k=1).astype(complex)
    np.testing.assert_allclose(backend_operator, np.sqrt(1.0 / lifetime) * lowering)


def test_channel_coupling_rate_is_jittable_and_differentiable() -> None:
    """A traced Lindblad rate reaches the physical L without host concretization."""
    import jax
    import jax.numpy as jnp

    operator = CanonicalOperator.from_dense(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex),
        dims=(2,),
        basis="native",
        subsystem_labels=("q",),
    )

    def coupling_element(rate):
        collapse = CollapseTerm(
            operator=operator,
            rate=rate,
            source="q",
            channel="relaxation",
        )
        slh = ResolvedSLH.from_terms(
            static_terms=(),
            dynamic_terms=(),
            collapse_terms=(collapse,),
        )
        return jnp.real(slh.L[0].to_dense()[0, 1])

    value, gradient = jax.jit(jax.value_and_grad(coupling_element))(jnp.asarray(0.25))

    np.testing.assert_allclose(value, 0.5)
    np.testing.assert_allclose(gradient, 1.0)
