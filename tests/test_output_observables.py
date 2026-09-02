"""Transient output fields derived from the solve-bound SLH model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from quchip import (
    Chip,
    CoherentInput,
    OutputAmplitude,
    OutputPhotonFlux,
    OutputQuadrature,
    PortNetwork,
    QuantumSequence,
    Resonator,
    Square,
)


def _one_port_chip(
    *,
    rate: float = 0.04,
    delay: float = 0.0,
    backend: str = "qutip",
) -> tuple[Chip, Resonator]:
    resonator = Resonator(freq=0.2, levels=4, label="r")
    network = PortNetwork(label="line")
    port = network.port("coupler", target=resonator, rate=rate)
    network.expose(
        "readout",
        input=port.input,
        output=port.output,
        delay=delay,
    )
    return Chip([resonator], port_network=network, frame="lab", backend=backend), resonator


def test_output_observable_specs_are_immutable_and_validate_the_exposure_label() -> None:
    """Output specifications are immutable and require a valid exposure label."""
    amplitude = OutputAmplitude("readout")
    quadrature = OutputQuadrature("readout", phase=0.3)
    flux = OutputPhotonFlux("readout")

    assert amplitude.exposure == quadrature.exposure == flux.exposure == "readout"
    assert quadrature.phase == pytest.approx(0.3)
    with pytest.raises(FrozenInstanceError):
        amplitude.exposure = "other"  # type: ignore[misc]


def test_output_amplitude_is_s_beta_plus_l_expectation() -> None:
    """Output amplitude evaluates the resolved S beta plus L relation."""
    rate = 0.04
    beta = 0.12 + 0.05j
    chip, resonator = _one_port_chip(rate=rate)
    sequence = QuantumSequence(chip)
    sequence.schedule(
        CoherentInput("readout"),
        envelope=Square(duration=2.0, amplitude=abs(beta)),
        phase=np.angle(beta),
    )
    times = np.linspace(0.0, 2.0, 41)

    result = sequence.simulate(
        tlist=times,
        e_ops={
            "field": OutputAmplitude("readout"),
            resonator: resonator.lowering_operator(),
        },
        partition=False,
    )

    lowering = result.observable_traces["r"]
    assert not isinstance(lowering, list)
    expected = beta + np.sqrt(rate) * lowering.raw
    np.testing.assert_allclose(result.expect("field"), expected, atol=2e-8)


def test_output_quadrature_uses_half_sum_convention() -> None:
    """Output quadrature uses the documented half-sum field convention."""
    chip, _ = _one_port_chip()
    phase = 0.37
    sequence = QuantumSequence(chip)
    sequence.schedule(
        CoherentInput("readout"),
        envelope=Square(duration=1.0, amplitude=0.15),
        phase=0.21,
    )

    result = sequence.simulate(
        tlist=np.linspace(0.0, 1.0, 21),
        e_ops={
            "field": OutputAmplitude("readout"),
            "quadrature": OutputQuadrature("readout", phase=phase),
        },
        partition=False,
    )

    np.testing.assert_allclose(
        result.expect("quadrature"),
        np.real(np.exp(-1j * phase) * result.expect("field")),
        atol=2e-8,
    )


def test_output_photon_flux_includes_spontaneous_emission() -> None:
    """Normally ordered output flux includes spontaneous emission from L."""
    rate = 0.04
    chip, resonator = _one_port_chip(rate=rate)
    times = np.linspace(0.0, 4.0, 41)

    result = QuantumSequence(chip).simulate(
        tlist=times,
        initial_state={resonator: 1},
        e_ops={"flux": OutputPhotonFlux("readout")},
        partition=False,
    )

    np.testing.assert_allclose(
        result.expect("flux"),
        rate * np.exp(-rate * times),
        atol=2e-7,
    )


def test_scattering_routes_the_direct_coherent_background() -> None:
    """Output fields include coherent background routed by scattering."""
    left = Resonator(freq=0.2, levels=2, label="left")
    right = Resonator(freq=0.3, levels=2, label="right")
    network = PortNetwork(scattering=[[0.0, 1.0], [1.0, 0.0]], label="swap")
    network.port("left", target=left, rate=0.04)
    network.port("right", target=right, rate=0.09)
    chip = Chip([left, right], port_network=network, frame="lab")
    sequence = QuantumSequence(chip)
    sequence.schedule(
        CoherentInput("left"),
        envelope=Square(duration=0.2, amplitude=0.2),
    )

    result = sequence.simulate(
        tlist=np.linspace(0.0, 0.2, 5),
        e_ops={
            "left_out": OutputAmplitude("left"),
            "right_out": OutputAmplitude("right"),
        },
        partition=False,
    )

    assert result.expect("left_out")[0] == pytest.approx(0.0)
    assert result.expect("right_out")[0] == pytest.approx(0.2)


def test_reference_delay_is_applied_reciprocally_to_emitted_flux() -> None:
    """Reference delay shifts emitted fields outward with the reciprocal convention."""
    rate = 0.04
    delay = 0.2
    chip, resonator = _one_port_chip(rate=rate, delay=delay)
    times = np.linspace(0.0, 1.0, 11)

    result = QuantumSequence(chip).simulate(
        tlist=times,
        initial_state={resonator: 1},
        e_ops={"flux": OutputPhotonFlux("readout")},
        partition=False,
    )

    expected = np.where(
        times < delay,
        0.0,
        rate * np.exp(-rate * (times - delay)),
    )
    np.testing.assert_allclose(result.expect("flux"), expected, atol=2e-7)


def test_reference_delay_applies_on_both_sides_of_direct_scattering() -> None:
    """Reference delay shifts incident and reported coherent fields reciprocally."""
    delay = 0.2
    chip, _ = _one_port_chip(delay=delay)
    sequence = QuantumSequence(chip)
    sequence.schedule(
        CoherentInput("readout"),
        envelope=Square(duration=0.5, amplitude=0.2),
    )
    times = np.linspace(0.0, 0.8, 9)

    result = sequence.simulate(
        tlist=times,
        e_ops={"field": OutputAmplitude("readout")},
        partition=False,
    )

    np.testing.assert_allclose(result.expect("field")[times < 2 * delay], 0.0, atol=2e-8)
    assert result.expect("field")[4] == pytest.approx(0.2, abs=2e-4)


def test_multiple_coherent_inputs_on_one_exposure_add_before_output_evaluation() -> None:
    """Incident fields on one exposure add before output evaluation."""
    chip, _ = _one_port_chip(rate=0.04)
    sequence = QuantumSequence(chip)
    sequence.schedule(CoherentInput("readout"), envelope=Square(duration=1.0, amplitude=0.1))
    sequence.schedule(
        CoherentInput("readout"),
        envelope=Square(duration=1.0, amplitude=0.2),
        phase=np.pi / 2,
    )

    result = sequence.simulate(
        tlist=np.linspace(0.0, 1.0, 11),
        e_ops={
            "field": OutputAmplitude("readout"),
            "flux": OutputPhotonFlux("readout"),
        },
        partition=False,
    )

    assert result.expect("field")[0] == pytest.approx(0.1 + 0.2j)
    assert result.expect("flux")[0] == pytest.approx(0.1**2 + 0.2**2)


def test_unknown_output_exposure_reports_the_available_boundary() -> None:
    """Output evaluation rejects unknown exposures and reports valid choices."""
    chip, _ = _one_port_chip()

    with pytest.raises(ValueError, match="Unknown output exposure.*readout"):
        QuantumSequence(chip).build_problem(
            tlist=np.linspace(0.0, 1.0, 11),
            e_ops={"field": OutputAmplitude("missing")},
        )


def test_output_observables_follow_sequence_batch_axes() -> None:
    """Output observables preserve ordinary sequence batch axes."""
    chip, _ = _one_port_chip()
    sequence = QuantumSequence(chip)
    handle = sequence.schedule(
        CoherentInput("readout"),
        envelope=Square(duration=0.5, amplitude=0.1),
    )

    result = sequence.simulate_batch(
        handle.vary("amplitude", [0.1, 0.2]),
        tlist=np.linspace(0.0, 0.5, 6),
        e_ops={"field": OutputAmplitude("readout")},
        progress=False,
    )

    assert result.expect("field").shape == (2, 6)
    np.testing.assert_allclose(result.expect("field")[:, 0], [0.1, 0.2], atol=2e-8)


def test_qutip_and_dynamiqs_output_observables_agree() -> None:
    """QuTiP and Dynamiqs agree on amplitude and photon-flux traces."""
    pytest.importorskip("dynamiqs")

    def traces(backend: str) -> tuple[np.ndarray, np.ndarray]:
        chip, _ = _one_port_chip(backend=backend)
        sequence = QuantumSequence(chip)
        sequence.schedule(
            CoherentInput("readout"),
            envelope=Square(duration=1.0, amplitude=0.15),
        )
        result = sequence.simulate(
            tlist=np.linspace(0.0, 1.0, 21),
            e_ops={
                "field": OutputAmplitude("readout"),
                "flux": OutputPhotonFlux("readout"),
            },
            partition=False,
        )
        return np.asarray(result.expect("field")), np.asarray(result.expect("flux"))

    qutip = traces("qutip")
    dynamiqs = traces("dynamiqs")
    np.testing.assert_allclose(dynamiqs[0], qutip[0], atol=2e-7)
    np.testing.assert_allclose(dynamiqs[1], qutip[1], atol=2e-7)


def test_dynamiqs_output_quadrature_is_differentiable() -> None:
    """Dynamiqs output quadrature remains differentiable through a solve."""
    pytest.importorskip("dynamiqs")
    import jax
    import jax.numpy as jnp

    chip, _ = _one_port_chip(backend="dynamiqs")

    def final_quadrature(amplitude: object) -> object:
        sequence = QuantumSequence(chip)
        sequence.schedule(
            CoherentInput("readout"),
            envelope=Square(duration=0.5, amplitude=amplitude),
        )
        result = sequence.simulate(
            tlist=jnp.linspace(0.0, 0.5, 6),
            e_ops={"I": OutputQuadrature("readout")},
            partition=False,
        )
        return result.expect("I")[-1]

    value, gradient = jax.jit(jax.value_and_grad(final_quadrature))(jnp.asarray(0.1))
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    assert gradient != 0.0
