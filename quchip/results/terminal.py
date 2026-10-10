"""Projective measurements of saved states and calibrated detector responses."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from quchip.results.receiver import gaussian_samples, validate_samples
from quchip.utils.jax_utils import contains_tracer, select_array_module
from quchip.utils.labeling import resolve_label
from quchip.utils.values import copy_value


def _namespace(*values: Any) -> Any:
    from jax.tree_util import tree_leaves
    from quchip.utils.jax_utils import is_jax_array
    return select_array_module(contains_tracer(values) or any(is_jax_array(v) for v in tree_leaves(values)))


def _probabilities(values: Any) -> Any:
    xp = _namespace(values)
    values = xp.real(xp.asarray(values))
    if not contains_tracer(values):
        if (not np.all(np.isfinite(values)) or np.any(values < -1e-7)
                or not np.allclose(np.sum(values, axis=-1), 1, atol=1e-6)):
            raise ValueError("Measurement requires normalized, nonnegative Born probabilities.")
    values = xp.maximum(values, 0)
    return values / xp.sum(values, axis=-1, keepdims=True)


@dataclass(frozen=True)
class IQReadout:
    """Calibrate one complex IQ distribution per ordered physical outcome.

    means has shape (outcomes,). iq_covariance is (2,2) or (outcomes,2,2) in
    the same signal units squared. They are complete conditional detector
    distributions, so do not add apparatus noise that the calibration already
    includes. Covariances must be positive definite. Wiring-derived readouts
    use fields in 1/sqrt(ns) and keep their integrated noise contributions.
    Means and IQ covariances follow the VNA engineering convention.

    Attributes
    ----------
    means : array_like
        Complex conditional means, one per physical outcome.
    iq_covariance : array_like
        Covariance with shape ``(outcomes, 2, 2)``.
    contributions : mapping or None
        Optional named integrated noise contributions.
    """

    means: Any
    iq_covariance: Any
    contributions: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        xp = _namespace(self.means, self.iq_covariance)
        means = xp.asarray(self.means, dtype=complex)
        covariance = xp.asarray(self.iq_covariance, dtype=float)
        if means.ndim != 1 or len(means) < 1:
            raise ValueError("Readout means must be a nonempty vector ordered by measurement outcomes.")
        if covariance.shape not in ((2, 2), (len(means), 2, 2)):
            raise ValueError("iq_covariance must have shape (2,2) or (outcomes,2,2).")
        if not contains_tracer((means, covariance)):
            if not np.all(np.isfinite(means)) or not np.all(np.isfinite(covariance)):
                raise ValueError("Readout calibration must be finite.")
            if (not np.allclose(covariance, np.swapaxes(covariance, -1, -2), atol=1e-12)
                    or np.any(np.linalg.eigvalsh(covariance) <= 0)):
                raise ValueError("Readout IQ covariance must be symmetric positive definite.")
        object.__setattr__(self, "means", copy_value(means, readonly=True))
        object.__setattr__(self, "iq_covariance", copy_value(
            xp.broadcast_to(covariance, (len(means), 2, 2)), readonly=True))
        object.__setattr__(self, "contributions", MappingProxyType(copy_value(
            dict(self.contributions or {}), readonly=True)))

    @classmethod
    def from_wiring(cls, chip: Any, output: Any, *, means: Any, frequency: Any,
                    receiver: Any, noise_frequencies: Any = None) -> IQReadout:
        """Build an IQ detector from wiring and conditional coherent fields.

        Use the same boundary-field units and vacuum assumptions as
        SimulationResult.iq_readout(). This method captures the chip's current
        wiring, whereas the result method uses the wiring that the simulation
        kept.

        Parameters
        ----------
        chip : Chip
            Chip whose current output wiring is captured.
        output : port object or str
            Output reference plane.
        means : array_like or mapping
            Engineering-convention conditional Markov-boundary fields in
            ``1/sqrt(ns)``. A mapping gives multiple boundary channels.
        frequency : scalar
            Carrier frequency in GHz.
        receiver : IQReceiver
            Boxcar receiver and optional digital transfer.
        noise_frequencies : array_like or None, optional
            Two-sided offsets in GHz. ``None`` uses the standard grid.
        """
        from quchip.analysis.field_noise import ReadoutWiring
        wiring = ReadoutWiring.capture(chip.resolve().slh, chip.backend.array_module)
        return wiring.iq_readout(output, means=means, frequency=frequency, receiver=receiver,
                                 noise_frequencies=noise_frequencies)

    def _weights(self, probabilities: Any) -> Any:
        weights = _probabilities(probabilities)
        if weights.shape[-1] != len(self.means):
            raise ValueError("Readout calibration must contain one distribution per physical outcome.")
        return weights

    def mean(self, probabilities: Any) -> Any:
        """Return the mean of the conditional IQ mixture.

        Parameters
        ----------
        probabilities : array_like
            Physical-outcome probabilities on the final axis.
        """
        return self._weights(probabilities) @ self.means

    def covariance(self, probabilities: Any) -> Any:
        """Return the mixture covariance, with conditional-mean separation.

        Parameters
        ----------
        probabilities : array_like
            Physical-outcome probabilities on the final axis.
        """
        weights = self._weights(probabilities)
        xp = _namespace(weights, self.means)
        centers = xp.stack((xp.real(self.means), xp.imag(self.means)), axis=-1)
        centered = centers - (weights @ centers)[..., None, :]
        return xp.sum(weights[..., :, None, None] * (
            self.iq_covariance + centered[..., :, :, None] * centered[..., :, None, :]), axis=-3)


@dataclass(frozen=True)
class StateSamples:
    """Independent measurement shots, with indices that select the ordered outcome tuples.

    physical_indices are Born draws. indices also includes assignment errors,
    if requested. iq is present only for a conditional IQ readout. Sweep axes
    follow the leading shot axis. No conditional state is returned.

    Attributes
    ----------
    devices : tuple of str
        Measured device labels.
    outcomes : tuple of tuple of int
        Ordered outcome tuples.
    axes : tuple
        Sweep-axis metadata.
    physical_indices, indices : array_like
        Born and recorded outcome indices.
    iq : array_like or None
        Conditional IQ draws, when requested.
    """

    devices: tuple[str, ...]
    outcomes: tuple[tuple[int, ...], ...]
    axes: tuple[tuple[str, Any], ...]
    physical_indices: Any
    indices: Any
    iq: Any = None

    def __post_init__(self) -> None:
        for name in ("axes", "physical_indices", "indices", "iq"):
            object.__setattr__(self, name, copy_value(getattr(self, name), readonly=True))

    def counts(self) -> dict[tuple[int, ...], Any]:
        """Count recorded outcomes along the shot axis, and keep sweep axes."""
        xp = _namespace(self.indices)
        return {outcome: xp.sum(self.indices == i, axis=0) for i, outcome in enumerate(self.outcomes)}


@dataclass(frozen=True)
class StateMeasurement:
    """Born probabilities in a captured local product measurement basis.

    Attributes
    ----------
    devices : tuple of str
        Measured device labels.
    outcomes : tuple of tuple of int
        Ordered product-basis outcomes.
    probabilities : array_like
        Normalized probabilities with outcomes on the final axis.
    axes : tuple
        Sweep-axis metadata.
    """

    devices: tuple[str, ...]
    outcomes: tuple[tuple[int, ...], ...]
    probabilities: Any
    axes: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        probabilities = _probabilities(self.probabilities)
        if probabilities.shape[-1] != len(self.outcomes):
            raise ValueError("Probabilities and outcome labels must have the same length.")
        object.__setattr__(self, "probabilities", copy_value(probabilities, readonly=True))
        object.__setattr__(self, "axes", copy_value(self.axes, readonly=True))

    def _assignment(self, assignment: Any) -> Any:
        xp = _namespace(self.probabilities, assignment)
        matrix = xp.asarray(assignment, dtype=float)
        size = len(self.outcomes)
        if matrix.shape != (size, size):
            raise ValueError("Assignment must have shape (recorded outcomes, physical outcomes).")
        if not contains_tracer(matrix):
            if (not np.all(np.isfinite(matrix)) or np.any(matrix < 0)
                    or not np.allclose(np.sum(matrix, axis=0), 1, atol=1e-8)):
                raise ValueError("Assignment columns must be nonnegative probabilities summing to one.")
        return matrix

    def recorded_probabilities(self, assignment: Any) -> Any:
        """Apply ``A[recorded, physical]`` to the Born probabilities.

        Parameters
        ----------
        assignment : array_like
            Column-stochastic ``(recorded, physical)`` assignment matrix.
        """
        return self.probabilities @ self._assignment(assignment).T

    def sample(self, count: int, *, seed: int | None = None, key: Any = None,
               assignment: Any = None, readout: IQReadout | None = None) -> StateSamples:
        """Draw Born outcomes, and optionally record assignment errors or conditional IQ.

        Assignment and IQ are separate detector models. IQ samples stay
        unclassified, and quchip does not infer a threshold or assignment
        matrix from them. Discrete labels are not differentiable. Use an
        explicit key under JAX.

        Parameters
        ----------
        count : int
            Positive number of independent shots.
        seed : int or None, optional
            NumPy random seed.
        key : jax.Array or None, optional
            JAX random key. Do not use with ``seed``.
        assignment : array_like or None, optional
            Column-stochastic ``(recorded, physical)`` assignment matrix.
        readout : IQReadout or None, optional
            Conditional IQ calibration for each physical outcome.
        """
        if assignment is not None and readout is not None:
            raise ValueError("Choose assignment or IQ readout; IQ samples are not automatically classified.")
        xp = validate_samples(count, seed, key, (self.probabilities, assignment,
                              None if readout is None else (readout.means, readout.iq_covariance)))
        shape = (count, *self.probabilities.shape[:-1])
        if key is None:
            rng = np.random.default_rng(seed)
            uniforms = xp.asarray(rng.uniform(size=(2, *shape)))
            noise_key = None
            noise_seed = int(rng.integers(0, 2**32))
        else:
            import jax.random
            choice_key, noise_key = jax.random.split(key)
            uniforms = jax.random.uniform(choice_key, (2, *shape))
            noise_seed = None
        physical = xp.sum(uniforms[0, ..., None] >= xp.cumsum(xp.asarray(self.probabilities), axis=-1), axis=-1)
        physical = xp.minimum(physical, len(self.outcomes) - 1)
        recorded, iq = physical, None
        if assignment is not None:
            conditional = xp.asarray(self._assignment(assignment)).T[physical]
            recorded = xp.sum(uniforms[1, ..., None] >= xp.cumsum(conditional, axis=-1), axis=-1)
            recorded = xp.minimum(recorded, len(self.outcomes) - 1)
        if readout is not None:
            readout._weights(self.probabilities)
            means = xp.asarray(readout.means)[physical]
            centers = xp.stack((xp.real(means), xp.imag(means)), axis=-1)
            draws = gaussian_samples(centers, xp.asarray(readout.iq_covariance)[physical],
                                     1, seed=noise_seed, key=noise_key)[0]
            iq = draws[..., 0] + 1j * draws[..., 1]
        return StateSamples(self.devices, self.outcomes, self.axes, physical, recorded, iq)


def measure_result(result: Any, devices: tuple[Any, ...], *, t: Any = None,
                   basis: Any = "energy") -> StateMeasurement:
    """Project kept states with captured basis maps and existing result coordinates."""
    from quchip.results.results import SimulationBatchResult
    from quchip.results.partitioned import PartitionedSimulationResult

    labels = tuple(resolve_label(device) for device in devices)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Measure one or more distinct devices, in the requested outcome order.")
    if isinstance(basis, Mapping):
        basis = {resolve_label(label): value for label, value in basis.items()}
        unknown = set(basis) - set(labels)
        if unknown:
            raise ValueError(f"Measurement basis names unmeasured devices: {sorted(unknown)}.")
    elif not isinstance(basis, str) and len(labels) != 1:
        raise ValueError("Joint custom bases require a device-to-unitary mapping.")
    if isinstance(result, SimulationBatchResult):
        points = [measure_result(point, devices, t=t, basis=basis) for point in result]
        if not points:
            raise ValueError("Cannot measure an empty batch.")
        if any(point.outcomes != points[0].outcomes for point in points):
            raise ValueError("Batch measurements require matching outcome dimensions.")
        xp = _namespace(*(point.probabilities for point in points))
        probabilities = xp.stack([point.probabilities for point in points])
        return StateMeasurement(labels, points[0].outcomes,
                                   probabilities.reshape((*result.shape, len(points[0].outcomes))), result.axes)
    if isinstance(result, PartitionedSimulationResult):
        groups: dict[int, list[str]] = {}
        for label in labels:
            groups.setdefault(result.partition.owner_of(label), []).append(label)
        points = [measure_result(result.components[index], tuple(group), t=t,
                                 basis={k: v for k, v in basis.items() if k in group}
                                 if isinstance(basis, Mapping) else basis)
                  for index, group in groups.items()]
        if len(points) == 1:
            return points[0]
        xp = _namespace(*(point.probabilities for point in points))
        dimensions = {label: max(outcome[i] for outcome in point.outcomes)+1
                      for point in points for i, label in enumerate(point.devices)}
        outcomes = tuple(product(*(range(dimensions[label]) for label in labels)))
        probabilities = xp.ones(len(outcomes))
        for point in points:
            digits = np.asarray(outcomes)[:, [labels.index(label) for label in point.devices]].T
            indices = np.ravel_multi_index(tuple(digits), tuple(dimensions[label] for label in point.devices))
            probabilities = probabilities * point.probabilities[xp.asarray(indices)]
        return StateMeasurement(labels, outcomes, probabilities)
    backend = result._backend
    selected = [result._resolve_device_idx(label)[0] for label in labels]
    state = result.state(t)
    array = backend.to_array(state)
    xp = _namespace(array, basis)
    ket = backend.is_ket(state)
    dims = tuple(result.dims)
    tensor = xp.asarray(array).reshape(dims if ket else dims + dims)
    if isinstance(basis, str):
        if basis not in ("energy", "solver"):
            raise ValueError("basis must be 'energy', 'solver', or local unitary columns in the energy basis.")
        custom: Mapping[str, Any] = {}
    else:
        custom = basis if isinstance(basis, Mapping) else {labels[0]: basis}
    for label, axis in zip(labels, selected):
        size = dims[axis]
        rotation = xp.asarray(custom.get(label, xp.eye(size)), dtype=complex)
        if rotation.shape != (size, size):
            raise ValueError(f"Measurement basis for {label!r} must have shape {(size, size)}.")
        if not contains_tracer(rotation) and not np.allclose(rotation.conj().T @ rotation, np.eye(size), atol=1e-8):
            raise ValueError("Measurement basis columns must be orthonormal.")
        if not isinstance(basis, str) or basis == "energy":
            captured = result._bases.get(label)
            if captured is None:
                raise ValueError("Energy measurement requires captured basis information; use basis='solver'.")
            energy = captured.energy_to_solver()
            if energy is not None:
                rotation = xp.asarray(energy) @ rotation
        tensor = xp.moveaxis(xp.tensordot(rotation.conj().T, tensor, axes=(1, axis)), 0, axis)
        if not ket:
            tensor = xp.moveaxis(xp.tensordot(rotation.T, tensor, axes=(1, axis+len(dims))), 0, axis+len(dims))
    diagonal = xp.abs(tensor)**2 if ket else xp.real(xp.diag(tensor.reshape((int(np.prod(dims)),)*2))).reshape(dims)
    remaining = [i for i in range(len(dims)) if i not in selected]
    probabilities = xp.transpose(diagonal, selected + remaining).reshape(
        (int(np.prod([dims[i] for i in selected])), -1)).sum(axis=-1)
    outcomes = tuple(product(*(range(dims[i]) for i in selected)))
    return StateMeasurement(labels, outcomes, probabilities)
