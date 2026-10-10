"""Dressed-state analysis owned by :class:`quchip.chip.chip.Chip`.

``ChipAnalysis`` diagonalizes the lab-frame static Hamiltonian and assigns
bare-state labels to dressed eigenvectors by overlap maximization. It also
exposes derived quantities: dressed eigenenergies, transition frequencies,
dispersive shifts, and effective subspace Hamiltonians. All results cache
against a structural signature and refresh automatically when a device or
coupling parameter changes.

Dressing is always computed in the lab frame, so the frame selection never
changes dressed data.

References
----------
Gambetta, J., Blais, A., Schuster, D. I., Wallraff, A., Frunzio, L.,
    Majer, J., Devoret, M. H., Girvin, S. M., & Schoelkopf, R. J.
    Qubit-photon interactions in a cavity: Measurement-induced dephasing
    and number splitting. PRA 74, 042318 (2006).
Koch, J., Yu, T. M., Gambetta, J., Houck, A. A., Schuster, D. I., Majer,
    J., Blais, A., Devoret, M. H., Girvin, S. M., & Schoelkopf, R. J.
    Charge-insensitive qubit design derived from the Cooper pair box.
    PRA 76, 042319 (2007).
Blais, A., Grimsmo, A. L., Girvin, S. M., & Wallraff, A. Circuit quantum
    electrodynamics. Rev. Mod. Phys. 93, 025005 (2021), §IV on dispersive
    regime and dressed-state labeling.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from quchip.backend import EigensystemData, Operator, State, _backend_context
from quchip.chip.dressing import (
    Labeling,
    _reference_amplitudes,
    assign_rowwise_greedy,
    label_eigensystem,
)
from quchip.chip.states import _bare_state_from_bases, normalize_device_state_mapping
from quchip.devices.base import BaseDevice, _validate_level_pair
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import contains_tracer, maybe_concrete_scalar
from quchip.utils.values import TracedKey, scoped_entry, scoped_hit
from quchip.utils.labeling import LabelKeyedDict, bare_label_from_mapping, resolve_label, top_components

if TYPE_CHECKING:
    from quchip.approximations import Approximation
    from quchip.chip.chip import Chip
    from quchip.control.drive import BaseDrive
    from quchip.engine.ir import EngineResult
    from quchip.engine.sectors import SectorModel


_STATE_OVERLAP_WARNING = 0.9
_DRESS_TRACING_ERROR = (
    "Dressing returns a concrete dict-keyed DressedResult and is not "
    "traceable under jax.jit/grad/vmap. Use Chip.energy(), Chip.freq(), "
    "Chip.dispersive_shift(), or Chip.state() inside transforms — they "
    "route through the array-only kernel in quchip.chip.dressing and stay "
    "differentiable."
)


def _phase_fixed_state(state: Any, anchor: Any, xp: Any) -> Any:
    """Set one dressed vector's assigned bare overlap to a nonnegative real value."""
    magnitude = xp.abs(anchor)
    threshold = xp.finfo(magnitude.dtype).eps
    safe_magnitude = xp.where(magnitude > threshold, magnitude, xp.asarray(1.0, dtype=magnitude.dtype))
    phase = xp.where(
        magnitude > threshold,
        xp.conj(anchor) / safe_magnitude,
        xp.asarray(1.0 + 0.0j, dtype=state.dtype),
    )
    return state * phase


@dataclass
class DressedResult:
    """Store a frozen snapshot of a dressed-state diagonalization.

    Attributes
    ----------
    eigenvalues : array-like
        Sorted dressed eigenvalues in GHz.
    eigenstates : array-like
        Backend eigenstate objects in the same order as ``eigenvalues``. They
        are materialized lazily on first access from the underlying
        :class:`~quchip.backend.containers.EigensystemData`. The dressing and
        sweep hot path never touches them, so the backend builds per-column
        kets only on request.
    state_map : dict[tuple[int, ...], int]
        Map from each bare product-basis label (one int per device) to the
        dressed eigenstate index with the largest overlap.
    dressed_eigenvalues : dict[tuple[int, ...], Any]
        Dressed eigenvalue for each assigned bare label, the direct lookup path
        for :meth:`Chip.energy`.
    assignment_overlaps : dict[tuple[int, ...], float]
        ``|⟨bare|dressed⟩|²`` of each assignment. Values below
        ``overlap_threshold`` flag hybridization.
    hybridized_labels : tuple[tuple[int, ...], ...]
        Bare labels whose assignment quality is below ``overlap_threshold``. A
        non-empty tuple triggers a user warning at dress time.
    bare_labels : tuple[tuple[int, ...], ...]
        Labels of the product energy levels exposed by the chip's resolved
        local dimensions, in chip order.
    bare_labels_by_dressed_index : dict[int, tuple[int, ...]]
        Inverse of :attr:`state_map`: dressed index → assigned bare label.
    eigenvector_matrix : array-like or None
        Dressed eigenvectors as columns, in the resolved solver basis, used by
        :meth:`ChipAnalysis.operator_in_dressed_basis` and
        :meth:`ChipAnalysis.state_components`.
    overlap_threshold : float
        Minimum overlap for a confident assignment.
    labeling : str
        Algorithm id (currently only ``"DE"``). It resolves to the confidence-ordered
        row-greedy overlap-matching policy run by :meth:`ChipAnalysis.dress`
        (:func:`~quchip.chip.dressing.assign_rowwise_greedy`).
    """

    eigenvalues: Any
    state_map: dict[tuple[int, ...], int]
    dressed_eigenvalues: dict[tuple[int, ...], Any]
    assignment_overlaps: dict[tuple[int, ...], float]
    hybridized_labels: tuple[tuple[int, ...], ...]
    bare_labels: tuple[tuple[int, ...], ...]
    bare_labels_by_dressed_index: dict[int, tuple[int, ...]]
    eigenvector_matrix: Any = None
    overlap_threshold: float = 0.5
    labeling: str = "DE"
    _eigensystem: EigensystemData | None = None

    @property
    def eigenstates(self) -> Any:
        """Backend eigenstate kets, materialized lazily from the eigensystem."""
        if self._eigensystem is None:
            raise RuntimeError(
                "DressedResult was constructed without an EigensystemData; "
                "eigenstates are unavailable."
            )
        return self._eigensystem.eigenstates


def _materialize_dressed_result(
    eigenvalues: Any,
    eigenvector_matrix: Any,
    eigensystem: EigensystemData,
    kernel_labeling: Labeling,
    *,
    overlap_threshold: float,
    labeling: str,
    warning_stacklevel: int,
) -> DressedResult:
    """Build the eager label-keyed result shared by chip and resolved dressing."""
    if contains_tracer((kernel_labeling.indices, kernel_labeling.overlaps)):
        raise RuntimeError(_DRESS_TRACING_ERROR)

    bare_labels = kernel_labeling.keys
    indices_np = np.asarray(kernel_labeling.indices)
    overlaps_np = np.asarray(kernel_labeling.overlaps)
    state_map: dict[tuple[int, ...], int] = {}
    bare_labels_by_dressed_index: dict[int, tuple[int, ...]] = {}
    assignment_overlaps: dict[tuple[int, ...], float] = {}
    dressed_eigenvalues: dict[tuple[int, ...], Any] = {}
    for k, bare_label in enumerate(bare_labels):
        index = int(indices_np[k])
        state_map[bare_label] = index
        bare_labels_by_dressed_index[index] = bare_label
        assignment_overlaps[bare_label] = float(overlaps_np[k])
        dressed_eigenvalues[bare_label] = eigenvalues[index]

    hybridized_labels = tuple(
        bare_label
        for bare_label in bare_labels
        if assignment_overlaps[bare_label] < overlap_threshold
    )
    if hybridized_labels:
        # Fixed text, so the default warning filter shows it once per call
        # site instead of once per sweep point.
        warnings.warn(
            "Strong hybridization detected during dressed-state assignment; "
            f"bare labels with overlap below {overlap_threshold} are approximate. "
            "Inspect DressedResult.hybridized_labels and "
            "DressedResult.assignment_overlaps for the affected states.",
            UserWarning,
            stacklevel=warning_stacklevel,
        )

    return DressedResult(
        eigenvalues=eigenvalues,
        state_map=state_map,
        dressed_eigenvalues=dressed_eigenvalues,
        assignment_overlaps=assignment_overlaps,
        hybridized_labels=hybridized_labels,
        bare_labels=bare_labels,
        bare_labels_by_dressed_index=bare_labels_by_dressed_index,
        eigenvector_matrix=eigenvector_matrix,
        overlap_threshold=float(overlap_threshold),
        labeling=labeling,
        _eigensystem=eigensystem,
    )


def dress_engine_result(
    result: "EngineResult",
    *,
    at_time: Any | None = None,
    overlap_threshold: float = 0.5,
    labeling: str = "DE",
) -> DressedResult:
    """Materialize the instantaneous dressed eigensystem of an engine snapshot."""
    if labeling != "DE":
        raise ValueError(f"Unsupported labeling {labeling!r}. Only 'DE' is implemented.")
    if result._dressing_context is None:
        raise RuntimeError(
            "EngineResult has no resolved dressing context; obtain it from Chip.resolve() "
            "or a built solve problem."
        )
    if result.dynamic_terms and at_time is None:
        raise ValueError(
            "A dynamic EngineResult requires dress(at_time=...); this is an "
            "instantaneous eigensystem, not a Floquet analysis."
        )
    context = result._dressing_context
    backend = context.backend
    hamiltonian = result.hamiltonian().matrix(t=at_time, backend=backend)
    if contains_tracer((hamiltonian, context.reference.local_vectors)):
        raise RuntimeError(_DRESS_TRACING_ERROR)
    native_hamiltonian = backend.from_array(
        hamiltonian,
        dims=[list(result.dims), list(result.dims)],
    )
    eigensystem = backend.eigensystem_data(native_hamiltonian)
    eigenvalues = eigensystem.eigenvalues
    eigenvector_matrix = eigensystem.eigenvector_matrix
    kernel_labeling = label_eigensystem(
        eigenvector_matrix,
        context.reference,
        policy=assign_rowwise_greedy,
    )
    return _materialize_dressed_result(
        eigenvalues,
        eigenvector_matrix,
        eigensystem,
        kernel_labeling,
        overlap_threshold=float(overlap_threshold),
        labeling=labeling,
        warning_stacklevel=4,
    )


def _labeled_eigensystem(engine_result: "EngineResult", backend: Any) -> tuple[Any, Any, Any, Labeling]:
    """Diagonalize a static analysis result and assign its bare product labels.

    Returns ``(eigenvalues, eigenvector_matrix, eigensystem, labeling)``;
    traced Hamiltonians stay traced.
    """
    from quchip.engine.assembly import _analysis_matrix_ghz

    context = engine_result._dressing_context
    if context is None:
        raise RuntimeError("Resolved analysis is missing its captured dressing reference.")
    dims = list(engine_result.dims)
    eigensystem = backend.eigensystem_data(
        backend.from_array(_analysis_matrix_ghz(engine_result), dims=[dims, dims])
    )
    labeling = label_eigensystem(eigensystem.eigenvector_matrix, context.reference, policy=assign_rowwise_greedy)
    return eigensystem.eigenvalues, eigensystem.eigenvector_matrix, eigensystem, labeling


def _is_eigenstate(state: State, result: "EngineResult", backend: Any) -> bool:
    """Return whether a concrete static result maps ``state`` onto a multiple of itself.

    The residual tolerance is ``1e-12`` times the Hamiltonian's angular
    spectral span.
    """
    operators = [term.coefficient * backend.from_canonical_operator(term.operator) for term in result.static_terms]
    image = sum(operators[1:], start=operators[0]) @ state
    residual = float(backend.norm(image - backend.overlap(state, image) * state))
    return residual <= 1e-12 * TWO_PI * result.metadata["spectral_bound_ghz"]


@dataclass(frozen=True)
class KerrMatrix:
    """Labeled dressed self-Kerr and cross-Kerr coefficients in GHz.

    ``labels`` follows chip device order, and ``values`` is a real symmetric
    square array. Diagonal entries are dressed anharmonicities, and
    off-diagonal entries are full-pull cross-Kerr shifts.

    Parameters
    ----------
    labels : tuple[str, ...]
        Device labels in chip order.
    values : array-like
        Real symmetric Kerr matrix in GHz.
    """

    labels: tuple[str, ...]
    values: Any

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        if not all(isinstance(label, str) for label in labels):
            raise TypeError("KerrMatrix labels must be strings.")
        if len(set(labels)) != len(labels):
            raise ValueError(f"KerrMatrix labels must be unique, got {labels!r}.")

        values = jnp.asarray(self.values)
        expected_shape = (len(labels), len(labels))
        if values.shape != expected_shape:
            raise ValueError(f"KerrMatrix values shape must be {expected_shape}, got {values.shape}.")
        if jnp.iscomplexobj(values):
            raise ValueError("KerrMatrix values must be real.")
        symmetric = maybe_concrete_scalar(
            jnp.allclose(values, values.T, rtol=0.0, atol=0.0, equal_nan=True)
        )
        if symmetric is not None and not bool(symmetric):
            raise ValueError("KerrMatrix values must be symmetric.")

        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "values", values)

    def _index(self, device: str | BaseDevice) -> int:
        label = resolve_label(device)
        try:
            return self.labels.index(label)
        except ValueError:
            raise KeyError(f"Unknown KerrMatrix label {label!r}. Available labels: {self.labels}.") from None

    def __getitem__(self, key: tuple[str | BaseDevice, str | BaseDevice]) -> Any:
        """Return one coefficient by device object or label on each axis."""
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("KerrMatrix lookup requires two devices: matrix[a, b].")
        row, column = key
        return self.values[self._index(row), self._index(column)]


jtu.register_pytree_node(
    KerrMatrix,
    lambda matrix: ((matrix.values,), matrix.labels),
    lambda labels, children: KerrMatrix(labels=labels, values=children[0]),
)


def kerr_entry(
    index_a: int,
    index_b: int,
    *,
    dims: tuple[int, ...],
    eigenvalues: Any,
    labeling: Labeling,
) -> Any:
    """Read one self- or cross-Kerr coefficient from a captured labeled spectrum."""

    def energy(label: tuple[int, ...]) -> Any:
        eigen_index = labeling.indices[np.ravel_multi_index(label, dims)]
        values = jnp.asarray(eigenvalues) if contains_tracer(eigen_index) else eigenvalues
        return values[eigen_index]

    return _kerr_coefficient(index_a, index_b, dims, energy)


def _kerr_coefficient(
    index_a: int,
    index_b: int,
    dims: tuple[int, ...],
    energy: Callable[[tuple[int, ...]], Any],
) -> Any:
    """Read one self- or cross-Kerr coefficient through a labeled dressed-energy lookup."""
    n_devices = len(dims)
    if not 0 <= index_a < n_devices or not 0 <= index_b < n_devices:
        raise IndexError(f"Kerr matrix indices must be in [0, {n_devices}), got {(index_a, index_b)}.")

    def excited(*excitations: tuple[int, int]) -> Any:
        label = [0] * n_devices
        for index, level in excitations:
            label[index] = level
        return energy(tuple(label))

    e0 = excited()
    if index_a == index_b:
        if dims[index_a] < 3:
            return jnp.asarray(jnp.nan, dtype=jnp.real(jnp.asarray(e0)).dtype)
        return excited((index_a, 2)) - 2.0 * excited((index_a, 1)) + e0
    return excited((index_a, 1), (index_b, 1)) - excited((index_a, 1)) - excited((index_b, 1)) + e0


class ChipAnalysis:
    """Dressed-state analysis, caching, and dressed-basis helpers.

    Every :class:`~quchip.chip.chip.Chip` owns one ``ChipAnalysis``, and the
    chip forwards common dressed quantities. The namespace exposes the frozen
    static contract via :meth:`engine_result` and keeps less-common analysis
    methods grouped.

    Caching: :meth:`dress` keys its cache on a structural signature covering
    the backend identity and the ``state_version`` of every device and
    coupling. Any change that increments a version invalidates the cache on the
    next access.

    Parameters
    ----------
    chip : Chip
        Chip whose static lab-frame Hamiltonian is analyzed under its approximation.
    """
    # The chip stores its `ChipAnalysis` as `chip._analysis`.

    def __init__(self, chip: "Chip") -> None:
        self._chip = chip
        self._dressed_result: DressedResult | None = None
        self._dressed_signature: tuple[Any, ...] | None = None
        # Cache entries are (key, trace scope, value); see quchip.utils.values.scoped_entry.
        self._array_cache: tuple[Any, Any, tuple[Any, Any, Any, Labeling]] | None = None
        self._sector_cache: tuple[Any, Any, SectorModel | None] | None = None
        self._engine_result_cache: tuple[Any, Any, EngineResult] | None = None
        self._ground_cache: tuple[Any, Any, Any] | None = None
        self._bare_labels_cache: tuple[
            tuple[tuple[int, ...], ...], dict[tuple[int, ...], int]
        ] | None = None
        self._bare_labels_signature: tuple[int, ...] | None = None

    def _analysis_signature(self) -> tuple[Any, ...]:
        """Hashable fingerprint covering every structural input to dressing.

        Retained effective terms, ports and the port network enter because they
        add coherent terms to the dressed Hamiltonian. Traced values are keyed by identity, so their
        results are reused only inside the trace that produced them.
        """
        from quchip.chip.chip import _operator_cache_value
        from quchip.declarative.parameters import component_fingerprint

        chip = self._chip

        def scalar(value: Any) -> Any:
            concrete = maybe_concrete_scalar(value)
            return TracedKey(id(value)) if concrete is None else concrete

        def operator(value: Any) -> Any:
            try:
                return _operator_cache_value(value)
            except ValueError:
                return TracedKey(id(value))

        def terms_key(terms: Any) -> Any:
            try:
                return terms.fingerprint()
            except ValueError:
                return TracedKey(id(terms))

        network = chip.port_network
        try:
            network_key: Any = None if network is None else network.fingerprint()
        except ValueError:
            network_key = TracedKey(id(network))
        return (
            f"{type(chip.backend).__module__}.{type(chip.backend).__qualname__}",
            chip.basis,
            chip.approximation,
            tuple(component_fingerprint(device, traced=True) for device in chip.devices),
            tuple(
                (
                    f"{type(coupling).__module__}.{type(coupling).__qualname__}",
                    coupling.device_a_label,
                    coupling.device_b_label,
                    component_fingerprint(coupling, traced=True),
                )
                for coupling in chip.couplings
            ),
            tuple(terms_key(terms) for terms in chip.effective_terms),
            tuple(
                (
                    port.label,
                    tuple(port.resolve_targets(chip)),
                    tuple((name, scalar(value)) for name, value in port.parameter_values().items()),
                    operator(port.operator),
                )
                for port in chip.ports
            ),
            network_key,
        )

    def engine_result(self, *, _local_resolution: Any | None = None) -> EngineResult:
        """Return the resolved static lab-frame contract used by analysis."""
        from quchip.engine.assembly import _build_static_analysis_result

        signature = self._analysis_signature()
        cache = self._engine_result_cache
        if scoped_hit(cache, signature):
            return cache[2]

        result = _build_static_analysis_result(
            self._chip,
            approximation=self._chip.approximation,
            _local_resolution=_local_resolution,
        )
        self._engine_result_cache = scoped_entry(signature, result, traced=result._contains_tracer())
        return result

    def _semantic_amplitudes(self, eigenvectors: Any, engine_result: EngineResult) -> Any:
        context = engine_result._dressing_context
        if context is None:
            raise RuntimeError("Resolved analysis is missing its captured dressing reference.")
        return _reference_amplitudes(context.reference, eigenvectors)

    def _canonical_bare_labels(self) -> tuple[tuple[int, ...], ...]:
        """Product energy-level labels in chip order."""
        return self._bare_labels_with_index()[0]

    def _semantic_dims(self) -> tuple[int, ...]:
        """Per-device energy-level dimensions exposed by this chip."""
        return tuple(
            device.resolved_dimension(self._chip.basis)
            for device in self._chip.devices
        )

    def _bare_labels_with_index(
        self,
    ) -> tuple[tuple[tuple[int, ...], ...], dict[tuple[int, ...], int]]:
        """Cached ``(level_labels, label → index)`` pair, keyed on semantic dimensions."""
        sig = self._semantic_dims()
        if self._bare_labels_cache is None or self._bare_labels_signature != sig:
            labels = tuple(itertools.product(*(range(d) for d in sig)))
            index_map = {label: idx for idx, label in enumerate(labels)}
            self._bare_labels_cache = (labels, index_map)
            self._bare_labels_signature = sig
        return self._bare_labels_cache

    def _state_label_from_mapping(
        self,
        device_states: Mapping[str | BaseDevice, int] | str | None = None,
        /,
        **device_state_kwargs: int,
    ) -> tuple[int, ...]:
        """Normalize a state mapping or shorthand into a validated chip-ordered label;
        unspecified devices default to level zero."""
        resolved = normalize_device_state_mapping(self._chip, device_states, device_state_kwargs)
        return self._label_from_resolved(resolved)

    def _label_from_resolved(self, resolved: Mapping[str, int]) -> tuple[int, ...]:
        """Validate integer types (excluding bool), nonnegative semantic-level bounds
        and device names, then build the chip-ordered label with unspecified levels zero."""
        semantic_dims = dict(zip(self._device_labels(), self._semantic_dims()))
        for device_label, value in resolved.items():
            _, device = self._chip._resolve_device_index(device_label)
            if isinstance(value, bool):
                raise ValueError(f"Level index for '{device.label}' must be an integer, got bool: {value!r}")
            if not isinstance(value, int):
                raise TypeError(f"Expected integer level index for '{device.label}', got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"Level index for '{device.label}' must be >= 0, got {value}")
            if value >= semantic_dims[device.label]:
                raise ValueError(
                    f"Level index {value} for '{device.label}' exceeds device dimension "
                    f"({semantic_dims[device.label]} semantic levels)"
                )
        return bare_label_from_mapping(self._device_labels(), resolved, {})

    def _label_from_plain_mapping(self, device_states: Mapping[Any, Any]) -> tuple[int, ...]:
        """Build a lookup label after validating device names only."""
        for device_label in device_states:
            self._chip._resolve_device_index(device_label)
        return bare_label_from_mapping(self._device_labels(), device_states, {})

    def _device_labels(self) -> tuple[str, ...]:
        """Chip device labels in tensor-product order."""
        return tuple(device.label for device in self._chip.devices)

    def _compute_array_labeled(
        self,
        engine_result: EngineResult | None = None,
    ) -> tuple[Any, Any, Any, Labeling]:
        """Return the array eigensystem and labeling; traced results are reused only inside their trace."""
        chip = self._chip
        signature = self._analysis_signature()
        if scoped_hit(self._array_cache, signature):
            return self._array_cache[2]

        if engine_result is None:
            engine_result = self.engine_result()
        else:
            self._engine_result_cache = scoped_entry(signature, engine_result, traced=engine_result._contains_tracer())
        result = _labeled_eigensystem(engine_result, chip.backend)
        # The 3rd slot carries the EigensystemData (lazy eigenstates) rather than
        # a materialized ket list — nothing on the hot path reads it. The cache
        # tracer-check covers only slots (0, 1, 3); touching slot 2 would force
        # the lazy ``eigenstates`` property and defeat the deferral.
        eigenvalues, vectors, _, labeling = result
        traced = contains_tracer((eigenvalues, vectors, labeling.indices, labeling.overlaps))
        self._array_cache = scoped_entry(signature, result, traced=traced)
        return result

    def _sector_model(self, *, _local_resolution: Any | None = None) -> SectorModel | None:
        """Return the chip's excitation-sector model, or ``None`` when its static model may change ``N``.

        Labeled dressed energies of a conserving chip come from the sector of
        each label. Results cache like the full eigensystem.
        """
        from quchip.chip.effective import conserves_excitation_number
        from quchip.engine.sectors import SectorModel

        signature = self._analysis_signature()
        if scoped_hit(self._sector_cache, signature):
            return self._sector_cache[2]
        chip = self._chip
        model = (
            SectorModel(chip, resolution=_local_resolution)
            if conserves_excitation_number(chip, chip.approximation)
            else None
        )
        self._sector_cache = scoped_entry(signature, model, traced=model is not None and model.traced)
        return model

    def _is_bare_label(self, label: tuple[int, ...]) -> bool:
        """Return whether ``label`` gives every device a resolved level."""
        dims = self._semantic_dims()
        return len(label) == len(dims) and all(0 <= level < dim for level, dim in zip(label, dims))

    def _sector_label(self, label: tuple[int, ...]) -> tuple[int, ...]:
        """Validate a bare label without enumerating the product basis."""
        if not self._is_bare_label(label):
            available = list(itertools.islice(itertools.product(*(range(d) for d in self._semantic_dims())), 10))
            raise KeyError(
                f"State label {label} is not a valid bare product-basis label. Available (first 10): {available}"
            )
        return tuple(label)

    def _labeled_lookup(
        self, *, _local_resolution: Any | None = None,
    ) -> Callable[[tuple[int, ...]], tuple[Any, Any, Any]]:
        """Return a map from a bare label to its dressed energy in GHz, assignment overlap and margin.

        A chip that conserves the total energy-level index diagonalizes only the
        sector of each requested label. Other chips diagonalize their complete
        static model once.
        """
        model = self._sector_model(_local_resolution=_local_resolution)
        if model is not None:
            def sector_entry(label: tuple[int, ...]) -> tuple[Any, Any, Any]:
                label = self._sector_label(label)
                system = model.eigensystem(sum(label))
                row = system.space.row(label)
                return system.energy(label), system.labeling.overlaps[row], system.labeling.margins[row]

            return sector_entry
        engine_result = self.engine_result(_local_resolution=_local_resolution)
        eigenvalues, _, _, labeling = self._compute_array_labeled(engine_result)

        def full_entry(label: tuple[int, ...]) -> tuple[Any, Any, Any]:
            row = self._bare_label_index(label)
            energy = self._eigenvalue_of_label(label, precomputed=(eigenvalues, labeling))
            return energy, labeling.overlaps[row], labeling.margins[row]

        return full_entry

    def _energy_lookup(self) -> Callable[[tuple[int, ...]], Any]:
        """Return a map from a bare label to its dressed energy in GHz."""
        lookup = self._labeled_lookup()
        return lambda label: lookup(label)[0]

    def _dressed_dimension(self, total: int) -> int:
        """Return the largest matrix that dressed queries up to total level index ``total`` diagonalize.

        Parameters
        ----------
        total : int
            Largest total energy-level index of the queried labels.
        """
        from quchip.chip.effective import conserves_excitation_number
        from quchip.engine.sectors import sector_size

        chip = self._chip
        if not conserves_excitation_number(chip, chip.approximation):
            return chip.total_dim
        dims = self._semantic_dims()
        return max(sector_size(dims, level_sum) for level_sum in range(total + 1))

    def _ground_ket(self, approximation: "Approximation") -> Any | None:
        """Return the all-ground-labeled lab-frame eigenstate for ``approximation``.

        The ket uses solver coordinates and has real, nonnegative overlap on
        the bare product. ``None`` means the bare product is already an
        eigenstate. Results cache against the analysis signature and
        ``approximation``; traced ones only inside their trace. A traced ket is differentiated by first-order
        perturbation theory over the same eigensystem, masking gaps at the
        tolerance of :func:`~quchip.engine.basis._differentiable_eigenpairs`.
        """
        from quchip.engine.assembly import _analysis_matrix_ghz, _build_static_analysis_result
        from quchip.engine.basis import _differentiable_eigenvector

        key = (self._analysis_signature(), approximation)
        if scoped_hit(self._ground_cache, key):
            return self._ground_cache[2]
        chip = self._chip
        backend = chip.backend
        # Reuse dressed analysis only when it describes the solve's approximation.
        same_approximation = approximation == chip.approximation
        static = (
            self.engine_result() if same_approximation
            else _build_static_analysis_result(chip, approximation=approximation)
        )
        bare = _bare_state_from_bases(chip, {}, static.bases)
        traced = static._contains_tracer()
        ket = None
        if traced or not _is_eigenstate(bare, static, backend):
            values, vectors, _, labeling = (
                self._compute_array_labeled(static) if same_approximation else _labeled_eigensystem(static, backend)
            )
            xp = backend.array_module
            index = labeling.indices[0]  # Label 0 is the all-ground product.
            column = xp.asarray(vectors)[:, index]
            if traced:
                # JAX's eigh derivative can be NaN at an unrelated exact tie.
                matrix = jnp.asarray(_analysis_matrix_ghz(static))
                column = _differentiable_eigenvector(matrix, values[index], values, vectors, column)
            ket = _phase_fixed_state(column, xp.vdot(xp.asarray(backend.to_array(bare)).reshape(-1), column), xp)
        self._ground_cache = scoped_entry(key, ket, traced=contains_tracer(ket))
        return ket

    def _bare_label_index(self, label: tuple[int, ...]) -> int:
        """Position of ``label`` in canonical bare-label order (Python int)."""
        bare_labels, index_map = self._bare_labels_with_index()
        try:
            return index_map[label]
        except KeyError:
            available = list(bare_labels)[:10]
            raise KeyError(
                f"State label {label} is not a valid bare product-basis label. Available (first 10): {available}"
            ) from None

    @staticmethod
    def _array_labeled_concrete(kernel_labeling: Labeling) -> bool:
        """Check array labeling for tracers without materializing lazy eigenstates."""
        return not contains_tracer((kernel_labeling.indices, kernel_labeling.overlaps))

    def _eigenvalue_of_label(
        self,
        label: tuple[int, ...],
        *,
        precomputed: tuple[Any, Any] | None = None,
    ) -> Any:
        """Gather the assigned eigenvalue without concretizing traced indices;
        accept precomputed arrays to share one eigensolve."""
        bare_idx = self._bare_label_index(label)
        if precomputed is None:
            eigenvalues, _, _, kernel_labeling = self._compute_array_labeled()
        else:
            eigenvalues, kernel_labeling = precomputed
        index = kernel_labeling.indices[bare_idx]
        if contains_tracer(index) and not contains_tracer(eigenvalues):
            import jax.numpy as jnp

            return jnp.asarray(eigenvalues)[index]
        return eigenvalues[index]

    def dress(
        self,
        *,
        overlap_threshold: float = 0.5,
        force: bool = False,
        labeling: str = "DE",
    ) -> DressedResult:
        """Diagonalize the lab-frame Hamiltonian and assign bare-state labels.

        Dressing keeps network-generated Hamiltonian terms, so the assigned eigenstates match the
        solver Hamiltonian and degenerate cascaded modes dress into superpositions. The assignment
        uses the ``label_eigensystem`` kernel (:mod:`quchip.chip.dressing`) with
        ``assign_rowwise_greedy``, i.e. confidence-ordered row-greedy matching as a pure
        ``lax.scan``. Its cost is ``O(D**2)`` in the Hilbert dimension, versus ``O(D**3)`` for the
        global variant.

        The result is identical to ``assign_global_greedy`` in the dispersive or weak-hybridization
        regime. It can differ only on strongly hybridized chips, where the assignment is already
        approximate and the labels are flagged. Bare labels whose best match is below
        ``overlap_threshold`` are flagged in :attr:`DressedResult.hybridized_labels` and trigger a
        user warning.

        :class:`DressedResult` is the eager, dict-materialized view.
        Materialization makes the assignment indices concrete and is **not
        traceable**. In ``jax.jit``/``grad``/``vmap``, call :meth:`energy`,
        :meth:`freq`, :meth:`dispersive_shift`, or :meth:`state`, which go
        directly through the array kernel.

        Parameters
        ----------
        overlap_threshold : float
            Confidence cutoff for the greedy assignment.
        force : bool
            Force a new calculation even if the signature matches.
        labeling : str
            Currently only ``"DE"`` is implemented.

        Returns
        -------
        DressedResult
            Cached result. Callers that modify it do so at their own risk.
        """
        if labeling != "DE":
            raise ValueError(f"Unsupported labeling {labeling!r}. Only 'DE' is implemented.")

        signature = self._analysis_signature()
        if (
            not force
            and self._dressed_result is not None
            and self._dressed_signature == signature
            and self._dressed_result.labeling == labeling
            and self._dressed_result.overlap_threshold == float(overlap_threshold)
        ):
            return self._dressed_result

        eigenvalues, eigenvector_matrix, eigensystem, kernel_labeling = self._compute_array_labeled()

        result = _materialize_dressed_result(
            eigenvalues,
            eigenvector_matrix,
            eigensystem,
            kernel_labeling,
            overlap_threshold=float(overlap_threshold),
            labeling=labeling,
            warning_stacklevel=3,
        )
        self._dressed_result = result
        self._dressed_signature = signature
        return result

    def _ensure_dressed(self) -> DressedResult:
        """Materialize the cached dict view only from concrete labeling."""
        _, _, _, kernel_labeling = self._compute_array_labeled()
        if not self._array_labeled_concrete(kernel_labeling):
            raise RuntimeError(_DRESS_TRACING_ERROR)
        if self._dressed_result is None or self._dressed_signature != self._analysis_signature():
            self.dress()
        assert self._dressed_result is not None
        return self._dressed_result

    @property
    def is_dressed(self) -> bool:
        """True if a cached dressed result is present and consistent."""
        return self._dressed_result is not None

    def energy(
        self,
        device_states: Mapping[str | BaseDevice, int] | None = None,
        /,
        **device_state_kwargs: int,
    ) -> Any:
        """Dressed eigenvalue (GHz) for the given bare-state label.

        Unspecified devices default to level 0. The method goes through the
        :func:`quchip.chip.dressing.label_eigensystem` array kernel, so it is
        safe in ``jax.jit``/``grad``/``vmap``, and gradients flow through
        ``eigenvalues[labeling.indices[bare_idx]]`` to all traced chip
        parameters.

        Parameters
        ----------
        device_states : mapping or None, default=None
            Bare product-state levels. Unspecified devices use level zero.
        **device_state_kwargs : int
            Per-device energy levels keyed by label.
        """
        resolved = normalize_device_state_mapping(self._chip, device_states, device_state_kwargs)
        label_t = self._label_from_plain_mapping(resolved)
        return self._energy_lookup()(label_t)

    def dressed_spectrum(self) -> Any:
        """Raw sorted eigenvalue array of the dressed Hamiltonian (GHz)."""
        return self._ensure_dressed().eigenvalues

    def _dressed_state(self, **device_states: int) -> Any:
        """Gather the assigned eigenvector column under tracing; eagerly use the dressed view.
        Global eigenvector phase remains gauge-dependent; populations and overlap magnitudes do not."""
        label_t = self._label_from_plain_mapping(device_states)
        engine = self.engine_result()
        _, eigenvector_matrix, _, kernel_labeling = self._compute_array_labeled(engine)
        if contains_tracer((eigenvector_matrix, kernel_labeling.indices)):
            bare_idx = self._bare_label_index(label_t)
            column = jnp.asarray(eigenvector_matrix)[:, kernel_labeling.indices[bare_idx]]
            dims = list(engine.dims)
            return self._chip.backend.from_array(column.reshape(-1, 1), dims=[dims, [1] * len(dims)])

        dressed = self._ensure_dressed()
        try:
            eigen_idx = dressed.state_map[label_t]
        except KeyError:
            available = list(dressed.state_map.keys())[:10]
            raise KeyError(
                f"State label {label_t} not found in state map. Available (first 10): {available}"
            ) from None
        overlap = dressed.assignment_overlaps[label_t]
        if overlap < _STATE_OVERLAP_WARNING:
            # Hops: _dressed_state -> ChipAnalysis.state -> Chip.state -> caller.
            warnings.warn(
                f"Dressed state label {label_t} has assignment overlap {overlap:.3f} "
                f"(< {_STATE_OVERLAP_WARNING:.3f}); chip.state() returns the assigned dressed "
                "eigenstate. Use chip.bare_state(...) for the product state.",
                UserWarning,
                stacklevel=4,
            )
        return dressed.eigenstates[eigen_idx]

    def _dressed_frequencies(self, *, _local_resolution: Any | None = None) -> dict[str, Any]:
        """Per-device dressed 0 → 1 transition frequencies (GHz)."""
        lookup = self._labeled_lookup(_local_resolution=_local_resolution)
        ground = (0,) * len(self._chip.devices)
        ground_energy = lookup(ground)[0]
        frequencies: dict[str, Any] = {}
        for index, device in enumerate(self._chip.devices):
            excited = list(ground)
            excited[index] = 1
            frequencies[device.label] = lookup(tuple(excited))[0] - ground_energy
        return frequencies

    def dressed_index(
        self,
        device_states: Mapping[str | BaseDevice, int] | None = None,
        /,
        **device_state_kwargs: int,
    ) -> int | None:
        """Return the dressed index assigned to a bare label.

        Parameters
        ----------
        device_states : mapping or None, default=None
            Bare product-state levels. Unspecified devices use level zero.
        **device_state_kwargs : int
            Per-device energy levels keyed by label.
        """
        label = self._state_label_from_mapping(device_states, **device_state_kwargs)
        return self._ensure_dressed().state_map.get(label)

    def bare_label(self, dressed_index: int) -> tuple[int, ...]:
        """Return the bare label assigned to a dressed index.

        Parameters
        ----------
        dressed_index : int
            Index in ascending order of dressed energy.
        """
        if isinstance(dressed_index, bool) or not isinstance(dressed_index, int):
            raise TypeError(f"dressed_index must be an integer, got {type(dressed_index).__name__}")
        try:
            return self._ensure_dressed().bare_labels_by_dressed_index[dressed_index]
        except KeyError:
            raise ValueError(f"No bare-state label assigned to dressed index {dressed_index}") from None

    def operator_in_dressed_basis(
        self,
        device: str | BaseDevice,
        op: str | Any,
        *,
        truncate: int | None = None,
    ) -> Operator:
        """Transform a local operator into the dressed eigenbasis.

        Computes ``U† O_embedded U``, where ``U`` is the dressed eigenvector
        matrix in solver coordinates, phase-fixed to the assigned bare-state
        convention of :meth:`drive_matrix_elements`. Optional truncation keeps
        the lowest ``truncate`` dressed levels.

        Parameters
        ----------
        device : str or BaseDevice
            Device whose local operator is embedded and transformed.
        op : str or Operator
            Operator name resolved from the device (e.g. ``"n"``, ``"a"``), or
            an operator already built in the local solver basis.
        truncate : int, optional
            Keep only the lowest ``truncate`` dressed levels of the result.
        """
        chip = self._chip
        backend = chip.backend
        dressed = self._ensure_dressed()
        from quchip.chip.observables import prepare_local_op
        from quchip.declarative.expr import materialize_expr

        idx, dev = chip._resolve_device_index(device)
        engine = self.engine_result()
        xp = backend.array_module
        raw_eigenvectors = xp.asarray(dressed.eigenvector_matrix, dtype=complex)
        amplitudes = xp.asarray(self._semantic_amplitudes(raw_eigenvectors, engine))
        bare_indices = [
            self._bare_label_index(dressed.bare_labels_by_dressed_index[index])
            for index in range(raw_eigenvectors.shape[1])
        ]
        anchors = amplitudes[bare_indices, xp.arange(raw_eigenvectors.shape[1])]
        U = _phase_fixed_state(raw_eigenvectors, anchors, xp)
        local_op = (
            prepare_local_op(dev, op, engine.bases[dev.label], backend)
            if isinstance(op, str) else materialize_expr(op, backend)
        )
        embedded = backend.embed(local_op, idx, engine.dims)
        op_array = backend.array_module.asarray(backend.to_array(embedded), dtype=complex)
        transformed = xp.conj(U).T @ op_array @ U
        if truncate is not None:
            if truncate <= 0:
                raise ValueError(f"truncate must be positive, got {truncate}")
            transformed = transformed[:truncate, :truncate]
            dims = [[truncate], [truncate]]
        else:
            dims = [[transformed.shape[0]], [transformed.shape[1]]]
        return backend.from_array(transformed, dims=dims)

    def drive_matrix_elements(
        self,
        transition: str | BaseDevice | tuple[Mapping[str | BaseDevice, int], Mapping[str | BaseDevice, int]],
        *,
        drives: Sequence[str | "BaseDrive"] | None = None,
    ) -> LabelKeyedDict:
        """Return dressed matrix elements of wired drive operators.

        The matrix convention is ``m_j^{fi} = <f~|D_j|i~>``, with the final
        dressed state as row index and the initial dressed state as column
        index. Each dressed eigenvector is phase-fixed so that its overlap with
        the assigned bare state is real and nonnegative. Relative matrix
        elements between different conditioned transitions are therefore
        independent of the backend's eigenvector phases.

        Passing a device selects its dressed ground-to-first-excitation
        transition, with every other device in its ground state. Passing
        ``(initial, final)`` mappings selects an arbitrary transition.

        Each drive must expose exactly one local Hamiltonian channel. Without
        applying the signal chain, the method returns the physical matrix
        element of each control line's drive operator, for combination with
        declared signal-chain phasors. This weak-drive projection gives
        effective driven-Hamiltonian coefficients. See Magesan and Gambetta,
        Phys. Rev. A 101, 052308 (2020), DOI 10.1103/PhysRevA.101.052308.

        Parameters
        ----------
        transition : str, BaseDevice, or tuple[mapping, mapping]
            Device shorthand, or ``(initial, final)`` bare-state mappings.
            Unspecified devices in each mapping default to level zero.
        drives : sequence[str or BaseDrive], optional
            Wired control lines to evaluate. ``None`` evaluates every line.
            Original or rebound drive objects resolve by label.

        Returns
        -------
        LabelKeyedDict
            Mapping from drive label to backend-native scalar matrix elements,
            addressable by drive object or label. On a JAX-capable backend, the
            values stay JAX-traceable.

        Raises
        ------
        ValueError
            If no control equipment is attached, if a selected line is not a
            device-target drive, or if a drive exposes zero or multiple
            channels.
        KeyError
            If a requested drive label is not in the attached equipment.
        TypeError
            If ``transition`` is neither a device reference nor a pair of state
            mappings.

        Examples
        --------
        >>> from quchip import Chip, ChargeDrive, ControlEquipment, DuffingTransmon
        >>> q = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q")
        >>> drive = ChargeDrive(q, label="xy")
        >>> chip = Chip([q], control_equipment=ControlEquipment([drive]))
        >>> elements = chip.drive_matrix_elements(q, drives=[drive])
        >>> abs(elements["xy"]) > 0
        True
        """
        equipment = self._chip.control_equipment
        if equipment is None:
            raise ValueError("drive_matrix_elements requires attached control equipment with wired drive lines")

        lines = equipment.lines
        by_label: dict[str, list[BaseDrive]] = {}
        for line in lines:
            by_label.setdefault(line.label, []).append(line)

        if drives is None:
            selected = lines
        else:
            selected = []
            available = [line.label for line in lines]
            for drive in drives:
                label = resolve_label(drive)
                matches = by_label.get(label, [])
                if not matches:
                    raise KeyError(f"No wired drive labeled '{label}'. Available drive lines: {available}")
                if len(matches) > 1:
                    raise ValueError(f"Drive label '{label}' is ambiguous across {len(matches)} wired lines")
                selected.append(matches[0])

        if isinstance(transition, (str, BaseDevice)):
            _, device = self._chip._resolve_device_index(transition)
            initial_label = self._state_label_from_mapping({})
            final_label = self._state_label_from_mapping({device: 1})
        elif (
            isinstance(transition, tuple)
            and len(transition) == 2
            and isinstance(transition[0], Mapping)
            and isinstance(transition[1], Mapping)
        ):
            initial_label = self._state_label_from_mapping(transition[0])
            final_label = self._state_label_from_mapping(transition[1])
        else:
            raise TypeError(
                "transition must be a device reference or an (initial_mapping, final_mapping) pair"
            )

        engine = self.engine_result()
        _, eigenvectors, _, labeling = self._compute_array_labeled(engine)
        xp = self._chip.backend.array_module
        U = xp.asarray(eigenvectors, dtype=complex)
        amplitudes = xp.asarray(self._semantic_amplitudes(U, engine))

        def phase_fixed_state(label: tuple[int, ...]) -> Any:
            bare_index = self._bare_label_index(label)
            state = U[:, labeling.indices[bare_index]]
            return _phase_fixed_state(state, amplitudes[bare_index, labeling.indices[bare_index]], xp)

        initial = phase_fixed_state(initial_label)
        final = phase_fixed_state(final_label)

        elements = LabelKeyedDict()
        backend = self._chip.backend
        for drive in selected:
            from quchip.control.drive import CouplingDrive

            if isinstance(drive, CouplingDrive) or drive.device_label is None:
                raise ValueError(
                    f"Drive '{drive.label}' targets a coupling; dressed drive matrix elements "
                    "currently require a device-target line"
                )
            device_index, device = self._chip._resolve_device_index(drive.device_label)
            from quchip.control.signal import AnalyticSignal
            from quchip.engine.ir import Constant

            with _backend_context(backend):
                authored = drive.hamiltonian(
                    device,
                    AnalyticSignal(program=Constant(1.0 + 0.0j)),
                )
            from quchip.declarative.expr import split_dynamic_hamiltonian
            from quchip.chip.observables import prepare_local_op

            channels = split_dynamic_hamiltonian(authored)
            if len(channels) != 1:
                raise ValueError(
                    f"Drive '{drive.label}' must expose exactly one local Hamiltonian channel; "
                    f"got {len(channels)}."
                )

            local_operator = prepare_local_op(device, channels[0][1], engine.bases[device.label], backend)
            operator = xp.asarray(backend.to_array(local_operator), dtype=complex)
            initial_tensor = initial.reshape(engine.dims)
            acted = xp.tensordot(operator, initial_tensor, axes=((1,), (device_index,)))
            acted = xp.moveaxis(acted, 0, device_index).reshape(-1)
            elements[drive.label] = xp.vdot(final, acted)
        return elements

    def state_components(
        self,
        state: int | Mapping[str | BaseDevice, int] | None = None,
        /,
        *,
        n_components: int = 5,
        **device_state_kwargs: int,
    ) -> dict[tuple[int, ...], float]:
        """Leading bare-basis probabilities ``|⟨bare|dressed⟩|²`` of a dressed eigenstate.

        ``state`` can be an ``int`` (direct dressed index) or a mapping of
        ``{device: Fock}`` (dressed index resolved through label matching).

        Parameters
        ----------
        state : int, mapping, or None, default=None
            Dressed index or bare label. ``None`` uses keyword levels.
        n_components : int, default=5
            Maximum number of components returned.
        **device_state_kwargs : int
            Per-device energy levels keyed by label.
        """
        if n_components <= 0:
            raise ValueError(f"n_components must be positive, got {n_components}")
        dressed = self._ensure_dressed()
        if isinstance(state, int) and not isinstance(state, bool):
            dressed_idx: int = state
        else:
            if state is not None and not isinstance(state, Mapping):
                raise TypeError(f"state must be an int or mapping, got {type(state).__name__}")
            mapping = state if isinstance(state, Mapping) else None
            resolved_idx = self.dressed_index(mapping, **device_state_kwargs)
            if resolved_idx is None:
                label = self._state_label_from_mapping(mapping, **device_state_kwargs)
                raise ValueError(f"No dressed-state index assigned to bare label {label}")
            dressed_idx = resolved_idx

        if dressed_idx < 0 or dressed_idx >= len(dressed.eigenvalues):
            raise ValueError(f"dressed state index {dressed_idx} out of range for dimension {len(dressed.eigenvalues)}")

        amplitudes = self._semantic_amplitudes(dressed.eigenvector_matrix, self.engine_result())
        return top_components(amplitudes, dressed.bare_labels, dressed_idx, n_components)

    def dispersive_shift(self, device_a: str | BaseDevice, device_b: str | BaseDevice) -> float:
        """Dressed cross-Kerr shift (GHz): ``E(1,1) − E(1,0) − E(0,1) + E(0,0)``.

        Equivalent to the static ZZ interaction strength between the two
        devices, with all other devices grounded. See Blais et al., RMP 93,
        025005 (2021), §IV.C.

        Parameters
        ----------
        device_a, device_b : str or BaseDevice
            Devices whose cross-Kerr shift is evaluated.

        Examples
        --------
        >>> from quchip import DuffingTransmon, Capacitive, Chip
        >>> q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        >>> q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.22, levels=3, label="q1")
        >>> chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.02)])
        >>> zz = chip.dispersive_shift(q0, q1)  # residual ZZ in GHz
        """
        index_a, _ = self._chip._resolve_device_index(device_a)
        index_b, _ = self._chip._resolve_device_index(device_b)
        return _kerr_coefficient(index_a, index_b, self._semantic_dims(), self._energy_lookup())

    def kerr_matrix(self) -> KerrMatrix:
        """Return the dressed self-Kerr and cross-Kerr matrix in GHz.

        Labels and axes follow chip device order. Diagonal entries are dressed
        anharmonicities, and are ``NaN`` when that device has fewer than three
        resolved levels. Off-diagonal entries use the full-pull
        ``E11 - E10 - E01 + E00`` convention.
        """
        energy = self._energy_lookup()
        dims = self._semantic_dims()
        n_devices = len(self._chip.devices)
        dtype = jnp.real(jnp.asarray(energy((0,) * n_devices))).dtype
        values = jnp.zeros((n_devices, n_devices), dtype=dtype)

        for row in range(n_devices):
            for column in range(row, n_devices):
                entry = jnp.real(_kerr_coefficient(row, column, dims, energy))
                values = values.at[row, column].set(entry)
                values = values.at[column, row].set(entry)

        return KerrMatrix(labels=self._device_labels(), values=values)

    def effective_subspace_hamiltonian(
        self,
        states: (
            list[Mapping[str | BaseDevice, int] | tuple[int, ...]]
            | tuple[Mapping[str | BaseDevice, int] | tuple[int, ...], ...]
        ),
    ) -> np.ndarray:
        """Effective Hamiltonian projected onto a labeled bare subspace.

        Returns a Hermitian matrix in the user-selected bare-state basis whose
        eigenvalues are exactly the dressed eigenenergies of the listed
        bare-label states. The truncated overlap block is Löwdin (``S^{-1/2}``)
        orthonormalized, the same des-Cloizeaux construction as
        :func:`quchip.analysis.effective_hamiltonian`. Hybridization with
        states *outside* the subspace therefore cannot leak absolute dressed
        energies into the off-diagonal elements. The method connects full-chip
        dressed physics to the low-dimensional effective models of dispersive
        gate design and state-transfer analysis.

        Parameters
        ----------
        states : sequence of mapping or tuple[int, ...]
            Bare-state labels spanning the subspace, each a
            ``{device: energy_level}`` mapping or a full chip-ordered level tuple.
        """
        # Every label of a sector has an assignment, so the sector route needs no dressed view.
        state_map = None if self._sector_model() is not None else self._ensure_dressed().state_map
        labels = []
        for state in states:
            label = state if isinstance(state, tuple) else self._state_label_from_mapping(state)
            if not (self._is_bare_label(label) if state_map is None else label in state_map):
                raise ValueError(f"No dressed-state assignment found for bare label {label}")
            labels.append(label)
        from quchip.analysis.effective_hamiltonian import _h_eff_on_basis

        return np.asarray(_h_eff_on_basis(self._chip, labels), dtype=complex)

    def dressed_anharmonicity(self, device: str | BaseDevice) -> float:
        """Return the dressed anharmonicity in GHz.

        Parameters
        ----------
        device : str or BaseDevice
            Device label or object. All other devices are grounded.
        """
        index, _ = self._chip._resolve_device_index(device)
        return _kerr_coefficient(index, index, self._semantic_dims(), self._energy_lookup())

    def transition_frequency(
        self,
        target: str | BaseDevice,
        lower: int,
        upper: int,
        when: dict[str | BaseDevice, int] | None = None,
    ) -> Any:
        """Return one optionally conditioned dressed transition in GHz.

        Unspecified spectators are grounded. The target cannot appear in
        ``when``. Traceable under ``jit``/``grad``/``vmap``.

        Parameters
        ----------
        target : str or BaseDevice
            Device whose transition is measured.
        lower, upper : int
            Lower and upper local energy levels.
        when : dict or None, default=None
            Conditioning levels for spectator devices.
        """
        idx_target, target_device = self._chip._resolve_device_index(target)
        _validate_level_pair(lower, upper, self._semantic_dims()[idx_target])

        conditioned = normalize_device_state_mapping(self._chip, when, {})
        if target_device.label in conditioned:
            raise ValueError(
                f"when may contain only spectators; target {target_device.label!r} was included."
            )

        lower_label = list(self._label_from_resolved(conditioned))
        upper_label = list(lower_label)
        lower_label[idx_target] = lower
        upper_label[idx_target] = upper

        lower_tuple = self._label_from_resolved(
            dict(zip(self._device_labels(), lower_label))
        )
        upper_tuple = self._label_from_resolved(
            dict(zip(self._device_labels(), upper_label))
        )

        lookup = self._labeled_lookup()
        energies = []
        for label in (lower_tuple, upper_tuple):
            energy, overlap, margin = lookup(label)
            overlap = maybe_concrete_scalar(overlap)
            margin = maybe_concrete_scalar(margin)
            if overlap is not None and margin is not None and (
                overlap < 0.5 or margin <= 1e-8
            ):
                raise ValueError(
                    "Dressed transition assignment is unreliable for bare label "
                    f"{label}: overlap={overlap:.6g}, margin={margin:.6g}. "
                    "Inspect chip.dress().assignment_overlaps or choose a better-resolved model."
                )
            energies.append(energy)
        return energies[1] - energies[0]

    def freq(
        self,
        target: str | BaseDevice | None = None,
        when: dict[str | BaseDevice, int] | None = None,
    ) -> dict[str, Any] | Any:
        """All dressed 0→1 frequencies (GHz), or one conditional transition.

        Parameters
        ----------
        target : str or BaseDevice, optional
            Device whose 0→1 transition is returned. ``None`` returns the
            full ``{device_label: frequency}`` dict for every device.
        when : dict[str | BaseDevice, int], optional
            Spectator Fock indices held fixed while the transition is
            evaluated. Unlisted devices stay in their ground state.
        """
        if target is None:
            return self._dressed_frequencies()
        return self.transition_frequency(target, 0, 1, when=when)

    def frame_info(self) -> dict[str, Any]:
        """Per-device frame reference frequency ``ω_ref,i`` (GHz).

        Resolves the chip's current frame spec and returns a flat
        ``{device_label: ω_ref,i}`` dict. The assembler subtracts these
        concrete frequencies as ``-Σ_i ω_ref,i n̂_i``. The method thus shows
        what the solver will solve without running it.

        Values are returned as the frame resolver produces them. They are
        concrete Python floats in ``"lab"``, ``"rotating"``, or scalar modes,
        and can be JAX tracers in ``dict`` mode when the user wired a traced
        reference frequency through. Traced values pass through unchanged to
        preserve differentiability.
        """
        # frame_info resolves the frame via the engine's own path
        # (`quchip.engine.frames.resolve_frame`). The assembler performs the
        # subtraction in `assembly._build_static_h0`.
        from quchip.engine.frames import resolve_frame

        resolved = resolve_frame(self._chip, self._chip.frame)
        return dict(resolved.frequencies)

    def state(
        self,
        device_states: Mapping[str | BaseDevice, int] | str | None = None,
        /,
        **device_state_kwargs: int,
    ) -> State:
        """Dressed eigenstate for local-energy product-state labels.

        Validates the mapping (rejects ``bool``, non-int, and out-of-range
        indices) and returns the assigned dressed eigenvector. After
        :meth:`Chip.set_state_order`, a ``str`` shorthand (e.g. ``"eg1"``) is
        parsed via :func:`~quchip.chip.states.normalize_device_state_mapping`.
        Use :meth:`Chip.bare_state` for arbitrary kets.

        If the requested label's assignment overlap is low, the method warns
        and names :meth:`Chip.bare_state` as the product-state alternative.

        Safe in ``jax.jit``/``grad``/``vmap``. Under tracing, the method
        selects the eigenvector column through the array kernel, so a dressed
        initial state is differentiable end-to-end.

        Parameters
        ----------
        device_states : mapping, str, or None, default=None
            Bare product-state label used for dressed assignment.
        **device_state_kwargs : int
            Per-device energy levels keyed by label.
        """
        resolved = normalize_device_state_mapping(self._chip, device_states, device_state_kwargs)
        self._label_from_resolved(resolved)
        return self._dressed_state(**resolved)
