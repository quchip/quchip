"""Dressed-state analysis owned by :class:`quchip.chip.chip.Chip`.

``ChipAnalysis`` diagonalizes the lab-frame static Hamiltonian, assigns
bare-state labels to dressed eigenvectors by overlap maximization, and
exposes derived quantities (dressed eigenenergies, transition
frequencies, dispersive shifts, effective subspace Hamiltonians). All
results cache against a structural signature and refresh automatically
when any device or coupling parameter mutates.

Dressing is always computed in the lab frame — frame selection never
alters dressed data.

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
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from quchip.backend import EigensystemData, Operator, State, _backend_context
from quchip.chip.dressing import (
    BareProductReference,
    Labeling,
    assign_rowwise_greedy,
    label_eigensystem,
)
from quchip.chip.states import normalize_device_state_mapping
from quchip.devices.base import BaseDevice
from quchip.engine.stage1_frames import resolve_frame
from quchip.utils.jax_utils import contains_tracer
from quchip.utils.labeling import LabelKeyedDict, bare_label_from_mapping, resolve_label, top_components

if TYPE_CHECKING:
    from quchip.chip.chip import Chip
    from quchip.control.drive import BaseDrive


_DRESS_TRACING_ERROR = (
    "Chip.dress() returns a concrete dict-keyed DressedResult and is not "
    "traceable under jax.jit/grad/vmap. Use Chip.energy(), Chip.freq(), "
    "Chip.dispersive_shift(), or Chip.state() inside transforms — they "
    "route through the array-only kernel in quchip.chip.dressing and stay "
    "differentiable."
)


def _phase_fixed_state(state: Any, bare_index: int, xp: Any) -> Any:
    """Set one dressed vector's assigned bare overlap to a nonnegative real value."""
    anchor = state[bare_index]
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
        Backend eigenstate objects in the same order as ``eigenvalues``.
        Materialized lazily on first access from the underlying
        :class:`~quchip.backend.containers.EigensystemData` — the dressing /
        sweep hot path never touches them, so the per-column backend kets are
        only built when a caller actually asks for states.
    state_map : dict[tuple[int, ...], int]
        Assignment from bare product-basis label (one int per device) to
        the dressed eigenstate index it overlaps most with.
    dressed_eigenvalues : dict[tuple[int, ...], Any]
        Dressed eigenvalue for each assigned bare label — direct lookup
        path for :meth:`Chip.energy`.
    assignment_overlaps : dict[tuple[int, ...], float]
        ``|⟨bare|dressed⟩|²`` of each assignment; values below
        ``overlap_threshold`` flag hybridization.
    hybridized_labels : tuple[tuple[int, ...], ...]
        Bare labels whose assignment quality is below ``overlap_threshold``.
        A non-empty tuple triggers a user warning at dress time.
    bare_labels : tuple[tuple[int, ...], ...]
        Full canonical product-basis label set (all combinations of
        ``range(device.levels)`` for every device, in chip order).
    bare_labels_by_dressed_index : dict[int, tuple[int, ...]]
        Inverse of :attr:`state_map` — dressed index → assigned bare label.
    eigenvector_matrix : array-like or None
        Columns = dressed eigenvectors in the bare product basis. Used
        for :meth:`ChipAnalysis.operator_in_dressed_basis` and
        :meth:`ChipAnalysis.state_components`.
    overlap_threshold : float
        Minimum overlap for a confident assignment.
    labeling : str
        Algorithm id (currently only ``"DE"``), resolved to the
        confidence-ordered row-greedy overlap-matching policy actually run
        by :meth:`ChipAnalysis.dress` (:func:`~quchip.chip.dressing.assign_rowwise_greedy`).
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


class ChipAnalysis:
    """Dressed-state analysis, caching, and dressed-basis helpers.

    Every :class:`~quchip.chip.chip.Chip` owns one ``ChipAnalysis`` as
    ``chip._analysis``. The chip class forwards its public dressed-state
    API here; users normally call methods on the chip, not this class
    directly.

    Caching: :meth:`dress` keys its cache on a structural signature
    covering backend identity, RWA policy, and the ``state_version`` of
    every device and coupling. Any mutation that bumps a version
    invalidates the cache on next access.
    """

    def __init__(self, chip: "Chip") -> None:
        self._chip = chip
        self._dressed_result: DressedResult | None = None
        self._dressed_signature: tuple[Any, ...] | None = None
        self._array_cache: tuple[Any, Any, Any, Labeling] | None = None
        self._array_signature: tuple[Any, ...] | None = None
        self._bare_labels_cache: tuple[
            tuple[tuple[int, ...], ...], dict[tuple[int, ...], int]
        ] | None = None
        self._bare_labels_signature: tuple[int, ...] | None = None

    def _analysis_signature(self) -> tuple[Any, ...]:
        """Hashable fingerprint covering every structural input to dressing."""
        chip = self._chip
        return (
            f"{type(chip.backend).__module__}.{type(chip.backend).__qualname__}",
            chip.rwa,
            tuple((device.label, device.state_version) for device in chip.devices),
            tuple(
                (
                    f"{type(coupling).__module__}.{type(coupling).__qualname__}",
                    coupling.device_a_label,
                    coupling.device_b_label,
                    coupling.state_version,
                )
                for coupling in chip.couplings
            ),
        )

    def _canonical_bare_labels(self) -> tuple[tuple[int, ...], ...]:
        """Full product-basis label set ``⨂_d range(d.levels)`` in chip order."""
        return self._bare_labels_with_index()[0]

    def _bare_labels_with_index(
        self,
    ) -> tuple[tuple[tuple[int, ...], ...], dict[tuple[int, ...], int]]:
        """Cached ``(bare_labels, label → index)`` pair, keyed on per-device dimensions."""
        sig = tuple(device.levels for device in self._chip.devices)
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
        """Merge a ``{device: Fock}`` mapping into a full chip-ordered label tuple.

        Unspecified devices default to Fock index 0. A ``str`` shorthand is
        parsed through :func:`~quchip.chip.states.normalize_device_state_mapping`.
        Validates each value as a non-negative ``int`` (rejecting ``bool``)
        within device bounds.
        """
        resolved = normalize_device_state_mapping(self._chip, device_states, device_state_kwargs)
        return self._label_from_resolved(resolved)

    def _label_from_resolved(self, resolved: Mapping[str, int]) -> tuple[int, ...]:
        """Validate an already-normalized ``{label: Fock}`` mapping into a full label tuple.

        Splits the validation pass out of :meth:`_state_label_from_mapping` so
        callers that already hold the normalized mapping (e.g. :meth:`state`)
        validate it without normalizing a second time. Unspecified devices
        default to Fock index 0; each value must be a non-negative ``int``
        (rejecting ``bool``) within device bounds.

        The Fock-bound and type checks are layered chip-side on top of the
        canonical :func:`~quchip.utils.labeling.bare_label_from_mapping`
        spec-to-tuple builder; :meth:`Chip._resolve_device_index` rejects
        unknown labels first with the device-specific message.
        """
        for device_label, value in resolved.items():
            _, device = self._chip._resolve_device_index(device_label)
            if isinstance(value, bool):
                raise ValueError(f"Fock index for '{device.label}' must be an integer, got bool: {value!r}")
            if not isinstance(value, int):
                raise TypeError(f"Expected integer Fock index for '{device.label}', got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"Fock index for '{device.label}' must be >= 0, got {value}")
            if value >= device.levels:
                raise ValueError(
                    f"Fock index {value} for '{device.label}' exceeds device dimension ({device.levels} levels)"
                )
        return bare_label_from_mapping(self._device_labels(), resolved, {})

    def _label_from_plain_mapping(self, device_states: Mapping[Any, Any]) -> tuple[int, ...]:
        """Merge a mapping into a bare-label tuple without the full validation pass.

        Used by lookup-only paths (:meth:`energy`, :meth:`_dressed_state`).
        Unknown labels are rejected eagerly with the device-specific message;
        the tuple itself is assembled by the canonical
        :func:`~quchip.utils.labeling.bare_label_from_mapping`.
        """
        for device_label in device_states:
            self._chip._resolve_device_index(device_label)
        return bare_label_from_mapping(self._device_labels(), device_states, {})

    def _device_labels(self) -> tuple[str, ...]:
        """Chip device labels in tensor-product order."""
        return tuple(device.label for device in self._chip.devices)

    def _compute_array_labeled(self) -> tuple[Any, Any, Any, Labeling]:
        """Pure-array path: ``(eigenvalues, eigenvector_matrix, eigenstates, labeling)``.

        Always returns the ``label_eigensystem`` kernel output directly.
        Cached against the structural signature only when the result is
        free of JAX tracers: under ``jit``/``grad``/``vmap`` the result is
        recomputed every call rather than stashing a tracer bound to a
        stale trace context.

        This is the trace-friendly primitive used by :meth:`energy`,
        :meth:`freq`, and :meth:`dispersive_shift`. :meth:`dress` is the
        eager dict-materialized view on top of this.
        """
        chip = self._chip
        signature = self._analysis_signature()
        if (
            self._array_cache is not None
            and self._array_signature == signature
            and not contains_tracer(self._array_cache)
        ):
            return self._array_cache

        saved_frame = chip._frame_spec
        chip._frame_spec = "lab"
        try:
            hamiltonian = chip.hamiltonian()
        finally:
            chip._frame_spec = saved_frame

        eigensystem = chip.backend.eigensystem_data(hamiltonian)
        eigenvalues = eigensystem.eigenvalues
        eigenvector_matrix = eigensystem.eigenvector_matrix

        evals_jax = jnp.asarray(eigenvalues)
        evecs_jax = jnp.asarray(eigenvector_matrix)

        dims = tuple(device.levels for device in chip.devices)
        reference = BareProductReference(dims=dims)
        labeling = label_eigensystem(evecs_jax, reference, policy=assign_rowwise_greedy)

        # The 3rd slot carries the EigensystemData (lazy eigenstates) rather than
        # a materialized ket list — nothing on the hot path reads it. The cache
        # tracer-check covers only slots (0, 1, 3); touching slot 2 would force
        # the lazy ``eigenstates`` property and defeat the deferral.
        result = (eigenvalues, eigenvector_matrix, eigensystem, labeling)
        if not contains_tracer((evals_jax, labeling.indices, labeling.overlaps)):
            self._array_cache = result
            self._array_signature = signature
        return result

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
        """True when the kernel labeling carries no JAX tracers.

        The dict-materialized :class:`DressedResult` view concretizes the
        assignment indices, so it can only be built when the kernel output is
        free of tracers. Centralizes the gate shared by :meth:`dress` and
        :meth:`_ensure_dressed`.
        """
        return not contains_tracer((kernel_labeling.indices, kernel_labeling.overlaps))

    def _eigenvalue_of_label(
        self,
        label: tuple[int, ...],
        *,
        precomputed: tuple[Any, Any] | None = None,
    ) -> Any:
        """Dressed eigenvalue (GHz) for a bare label, gathered through the array kernel.

        Resolves *label* to its bare-product index and gathers
        ``eigenvalues[kernel_labeling.indices[bare_idx]]``. The gather stays a
        JAX-indexable op on the kernel output, so it is differentiable w.r.t.
        any traced chip parameter — no Python concretization of the
        eigenvalue. Pass *precomputed* =
        ``(eigenvalues, kernel_labeling)`` to share a single
        :meth:`_compute_array_labeled` across several labels (transition
        frequencies, dispersive shifts, anharmonicities).
        """
        bare_idx = self._bare_label_index(label)
        if precomputed is None:
            eigenvalues, _, _, kernel_labeling = self._compute_array_labeled()
        else:
            eigenvalues, kernel_labeling = precomputed
        return eigenvalues[kernel_labeling.indices[bare_idx]]

    def dress(
        self,
        *,
        overlap_threshold: float = 0.5,
        force: bool = False,
        labeling: str = "DE",
    ) -> DressedResult:
        """Diagonalize the lab-frame Hamiltonian and assign bare-state labels.

        Assignment goes through the ``label_eigensystem`` kernel
        (:mod:`quchip.chip.dressing`) with ``assign_rowwise_greedy`` —
        confidence-ordered row-greedy matching as a pure ``lax.scan``,
        ``O(D**2)`` in the Hilbert dimension versus the ``O(D**3)`` global
        variant. It is identical to ``assign_global_greedy`` in the
        dispersive / weak-hybridization regime and can differ only on
        strongly-hybridized chips, where the assignment is already
        approximate (those labels are flagged below). Bare labels whose
        best match is below ``overlap_threshold`` are flagged
        :attr:`DressedResult.hybridized_labels` and trigger a user warning.

        :class:`DressedResult` is the eager, dict-materialized view.
        Materialization concretizes the assignment indices and is **not
        traceable** — call :meth:`energy`, :meth:`freq`,
        :meth:`dispersive_shift`, or :meth:`state` from inside
        ``jax.jit``/``grad``/``vmap``; those route through the array
        kernel directly.

        Parameters
        ----------
        overlap_threshold : float
            Confidence cutoff for the greedy assignment.
        force : bool
            Force recomputation even if the signature matches.
        labeling : str
            Currently only ``"DE"`` is implemented.

        Returns
        -------
        DressedResult
            Cached result — mutate at the caller's own risk.
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

        if not self._array_labeled_concrete(kernel_labeling):
            raise RuntimeError(_DRESS_TRACING_ERROR)

        # The kernel already materialized the canonical product-basis keys in
        # Kronecker order (``BareProductReference``); reuse them instead of a
        # second ``itertools.product`` over the device dims.
        bare_labels = kernel_labeling.keys
        indices_np = np.asarray(kernel_labeling.indices)
        overlaps_np = np.asarray(kernel_labeling.overlaps)

        state_map: dict[tuple[int, ...], int] = {}
        bare_labels_by_dressed_index: dict[int, tuple[int, ...]] = {}
        assignment_overlaps: dict[tuple[int, ...], float] = {}
        dressed_eigenvalues: dict[tuple[int, ...], Any] = {}
        for k, bare_label in enumerate(bare_labels):
            idx = int(indices_np[k])
            state_map[bare_label] = idx
            bare_labels_by_dressed_index[idx] = bare_label
            assignment_overlaps[bare_label] = float(overlaps_np[k])
            dressed_eigenvalues[bare_label] = eigenvalues[idx]

        hybridized_labels = tuple(
            bare_label for bare_label in bare_labels
            if assignment_overlaps[bare_label] < overlap_threshold
        )
        if hybridized_labels:
            preview = ", ".join(
                f"{label} ({assignment_overlaps[label]:.3f})"
                for label in sorted(hybridized_labels, key=lambda item: assignment_overlaps[item])[:4]
            )
            warnings.warn(
                "Strong hybridization detected during dressed-state assignment; "
                "bare labels are approximate for "
                f"{len(hybridized_labels)} states. Lowest-overlap labels: "
                f"{preview}. Inspect DressedResult.assignment_overlaps for "
                "full assignment quality.",
                UserWarning,
                stacklevel=2,
            )

        result = DressedResult(
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
        self._dressed_result = result
        self._dressed_signature = signature
        return result

    def _ensure_dressed(self) -> DressedResult:
        """Return the cached :class:`DressedResult`, diagonalizing if needed.

        Under JAX tracing the dict view cannot be materialized (its assignment
        indices are tracers), so this raises :class:`RuntimeError` in that
        case — trace-sensitive consumers should route through
        :meth:`_compute_array_labeled`.
        """
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

        Unspecified devices default to Fock index 0. Routes through the
        :func:`quchip.chip.dressing.label_eigensystem` array kernel, so
        this is safe inside ``jax.jit``/``grad``/``vmap`` — gradients
        flow through ``eigenvalues[labeling.indices[bare_idx]]`` to any
        traced chip parameters.
        """
        resolved = normalize_device_state_mapping(self._chip, device_states, device_state_kwargs)
        label_t = self._label_from_plain_mapping(resolved)
        return self._eigenvalue_of_label(label_t)

    def dressed_spectrum(self) -> Any:
        """Raw sorted eigenvalue array of the dressed Hamiltonian (GHz)."""
        return self._ensure_dressed().eigenvalues

    def _dressed_state(self, **device_states: int) -> Any:
        """Dressed eigenstate (as a backend ket) for a bare-state label.

        Eager: the cached dict view (with hybridization warnings).
        Traced: selects the assigned eigenvector column straight through
        the :func:`label_eigensystem` array kernel —
        ``evecs[:, labeling.indices[bare_idx]]`` — so dressed initial
        states stay differentiable end-to-end.
        The eigenvector's global phase is gauge-dependent (``eigh``
        column convention); populations and ``|overlap|`` are unaffected.
        """
        label_t = self._label_from_plain_mapping(device_states)
        _, eigenvector_matrix, _, kernel_labeling = self._compute_array_labeled()
        if contains_tracer((eigenvector_matrix, kernel_labeling.indices)):
            bare_idx = self._bare_label_index(label_t)
            column = jnp.asarray(eigenvector_matrix)[:, kernel_labeling.indices[bare_idx]]
            dims = [device.levels for device in self._chip.devices]
            return self._chip.backend.from_array(column.reshape(-1, 1), dims=[dims, [1] * len(dims)])

        dressed = self._ensure_dressed()
        try:
            eigen_idx = dressed.state_map[label_t]
        except KeyError:
            available = list(dressed.state_map.keys())[:10]
            raise KeyError(
                f"State label {label_t} not found in state map. Available (first 10): {available}"
            ) from None
        return dressed.eigenstates[eigen_idx]

    def _dressed_frequencies(self) -> dict[str, float]:
        """Per-device dressed 0 → 1 transition frequencies (GHz)."""
        return {device.label: self._transition_freq(device) for device in self._chip.devices}

    def dressed_index(
        self,
        device_states: Mapping[str | BaseDevice, int] | None = None,
        /,
        **device_state_kwargs: int,
    ) -> int | None:
        """Dressed-state index assigned to a bare label, or ``None`` if unassigned."""
        label = self._state_label_from_mapping(device_states, **device_state_kwargs)
        return self._ensure_dressed().state_map.get(label)

    def bare_label(self, dressed_index: int) -> tuple[int, ...]:
        """Bare-state label assigned to a dressed-state index."""
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

        Computes ``U† O_embedded U`` where ``U`` is the dressed
        eigenvector matrix (columns are dressed eigenstates in the bare
        product basis), phase-fixed to the assigned bare-state convention
        used by :meth:`drive_matrix_elements`. Optional truncation keeps the
        lowest ``truncate`` dressed levels.

        Parameters
        ----------
        device : str or BaseDevice
            Device whose local operator is embedded and transformed.
        op : str or Operator
            Operator name resolved off the device (e.g. ``"n"``, ``"a"``)
            or an already-built local-space operator.
        truncate : int, optional
            Keep only the lowest ``truncate`` dressed levels of the result.
        """
        chip = self._chip
        backend = chip.backend
        dressed = self._ensure_dressed()
        idx, dev = chip._resolve_device_index(device)
        xp = backend.array_module
        raw_eigenvectors = xp.asarray(dressed.eigenvector_matrix, dtype=complex)
        U = xp.stack([
            _phase_fixed_state(
                raw_eigenvectors[:, dressed_index],
                self._bare_label_index(dressed.bare_labels_by_dressed_index[dressed_index]),
                xp,
            )
            for dressed_index in range(raw_eigenvectors.shape[1])
        ], axis=1)
        # Resolve the operator name straight off the device (it owns the
        # vocabulary) rather than round-tripping through chip.observable.
        with _backend_context(backend):
            local_op = dev.local_operator(op) if isinstance(op, str) else op
        embedded = backend.embed(local_op, idx, chip.dims)
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

        The matrix convention is ``m_j^{fi} = <f~|D_j|i~>``: the final
        dressed state is the row index and the initial dressed state is the
        column index. Each dressed eigenvector is phase-fixed so that its
        overlap with the assigned bare state is real and nonnegative. This
        makes relative matrix elements between different conditioned
        transitions independent of the backend's eigenvector phases. Passing
        a device selects the dressed ground-to-first-excitation transition of
        that device with every other device in its ground state. Passing
        ``(initial, final)`` mappings selects an arbitrary transition.

        Each drive must expose exactly one local Hamiltonian channel. The
        signal chain is not applied: this method returns the physical matrix
        element of each control line's drive operator, which can then be
        combined with declared signal-chain phasors. This is the weak-drive
        projection used for effective driven-Hamiltonian coefficients; see
        Magesan and Gambetta, Phys. Rev. A 101, 052308 (2020),
        DOI 10.1103/PhysRevA.101.052308.

        Parameters
        ----------
        transition : str, BaseDevice, or tuple[mapping, mapping]
            Device shorthand, or ``(initial, final)`` bare-state mappings.
            Unspecified devices in either mapping default to level zero.
        drives : sequence[str or BaseDrive], optional
            Wired control lines to evaluate. ``None`` evaluates every line.
            Original or rebound drive objects resolve by label.

        Returns
        -------
        LabelKeyedDict
            Drive-label mapping of backend-native scalar matrix elements.
            Values remain JAX-traceable on a JAX-capable backend; each entry
            is addressable by the drive object or its label.

        Raises
        ------
        ValueError
            If no control equipment is attached, a selected line is not a
            device-target drive, or a drive exposes zero or multiple channels.
        KeyError
            If a requested drive label is absent from the attached equipment.
        TypeError
            If ``transition`` is neither a device reference nor a pair of
            state mappings.

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

        _, eigenvectors, _, labeling = self._compute_array_labeled()
        xp = self._chip.backend.array_module
        U = xp.asarray(eigenvectors, dtype=complex)

        def phase_fixed_state(label: tuple[int, ...]) -> Any:
            bare_index = self._bare_label_index(label)
            state = U[:, labeling.indices[bare_index]]
            return _phase_fixed_state(state, bare_index, xp)

        initial = phase_fixed_state(initial_label)
        final = phase_fixed_state(final_label)

        elements = LabelKeyedDict()
        backend = self._chip.backend
        for drive in selected:
            if drive.target_kind != "device" or drive.device_label is None:
                raise ValueError(
                    f"Drive '{drive.label}' targets {drive.target_kind!r}; dressed drive matrix elements "
                    "currently require a device-target line"
                )
            device_index, device = self._chip._resolve_device_index(drive.device_label)
            with _backend_context(backend):
                channels = drive.local_channels(device)
            if len(channels) != 1:
                raise ValueError(
                    f"Drive '{drive.label}' exposes {len(channels)} local Hamiltonian channels; "
                    "drive_matrix_elements requires exactly one unambiguous operator"
                )
            operator = xp.asarray(backend.to_array(channels[0].operator), dtype=complex)
            initial_tensor = initial.reshape(self._chip.dims)
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

        ``state`` may be an ``int`` (direct dressed index) or a mapping
        of ``{device: Fock}`` (dressed index resolved via label matching).
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

        return top_components(dressed.eigenvector_matrix, dressed.bare_labels, dressed_idx, n_components)

    def dispersive_shift(self, device_a: str | BaseDevice, device_b: str | BaseDevice) -> float:
        """Dressed cross-Kerr shift (GHz): ``E(1,1) − E(1,0) − E(0,1) + E(0,0)``.

        Equivalent to the static ZZ interaction strength between the two
        devices with all others grounded. See Blais et al., RMP 93,
        025005 (2021), §IV.C.

        Parameters
        ----------
        device_a, device_b : str or BaseDevice
            The two devices whose cross-Kerr shift is evaluated.

        Examples
        --------
        >>> from quchip import DuffingTransmon, Capacitive, Chip
        >>> q0 = DuffingTransmon(freq=5.0, anharmonicity=-0.25, levels=3, label="q0")
        >>> q1 = DuffingTransmon(freq=5.2, anharmonicity=-0.22, levels=3, label="q1")
        >>> chip = Chip([q0, q1], couplings=[Capacitive(q0, q1, g=0.02)])
        >>> zz = chip.dispersive_shift(q0, q1)  # residual ZZ in GHz
        """
        eigenvalues, _, _, kernel_labeling = self._compute_array_labeled()
        precomputed = (eigenvalues, kernel_labeling)

        def e(a: int, b: int) -> Any:
            label = self._label_from_plain_mapping({device_a: a, device_b: b})
            return self._eigenvalue_of_label(label, precomputed=precomputed)

        return e(1, 1) - e(1, 0) - e(0, 1) + e(0, 0)

    def effective_subspace_hamiltonian(
        self,
        states: (
            list[Mapping[str | BaseDevice, int] | tuple[int, ...]]
            | tuple[Mapping[str | BaseDevice, int] | tuple[int, ...], ...]
        ),
    ) -> np.ndarray:
        """Effective Hamiltonian projected onto a labeled bare subspace.

        Returns a Hermitian matrix in the user-chosen bare-state basis
        whose eigenvalues are exactly the dressed eigenenergies of the
        listed bare-label states. The truncated overlap block is Löwdin
        (``S^{-1/2}``) orthonormalized — the same des-Cloizeaux
        construction as :func:`quchip.analysis.effective_hamiltonian` —
        so hybridization with states *outside* the subspace cannot leak
        absolute dressed energies into the off-diagonal elements.
        Bridges full-chip dressed physics to the kind of low-dimensional
        effective model used in dispersive gate design and
        state-transfer analysis.

        Parameters
        ----------
        states : sequence of mapping or tuple[int, ...]
            The bare-state labels spanning the subspace, each a
            ``{device: Fock}`` mapping or a full chip-ordered index tuple.
        """
        dressed = self._ensure_dressed()
        label_to_bare_index = {label: idx for idx, label in enumerate(dressed.bare_labels)}

        dressed_indices: list[int] = []
        bare_indices: list[int] = []
        for state in states:
            label = state if isinstance(state, tuple) else self._state_label_from_mapping(state)
            if label not in dressed.state_map:
                raise ValueError(f"No dressed-state assignment found for bare label {label}")
            dressed_indices.append(dressed.state_map[label])
            bare_indices.append(label_to_bare_index[label])

        evec = np.asarray(dressed.eigenvector_matrix, dtype=complex)
        overlap = evec[np.asarray(bare_indices), :][:, np.asarray(dressed_indices)]
        energies = np.asarray(dressed.eigenvalues, dtype=complex)[np.asarray(dressed_indices)]

        # Löwdin-orthonormalize the truncated block: S^{-1/2} makes the row
        # vectors unitary on the subspace, so the result is unitarily similar
        # to diag(energies) no matter how much weight the dressed states carry
        # outside the chosen bare labels.
        gram_vals, gram_vecs = np.linalg.eigh(overlap @ overlap.conj().T)
        inv_sqrt = gram_vecs @ np.diag(gram_vals**-0.5) @ gram_vecs.conj().T
        effective = inv_sqrt @ (overlap @ np.diag(energies) @ overlap.conj().T) @ inv_sqrt
        return 0.5 * (effective + effective.conj().T)

    def dressed_anharmonicity(self, device: str | BaseDevice) -> float:
        """Dressed anharmonicity (GHz): ``E_2 − 2·E_1 + E_0``, others grounded."""
        eigenvalues, _, _, kernel_labeling = self._compute_array_labeled()
        precomputed = (eigenvalues, kernel_labeling)

        def e(n: int) -> Any:
            label = self._label_from_plain_mapping({device: n})
            return self._eigenvalue_of_label(label, precomputed=precomputed)

        return e(2) - 2.0 * e(1) + e(0)

    def _transition_freq(
        self,
        target: str | BaseDevice,
        when: dict[str | BaseDevice, int] | None = None,
    ) -> Any:
        """Conditional 0 → 1 dressed transition frequency of *target*.

        Other devices are grounded by default; *when* specifies any
        non-ground spectators. Traceable under ``jit``/``grad``/``vmap``.
        """
        idx_target, _ = self._chip._resolve_device_index(target)
        ground_label = [0] * len(self._chip.devices)
        if when is not None:
            for device_key, value in when.items():
                if isinstance(value, bool):
                    raise ValueError(f"Fock index must be an integer, got bool: {value!r}")
                idx, _ = self._chip._resolve_device_index(device_key)
                ground_label[idx] = value

        excited_label = list(ground_label)
        excited_label[idx_target] += 1

        eigenvalues, _, _, kernel_labeling = self._compute_array_labeled()
        precomputed = (eigenvalues, kernel_labeling)
        return (
            self._eigenvalue_of_label(tuple(excited_label), precomputed=precomputed)
            - self._eigenvalue_of_label(tuple(ground_label), precomputed=precomputed)
        )

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
            evaluated; unlisted devices stay in their ground state.
        """
        if target is None:
            return self._dressed_frequencies()
        return self._transition_freq(target, when=when)

    def frame_info(self) -> dict[str, Any]:
        """Per-device frame reference frequency ``ω_ref,i`` (GHz).

        Resolves the chip's current frame spec through the same path the
        engine uses (:func:`quchip.engine.stage1_frames.resolve_frame`) and
        returns a flat ``{device_label: ω_ref,i}`` dict. These are the
        concrete frequencies the assembler will subtract as
        ``-Σ_i ω_ref,i n̂_i`` in :func:`stage2_assembly._build_static_h0`,
        so it exposes what will be solved without running the solver.

        Values are returned as produced by the frame resolver: concrete
        Python floats in ``"lab"``, ``"rotating"``, or scalar modes, and
        potentially JAX tracers in ``dict`` mode when the user wired a
        traced reference frequency through. Traced values are passed
        through unchanged to preserve differentiability.
        """
        resolved = resolve_frame(self._chip, self._chip.frame)
        return dict(resolved.frequencies)

    def state(
        self,
        device_states: Mapping[str | BaseDevice, int] | str | None = None,
        /,
        **device_state_kwargs: int,
    ) -> State:
        """Dressed eigenstate for Fock-indexed bare-state labels.

        Validates the mapping (rejects ``bool``, non-int, and
        out-of-range indices) and returns the assigned dressed
        eigenvector. A ``str`` shorthand (e.g. ``"eg1"``) is parsed
        through :func:`~quchip.chip.states.normalize_device_state_mapping`
        when :meth:`Chip.set_state_order` has been called. Use
        :meth:`Chip.bare_state` for arbitrary kets.

        Safe inside ``jax.jit``/``grad``/``vmap``: under tracing the
        eigenvector column is selected through the array kernel, so a
        dressed initial state is differentiable end-to-end.
        """
        resolved = normalize_device_state_mapping(self._chip, device_states, device_state_kwargs)
        self._label_from_resolved(resolved)
        return self._dressed_state(**resolved)
