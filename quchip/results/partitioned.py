"""Combined view over per-component solves of a partitioned chip.

Holds the K component :class:`~quchip.results.results.SimulationResult`
objects and answers observable queries locally. The joint state is never
materialized unless explicitly requested — rebuilding it costs the full
tensor-product space the partition avoided.
"""

from __future__ import annotations

import warnings
from typing import Any

from quchip.utils.labeling import resolve_label

_JOINT_WARNING = (
    "Materializing the joint state of a partitioned result rebuilds the full "
    "tensor-product space the partition avoided."
)


class PartitionedSimulationResult:
    """Result of one partitioned solve: K component results + a key plan."""

    def __init__(self, component_results: list, partition: Any, key_plan: dict) -> None:
        """Wrap the per-component solves produced by one partitioned run.

        *component_results* must align with ``partition.components`` — the
        result at index ``i`` is the solve of ``partition.components[i].chip``.
        Callers that build this by hand (rather than via
        :func:`~quchip.engine.partitioned.maybe_simulate_partitioned`) must
        preserve that order.
        """
        if len(component_results) != len(partition.components):
            raise ValueError(
                f"component_results has {len(component_results)} entries but partition has "
                f"{len(partition.components)} components; they must align one-to-one, in "
                "partition.components order."
            )
        self._results = tuple(component_results)
        self._partition = partition
        self._key_plan = dict(key_plan)
        self.times = self._results[0].times

    @property
    def components(self) -> tuple:
        return self._results

    @property
    def partition(self) -> Any:
        return self._partition

    @property
    def device_order(self) -> tuple[str, ...]:
        """Return the parent chip's original device-label order.

        This is *not* the concatenation of each component's labels (which
        follows connected-component discovery order and can interleave
        differently whenever the chip's device order doesn't already group
        each component's members together) — it is the order
        :attr:`states` and :attr:`final_state` are permuted into so they
        match the joint solve exactly.
        """
        return self._partition.chip_order

    def _current_order(self) -> list[str]:
        """Return device labels in the order the per-component states concatenate into."""
        return [label for comp in self._partition.components for label in comp.labels]

    def _permutation_to_chip_order(self) -> tuple[list[int], list[int]]:
        """Return ``(dims, order)`` permuting a concatenated-component state into chip order.

        ``dims`` are the per-device levels in the *current* (concatenated
        component) order; ``order`` follows
        :meth:`~quchip.backend.protocol.Backend.permute_state`'s
        ``numpy.transpose`` convention.
        """
        current_labels = self._current_order()
        dims = [d for result in self._results for d in result.dims]
        index_of = {label: i for i, label in enumerate(current_labels)}
        order = [index_of[label] for label in self._partition.chip_order]
        return dims, order

    def _normalize_key(self, key: Any) -> Any:
        return tuple(resolve_label(k) for k in key) if isinstance(key, tuple) else resolve_label(key)

    def _local_values(self, entry: Any, index: int | None) -> Any:
        effective = entry.index if entry.index is not None else index
        return self._results[entry.component].expect(entry.key, effective)

    def expect(self, key: Any, index: int | None = None) -> Any:
        entry = self._key_plan.get(self._normalize_key(key))
        if entry is None:
            raise KeyError(f"No observable {key!r} in this partitioned result. Known: {list(self._key_plan)}")
        from quchip.chip.partition import CrossEop

        if isinstance(entry, CrossEop):
            if index is not None:
                raise ValueError(
                    f"Cross-component e_ops key {key!r} is a single correlator trace; "
                    f"'index' must be None, got {index!r}."
                )
            return self._local_values(entry.a, None) * self._local_values(entry.b, None)
        return self._local_values(entry, index)

    def expect_final(self, key: Any, index: int | None = None) -> Any:
        return self.expect(key, index)[-1]

    expect_values = expect

    def _owner_result(self, device: Any) -> Any:
        return self._results[self._partition.owner_of(device)]

    def population(self, device: Any, level: int = 0) -> Any:
        return self._owner_result(device).population(device, level)

    def population_array(self, device: Any, level: int = 0) -> Any:
        return self._owner_result(device).population_array(device, level)

    def check_truncation(self, threshold: float = 1e-3) -> dict[str, float]:
        """Run each component's truncation check and merge the per-device results.

        Mirrors :meth:`~quchip.results.results.SimulationResult.check_truncation`'s
        return shape (a ``dict`` keyed by device label) for duck-typing parity
        between a joint and a partitioned result.
        """
        merged: dict[str, float] = {}
        for result in self._results:
            merged.update(result.check_truncation(threshold=threshold))
        return merged

    def _chip_order_permutation(self) -> tuple[list[int], list[int]] | None:
        """Return ``(dims, order)`` for :meth:`~quchip.backend.protocol.Backend.permute_state`.

        Returns ``None`` when the concatenated component order already
        matches :attr:`device_order` and permuting would be a no-op.
        """
        dims, order = self._permutation_to_chip_order()
        if order == list(range(len(order))):
            return None
        return dims, order

    def _collect_component_ket_trajectories(self) -> list:
        """Collect each component's saved-state trajectory; requires every component ket-valued.

        Raises if a component solve didn't retain states, or stored density
        matrices instead of kets — joint-state reconstruction of a
        *trajectory* only supports the all-ket case (see :attr:`final_state`
        for the mixed ket/density-matrix case, which is well-defined for a
        single final state).
        """
        trajectories = []
        for result in self._results:
            if result.states is None:
                raise RuntimeError("A component solve did not store states.")
            if not result._is_ket_trajectory():
                raise NotImplementedError(
                    "Joint-state reconstruction is implemented for ket trajectories only."
                )
            trajectories.append(result.states)
        return trajectories

    def _promote_to_common_state_kind(self, backend: Any, states: list) -> list:
        """Promote every state to a density matrix when the list mixes kets and density matrices.

        A tensor product of component kets is itself a valid joint ket, and
        a tensor product of component density matrices is a valid joint
        density matrix — but a mix of the two is neither: tensoring a ket
        with a density matrix is a shape mismatch, not a physical state.
        """
        kets = [backend.is_ket(s) for s in states]
        if any(kets) and not all(kets):
            return [backend.as_density_matrix(s) for s in states]
        return states

    @property
    def states(self) -> list:
        """Reconstruct the joint-state trajectory (ket trajectories only), in :attr:`device_order`.

        Component states tensor together in connected-component discovery
        order, which can interleave differently from the parent chip's own
        device order; each reconstructed step is permuted
        (:meth:`~quchip.backend.protocol.Backend.permute_state`) into
        :attr:`device_order` so the result matches a joint solve of the
        original chip exactly.

        See also :attr:`final_state`, which — unlike this accessor —
        intentionally also accepts density-matrix components: a tensor
        product of component density matrices is itself a valid joint
        state, whereas a per-step list of joint kets is only well-defined
        when every component stayed pure.
        """
        warnings.warn(_JOINT_WARNING, UserWarning, stacklevel=2)
        backend = self._results[0]._backend
        per_component_trajectories = self._collect_component_ket_trajectories()
        joint_steps = [backend.tensor_states(*step) for step in zip(*per_component_trajectories)]
        permutation = self._chip_order_permutation()
        if permutation is None:
            return joint_steps
        dims, order = permutation
        return [backend.permute_state(state, dims, order) for state in joint_steps]

    @property
    def final_state(self) -> Any:
        """Reconstruct the joint final state, in :attr:`device_order` — a ket if
        every component stayed pure, otherwise a density matrix.

        When components disagree, every component is first promoted to a
        density matrix (:meth:`_promote_to_common_state_kind`) before
        tensoring, giving a valid joint density matrix. Components tensor
        together in connected-component discovery order, which can
        interleave differently from the parent chip's own device order; the
        result is permuted (:meth:`~quchip.backend.protocol.Backend.permute_state`)
        into :attr:`device_order` so it matches a joint solve of the
        original chip exactly.
        """
        warnings.warn(_JOINT_WARNING, UserWarning, stacklevel=2)
        backend = self._results[0]._backend
        finals = self._promote_to_common_state_kind(backend, [r.final_state for r in self._results])
        joint = backend.tensor_states(*finals)
        permutation = self._chip_order_permutation()
        if permutation is None:
            return joint
        dims, order = permutation
        return backend.permute_state(joint, dims, order)

    def describe(self) -> str:
        lines = [f"PartitionedSimulationResult: {len(self._results)} components"]
        for comp, result in zip(self._partition.components, self._results):
            lines.append(f"- {list(comp.labels)}: dims={tuple(result.dims)}, solver={result.solver}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PartitionedSimulationResult({len(self._results)} components, "
            f"devices={list(self.device_order)})"
        )

    def __getattr__(self, name: str) -> Any:
        """Raise a directed failure for any accessor this class doesn't implement.

        ``PartitionedSimulationResult`` only aggregates the surface defined
        above (``expect``, ``population``, ``states``/``final_state``,
        ``check_truncation``, ...) — it does not re-implement every
        :class:`~quchip.results.results.SimulationResult` method. Reach the
        missing member either per component (``result.components[i].<name>``)
        or by re-running with ``partition=False`` for a full-fidelity joint
        :class:`~quchip.results.results.SimulationResult` that has it.
        """
        raise AttributeError(
            f"PartitionedSimulationResult has no '{name}'. Use "
            f"'.components[i].{name}' for a per-component result, or "
            "simulate(..., partition=False) for a full-fidelity joint "
            "SimulationResult that implements it."
        )
