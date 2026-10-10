"""Combined view over per-component solves of a partitioned chip.

Holds the K component :class:`~quchip.results.results.SimulationResult` objects
and answers observable queries locally. The joint state is materialized only on
explicit request, because rebuilding it costs the full tensor-product space
that the partition avoided.
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
    """Result of one partitioned solve: K component results plus a key plan.

    Attributes
    ----------
    component_results : sequence of SimulationResult
        Per-component results in ``partition.components`` order.
    partition : object
        Component and device ownership metadata.
    key_plan : mapping
        Mapping from requested observable keys to component traces.
    """

    def __init__(self, component_results: list, partition: Any, key_plan: dict) -> None:
        """Wrap the per-component solves produced by one partitioned run.

        *component_results* must align with ``partition.components``. The
        result at index ``i`` solves ``partition.components[i].chip``. When you
        build this object by hand instead of via
        :func:`~quchip.engine.partitioned.maybe_simulate_partitioned`, keep
        that order.
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
    def dissipation(self) -> bool:
        """Return the shared dissipation choice of this partitioned solve."""
        choices = {result.dissipation for result in self._results}
        if len(choices) != 1:
            raise ValueError("Component results have different dissipation choices.")
        return choices.pop()

    @property
    def partition(self) -> Any:
        return self._partition

    @property
    def device_order(self) -> tuple[str, ...]:
        """Return the parent chip's original device-label order.

        This is *not* the concatenation of each component's labels, which
        follows connected-component discovery order. That order can interleave
        differently when the chip's device order does not already group each
        component's members together. The result permutes :attr:`states` and
        :attr:`final_state` into this order, so they match the joint solve
        exactly.
        """
        return self._partition.chip_order

    def _normalize_key(self, key: Any) -> Any:
        return tuple(resolve_label(k) for k in key) if isinstance(key, tuple) else resolve_label(key)

    def _local_values(self, entry: Any, index: int | None) -> Any:
        effective = entry.index if entry.index is not None else index
        return self._results[entry.component].expect(entry.key, effective)

    def expect(self, key: Any, index: int | None = None) -> Any:
        """Return one captured expectation trace.

        Parameters
        ----------
        key : object or str
            Captured observable key.
        index : int or None, optional
            Entry of a list-valued observable.
        """
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

    def observable_at(self, t: Any, values: Any, *, method: str = "exact") -> Any:
        """Select or linearly interpolate observable values on the shared grid.

        Parameters
        ----------
        t : scalar or array_like
            Query times in ns.
        values : array_like
            Saved values with time on the final axis.
        method : {"exact", "nearest", "interpolate"}, default="exact"
            Time-selection rule.
        """
        return self._results[0].observable_at(t, values, method=method)

    def expect_final(self, key: Any, index: int | None = None) -> Any:
        """Return the final expectation value for an observable.

        Parameters
        ----------
        key : object or str
            Captured observable key.
        index : int or None, optional
            Entry of a list-valued observable.
        """
        return self.expect(key, index)[-1]

    expect_values = expect

    def _owner_result(self, device: Any) -> Any:
        return self._results[self._partition.owner_of(device)]

    def iq_readout(self, output: Any, **kwargs: Any) -> Any:
        """Build a detector from the component owning an exposed output.

        Parameters
        ----------
        output : port object or str
            External output reference plane.
        **kwargs
            Forwarded to :meth:`SimulationResult.iq_readout`.
        """
        label = resolve_label(output)
        owners = [result for result in self._results if result._readout_wiring is not None
                  and label in result._readout_wiring.exposed]
        if len(owners) != 1:
            raise ValueError(f"Output {label!r} must belong to exactly one partition; inspect component results.")
        return owners[0].iq_readout(output, **kwargs)

    def measure(self, *devices: Any, t: Any = None, basis: Any = "energy") -> Any:
        """Measure kept states in local energy bases, without further evolution.

        Pass multiple devices for joint outcomes, t for an exact saved time, or
        basis='solver'. Write custom local unitary columns in the captured energy
        basis of the stored integration frame. Give one matrix for one device, or
        a device mapping. No phase-frame conversion is applied. Samples at
        different times represent independently terminated experiments.

        Parameters
        ----------
        *devices : device object or str
            Measured devices. An empty selection measures all devices.
        t : scalar, array_like, or None, optional
            Saved time or times in ns. ``None`` selects the final state.
        basis : {"energy", "solver"}, array_like, or mapping, default="energy"
            Captured local measurement basis.
        """
        from quchip.results.terminal import measure_result
        return measure_result(self, devices, t=t, basis=basis)

    def population(self, device: Any, level: int = 0) -> Any:
        """Return occupation of a captured isolated energy level.

        Parameters
        ----------
        device : device object or str
            Captured device.
        level : int, default=0
            Zero-based isolated energy level.
        """
        return self._owner_result(device).population(device, level)

    def check_truncation(self, threshold: float = 1e-3) -> dict[str, float]:
        """Run each component's truncation check and merge the per-device results.

        The return shape is the same as for
        :meth:`~quchip.results.results.SimulationResult.check_truncation` (a
        ``dict`` keyed by device label), which gives duck-typing parity between
        joint and partitioned results.

        Parameters
        ----------
        threshold : float, default=1e-3
            Maximum accepted boundary population.
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
        current_labels = [label for comp in self._partition.components for label in comp.labels]
        index_of = {label: i for i, label in enumerate(current_labels)}
        order = [index_of[label] for label in self._partition.chip_order]
        if order == list(range(len(order))):
            return None
        dims = [d for result in self._results for d in result.dims]
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
        order, which can interleave differently from the parent chip's device
        order. Each reconstructed step is permuted
        (:meth:`~quchip.backend.protocol.Backend.permute_state`) into
        :attr:`device_order`, so the result matches a joint solve of the
        original chip exactly.

        See also :attr:`final_state`. Unlike this accessor, :attr:`final_state`
        intentionally also accepts density-matrix components, because a tensor
        product of component density matrices is itself a valid joint state. A
        per-step list of joint kets is well-defined only when every component
        stayed pure.
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
        """Reconstruct the joint final state in :attr:`device_order`, as a ket
        if all components stayed pure, else a density matrix.

        When components disagree, it first promotes every component to a density
        matrix and then tensors them into a valid joint density matrix. Components
        tensor together in connected-component discovery order, which can
        interleave differently from the parent chip's device order. The result is
        permuted (:meth:`~quchip.backend.protocol.Backend.permute_state`) into
        :attr:`device_order`, so it matches a joint solve of the original chip
        exactly.
        """
        # Promotion of the components to a common density-matrix state kind uses
        # `_promote_to_common_state_kind`.
        return self._joint_state([r.final_state for r in self._results])

    def state_at(self, t: Any, *, method: str = "exact") -> Any:
        """Reconstruct a joint state at a saved time.

        Parameters
        ----------
        t : scalar
            Query time in ns.
        method : {"exact", "nearest"}, default="exact"
            Time-selection rule.
        """
        return self._joint_state([r.state_at(t, method=method) for r in self._results])

    def _joint_state(self, states: list) -> Any:
        warnings.warn(_JOINT_WARNING, UserWarning, stacklevel=3)
        backend = self._results[0]._backend
        joint = backend.tensor_states(*self._promote_to_common_state_kind(backend, states))
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
        """Raise a directed failure for any accessor that this class does not implement.

        ``PartitionedSimulationResult`` aggregates only the surface defined
        above (``expect``, ``population``, ``states``/``final_state``,
        ``check_truncation``, ...). It does not re-implement every
        :class:`~quchip.results.results.SimulationResult` method. Use a missing
        member per component (``result.components[i].<name>``), or rerun with
        ``partition=False`` for a full-fidelity joint
        :class:`~quchip.results.results.SimulationResult` that has it.
        """
        raise AttributeError(
            f"PartitionedSimulationResult has no '{name}'. Use "
            f"'.components[i].{name}' for a per-component result, or "
            "simulate(..., partition=False) for a full-fidelity joint "
            "SimulationResult that implements it."
        )
