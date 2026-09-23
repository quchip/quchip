"""Structural lowering for exact passive-linear input-output response."""

from __future__ import annotations

from typing import Any

import jax

from quchip.approximations import Approximation
from quchip.chip.ports import Port
from quchip.declarative.expr import PhysicsExpr, _bound_values, _walk_expr, materialize_expr
from quchip.devices.spaces import FockSpace
from quchip.engine.assembly import _apply_2pi_scalar
from quchip.engine.ir import LinearResponseProblem
from quchip.engine.output_network import pad_mixing
from quchip.engine.reference import cw_transfer
from quchip.utils.jax_utils import maybe_concrete_scalar


class _UnsupportedLinearModel(Exception):
    """Signal that a valid chip requires the general stationary solver."""


def try_build_linear_response_problem(
    chip: Any,
    frequencies: Any,
    *,
    plane_labels: tuple[str, ...],
) -> LinearResponseProblem | None:
    """Return a compact passive-linear problem, or ``None`` for fallback."""
    try:
        return _build_linear_response_problem(chip, frequencies, plane_labels=plane_labels)
    except _UnsupportedLinearModel:
        return None


def _build_linear_response_problem(
    chip: Any,
    frequencies: Any,
    *,
    plane_labels: tuple[str, ...],
) -> LinearResponseProblem:
    network = chip.port_network
    if network is None:
        raise _UnsupportedLinearModel
    if chip.dynamic_contributions() or chip.effective_terms:
        raise _UnsupportedLinearModel

    labels = tuple(device.label for device in chip.devices)
    mode_index = {label: index for index, label in enumerate(labels)}
    if any(
        not isinstance(device.local_space(), FockSpace)
        or chip.resolve_basis(device) != "native"
        for device in chip.devices
    ):
        raise _UnsupportedLinearModel
    if any(device.thermal_occupation is not None for device in chip.devices):
        raise _UnsupportedLinearModel

    backend = chip.backend
    xp = backend.array_module
    hamiltonian = xp.zeros((len(labels), len(labels)), dtype=complex)
    for device in chip.devices:
        hamiltonian = _add_hamiltonian_expr(
            hamiltonian,
            device.unresolved_hamiltonian(),
            chip.approximation,
            mode_index,
            backend,
            local=True,
        )
    for coupling in chip.couplings:
        hamiltonian = _add_hamiltonian_expr(
            hamiltonian,
            coupling.interaction_hamiltonian(),
            chip.approximation,
            mode_index,
            backend,
        )

    raw_ports: dict[str, Any] = {}
    for port in network.ports:
        raw_ports[port.label] = _port_coupling_vector(port, chip, mode_index, backend)

    compiled = network._compile()
    if compiled.output_network is not None:
        raise _UnsupportedLinearModel
    exposure_couplings = xp.stack(
        [
            sum(
                (
                    xp.asarray(coefficient) * raw_ports[source]
                    for source, coefficient in channel.coupling.items()
                ),
                start=xp.zeros((len(labels),), dtype=complex),
            )
            for channel in compiled.channels
        ]
    )
    for downstream, upstream, coefficient in compiled.generated_pairs:
        product = xp.outer(
            xp.conj(raw_ports[downstream]),
            xp.asarray(coefficient) * raw_ports[upstream],
        )
        hamiltonian = hamiltonian + (product - xp.conj(xp.swapaxes(product, -1, -2))) / (2j)

    hidden_couplings: list[Any] = []
    for operator, rate, _support, _source, _channel, _paths, owner in chip._collapse_contributions_with_owners():
        if isinstance(owner, Port):
            continue
        vector = _linear_operator_vector(operator, mode_index, backend)
        resolved_rate = _scalar_value(rate, backend)
        hidden_couplings.append(xp.sqrt(xp.asarray(resolved_rate)) * vector)

    couplings = (
        exposure_couplings
        if not hidden_couplings
        else xp.concatenate((exposure_couplings, xp.stack(hidden_couplings)), axis=0)
    )
    network_size = len(compiled.channels)
    full_size = network_size + len(hidden_couplings)
    scattering = pad_mixing(xp.asarray(compiled.scattering, dtype=complex), full_size, xp)
    external_labels = tuple(channel.exposure.label for channel in compiled.channels if not channel.exposure._hidden)
    try:
        plane_indices = tuple(external_labels.index(label) for label in plane_labels)
    except ValueError as error:
        raise ValueError(
            f"Unknown linear-response exposure. Available exposures: {list(external_labels)}"
        ) from error
    frequency_values = xp.asarray(frequencies, dtype=float)
    external_planes = [channel.reference for channel in compiled.channels if not channel.exposure._hidden]

    def transfer_columns(runs: list[Any]) -> Any:
        return xp.stack(
            [
                xp.broadcast_to(cw_transfer(run, frequency_values, xp), frequency_values.shape)
                for run in runs
            ],
            axis=1,
        )

    inbound_transfer = transfer_columns([plane.inbound for plane in external_planes])
    outbound_transfer = transfer_columns([plane.outbound for plane in external_planes])
    return LinearResponseProblem(
        frequencies=frequencies,
        mode_labels=labels,
        hamiltonian=hamiltonian,
        couplings=couplings,
        scattering=scattering,
        plane_indices=plane_indices,
        inbound_transfer=inbound_transfer,
        outbound_transfer=outbound_transfer,
        field_channels=network._field_channels(compiled),
    )


def _add_hamiltonian_expr(
    matrix: Any,
    expression: Any,
    approximation: Approximation,
    mode_index: dict[str, int],
    backend: Any,
    *, local: bool = False,
) -> Any:
    if not isinstance(expression, PhysicsExpr):
        raise _UnsupportedLinearModel
    for coefficient, factors in _operator_terms(expression, backend):
        weights = tuple(
            sum(1 if kind == "adag" else -1 for factor_label, kind in factors if factor_label == label)
            for label in expression.labels
        )
        if local and sum(weights) != 0:
            # Active local terms can be stationary in their authored frame;
            # a weight-only RWA cannot prove that they are off resonance.
            raise _UnsupportedLinearModel
        if not approximation.keeps_operator_band(weights):
            continue
        if not factors:
            continue
        creators = [label for label, kind in factors if kind == "adag"]
        annihilators = [label for label, kind in factors if kind == "a"]
        if len(factors) != 2 or len(creators) != 1 or len(annihilators) != 1:
            raise _UnsupportedLinearModel
        matrix = _add_entry(
            matrix,
            mode_index[creators[0]],
            mode_index[annihilators[0]],
            _apply_2pi_scalar(coefficient),
        )
    return matrix


def _port_coupling_vector(port: Port, chip: Any, mode_index: dict[str, int], backend: Any) -> Any:
    xp = backend.array_module
    targets = port.resolve_targets(chip)
    if port.operator is None or isinstance(port.operator, str) and port.operator == "a":
        if len(targets) != 1:
            raise _UnsupportedLinearModel
        vector = xp.zeros((len(mode_index),), dtype=complex)
        vector = _add_entry(vector, mode_index[targets[0]], None, 1.0)
    elif isinstance(port.operator, PhysicsExpr):
        vector = _linear_operator_vector(port.operator, mode_index, backend)
    else:
        raise _UnsupportedLinearModel
    return (
        xp.exp(1j * xp.asarray(port.phase))
        * xp.sqrt(xp.asarray(port.rate_value(chip)))
        * vector
    )


def _linear_operator_vector(expression: Any, mode_index: dict[str, int], backend: Any) -> Any:
    if not isinstance(expression, PhysicsExpr):
        raise _UnsupportedLinearModel
    xp = backend.array_module
    vector = xp.zeros((len(mode_index),), dtype=complex)
    saw_term = False
    for coefficient, factors in _operator_terms(expression, backend):
        if len(factors) != 1 or factors[0][1] != "a":
            raise _UnsupportedLinearModel
        vector = _add_entry(vector, mode_index[factors[0][0]], None, coefficient)
        saw_term = True
    if not saw_term:
        raise _UnsupportedLinearModel
    return vector


def _operator_terms(expression: PhysicsExpr, backend: Any) -> list[tuple[Any, tuple[tuple[str, str], ...]]]:
    bindings = _bound_values(expression)

    def expand(node: PhysicsExpr) -> list[tuple[Any, tuple[tuple[str, str], ...]]]:
        if not node.labels:
            return [(_scalar_value(node, backend, bindings=bindings), ())]
        if node.kind == "level":
            hamiltonian = node.args[0]
            if not isinstance(hamiltonian, PhysicsExpr):
                raise _UnsupportedLinearModel
            number = ((node.labels[0], "adag"), (node.labels[0], "a"))
            # Only a positive harmonic spectrum proves that energy order is
            # Fock order. Unknown or reordered spectra use the general solver.
            with jax.ensure_compile_time_eval():
                terms = expand(hamiltonian)
                if any(factors not in ((), number) for _, factors in terms):
                    raise _UnsupportedLinearModel
                frequency = maybe_concrete_scalar(sum(coefficient for coefficient, factors in terms if factors))
            if frequency is None or not 0 < frequency < float("inf"):
                raise _UnsupportedLinearModel
            return [(1.0, number)]
        if node.kind == "op":
            if type(node.args[1]) is not FockSpace:
                raise _UnsupportedLinearModel
            name = node.args[0]
            factors: tuple[tuple[str, str], ...]
            if name == "a":
                factors = ((node.labels[0], "a"),)
            elif name == "adag":
                factors = ((node.labels[0], "adag"),)
            elif name == "n":
                factors = ((node.labels[0], "adag"), (node.labels[0], "a"))
            elif name == "I":
                factors = ()
            else:
                raise _UnsupportedLinearModel
            return [(1.0, factors)]
        if node.kind == "embed":
            return expand(node.args[0])
        if node.kind in {"add", "sub"}:
            left = expand(node.args[0])
            right = expand(node.args[1])
            return left + ([(-coefficient, factors) for coefficient, factors in right] if node.kind == "sub" else right)
        if node.kind == "scale":
            scalar, operator = node.args
            value = _scalar_value(scalar, backend, bindings=bindings)
            return [(value * coefficient, factors) for coefficient, factors in expand(operator)]
        if node.kind in {"matmul", "tensor"}:
            left, right = expand(node.args[0]), expand(node.args[1])
            return [
                (left_coefficient * right_coefficient, left_factors + right_factors)
                for left_coefficient, left_factors in left
                for right_coefficient, right_factors in right
            ]
        raise _UnsupportedLinearModel

    return expand(expression)


def _scalar_value(
    expression: Any,
    backend: Any,
    *,
    bindings: dict[str, Any] | None = None,
) -> Any:
    if not isinstance(expression, PhysicsExpr):
        return expression
    allowed = {"literal", "parameter", "function", "add", "sub", "mul", "scale", "pow"}
    if any(node.labels or node.kind not in allowed for node in _walk_expr(expression)):
        raise _UnsupportedLinearModel
    return materialize_expr(expression, backend, bindings=bindings)


def _add_entry(array: Any, row: int, column: int | None, value: Any) -> Any:
    index = row if column is None else (row, column)
    if hasattr(array, "at"):
        return array.at[index].add(value)
    array[index] += value
    return array
