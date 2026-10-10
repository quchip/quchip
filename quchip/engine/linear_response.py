"""Structural lowering for exact passive-linear input-output response."""

from __future__ import annotations

from typing import Any

import jax

from quchip.approximations import Approximation
from quchip.chip.ports import Port
from quchip.declarative.expr import PhysicsExpr, _bound_values, _walk_expr, materialize_expr
from quchip.devices.base import BaseDevice
from quchip.devices.spaces import FockSpace
from quchip.engine.assembly import _apply_2pi_scalar
from quchip.engine.ir import LinearResponseProblem
from quchip.engine.output_network import pad_mixing
from quchip.engine.reference import cw_transfer
from quchip.utils.jax_utils import maybe_concrete_scalar


class _UnsupportedLinearModel(Exception):
    """Signal that a valid chip requires the general stationary solver."""


def is_linear_mode(device: Any, backend: Any) -> bool:
    """Return if the authored local Hamiltonian is a passive harmonic Fock mode."""
    from quchip.approximations import Exact

    if not isinstance(device.local_space(), FockSpace) or device._time_terms():
        return False
    try:
        _add_hamiltonian_expr(backend.array_module.zeros((1, 1), dtype=complex),
                              device.unresolved_hamiltonian(), Exact(),
                              {device.label: 0}, backend, local=True)
    except _UnsupportedLinearModel:
        return False
    return True


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


def try_build_weak_probe_problem(
    chip: Any,
    frequencies: Any,
    *,
    plane_labels: tuple[str, ...],
) -> tuple[LinearResponseProblem, str] | None:
    """Return a compact problem for an infinitesimal probe and its route, or ``None``.

    A passive harmonic chip keeps its mode equations (``"linear_response"``).
    An excitation-conserving chip with a stationary vacuum is projected onto its
    vacuum and one-excitation states (``"vacuum_response"``). Both are exact for
    an infinitesimal probe. Only the harmonic route stays exact for a finite
    probe or noisy inputs.
    """
    problem = try_build_linear_response_problem(chip, frequencies, plane_labels=plane_labels)
    if problem is not None:
        return problem, "linear_response"
    try:
        return _build_vacuum_response_problem(chip, frequencies, plane_labels=plane_labels), "vacuum_response"
    except _UnsupportedLinearModel:
        return None


def _build_vacuum_response_problem(
    chip: Any,
    frequencies: Any,
    *,
    plane_labels: tuple[str, ...],
) -> LinearResponseProblem:
    """Project an excitation-conserving chip onto its vacuum and one-excitation states.

    The static Hamiltonian must conserve the total energy-level index ``N``,
    every port must lower ``N`` by one, and every other channel must either
    lower ``N`` by one or conserve it. With vacuum inputs the vacuum is then
    stationary, and to first order in the probe the coherences ``|1_j><0|``
    evolve in the one-excitation block alone, whatever the anharmonicities,
    cross-Kerr terms or cutoffs. A conserving channel with ``L|0> = c|0>`` adds
    ``c* L_1 - |c|²/2 - (L†L)_1 / 2`` to that block.
    """
    from quchip.chip.effective import authored_excitation_changes, conserves_excitation_number

    if (chip.port_network is None or chip.dynamic_contributions()
            or not conserves_excitation_number(chip, chip.approximation)):
        raise _UnsupportedLinearModel
    backend = chip.backend
    labels = tuple(device.label for device in chip.devices)
    resolved = chip.resolve(frame="lab")
    bases = resolved.bases
    for operator, rate, support, _source, _channel, _paths, owner in chip._collapse_contributions_with_owners(bases):
        if maybe_concrete_scalar(_scalar_value(rate, backend)) == 0.0:
            continue
        operator_labels = tuple(labels[index] for index in support) if support else labels
        changes = authored_excitation_changes(operator, operator_labels, backend, bases)
        lowering = changes is not None and changes <= {1}
        if not lowering and (changes != {0} or isinstance(owner, Port)):
            raise _UnsupportedLinearModel

    slh = resolved.slh
    if slh.H.dynamic_terms or slh.output_network is not None:
        raise _UnsupportedLinearModel
    if any(channel.input_occupation is not None
           and maybe_concrete_scalar(channel.input_occupation) != 0.0 for channel in slh.channels):
        raise _UnsupportedLinearModel

    xp = backend.array_module
    records = [bases[label] for label in labels]
    excited = tuple(index for index, record in enumerate(records) if record.resolved_dim > 1)

    def product_state(raised: int | None) -> Any:
        state = xp.ones((1,), dtype=complex)
        for index, record in enumerate(records):
            state = xp.kron(state, xp.asarray(record.energy_state(1 if index == raised else 0), dtype=complex))
        return state

    vacuum = product_state(None)
    states = xp.stack([product_state(index) for index in excited], axis=1)
    identity = xp.eye(len(excited), dtype=complex)
    hamiltonian = sum(
        (term.coefficient * xp.asarray(term.operator.to_dense(), dtype=complex) for term in slh.H.static_terms),
        start=xp.zeros((vacuum.shape[0],) * 2, dtype=complex),
    )
    block = states.conj().T @ hamiltonian @ states - (vacuum.conj() @ hamiltonian @ vacuum) * identity
    rows = []
    for channel in slh.channels:
        coupling = xp.asarray(channel.coupling.to_dense(), dtype=complex)
        lifted = coupling @ states
        row = vacuum.conj() @ lifted
        shift = vacuum.conj() @ coupling @ vacuum
        block = block + 1j * (
            xp.conj(shift) * (states.conj().T @ lifted)
            - 0.5 * xp.abs(shift) ** 2 * identity
            - 0.5 * (lifted.conj().T @ lifted - xp.outer(row.conj(), row))
        )
        rows.append(row)

    keys = tuple(channel.key for channel in slh.channels)
    external = tuple(channel.key for channel in slh.external_channels)
    if any(label not in external for label in plane_labels):
        raise ValueError(f"Unknown linear-response exposure. Available exposures: {list(external)}")
    plane_indices = tuple(keys.index(label) for label in plane_labels)
    frequency_values = xp.asarray(frequencies, dtype=float)

    def transfer_columns(runs: list[Any]) -> Any:
        return xp.stack(
            [xp.broadcast_to(cw_transfer(run, frequency_values, xp), frequency_values.shape) for run in runs],
            axis=1,
        )

    return LinearResponseProblem(
        frequencies=frequencies,
        mode_labels=tuple(labels[index] for index in excited),
        hamiltonian=block,
        couplings=xp.stack(rows),
        scattering=xp.asarray(slh.S, dtype=complex),
        plane_indices=plane_indices,
        inbound_transfer=transfer_columns([channel.reference.inbound for channel in slh.channels]),
        outbound_transfer=transfer_columns([channel.reference.outbound for channel in slh.channels]),
    )


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
        if (len(targets) != 1
                or type(chip[targets[0]]).lowering_operator is not BaseDevice.lowering_operator):
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
