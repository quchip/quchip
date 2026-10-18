"""Structural lowering for exact passive-linear input-output response."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from quchip.approximations import Approximation
from quchip.backend import _backend_context
from quchip.chip.ports import Port
from quchip.declarative.expr import PhysicsExpr, _bound_values, _walk_expr, materialize_expr
from quchip.devices.base import BaseDevice
from quchip.devices.spaces import FockSpace
from quchip.engine.assembly import _apply_2pi_scalar
from quchip.engine.ir import LinearResponseProblem
from quchip.engine.output_network import pad_mixing
from quchip.engine.reference import ReferencePlane, cw_transfer
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import concrete_array_module, maybe_concrete_scalar, raise_if

# Two ill-conditioned poles closer than this multiple of their round-off are not resolved.
_POLE_RESOLUTION = 100.0


class _UnsupportedLinearModel(Exception):
    """Signal that a valid chip requires the general stationary solver.

    The message, when there is one, says which part of the model the route does not support.
    """


def is_linear_mode(device: Any, backend: Any) -> bool:
    """Return if the authored local Hamiltonian is a passive harmonic Fock mode."""
    from quchip.approximations import Exact

    if not isinstance(device.local_space(), FockSpace) or device._time_terms():
        return False
    try:
        with _backend_context(backend):
            authored = device.unresolved_hamiltonian()
        _add_hamiltonian_expr(backend.array_module.zeros((1, 1), dtype=complex),
                              authored, Exact(), {device.label: 0}, backend, local=True)
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
        with _backend_context(chip.backend):
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
        with _backend_context(chip.backend):
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
    ``c* L_1 - |c|²/2 - (L†L)_1 / 2`` to that block. Every block comes from the
    terms' local operators, so the full product space is never formed.
    """
    model = _sector_response_model(chip, plane_labels)
    sectors, xp = model.sectors, chip.backend.array_module
    labels = tuple(device.label for device in chip.devices)
    excited = np.argmax(sectors.space(1).levels, axis=1)
    order = np.argsort(excited, kind="stable")
    identity = xp.eye(order.size, dtype=complex)
    block = TWO_PI * (xp.asarray(sectors.hamiltonian(1)) - xp.asarray(sectors.hamiltonian(0))[0, 0] * identity)
    rows = []
    for index in range(len(model.channels)):
        shift = model.channel_block(index, 0, 0)[0, 0]
        kept = model.channel_block(index, 1, 1)
        block = block + 1j * (
            xp.conj(shift) * kept - 0.5 * xp.abs(shift) ** 2 * identity - 0.5 * kept.conj().T @ kept
        )
        rows.append(model.channel_block(index, 0, 1)[0])
    inbound, outbound = model.transfers(frequencies)
    problem = LinearResponseProblem(
        frequencies=frequencies,
        mode_labels=tuple(labels[index] for index in excited[order]),
        hamiltonian=block[np.ix_(order, order)],
        couplings=xp.stack(rows)[:, order],
        scattering=model.scattering,
        plane_indices=model.plane_indices,
        inbound_transfer=inbound,
        outbound_transfer=outbound,
    )

    return _with_mode_responses(problem, model, order)


def vacuum_drift(chip: Any) -> tuple[tuple[str, ...], Any] | None:
    """Return the mode labels and the drift ``-iH - C†C/2`` of the vacuum response, or ``None``.

    The modes are the one-excitation states of the devices, followed by the
    retained poles. ``None`` means that the chip has no vacuum one-excitation
    model.
    """
    try:
        problem = _build_vacuum_response_problem(chip, np.zeros(1), plane_labels=())
    except _UnsupportedLinearModel:
        return None
    return problem.mode_labels, response_drift(problem)


def response_drift(problem: LinearResponseProblem) -> Any:
    """Return the drift ``A = -iH - C†C/2`` of a weak-probe problem, in rad/ns.

    The backends solve ``S(f) = S_d + C (-i 2π f - A)^-1 (-C† S_d)`` with this
    drift, so its eigenvalues are the poles of the response.
    """
    xp = concrete_array_module(problem.hamiltonian, problem.couplings)
    couplings = xp.asarray(problem.couplings, dtype=complex)
    return -1j * xp.asarray(problem.hamiltonian, dtype=complex) - 0.5 * couplings.conj().T @ couplings


@dataclass(frozen=True)
class ResponsePoles:
    """Eigensystem of a weak-probe drift, with the poles in increasing frequency.

    Attributes
    ----------
    values : array_like
        Eigenvalues ``λ_p`` of the drift in rad/ns. They stay differentiable on
        a fixed structure away from degenerate poles.
    vectors : array_like
        Right eigenvectors as columns, with the derivative stopped, for
        discrete choices and checks.
    inverse : array_like
        Inverse of ``vectors``. Its rows are the left eigenvectors.
    drift : array_like
        The drift ``A`` in rad/ns.
    """

    values: Any
    vectors: Any
    inverse: Any
    drift: Any

    def modes(self) -> tuple[Any, Any]:
        """Return the right eigenvectors and their inverse in the order of ``values``.

        A JAX derivative that reaches them through the drift raises, because
        JAX does not differentiate the eigenvectors of a general matrix.
        """
        if concrete_array_module(self.drift) is np:
            return self.vectors, self.inverse
        return _undifferentiated_eigenvectors(self.vectors, self.inverse, self.drift)


@jax.custom_jvp
def _undifferentiated_eigenvectors(vectors: Any, inverse: Any, drift: Any) -> tuple[Any, Any]:
    return vectors, inverse


@_undifferentiated_eigenvectors.defjvp
def _undifferentiated_eigenvectors_jvp(primals: Any, tangents: Any) -> Any:
    raise NotImplementedError("The composition and the residues of the poles need eigenvector derivatives, "
                              "which JAX does not define for a general matrix.")


def _one_to_one(targets: Any, candidates: Any) -> Any:
    """Return, for each target in order, the index of the nearest candidate that no earlier target took."""
    xp = concrete_array_module(targets, candidates)
    taken = xp.zeros(candidates.shape[0], dtype=bool)
    picks = []
    for target in targets:
        pick = xp.argmin(xp.where(taken, xp.inf, xp.abs(candidates - target)))
        picks.append(pick)
        taken = taken | (xp.arange(candidates.shape[0]) == pick)
    return xp.stack(picks)


def response_poles(problem: LinearResponseProblem) -> ResponsePoles:
    """Return the poles of a weak-probe response, the eigenvalues of its drift.

    The eigensolves subtract the mean frequency of the drift, which keeps the
    pole splittings accurate. They keep its damping, so adding the mean back
    does not cancel digits of a narrow linewidth. Two ill-conditioned poles that are too close to
    resolve raise. This happens near an exceptional point, where two poles
    merge.
    """
    drift = response_drift(problem)
    xp = concrete_array_module(drift)
    count = drift.shape[0]
    center = 1j * xp.imag(xp.trace(drift)) / count
    shifted = drift - center * xp.eye(count)
    fixed = shifted if xp is np else jax.lax.stop_gradient(shifted)
    fxp = concrete_array_module(fixed)
    found, vectors = fxp.linalg.eig(fixed)
    values = found
    if xp is not np:
        values = jnp.linalg.eigvals(shifted)
        # Pair each differentiable eigenvalue with one fixed eigenpair. Repeated eigenvalues take distinct ones.
        nearest = _one_to_one(jax.lax.stop_gradient(values), found)
        found, vectors = found[nearest], vectors[:, nearest]
    order = fxp.argsort(-fxp.imag(found))
    found, vectors, values = found[order], vectors[:, order], values[order]
    inverse = fxp.linalg.inv(vectors)
    # A perturbation of the drift moves each eigenvalue by its condition number times its size.
    condition = fxp.linalg.norm(inverse, axis=1) * fxp.linalg.norm(vectors, axis=0)
    limit = (_POLE_RESOLUTION * np.finfo(float).eps * fxp.linalg.norm(fixed)
             * (condition[:, None] + condition[None, :]))
    meeting = ((fxp.abs(found[:, None] - found[None, :]) <= limit)
               & (fxp.minimum(condition[:, None], condition[None, :]) > 10.0) & ~fxp.eye(count, dtype=bool))
    frequencies = -fxp.imag(found + (center if xp is np else jax.lax.stop_gradient(center))) / TWO_PI
    raise_if(fxp.any(meeting), _unresolved_poles, meeting, frequencies)
    return ResponsePoles(values=values + center, vectors=vectors, inverse=inverse, drift=drift)


def _unresolved_poles(meeting: Any, frequencies: Any) -> ValueError:
    pairs = np.argwhere(np.triu(np.asarray(meeting), 1))
    places = ", ".join(f"{float(np.asarray(frequencies)[i]):.9g}" for i in sorted({int(i) for i, _ in pairs}))
    return ValueError(
        f"Poles of the weak-probe response near {places} GHz are ill-conditioned and too close to resolve "
        "reliably. This happens near an exceptional point, where two poles merge."
    )


def _with_mode_responses(problem: LinearResponseProblem, model: Any, order: Any) -> LinearResponseProblem:
    """Compose removed harmonic poles at their original port locations in mode space.

    For one pole, elimination of its amplitude gives
    ``S(f) = I - v v† / (sum(|v|²)/2 + kappa_i/2 - i 2π(f-f_r))``.
    Keeping this one scalar unknown in the frequency solve composes the full
    rank-one response, including reflections and later scattering. It does not
    add a device or a quantum state to the reduced chip.
    """
    sectors = model.sectors
    chip, compiled = sectors.chip, sectors.network
    network = chip.port_network
    modes = network._mode_responses
    if not modes:
        return problem
    backend = chip.backend
    xp = backend.array_module
    count, added = len(problem.mode_labels), len(modes)
    size = count + added
    raw = {}
    for port in network.ports:
        matrix, support = sectors.port_couplings()[port.label]
        kept = xp.asarray(sectors.block(matrix, support, 0, 1))[0, order]
        amplitude = xp.sqrt(xp.asarray(_scalar_value(port.rate_value(chip), backend))) * xp.exp(
            1j * xp.asarray(_scalar_value(port.phase, backend)))
        poles = xp.stack([amplitude * xp.sqrt(xp.asarray(_scalar_value(mode.weight, backend)))
                          if port.label in mode.ports else xp.asarray(0j) for mode in modes])
        raw[port.label] = xp.concatenate((kept, poles))
    h = xp.pad(problem.hamiltonian, ((0, added), (0, added)))
    frequencies = xp.asarray([_scalar_value(mode.freq, backend) for mode in modes])
    internal_rates = xp.asarray([_scalar_value(mode.internal_rate, backend) for mode in modes])
    # Internal loss damps only the pole amplitude. The diagonal -i kappa_i/2
    # gives the drift of an unread loss channel and stays differentiable at zero loss.
    h = h + xp.diag(xp.concatenate((xp.zeros(count), TWO_PI * frequencies - 0.5j * internal_rates)))
    # The retained block already contains the network Hamiltonian. Add only
    # the new rows and columns of each series-product term.
    for downstream, upstream, coefficient in compiled.generated_pairs:
        product = xp.outer(xp.conj(raw[downstream]), coefficient * raw[upstream])
        term = (product - product.conj().T) / (2j)
        h = h + term - xp.pad(term[:count, :count], ((0, added), (0, added)))
    exposed = xp.stack([
        sum((coefficient * raw[source] for source, coefficient in entry.coupling.items()),
            start=xp.zeros(size, dtype=complex)) for entry in compiled.channels
    ])
    hidden = xp.pad(problem.couplings[len(compiled.channels):], ((0, 0), (0, added)))
    return replace(
        problem, mode_labels=(*problem.mode_labels, *(mode.label for mode in modes)),
        hamiltonian=h, couplings=xp.concatenate((exposed, hidden), axis=0),
    )


@dataclass(frozen=True)
class _SectorResponseModel:
    """Sector operators and the compiled vacuum-input boundary shared by weak probes."""

    sectors: Any
    channels: list[list[tuple[Any, Any, tuple[int, ...]]]]
    scattering: Any
    plane_indices: tuple[int, ...]
    references: list[Any]
    conserving_indices: tuple[int, ...]
    keys: tuple[str, ...] = ()

    def channel_block(self, index: int, row_total: int, column_total: int) -> Any:
        xp = self.sectors.chip.backend.array_module
        shape = (self.sectors.space(row_total).size, self.sectors.space(column_total).size)
        return sum(
            (xp.asarray(coefficient) * xp.asarray(self.sectors.block(matrix, support, row_total, column_total))
             for coefficient, matrix, support in self.channels[index]),
            start=xp.zeros(shape, dtype=complex),
        )

    def transfers(self, frequencies: Any) -> tuple[Any, Any]:
        xp = self.sectors.chip.backend.array_module
        values = xp.atleast_1d(xp.asarray(frequencies, dtype=float))

        def columns(runs: list[Any]) -> Any:
            return xp.stack([xp.broadcast_to(cw_transfer(run, values, xp), values.shape) for run in runs], axis=1)

        return (columns([reference.inbound for reference in self.references]),
                columns([reference.outbound for reference in self.references]))


def _sector_response_model(chip: Any, plane_labels: tuple[str, ...]) -> _SectorResponseModel:
    """Assemble every channel on its local support and require a conserving vacuum-input model.

    A chip without a port network keeps only its device channels, and then
    ``plane_labels`` must be empty.
    """
    from quchip.engine.excitation import authored_excitation_changes, lowers_by_one

    network = chip.port_network
    if network is None and plane_labels:
        raise _UnsupportedLinearModel("The chip has no port network.")
    if chip.dynamic_contributions():
        raise _UnsupportedLinearModel("The chip has time-dependent contributions.")
    sectors = chip.analysis._sector_model()
    if sectors is None:
        raise _UnsupportedLinearModel("The static model does not conserve the total excitation N.")
    backend = chip.backend
    labels = tuple(device.label for device in chip.devices)
    channels: list[list[tuple[Any, Any, tuple[int, ...]]]] = []
    hidden_keys: list[str] = []
    occurrences: dict[str, int] = {}
    conserving_indices = []
    for operator, rate, support, source, name, _paths, owner in sectors.contributions:
        operator_labels = tuple(labels[index] for index in support) if support else labels
        changes = authored_excitation_changes(operator, operator_labels, backend, sectors.bases)
        if maybe_concrete_scalar(_scalar_value(rate, backend)) != 0.0:
            if not lowers_by_one(changes) and (changes != {0} or isinstance(owner, Port)):
                raise _UnsupportedLinearModel(
                    f"Channel {name!r} of {source!r} does not lower N by one" + (
                        "." if isinstance(owner, Port) else " or conserve it."))
        if not isinstance(owner, Port):
            if changes == {0}:
                conserving_indices.append(len(channels))
            # Hidden channels keep the keys and order of the resolved SLH model.
            channels.append([(1.0, sectors.channel_operator(operator, rate, support), support)])
            key = f"hidden.{source}.{name}"
            occurrences[key] = occurrences.get(key, 0) + 1
            hidden_keys.append(key if occurrences[key] == 1 else f"{key}#{occurrences[key]}")

    compiled = sectors.network
    entries = () if compiled is None else compiled.channels
    if compiled is not None and compiled.output_network is not None:
        raise _UnsupportedLinearModel("An output network mixes the outputs.")
    if network is not None and any(
            field.input_occupation is not None and maybe_concrete_scalar(field.input_occupation) != 0.0
            for field in network._field_channels(compiled)):
        raise _UnsupportedLinearModel("An input carries thermal noise.")
    if entries:
        ports = sectors.port_couplings()
        channels[:0] = [[(coefficient, *ports[source]) for source, coefficient in entry.coupling.items()]
                        for entry in entries]

    xp = backend.array_module
    keys = (*(entry.exposure.label for entry in entries), *hidden_keys)
    external = tuple(entry.exposure.label for entry in entries if not entry.exposure._hidden)
    if any(label not in external for label in plane_labels):
        raise ValueError(f"Unknown linear-response exposure. Available exposures: {list(external)}")
    plane_indices = tuple(keys.index(label) for label in plane_labels)
    references = [entry.reference for entry in entries] + [ReferencePlane()] * len(hidden_keys)
    scattering = (pad_mixing(xp.asarray(compiled.scattering, dtype=complex), len(keys), xp)
                  if entries else xp.eye(len(keys), dtype=complex))

    return _SectorResponseModel(sectors, channels, scattering, plane_indices, references,
                                tuple(len(entries) + i for i in conserving_indices), keys)


def _build_linear_response_problem(
    chip: Any,
    frequencies: Any,
    *,
    plane_labels: tuple[str, ...],
) -> LinearResponseProblem:
    network = chip.port_network
    if network is None:
        raise _UnsupportedLinearModel
    # Retained mode responses enter only the vacuum weak-probe composition.
    if chip.dynamic_contributions() or chip.effective_terms or network._mode_responses:
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
        xp.exp(1j * xp.asarray(_scalar_value(port.phase, backend)))
        * xp.sqrt(xp.asarray(_scalar_value(port.rate_value(chip), backend)))
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
    value = expression
    if isinstance(expression, PhysicsExpr):
        allowed = {"literal", "parameter", "function", "add", "sub", "mul", "scale", "pow"}
        if any(node.labels or node.kind not in allowed for node in _walk_expr(expression)):
            raise _UnsupportedLinearModel
        value = materialize_expr(expression, backend, bindings=bindings)
    # Compact response arrays consume scalars without constructing native operators.
    validate = getattr(backend, "_validate_input", None)
    if validate is not None:
        validate(value)
    return value


def _add_entry(array: Any, row: int, column: int | None, value: Any) -> Any:
    index = row if column is None else (row, column)
    if hasattr(array, "at"):
        return array.at[index].add(value)
    array[index] += value
    return array
