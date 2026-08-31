"""Engine-owned input-output assembly for stationary port calculations.

Port channels enter the engine once as :class:`~quchip.engine.ir.PortTerm`
objects.  This module derives coherent input Hamiltonians, output coupling
operators, and dense Liouvillians from those resolved terms so analysis code
never reconstructs channel physics from the authored chip.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from quchip.engine.ir import CanonicalOperator, EngineResult, StaticTerm
from quchip.utils.jax_utils import maybe_concrete_scalar


# These routines form a dense d^2 by d^2 Liouvillian.  d=16 is already a
# 256-by-256 complex matrix; larger stationary states remain available through
# backend-native steady-state solvers without paying this analysis cost.
MAX_DENSE_INPUT_OUTPUT_DIMENSION = 16


def port_operators(engine: EngineResult, backend: Any) -> dict[str, CanonicalOperator]:
    """Return each resolved ``L_p = exp(i phi) sqrt(kappa) A_p`` operator."""
    xp = backend.array_module
    operators: dict[str, CanonicalOperator] = {}
    for term in engine.port_terms:
        values = (
            xp.exp(1j * xp.asarray(term.phase))
            * xp.sqrt(xp.asarray(term.rate))
            * term.operator.to_dense()
        )
        operators[term.label] = CanonicalOperator.from_dense(
            values,
            dims=term.operator.dims,
            basis=term.operator.basis,
            subsystem_labels=term.operator.subsystem_labels,
            tag=f"port:{term.label}",
        )
    return operators


def add_port_inputs(
    engine: EngineResult,
    backend: Any,
    tones: tuple[tuple[str, Any, Any], ...],
) -> EngineResult:
    """Add stationary ``i(beta L† - beta* L)`` terms in angular units."""
    if not tones:
        return engine
    xp = backend.array_module
    operators = port_operators(engine, backend)
    port_terms = {term.label: term for term in engine.port_terms}
    terms = list(engine.static_terms)
    for port_label, frequency, amplitude in tones:
        if port_label not in port_terms:
            raise ValueError(f"Unknown resolved port {port_label!r}.")
        if not same_frequency(port_terms[port_label].frame_frequency, frequency):
            raise ValueError(
                f"Tone at {frequency!r} GHz is not stationary for port {port_label!r} "
                f"in its resolved frame ({port_terms[port_label].frame_frequency!r} GHz). "
                "Use QuantumSequence for time evolution."
            )
        coupling = operators[port_label]
        values = coupling.to_dense()
        h_input = 1j * (
            xp.asarray(amplitude) * xp.conj(xp.swapaxes(values, -1, -2))
            - xp.conj(xp.asarray(amplitude)) * values
        )
        terms.append(
            StaticTerm(
                CanonicalOperator.from_dense(
                    h_input,
                    dims=coupling.dims,
                    basis=coupling.basis,
                    subsystem_labels=coupling.subsystem_labels,
                    tag=f"input:{port_label}",
                ),
                origin="port",
                metadata={"port": port_label},
            )
        )
    return replace(engine, static_terms=tuple(terms))


def dense_liouvillian(engine: EngineResult, backend: Any, *, operation: str) -> Any:
    """Build the dense static Liouvillian used by response/correlation algebra."""
    dimension = int(np.prod(engine.dims, dtype=int))
    if dimension > MAX_DENSE_INPUT_OUTPUT_DIMENSION:
        raise ValueError(
            f"{operation} currently supports total Hilbert dimension <= "
            f"{MAX_DENSE_INPUT_OUTPUT_DIMENSION}; got {dimension}. Reduce truncation "
            "or use backend-native correlation tools."
        )
    xp = backend.array_module
    terms = [
        xp.asarray(term.coefficient) * xp.asarray(term.operator.to_dense())
        for term in engine.static_terms
    ]
    if not terms:
        raise ValueError(f"{operation} requires a static Hamiltonian.")
    hamiltonian = sum(terms[1:], start=terms[0])
    identity = xp.eye(dimension, dtype=complex)
    liouvillian = -1j * (
        xp.kron(identity, hamiltonian)
        - xp.kron(xp.swapaxes(hamiltonian, -1, -2), identity)
    )
    for term in engine.collapse_terms:
        collapse = xp.sqrt(xp.asarray(term.rate)) * xp.asarray(term.operator.to_dense())
        cdc = xp.conj(xp.swapaxes(collapse, -1, -2)) @ collapse
        liouvillian = liouvillian + xp.kron(xp.conj(collapse), collapse)
        liouvillian = liouvillian - 0.5 * xp.kron(identity, cdc)
        liouvillian = liouvillian - 0.5 * xp.kron(xp.swapaxes(cdc, -1, -2), identity)
    return liouvillian


def small_signal_response(
    engine: EngineResult,
    state: Any,
    backend: Any,
    operators: dict[str, CanonicalOperator],
    input_label: str,
    output_labels: tuple[str, ...],
) -> dict[str, Any]:
    """Solve the zero-frequency linear response around one stationary state."""
    xp = backend.array_module
    rho = xp.asarray(backend.to_array(state), dtype=complex)
    dimension = rho.shape[0]
    input_operator = xp.asarray(operators[input_label].to_dense())
    input_dag = xp.conj(xp.swapaxes(input_operator, -1, -2))
    source = -(input_dag @ rho - rho @ input_dag).T.reshape(-1)
    matrix = dense_liouvillian(engine, backend, operation="Small-signal VNA response")
    trace_row = xp.eye(dimension, dtype=complex).T.reshape(-1)
    constrained = matrix.at[0].set(trace_row) if hasattr(matrix, "at") else matrix.copy()
    if not hasattr(matrix, "at"):
        constrained[0] = trace_row
    source = source.at[0].set(0.0) if hasattr(source, "at") else source.copy()
    if not hasattr(source, "at"):
        source[0] = 0.0
    derivative = xp.linalg.solve(constrained, source).reshape((dimension, dimension)).T
    return {
        label: (1.0 if label == input_label else 0.0)
        - xp.trace(xp.asarray(operators[label].to_dense()) @ derivative)
        for label in output_labels
    }


def same_frequency(first: Any, second: Any) -> bool:
    """Compare concrete frequencies without forcing traced values to Python."""
    if first is second:
        return True
    first_value = maybe_concrete_scalar(first)
    second_value = maybe_concrete_scalar(second)
    return (
        first_value is not None
        and second_value is not None
        and first_value == second_value
    )
