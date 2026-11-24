"""qutip-qip interoperability with ``to_qutip_qip`` and ``from_qutip_qip``.

``to_qutip_qip`` turns a gate set into a qutip-qip processor that runs
``QubitCircuit`` objects on the chip's full Hamiltonian
(:class:`~quchip.interop.qutip_qip.processor.ChipProcessor`).
``from_qutip_qip`` turns a ``QubitCircuit`` into gate-set instructions. Both
use the circuit API of qutip-qip 0.5. This module imports qutip-qip only when
one of them is called.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from quchip.gateset.ideals import VIRTUAL, canonical_name, rule_table

if TYPE_CHECKING:
    from quchip.gateset import GateSet

__all__ = ["from_qutip_qip", "to_qutip_qip"]

#: Gates that qutip-qip resolves a gate into when the gate set has no rule for it.
BASIS = ["CX", "RX", "RY", "RZ"]


def _require_qutip_qip() -> None:
    """Import qutip-qip on demand so quchip remains usable without it."""
    try:
        import qutip_qip
    except ModuleNotFoundError:
        raise ImportError(
            "qutip-qip is required for this feature. "
            "Install it with:  pip install quchip[qutip-qip]"
        ) from None
    version = tuple(int(part) for part in qutip_qip.__version__.split(".")[:2])
    if version < (0, 5):
        raise ImportError(f"quchip needs qutip-qip 0.5 or later; found qutip-qip {qutip_qip.__version__}.")


def to_qutip_qip(gates: "GateSet") -> Any:
    """Return a qutip-qip processor that runs circuits with a gate set.

    Circuit qubit ``i`` is ``gates.qubits[i]``. A native without a
    calibration keeps its template values, which for the built-in natives are
    closed-form seeds from the chip, and gets fitted virtual-Z phases
    (``gates.calibrate(max_nfev=0)``), with a warning. Calibrate the gate set
    first to use calibrated gates.

    Parameters
    ----------
    gates : GateSet
        Natives and decomposition rules of the processor.

    Returns
    -------
    ChipProcessor
        A ``qutip_qip.device.ModelProcessor``; see
        :class:`~quchip.interop.qutip_qip.processor.ChipProcessor`.

    Raises
    ------
    ImportError
        The optional ``quchip[qutip-qip]`` dependency is unavailable.
    """
    _require_qutip_qip()
    from .processor import ChipProcessor

    missing = [key for key in gates.natives if key not in gates._calibrations]
    if missing:
        names = ", ".join(f"{name} on {', '.join(qubits)}" for name, qubits in missing)
        warnings.warn(f"to_qutip_qip uses the closed-form seeds of {names} and fits only their virtual-Z phases. "
                      "Call gates.calibrate() first to use calibrated gates.", UserWarning, stacklevel=2)
        gates.calibrate(*missing, max_nfev=0)
    return ChipProcessor(gates)


def from_qutip_qip(circuit: Any, gates: "GateSet") -> list[tuple[Any, ...]]:
    """Convert a qutip-qip circuit into gate-set instructions.

    Circuit qubit ``i`` becomes ``gates.qubits[i]``. A gate whose name the
    gate set has as a native or a rule keeps its name and angles, and RZ stays
    virtual, as does PHASE with its global phase. qutip-qip first resolves
    every other gate into CX, RX, RY, and RZ. The circuit's global phase
    becomes a ``GLOBALPHASE`` instruction.

    Parameters
    ----------
    circuit : qutip_qip.circuit.QubitCircuit or iterable of qutip_qip.circuit.GateInstruction
        Circuit to convert. Measurements and classically controlled gates are
        not supported.
    gates : GateSet
        Gate set whose qubits, natives, and rules the instructions use.

    Returns
    -------
    list of tuple
        ``(name, qubits, params)`` instructions for
        :meth:`~quchip.gateset.GateSet.schedule`,
        :meth:`~quchip.gateset.GateSet.unitary`, and
        :meth:`~quchip.gateset.GateSet.simulate`.

    Raises
    ------
    ImportError
        The optional ``quchip[qutip-qip]`` dependency is unavailable.
    """
    _require_qutip_qip()
    from qutip_qip.circuit import QubitCircuit

    labels = gates.qubits
    known = {name for name, _ in gates.natives} | set(rule_table(gates._user_rules)) | set(VIRTUAL)
    out: list[tuple[Any, ...]] = []
    if isinstance(circuit, QubitCircuit):
        if circuit.num_qubits > len(labels):
            raise ValueError(f"The circuit has {circuit.num_qubits} qubits; the gate set has {len(labels)}.")
        if circuit.global_phase:
            out.append(("GLOBALPHASE", (), (float(circuit.global_phase),)))
        circuit = circuit.instructions
    for instruction in circuit:
        if not instruction.is_gate_instruction():
            raise ValueError("Measurements are not supported; read the final state instead.")
        gate = instruction.operation
        if instruction.cbits:
            raise ValueError(f"{gate.name} has classical controls, which are not supported.")
        name = canonical_name(gate.name)
        if name == "PHASE":
            # PHASE(φ) = diag(1, e^{iφ}) is a virtual RZ(φ) with the global phase φ/2.
            angle = float(gate.arg_value[0])
            out += [("RZ", (labels[instruction.qubits[0]],), (angle,)), ("GLOBALPHASE", (), (angle / 2,))]
            continue
        if name not in known:
            single = QubitCircuit(len(labels))
            single.add_gate(gate, targets=instruction.targets, controls=instruction.controls)
            try:
                resolved = single.resolve_gates(basis=BASIS)
            except NotImplementedError:
                raise ValueError(f"The gate set has no rule for {gate.name}, and qutip-qip cannot resolve it. "
                                 "Add one with gates.rule().") from None
            unknown = sorted({item.operation.name for item in resolved.instructions
                              if canonical_name(item.operation.name) not in known})
            if unknown:
                raise ValueError(f"qutip-qip resolves {gate.name} into {unknown}, which the gate set does not have.")
            out.extend(from_qutip_qip(resolved, gates))
            continue
        qubits = tuple(labels[i] for i in instruction.qubits)
        params = tuple(float(v) for v in gate.arg_value) if gate.is_parametric else ()
        out.append((name, qubits, params))
    return out
