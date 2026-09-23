"""Composite import — scqubits ``HilbertSpace`` -> quchip :class:`~quchip.chip.chip.Chip`.

An scqubits ``HilbertSpace`` bundles a list of subsystems and a list of
``InteractionTerm`` couplings between them. :func:`import_hilbertspace` imports
each subsystem individually through the shipped device mappings
(:mod:`quchip.interop.scqubits.devices`), preserving order and each subsystem's
``id_str`` as the device label, then transcribes every ``InteractionTerm`` into
a callable-form :class:`~quchip.chip.couplings.Coupling` whose operator matrices
are the term's subsystem operators expressed in the *gauge of the imported
device* they act on.

An imported ``HilbertSpace`` is a frozen snapshot of the source's truncated
subsystem model. Each subsystem becomes an
:class:`~quchip.interop.eigenbasis.EigenbasisDevice`, and interaction factors
remain in that same source eigenbasis. This reproduces scqubits' truncation and
gauge exactly without introducing a second projection path. Importing an
individual supported device still reconstructs the live differentiable quchip
model from its circuit parameters.

Only pairwise ``InteractionTerm`` products of two operators are translated.
Each term's operator matrices are a
frozen snapshot at the source parameter point, so the coupling is not
differentiable with respect to the source circuit parameters (the same
frozen-snapshot contract :class:`~quchip.interop.eigenbasis.EigenbasisDevice`
carries). ``InteractionTermStr`` string expressions and non-pairwise products
raise :class:`NotImplementedError` rather than importing a partial model.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

from quchip.chip.chip import Chip
from quchip.chip.coupling_base import BaseCoupling
from quchip.chip.couplings import Capacitive, Coupling, CrossKerr, TunableCapacitive
from quchip.devices.base import BaseDevice
from quchip.devices.protocols import ChargeCoupled
from quchip.interop.base import export_object
from quchip.interop.eigenbasis import EigenbasisDevice
from quchip.utils.jax_utils import maybe_concrete_scalar

_SUPPORTED_EXPORT_COUPLINGS = "Capacitive, TunableCapacitive, CrossKerr, or product-form Coupling"


def _eigenbasis_matrix(subsys: Any, operator: Any) -> np.ndarray:
    """Return a truncated eigenbasis operator: call with energy_esys=True
    (or no arguments for Fock methods), or project a raw native matrix as V† O V."""
    if callable(operator):
        try:
            matrix = operator(energy_esys=True)
        except TypeError:
            matrix = operator()
        return np.asarray(matrix, dtype=complex)

    native = np.asarray(operator.todense() if hasattr(operator, "todense") else operator, dtype=complex)
    _, evecs = subsys.eigensys(evals_count=subsys.truncated_dim)
    v = np.asarray(evecs, dtype=complex)[:, : subsys.truncated_dim]
    return v.conj().T @ native @ v


def _device_gauge_matrix(subsys: Any, operator: Any, device: Any) -> np.ndarray:
    """Return a source operator in the frozen subsystem eigenbasis."""
    matrix = _eigenbasis_matrix(subsys, operator)
    dimension = device.local_space().dimension
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"Interaction factor for {device.label!r} has shape {matrix.shape}, "
            f"expected {(dimension, dimension)}."
        )
    return matrix


def _projected_source_operator(subsys: Any, names: tuple[str, ...], esys: Any) -> Any | None:
    """Return the first available source operator projected with ``esys``."""
    for name in names:
        operator = getattr(subsys, name, None)
        if operator is None:
            continue
        try:
            return np.asarray(operator(energy_esys=esys), dtype=complex)
        except TypeError:
            try:
                return np.asarray(operator(), dtype=complex)
            except ValueError:
                continue
        except ValueError:
            continue
    return None


def _snapshot_subsystem(subsys: Any) -> EigenbasisDevice:
    """Freeze one scqubits subsystem exactly at its HilbertSpace truncation."""
    levels = int(subsys.truncated_dim)
    esys = subsys.eigensys(evals_count=levels)
    return EigenbasisDevice(
        esys[0],
        charge_operator=_projected_source_operator(
            subsys,
            ("n_operator", "n_theta_operator"),
            esys,
        ),
        phase_operator=_projected_source_operator(subsys, ("phi_operator",), esys),
        levels=levels,
        label=getattr(subsys, "id_str", None),
        source_type=f"scqubits.{type(subsys).__name__}",
    )


def _product_interaction(
    g_strength: complex,
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    add_hc: bool,
) -> Callable[[Any, Any, Any], Any]:
    """Fold possibly complex g into the first frozen factor; the host Coupling
    keeps g=1. Add the Hermitian conjugate when requested."""
    matrix_a_g = g_strength * matrix_a
    matrix_b = np.asarray(matrix_b, dtype=complex)

    def interaction(_a: Any, _b: Any, bk: Any) -> Any:
        m = bk.tensor(bk.from_array(matrix_a_g), bk.from_array(matrix_b))
        if add_hc:
            return m + bk.dag(m)
        return m

    return interaction


def _coupling_from_term(
    term: Any,
    subsystems: list[Any],
    devices: list[Any],
    index: int,
) -> Coupling:
    """Translate one pairwise operator-product term; reject string expressions
    and non-pairwise forms rather than importing a partial interaction."""
    from scqubits.core.hilbert_space import InteractionTermStr

    if isinstance(term, InteractionTermStr):
        raise NotImplementedError(
            "string-expression interactions are not translated; re-express as operator products"
        )

    operator_list = list(term.operator_list)
    if len(operator_list) != 2:
        raise NotImplementedError(
            f"only pairwise interaction terms are translated; term {index} couples "
            f"{len(operator_list)} operators. Re-express as two-operator products."
        )

    (index_a, op_a), (index_b, op_b) = operator_list
    matrix_a = _device_gauge_matrix(subsystems[index_a], op_a, devices[index_a])
    matrix_b = _device_gauge_matrix(subsystems[index_b], op_b, devices[index_b])

    # An scqubits InteractionTerm is a complete bilinear operator. The
    # imported chip therefore defaults to Exact so its spectrum matches.
    return Coupling(
        devices[index_a],
        devices[index_b],
        g=1.0,
        interaction=_product_interaction(term.g_strength, matrix_a, matrix_b, bool(term.add_hc)),
        label=f"scq_interaction_{index}",
    )


def import_hilbertspace(hs: Any, **opts: Any) -> Chip:
    """Import an scqubits ``HilbertSpace`` into a quchip :class:`Chip`.

    Each subsystem is frozen at its source truncation (order and ``id_str``
    preserved); each ``InteractionTerm`` becomes a callable-form
    :class:`~quchip.chip.couplings.Coupling` in the same eigenbasis gauge.

    Parameters
    ----------
    hs : scqubits.HilbertSpace
        The composite system to import.
    **opts
        ``frame`` and ``approximation`` are forwarded to :class:`Chip`.
        Device-level options are not forwarded: every subsystem imports
        at its own ``truncated_dim`` and native noise defaults.

    Raises
    ------
    NotImplementedError
        A string-expression (``InteractionTermStr``) or non-pairwise
        interaction term is present.
    """
    subsystems = list(hs.subsystem_list)
    devices: list[BaseDevice] = [_snapshot_subsystem(subsys) for subsys in subsystems]
    couplings: list[BaseCoupling] = [
        _coupling_from_term(term, subsystems, devices, index)
        for index, term in enumerate(hs.interaction_list)
    ]

    from quchip.approximations import Exact

    chip_kwargs: dict[str, Any] = {"approximation": Exact()}
    for key in ("frame", "approximation"):
        if key in opts:
            chip_kwargs[key] = opts[key]

    return Chip(devices=devices, couplings=couplings, **chip_kwargs)


def _concrete_strength(value: Any, coupling: Any) -> Any:
    """Require an eager scalar; traced strengths cannot populate a static scqubits object."""
    scalar = maybe_concrete_scalar(value)
    if scalar is None:
        raise ValueError(
            f"export_chip requires a concrete coupling strength for {coupling.label!r}; "
            "call outside jit/grad or substitute concrete values first."
        )
    return scalar


def _coupling_product_factors(
    coupling: Any,
    backend: Any,
    bases: Any,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Return the complete interaction as (g, A, B), densified through backend.

    Capacitive factors use authored charge operators or a+a†; cross-Kerr uses
    number operators; product Coupling uses its authored factors. Reject opaque
    callable forms. scqubits applies no rotating-wave filtering to these products."""

    def matrix(device: Any, op: Any) -> np.ndarray:
        from quchip.declarative.expr import materialize_expr

        authored = np.asarray(
            backend.to_array(materialize_expr(op, backend)),
            dtype=complex,
        )
        return np.asarray(bases[device.label].transform_operator(authored), dtype=complex)

    device_a, device_b = coupling.device_a, coupling.device_b
    # coupling_strength is the one scalar-strength property every coupling
    # type defines (BaseCoupling.coupling_strength): g for Capacitive/
    # Coupling, g_0 for TunableCapacitive, chi for CrossKerr. Reading it
    # uniformly here means a new coupling type with its own scalar-strength
    # field needs no change to this dispatch — only the operator structure
    # below is type-specific.
    g = _concrete_strength(coupling.coupling_strength, coupling)

    if isinstance(coupling, (TunableCapacitive, Capacitive)):
        return g, _charge_factor(device_a, matrix), _charge_factor(device_b, matrix)
    if isinstance(coupling, CrossKerr):
        return (
            g,
            matrix(device_a, device_a.energy_level_operator()),
            matrix(device_b, device_b.energy_level_operator()),
        )
    if isinstance(coupling, Coupling):
        if coupling._interaction is not None:
            raise NotImplementedError(
                f"callable-form Coupling {coupling.label!r} carries an opaque two-device closure that "
                f"does not factorize into a single operator product; scqubits export supports "
                f"{_SUPPORTED_EXPORT_COUPLINGS}. Re-express it in product form (op_a, op_b)."
            )
        # op_a/op_b are both non-None in product form (guaranteed by Coupling.__init__,
        # given _interaction is None here).
        assert coupling._op_a is not None and coupling._op_b is not None
        return (
            g,
            matrix(device_a, coupling._op_a(device_a)),
            matrix(device_b, coupling._op_b(device_b)),
        )

    raise NotImplementedError(
        f"{type(coupling).__name__} {coupling.label!r} is not exportable to scqubits; "
        f"supported couplings are {_SUPPORTED_EXPORT_COUPLINGS}."
    )


def _charge_factor(device: Any, matrix: Any) -> np.ndarray:
    """Return one endpoint's physical charge-like operator in solver space."""
    if isinstance(device, ChargeCoupled):
        return matrix(device, device.charge_coupling_operator())
    return matrix(device, device.lowering_operator() + device.raising_operator())


def _lift_to_native(subsys: Any, matrix: np.ndarray) -> np.ndarray:
    """Lift with V O V† so scqubits projects back to the retained matrix.

    Keeping ndarray product factors, rather than opaque full-space Qobjs, permits re-import."""
    _, evecs = subsys.eigensys(evals_count=subsys.truncated_dim)
    v = np.asarray(evecs, dtype=complex)[:, : subsys.truncated_dim]
    return v @ matrix @ v.conj().T


def _warn_if_cross_basis(device: Any, subsys: Any) -> None:
    """Warn when source and exported native dimensions differ: spectra then
    agree only to cross-discretization accuracy. Charge-basis transmons map directly."""
    eigenvectors = getattr(device, "eigenvectors", None)
    if eigenvectors is None:
        return
    native_dim = int(np.asarray(eigenvectors()).shape[0])
    if native_dim != subsys.hilbertdim():
        warnings.warn(
            f"Exported device {device.label!r} diagonalizes a native basis of a different "
            f"dimension than its scqubits subsystem, which rebuilds the spectrum in a different "
            f"native basis; the two composites agree only to the cross-discretization level.",
            UserWarning,
            stacklevel=3,
        )


def _check_approximation_exportable(chip: Chip, coupling: Any) -> None:
    """Reject filtered non-diagonal products: scqubits exports them in full.
    Cross-Kerr is exempt because rotating-wave filtering leaves it unchanged."""
    if isinstance(coupling, CrossKerr):
        return
    approximation_sensitive = isinstance(coupling, (Capacitive, TunableCapacitive)) or (
        isinstance(coupling, Coupling) and coupling._interaction is None
    )
    if approximation_sensitive and chip.approximation.filters_terms:
        raise ValueError(
            f"Coupling {coupling.label!r} is filtered by {type(chip.approximation).__name__}, "
            "but scqubits export emits the complete operator product. Resolve or clone the chip "
            "with Exact() before export."
        )


def export_chip(chip: Chip, **opts: Any) -> Any:
    """Export a quchip :class:`Chip` to an scqubits ``HilbertSpace``.

    Each device exports through the shipped device mappings
    (:mod:`quchip.interop.scqubits.devices`) in chip order, and every
    :class:`~quchip.chip.couplings.Coupling` factorizes into a scalar strength
    and two device operators (see :func:`_coupling_product_factors`) added as
    one ``InteractionTerm`` per edge. scqubits carries the *bare diagonal*
    energies of each subsystem (gauge-invariant) plus these interaction
    matrices, so the whole composite lives in one consistent gauge — quchip's —
    and its dressed spectrum reproduces the chip's.

    Couplings are exported in their complete operator form: scqubits
    interaction terms are bare operator products and apply no approximation
    strategy of their own. Exporting a chip whose ``RWA()`` strategy
    filters a :class:`~quchip.chip.couplings.Capacitive`,
    :class:`~quchip.chip.couplings.TunableCapacitive`, or product-form
    :class:`~quchip.chip.couplings.Coupling` therefore fails closed with
    :class:`ValueError`: silently exporting the full form anyway would
    reproduce different physics than the chip's own resolved dynamics.
    :class:`~quchip.chip.couplings.CrossKerr` is exempt because
    ``n̂_a n̂_b`` conserves excitation number and survives ``RWA()``
    unchanged. Resolve or clone the chip with
    :class:`~quchip.approximations.Exact` before export.

    Chip-level control equipment and baths have no scqubits counterpart (it
    models neither drives nor dissipation) and are dropped with a single
    :class:`UserWarning` naming what was dropped.

    Parameters
    ----------
    chip : Chip
        The composite system to export. Coupling strengths must be concrete —
        a strength carrying a JAX tracer raises :class:`ValueError`.

    Raises
    ------
    NotImplementedError
        The chip has a PortNetwork, retained effective terms, intrinsic
        time-dependent Hamiltonians, or a coupling is neither
        :class:`~quchip.chip.couplings.Capacitive`,
        :class:`~quchip.chip.couplings.TunableCapacitive`,
        :class:`~quchip.chip.couplings.CrossKerr`, nor a product-form
        :class:`~quchip.chip.couplings.Coupling`.
    ValueError
        A coupling strength is a JAX tracer rather than a concrete value, or
        a Capacitive/TunableCapacitive/product-form Coupling is filtered by
        :class:`~quchip.approximations.RWA` (see above).
    LookupError
        A device has no registered scqubits export mapping.
    TypeError
        An unexpected keyword option is passed (composite export takes none).

    Other Parameters
    ----------------
    **opts
        No keyword options are accepted for composite export. Any supplied
        key raises ``TypeError``.
    """
    import scqubits

    if opts:
        raise TypeError(
            f"export_chip got unexpected keyword argument(s): {', '.join(sorted(opts))}. "
            "Composite export takes no options."
        )

    if chip.effective_terms:
        raise NotImplementedError("scqubits export does not represent retained effective terms.")
    if chip.dynamic_contributions():
        raise NotImplementedError("scqubits export does not represent intrinsic time-dependent Hamiltonians.")
    if chip.port_network is not None:
        raise NotImplementedError(
            "scqubits export does not represent PortNetwork interactions. "
            "The attached network can contribute to both the Hamiltonian and dissipation."
        )

    dropped: list[str] = []
    if chip.control_equipment is not None:
        dropped.append("control equipment (drive lines and signal chain)")
    if chip.baths:
        dropped.append("chip-level baths")
    if dropped:
        warnings.warn(
            f"scqubits models neither drives nor dissipation; dropping {' and '.join(dropped)} "
            f"from the exported HilbertSpace.",
            UserWarning,
            stacklevel=2,
        )

    subsystems = []
    for device in chip.devices:
        subsys = export_object(device, "scqubits")
        _warn_if_cross_basis(device, subsys)
        subsystems.append(subsys)
    label_to_subsys = {device.label: subsys for device, subsys in zip(chip.devices, subsystems)}

    hs = scqubits.HilbertSpace(subsystems)  # type: ignore[abstract]  # scqubits stub marks HilbertSpace abstract
    backend = chip.backend
    bases = chip.resolve().bases
    for coupling in chip.couplings:
        _check_approximation_exportable(chip, coupling)
        g, matrix_a, matrix_b = _coupling_product_factors(coupling, backend, bases)
        subsys_a = label_to_subsys[coupling.device_a_label]
        subsys_b = label_to_subsys[coupling.device_b_label]
        hs.add_interaction(
            g=g,
            op1=(_lift_to_native(subsys_a, matrix_a), subsys_a),
            op2=(_lift_to_native(subsys_b, matrix_b), subsys_b),
            add_hc=False,
        )
    return hs
