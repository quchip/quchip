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
    """Return *operator* on *subsys* as a truncated energy-eigenbasis matrix.

    scqubits stores an ``InteractionTerm`` operator either as a bound method
    (evaluated in the eigenbasis on demand) or as a raw matrix in the
    subsystem's native basis:

    * A callable is invoked with ``energy_esys=True`` so scqubits returns the
      ``truncated_dim x truncated_dim`` eigenbasis matrix directly. Operators
      that are already in their eigenbasis (e.g. an ``Oscillator`` in the Fock
      basis) expose a no-argument method; those are called bare, mirroring
      scqubits' own ``identity_wrap`` fallback.
    * A raw native-basis matrix is projected with the subsystem's eigenvectors,
      ``V^\\dagger O V`` with ``V`` the lowest ``truncated_dim`` columns of
      ``subsys.eigensys`` — the same projection scqubits applies internally.
    """
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
    """Build the callable-form interaction ``g·A⊗B`` (plus h.c. when *add_hc*).

    ``g_strength`` is folded into the first factor so the returned closure needs
    no scalar prefactor and the host :class:`~quchip.chip.couplings.Coupling`
    keeps ``g = 1.0`` (``g_strength`` may be complex, which the coupling's real
    ``g`` could not carry). The closure builds ``M = A_g ⊗ B`` from the frozen
    matrices via the backend and returns ``M + M^\\dagger`` for the type-2
    (``add_hc``) interaction, matching scqubits' ``V = g A B + h.c.``.
    """
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
    """Transcribe one scqubits ``InteractionTerm`` into a quchip ``Coupling``.

    Rejects string-expression interactions
    (``InteractionTermStr``) and products of other than two operators. Both
    raise :class:`NotImplementedError` with a re-expression hint rather than
    importing a partial interaction.
    """
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
    """Return *value* as a concrete scalar, or raise on a JAX tracer.

    Export is eager: a coupling strength carrying a
    tracer (inside ``jit``/``grad``) cannot be written into a static scqubits
    object, so it fails here rather than silently dropping the swept value.
    """
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
    r"""Return ``(g, A, B)`` reproducing the complete ``H_int = g·A⊗B``.

    Each supported coupling factorizes into a scalar strength and two device
    operators; the factors are the coupling's own operator definitions,
    evaluated on the endpoint devices and densified through *backend*, so the
    exported interaction is term-for-term identical to the one quchip assembles
    (:meth:`~quchip.chip.coupling_base.BaseCoupling.interaction_hamiltonian`):

    * :class:`~quchip.chip.couplings.Capacitive` /
      :class:`~quchip.chip.couplings.TunableCapacitive` — the full product of
      each endpoint's physical charge-like factor. Devices implementing
      :class:`~quchip.devices.protocols.ChargeCoupled` supply their authored
      charge operator; other devices use ``a + a†``. The complete form is
      always exported because scqubits interaction terms apply no rotating-wave
      truncation of their own.
    * :class:`~quchip.chip.couplings.CrossKerr` — ``χ·n̂_a n̂_b``, so
      ``A = n̂_a`` and ``B = n̂_b``.
    * product-form :class:`~quchip.chip.couplings.Coupling` — the user's
      ``g·op_a(a)⊗op_b(b)``, so ``A``/``B`` are exactly those two factors.

    A callable-form :class:`~quchip.chip.couplings.Coupling` (whose interaction
    is an opaque two-device closure, not a factorizable product) and any other
    coupling type raise :class:`NotImplementedError` naming the supported set.
    """

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
    r"""Lift a truncated-eigenbasis operator into *subsys*' native basis.

    scqubits assembles an ``op1``/``op2`` interaction by projecting each raw
    matrix from the subsystem's *native* basis into its truncated eigenbasis
    (``V^\dagger O V`` with ``V`` the native eigenvectors). quchip supplies the
    operator already in the truncated eigenbasis, so the inverse lift
    ``V O V^\dagger`` is applied first: scqubits' projection then recovers the
    quchip matrix exactly (``V^\dagger V = I`` on the kept subspace). For a
    subsystem whose native dimension already equals its truncated dimension
    (an ``Oscillator``) the lift is the identity. This keeps the exported
    interaction a plain ndarray product term — re-importable through
    :func:`import_hilbertspace` unchanged — rather than an opaque full-space
    ``qobj`` scqubits' assembly would take verbatim but the importer could not
    factorize.
    """
    _, evecs = subsys.eigensys(evals_count=subsys.truncated_dim)
    v = np.asarray(evecs, dtype=complex)[:, : subsys.truncated_dim]
    return v @ matrix @ v.conj().T


def _warn_if_cross_basis(device: Any, subsys: Any) -> None:
    """Warn when *device* and its exported *subsys* diagonalize different-sized bases.

    A device whose authored basis dimension differs from the exported
    subsystem's native dimension — for example a fluxonium phase grid versus
    scqubits' oscillator cutoff — is reconstructed in a different numerical
    discretization. Its exported spectrum therefore agrees only to the
    cross-discretization accuracy. A charge-basis transmon exports one-to-one
    and does not trigger this warning.
    """
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
    """Reject a filtered interaction that scqubits would export in full.

    scqubits export always emits the complete operator product
    (:func:`_coupling_product_factors`). For :class:`~quchip.chip.couplings.Capacitive`,
    :class:`~quchip.chip.couplings.TunableCapacitive`, and product-form
    :class:`~quchip.chip.couplings.Coupling`, the filtered form is a genuinely
    different operator than the complete form, so exporting one of these under a
    `RWA` would silently reproduce different physics than the
    chip's own dressed dynamics. :class:`~quchip.chip.couplings.CrossKerr` is
    exempt: its interaction is diagonal in the excitation-number basis, so RWA
    masking is a no-op on it.
    """
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
        A coupling is neither :class:`~quchip.chip.couplings.Capacitive`,
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
    """
    import scqubits

    if opts:
        raise TypeError(
            f"export_chip got unexpected keyword argument(s): {', '.join(sorted(opts))}. "
            "Composite export takes no options."
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
