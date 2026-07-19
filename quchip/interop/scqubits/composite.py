"""Composite import — scqubits ``HilbertSpace`` -> quchip :class:`~quchip.chip.chip.Chip`.

An scqubits ``HilbertSpace`` bundles a list of subsystems and a list of
``InteractionTerm`` couplings between them. :func:`import_hilbertspace` imports
each subsystem individually through the shipped device mappings
(:mod:`quchip.interop.scqubits.devices`), preserving order and each subsystem's
``id_str`` as the device label, then transcribes every ``InteractionTerm`` into
a callable-form :class:`~quchip.chip.couplings.Coupling` whose operator matrices
are the term's subsystem operators expressed in the *gauge of the imported
device* they act on.

Gauge consistency (Principle 3). Each imported circuit device
(:class:`~quchip.devices.circuit.CircuitDevice`, e.g. ``ChargeBasisTransmon``)
re-diagonalizes its own native Hamiltonian, fixing an eigenvector gauge — an
arbitrary per-eigenvector phase/sign — that need not agree with scqubits' own
diagonalization of the same Hamiltonian. The undriven spectrum is
gauge-invariant, but a drive on such a device that also participates in a
coupling mixes the two gauges, corrupting the driven dynamics. So a coupling
factor is projected through the *device's* eigenvectors
(:meth:`~quchip.devices.circuit.CircuitDevice.project_operator`) whenever the
device re-diagonalizes the same native basis scqubits stores the operator in.

Limitation (declared, Principle 12). When the imported device's native basis
differs in dimension from scqubits' native basis for that subsystem — e.g. a
:class:`~quchip.devices.fluxonium.Fluxonium` (quchip phase grid vs scqubits'
harmonic-oscillator basis) — the factor cannot be re-projected into the device
gauge and is frozen in the *source's* eigenbasis gauge instead; a
:class:`UserWarning` names the device, and driven dynamics through it may carry
gauge-inconsistent matrix elements. Declarative devices without a native basis
of their own (``Resonator``, ``KerrCavity``, whose Fock eigenbasis matches
scqubits ``Oscillator``'s by construction) and
:class:`~quchip.interop.eigenbasis.EigenbasisDevice` (whose native basis *is*
scqubits' eigenbasis) are already in the device gauge and take the source
eigenbasis path without warning.

Scope of v1 (declared, Principle 12): only pairwise ``InteractionTerm``
products of two operators are translated. Each term's operator matrices are a
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
from quchip.interop.base import export_object, import_object
from quchip.interop.eigenbasis import EigenbasisDevice
from quchip.utils.jax_utils import maybe_concrete_scalar

_SUPPORTED_EXPORT_COUPLINGS = "Capacitive, TunableCapacitive, CrossKerr, or product-form Coupling"


def _native_matrix(operator: Any) -> np.ndarray:
    """Return *operator* as a dense matrix in scqubits' native (undiagonalized) basis.

    A bound method is called bare (no ``energy_esys=``) so scqubits returns the
    operator in the subsystem's native basis rather than its eigenbasis; raw
    ndarray / sparse entries are already native and are densified.
    """
    if callable(operator):
        matrix = operator()
        return np.asarray(matrix.todense() if hasattr(matrix, "todense") else matrix, dtype=complex)
    return np.asarray(operator.todense() if hasattr(operator, "todense") else operator, dtype=complex)


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
    r"""Return *operator* on *subsys* in the eigenvector gauge of *device*.

    When *device* re-diagonalizes its native Hamiltonian
    (:class:`~quchip.devices.circuit.CircuitDevice` subclasses, which expose
    :meth:`~quchip.devices.circuit.CircuitDevice.project_operator`) and its
    native dimension matches scqubits' native dimension for *subsys*, the
    native-basis operator is projected through the *device's* own eigenvectors,
    ``V_dev^\dagger O V_dev`` — so the coupling factor and the device's drive
    operators live in one consistent gauge.

    Otherwise the source's eigenbasis gauge is used (see
    :func:`_eigenbasis_matrix`). Two of those cases are already in the device
    gauge and pass silently: a device with no ``project_operator`` (declarative
    ``Resonator`` / ``KerrCavity``, matching scqubits ``Oscillator``'s Fock
    basis) and :class:`~quchip.interop.eigenbasis.EigenbasisDevice` (whose
    native basis *is* scqubits' eigenbasis, so its native dimension equals the
    truncated eigenbasis dimension). The remaining case — a device that
    diagonalizes a *different* native basis (e.g. fluxonium) — cannot be
    re-projected and warns, since driven dynamics through it may carry
    gauge-inconsistent matrix elements.

    The native-basis operator (:func:`_native_matrix`) is only ever
    materialized when there is a real chance of using it: an
    :class:`~quchip.interop.eigenbasis.EigenbasisDevice` never re-diagonalizes
    a native basis of its own (its "native basis" *is* the truncated
    eigenbasis scqubits already handed over), so it short-circuits to the
    eigenbasis path immediately; for every other project-capable device, the
    native dimension is compared against ``subsys.hilbertdim()`` — a cheap
    lookup, not a matrix build — before ever calling :func:`_native_matrix`.
    A ZeroPi subsystem's native basis is a dense joint phi-grid / charge-basis
    product space (thousands of dimensions); densifying its operator just to
    discover the dimensions do not match wastes gigabytes for nothing.
    """
    project = getattr(device, "project_operator", None)
    if project is None or isinstance(device, EigenbasisDevice):
        return _eigenbasis_matrix(subsys, operator)

    native_dim = int(np.asarray(device.eigenvectors()).shape[0])
    if native_dim != subsys.hilbertdim():
        eigenbasis = _eigenbasis_matrix(subsys, operator)
        if native_dim != eigenbasis.shape[0]:
            warnings.warn(
                f"Imported device {device.label!r} diagonalizes a native basis of a "
                f"different dimension than scqubits' for this subsystem, so its coupling "
                f"matrix is frozen in the source's eigenbasis gauge rather than the device "
                f"gauge; driven dynamics through this device may carry gauge-inconsistent "
                f"matrix elements.",
                UserWarning,
                stacklevel=2,
            )
        return eigenbasis

    return np.asarray(project(_native_matrix(operator)), dtype=complex)


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

    Rejects the two shapes v1 does not model: string-expression interactions
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

    # An scqubits InteractionTerm is a full bilinear operator with no
    # rotating-wave truncation of its own, so the imported coupling is
    # marked rwa=False — the imported chip reproduces the scqubits
    # composite spectrum exactly, independent of the chip's RWA default.
    return Coupling(
        devices[index_a],
        devices[index_b],
        g=1.0,
        interaction=_product_interaction(term.g_strength, matrix_a, matrix_b, bool(term.add_hc)),
        rwa=False,
        label=f"scq_interaction_{index}",
    )


def import_hilbertspace(hs: Any, **opts: Any) -> Chip:
    """Import an scqubits ``HilbertSpace`` into a quchip :class:`Chip`.

    Each subsystem imports through the shipped device mappings (order and
    ``id_str`` label preserved); each ``InteractionTerm`` becomes a
    callable-form :class:`~quchip.chip.couplings.Coupling`.

    Parameters
    ----------
    hs : scqubits.HilbertSpace
        The composite system to import.
    **opts
        ``frame`` and ``rwa`` are forwarded to the :class:`Chip` constructor.
        Device-level options are not forwarded in v1: every subsystem imports
        at its own ``truncated_dim`` and native noise defaults.

    Raises
    ------
    NotImplementedError
        A string-expression (``InteractionTermStr``) or non-pairwise
        interaction term is present.
    LookupError
        A subsystem has no registered device mapping.
    """
    subsystems = list(hs.subsystem_list)
    devices = [import_object(subsys) for subsys in subsystems]
    couplings: list[BaseCoupling] = [
        _coupling_from_term(term, subsystems, devices, index)
        for index, term in enumerate(hs.interaction_list)
    ]

    chip_kwargs: dict[str, Any] = {}
    for key in ("frame", "rwa"):
        if key in opts:
            chip_kwargs[key] = opts[key]

    return Chip(devices=devices, couplings=couplings, **chip_kwargs)


def _concrete_strength(value: Any, coupling: Any) -> Any:
    """Return *value* as a concrete scalar, or raise on a JAX tracer.

    Export is eager and terminal (Principle 2): a coupling strength carrying a
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


def _coupling_product_factors(coupling: Any, backend: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    r"""Return ``(g, A, B)`` reproducing the coupling's non-RWA ``H_int = g·A⊗B``.

    Each supported coupling factorizes into a scalar strength and two device
    operators; the factors are the coupling's own operator definitions,
    evaluated on the endpoint devices and densified through *backend*, so the
    exported interaction is term-for-term identical to the one quchip assembles
    (:meth:`~quchip.chip.couplings.CouplingModel.interaction_hamiltonian` at
    ``rwa=False``):

    * :class:`~quchip.chip.couplings.Capacitive` /
      :class:`~quchip.chip.couplings.TunableCapacitive` — the full dipole form
      ``g·(a + a†)(b + b†)``, so ``A = a + a†`` and ``B = b + b†``. The full
      (non-RWA) form is always exported: scqubits interaction terms are the
      bare operator product, with no rotating-wave truncation of their own.
    * :class:`~quchip.chip.couplings.CrossKerr` — ``χ·n̂_a n̂_b``, so
      ``A = n̂_a`` and ``B = n̂_b``.
    * product-form :class:`~quchip.chip.couplings.Coupling` — the user's
      ``g·op_a(a)⊗op_b(b)``, so ``A``/``B`` are exactly those two factors.

    A callable-form :class:`~quchip.chip.couplings.Coupling` (whose interaction
    is an opaque two-device closure, not a factorizable product) and any other
    coupling type raise :class:`NotImplementedError` naming the supported set.
    """

    def matrix(op: Any) -> np.ndarray:
        return np.asarray(backend.to_array(op), dtype=complex)

    device_a, device_b = coupling.device_a, coupling.device_b
    # coupling_strength is the one scalar-strength property every coupling
    # type defines (BaseCoupling.coupling_strength): g for Capacitive/
    # Coupling, g_0 for TunableCapacitive, chi for CrossKerr. Reading it
    # uniformly here means a new coupling type with its own scalar-strength
    # field needs no change to this dispatch — only the operator structure
    # below is type-specific.
    g = _concrete_strength(coupling.coupling_strength, coupling)

    if isinstance(coupling, (TunableCapacitive, Capacitive)):
        return g, _quadrature(device_a, matrix), _quadrature(device_b, matrix)
    if isinstance(coupling, CrossKerr):
        return g, matrix(device_a.number_operator()), matrix(device_b.number_operator())
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
        return g, matrix(coupling._op_a(device_a)), matrix(coupling._op_b(device_b))

    raise NotImplementedError(
        f"{type(coupling).__name__} {coupling.label!r} is not exportable to scqubits; "
        f"supported couplings are {_SUPPORTED_EXPORT_COUPLINGS}."
    )


def _quadrature(device: Any, matrix: Any) -> np.ndarray:
    """Return the unnormalized quadrature ``a + a†`` on *device* as a matrix."""
    return matrix(device.lowering_operator()) + matrix(device.raising_operator())


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

    Mirrors the import-side gauge caveat (:func:`_device_gauge_matrix`). A device
    that re-diagonalizes its own native Hamiltonian (exposes ``eigenvectors()``)
    at a native dimension that differs from the exported subsystem's
    ``hilbertdim()`` — e.g. a :class:`~quchip.devices.fluxonium.Fluxonium`, whose
    phase grid does not match scqubits' harmonic-oscillator ``cutoff`` — cannot
    hand scqubits its own native basis; scqubits rebuilds the spectrum in its
    basis instead, so the two composites agree only to the cross-discretization
    level (the same ``atol=1e-6`` the per-device import already carries). A
    :class:`~quchip.devices.transmon.ChargeBasisTransmon` (whose charge basis
    exports one-to-one) does not trigger it.
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


def _check_rwa_exportable(chip: Chip, coupling: Any) -> None:
    """Raise when *coupling* factorizes RWA-sensitively and the chip resolves ``rwa=True`` for it.

    scqubits export always emits the full (non-RWA) operator product
    (:func:`_coupling_product_factors`). For :class:`~quchip.chip.couplings.Capacitive`,
    :class:`~quchip.chip.couplings.TunableCapacitive`, and product-form
    :class:`~quchip.chip.couplings.Coupling`, the RWA form is a genuinely
    different operator than the full form, so exporting one of these under a
    resolved ``rwa=True`` would silently reproduce different physics than the
    chip's own dressed dynamics. :class:`~quchip.chip.couplings.CrossKerr` is
    exempt: its interaction is diagonal in the excitation-number basis, so RWA
    masking is a no-op on it.
    """
    if isinstance(coupling, CrossKerr):
        return
    rwa_sensitive = isinstance(coupling, (Capacitive, TunableCapacitive)) or (
        isinstance(coupling, Coupling) and coupling._interaction is None
    )
    if rwa_sensitive and chip.resolve_rwa(coupling):
        raise ValueError(
            f"Coupling {coupling.label!r} resolves rwa=True on this chip, but scqubits export always "
            "emits the full (non-RWA) operator product — exporting it would silently reproduce "
            "different physics than the chip's RWA-resolved dynamics. Export an explicitly non-RWA "
            "chip (Chip(..., rwa=False)), or set rwa=False on this coupling, to proceed."
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

    Couplings are exported in their **full (non-RWA) operator form**: scqubits
    interaction terms are bare operator products and apply no rotating-wave
    truncation of their own. Exporting a chip whose *resolved* RWA actually
    masks a :class:`~quchip.chip.couplings.Capacitive`,
    :class:`~quchip.chip.couplings.TunableCapacitive`, or product-form
    :class:`~quchip.chip.couplings.Coupling` therefore fails closed with
    :class:`ValueError`: silently exporting the full form anyway would
    reproduce different physics than the chip's own RWA-resolved dynamics.
    :class:`~quchip.chip.couplings.CrossKerr` is exempt — its RWA and full
    forms coincide, since ``n̂_a n̂_b`` conserves excitation number and is
    never touched by the RWA mask. Export an explicitly non-RWA chip
    (``Chip(..., rwa=False)``, or ``rwa=False`` on the coupling) to proceed.

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
        a Capacitive/TunableCapacitive/product-form Coupling resolves
        ``rwa=True`` on the chip (see above).
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
    for coupling in chip.couplings:
        _check_rwa_exportable(chip, coupling)
        g, matrix_a, matrix_b = _coupling_product_factors(coupling, backend)
        subsys_a = label_to_subsys[coupling.device_a_label]
        subsys_b = label_to_subsys[coupling.device_b_label]
        hs.add_interaction(
            g=g,
            op1=(_lift_to_native(subsys_a, matrix_a), subsys_a),
            op2=(_lift_to_native(subsys_b, matrix_b), subsys_b),
            add_hc=False,
        )
    return hs
