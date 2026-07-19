"""Shipped scqubits <-> quchip device mappings.

Each :class:`~quchip.interop.base.ModelMapping` here transcribes one scqubits
circuit-QED object into the quchip device that carries the same spectrum, and
(where the inversion is well-defined) back again. Import reads the source
object's native parameters and hands them to the matching quchip constructor,
which rebuilds the Hamiltonian from those parameters — so the imported device
stays differentiable in them. Export reads the concrete device parameters,
guards each against JAX tracers via :func:`maybe_concrete_scalar`, and
reconstructs the scqubits object.

Every mapping's docstring states its parameter translation; these are the
reference examples for authoring further mappings.
"""

from __future__ import annotations

from typing import Any, cast

from quchip.devices import DuffingTransmon, KerrCavity, Resonator
from quchip.devices.fluxonium import Fluxonium
from quchip.devices.transmon import ChargeBasisTransmon
from quchip.interop.base import ModelMapping
from quchip.interop.eigenbasis import EigenbasisDevice
from quchip.utils.jax_utils import maybe_concrete_scalar

_EXPORT_GUARD_MESSAGE = (
    "to_scqubits requires concrete parameters; call outside jit/grad or "
    "substitute concrete values first."
)


def _concrete_params(device: Any, names: tuple[str, ...]) -> dict[str, float]:
    """Read named device attributes as concrete floats, or raise on a tracer.

    Returns a name -> value dict. Raises :class:`ValueError` when any value is
    a JAX tracer (``maybe_concrete_scalar`` returns ``None``), so export never
    silently drops a swept or differentiated parameter.
    """
    vals = {name: maybe_concrete_scalar(getattr(device, name)) for name in names}
    if any(v is None for v in vals.values()):
        raise ValueError(_EXPORT_GUARD_MESSAGE)
    return cast(dict[str, float], vals)


class TransmonMapping(ModelMapping):
    """Map ``scqubits.Transmon`` to and from :class:`ChargeBasisTransmon`.

    Both sides diagonalize the Cooper-pair-box Hamiltonian
    :math:`H = 4 E_C (\\hat n - n_g)^2 - E_J \\cos\\hat\\varphi` in the integer
    charge basis, so the translation is a direct parameter copy:

    ==================  ======================
    scqubits            quchip
    ==================  ======================
    ``EC``              ``E_C``
    ``EJ``              ``E_J``
    ``ng``              ``n_g``
    ``ncut``            ``num_basis = 2*ncut + 1``
    ``truncated_dim``   ``levels``
    ==================  ======================
    """

    source = "scqubits.Transmon"
    target = ChargeBasisTransmon

    def import_model(self, obj: Any, *, levels: int | None = None, label: str | None = None,
                     **noise_kwargs: Any) -> ChargeBasisTransmon:
        coupling_channel = noise_kwargs.pop("coupling_channel", None)
        return ChargeBasisTransmon(
            E_C=obj.EC,
            E_J=obj.EJ,
            n_g=obj.ng,
            levels=levels or obj.truncated_dim,
            num_basis=2 * obj.ncut + 1,
            label=label or getattr(obj, "id_str", None),
            coupling_channel=coupling_channel,
            **noise_kwargs,
        )

    def export_model(self, device: Any, **opts: Any) -> Any:
        import scqubits

        vals = _concrete_params(device, ("E_C", "E_J", "n_g"))
        return scqubits.Transmon(
            EJ=vals["E_J"],
            EC=vals["E_C"],
            ng=vals["n_g"],
            ncut=(device.num_basis - 1) // 2,
            truncated_dim=device.levels,
            id_str=device.label,
        )


class TunableTransmonMapping(ModelMapping):
    """Map ``scqubits.TunableTransmon`` to :class:`ChargeBasisTransmon` (import-only).

    The flux-tunable SQUID transmon has a flux-dependent effective Josephson
    energy

    .. math::

       E_J(\\Phi) = E_J^{\\max}
           \\sqrt{\\cos^2(\\pi\\Phi) + d^2 \\sin^2(\\pi\\Phi)},

    with ``d`` the junction asymmetry. Import evaluates :math:`E_J(\\Phi)` at
    the object's ``flux`` and hands the resulting fixed-frequency transmon to
    :class:`ChargeBasisTransmon`; the remaining parameters copy across exactly
    as in :class:`TransmonMapping`. There is no export: a single frequency does
    not determine ``(EJmax, d, flux)``.
    """

    source = "scqubits.TunableTransmon"
    target = None

    def import_model(self, obj: Any, *, levels: int | None = None, label: str | None = None,
                     **noise_kwargs: Any) -> ChargeBasisTransmon:
        import numpy as np

        effective_e_j = obj.EJmax * np.sqrt(
            np.cos(np.pi * obj.flux) ** 2 + obj.d**2 * np.sin(np.pi * obj.flux) ** 2
        )
        coupling_channel = noise_kwargs.pop("coupling_channel", None)
        return ChargeBasisTransmon(
            E_C=obj.EC,
            E_J=effective_e_j,
            n_g=obj.ng,
            levels=levels or obj.truncated_dim,
            num_basis=2 * obj.ncut + 1,
            label=label or getattr(obj, "id_str", None),
            coupling_channel=coupling_channel,
            **noise_kwargs,
        )


class FluxoniumMapping(ModelMapping):
    """Map ``scqubits.Fluxonium`` to and from :class:`~quchip.devices.fluxonium.Fluxonium`.

    Parameter copy across the three circuit energies plus the external flux:

    ==================  ======================
    scqubits            quchip
    ==================  ======================
    ``EC``              ``E_C``
    ``EJ``              ``E_J``
    ``EL``              ``E_L``
    ``flux``            ``phi_ext``
    ``truncated_dim``   ``levels``
    ==================  ======================

    The native discretizations differ — scqubits uses a harmonic-oscillator
    basis of size ``cutoff``, quchip a plane-wave phase grid of ``num_basis``
    points — so quchip keeps its own default grid rather than mirroring
    ``cutoff``. The physics is identical; only the basis is not, which is why
    the spectrum agreement is at ``atol=1e-6`` rather than machine precision.
    Export uses scqubits' ``cutoff=110`` default.
    """

    source = "scqubits.Fluxonium"
    target = Fluxonium

    def import_model(self, obj: Any, *, levels: int | None = None, label: str | None = None,
                     **noise_kwargs: Any) -> Fluxonium:
        return Fluxonium(
            E_C=obj.EC,
            E_J=obj.EJ,
            E_L=obj.EL,
            phi_ext=obj.flux,
            levels=levels or obj.truncated_dim,
            label=label or getattr(obj, "id_str", None),
            **noise_kwargs,
        )

    def export_model(self, device: Any, **opts: Any) -> Any:
        import scqubits

        vals = _concrete_params(device, ("E_C", "E_J", "E_L", "phi_ext"))
        return scqubits.Fluxonium(
            EJ=vals["E_J"],
            EC=vals["E_C"],
            EL=vals["E_L"],
            flux=vals["phi_ext"],
            cutoff=110,
            truncated_dim=device.levels,
            id_str=device.label,
        )


class OscillatorMapping(ModelMapping):
    """Map ``scqubits.Oscillator`` to and from :class:`~quchip.devices.resonator.Resonator`.

    A harmonic oscillator :math:`H = E_{\\rm osc}\\, a^\\dagger a` maps to the
    resonator :math:`H = \\omega\\, \\hat n` with ``freq = E_osc`` and
    ``levels = truncated_dim``. The scqubits ``l_osc`` (an operator-definition
    convention) has no spectral effect and is dropped.
    """

    source = "scqubits.Oscillator"
    target = Resonator

    def import_model(self, obj: Any, *, levels: int | None = None, label: str | None = None,
                     **noise_kwargs: Any) -> Resonator:
        return Resonator(
            freq=obj.E_osc,
            levels=levels or obj.truncated_dim,
            label=label or getattr(obj, "id_str", None),
            **noise_kwargs,
        )

    def export_model(self, device: Any, **opts: Any) -> Any:
        import scqubits

        vals = _concrete_params(device, ("freq",))
        return scqubits.Oscillator(E_osc=vals["freq"], truncated_dim=device.levels, id_str=device.label)


class KerrOscillatorMapping(ModelMapping):
    """Map ``scqubits.KerrOscillator`` to :class:`~quchip.devices.kerr_cavity.KerrCavity` (import-only).

    scqubits writes the Kerr oscillator as
    :math:`H = E_{\\rm osc}\\, a^\\dagger a - K\\, a^\\dagger a^\\dagger a a`,
    whose eigenvalues are :math:`E_n = (E_{\\rm osc} + K)\\, n - K\\, n^2`.
    quchip's :class:`KerrCavity` writes :math:`H = \\omega\\, \\hat n - K'\\,
    \\hat n(\\hat n - 1)`, with eigenvalues
    :math:`E_n = \\omega\\, n - K'\\, n(n-1) = (\\omega + K')\\, n - K'\\, n^2`.

    Matching term by term: the :math:`n^2` coefficient gives ``kerr = K``, and
    the :math:`n` coefficient (with ``kerr = K`` already fixed) gives
    ``freq = E_osc``. Both spectra sit at :math:`E_0 = 0`, so the translation
    is the direct copy ``freq = E_osc``, ``kerr = K`` — no sign flip. Import
    only: :class:`KerrCavity` requires ``kerr >= 0`` and models the ``K > 0``
    (self-focusing) branch scqubits uses.
    """

    source = "scqubits.KerrOscillator"
    target = KerrCavity

    def import_model(self, obj: Any, *, levels: int | None = None, label: str | None = None,
                     **noise_kwargs: Any) -> KerrCavity:
        return KerrCavity(
            freq=obj.E_osc,
            kerr=obj.K,
            levels=levels or obj.truncated_dim,
            label=label or getattr(obj, "id_str", None),
            **noise_kwargs,
        )


class GenericQubitMapping(ModelMapping):
    """Map ``scqubits.GenericQubit`` to :class:`DuffingTransmon` (import-only).

    The generic two-level system :math:`H = \\tfrac12 E\\, \\sigma_z` has level
    splitting ``E``. It maps to a two-level :class:`DuffingTransmon` with
    ``freq = E``, ``anharmonicity = 0`` (irrelevant at two levels), and
    ``levels = 2``.
    """

    source = "scqubits.GenericQubit"
    target = None

    def import_model(self, obj: Any, *, levels: int | None = None, label: str | None = None,
                     **noise_kwargs: Any) -> DuffingTransmon:
        return DuffingTransmon(
            freq=obj.E,
            anharmonicity=0.0,
            levels=levels or 2,
            label=label or getattr(obj, "id_str", None),
            **noise_kwargs,
        )


class DuffingTransmonMapping(ModelMapping):
    """Map :class:`DuffingTransmon` to ``scqubits.Transmon`` (export-only).

    A Duffing transmon is specified by ``(freq, anharmonicity)``; scqubits'
    ``Transmon.find_EJ_EC`` inverts that pair to the ``(EJ, EC)`` that best
    reproduce the same 0->1 splitting and anharmonicity, from which the
    charge-basis transmon is built (``truncated_dim = device.levels``).
    ``ncut`` (default 30, scqubits' own inversion default) is an export
    option, passed identically to both ``find_EJ_EC`` and the reconstructed
    ``Transmon`` so the two never disagree. :class:`DuffingTransmon` has no
    offset-charge concept of its own to translate, so the reconstructed
    transmon is built at the charge sweet spot ``ng=0``. Import-only in the
    other direction is already covered by :class:`TransmonMapping`.
    """

    library = "scqubits"
    source = None
    target = DuffingTransmon

    def export_model(self, device: Any, *, ncut: int = 30, **opts: Any) -> Any:
        import scqubits

        vals = _concrete_params(device, ("freq", "anharmonicity"))
        e_j, e_c = scqubits.Transmon.find_EJ_EC(
            E01=vals["freq"], anharmonicity=vals["anharmonicity"], ncut=ncut
        )
        return scqubits.Transmon(
            EJ=e_j, EC=e_c, ng=0.0, ncut=ncut, truncated_dim=device.levels, id_str=device.label
        )


class ZeroPiMapping(ModelMapping):
    """Map ``scqubits.ZeroPi`` to :class:`~quchip.interop.eigenbasis.EigenbasisDevice` (import-only).

    ZeroPi is a two-mode circuit (:math:`\\phi`, :math:`\\theta`) diagonalized
    on a joint phi-grid / charge-basis product space; quchip has no native
    model for it, since none of its circuit devices carry a second coordinate.
    Rather than reimplementing that two-mode Hamiltonian, this mapping takes
    the exact-lane recipe: diagonalize with scqubits once, then hand the
    resulting energies and eigenbasis-projected operators to
    :class:`~quchip.interop.eigenbasis.EigenbasisDevice`, which treats an
    already-diagonal spectrum as its native basis (see that class's
    docstring). The one ``obj.eigensys(...)`` call is reused for both
    operators via scqubits' ``energy_esys=`` argument
    (:meth:`~scqubits.core.qubit_base.QubitBaseClass.process_op`), so the
    (comparatively expensive) sparse diagonalization runs exactly once.

    The snapshot reproduces ``obj``'s spectrum and charge/phase matrix
    elements exactly, at the parameter point it was taken at, but it is a
    frozen numeric table, not a Hamiltonian recipe: unlike the parametric
    mappings above, the imported device is not differentiable with respect
    to ZeroPi's circuit parameters (``EJ``, ``EL``, ``ECJ``, ``EC``, ``ng``,
    ``flux``). This is the reference recipe for wrapping any other
    scqubits (or third-party) type quchip has no native model for.
    """

    source = "scqubits.ZeroPi"
    target = None

    def import_model(self, obj: Any, *, levels: int | None = None, label: str | None = None,
                      **noise_kwargs: Any) -> EigenbasisDevice:
        import numpy as np

        levels = levels or obj.truncated_dim
        esys = obj.eigensys(evals_count=levels)
        energies = esys[0]
        return EigenbasisDevice(
            energies,
            charge_operator=np.asarray(obj.n_theta_operator(energy_esys=esys)),
            phase_operator=np.asarray(obj.phi_operator(energy_esys=esys)),
            levels=levels,
            label=label or getattr(obj, "id_str", None),
            source_type="scqubits.ZeroPi",
            **noise_kwargs,
        )
