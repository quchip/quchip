"""Shipped scqubits <-> quchip device mappings.

Each :class:`~quchip.interop.base.ModelMapping` here transcribes one scqubits
circuit-QED object into the quchip device that carries the same spectrum. Where
the inversion is well-defined, the mapping also goes back. Import reads the
source object's native parameters and passes them to the matching quchip
constructor. The constructor rebuilds the Hamiltonian from those parameters, so
the imported device stays differentiable in them. Export reads the concrete
device parameters, guards each against JAX tracers via
:func:`maybe_concrete_scalar`, and reconstructs the scqubits object.

Each mapping's docstring states its parameter translation and serves as a
reference example for new mappings. scqubits stores energies in its global
unit, ``scqubits.get_units()`` (GHz by default). Every energy is converted
between that unit and quchip's GHz at this boundary.
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


def _units_per_ghz() -> float:
    """Return the number of scqubits energy units in one GHz under ``scqubits.get_units()``."""
    from scqubits.core import units

    return units.units_scale_factor("GHz") / units.units_scale_factor()


def _export_levels(device: Any) -> int:
    """Return the device's requested retained dimension, or its authored size."""
    return int(device.projection_levels or device.levels)


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
        units = _units_per_ghz()
        return ChargeBasisTransmon(
            E_C=obj.EC / units,
            E_J=obj.EJ / units,
            n_g=obj.ng,
            levels=levels or obj.truncated_dim,
            num_basis=2 * obj.ncut + 1,
            basis="eigen",
            label=cast(str, label or getattr(obj, "id_str", None)),
            coupling_channel=coupling_channel,
            **noise_kwargs,
        )

    def export_model(self, device: Any, **opts: Any) -> Any:
        import scqubits

        vals = _concrete_params(device, ("E_C", "E_J", "n_g"))
        units = _units_per_ghz()
        return scqubits.Transmon(
            EJ=vals["E_J"] * units,
            EC=vals["E_C"] * units,
            ng=vals["n_g"],
            ncut=(device.num_basis - 1) // 2,
            truncated_dim=_export_levels(device),
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
    the object's ``flux`` and passes the resulting fixed-frequency transmon to
    :class:`ChargeBasisTransmon`. The other parameters copy across exactly as
    in :class:`TransmonMapping`. There is no export, because a single frequency
    does not set ``(EJmax, d, flux)``.
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
        units = _units_per_ghz()
        return ChargeBasisTransmon(
            E_C=obj.EC / units,
            E_J=effective_e_j / units,
            n_g=obj.ng,
            levels=levels or obj.truncated_dim,
            num_basis=2 * obj.ncut + 1,
            basis="eigen",
            label=cast(str, label or getattr(obj, "id_str", None)),
            coupling_channel=coupling_channel,
            **noise_kwargs,
        )


class FluxoniumMapping(ModelMapping):
    """Map ``scqubits.Fluxonium`` to and from :class:`~quchip.devices.fluxonium.Fluxonium`.

    Parameter copy across the three circuit energies and the external flux:

    ==================  ======================
    scqubits            quchip
    ==================  ======================
    ``EC``              ``E_C``
    ``EJ``              ``E_J``
    ``EL``              ``E_L``
    ``flux``            ``phi_ext``
    ``truncated_dim``   ``levels``
    ==================  ======================

    The native discretizations differ: scqubits uses a harmonic-oscillator basis of
    size ``cutoff``, whereas quchip uses a finite-difference phase grid of
    ``num_basis`` points. Therefore quchip keeps its own default grid and does not
    mirror ``cutoff``. Both represent the fluxonium Hamiltonian, but check each
    discretization's convergence for the selected parameters. Export uses the
    scqubits default ``cutoff=110``.
    """

    source = "scqubits.Fluxonium"
    target = Fluxonium

    def import_model(self, obj: Any, *, levels: int | None = None, label: str | None = None,
                     **noise_kwargs: Any) -> Fluxonium:
        units = _units_per_ghz()
        return Fluxonium(
            E_C=obj.EC / units,
            E_J=obj.EJ / units,
            E_L=obj.EL / units,
            phi_ext=obj.flux,
            levels=levels or obj.truncated_dim,
            basis="eigen",
            label=cast(str, label or getattr(obj, "id_str", None)),
            **noise_kwargs,
        )

    def export_model(self, device: Any, **opts: Any) -> Any:
        import scqubits

        vals = _concrete_params(device, ("E_C", "E_J", "E_L", "phi_ext"))
        units = _units_per_ghz()
        return scqubits.Fluxonium(
            EJ=vals["E_J"] * units,
            EC=vals["E_C"] * units,
            EL=vals["E_L"] * units,
            flux=vals["phi_ext"],
            cutoff=110,
            truncated_dim=_export_levels(device),
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
            freq=obj.E_osc / _units_per_ghz(),
            levels=levels or obj.truncated_dim,
            label=cast(str, label or getattr(obj, "id_str", None)),
            **noise_kwargs,
        )

    def export_model(self, device: Any, **opts: Any) -> Any:
        import scqubits

        vals = _concrete_params(device, ("freq",))
        return scqubits.Oscillator(
            E_osc=vals["freq"] * _units_per_ghz(), truncated_dim=device.levels, id_str=device.label
        )


class KerrOscillatorMapping(ModelMapping):
    """Map ``scqubits.KerrOscillator`` to :class:`~quchip.devices.kerr_cavity.KerrCavity` (import-only).

    scqubits writes the Kerr oscillator as
    :math:`H = E_{\\rm osc}\\, a^\\dagger a - K\\, a^\\dagger a^\\dagger a a`,
    with the eigenvalues :math:`E_n = (E_{\\rm osc} + K)\\, n - K\\, n^2`. The
    quchip :class:`KerrCavity` writes
    :math:`H = \\omega\\, \\hat n - K'\\, \\hat n(\\hat n - 1)`, with the
    eigenvalues
    :math:`E_n = \\omega\\, n - K'\\, n(n-1) = (\\omega + K')\\, n - K'\\, n^2`.

    Match the terms one by one. The :math:`n^2` coefficient gives ``kerr = K``.
    With ``kerr = K`` fixed, the :math:`n` coefficient gives ``freq = E_osc``.
    Both spectra start at :math:`E_0 = 0`, so the translation is the direct
    copy ``freq = E_osc``, ``kerr = K``, with no sign flip. Import only:
    :class:`KerrCavity` requires ``kerr >= 0``, and it models the ``K > 0``
    (self-focusing) branch that scqubits uses.
    """

    source = "scqubits.KerrOscillator"
    target = KerrCavity

    def import_model(self, obj: Any, *, levels: int | None = None, label: str | None = None,
                     **noise_kwargs: Any) -> KerrCavity:
        units = _units_per_ghz()
        return KerrCavity(
            freq=obj.E_osc / units,
            kerr=obj.K / units,
            levels=levels or obj.truncated_dim,
            label=cast(str, label or getattr(obj, "id_str", None)),
            **noise_kwargs,
        )


class GenericQubitMapping(ModelMapping):
    """Map ``scqubits.GenericQubit`` to :class:`DuffingTransmon` (import-only).

    The generic two-level system :math:`H = \\tfrac12 E\\, \\sigma_z` has the
    level splitting ``E``. It maps to a two-level :class:`DuffingTransmon` with
    ``freq = E``, ``anharmonicity = 0`` (irrelevant at two levels), and
    ``levels = 2``.
    """

    source = "scqubits.GenericQubit"
    target = None

    def import_model(self, obj: Any, *, levels: int | None = None, label: str | None = None,
                     **noise_kwargs: Any) -> DuffingTransmon:
        return DuffingTransmon(
            freq=obj.E / _units_per_ghz(),
            anharmonicity=0.0,
            levels=levels or 2,
            label=cast(str, label or getattr(obj, "id_str", None)),
            **noise_kwargs,
        )


class DuffingTransmonMapping(ModelMapping):
    """Map :class:`DuffingTransmon` to ``scqubits.Transmon`` (export-only).

    A Duffing transmon is specified by ``(freq, anharmonicity)``. The scqubits
    ``Transmon.find_EJ_EC`` inverts that pair to the ``(EJ, EC)`` that best
    reproduce the same 0->1 splitting and anharmonicity. The export builds the
    charge-basis transmon from these values
    (``truncated_dim = device.levels``). ``ncut`` (default 30, the scqubits
    inversion default) is an export option. The same ``ncut`` goes to
    ``find_EJ_EC`` and to the reconstructed ``Transmon``, so the two always
    agree. :class:`DuffingTransmon` has no offset-charge concept of its own to
    translate, so the export builds the transmon at the charge sweet spot
    ``ng=0``. The other direction is import only and already covered by
    :class:`TransmonMapping`.
    """

    library = "scqubits"
    source = None
    target = DuffingTransmon

    def export_model(self, device: Any, *, ncut: int = 30, **opts: Any) -> Any:
        import scqubits

        vals = _concrete_params(device, ("freq", "anharmonicity"))
        units = _units_per_ghz()
        e_j, e_c = scqubits.Transmon.find_EJ_EC(
            E01=vals["freq"] * units, anharmonicity=vals["anharmonicity"] * units, ncut=ncut
        )
        return scqubits.Transmon(
            EJ=e_j, EC=e_c, ng=0.0, ncut=ncut, truncated_dim=device.levels, id_str=device.label
        )


class ZeroPiMapping(ModelMapping):
    """Map ``scqubits.ZeroPi`` to :class:`~quchip.interop.eigenbasis.EigenbasisDevice` (import-only).

    ZeroPi is a two-mode circuit (:math:`\\phi`, :math:`\\theta`), which
    scqubits diagonalizes on a joint phi-grid / charge-basis product space.
    There is no native quchip model for it, because none of its circuit devices
    carry a second coordinate. Instead of reimplementing that two-mode
    Hamiltonian, this mapping takes the frozen-snapshot path and diagonalizes
    with scqubits. It passes the resulting energies and eigenbasis-projected
    operators to :class:`~quchip.interop.eigenbasis.EigenbasisDevice`. That
    class uses an already-diagonal spectrum as its native basis (see its
    docstring).

    The snapshot reproduces the spectrum and charge/phase matrix elements of
    ``obj`` exactly at its parameter point, but it is a frozen numeric table,
    not a parametric Hamiltonian model. Unlike the parametric mappings above,
    the imported device is not differentiable in the ZeroPi circuit parameters
    (``EJ``, ``EL``, ``ECJ``, ``EC``, ``ng``, ``flux``). This path is the
    reference for wrapping any other scqubits (or third-party) type that has no
    native quchip model.
    """
    # The one `obj.eigensys(...)` call is reused for both operators via the
    # scqubits `energy_esys=` argument (`process_op`), so the expensive sparse
    # diagonalization runs exactly once.

    source = "scqubits.ZeroPi"
    target = None

    def import_model(self, obj: Any, *, levels: int | None = None, label: str | None = None,
                      **noise_kwargs: Any) -> EigenbasisDevice:
        import numpy as np

        levels = levels or obj.truncated_dim
        esys = obj.eigensys(evals_count=levels)
        energies = esys[0] / _units_per_ghz()
        return EigenbasisDevice(
            energies,
            charge_operator=np.asarray(obj.n_theta_operator(energy_esys=esys)),
            phase_operator=np.asarray(obj.phi_operator(energy_esys=esys)),
            levels=levels,
            label=label or getattr(obj, "id_str", None),
            source_type="scqubits.ZeroPi",
            **noise_kwargs,
        )
