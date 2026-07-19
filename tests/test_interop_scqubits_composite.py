"""Composite-import tests: scqubits ``HilbertSpace`` -> quchip ``Chip``.

scqubits is the oracle. A ``HilbertSpace`` of a transmon capacitively coupled
to an oscillator imports to a ``Chip`` whose dressed spectrum must reproduce
``HilbertSpace.eigenvals`` at the same parameter point. The two
not-yet-supported interaction shapes (string expressions and non-pairwise
operator products) must raise a guiding ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

scq = pytest.importorskip("scqubits")

from quchip import Chip  # noqa: E402
from quchip.chip.baths import Bath  # noqa: E402
from quchip.chip.couplings import Capacitive, Coupling, CrossKerr, TunableCapacitive  # noqa: E402
from quchip.control.drive import ChargeDrive  # noqa: E402
from quchip.control.equipment import ControlEquipment  # noqa: E402
from quchip.devices import Resonator  # noqa: E402
from quchip.devices.fluxonium import Fluxonium  # noqa: E402
from quchip.devices.transmon import ChargeBasisTransmon  # noqa: E402
from quchip.interop.scqubits import from_scqubits, to_scqubits  # noqa: E402
from quchip.interop.scqubits.composite import (  # noqa: E402
    _device_gauge_matrix,
    _product_interaction,
)


def _ground_shifted(evals: np.ndarray) -> np.ndarray:
    """Return the real spectrum sorted and shifted so the ground sits at zero."""
    evals = np.real(np.sort(np.asarray(evals)))
    return evals - evals[0]


def _transmon_oscillator_hilbertspace() -> Any:  # type: ignore[valid-type]
    """Build the reference transmon <-> oscillator ``HilbertSpace``."""
    tmon = scq.Transmon(EJ=30.0, EC=0.2, ng=0.25, ncut=31, truncated_dim=4, id_str="tmon")
    osc = scq.Oscillator(E_osc=6.0, truncated_dim=3, id_str="osc")
    hs = scq.HilbertSpace([tmon, osc])
    hs.add_interaction(g=0.035, op1=tmon.n_operator, op2=osc.creation_operator, add_hc=True)
    return hs


# ---------------------------------------------------------------------------
# Composite structure
# ---------------------------------------------------------------------------


def test_hilbertspace_imports_to_chip_structure():
    """Two subsystems and one InteractionTerm import to 2 devices + 1 coupling."""
    hs = _transmon_oscillator_hilbertspace()
    chip = from_scqubits(hs)

    assert isinstance(chip, Chip)
    assert len(chip.devices) == 2
    assert isinstance(chip.devices[0], ChargeBasisTransmon)
    assert isinstance(chip.devices[1], Resonator)
    assert [d.label for d in chip.devices] == ["tmon", "osc"]
    assert len(chip.couplings) == 1
    assert chip.couplings[0].label == "scq_interaction_0"


def test_hilbertspace_dressed_spectrum_matches_oracle():
    """The imported chip's dressed spectrum matches ``hs.eigenvals`` (physics gate)."""
    hs = _transmon_oscillator_hilbertspace()
    chip = from_scqubits(hs)

    got = _ground_shifted(np.asarray(chip.dress().eigenvalues)[:6])
    want = _ground_shifted(hs.eigenvals(evals_count=6))
    np.testing.assert_allclose(got, want, atol=1e-6)


# ---------------------------------------------------------------------------
# Gauge consistency (device-gauge coupling factors)
# ---------------------------------------------------------------------------


def test_coupling_factor_shares_imported_device_gauge():
    """The transmon-side coupling factor equals the device's own charge operator.

    scqubits' eigenvector gauge and the imported ``ChargeBasisTransmon``'s own
    ``jnp.eigh`` gauge differ by a per-eigenvector sign; a coupling factor must
    be transcribed in the *device's* gauge so it agrees elementwise (signs and
    all) with the operator the device hands its own drives.
    """
    hs = _transmon_oscillator_hilbertspace()
    tmon = hs.subsystem_list[0]
    chip = from_scqubits(hs)
    device = chip.devices[0]

    factor = _device_gauge_matrix(tmon, tmon.n_operator, device)
    np.testing.assert_allclose(
        factor, np.asarray(device.charge_coupling_operator()), atol=1e-8
    )


def test_driven_proxy_spectrum_gauge_consistent():
    """A device-gauge charge term added on top of the import matches the oracle.

    Models a static charge perturbation present on *both* sides: scqubits carries
    it as an extra ``n ⊗ I`` interaction (its own gauge for both terms); quchip
    imports the exchange coupling and then adds a charge term built from the
    device's *own* ``charge_coupling_operator``. Only when the imported exchange
    factor also lives in the device gauge do the two quchip terms share one gauge
    and reproduce the oracle spectrum. Under scqubits' eigenvector gauge the two
    quchip terms clash and the dressed spectrum drifts by ~3e-3 GHz.
    """
    tmon_o = scq.Transmon(EJ=30.0, EC=0.2, ng=0.25, ncut=31, truncated_dim=4, id_str="tmon")
    osc_o = scq.Oscillator(E_osc=6.0, truncated_dim=3, id_str="osc")
    hs_oracle = scq.HilbertSpace([tmon_o, osc_o])
    hs_oracle.add_interaction(g=0.035, op1=tmon_o.n_operator, op2=osc_o.creation_operator, add_hc=True)
    hs_oracle.add_interaction(
        g=0.5, op1=tmon_o.n_operator, op2=(np.eye(osc_o.truncated_dim), osc_o), add_hc=False
    )
    want = _ground_shifted(hs_oracle.eigenvals(evals_count=6))

    chip_exchange = from_scqubits(_transmon_oscillator_hilbertspace())
    device_t, device_osc = chip_exchange.devices
    charge_term = Coupling(
        device_t,
        device_osc,
        g=1.0,
        interaction=_product_interaction(
            0.5, np.asarray(device_t.charge_coupling_operator()), np.eye(device_osc.levels), False
        ),
        rwa=False,
        label="charge_proxy",
    )
    chip = Chip(
        devices=list(chip_exchange.devices),
        couplings=list(chip_exchange.couplings) + [charge_term],
    )

    got = _ground_shifted(np.asarray(chip.dress().eigenvalues)[:6])
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_fluxonium_coupling_warns_and_reproduces_spectrum():
    """A cross-basis fluxonium coupling warns yet still tracks the undriven spectrum.

    quchip's fluxonium lives on a phase grid, scqubits' on a harmonic-oscillator
    basis; the coupling factor cannot be re-projected into the device gauge and is
    frozen in the source's, which is flagged. The undriven dressed spectrum is
    gauge-invariant, so it still matches the oracle at the cross-basis tolerance.
    """
    flx = scq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=110, truncated_dim=5, id_str="flx")
    osc = scq.Oscillator(E_osc=6.0, truncated_dim=4, id_str="osc")
    hs = scq.HilbertSpace([flx, osc])
    hs.add_interaction(g=0.02, op1=flx.n_operator, op2=osc.creation_operator, add_hc=True)

    with pytest.warns(UserWarning, match="gauge-inconsistent"):
        chip = from_scqubits(hs)

    got = _ground_shifted(np.asarray(chip.dress().eigenvalues)[:6])
    want = _ground_shifted(hs.eigenvals(evals_count=6))
    np.testing.assert_allclose(got, want, rtol=5e-4)


def _small_zero_pi_hilbertspace(id_str: str = "zp") -> Any:
    """Build a deliberately small ZeroPi + Oscillator ``HilbertSpace`` so the test stays a few seconds."""
    grid = scq.Grid1d(-19.0, 19.0, 200)
    zp = scq.ZeroPi(
        grid=grid, EJ=10.0, EL=0.04, ECJ=20.0, EC=0.04, ng=0.1, flux=0.23,
        ncut=30, truncated_dim=4, id_str=id_str,
    )
    osc = scq.Oscillator(E_osc=0.5, truncated_dim=3, id_str="osc")
    hs = scq.HilbertSpace([zp, osc])
    hs.add_interaction(g=0.01, op1=zp.n_theta_operator, op2=osc.creation_operator, add_hc=True)
    return hs, zp


def test_zeropi_in_hilbertspace():
    """A ZeroPi subsystem imports cleanly; its factor is already device gauge.

    ZeroPi imports as an :class:`EigenbasisDevice`, whose native basis *is*
    scqubits' eigenbasis — the source-eigenbasis path is the device gauge, so
    the import must not warn even though its coupling factor (frozen in that
    eigenbasis) cannot be re-projected.
    """
    import warnings

    hs, zp = _small_zero_pi_hilbertspace()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chip = from_scqubits(hs)

    got = _ground_shifted(np.asarray(chip.dress().eigenvalues)[:6])
    want = _ground_shifted(hs.eigenvals(evals_count=6))
    np.testing.assert_allclose(got, want, atol=1e-6)

    device = chip.devices[0]
    factor = _device_gauge_matrix(zp, zp.n_theta_operator, device)
    np.testing.assert_allclose(
        factor, np.asarray(device.charge_coupling_operator()), atol=1e-8
    )


def test_zeropi_gauge_matrix_never_densifies_native_operator(monkeypatch):
    """The EigenbasisDevice gauge path never materializes ZeroPi's dense native operator.

    ZeroPi's native basis is a joint phi-grid / charge-basis product space —
    thousands of dimensions, densifying to gigabytes. Since an
    :class:`EigenbasisDevice`'s native basis always *is* its (already small)
    truncated eigenbasis, the gauge projection must short-circuit before ever
    calling the native-matrix accessor.
    """
    import quchip.interop.scqubits.composite as composite_mod

    def _must_not_be_called(operator):
        raise AssertionError("_native_matrix must not be called for an EigenbasisDevice-imported subsystem")

    monkeypatch.setattr(composite_mod, "_native_matrix", _must_not_be_called)

    hs, _zp = _small_zero_pi_hilbertspace()
    chip = from_scqubits(hs)
    assert len(chip.devices) == 2


# ---------------------------------------------------------------------------
# Unsupported interaction shapes
# ---------------------------------------------------------------------------


def test_string_expression_interaction_raises():
    """An ``InteractionTermStr`` interaction is rejected with guidance."""
    tmon = scq.Transmon(EJ=30.0, EC=0.2, ng=0.25, ncut=31, truncated_dim=4, id_str="tmon")
    osc = scq.Oscillator(E_osc=6.0, truncated_dim=3, id_str="osc")
    hs = scq.HilbertSpace([tmon, osc])
    hs.add_interaction(
        expr="g * n * (ad + a)",
        op1=("n", tmon.n_operator),
        op2=("ad", osc.creation_operator),
        op3=("a", osc.annihilation_operator),
        const={"g": 0.02},
    )

    with pytest.raises(NotImplementedError, match="string-expression interactions"):
        from_scqubits(hs)


def test_non_pairwise_interaction_raises():
    """An interaction with other than two operators is rejected."""
    tmon = scq.Transmon(EJ=30.0, EC=0.2, ng=0.25, ncut=31, truncated_dim=4, id_str="tmon")
    osc_a = scq.Oscillator(E_osc=6.0, truncated_dim=3, id_str="osc_a")
    osc_b = scq.Oscillator(E_osc=7.0, truncated_dim=3, id_str="osc_b")
    hs = scq.HilbertSpace([tmon, osc_a, osc_b])
    hs.add_interaction(
        g=0.01,
        op1=tmon.n_operator,
        op2=osc_a.creation_operator,
        op3=osc_b.creation_operator,
        add_hc=True,
    )

    with pytest.raises(NotImplementedError):
        from_scqubits(hs)


# ---------------------------------------------------------------------------
# Options pass-through
# ---------------------------------------------------------------------------


def test_frame_and_rwa_forwarded_to_chip():
    """``frame=`` / ``rwa=`` options reach the constructed ``Chip``."""
    hs = _transmon_oscillator_hilbertspace()
    chip = from_scqubits(hs, rwa=False)
    assert chip.rwa is False


# ---------------------------------------------------------------------------
# Composite export — Chip -> HilbertSpace
# ---------------------------------------------------------------------------


def _transmon_oscillator_chip() -> Chip:
    """A quchip transmon capacitively coupled to a resonator (non-RWA form).

    The coupling carries ``rwa=False`` so the chip's own dressed spectrum uses
    the full ``(a + a†)(b + b†)`` form — the same form export emits — making
    the export oracle a clean, approximation-free comparison.
    """
    tmon = ChargeBasisTransmon(E_C=0.2, E_J=30.0, n_g=0.25, levels=4, num_basis=63, label="tmon")
    res = Resonator(freq=6.0, levels=3, label="osc")
    return Chip([tmon, res], couplings=[Capacitive(tmon, res, g=0.035, rwa=False)], rwa=False)


def test_chip_exports_to_hilbertspace_structure():
    """A 2-device, 1-coupling chip exports to 2 subsystems + 1 interaction."""
    hs = to_scqubits(_transmon_oscillator_chip())

    assert isinstance(hs, scq.HilbertSpace)
    assert len(hs.subsystem_list) == 2
    assert isinstance(hs.subsystem_list[0], scq.Transmon)
    assert isinstance(hs.subsystem_list[1], scq.Oscillator)
    assert [s.id_str for s in hs.subsystem_list] == ["tmon", "osc"]
    assert len(hs.interaction_list) == 1


def test_export_spectrum_matches_chip_oracle():
    """The exported HilbertSpace spectrum matches the chip's dressed spectrum.

    quchip is the oracle here: the exported scqubits composite must reproduce
    the chip's own dressed energies (physics gate) to ``1e-6``.
    """
    chip = _transmon_oscillator_chip()
    hs = to_scqubits(chip)

    got = _ground_shifted(hs.eigenvals(evals_count=6))
    want = _ground_shifted(np.asarray(chip.dress().eigenvalues)[:6])
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_export_import_round_trip_spectrum_stable():
    """``import(export(chip))`` reproduces the original chip's dressed spectrum."""
    chip = _transmon_oscillator_chip()
    round_trip = from_scqubits(to_scqubits(chip))

    got = _ground_shifted(np.asarray(round_trip.dress().eigenvalues)[:6])
    want = _ground_shifted(np.asarray(chip.dress().eigenvalues)[:6])
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_export_crosskerr_coupling_matches_oracle():
    """A ``CrossKerr`` chip exports its ``χ·n̂_a n̂_b`` interaction faithfully."""
    tmon = ChargeBasisTransmon(E_C=0.2, E_J=30.0, n_g=0.25, levels=4, num_basis=63, label="tmon")
    res = Resonator(freq=6.0, levels=3, label="osc")
    chip = Chip([tmon, res], couplings=[CrossKerr(tmon, res, chi=0.01)])

    hs = to_scqubits(chip)
    got = _ground_shifted(hs.eigenvals(evals_count=6))
    want = _ground_shifted(np.asarray(chip.dress().eigenvalues)[:6])
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_export_product_coupling_matches_oracle():
    """A product-form ``Coupling`` exports its two operator factors faithfully."""
    tmon = ChargeBasisTransmon(E_C=0.2, E_J=30.0, n_g=0.25, levels=4, num_basis=63, label="tmon")
    res = Resonator(freq=6.0, levels=3, label="osc")
    coupling = Coupling(
        tmon,
        res,
        g=0.03,
        op_a=lambda d: d.number_operator(),
        op_b=lambda d: d.lowering_operator() + d.raising_operator(),
    )
    # Export writes the full (non-RWA) product operator, so the chip's own
    # dressed spectrum must use the full form too — rwa=False.
    chip = Chip([tmon, res], couplings=[coupling], rwa=False)

    hs = to_scqubits(chip)
    got = _ground_shifted(hs.eigenvals(evals_count=6))
    want = _ground_shifted(np.asarray(chip.dress().eigenvalues)[:6])
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_export_raises_when_rwa_resolves_true_on_rwa_sensitive_coupling():
    """Capacitive/TunableCapacitive/product-form couplings raise on export when the chip resolves rwa=True.

    scqubits export always emits the full (non-RWA) operator product; a chip
    that actually resolves rwa=True for one of these couplings would silently
    export different physics than its own RWA-resolved dressed dynamics.
    Explicitly non-RWA export (exercised by ``test_export_*_matches_oracle``
    above) is the other half of this regression: it must keep succeeding.
    """
    def _pair():
        tmon = ChargeBasisTransmon(E_C=0.2, E_J=30.0, n_g=0.25, levels=4, num_basis=63, label="tmon")
        res = Resonator(freq=6.0, levels=3, label="osc")
        return tmon, res

    tmon, res = _pair()
    with pytest.raises(ValueError, match="rwa"):
        to_scqubits(Chip([tmon, res], couplings=[Capacitive(tmon, res, g=0.035)]))

    tmon, res = _pair()
    with pytest.raises(ValueError, match="rwa"):
        to_scqubits(Chip([tmon, res], couplings=[TunableCapacitive(tmon, res, g_0=0.035)]))

    tmon, res = _pair()
    product_coupling = Coupling(
        tmon, res, g=0.03,
        op_a=lambda d: d.number_operator(),
        op_b=lambda d: d.lowering_operator() + d.raising_operator(),
    )
    with pytest.raises(ValueError, match="rwa"):
        to_scqubits(Chip([tmon, res], couplings=[product_coupling]))


def test_export_callable_coupling_raises():
    """A callable-form ``Coupling`` is not exportable — it raises with guidance."""
    tmon = ChargeBasisTransmon(E_C=0.2, E_J=30.0, n_g=0.25, levels=4, num_basis=63, label="tmon")
    res = Resonator(freq=6.0, levels=3, label="osc")
    coupling = Coupling(
        tmon,
        res,
        g=0.03,
        interaction=lambda a, b, bk: bk.tensor(a.number_operator(), b.lowering_operator()),
    )
    chip = Chip([tmon, res], couplings=[coupling])

    with pytest.raises(NotImplementedError, match="Capacitive"):
        to_scqubits(chip)


def test_export_drops_control_equipment_with_warning():
    """Chip-level control equipment is dropped with a single warning."""
    tmon = ChargeBasisTransmon(E_C=0.2, E_J=30.0, n_g=0.25, levels=4, num_basis=63, label="tmon")
    res = Resonator(freq=6.0, levels=3, label="osc")
    drive = ChargeDrive(tmon, label="d0")
    chip = Chip(
        [tmon, res],
        couplings=[Capacitive(tmon, res, g=0.035, rwa=False)],
        control_equipment=ControlEquipment([drive]),
        rwa=False,
    )

    with pytest.warns(UserWarning, match="control equipment"):
        hs = to_scqubits(chip)
    assert isinstance(hs, scq.HilbertSpace)


def test_export_tunable_capacitive_matches_oracle():
    """A ``TunableCapacitive`` chip exports its effective dipole coupling faithfully.

    Exported in its full ``g_0·(a + a†)(b + b†)`` form (same as ``Capacitive``);
    the chip carries ``rwa=False`` so its own dressed spectrum uses that form too,
    making the export oracle approximation-free.
    """
    tmon = ChargeBasisTransmon(E_C=0.2, E_J=30.0, n_g=0.25, levels=4, num_basis=63, label="tmon")
    res = Resonator(freq=6.0, levels=3, label="osc")
    chip = Chip(
        [tmon, res],
        couplings=[TunableCapacitive(tmon, res, g_0=0.035, rwa=False)],
        rwa=False,
    )

    hs = to_scqubits(chip)
    got = _ground_shifted(hs.eigenvals(evals_count=6))
    want = _ground_shifted(np.asarray(chip.dress().eigenvalues)[:6])
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_export_drops_baths_with_warning():
    """A chip-level ``Bath`` is dropped with a single warning naming baths."""
    tmon = ChargeBasisTransmon(E_C=0.2, E_J=30.0, n_g=0.25, levels=4, num_basis=63, label="tmon")
    res = Resonator(freq=6.0, levels=3, label="osc")
    chip = Chip(
        [tmon, res],
        couplings=[Capacitive(tmon, res, g=0.035, rwa=False)],
        baths=[Bath("thermal", temperature=20.0)],
        rwa=False,
    )

    with pytest.warns(UserWarning, match="baths"):
        hs = to_scqubits(chip)
    assert isinstance(hs, scq.HilbertSpace)


def test_export_fluxonium_warns_cross_basis():
    """Exporting a fluxonium (phase grid vs scqubits' cutoff basis) warns.

    The fluxonium's native dimension differs from its exported scqubits
    subsystem's, so the composites agree only to the cross-discretization level;
    a transmon+resonator export (matching bases) must not warn.
    """
    import warnings

    flx = Fluxonium(E_C=2.5, E_J=8.9, E_L=0.5, phi_ext=0.5, levels=4, label="flx")
    res = Resonator(freq=6.0, levels=4, label="osc")
    chip = Chip([flx, res], couplings=[Capacitive(flx, res, g=0.02, rwa=False)], rwa=False)

    with pytest.warns(UserWarning, match="cross-discretization"):
        assert isinstance(to_scqubits(chip), scq.HilbertSpace)

    # The transmon+resonator export shares scqubits' bases and must stay silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(to_scqubits(_transmon_oscillator_chip()), scq.HilbertSpace)
