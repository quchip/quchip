"""pyEPR and Quantum Metal adapters read EPR matrices through public attributes only.

The stand-ins below expose the attributes ``from_pyepr`` reads from a pyEPR
``QuantumAnalysis`` (``variations``, ``get_epr_base_matrices``, ``Qs`` and
``modes``) and from a Quantum Metal ``EPRanalysis`` (``sim.renderer``). A
comparison with pyEPR itself is in ``test_interop_epr_physics.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import quchip
from quchip import EPRModel
from quchip.interop.pyepr import from_pyepr

pytestmark = pytest.mark.unit

PARTICIPATIONS = np.array([[0.95, 0.003], [0.01, 0.92], [0.02, 0.03]])
SIGNS = np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
FREQS = np.array([4.8, 5.3, 7.1])
ENERGIES = np.array([13.6, 14.9])


class QuantumAnalysisStub:
    """pyEPR QuantumAnalysis attributes: per-variation matrices, HFSS Qs and analysed modes."""

    def __init__(self, matrices, *, qs=None, modes=None):
        self.variations = list(matrices)
        self._matrices = matrices
        if qs is not None:
            self.Qs = qs
        if modes is not None:
            self.modes = modes

    def get_epr_base_matrices(self, variation):
        participations, signs, freqs, energies = self._matrices[variation]
        # pyEPR returns (PJ, SJ, Om, EJ, PHI_zpf, PJ_cap, n_zpf) with diagonal Om and EJ in GHz.
        return participations, signs, np.diagflat(freqs), np.diagflat(energies), None, None, None


def _analysis(**kwargs):
    return QuantumAnalysisStub({"0": (PARTICIPATIONS, SIGNS, FREQS, ENERGIES)}, **kwargs)


def test_single_variation_imports_pyepr_matrices():
    """The only variation's normalized participations, signs, frequencies and E_J become the model."""
    model = from_pyepr(_analysis())

    assert isinstance(model, EPRModel)
    assert model.labels == ("mode_0", "mode_1", "mode_2")
    np.testing.assert_array_equal(model.freqs, FREQS)
    np.testing.assert_array_equal(model.participations, PARTICIPATIONS)
    np.testing.assert_array_equal(model.signs, SIGNS)
    np.testing.assert_array_equal(model.junction_energies, ENERGIES)
    assert model.quality_factors is None


def test_variation_selection_is_explicit_when_several_exist():
    """A variation key or integer selects its matrices; None is rejected when ambiguous."""
    shifted = (PARTICIPATIONS, SIGNS, FREQS + 0.1, ENERGIES)
    analysis = QuantumAnalysisStub({"0": (PARTICIPATIONS, SIGNS, FREQS, ENERGIES), "1": shifted})

    np.testing.assert_array_equal(from_pyepr(analysis, 1).freqs, FREQS + 0.1)
    np.testing.assert_array_equal(from_pyepr(analysis, "0").freqs, FREQS)
    with pytest.raises(ValueError, match=r"Choose one of the analysed variations: \['0', '1'\]"):
        from_pyepr(analysis)
    with pytest.raises(ValueError, match="Unknown variation"):
        from_pyepr(analysis, "7")


def test_modes_and_junctions_select_pyepr_positions():
    """modes and junctions index pyEPR's analysed-mode and junction order; labels default to positions."""
    model = from_pyepr(_analysis(), modes=[0, 2], junctions=[1])

    assert model.labels == ("mode_0", "mode_2")
    np.testing.assert_array_equal(model.freqs, FREQS[[0, 2]])
    np.testing.assert_array_equal(model.participations, PARTICIPATIONS[[0, 2]][:, [1]])
    np.testing.assert_array_equal(model.signs, SIGNS[[0, 2]][:, [1]])
    np.testing.assert_array_equal(model.junction_energies, ENERGIES[[1]])
    assert from_pyepr(_analysis(), labels=["q0", "q1", "r"]).labels == ("q0", "q1", "r")


def test_quality_factors_follow_the_solved_mode_numbers():
    """HFSS Qs are read by analysed mode number; lossless or invalid Q becomes inf."""
    solved_qs = {"0": {0: 3e5, 1: 1.5e6, 2: -1.0, 3: 2e4, 4: np.inf}}
    model = from_pyepr(_analysis(qs=solved_qs, modes={"0": [1, 2, 3]}))

    np.testing.assert_array_equal(model.quality_factors, [1.5e6, np.inf, 2e4])
    lossless = {"0": {1: np.inf, 2: np.inf, 3: np.inf}}
    assert from_pyepr(_analysis(qs=lossless, modes={"0": [1, 2, 3]})).quality_factors is None
    assert from_pyepr(_analysis(qs=solved_qs)).quality_factors is None


def test_metal_epr_analysis_is_read_through_its_renderer():
    """A Quantum Metal EPRanalysis provides its renderer's QuantumAnalysis and fails clearly before it runs."""
    metal = SimpleNamespace(sim=SimpleNamespace(renderer=SimpleNamespace(epr_quantum_analysis=_analysis())))
    np.testing.assert_array_equal(from_pyepr(metal).freqs, FREQS)

    not_run = SimpleNamespace(sim=SimpleNamespace(renderer=SimpleNamespace(epr_quantum_analysis=None)))
    with pytest.raises(ValueError, match="run its EPR spectrum analysis"):
        from_pyepr(not_run)
    with pytest.raises(TypeError, match="PyaedtDistributedAnalysis, or a Quantum Metal EPRanalysis"):
        from_pyepr(object())


def _pyaedt_analysis(**results):
    """pyEPR PyaedtDistributedAnalysis attributes, which stay None until do_EPR_analysis() runs."""
    fields = dict(freqs_GHz=None, Ljs=None, PJ=None, SJ=None)
    fields.update(results)
    return SimpleNamespace(**fields)


def test_pyaedt_analysis_imports_its_arrays():
    """A PyAEDT analysis's frequencies, participations, signs and inductances become the model."""
    inductances = np.array([12e-9, 11e-9])
    analysis = _pyaedt_analysis(freqs_GHz=FREQS, Ljs=inductances, PJ=PARTICIPATIONS, SJ=SIGNS)
    model = from_pyepr(analysis, modes=[1, 0], junctions=[1], labels=["b", "a"])

    reference = EPRModel(FREQS, PARTICIPATIONS, junction_inductances=inductances)
    np.testing.assert_array_equal(model.freqs, FREQS[[1, 0]])
    np.testing.assert_array_equal(model.participations, PARTICIPATIONS[[1, 0]][:, [1]])
    np.testing.assert_array_equal(model.signs, SIGNS[[1, 0]][:, [1]])
    np.testing.assert_allclose(model.junction_energies, reference.junction_energies[[1]], rtol=1e-15)
    assert model.labels == ("b", "a") and model.quality_factors is None


def test_pyaedt_analysis_needs_results_and_no_variation():
    """A PyAEDT analysis must have run do_EPR_analysis() and holds exactly one variation."""
    with pytest.raises(ValueError, match="do_EPR_analysis"):
        from_pyepr(_pyaedt_analysis())
    solved = _pyaedt_analysis(freqs_GHz=FREQS, Ljs=[12e-9, 11e-9], PJ=PARTICIPATIONS, SJ=SIGNS)
    with pytest.raises(ValueError, match="one solved variation"):
        from_pyepr(solved, variation="0")


def test_from_pyepr_is_a_lazy_top_level_export():
    """quchip.from_pyepr resolves to the adapter and appears in dir(quchip)."""
    assert quchip.from_pyepr is from_pyepr
    assert "from_pyepr" in dir(quchip)
