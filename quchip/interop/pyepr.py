"""Read pyEPR and Quantum Metal EPR results into :class:`~quchip.interop.epr.EPRModel`.

pyEPR's ``QuantumAnalysis`` holds the normalized participation, sign,
frequency and junction-energy matrices of each analysed variation. The
``EPRanalysis`` of Quantum Metal, imported as ``qiskit_metal``, runs pyEPR
through its HFSS renderer and keeps that ``QuantumAnalysis`` on the renderer.
pyEPR's ``PyaedtDistributedAnalysis`` extracts one variation over PyAEDT and
keeps its frequencies, participations, signs and inductances as arrays. All
three are read through their public attributes, so this module imports
neither package.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from quchip.interop.epr import _GHZ_HENRY, EPRModel


def _quantum_analysis(analysis: Any) -> Any:
    """Return the pyEPR ``QuantumAnalysis`` held by *analysis*."""
    if hasattr(analysis, "get_epr_base_matrices"):
        return analysis
    renderer = getattr(getattr(analysis, "sim", None), "renderer", None)
    quantum = getattr(renderer, "epr_quantum_analysis", None)
    if quantum is not None:
        return quantum
    if renderer is not None:
        raise ValueError(
            "This Quantum Metal EPRanalysis has no pyEPR QuantumAnalysis yet; "
            "run its EPR spectrum analysis first, for example with run_epr()."
        )
    raise TypeError(
        "Expected a pyEPR QuantumAnalysis or PyaedtDistributedAnalysis, or a Quantum Metal EPRanalysis, "
        f"got {type(analysis).__name__}."
    )


def _variation(quantum: Any, variation: Any) -> str:
    """Return the requested variation key, or the only analysed one."""
    available = [str(key) for key in quantum.variations]
    if variation is None:
        if len(available) != 1:
            raise ValueError(f"Choose one of the analysed variations: {available}.")
        return available[0]
    if str(variation) not in available:
        raise ValueError(f"Unknown variation {variation!r}; analysed variations are {available}.")
    return str(variation)


def _quality_factors(quantum: Any, variation: str, n_modes: int) -> np.ndarray | None:
    """Return HFSS eigenmode quality factors of the analysed modes, or ``None``.

    pyEPR indexes ``Qs`` by solved-mode number and lists the analysed mode
    numbers of each variation in ``modes``.
    """
    try:
        series = quantum.Qs[variation]
        numbers = list(quantum.modes[variation])
    except (AttributeError, KeyError, TypeError):
        return None
    if series is None or len(numbers) != n_modes:
        return None
    lookup = getattr(series, "loc", series)
    values = np.array([float(lookup[number]) for number in numbers], dtype=float)
    # HFSS reports Q = f / 0 for a mode without loss; keep it as an explicit inf.
    values[~(np.isfinite(values) & (values > 0))] = np.inf
    return None if np.all(np.isinf(values)) else values


def _pyaedt_results(analysis: Any, variation: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the frequencies, participations, signs and junction energies of a PyAEDT analysis."""
    if variation is not None:
        raise ValueError("A pyEPR PyaedtDistributedAnalysis holds one solved variation; omit variation.")
    if analysis.PJ is None:
        raise ValueError("This pyEPR PyaedtDistributedAnalysis has no results yet; run do_EPR_analysis() first.")
    return (
        np.asarray(analysis.freqs_GHz, dtype=float),
        np.asarray(analysis.PJ, dtype=float),
        np.asarray(analysis.SJ, dtype=float),
        _GHZ_HENRY / np.asarray(analysis.Ljs, dtype=float),
    )


def from_pyepr(
    analysis: Any,
    variation: Any = None,
    *,
    modes: Sequence[int] | None = None,
    junctions: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
) -> EPRModel:
    """Import one variation of a pyEPR or Quantum Metal EPR analysis.

    Parameters
    ----------
    analysis : pyEPR.QuantumAnalysis, pyEPR PyaedtDistributedAnalysis or Quantum Metal EPRanalysis
        pyEPR quantum analysis; pyEPR's PyAEDT analysis after
        ``do_EPR_analysis()``; or a Quantum Metal ``EPRanalysis`` whose EPR
        spectrum analysis has run.
    variation : str, int or None, default None
        pyEPR variation key, such as ``"0"``. ``None`` selects the only
        analysed variation and raises when there are several. A PyAEDT
        analysis holds one variation and requires ``None``.
    modes : sequence of int or None, default None
        Positions in pyEPR's analysed-mode order, as accepted by
        ``QuantumAnalysis.analyze_variation``. ``None`` keeps every mode.
    junctions : sequence of int or None, default None
        Junction positions in pyEPR's junction order. ``None`` keeps every
        junction.
    labels : sequence of str or None, default None
        Labels of the selected modes. ``None`` gives ``"mode_<position>"``.

    Returns
    -------
    EPRModel
        HFSS mode frequencies in GHz, pyEPR's normalized participations and
        signs, junction energies from pyEPR's ``Ljs``, and the finite HFSS
        eigenmode quality factors, if any. Lossless modes carry ``inf``. A
        PyAEDT analysis gives its own participations and no quality factors.

    Raises
    ------
    TypeError
        *analysis* is not one of the supported types.
    ValueError
        The variation is ambiguous or unknown, a Quantum Metal analysis has
        not run its EPR spectrum analysis, or a PyAEDT analysis has no results
        or was given a variation.

    Notes
    -----
    The participations are pyEPR's normalized matrix, so a first-order
    :meth:`EPRModel.chip` reproduces pyEPR's ``chi_O1`` and ``f_1`` results
    for the same modes and junctions, with quchip's sign convention and units.
    A PyAEDT analysis stores the participations its ``analyze()`` uses, so the
    cosine chip reproduces that diagonalization at equal truncation.

    References
    ----------
    Minev et al., npj Quantum Inf. 7, 131 (2021),
    https://doi.org/10.1038/s41534-021-00461-8.
    """
    quality: np.ndarray | None = None
    if hasattr(analysis, "PJ") and hasattr(analysis, "freqs_GHz"):
        freqs, participations, signs, energies = _pyaedt_results(analysis, variation)
    else:
        quantum = _quantum_analysis(analysis)
        key = _variation(quantum, variation)
        participations, signs, omega, josephson = (np.asarray(m, dtype=float) for m in
                                                   quantum.get_epr_base_matrices(key)[:4])
        freqs = np.diag(omega)
        energies = np.diag(josephson)
        quality = _quality_factors(quantum, key, freqs.shape[0])

    rows = list(range(freqs.shape[0])) if modes is None else [int(mode) for mode in modes]
    columns = list(range(energies.shape[0])) if junctions is None else [int(j) for j in junctions]
    return EPRModel(
        freqs[rows],
        participations[np.ix_(rows, columns)],
        junction_energies=energies[columns],
        signs=signs[np.ix_(rows, columns)],
        labels=tuple(f"mode_{row}" for row in rows) if labels is None else labels,
        quality_factors=None if quality is None else quality[rows],
    )


__all__ = ["from_pyepr"]
