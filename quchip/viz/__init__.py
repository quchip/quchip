"""Lazy-loaded Matplotlib/pyvis visualization helpers."""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "plot_energy_levels": ("quchip.viz.chip", "plot_energy_levels"),
    "plot_expectation": ("quchip.viz.results", "plot_expectation"),
    "plot_graph": ("quchip.viz.chip", "plot_graph"),
    "plot_populations": ("quchip.viz.results", "plot_populations"),
    "plot_sequence": ("quchip.viz.control", "plot_sequence"),
    "plot_state": ("quchip.viz.results", "plot_state"),
    "plot_wavefunction": ("quchip.viz.device", "plot_wavefunction"),
    "plot_wigner": ("quchip.viz.results", "plot_wigner"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(f"module 'quchip.viz' has no attribute {name!r}")
