"""Public package surface for quchip.

This module re-exports the primary user-facing objects that make up a
quchip study: device models (:class:`DuffingTransmon`,
:class:`Resonator`, ...), chip topology and analysis
(:class:`Chip`, :class:`Capacitive`, :class:`DressedResult`, ...),
classical control (drives, envelopes, sequences, crosstalk), the
engine entry points (:func:`simulate`, :func:`build_problem`,
:func:`solve_problem`, :func:`solve_many`), backend selection helpers,
backend-agnostic result containers, sweep helpers
(:class:`Sweep`, :class:`SpectrumSweep`), post-hoc analysis
(:func:`effective_hamiltonian`, :func:`analyze_cross_resonance`, ...),
the immutable physical constants, and :func:`enable_compilation_cache`.

Visualization helpers and optional third-party interop (pyvis, scqubits,
matplotlib-based plots, ...) are loaded lazily through the module-level
:func:`__getattr__` so that ``import quchip`` stays fast and does not
force any optional dependency on the core install.
"""

from __future__ import annotations

# ruff: noqa: E402

# JAX defaults to float32, which destroys precision for physics simulation.
# Enable x64 globally so envelope/Hamiltonian math, dressed-state analysis,
# and traced gradients all stay in double precision regardless of which
# backend the user picks. Must run before any quchip submodule import so
# JAX-derived dtypes are float64 from the start.
import jax as _jax

_jax.config.update("jax_enable_x64", True)
del _jax

import os  # noqa: E402
from importlib import import_module  # noqa: E402

__version__ = "0.1.0"

from quchip.analysis import (  # noqa: E402
    CRHamiltonianResult,
    CRSusceptibilityResult,
    DispersiveReadoutResult,
    EffectiveHamiltonianResult,
    StaticZZResult,
    analyze_cross_resonance,
    analyze_cr_susceptibility,
    analyze_dispersive_readout,
    analyze_static_zz,
    effective_hamiltonian,
)
from quchip.backend import get_default_backend, set_default_backend  # noqa: E402
from quchip.chip import (
    ActivePatchResult,
    Bath,
    Capacitive,
    Chip,
    ChipTransform,
    Coupling,
    CrossKerr,
    DressedResult,
    EliminationResult,
    TunableCapacitive,
    active_patch,
    eliminate,
    register_elimination_target,
    register_reduction_method,
    register_retarget_rule,
)
from quchip.control import (
    BaseDrive,
    ChargeDrive,
    ControlEquipment,
    Crosstalk,
    CrosstalkMatrix,
    Delay,
    DriveChannel,
    DriveModulation,
    DriveSignalSpec,
    FluxDrive,
    Gain,
    Gaussian,
    GaussianEdge,
    LinearRamp,
    ParametricDrive,
    PhaseDrive,
    SignalTransform,
    Square,
    SquareWithGaussianEdges,
    TwoPhotonDrive,
)
from quchip.control.batch import ProblemBatch
from quchip.control.sequence import QuantumSequence
from quchip.declarative import (
    CouplingModel,
    DeviceModel,
    EnvelopeShape,
    Modulation,
    Parameter,
    Scalar,
    parameter,
    qnp,
)
from quchip.devices.base import NoiseChannel
from quchip.devices.circuit import CircuitDevice
from quchip.devices.fluxonium import Fluxonium
from quchip.devices.kerr_cavity import KerrCavity
from quchip.devices.protocols import ChargeCoupled, FluxCoupled, PhaseCoupled
from quchip.devices.resonator import Resonator
from quchip.devices.transmon.charge_basis import ChargeBasisTransmon
from quchip.devices.transmon.duffing import DuffingTransmon
from quchip.devices.transmon.flux_tunable import FluxTunableTransmon
from quchip.engine import build_problem, simulate, solve_many, solve_problem
from quchip.interop import EigenbasisDevice, ModelMapping
from quchip.inverse_design import FitADressResult, ObservableReport, fit_a_dress
from quchip.results import ObservableTrace, PartitionedSimulationResult, SimulationBatchResult, SimulationResult
from quchip.sweep import SpectrumSweep, Sweep, ZippedSweep
from quchip.utils.constants import Phi_0, hbar, k_B

_LAZY_INTEROP_EXPORTS = {
    "from_scqubits": ("quchip.interop.scqubits", "from_scqubits"),
    "to_scqubits": ("quchip.interop.scqubits", "to_scqubits"),
}

_LAZY_VIZ_EXPORTS = {
    "plot_energy_levels": ("quchip.viz.chip", "plot_energy_levels"),
    "plot_expectation": ("quchip.viz.results", "plot_expectation"),
    "plot_graph": ("quchip.viz.chip", "plot_graph"),
    "plot_populations": ("quchip.viz.results", "plot_populations"),
    "plot_sequence": ("quchip.viz.control", "plot_sequence"),
    "plot_state": ("quchip.viz.results", "plot_state"),
    "plot_wavefunction": ("quchip.viz.device", "plot_wavefunction"),
    "plot_wigner": ("quchip.viz.results", "plot_wigner"),
}

__all__ = [
    # Version
    "__version__",
    # Declarative extension API
    "CouplingModel",
    "DeviceModel",
    "EnvelopeShape",
    "Parameter",
    "Scalar",
    "Modulation",
    "parameter",
    "qnp",
    # Devices
    "ChargeBasisTransmon",
    "CircuitDevice",
    "DuffingTransmon",
    "FluxTunableTransmon",
    "Fluxonium",
    "KerrCavity",
    "NoiseChannel",
    "Resonator",
    # Coupling Protocols
    "ChargeCoupled",
    "PhaseCoupled",
    "FluxCoupled",
    # Chip topology
    "Chip",
    "Capacitive",
    "Coupling",
    "CrossKerr",
    "TunableCapacitive",
    "DressedResult",
    "Bath",
    "ChipTransform",
    "EliminationResult",
    "eliminate",
    "ActivePatchResult",
    "active_patch",
    "register_retarget_rule",
    "register_elimination_target",
    "register_reduction_method",
    # Pulse envelopes
    "Gaussian",
    "LinearRamp",
    "GaussianEdge",
    "Square",
    "SquareWithGaussianEdges",
    # Simulation
    "simulate",
    "build_problem",
    "solve_problem",
    "solve_many",
    "ObservableTrace",
    "SimulationBatchResult",
    "SimulationResult",
    "PartitionedSimulationResult",
    # Sequence
    "QuantumSequence",
    "ProblemBatch",
    # Backend management
    "get_default_backend",
    "set_default_backend",
    "enable_compilation_cache",
    # Constants
    "k_B",
    "hbar",
    "Phi_0",
    # Control
    "BaseDrive",
    "SignalTransform",
    "Delay",
    "Gain",
    "DriveSignalSpec",
    "DriveModulation",
    "ChargeDrive",
    "DriveChannel",
    "FluxDrive",
    "ParametricDrive",
    "PhaseDrive",
    "TwoPhotonDrive",
    "ControlEquipment",
    "Crosstalk",
    "CrosstalkMatrix",
    # Sweep
    "SpectrumSweep",
    "Sweep",
    "ZippedSweep",
    # Inverse design
    "fit_a_dress",
    "FitADressResult",
    "ObservableReport",
    # Analysis
    "analyze_cross_resonance",
    "analyze_cr_susceptibility",
    "analyze_dispersive_readout",
    "analyze_static_zz",
    "effective_hamiltonian",
    "CRHamiltonianResult",
    "CRSusceptibilityResult",
    "DispersiveReadoutResult",
    "EffectiveHamiltonianResult",
    "StaticZZResult",
    # Third-party interop
    "ModelMapping",
    "EigenbasisDevice",
    # Interop (lazy)
    "from_scqubits",
    "to_scqubits",
    # Visualization (lazy)
    "plot_energy_levels",
    "plot_expectation",
    "plot_graph",
    "plot_populations",
    "plot_sequence",
    "plot_state",
    "plot_wavefunction",
    "plot_wigner",
]


def enable_compilation_cache(path: str | None = None, *, min_compile_time_secs: float = 0.01) -> str:
    """Enable JAX's on-disk persistent compilation cache (opt-in).

    JAX compiles each traced kernel (the ``label_eigensystem`` scan, ``eigh`` /
    ``diag`` / broadcast kernels, dynamiqs solver steps, ...) to XLA when it
    is called for the first time in a process. Those compilations are
    memoized in memory within a process, but every *new* process re-pays
    them. This helper points JAX at a per-user on-disk cache so a later
    process whose ``(shape, policy, jaxlib)`` fingerprint matches loads the
    compiled executable instead of recompiling.

    Fully transparent to ``jit`` / ``grad`` / ``vmap`` — it only changes where
    compiled artifacts are stored, never traced values or physics. It is opt-in:
    the package never writes to the user's disk unless this is called
    explicitly. JAX keys cache entries on the
    ``jaxlib`` version, so a toolchain upgrade produces fresh entries rather
    than loading a stale executable.

    Parameters
    ----------
    path
        Cache directory. Defaults to ``$XDG_CACHE_HOME/quchip/jax`` (or
        ``~/.cache/quchip/jax``) — a per-user path, never a shared/system one,
        so the cache stays inside the user's trust boundary.
    min_compile_time_secs
        Skip caching kernels that compile faster than this. The default
        ``0.01`` captures quchip's sub-second kernels that JAX's ``~1s`` default
        would otherwise never persist.

    Returns
    -------
    str
        The absolute cache directory now in use.
    """
    import jax

    if path is None:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        path = os.path.join(base, "quchip", "jax")
    cache_dir = os.path.abspath(path)
    os.makedirs(cache_dir, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return cache_dir


def __getattr__(name: str):
    """Resolve lazy exports on first access.

    Looked up in order: optional third-party interop, then visualization
    helpers. Any attribute not present in either table raises the
    standard :class:`AttributeError` so ``hasattr`` and ``dir``
    behave correctly.
    """
    for lazy in (_LAZY_INTEROP_EXPORTS, _LAZY_VIZ_EXPORTS):
        if name in lazy:
            module_name, attr_name = lazy[name]
            return getattr(import_module(module_name), attr_name)
    raise AttributeError(f"module 'quchip' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazy-export names alongside eagerly bound module attributes.

    Without this, ``dir(quchip)`` would only show names already present
    in ``globals()`` — a lazy export resolves fine through
    :func:`__getattr__` (so ``hasattr``/direct access work) but would
    never appear in ``dir()`` or tab-completion until first accessed.
    """
    return sorted(set(globals()) | set(_LAZY_INTEROP_EXPORTS) | set(_LAZY_VIZ_EXPORTS))
