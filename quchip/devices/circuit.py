r"""CircuitDevice — shared base for diagonalize/truncate/project device models.

Concrete subclasses (fluxonium, charge-basis transmon) implement three
methods that live entirely in the *native* basis (phase grid or integer
charge basis):

* :meth:`_build_native_hamiltonian` — total Hamiltonian in native basis.
* :meth:`_native_charge_operator` — charge operator :math:`\hat n` in native basis.
* :meth:`_native_phase_operator` — phase-space coupling operator
  (:math:`\hat\varphi` or :math:`\sin\hat\varphi` depending on basis)
  in native basis.

The base class handles:

* JAX-traceable diagonalization (:func:`_truncated_eigh`, an ``eigh``
  whose reverse pass involves only kept-level gaps, so degenerate
  discarded levels cannot NaN circuit-parameter gradients) cached on the
  :attr:`BaseDevice.state_version` counter (tracer-safe via
  :func:`quchip.utils.jax_utils.contains_tracer`).
* Truncation to ``levels`` lowest eigenstates; spectrum shifted so
  :math:`E_0 = 0`.
* :meth:`project_operator` transforming a native-basis matrix ``O`` to
  :math:`V^\dagger O V` in the truncated eigenbasis.
* Physical-coupling accessors (:meth:`charge_coupling_operator`,
  :meth:`phase_coupling_operator`) consumed by drives that dispatch via
  :mod:`quchip.devices.protocols`.
* Fermi-golden-rule Lindblad channels in :meth:`collapse_operators`.

Init-order contract (subclass discipline)
-----------------------------------------
Subclass ``__init__`` must set all native-Hamiltonian parameters
(``E_C``, ``E_J``, ``E_L``, ``phi_ext``, ``num_basis``, …) before
returning, and must not call ``_eigensys`` from within ``__init__``.
Violating this would cache against a partially-constructed Hamiltonian.
The base class enforces this by instantiating lazily: no accessor is
called during ``__init__``. Mutation tracking — which gates
:meth:`~quchip.devices.base.BaseDevice._validate_param_write` — switches on
automatically once the outermost ``__init__`` returns
(:class:`~quchip.utils.state_versioning.StateVersioned`); subclasses do not
call ``_finish_init`` by hand.

See ``docs/superpowers/specs/2026-04-21-circuit-level-devices-design.md``.
"""

from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Any, ClassVar, Literal

import jax
import jax.numpy as jnp

from quchip.backend.protocol import Operator
from quchip.devices.base import (
    BaseDevice,
    NoiseChannel,
    _pure_dephasing_channel,
    _thermal_emission_channel,
)
from quchip.utils.jax_utils import maybe_concrete_scalar
from quchip.utils.jax_utils import contains_tracer


def _level_projector(levels: int, i: int, j: int) -> Any:
    """|i><j| as a (levels, levels) complex jnp array."""
    return jnp.zeros((levels, levels), dtype=jnp.complex128).at[i, j].set(1.0)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _truncated_eigh(matrix: Any, keep: int) -> tuple[Any, Any]:
    r"""Lowest ``keep`` eigenpairs of a Hermitian ``matrix``, with a truncation-aware gradient.

    Forward-identical to ``jnp.linalg.eigh`` followed by ``[:keep]`` /
    ``[:, :keep]`` truncation. The custom reverse pass exists because the
    stock ``eigh`` VJP divides by *every* eigenvalue gap: a native basis
    whose high-lying (discarded) levels are numerically degenerate — e.g.
    the ±n charge pairs of a transmon at the ``n_g = 0`` and ``n_g = 0.5``
    sweet spots — produces ``0 · ∞ = NaN`` there even though those levels
    carry zero cotangent. Here only kept-versus-all gaps enter, guarded by
    the double-``where`` pattern, so circuit-parameter gradients stay
    finite whenever the *kept* levels are non-degenerate. An exactly
    degenerate kept pair still has no well-defined eigenvector gradient;
    its guarded cross term is zeroed rather than NaN.
    """
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    return eigvals[:keep], eigvecs[:, :keep]


def _truncated_eigh_fwd(matrix: Any, keep: int) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    return (eigvals[:keep], eigvecs[:, :keep]), (eigvals, eigvecs)


def _truncated_eigh_bwd(keep: int, residuals: tuple[Any, Any], cotangents: tuple[Any, Any]) -> tuple[Any]:
    eigvals, eigvecs = residuals
    vals_bar, vecs_bar = cotangents
    kept_vecs = eigvecs[:, :keep]

    # dλ_i = v_iᴴ dH v_i  →  H̄ += Σ_i v_i λ̄_i v_iᴴ.
    grad = (kept_vecs * vals_bar[None, :]) @ kept_vecs.conj().T

    # dv_i = Σ_{j≠i} v_j (v_jᴴ dH v_i)/(λ_i − λ_j): denominators pair a
    # kept level i with any level j. Gaps below ``tol`` (exact kept-pair
    # degeneracies; the j = i diagonal) are excluded by double-where.
    gaps = eigvals[None, :keep] - eigvals[:, None]
    tol = 1e-9 * jnp.max(jnp.abs(eigvals))
    resolved = jnp.abs(gaps) > tol
    inv_gaps = jnp.where(resolved, 1.0 / jnp.where(resolved, gaps, 1.0), 0.0)
    overlaps = eigvecs.conj().T @ vecs_bar
    grad = grad + eigvecs @ (inv_gaps * overlaps) @ kept_vecs.conj().T

    # eigh differentiates within the Hermitian submanifold.
    grad = 0.5 * (grad + grad.conj().T)
    return (grad.astype(eigvecs.dtype),)


_truncated_eigh.defvjp(_truncated_eigh_fwd, _truncated_eigh_bwd)


def _golden_rule_emission_channel(device: "CircuitDevice") -> list[Operator]:
    r"""Fermi-golden-rule relaxation / absorption channels for a circuit device.

    The T1 normalization
    :math:`\gamma_{ij} = (1/T_1) \cdot |N_{ij}|^2 / |N_{01}|^2`
    assumes T1 is set by a specific bath coupling operator ``N`` whose
    choice is fixed at construction by ``coupling_channel``:

    * ``coupling_channel='charge'`` → ``N = n̂`` (charge-operator
      relaxation; Breuer & Petruccione §3.4; Smith et al. PRX
      Quantum **2**, 010339 (2021) §III.B).
    * ``coupling_channel='flux'`` → ``N = ∂H/∂φ_ext`` up to a
      global scale, which for a phase-basis device reduces to
      ``φ̂`` (constant terms in ``∂H/∂φ_ext`` only contribute on
      the diagonal and drop out of off-diagonal matrix elements,
      so ``|<i|φ̂|j>|²`` is the correct relative weight). Use this
      at the fluxonium flux sweet spot, where flux-noise-limited
      T1 dominates charge relaxation.

    When ``collapse_model='ladder'``, defers to the structural Fock-ladder
    channel from :class:`~quchip.devices.base.BaseDevice`. All returned
    operators are pure JAX arrays (rank-1 projectors built directly from
    :mod:`jax.numpy`), fully JAX-traceable independent of the active
    default backend.
    """
    if device.collapse_model == "ladder":
        return _thermal_emission_channel(device)
    if device.T1 is None:
        return []

    if device.coupling_channel == "flux":
        N = device.phase_coupling_operator()
        channel_name = "flux"
        op_symbol = "φ̂"
    else:
        N = device.charge_coupling_operator()
        channel_name = "charge"
        op_symbol = "n̂"
    gamma_0 = 1.0 / device.T1
    normalization = jnp.abs(N[0, 1]) ** 2

    norm_concrete = maybe_concrete_scalar(normalization)
    if norm_concrete is not None and norm_concrete < 1e-24:
        raise ValueError(
            f"|<0|{op_symbol}|1>| ≈ 0 — no {channel_name}-coupled T1 channel to the 0→1 "
            "transition. Check phi_ext (dark at some symmetry points), switch "
            "coupling_channel (e.g. 'charge' ↔ 'flux'), or pass collapse_model='ladder' "
            "to use structural Fock channels."
        )

    ops: list[Operator] = []
    for i in range(1, device.levels):
        for j in range(i):
            matrix_element_sq = jnp.abs(N[i, j]) ** 2
            rate_ratio = matrix_element_sq / normalization
            ratio_concrete = maybe_concrete_scalar(rate_ratio)
            if ratio_concrete is not None and ratio_concrete < device.collapse_rate_threshold:
                continue
            rate = gamma_0 * rate_ratio
            ops.extend(
                BaseDevice._emission_pair(
                    rate,
                    device.thermal_population,
                    _level_projector(device.levels, j, i),
                    _level_projector(device.levels, i, j),
                )
            )
    return ops


def _eigenbasis_dephasing_channel(device: "CircuitDevice") -> list[Operator]:
    """Pure dephasing on the level-index operator ``diag(0, 1, 2, …)``.

    Same ``gamma_phi`` algebra and ``sqrt(2*gamma_phi)`` normalization as
    the base-device channel (see :meth:`BaseDevice._dephasing_op`), applied
    to the circuit device's eigenbasis level index. Physically incomplete
    for fluxonium at flux-sensitive points (sweet-spot flux protection is
    not captured). ``collapse_model='ladder'`` defers to the base channel.
    """
    if device.collapse_model == "ladder":
        return _pure_dephasing_channel(device)
    gamma_phi = BaseDevice._dephasing_rate(device.T1, device.T2)
    if gamma_phi is None:
        return []
    level_index_op = jnp.diag(jnp.arange(device.levels, dtype=jnp.complex128))
    return [BaseDevice._dephasing_op(gamma_phi, level_index_op)]


def _check_positive_energy(name: str, value: Any) -> None:
    """Raise ``ValueError`` if *value* is a concrete non-positive scalar.

    JAX tracers pass through unchecked: validation runs only
    on values :func:`maybe_concrete_scalar` can read as a concrete scalar.
    Shared by :class:`~quchip.devices.fluxonium.Fluxonium` and
    :class:`~quchip.devices.transmon.charge_basis.ChargeBasisTransmon` to
    validate ``E_C``, ``E_J``, ``E_L``.
    """
    concrete = maybe_concrete_scalar(value)
    if concrete is not None and concrete <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


class CircuitDevice(BaseDevice):
    """Abstract base for devices built by diagonalizing a native-basis circuit.

    Conforms to the :class:`~quchip.devices.protocols.ChargeCoupled` and
    :class:`~quchip.devices.protocols.PhaseCoupled` Protocols by default;
    subclasses may additionally conform to :class:`FluxCoupled` by
    defining :meth:`flux_coupling_operator`.

    Subclasses must implement:

    * :meth:`_build_native_hamiltonian`
    * :meth:`_native_charge_operator`
    * :meth:`_native_phase_operator`
    """

    # Coupling channels this circuit device's Fermi-golden-rule bath model
    # supports. A subclass with no flux degree of freedom (e.g.
    # ChargeBasisTransmon) narrows this to ("charge",) so both construction
    # and post-construction writes reject coupling_channel='flux'.
    _ALLOWED_COUPLING_CHANNELS: tuple[str, ...] = ("charge", "flux")

    # Subset of tunable_param_names that are energy-valued (must stay
    # positive on every write, checked via _check_positive_energy).
    # Non-energy tunables (phi_ext, n_g) are excluded on purpose — they are
    # not sign-constrained. Concrete subclasses declare this explicitly.
    _ENERGY_PARAM_NAMES: tuple[str, ...] = ()

    #: Declared approximation-regime statement surfaced by
    #: :meth:`physics_notes`, mirroring
    #: :attr:`~quchip.declarative.models.DeviceModel.approximation` — this
    #: class does not inherit from ``DeviceModel``, so the attribute and its
    #: surfacing are declared here directly.
    approximation: str | None = None

    def __init__(
        self,
        levels: int,
        label: str | None = None,
        *,
        collapse_model: Literal["fermi_golden", "ladder"] = "fermi_golden",
        coupling_channel: Literal["charge", "flux"] | None = None,
        collapse_rate_threshold: float = 1e-8,
        **noise_kwargs: float | None,
    ) -> None:
        super().__init__(levels=levels, label=label, **noise_kwargs)
        if collapse_model not in ("fermi_golden", "ladder"):
            raise ValueError(f"collapse_model must be 'fermi_golden' or 'ladder', got {collapse_model!r}")
        if collapse_rate_threshold < 0:
            raise ValueError(f"collapse_rate_threshold must be non-negative, got {collapse_rate_threshold}")
        if coupling_channel is not None and coupling_channel not in self._ALLOWED_COUPLING_CHANNELS:
            raise ValueError(
                f"{type(self).__name__} coupling_channel must be one of "
                f"{self._ALLOWED_COUPLING_CHANNELS} or None, got {coupling_channel!r}"
            )
        self._check_t1_channel_contract(noise_kwargs.get("T1"), collapse_model, coupling_channel)
        self.collapse_model = collapse_model
        self.coupling_channel = coupling_channel
        self.collapse_rate_threshold = float(collapse_rate_threshold)

    def _check_t1_channel_contract(self, T1: Any, collapse_model: Any, coupling_channel: Any) -> None:
        """Reject ``fermi_golden`` with ``T1`` set but no bath coupling channel.

        Run by the constructor and by every post-construction write to any
        of the three fields, so a valid object can never be mutated into an
        invalid ``(T1, collapse_model, coupling_channel)`` combination.
        """
        if collapse_model == "fermi_golden" and T1 is not None and coupling_channel is None:
            raise ValueError(
                "collapse_model='fermi_golden' with T1 set requires an explicit "
                "coupling_channel. The FG normalization uses matrix elements of a "
                "specific bath coupling operator and the correct choice depends on "
                "the device and operating point (e.g. fluxonium at the flux sweet "
                "spot is flux-noise-dominated, not charge-coupled). Set a "
                "coupling_channel from "
                f"{self._ALLOWED_COUPLING_CHANNELS}, or use collapse_model='ladder'."
            )

    def _validate_param_write(self, name: str, value: Any) -> None:
        """Give post-construction writes the same validation as the constructor.

        Extends :meth:`~quchip.devices.base.BaseDevice._validate_param_write`
        with the circuit-level checks the constructor runs: declared energy
        parameter positivity (:attr:`_ENERGY_PARAM_NAMES`), the
        ``collapse_model`` / ``coupling_channel`` literals,
        ``collapse_rate_threshold`` non-negativity, the ``levels`` /
        ``num_basis`` truncation bounds, and the
        coupling-channel-required-for-T1 constructor contract. Checks apply
        to concrete scalars only; traced writes flow through unchecked.
        """
        super()._validate_param_write(name, value)
        if name in self._ENERGY_PARAM_NAMES:
            _check_positive_energy(name, value)
        elif name == "collapse_model":
            if value not in ("fermi_golden", "ladder"):
                raise ValueError(f"collapse_model must be 'fermi_golden' or 'ladder', got {value!r}")
            self._check_t1_channel_contract(self.T1, value, self.coupling_channel)
        elif name == "coupling_channel":
            if value is not None and value not in self._ALLOWED_COUPLING_CHANNELS:
                raise ValueError(
                    f"{type(self).__name__} coupling_channel must be one of "
                    f"{self._ALLOWED_COUPLING_CHANNELS} or None, got {value!r}"
                )
            self._check_t1_channel_contract(self.T1, self.collapse_model, value)
        elif name == "collapse_rate_threshold":
            concrete = maybe_concrete_scalar(value)
            if concrete is not None and concrete < 0:
                raise ValueError(f"collapse_rate_threshold must be non-negative, got {value}")
        elif name == "levels":
            num_basis = getattr(self, "num_basis", None)
            if num_basis is not None and value > num_basis:
                raise ValueError(f"levels ({value}) cannot exceed num_basis ({num_basis})")
        elif name == "num_basis":
            if value < 3:
                raise ValueError(f"num_basis must be >= 3, got {value}")
            if value < self.levels:
                raise ValueError(f"num_basis ({value}) cannot be smaller than levels ({self.levels})")
        elif name == "T1":
            self._check_t1_channel_contract(value, self.collapse_model, self.coupling_channel)

    # ------------------------------------------------------------------
    # Abstract native-basis interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_native_hamiltonian(self) -> Any:
        """Return the full Hamiltonian ``H`` in the native basis."""

    @abstractmethod
    def _native_charge_operator(self) -> Any:
        r"""Return the charge operator :math:`\hat n` in the native basis."""

    @abstractmethod
    def _native_phase_operator(self) -> Any:
        r"""Return the phase-space coupling operator in the native basis.

        On a phase-basis device this is :math:`\hat\varphi`; on an
        integer-charge-basis device this is typically :math:`\sin\hat\varphi`
        (since :math:`\hat\varphi` is not single-valued in the charge basis).
        """

    # ------------------------------------------------------------------
    # Eigendecomposition with state_version-keyed cache
    # ------------------------------------------------------------------

    def _eigensys(self) -> tuple[Any, Any]:
        """Return ``(eigvals_shifted, V_truncated)`` in the truncated eigenbasis.

        ``eigvals_shifted`` is a 1D array of length ``levels``, with
        :math:`E_0 = 0`; ``V_truncated`` is the ``(num_basis, levels)``
        eigenvector matrix.

        Cached on :attr:`state_version`. The cache is written only when
        :func:`~quchip.utils.jax_utils.contains_tracer` returns ``False``
        on the native-Hamiltonian payload, so a tracer bound to the wrong
        JAX trace context is never cached.
        """
        cached = getattr(self, "_eigensys_cache", None)
        if cached is not None and cached[0] == self._state_version:
            return cached[1]

        H_native = self._build_native_hamiltonian()
        eigvals, V_trunc = _truncated_eigh(H_native, self.levels)
        eigvals = eigvals - eigvals[0]
        result = (eigvals, V_trunc)

        if not contains_tracer(H_native):
            object.__setattr__(self, "_eigensys_cache", (self._state_version, result))
        return result

    # ------------------------------------------------------------------
    # Public operator surface
    # ------------------------------------------------------------------

    def hamiltonian(self) -> Operator:
        """Diagonal ``diag(0, E_01, E_02, …)`` in the truncated eigenbasis.

        Returned as a backend-native diagonal operator — the engine relies
        on ``chip.hamiltonian()`` producing the same sparse layout as
        :meth:`number_operator` so layout-aware backends (e.g. dynamiqs
        sparse-DIA) do not silently densify when assembling ``H₀``.
        """
        from quchip.backend import get_default_backend

        eigvals, _ = self._eigensys()
        return get_default_backend().diag(eigvals.astype(jnp.complex128), dims=[[self.levels]])

    def eigenenergies(self) -> Any:
        """Return the truncated eigenvalue array, shape ``(levels,)``, with :math:`E_0 = 0`."""
        eigvals, _ = self._eigensys()
        return eigvals

    def eigenvectors(self) -> Any:
        """Return the truncated eigenvector matrix ``V``, shape ``(num_basis, levels)``."""
        _, V = self._eigensys()
        return V

    def project_operator(self, native_op: Any) -> Any:
        r"""Transform a native-basis operator ``O`` into the truncated eigenbasis.

        Returns :math:`V^\dagger O V`.
        """
        _, V = self._eigensys()
        return V.conj().T @ native_op @ V

    @property
    def freq(self) -> Any:
        r"""Bare 0→1 transition frequency :math:`E_1 - E_0` (GHz)."""
        eigvals, _ = self._eigensys()
        return eigvals[1]

    # ------------------------------------------------------------------
    # Protocol conformance (ChargeCoupled / PhaseCoupled)
    # ------------------------------------------------------------------

    def charge_coupling_operator(self) -> Operator:
        r"""Return the physical charge operator :math:`V^\dagger \hat n V` in the eigenbasis.

        Returned as a dense, trace-safe array-like (see
        :mod:`quchip.devices.protocols`); backend composition entry
        points coerce it to native form on use.
        """
        return self.project_operator(self._native_charge_operator())

    def phase_coupling_operator(self) -> Operator:
        r"""Return the physical phase-coupling operator in the eigenbasis.

        Returns :math:`V^\dagger \hat\varphi V` on a phase-basis device
        (fluxonium); returns :math:`V^\dagger \sin\hat\varphi V` on an
        integer-charge-basis device (charge-basis transmon), since
        :math:`\hat\varphi` is not single-valued there.
        """
        return self.project_operator(self._native_phase_operator())

    # ------------------------------------------------------------------
    # Fermi-golden-rule Lindblad channels
    # ------------------------------------------------------------------

    # Fermi-golden-rule relaxation plus level-index dephasing, declared as
    # noise channels (see the module-level builds for the physics). Both
    # builds defer to the base-device channels when
    # ``collapse_model='ladder'``.
    _noise_channels: ClassVar[tuple[NoiseChannel, ...]] = (
        NoiseChannel(
            "golden_rule_emission", ("T1", "thermal_population"), _golden_rule_emission_channel
        ),
        NoiseChannel("pure_dephasing", ("T2",), _eigenbasis_dephasing_channel),
    )

    # ------------------------------------------------------------------
    # Declared approximations
    # ------------------------------------------------------------------

    def _truncation_note(self) -> str:
        """Return the Hilbert-truncation note for a diagonalized native Hamiltonian."""
        return f"Hilbert truncation: {self.levels} lowest eigenstates of the diagonalized native Hamiltonian"

    def physics_notes(self) -> list[str]:
        """Return declared diagonalization and collapse-model assumptions."""
        notes = super().physics_notes()
        if self.approximation:
            notes.append(self.approximation)
        notes.append("Native-basis exact diagonalization; lowest eigenstates kept (no Duffing expansion)")
        if self.collapse_model == "fermi_golden":
            if self.T1 is not None:
                notes.append(
                    f"T1 model: fermi_golden, {self.coupling_channel}-coupled relaxation "
                    f"(Markov bath with flat S(ω) at ω_01; |N_ij|²/|N_01|² rate weights)"
                )
            elif self.T2 is not None:
                notes.append("T2 model: fermi_golden level-index dephasing (flux-noise structure not captured)")
        else:
            notes.append("Collapse model: structural Fock-ladder (ignores eigenstate matrix elements)")
        return notes

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Extend :meth:`BaseDevice.to_dict` with circuit-level collapse params."""
        data = super().to_dict()
        data["collapse_model"] = self.collapse_model
        data["coupling_channel"] = self.coupling_channel
        data["collapse_rate_threshold"] = float(self.collapse_rate_threshold)
        return data

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover — representation only
        return f"{type(self).__name__}(label={self.label!r}, levels={self.levels})"
