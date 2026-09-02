"""Band-decompose dict-form ``e_ops`` and demodulate expectations post-solve.

The simulation is performed in the resolved rotating frame;
user-facing observables, however, live in the control frame (where
matrix elements are labeled by detunings ``Δ = ω_drive − ω_frame``).
This module performs the two operations that tie those frames together:

* **Pre-solve (:func:`decompose_eops`):** each user operator is split
  into excitation-change bands ``w = col − row`` (single-mode) or
  ``(Δa, Δb)`` (two-mode) via :mod:`quchip.engine.bands`. Each band is
  embedded into the full product Hilbert space and carries
  :class:`BandMeta` so the post-solve step knows which phase to apply.

* **Post-solve (:func:`recombine_expect`):** each band expectation
  ``⟨O_w⟩(t)`` is multiplied by ``exp(±i · 2π · ω_demod · w · t)``
  (the sign selects demodulation vs. remodulation) and summed over
  bands. Physically this is the inverse rotating-frame
  transformation applied to each band — equivalent to moving the
  observable from the simulation frame to the control frame.

Output-field specs take a parallel path: they lower the resolved ``L`` and
``L dagger L`` moments before the solve, then reconstruct ``S beta + L`` from
the solve-bound coherent inputs and propagate it to the exposure reference
plane afterward.

Observable reconstruction reads per-device demodulation frequencies from
:class:`~quchip.engine.ir.ResolvedFrame` and forms the demodulation
phase ``exp(i · 2π · ω · w · t)``. This is the inverse of the
rotating-frame shift applied to the Hamiltonian during assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from quchip.backend import Backend, SolverResult
from quchip.chip.chip import Chip
from quchip.engine.ir import ResolvedFrame
from quchip.observables import (
    OutputAmplitude,
    OutputObservable,
    OutputPhotonFlux,
    OutputQuadrature,
    is_output_observable,
)
from quchip.results.results import ObservableTrace
from quchip.utils.constants import TWO_PI
from quchip.utils.jax_utils import array_namespace as _array_namespace
from quchip.utils.labeling import resolve_label
from quchip.engine.bands import embed_single_mode_bands, local_mode_bands
from quchip.engine.basis import semantic_to_solver_transform

EOpKey = str | tuple[str, str]
BandWeight = int | tuple[int, int]
DeviceLabels = str | tuple[str, str]


@dataclass(frozen=True)
class BandMeta:
    """Post-solve metadata for one flattened e_ops band.

    ``key`` is the original dict key, ``weight`` is the excitation-change
    weight (``int`` single-device, ``tuple[int, int]`` two-device),
    ``device_labels`` drives phase lookup, and ``sub_index`` distinguishes
    entries when the user passed a list of operators for the same key.
    """

    key: EOpKey
    weight: BandWeight
    device_labels: DeviceLabels
    sub_index: int | None = None


@dataclass(frozen=True)
class OutputMeta:
    """Reconstruction recipe for one solver-facing output operator."""

    key: EOpKey
    observable: OutputObservable
    role: Literal["amplitude", "flux"]


def _resolve_eop_key(key: Any) -> EOpKey:
    if isinstance(key, str):
        return key
    if isinstance(key, tuple) and len(key) == 2:
        return (resolve_label(key[0]), resolve_label(key[1]))
    return resolve_label(key)


def decompose_eops(
    e_ops_dict: Mapping[EOpKey, Any],
    chip: Chip,
    backend: Backend,
    bases: Mapping[str, Any] | None = None,
    engine_result: Any | None = None,
) -> tuple[list[Any], list[BandMeta | OutputMeta]]:
    """Flatten dict-form ``e_ops`` into ``(ops, meta)`` ready for the solver.

    Supported key shapes:

    * ``"device_label"`` (or a device object resolved to its label) with
      a single local operator — the operator is split into single-mode
      bands on that device.
    * ``("label_a", "label_b")`` with a tuple ``(op_a, op_b)`` — each
      operator is split into single-mode bands on its device and all
      band pairs are tensored, producing two-body ``(w_a, w_b)`` bands.
    * A string trace name with an ``OutputAmplitude``, ``OutputQuadrature``,
      or ``OutputPhotonFlux`` value — the required resolved SLH moments are
      lowered directly and reconstructed after the solve.

    For each band, the returned :class:`BandMeta` records the original
    key, the weight, and the sub-index (distinguishing entries when the
    user passed a list of operators for the same key). The matching
    ``ops`` list is ready for direct consumption by the backend solver.
    """
    if bases is None:
        bases = chip.resolve().bases

    flat_ops: list[Any] = []
    meta: list[BandMeta | OutputMeta] = []

    dims = chip.dims

    for raw_key, val in e_ops_dict.items():
        key = _resolve_eop_key(raw_key)

        if is_output_observable(val):
            if not isinstance(key, str):
                raise ValueError("Output observable trace names must be strings.")
            if engine_result is None:
                raise ValueError(
                    "Output observables require a resolved transient solve; "
                    "pass them to QuantumSequence.simulate(..., e_ops=...)."
                )
            _append_output_eops(
                flat_ops,
                meta,
                key=key,
                observable=val,
                engine_result=engine_result,
                backend=backend,
            )
            continue

        if isinstance(key, str):
            device_label = key
            if isinstance(val, tuple):
                raise ValueError(f"Single-device key {device_label!r} requires one operator, got tuple value")

            dev_idx = chip.device_index(device_label)
            dev = chip.device_map[device_label]
            basis = bases[device_label]
            dev_dim = basis.resolved_dim

            if isinstance(val, list):
                ops_with_idx: list[tuple[int | None, Any]] = [(i, op) for i, op in enumerate(val)]
            else:
                ops_with_idx = [(None, val)]

            for sub_idx, op in ops_with_idx:
                from quchip.chip.observables import prepare_local_op

                op = prepare_local_op(dev, op, basis, backend)
                for weight, embedded in embed_single_mode_bands(
                    backend,
                    op,
                    device_index=dev_idx,
                    dim=dev_dim,
                    label=device_label,
                    dims=dims,
                    semantic_to_solver=semantic_to_solver_transform(dev, basis),
                ):
                    flat_ops.append(embedded)
                    meta.append(
                        BandMeta(
                            key=device_label,
                            weight=weight,
                            device_labels=device_label,
                            sub_index=sub_idx,
                        )
                    )
            continue

        if isinstance(key, tuple) and len(key) == 2:
            label_a, label_b = key
            if not (isinstance(val, tuple) and len(val) == 2):
                raise ValueError(f"Two-device key {key!r} requires a tuple value (op_a, op_b)")

            op_a, op_b = val
            idx_a = chip.device_index(label_a)
            idx_b = chip.device_index(label_b)
            dev_a = chip.device_map[label_a]
            dev_b = chip.device_map[label_b]
            basis_a = bases[label_a]
            basis_b = bases[label_b]
            dim_a = basis_a.resolved_dim
            dim_b = basis_b.resolved_dim

            from quchip.chip.observables import prepare_local_op

            op_a = prepare_local_op(dev_a, op_a, basis_a, backend)
            op_b = prepare_local_op(dev_b, op_b, basis_b, backend)

            for w_a, band_a in local_mode_bands(
                backend,
                op_a,
                dim=dim_a,
                label=label_a,
                semantic_to_solver=semantic_to_solver_transform(dev_a, basis_a),
            ):
                for w_b, band_b in local_mode_bands(
                    backend,
                    op_b,
                    dim=dim_b,
                    label=label_b,
                    semantic_to_solver=semantic_to_solver_transform(dev_b, basis_b),
                ):
                    product = backend.tensor(band_a, band_b)
                    embedded = backend.embed_two_body(product, idx_a, idx_b, dims)
                    flat_ops.append(embedded)
                    meta.append(
                        BandMeta(
                            key=(label_a, label_b),
                            weight=(w_a, w_b),
                            device_labels=(label_a, label_b),
                        )
                    )
            continue

        raise ValueError("e_ops dict keys must be device labels (str) or 2-tuples of device labels")

    return flat_ops, meta


def _append_output_eops(
    flat_ops: list[Any],
    meta: list[BandMeta | OutputMeta],
    *,
    key: EOpKey,
    observable: OutputObservable,
    engine_result: Any,
    backend: Backend,
) -> None:
    """Lower one field recipe to the moments the backend must evaluate."""
    external = {channel.key: channel for channel in engine_result.slh.external_channels}
    channel = external.get(observable.exposure)
    if channel is None:
        raise ValueError(
            f"Unknown output exposure {observable.exposure!r}. "
            f"Available exposures: {list(external)}."
        )

    coupling = channel.coupling
    flat_ops.append(backend.from_canonical_operator(coupling))
    meta.append(OutputMeta(key=key, observable=observable, role="amplitude"))
    if not isinstance(observable, OutputPhotonFlux):
        return

    values = coupling.to_dense()
    xp = backend.array_module
    number = xp.conj(xp.swapaxes(values, -1, -2)) @ values
    from quchip.engine.ir import CanonicalOperator

    flat_ops.append(
        backend.from_canonical_operator(
            CanonicalOperator.from_dense(
                number,
                dims=coupling.dims,
                basis=coupling.basis,
                subsystem_labels=coupling.subsystem_labels,
                tag=f"output_flux:{observable.exposure}",
            )
        )
    )
    meta.append(OutputMeta(key=key, observable=observable, role="flux"))


def recombine_expect(
    flat_expect: Sequence[Any],
    meta_list: Sequence[BandMeta],
    tlist: Sequence[float] | Any,
    frame_freqs: Mapping[str, float],
    *,
    direction: str = "demodulate",
) -> tuple[dict[EOpKey, Any], dict[EOpKey, Any]]:
    """Recombine flattened expectations into ``(band_sum, phase_corrected)`` dicts.

    For each band with weight ``w`` on device(s) with demodulation
    frequency ``ω_demod``, the correction factor is

    .. math::
        \\exp\\!\\big(\\pm i \\cdot 2\\pi \\cdot \\omega_{\\text{demod}}
                              \\cdot w \\cdot t \\big),

    with the sign selected by *direction*: ``"demodulate"`` (``+1``)
    moves the expectation from the simulation's rotating frame back to
    the control frame; ``"remodulate"`` (``−1``) does the reverse.

    Returns two dicts: ``band_sum`` is the raw per-key sum over bands
    (no phase correction) and ``phase_corrected`` is the physically
    meaningful expectation in the control frame.
    """
    if direction not in ("demodulate", "remodulate"):
        raise ValueError(f"direction must be 'demodulate' or 'remodulate', got {direction!r}")
    if len(flat_expect) != len(meta_list):
        raise ValueError(
            f"flat_expect and meta_list must have the same length, got {len(flat_expect)} and {len(meta_list)}"
        )

    sign = +1 if direction == "demodulate" else -1
    sample = flat_expect[0] if flat_expect else tlist
    xp = _array_namespace(sample)
    times = xp.asarray(tlist, dtype=float)

    _AccKey = tuple[EOpKey, int | None]
    band_sum_acc: dict[_AccKey, Any] = {}
    corrected_acc: dict[_AccKey, Any] = {}
    _has_sub: dict[EOpKey, bool] = {}

    for series, meta in zip(flat_expect, meta_list):
        values = xp.asarray(series, dtype=complex)
        if values.shape != times.shape:
            raise ValueError(f"Expectation series shape {values.shape} does not match tlist shape {times.shape}")

        acc_key: _AccKey = (meta.key, meta.sub_index)
        if meta.sub_index is not None:
            _has_sub[meta.key] = True

        if acc_key not in band_sum_acc:
            band_sum_acc[acc_key] = xp.zeros_like(values, dtype=complex)
            corrected_acc[acc_key] = xp.zeros_like(values, dtype=complex)

        band_sum_acc[acc_key] += values

        if isinstance(meta.weight, int):
            if not isinstance(meta.device_labels, str):
                raise ValueError("Single-device weight requires string device label")
            omega_eff = frame_freqs[meta.device_labels] * meta.weight
        else:
            if not isinstance(meta.device_labels, tuple) or len(meta.device_labels) != 2:
                raise ValueError("Two-device weight requires 2-tuple device labels")
            label_a, label_b = meta.device_labels
            w_a, w_b = meta.weight
            omega_eff = frame_freqs[label_a] * w_a + frame_freqs[label_b] * w_b

        phase = xp.exp(sign * 1j * TWO_PI * omega_eff * times)
        corrected_acc[acc_key] += phase * values

    band_sum: dict[EOpKey, Any] = {}
    phase_corrected: dict[EOpKey, Any] = {}

    for (orig_key, sub_idx), bs_arr in band_sum_acc.items():
        pc_arr = corrected_acc[(orig_key, sub_idx)]
        if orig_key in _has_sub:
            if orig_key not in band_sum:
                band_sum[orig_key] = []
                phase_corrected[orig_key] = []
            band_sum[orig_key].append(bs_arr)
            phase_corrected[orig_key].append(pc_arr)
        else:
            band_sum[orig_key] = bs_arr
            phase_corrected[orig_key] = pc_arr

    return band_sum, phase_corrected


def build_observable_traces(
    solver_result: SolverResult,
    tlist: np.ndarray,
    chip: Chip,
    *,
    dict_meta: list[BandMeta | OutputMeta],
    resolved_frame: ResolvedFrame,
    engine_result: Any,
) -> dict[EOpKey, ObservableTrace | list[ObservableTrace]]:
    """Wrap solver expectations into phase-corrected :class:`ObservableTrace` objects.

    Pulls the flat per-band expectations out of *solver_result*, calls
    :func:`recombine_expect` with the demodulation frequencies from
    *resolved_frame*, and returns a dict keyed by the user's original
    e_ops keys. Each value is an :class:`ObservableTrace` (or a list
    when the user supplied multiple operators for the same key).
    """
    del chip  # reconstruction uses resolved solve metadata
    raw_expect = solver_result.expect
    if isinstance(raw_expect, dict):
        flat_expect: Sequence[Any] = list(raw_expect.values())
    else:
        flat_expect = raw_expect or []
    ordinary_expect = [
        values
        for values, meta in zip(flat_expect, dict_meta, strict=True)
        if isinstance(meta, BandMeta)
    ]
    ordinary_meta = [meta for meta in dict_meta if isinstance(meta, BandMeta)]
    band_sum, phase_corrected = recombine_expect(
        flat_expect=ordinary_expect,
        meta_list=ordinary_meta,
        tlist=tlist,
        frame_freqs=resolved_frame.demod_freqs,
        direction="demodulate",
    )
    traces: dict[EOpKey, ObservableTrace | list[ObservableTrace]] = {}
    for key, values in phase_corrected.items():
        raw = band_sum[key]
        if isinstance(values, list):
            traces[key] = [
                ObservableTrace(values=value, raw=raw_value)
                for value, raw_value in zip(values, raw)
            ]
        else:
            traces[key] = ObservableTrace(values=values, raw=raw)
    traces.update(
        _build_output_traces(
            flat_expect,
            dict_meta,
            tlist=tlist,
            engine_result=engine_result,
        )
    )
    return traces


def _build_output_traces(
    flat_expect: Sequence[Any],
    meta_list: Sequence[BandMeta | OutputMeta],
    *,
    tlist: Any,
    engine_result: Any,
) -> dict[EOpKey, ObservableTrace]:
    """Reconstruct ``b_out = S b_in + L`` moments at reference planes."""
    output_pairs = [
        (values, meta)
        for values, meta in zip(flat_expect, meta_list, strict=True)
        if isinstance(meta, OutputMeta)
    ]
    if not output_pairs:
        return {}

    sample = output_pairs[0][0]
    xp = _array_namespace(sample)
    times = xp.asarray(tlist, dtype=float)
    external = engine_result.slh.external_channels
    exposure_index = {channel.key: index for index, channel in enumerate(external)}
    incident = [xp.zeros_like(times, dtype=complex) for _ in external]
    from quchip.engine.ir import evaluate_signal_program

    for bound in engine_result.coherent_inputs:
        index = exposure_index.get(bound.exposure)
        if index is not None:
            incident[index] = incident[index] + evaluate_signal_program(
                bound.beta,
                times,
                xp=xp,
            )

    grouped: dict[EOpKey, dict[str, Any]] = {}
    specs: dict[EOpKey, OutputObservable] = {}
    for values, meta in output_pairs:
        grouped.setdefault(meta.key, {})[meta.role] = xp.asarray(values)
        specs[meta.key] = meta.observable

    traces: dict[EOpKey, ObservableTrace] = {}
    for key, moments in grouped.items():
        observable = specs[key]
        output_index = exposure_index[observable.exposure]
        channel = external[output_index]
        direct = xp.zeros_like(times, dtype=complex)
        for input_index, beta in enumerate(incident):
            direct = direct + xp.asarray(engine_result.slh.S[output_index, input_index]) * beta

        frame_frequency = (
            0.0
            if channel.collapse.frame_frequency is None
            else channel.collapse.frame_frequency
        )
        carrier = xp.exp(-1j * TWO_PI * xp.asarray(frame_frequency) * times)
        lowering = carrier * moments["amplitude"]
        field = direct + lowering

        if isinstance(observable, OutputAmplitude):
            boundary = field
        elif isinstance(observable, OutputQuadrature):
            boundary = xp.real(xp.exp(-1j * xp.asarray(observable.phase)) * field)
        else:
            number = xp.real(moments["flux"])
            boundary = (
                xp.abs(direct) ** 2
                + 2.0 * xp.real(xp.conj(direct) * lowering)
                + number
            )
        values = _delay_trace(boundary, times, channel.reference_delay, xp)
        traces[key] = ObservableTrace(values=values, raw=boundary)
    return traces


def _delay_trace(values: Any, times: Any, delay: Any, xp: Any) -> Any:
    """Propagate one boundary trace to its reciprocal external reference plane."""
    from quchip.utils.jax_utils import maybe_concrete_scalar

    concrete = maybe_concrete_scalar(delay)
    if concrete is not None and concrete == 0.0:
        return values
    query = times - xp.asarray(delay)
    real = xp.interp(query, times, xp.real(values), left=0.0, right=0.0)
    if not xp.iscomplexobj(values):
        return real
    imag = xp.interp(query, times, xp.imag(values), left=0.0, right=0.0)
    return real + 1j * imag
