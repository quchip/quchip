"""Validate numerical outcomes at the Dynamiqs integration boundary.

Dynamiqs 0.3.6 discards this status after integration. Keep the adapter local:
native integrators still own equations, saving, controllers and adjoints.
"""

from __future__ import annotations

import warnings
from typing import Any

import diffrax as dx
import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from quchip.backend.containers import BatchSolveError
from quchip.utils.jax_utils import contains_tracer


def solve_with_status(H: Any, jumps: list[Any], state: Any, times: Any,
                      observables: Any, *, solver: str, method: Any, gradient: Any = None,
                      **options: Any) -> tuple[Any, Any]:
    """Run one native lane, retaining status for Diffrax-backed methods.

    ``options`` are the native solver keyword arguments (``save_states``, ...).
    """
    ordinary = (dq.method.Euler, dq.method.Dopri5, dq.method.Dopri8,
                dq.method.Tsit5, dq.method.Kvaerno3, dq.method.Kvaerno5)
    rouchon = (dq.method.Rouchon1, dq.method.Rouchon2, dq.method.Rouchon3)
    if type(method) not in ordinary + (rouchon if solver == "mesolve" else ()):
        kwargs = dict(exp_ops=observables, method=method, gradient=gradient, **options)
        result = (dq.mesolve(H, jumps, state, times, **kwargs) if solver == "mesolve"
                  else dq.sesolve(H, state, times, **kwargs))
        return result, expm_status(result)

    try:
        from dynamiqs._checks import check_times
        from dynamiqs.integrators.apis.mesolve import _check_mesolve_args
        from dynamiqs.integrators.apis.sesolve import _check_sesolve_args
        from dynamiqs.integrators.core import diffrax_integrator, rouchon_integrator
        module = rouchon_integrator if type(method) in rouchon else diffrax_integrator
        constructor = getattr(module, f"{solver}_{type(method).__name__.lower()}_integrator_constructor")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Native batch status requires a compatible dynamiqs integrator interface.") from exc

    H = H if isinstance(H, dq.TimeQArray) else dq.constant(H)
    observables = observables or None
    extra = {}
    if solver == "mesolve":
        jumps = [dq.constant(jump) for jump in jumps]
        _check_mesolve_args(H, jumps, state, observables)
        extra["Ls"] = jumps
    else:
        _check_sesolve_args(H, state, observables)
        extra["result_class"] = dq.SESolveResult  # master-equation constructors bind their own
    method.assert_supports_gradient(gradient)
    integrator = constructor(
        H=H, y0=state, ts=check_times(times, "tsave"), Es=observables,
        method=method, gradient=gradient, options=dq.Options(**options).initialise(), **extra,
    )
    # Newer dynamiqs passes time to save callbacks and restores native trees
    # from its device-specific buffers. Keep preparation and restoration paired.
    prepare_save = getattr(diffrax_integrator, "_saveat", None)
    if prepare_save is not None:
        saveat, restore = prepare_save(integrator.ts, integrator.save, integrator.y0)
    else:
        saveat = dx.SaveAt(subs=[
            dx.SubSaveAt(ts=integrator.ts, fn=lambda t, y, args: integrator.save(y)),
            dx.SubSaveAt(t1=True),
        ])
        restore = None
    progress_meter = integrator.options.progress_meter
    convert_progress = getattr(diffrax_integrator, "progress_meter_to_diffrax", None)
    progress_meter = (convert_progress(progress_meter) if convert_progress is not None
                      else progress_meter.to_diffrax())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Native complex-integration warning policy.
        solution = dx.diffeqsolve(
            integrator.terms, integrator.diffrax_solver,
            t0=integrator.t0, t1=integrator.t1, dt0=integrator.dt0, y0=integrator.y0,
            saveat=saveat, stepsize_controller=integrator.stepsize_controller,
            adjoint=integrator.adjoint, max_steps=integrator.max_steps,
            progress_meter=progress_meter, throw=False,
        )
    saved_values = restore(solution.ys) if restore is not None else solution.ys
    saved = integrator.postprocess_saved(*saved_values)
    return integrator.result(saved, infos=integrator.infos(solution.stats)), solution.result


def expm_status(result: Any) -> Any:
    """Expm has no termination status and can return NaNs on overflow."""
    finite = jnp.all(jnp.isfinite(result.states.to_jax()))
    if result.expects is not None:
        finite = finite & jnp.all(jnp.isfinite(result.expects))
    return dx.RESULTS.where(finite, dx.RESULTS.successful, dx.RESULTS.nonfinite)


def require_success(result: Any, status: Any, parameters: Any, *, traced_context: Any = (),
                    single: bool = False) -> Any:
    """Reject partial native solutions, retaining numerical context under JIT."""
    def check(code, original_index, values):
        if bool(np.asarray(code != dx.RESULTS.successful)):
            if single:
                raise RuntimeError(f"Integration failed: {dx.RESULTS[code]}")
            params = jax.tree.map(lambda value: np.asarray(value).item() if eqx.is_array(value)
                                  and value.ndim == 0 else value, values)
            raise BatchSolveError(int(original_index), dx.RESULTS[code], params)
        return np.asarray(False)

    context = tuple(enumerate(parameters))
    if not contains_tracer(status):
        failed = np.flatnonzero(np.asarray(status != dx.RESULTS.successful))
        if len(failed):
            concrete_index = int(failed[0])
            check(jax.tree.map(lambda leaf: leaf[concrete_index], status), *context[concrete_index])
        return result
    context = traced_context or context
    index = jnp.argmax(status != dx.RESULTS.successful)
    # Select one lane on device; do not transfer a scalar buffer per sweep point.
    original_index = jnp.asarray([point for point, _ in context])[index]
    values = jax.tree.map(lambda *leaves: jnp.stack(leaves)[index],
                          *(values for _, values in context))
    code = jax.tree.map(lambda leaf: leaf[index], status)
    stopped = jax.tree.map(lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value)
                           else value, (code, original_index, values))
    dynamic, static = eqx.partition(stopped, eqx.is_array)
    # The effect must survive gradient-only consumers. A pure error guard's
    # checked primal can be removed when differentiation returns only its tangent.
    failed = jax.experimental.io_callback(
        lambda values: check(*eqx.combine(values, static)),
        jax.ShapeDtypeStruct((), jnp.bool_), dynamic, ordered=False,
    )
    return eqx.error_if(result, failed, "Batch integration failed.")
