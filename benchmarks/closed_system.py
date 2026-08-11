"""Compare quchip with direct QuTiP and dynamiqs closed-system solves."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quchip-matplotlib"))

LEVELS = 3
BASE_FREQ = 5.0
FREQ_STEP = 0.1
ANHARMONICITY = -0.25
COUPLING_G = 0.01
DRIVE_AMP = 0.02
DRIVE_SIGMAS = 3.0
PULSE_DURATION = 40.0
N_TIMES = 401
ATOL = 1e-8
RTOL = 1e-6
MAX_STEPS = 1_000_000
PARITY_TOL = 1e-5
TWO_PI = 2.0 * np.pi


def frequencies(n: int) -> list[float]:
    """Return the frequency ladder used by every implementation."""
    return [BASE_FREQ + FREQ_STEP * index for index in range(n)]


def time_grid() -> np.ndarray:
    """Return the shared pulse sampling grid."""
    return np.linspace(0.0, PULSE_DURATION, N_TIMES)


def gaussian_envelope(t: Any, *, xp: Any = np) -> Any:
    """Evaluate the shared Gaussian control envelope."""
    center = PULSE_DURATION / 2.0
    sigma = PULSE_DURATION / (2.0 * DRIVE_SIGMAS)
    return xp.exp(-((t - center) ** 2) / (2.0 * sigma**2))


def _timed(call: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    result = call()
    return time.perf_counter() - start, result


def _qutip_site(op: Any, index: int, n: int, levels: int) -> Any:
    import qutip as qt

    operators = [qt.qeye(levels)] * n
    operators[index] = op
    return qt.tensor(operators)


def _build_native_qutip(n: int, levels: int) -> dict[str, Any]:
    import qutip as qt

    lowering = qt.destroy(levels)
    number = qt.num(levels)
    a_ops = [_qutip_site(lowering, index, n, levels) for index in range(n)]
    n_ops = [_qutip_site(number, index, n, levels) for index in range(n)]
    hamiltonian: list[Any] = [
        sum(
            (TWO_PI * (ANHARMONICITY / 2.0) * (op * op - op) for op in n_ops),
            0 * n_ops[0],
        )
    ]
    angular_detuning = TWO_PI * FREQ_STEP
    minus = qt.coefficient(lambda t: np.exp(-1j * angular_detuning * t))
    plus = qt.coefficient(lambda t: np.exp(1j * angular_detuning * t))
    envelope = qt.coefficient(lambda t: gaussian_envelope(t))
    for index in range(n - 1):
        exchange = TWO_PI * COUPLING_G * (a_ops[index].dag() * a_ops[index + 1])
        hamiltonian.extend(([exchange, minus], [exchange.dag(), plus]))
    drive_op = TWO_PI * (DRIVE_AMP / 2.0) * (1j * (a_ops[0] - a_ops[0].dag()))
    hamiltonian.append([drive_op, envelope])
    options = {"atol": ATOL, "rtol": RTOL, "nsteps": MAX_STEPS, "store_states": False}
    return {
        "runner": qt.SESolver(qt.QobjEvo(hamiltonian), options=options),
        "state": qt.tensor([qt.basis(levels, 0)] * n),
        "observables": n_ops,
    }


def _solve_native_qutip(model: dict[str, Any]) -> np.ndarray:
    result = model["runner"].run(model["state"], time_grid(), e_ops=model["observables"])
    return np.real(np.asarray(result.expect))


def _dynamiqs_site(op: Any, index: int, n: int, levels: int) -> Any:
    import dynamiqs as dq

    return dq.tensor(*[op if site == index else dq.eye(levels) for site in range(n)])


def _build_native_dynamiqs(n: int, levels: int) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    import dynamiqs as dq
    import jax.numpy as jnp

    lowering = dq.destroy(levels)
    number = dq.number(levels)
    a_ops = [_dynamiqs_site(lowering, index, n, levels) for index in range(n)]
    n_ops = [_dynamiqs_site(number, index, n, levels) for index in range(n)]
    state = dq.tensor(*[dq.fock(levels, 0) for _ in range(n)])
    method = dq.method.Tsit5(rtol=RTOL, atol=ATOL, max_steps=MAX_STEPS)
    options = dq.Options(save_states=False, progress_meter=False)

    @jax.jit
    def solve(amplitude: Any) -> Any:
        hamiltonian: Any = 0.0 * n_ops[0]
        for op in n_ops:
            hamiltonian = hamiltonian + TWO_PI * (ANHARMONICITY / 2.0) * (op @ op - op)
        angular_detuning = TWO_PI * FREQ_STEP
        for index in range(n - 1):
            exchange = TWO_PI * COUPLING_G * (dq.dag(a_ops[index]) @ a_ops[index + 1])
            hamiltonian = hamiltonian + dq.modulated(
                lambda t, w=angular_detuning: jnp.exp(-1j * w * t), exchange
            )
            hamiltonian = hamiltonian + dq.modulated(
                lambda t, w=angular_detuning: jnp.exp(1j * w * t), dq.dag(exchange)
            )
        drive_op = 1j * (a_ops[0] - dq.dag(a_ops[0]))
        hamiltonian = hamiltonian + dq.modulated(
            lambda t: (TWO_PI * amplitude / 2.0) * gaussian_envelope(t, xp=jnp) + 0.0j,
            drive_op,
        )
        result = dq.sesolve(
            hamiltonian,
            state,
            time_grid(),
            exp_ops=n_ops,
            method=method,
            options=options,
        )
        return result.expects

    return {"solve": solve}


def _solve_native_dynamiqs(model: dict[str, Any]) -> np.ndarray:
    return np.real(np.asarray(model["solve"](DRIVE_AMP)))


def _build_quchip(n: int, levels: int, family: str) -> tuple[Any, list[str]]:
    import quchip
    from quchip import Capacitive, ChargeDrive, Chip, DuffingTransmon, Gaussian, QuantumSequence
    from quchip.engine import solve_problem

    del quchip, solve_problem
    freqs = frequencies(n)
    devices: list[Any] = [
        DuffingTransmon(
            freq=freq,
            anharmonicity=ANHARMONICITY,
            levels=levels,
            label=f"q{index}",
        )
        for index, freq in enumerate(freqs)
    ]
    for device, freq in zip(devices, freqs):
        device.reference_freq = freq
    couplings: list[Any] = [
        Capacitive(devices[index], devices[index + 1], g=COUPLING_G) for index in range(n - 1)
    ]
    chip = Chip(
        devices,
        couplings or None,
        frame={device: freq for device, freq in zip(devices, freqs)},
        rwa=True,
        backend=family,
    )
    drive = ChargeDrive(devices[0], label="d0")
    chip.wire(drive)
    sequence = QuantumSequence(chip)
    sequence.schedule(
        drive,
        envelope=Gaussian(duration=PULSE_DURATION, sigmas=DRIVE_SIGMAS, amplitude=DRIVE_AMP),
        freq=freqs[0],
    )
    if family == "qutip":
        options: dict[str, Any] = {"atol": ATOL, "rtol": RTOL, "nsteps": MAX_STEPS, "store_states": False}
    else:
        import dynamiqs as dq

        options = {
            "method": dq.method.Tsit5(rtol=RTOL, atol=ATOL, max_steps=MAX_STEPS),
            "store_states": False,
        }
    problem = sequence.build_problem(
        tlist=time_grid(),
        e_ops=chip.e_ops(**{device.label: "n" for device in devices}),  # type: ignore[arg-type]
        initial_state=chip.bare_state(**{device.label: 0 for device in devices}),
        options=options,
    )
    return problem, [device.label for device in devices]


def _solve_quchip(model: tuple[Any, list[str]]) -> np.ndarray:
    from quchip.engine import solve_problem

    problem, labels = model
    result = solve_problem(problem, check_truncation=False)
    return np.stack([np.real(np.asarray(result.expect(label))) for label in labels])


def _worker(args: argparse.Namespace) -> None:
    row: dict[str, Any] = {
        "N": args.n,
        "levels": args.levels,
        "dim": args.levels**args.n,
        "family": args.family,
        "path": args.path,
    }
    try:
        build_call: Callable[[], Any]
        solve: Callable[[Any], np.ndarray]
        if args.family == "qutip":
            import qutip

            del qutip
        else:
            import dynamiqs
            import jax

            del dynamiqs, jax
        if args.path == "native":
            if args.family == "qutip":
                build, solve = _build_native_qutip, _solve_native_qutip
            else:
                build, solve = _build_native_dynamiqs, _solve_native_dynamiqs

            def build_call() -> Any:
                return build(args.n, args.levels)

        else:
            import quchip

            imported = Path(quchip.__file__).resolve()
            expected = Path(args.source_root).resolve()
            if not imported.is_relative_to(expected):
                raise RuntimeError(f"loaded quchip from {imported}, expected it under {expected}")

            def build_call() -> Any:
                return _build_quchip(args.n, args.levels, args.family)

            solve = _solve_quchip

        cold_build_s, model = _timed(build_call)
        first_solve_s, traces = _timed(lambda: solve(model))
        for _ in range(args.warmup):
            solve(model)
        warm_samples = [_timed(lambda: solve(model))[0] for _ in range(args.solve_repeat)]
        build_samples = [_timed(build_call)[0] for _ in range(args.build_repeat)]
        np.save(args.trace_out, traces)
        row.update(
            cold_build_s=cold_build_s,
            build_s=float(np.median(build_samples)),
            build_samples_s=build_samples,
            first_solve_s=first_solve_s,
            warm_solve_s=float(np.median(warm_samples)),
            warm_samples_s=warm_samples,
        )
        if args.reference:
            reference = np.load(args.reference)
            parity = float(np.max(np.abs(traces - reference)))
            row.update(parity=parity, status="ok" if parity < PARITY_TOL else "parity-fail")
        else:
            row.update(parity=0.0, status="ok")
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    Path(args.row_out).write_text(json.dumps(row, indent=2) + "\n")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _provenance(repo: Path, main_ref: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "head_branch": _git(repo, "branch", "--show-current") or "detached",
        "head_commit": _git(repo, "rev-parse", "HEAD"),
        "head_dirty": bool(_git(repo, "status", "--porcelain", "--untracked-files=no")),
        "main_ref": main_ref,
        "main_commit": _git(repo, "rev-parse", main_ref),
        "qutip": _package_version("qutip"),
        "dynamiqs": _package_version("dynamiqs"),
        "jax": _package_version("jax"),
        "numpy": _package_version("numpy"),
        "build_repeat": args.build_repeat,
        "warmup": args.warmup,
        "solve_repeat": args.solve_repeat,
        "parity_tol": PARITY_TOL,
        "constants": {
            "levels": args.levels,
            "coupling_g": COUPLING_G,
            "drive_amp": DRIVE_AMP,
            "atol": ATOL,
            "rtol": RTOL,
            "n_times": N_TIMES,
        },
    }


def _extract_ref(repo: Path, ref: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", ref], cwd=repo, check=True, stdout=subprocess.PIPE
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        handle.extractall(destination)


def _invoke_worker(
    *,
    path_name: str,
    family: str,
    n: int,
    args: argparse.Namespace,
    source_root: Path | None,
    reference: Path | None,
    temp_dir: Path,
) -> tuple[dict[str, Any], Path]:
    stem = f"{n}-{family}-{path_name}"
    row_path = temp_dir / f"{stem}.json"
    trace_path = temp_dir / f"{stem}.npy"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--path",
        path_name,
        "--family",
        family,
        "--n",
        str(n),
        "--levels",
        str(args.levels),
        "--build-repeat",
        str(args.build_repeat),
        "--warmup",
        str(args.warmup),
        "--solve-repeat",
        str(args.solve_repeat),
        "--row-out",
        str(row_path),
        "--trace-out",
        str(trace_path),
    ]
    if source_root is not None:
        command.extend(("--source-root", str(source_root)))
    if reference is not None:
        command.extend(("--reference", str(reference)))
    environment = dict(os.environ)
    if source_root is not None:
        environment["PYTHONPATH"] = str(source_root)
    subprocess.run(command, check=True, env=environment)
    return json.loads(row_path.read_text()), trace_path


def _parse_rungs(value: str) -> list[int]:
    rungs = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not rungs or any(n < 1 for n in rungs) or len(rungs) != len(set(rungs)):
        raise ValueError("--rungs must contain unique positive comma-separated integers")
    return rungs


def _run(args: argparse.Namespace) -> None:
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    provenance = _provenance(repo, args.main_ref, args)
    if provenance["head_dirty"] and not args.allow_dirty:
        raise SystemExit("current checkout is dirty; commit it first or pass --allow-dirty")
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def persist(complete: bool) -> None:
        output.write_text(
            json.dumps(
                {"schema_version": 2, "provenance": provenance, "complete": complete, "rows": rows},
                indent=2,
            )
            + "\n"
        )

    with tempfile.TemporaryDirectory(prefix="quchip-main-") as main_temp, tempfile.TemporaryDirectory(
        prefix="quchip-benchmark-"
    ) as cell_temp:
        main_root = Path(main_temp)
        cell_root = Path(cell_temp)
        _extract_ref(repo, args.main_ref, main_root)
        for rung_index, n in enumerate(_parse_rungs(args.rungs)):
            qutip_native, reference_trace = _invoke_worker(
                path_name="native",
                family="qutip",
                n=n,
                args=args,
                source_root=None,
                reference=None,
                temp_dir=cell_root,
            )
            rows.append(qutip_native)
            persist(False)
            print(f"[dim={args.levels**n} qutip/native] {qutip_native['status']}", flush=True)
            reference = reference_trace if reference_trace.exists() else None
            revisions: tuple[tuple[str, Path], ...] = (("head", repo), ("main", main_root))
            if rung_index % 2:
                revisions = tuple(reversed(revisions))
            for path_name, source_root in revisions:
                row, _ = _invoke_worker(
                    path_name=path_name,
                    family="qutip",
                    n=n,
                    args=args,
                    source_root=source_root,
                    reference=reference,
                    temp_dir=cell_root,
                )
                rows.append(row)
                persist(False)
                print(f"[dim={args.levels**n} qutip/{path_name}] {row['status']}", flush=True)

            dynamiqs_native, _ = _invoke_worker(
                path_name="native",
                family="dynamiqs",
                n=n,
                args=args,
                source_root=None,
                reference=reference,
                temp_dir=cell_root,
            )
            rows.append(dynamiqs_native)
            persist(False)
            print(f"[dim={args.levels**n} dynamiqs/native] {dynamiqs_native['status']}", flush=True)
            for path_name, source_root in revisions:
                row, _ = _invoke_worker(
                    path_name=path_name,
                    family="dynamiqs",
                    n=n,
                    args=args,
                    source_root=source_root,
                    reference=reference,
                    temp_dir=cell_root,
                )
                rows.append(row)
                persist(False)
                print(f"[dim={args.levels**n} dynamiqs/{path_name}] {row['status']}", flush=True)
    persist(True)
    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        cells = ", ".join(f"{row['family']}/{row['path']}/N={row['N']}" for row in failures)
        raise SystemExit(f"benchmark failed after writing {output}: {cells}")
    print(f"wrote {output}")


def _report(args: argparse.Namespace) -> None:
    from reporting import load_result, render_markdown, render_plots

    document = load_result(Path(args.data))
    summary = render_markdown(document)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary)
    outputs = render_plots(document, Path(args.out_dir))
    print(f"wrote {summary_path} and {len(outputs)} plots")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="record closed-system benchmark rows")
    run.add_argument("--main-ref", default="main")
    run.add_argument("--levels", type=int, default=LEVELS)
    run.add_argument("--rungs", default="1,3,5,7")
    run.add_argument("--build-repeat", type=int, default=3)
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--solve-repeat", type=int, default=3)
    run.add_argument("--out", default="benchmark-output/closed-system.json")
    run.add_argument("--allow-dirty", action="store_true")
    run.set_defaults(handler=_run)

    report = subparsers.add_parser("report", help="validate and render a recorded result")
    report.add_argument("--data", default="benchmark-output/closed-system.json")
    report.add_argument("--out-dir", default="benchmark-output")
    report.add_argument("--summary", default="benchmark-output/summary.md")
    report.set_defaults(handler=_report)

    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--path", choices=("head", "main", "native"), required=True)
    worker.add_argument("--family", choices=("qutip", "dynamiqs"), required=True)
    worker.add_argument("--n", type=int, required=True)
    worker.add_argument("--levels", type=int, required=True)
    worker.add_argument("--build-repeat", type=int, required=True)
    worker.add_argument("--warmup", type=int, required=True)
    worker.add_argument("--solve-repeat", type=int, required=True)
    worker.add_argument("--source-root")
    worker.add_argument("--reference")
    worker.add_argument("--row-out", required=True)
    worker.add_argument("--trace-out", required=True)
    worker.set_defaults(handler=_worker)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a benchmark command."""
    args = _parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
