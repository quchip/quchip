"""Check parameter descriptions on the supported user-facing Python surface.

Run with the docs extra installed. This checks structure, not scientific
accuracy: units, options, defaults, shapes, and references still need review.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
from collections.abc import Iterator
from importlib.util import find_spec
from typing import Any

from sphinx.ext.napoleon import Config
from sphinx.ext.napoleon.docstring import NumpyDocstring


# Package exports are authoritative. These additional types are returned by
# public workflows without being re-exported at the package root.
PUBLIC_MODULES = (
    "quchip", "quchip.analysis", "quchip.approximations", "quchip.backend",
    "quchip.chip", "quchip.control", "quchip.declarative", "quchip.devices",
    "quchip.engine", "quchip.extensions", "quchip.inverse_design",
    "quchip.interop", "quchip.results", "quchip.utils", "quchip.viz",
)
RETURNED_TYPES = (
    "quchip.sweep.SpectrumSweepResult",
    "quchip.control.batch.BatchAxis",
    "quchip.control.batch.ZippedBatchAxis",
    "quchip.control.batch.PulseHandle",
    "quchip.control.batch.DelayHandle",
    "quchip.control.field.CoherentInput",
    "quchip.chip.analysis.ChipAnalysis",
    "quchip.engine.excitation.ExcitationStructure",
    "quchip.engine.excitation.ExcitationTerm",
    "quchip.engine.routes.SolveRoute",
    "quchip.engine.ir.ResolvedFrame",
    "quchip.chip.port_network.FieldTerminal",
    "quchip.chip.port_network.SLHComponent",
    "quchip.chip.port_network.IncludedNetwork",
    "quchip.observables.OutputField",
    "quchip.backend.containers.LinearResponseSolverResult",
    "quchip.backend.containers.PreparedStationary",
)


def documented_fields(doc: str) -> tuple[set[str], set[str]]:
    """Read nonempty parameter and attribute descriptions using Sphinx's parser."""
    rendered = str(NumpyDocstring(doc, config=Config(napoleon_use_ivar=True)))
    params, attributes = set(), set()
    # Field bodies may wrap onto subsequent indented lines.
    pattern = r"^:(param|ivar) ([^:]+):([^\n]*(?:\n[ \t]+[^\n]*)*)"
    for kind, name, body in re.findall(pattern, rendered, flags=re.MULTILINE):
        if body.strip():
            target = params if kind == "param" else attributes
            target.update(part.strip().lstrip("\\*") for part in name.split(","))
    return params, attributes


def public_objects() -> Iterator[tuple[str, Any]]:
    """Yield exports and explicitly listed returned types, deduplicated by identity."""
    seen = set()
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            obj = getattr(module, name)
            if id(obj) not in seen:
                seen.add(id(obj))
                yield f"{module_name}.{name}", obj
    extra_types = list(RETURNED_TYPES)
    if find_spec("dynamiqs") is not None:
        extra_types.append("quchip.backend.dynamiqs.DynamiqsBackend")
    for path in extra_types:
        module_name, name = path.rsplit(".", 1)
        obj = getattr(importlib.import_module(module_name), name)
        if id(obj) not in seen:
            seen.add(id(obj))
            yield path, obj


def check_callable(name: str, obj: Any) -> dict[str, Any] | None:
    """Compare a callable's real signature with its authored descriptions."""
    if inspect.isclass(obj) and getattr(obj, "_is_protocol", False):
        return None  # Protocols describe structural interfaces, not constructors.
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    parameters = {
        key: value for key, value in signature.parameters.items()
        if key not in {"self", "cls"} and not key.startswith("_")
    }
    doc = inspect.getdoc(obj) or ""
    documented, attributes = documented_fields(doc)
    if inspect.isclass(obj):
        documented |= attributes
    missing = sorted(set(parameters) - documented)
    accepts_keywords = any(p.kind == p.VAR_KEYWORD for p in parameters.values())
    # Expanded keyword descriptions are valid when **kwargs accepts them.
    extra = sorted(documented - set(parameters)) if not accepts_keywords else []
    if inspect.isclass(obj):
        extra = sorted(set(extra) - attributes)
    if not missing and not extra:
        return None
    return {"name": name, "file": inspect.getsourcefile(obj), "missing": missing, "extra": extra}


def audit() -> tuple[int, list[dict[str, Any]]]:
    """Check public callables and inherited methods once per implementation."""
    issues = []
    checked = 0
    seen_methods = set()
    for name, obj in public_objects():
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        if not getattr(obj, "__module__", "").startswith("quchip"):
            continue  # Re-exported typing aliases do not define quchip callables.
        checked += 1
        issue = check_callable(name, obj)
        if issue:
            issues.append(issue)
        if not inspect.isclass(obj):
            continue
        for member_name, member in inspect.getmembers(obj, inspect.isroutine):
            if member_name.startswith("_") or not getattr(member, "__module__", "").startswith("quchip"):
                continue
            identity = (member.__module__, member.__qualname__)
            if identity in seen_methods:
                continue
            seen_methods.add(identity)
            checked += 1
            issue = check_callable(f"{name}.{member_name}", member)
            if issue:
                issues.append(issue)
    return checked, issues


def main() -> int:
    """Print missing or obsolete entries and fail when any remain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the complete report as JSON.")
    parser.add_argument("--path", help="Only report files containing this path fragment.")
    args = parser.parse_args()
    checked, issues = audit()
    if args.path:
        issues = [issue for issue in issues if args.path in (issue["file"] or "")]
    if args.json:
        print(json.dumps({"checked": checked, "issues": issues}, indent=2))
    else:
        for issue in issues:
            print(f"{issue['name']}: missing={issue['missing']}, obsolete={issue['extra']}")
        print(f"{checked} public callables inspected; {len(issues)} with parameter-documentation gaps.")
    return bool(issues)


if __name__ == "__main__":
    raise SystemExit(main())
