# Dressed-Chip Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `fit_a_dress(desired)` compile component-owned numerical dressed constraints without evaluating `desired`, while retaining explicit legacy behavior during migration.

**Architecture:** Devices and couplings expose small inverse-design policy hooks. Constraint compilation produces normalized `TargetSpec` records and a conservative free-parameter selection; the existing SciPy/JAX solver consumes that plan. New `constraints`, `vary`, and `start` inputs coexist with legacy keywords without changing their meaning.

**Tech Stack:** Python, dataclasses, NumPy, SciPy least-squares, JAX, pytest, Ruff, Sphinx.

---

## Task 1: Component-owned defaults

**Files:**
- Modify: `quchip/devices/base.py`
- Modify: `quchip/devices/transmon/duffing.py`
- Modify: `quchip/devices/transmon/flux_tunable.py`
- Modify: `quchip/devices/resonator.py`
- Modify: `quchip/chip/coupling_base.py`
- Modify: `quchip/chip/couplings.py`
- Test: `tests/test_fit_a_dress.py`

- [x] Write tests proving that supported devices expose numeric dressed target declarations and that capacitive couplings declare `cross_kerr` as their default.
- [x] Run the focused tests and confirm they fail because the policy hooks do not exist.
- [x] Add minimal class-owned hooks returning target-field mappings, default free parameters, and the coupling observable.
- [x] Run the focused tests and confirm they pass.

## Task 2: Desired-chip constraint compilation

**Files:**
- Modify: `quchip/inverse_design/observables.py`
- Modify: `quchip/inverse_design/types.py`
- Test: `tests/test_fit_a_dress.py`

- [x] Write tests proving compilation reads declared numbers without calling `Chip.freq`, `dressed_anharmonicity`, or `static_zz` on the desired chip.
- [x] Write tests for additive near-edge and far-edge `exchange_rate`/`cross_kerr` constraints, replacement, and removal.
- [x] Run the focused tests and confirm the new compiler is absent.
- [x] Implement canonical observable normalization and desired-chip target compilation.
- [x] Run the focused tests and confirm they pass.

## Task 3: Solver integration and manual controls

**Files:**
- Modify: `quchip/inverse_design/fit.py`
- Modify: `quchip/inverse_design/types.py`
- Test: `tests/test_fit_a_dress.py`

- [x] Write an end-to-end test in which a capacitive edge's declared scalar is fitted as a full cross-Kerr target and the output bare `g` differs from that target.
- [x] Write tests for `vary`, `start`, default sign selection, and legacy keyword compatibility.
- [x] Run the focused tests and confirm the public behavior fails.
- [x] Integrate the desired-chip compiler with the solver while leaving the legacy path intact.
- [x] Run all inverse-design tests and fix only behavior required by the design.

## Task 4: Receipts and documentation

**Files:**
- Modify: `quchip/inverse_design/types.py`
- Modify: `docs/guides/defining-and-inspecting-a-chip.md`
- Modify: `docs/cookbook.md`
- Modify: `examples/02_reduce_and_replay.md`
- Modify: `examples/02_reduce_and_replay.ipynb`
- Test: `tests/examples/test_progressive_guides.py`

- [x] Add structured source/convention metadata and a concise result summary.
- [x] Replace the existing manual fit example with the desired-chip-first form and show the numerical target-to-bare-parameter distinction.
- [x] Re-execute paired examples and synchronize Markdown output.
- [x] Run focused tests, Ruff, output synchronization, Sphinx, prose audit, and `git diff --check`.
