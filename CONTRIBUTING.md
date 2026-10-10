# Contributing to quchip

Contributions from scientists and developers are welcome, including bug reports, physics discrepancies, documentation corrections, and code changes. Open a GitHub Issue for a concrete problem or model request, and use GitHub Discussions for open-ended questions.

## Development setup

quchip requires Python 3.11 or newer.

```bash
git clone https://github.com/quchip/quchip.git
cd quchip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,test,dynamiqs]'
```

QuTiP is the default backend. The `dynamiqs` extra lets you run the JAX-native solver, gradient, and batching tests.

## Tests

Each PR that adds a feature must include new or extended `unit` and/or `e2e`
tests for the added behavior. A PR that adds physics must also add or extend
tests in `validation`. These tests must use an independent numerical
reference, an analytic limit, or a convergence check suited to the claim. Keep
a cheap, representative physics check in the daily suites. Assert physical
quantities and meaningful tolerances. Execution, output shape, or a finite
gradient alone does not show correctness.

During development, use the two daily suites:

```bash
python -m pytest -m unit
python -m pytest -m e2e
```

`unit` checks small deterministic components and API contracts. `e2e` covers
integration workflows and physics, including every physics sentinel. On a
developer machine, aim for under 30 seconds for units and 3–5 minutes for
E2E, as runtime targets rather than guarantees across machines or optional
dependencies. To inspect regressions, use `--durations=20`. To limit runaway
checks, CI stops the unit step after two minutes and the E2E step after ten
minutes.

Expensive gradients, convergence studies, full guide replays, and exhaustive
checks stay in `validation`. Run them separately, or run all tests:

```bash
python -m pytest -m validation --durations=20
python -m pytest
```

Validation runs in one pytest process, even in pre-merge CI. Measure
changes with this sequential command, because parallel wall time does not
show that a suite is cheaper. Five minutes is the target, not a measured CI
guarantee. When you decrease repeated setup or solver work, keep
independent numerical references, failure paths, and statistical sample
sizes.

Each collected test belongs to exactly one of these suites. Mark small test modules with
`pytestmark = pytest.mark.unit` and expensive functions with `@pytest.mark.validation`. For an
expensive parameter value, use `pytest.param(value, marks=pytest.mark.validation)`. Unmarked tests
go into `e2e`, so new tests run in PR CI by default. Conflicting suite markers cause collection to
fail. Keep meaningful cheap checks. Select validation candidates by cost and coverage overlap, but
keep representative numerical oracles in E2E. Coverage overlap alone does not show that two tests
check the same assertion.

The location markers `core`, `physics_sentinel`, `extended`, and `examples`
stay available for focused work. They carry no runtime budget.

### CI and merging

| Event | Checks |
| --- | --- |
| PR opened or updated | Lint/types, then separate `unit` and `e2e` runs on Python 3.11 and 3.12 |
| PR marked ready for review, or updated while ready | `validation` on Python 3.11 |
| Push to `main` | Relevant docs build/deployment; no repeated test or benchmark run |
| Manual pre-merge run | `validation` on Python 3.11 |
| Manual benchmark run | Full benchmark ladder against the selected comparison ref |
| Weekly dependency canary | `unit` and `e2e` on Python 3.11 and 3.12 with fresh dependencies |

Open new PRs as drafts (`gh pr create --draft`). Drafts run the fast checks,
and marking a PR ready for review starts validation. While the PR is ready,
each new push reruns validation for the updated commit. If you change the PR
back to draft, a running validation workflow stops.

Together, the required fast and pre-merge checks cover the full suite, and they
do not repeat the fast tests on Python 3.11. Documentation, workflow changes, and
release metadata use lightweight validation. A version-only edit to
`quchip/__init__.py` is release metadata. Other source changes, tests, executable
examples (including Markdown), tooling, and repository rules must use test lanes.
If the change list is unreadable, classification fails.

The required `pre-merge full suite` check skips validation for drafts and
lightweight changes. It keeps its name for branch protection. Manual dispatch
always runs validation on the selected ref, but scheduled dependency checks
do not.

The `main` ruleset requires the PR branch to include the current `main`
before merging. Its configuration is recorded in
[`.github/rulesets/main.json`](.github/rulesets/main.json). A change to this
file does not update GitHub automatically. An administrator must apply it to
ruleset `21738854` before merging the workflow changes that remove
post-merge test runs. Existing PRs keep the required check names.

Tests that need dynamiqs use the `optional_backend` marker and call `pytest.importorskip("dynamiqs")`, so they skip cleanly when dynamiqs is not available.

Test public behavior and physical invariants, not implementation details. A behavior-preserving refactor should not require mechanical test edits. See [Change-Detector Tests Considered Harmful](https://testing.googleblog.com/2015/01/testing-on-toilet-change-detector-tests.html).

Before opening a pull request, run:

```bash
python -m pytest -m unit
python -m pytest -m e2e
ruff check .
python -m mypy quchip tests/typing/external_declarative_models.py
```

During physics development, run new or changed validation tests locally.
When the PR is ready to merge, run the complete validation suite by marking
it ready for review or via a manual pre-merge run, as described above.

Ruff uses a 120-character line limit. Public API docstrings use NumPy-style sections and imperative summaries that end with periods. Each test has a one-line docstring that states the invariant under test.

Document every parameter of public constructors, functions, and methods,
including keyword-only and inherited arguments. Where applicable, give accepted
forms, defaults, units, allowed choices or ranges, and the meaning of `None`.
Expand forwarded options or link to their precise contract. Keep descriptions
short. Put equations and derivations in `Notes`, with primary physics
references. Describe result attributes and return values with shapes, axis
order, and units.

With the docs extra installed, run `python tools/check_api_docs.py`. It finds
missing descriptions and obsolete parameter names on package exports and on
the returned types it lists, and examines structure only. Review numerical
conventions and references against the implementation, then inspect rendered
API pages. A successful Sphinx build alone does not show completeness. The
check includes the dynamiqs backend when that extra is installed. The docs job
installs `.[docs,dynamiqs]`, so it inspects both backend implementations and
can build the complete reference with warnings treated as errors.

## Examples and notebooks

The [guides](docs/guides/index.md) and [cookbook](docs/cookbook.md) explain how to use quchip, including model construction, parameter changes, sweeps and result interpretation. Use the same workflows in contributed examples. [Writing examples](docs/contribute/writing-examples.md) covers organization, figures and numerical checks.

Readable Jupytext Markdown is the canonical source. Commit its executed `.ipynb` partner with the `python3` kernel, identical code cells, and inspected outputs. Use standard fenced Python cells, not percent-format scripts.

Notebook outputs stay in the executed `.ipynb`. Save only figures selected for the website under `docs/images/`. Include the canonical Markdown from its page under `docs/guides/`. Figures load `docs/_static/quchip.mplstyle` and save a single light `.svg`. Then `python3 tools/theme_figures.py` derives the `-dark.svg`, `.pdf`, and README `-dark.png` variants and rejects colors outside the palette.

From the repository root, replace `<name>` with the example stem and run:

```bash
jupytext --sync examples/<name>.md
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 \
  --ExecutePreprocessor.record_timing=False \
  examples/<name>.ipynb
jupytext --diff --diff-format md examples/<name>.ipynb examples/<name>.md
jupytext --to md --test-strict examples/<name>.ipynb
```

The diff must be empty and the strict round trip must pass. Before you commit, inspect the executed notebook. Setup cells stay silent, stdout contains only intended receipts, and each intended figure appears once. Run the focused example test and the project checks above.

## Physics conventions

Before you change Hamiltonians, frames, approximations, observables, dissipation, or solver assembly, read [PHYSICS.md](PHYSICS.md).

- Use ordinary GHz for frequencies and energies, ns for time, and mK for temperature.
- Keep traced paths compatible with JAX. Do not call `float()`, `int()`, or `bool()` on traced values, and do not branch on traced values with Python. Use `jax.numpy` and JAX control flow.
- State approximations, omitted terms, and validity regimes through the model's declared approximation and `physics_notes()`.
- Use derived tolerances based on an analytic limit, truncation error, solver convergence, an independent calculation, or a cited reference. State the basis next to the assertion. Never tune a tolerance only to make a test pass.

## Pull requests

Keep pull requests small and focused. Explain the physical or user-visible change, include tests for changed behavior, and update the applicable documentation.

Before you start a large change or add a device, coupling, drive, envelope, or noise model, open an issue. Include the model Hamiltonian, assumptions, intended use, and a reference when available. This lets contributors agree on the scope before implementation.

## Releases

Keep release notes in `CHANGELOG.md` under `## [version] - YYYY-MM-DD`.
Each entry has two parts:

1. `### 0.x series highlights`: a short cumulative overview of the main
   capabilities the feature series introduced. While quchip is pre-1.0, a new
   series starts at `0.x.0` (for example, 0.4.0). Copy the overview into later
   patch releases. Update it only for substantial additions. Describe these as
   series capabilities so that older features do not appear new in the current
   patch.
2. `### Changes since <previous release>`: changes since the immediately
   preceding published tag, with a full comparison link. Use the following
   `####` categories in this order, and omit empty categories: Changes and
   migration, New features, Fixes, Performance, Compatibility, Documentation and
   examples, Development. At the first release of a series, the overview
   summarizes the features that this part gives in detail. Later patches keep
   the overview and list only their own changes here.

Write one short bullet for each user-visible outcome, with a PR link. Combine
related implementation changes. Start the series overview with new scientific
capabilities, such as solvers and device models. Where applicable, state
supported equations, backends, and experimental limits. Keep internal
refactors and CI details brief in Development. Performance claims require
measured evidence.

Changes to defaults, approximations, units, signs, initial states, or tolerances
must explain the old and new behavior and each necessary user action, even when
signatures do not change. Intentional changes to documented behavior go in
Changes and migration. Fixes describe the affected case and the correction, and
flag numerical consequences. Deprecations identify the replacement and the
planned removal release. Keep migration steps in the release entry, unless a
longer guide is necessary.

Before you tag, review the full change set since the previous published tag.
Move `Unreleased` into the dated entry, and run
`python tools/release_notes.py <version>` to inspect the release body. The docs
include the changelog. After the tag workflow publishes to PyPI, it uses that
same section for GitHub. Do not keep separate copies.

For editorial corrections after publication, merge the changelog update and
regenerate the GitHub release body with the same command. Then verify both the
published release and the deployed docs. Do not change the published tag and
package. Code changes require a new release.

## Policies

- There is no CLA or DCO. When you open a pull request, you agree that your contribution is supplied under the project's license.
- If you used AI assistance, disclose it in the pull request description. The contributor stays accountable for every claim and change.
- Be respectful and constructive in issues, discussions, and reviews. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
