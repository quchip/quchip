# Contributing to quchip

Contributions from scientists and developers are welcome. Bug reports, physics discrepancies, documentation corrections, and code changes are all useful. Open a GitHub Issue for a concrete problem or model request. Use GitHub Discussions for open-ended questions.

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

QuTiP is the default backend. The `dynamiqs` extra enables JAX-native solver, gradient, and batching tests.

## Tests

Every PR adding a feature must include new or extended `unit` and/or `e2e`
tests for the added behavior. PRs adding physics must also add or extend tests
in `validation`, with an independent numerical reference, an analytic limit,
or a convergence check appropriate to the claim. Keep a representative cheap
physics check in the daily suites as well. Assert physical quantities and
meaningful tolerances; execution, output shape, or a finite gradient alone
does not establish correctness.

Use the two daily suites while developing:

```bash
python -m pytest -m unit
python -m pytest -m e2e
```

`unit` checks small deterministic components and API contracts. `e2e` covers
integration workflows and physics, including every physics sentinel. Aim for
under 30 seconds for units and 3–5 minutes for E2E on a developer machine;
use `--durations=20` to inspect regressions. These are runtime targets, not
guarantees across machines or optional dependencies. CI stops the unit step
after two minutes and the E2E step after ten minutes to bound runaway checks.

Expensive gradients, convergence studies, full guide replays, and exhaustive
checks remain in `validation`.
Run them separately, or run all tests:

```bash
python -m pytest -m validation --durations=20
python -m pytest
```

Validation runs in one pytest process, including in pre-merge CI. Measure
changes with this sequential command; parallel wall time does not establish
a cheaper suite. Five minutes is the target, not a measured CI guarantee.
Preserve independent numerical references, failure paths and statistical
sample sizes when reducing repeated setup or solver work.

Every collected test belongs to exactly one of these suites. Mark small test
modules with `pytestmark = pytest.mark.unit` and expensive functions with
`@pytest.mark.validation`. For an expensive parameter value, use
`pytest.param(value, marks=pytest.mark.validation)`. Unmarked tests enter `e2e`,
so new tests run in PR CI by default. Conflicting suite markers fail collection. Keep meaningful cheap
checks; choose validation candidates by cost and coverage overlap, while
retaining representative numerical oracles in E2E. Coverage overlap alone
does not show that two tests check the same assertion.

The location markers `core`, `physics_sentinel`, `extended`, and `examples`
remain available for focused work. They do not promise a runtime budget.

### CI and merging

| Event | Checks |
| --- | --- |
| PR opened or updated | Lint/types, then separate `unit` and `e2e` runs on Python 3.11 and 3.12 |
| PR marked ready for review, or updated while ready | `validation` on Python 3.11 |
| Push to `main` | Relevant docs build/deployment; no repeated test or benchmark run |
| Manual pre-merge run | `validation` on Python 3.11 |
| Manual benchmark run | Full benchmark ladder against the selected comparison ref |
| Weekly dependency canary | `unit` and `e2e` on Python 3.11 and 3.12 with fresh dependencies |

Open new PRs as drafts (`gh pr create --draft`). Drafts run the fast checks;
marking a PR ready for review starts validation. Further pushes while ready
rerun validation for the updated commit. Converting back to draft cancels a
running validation workflow.

The required fast and pre-merge checks together cover the full suite without
repeating the fast tests on Python 3.11. Documentation, workflow changes, and
release metadata use lightweight validation. A version-only edit to
`quchip/__init__.py` is release metadata; other source changes, tests, executable
examples (including Markdown), tooling, and repository rules require test lanes.
An unreadable change list fails classification.

The required `pre-merge full suite` check skips validation for drafts and
lightweight changes. Its name is retained for branch protection. Manual dispatch
always runs validation on the selected ref; scheduled dependency checks do not.

The `main` ruleset requires the PR branch to include current `main` before
merging. Its configuration is recorded in
[`.github/rulesets/main.json`](.github/rulesets/main.json); changing this file
does not update GitHub automatically. An administrator must apply it to
ruleset `21738854` before merging the workflow changes that remove post-merge
test runs. Required check names are preserved for existing PRs.

Tests that require dynamiqs use the `optional_backend` marker and call `pytest.importorskip("dynamiqs")`, so they skip cleanly when dynamiqs is unavailable.

Test public behavior and physical invariants, not implementation details. A behavior-preserving refactor should not require mechanical test edits; see [Change-Detector Tests Considered Harmful](https://testing.googleblog.com/2015/01/testing-on-toilet-change-detector-tests.html).

Before opening a pull request, run:

```bash
python -m pytest -m unit
python -m pytest -m e2e
ruff check .
python -m mypy quchip tests/typing/external_declarative_models.py
```

Run new or changed validation tests locally while developing the physics.
Run the complete validation suite at merge readiness by marking the PR ready
for review, or through a manual pre-merge run, as described above.

Ruff uses a 120-character line limit. Public API docstrings use NumPy-style sections and imperative summaries ending with periods. Every test has a one-line docstring stating the invariant under test.

Document every public constructor, function, and method parameter, including
keyword-only and inherited arguments. Give accepted forms, defaults, units,
allowed choices or ranges, and the meaning of `None` where applicable. Expand
forwarded options or link to their precise contract. Keep descriptions short;
put equations and derivations in `Notes`, with primary physics references.
Describe result attributes and return values with shapes, axis order, and units.

With the docs extra installed, run `python tools/check_api_docs.py` to detect
missing descriptions and obsolete parameter names on package exports and the
returned types listed in that tool. This checks structure; review numerical
conventions and references against the implementation, then inspect rendered
API pages. A successful Sphinx build alone does not establish completeness.
The check includes the dynamiqs backend when that extra is installed. The docs
job installs `.[docs,dynamiqs]` so both backend implementations are inspected
and the complete reference can be built with warnings treated as errors.

## Examples and notebooks

The [guides](docs/guides/index.md) and [cookbook](docs/cookbook.md) explain how to use quchip, including model construction, parameter changes, sweeps and result interpretation. Follow the same workflows in contributed examples. [Writing examples](docs/contribute/writing-examples.md) covers organization, figures and numerical checks.

Readable Jupytext Markdown is the canonical source. Commit its executed `.ipynb` partner with the `python3` kernel, identical code cells, and inspected outputs. Use standard fenced Python cells rather than percent-format scripts.

Notebook outputs remain in the executed `.ipynb`; save only figures selected for the website under `docs/images/` and include the canonical Markdown from its page under `docs/guides/`. Figures load `docs/_static/quchip.mplstyle`, save a single light `.svg`, and `python3 tools/theme_figures.py` derives the `-dark.svg`, `.pdf`, and README `-dark.png` variants (it rejects colors outside the palette).

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

The diff must be empty and the strict round trip must pass. Inspect the executed notebook before committing: setup cells stay silent, stdout contains only intended receipts, and every intended figure appears once. Run the focused example test and the project checks above.

## Physics conventions

Read [PHYSICS.md](PHYSICS.md) before changing Hamiltonians, frames, approximations, observables, dissipation, or solver assembly.

- Use ordinary GHz for frequencies and energies, ns for time, and mK for temperature.
- Keep traced paths compatible with JAX. Do not call `float()`, `int()`, or `bool()` on traced values or branch on them with Python. Use `jax.numpy` and JAX control flow.
- State approximations, omitted terms, and validity regimes through the model's declared approximation and `physics_notes()`.
- Use derived tolerances based on an analytic limit, truncation error, solver convergence, an independent calculation, or a cited reference. State the basis next to the assertion. Never tune a tolerance only to make a test pass.

## Pull requests

Keep pull requests small and focused. Explain the physical or user-visible change, include tests for changed behavior, and update the relevant documentation.

Open an issue before starting a large change or adding a device, coupling, drive, envelope, or noise model. Include the model Hamiltonian, assumptions, intended use, and a reference when available so the scope can be agreed before implementation.

## Releases

Keep release notes in `CHANGELOG.md` under `## [version] - YYYY-MM-DD`.
Each entry has two parts:

1. `### 0.x series highlights`: a short cumulative overview of the main
   capabilities introduced in that feature series. While quchip is pre-1.0,
   a new series starts at `0.x.0` (for example, 0.4.0). Carry the overview into
   later patch releases, updating it only for substantial additions. Describe
   these as series capabilities so older features are not presented as new
   in the current patch.
2. `### Changes since <previous release>`: changes since the immediately
   preceding published tag, with a full comparison link. Use the following
   `####` categories in order, omitting empty categories: Changes and migration,
   New features, Fixes, Performance, Compatibility, Documentation and examples,
   Development. At the first release of a series, the overview summarizes the
   features detailed here; later patches retain the overview and list only
   their own changes here.

Write one short bullet per user-visible outcome, with a PR link. Combine
related implementation changes. Lead the series overview with new scientific
capabilities, such as solvers and device models. State supported equations,
backends, and experimental limits where relevant. Keep internal refactors and
CI details brief in Development; performance claims need measured evidence.

Changes to defaults, approximations, units, signs, initial states, or tolerances
must explain the old and new behavior and any required user action, even when
signatures are unchanged. Intentional changes to documented behavior belong in
Changes and migration. Fixes describe the affected case and correction;
flag numerical consequences there too. Deprecations identify the replacement
and planned removal release. Keep migration steps in the release entry unless
a longer guide is needed.

Before tagging, review the entire change set since the previous published tag,
move `Unreleased` into the dated entry, and run
`python tools/release_notes.py <version>` to inspect the release body. The docs
include the changelog, and the tag workflow uses that same section for GitHub
after publishing to PyPI. Avoid separately maintained copies.

For editorial corrections after publication, merge the changelog update,
regenerate the GitHub release body with the same command, and verify both the
published release and deployed docs. Keep the published tag and package intact;
code changes require another release.

## Policies

- There is no CLA or DCO. By opening a pull request you agree that your contribution is provided under the project's license.
- If AI assistance was used, disclose it in the pull request description. The contributor remains accountable for every claim and change.
- Be respectful and constructive in issues, discussions, and reviews. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
