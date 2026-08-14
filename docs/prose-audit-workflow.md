# Prose audit workflow

This workflow scans repository prose for named writing patterns without guessing
whether AI wrote it. It covers Python docstrings, Markdown prose, and visible or
accessibility text in rendered HTML.

## Sources and rule contract

The primary contract comes from
[`petergyang/no-ai-slop`](https://github.com/petergyang/no-ai-slop) at commit
`d30eddb9e04562234f2070b5ee63ca4649d9a05e`: preserve the writer's voice, make
the minimum effective edit, name each detected pattern, quote its evidence, and
do not assign an authorship score.

The contextual signals come from
[`hardikpandya/stop-slop`](https://github.com/hardikpandya/stop-slop) at commit
`8da1f030185bdfe8471220585162991eaeb970e9`. Its exact filler, puffery,
attribution, and rhetorical patterns are useful candidates. Its blanket rules
against passive voice, adverbs, em dashes, Wh-word openings, and three-item lists
are review signals only. Applying those bans mechanically would damage physics
definitions, API reference material, and intentional project voice.

## Run the inventory

From the repository root:

```bash
.venv/bin/python -m tools.prose_audit.cli \
  --root . \
  --rendered-html docs/_build/html \
  --extra-path TRUTH.md \
  --extra-path EXAMPLES_PLAN.md \
  --output docs/_build/prose-audit/candidates.json
```

The command discovers tracked and untracked, non-ignored Python, Markdown, and
HTML files. Repeat `--extra-path` for each deliberately ignored authored source
that belongs in the audit, such as `TRUTH.md` or `EXAMPLES_PLAN.md`;
`--rendered-html` adds ignored Sphinx output as a derived verification set. The
output remains under ignored `docs/_build/`.

The command exits nonzero for an extraction failure or an unaccounted target. A
style candidate does not fail the command.

## What extraction includes

- Python: module, class, function, and async-function docstrings, with owning
  symbols and physical source lines.
- Markdown: headings, paragraphs, lists, tables, block quotations, link labels,
  and image alt text.
- HTML: visible text plus `alt` and `aria-label` text.

The extractors exclude Python comments and ordinary string literals; Markdown
code fences, inline syntax, link destinations, and display math; and HTML
scripts, styles, navigation, code listings, and generated source views.

## Adjudicate candidates

The deterministic candidate list is only the discovery pass. Split a large
repository into non-overlapping semantic lanes—authored Markdown, production
docstrings, test/tooling docstrings, and rendered-only text—and read every
passage in each lane, including passages with no lexical match. A scout must
return exact `file:line` evidence, the named pattern, confidence, contextual
reasoning, and a preserve/repair recommendation. A separate triage pass then
re-reads each proposal in its owning section or symbol.

Give each proposed issue one disposition:

- **confirmed**: the named pattern weakens this passage in context;
- **dismissed**: the match is legitimate technical or intentional prose; or
- **duplicate-generated**: rendered HTML repeats an authored source passage.

High-confidence lexical matches are still candidates, not verdicts. Protect
equations, API identifiers, domain terms, qualified claims, uncertainty,
necessary passives, citations, accessibility text, and deliberate examples.
Check technical claims against `TRUTH.md`, `PHYSICS.md`, implementation, or tests
before treating them as writing problems.

Review contextual signals in clusters, then inspect every member of a suspicious
cluster in its source. Typical legitimate clusters include:

- passive voice where a physical object or transformation matters more than an
  actor;
- three-item or longer lists that enumerate parameters, states, solver paths,
  or invariants;
- Wh-word openings in headings and NumPy-style parameter clauses;
- em dashes separating equations, qualifications, or deliberate contrasts; and
- adverbs that distinguish actual from nominal behavior or carry mathematical
  precision.

Rendered Sphinx candidates must be mapped back to their Markdown or docstring.
Only prose introduced by the rendering layer stays attached to an HTML path.

Review the authored entries in the JSON `passages` array in file order. Apply
the portability test to short standalone claims; check module introductions for
catalogues that duplicate the API; compare repeated explanations across module,
class, and helper docstrings; inspect test prose for generic rigor claims; and
inspect section endings for generic recaps. These semantic patterns often have
no banned word. Preserve specific technical details and deliberate cadence.

## Report and verify

Record each confirmed finding with `file:line`, the smallest useful quotation,
rule provenance, confidence, contextual rationale, and a short repair direction.
Also record representative dismissals and the complete inventory totals.

Keep the working findings ledger at
`docs/_build/prose-audit/prose-audit-findings.md`. It contains audit-process
notes and quotations of known-bad prose, so it should enter the published
documentation only through an explicit publication decision.

Run:

```bash
MPLCONFIGDIR=/tmp/quchip-matplotlib \
  .venv/bin/python -m pytest tests/test_prose_audit.py -q
.venv/bin/ruff check tools/prose_audit tests/test_prose_audit.py
git diff --check
```

Finally regenerate `candidates.json`. Its `failures` list must be empty,
`inventory.unaccounted_targets` must be zero, and every quoted finding must
still match its source line.

The audit does not rewrite prose or establish a CI gate. Remediation is a
separate reviewable change.
