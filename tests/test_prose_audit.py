"""Format-aware coverage for the repository prose-audit workflow."""

from __future__ import annotations

import json
import subprocess

from tools.prose_audit import Passage
from tools.prose_audit.cli import build_report, main
from tools.prose_audit.extract import extract_html, extract_markdown, extract_python
from tools.prose_audit.rules import find_candidates


def test_extract_python_returns_only_docstrings_with_source_owners(tmp_path) -> None:
    """Comments and ordinary string literals do not enter the prose inventory."""
    path = tmp_path / "sample.py"
    source = '''"""Module summary."""

# A prose comment is outside the requested audit scope.
class Device:
    """Device summary."""

    note = "ordinary string"

    def solve(self):
        """Solve the model."""
        return "result text"


async def prepare():
    """Prepare the model."""
'''

    passages = extract_python(path, source)

    assert [(item.line, item.owner, item.text) for item in passages] == [
        (1, "<module>", "Module summary."),
        (5, "Device", "Device summary."),
        (10, "Device.solve", "Solve the model."),
        (15, "prepare", "Prepare the model."),
    ]
    assert {item.kind for item in passages} == {"docstring"}


def test_extract_markdown_keeps_authored_prose_and_excludes_literal_syntax(tmp_path) -> None:
    """Markdown structure remains reviewable without treating code or math as prose."""
    path = tmp_path / "guide.md"
    source = """# Device guide

Model the `Chip` with the [physics guide](https://example.test/physics).
Keep the declared approximation explicit.

- First authored item.
- Second authored item.

![Pulse response for the readout resonator](pulse.png)

```python
print("Here's the thing: code is not prose")
```

$$
H = 2\\pi f a^\\dagger a
$$
"""

    passages = extract_markdown(path, source)
    combined = "\n".join(item.text for item in passages)

    assert [(item.line, item.kind) for item in passages] == [
        (1, "markdown-heading"),
        (3, "markdown-prose"),
        (6, "markdown-list"),
        (7, "markdown-list"),
        (9, "markdown-alt"),
    ]
    assert "Device guide" in combined
    assert "Model the Chip with the physics guide." in combined
    assert "Keep the declared approximation explicit." in combined
    assert "Pulse response for the readout resonator" in combined
    assert "example.test" not in combined
    assert "print" not in combined
    assert "2\\pi" not in combined


def test_extract_markdown_handles_front_matter_and_myst_figure_prose(tmp_path) -> None:
    """Jupytext metadata stays out while MyST accessibility text and captions stay in."""
    path = tmp_path / "figure.md"
    source = """---
jupyter:
  kernelspec:
    name: python3
---

```{figure} response.png
:width: 560px
:alt: Conditional resonator response

The two paths separate over the declared readout pulse.
```
"""

    passages = extract_markdown(path, source)

    assert [(item.line, item.kind, item.text) for item in passages] == [
        (9, "markdown-alt", "Conditional resonator response"),
        (11, "markdown-prose", "The two paths separate over the declared readout pulse."),
    ]


def test_extract_html_keeps_visible_content_and_accessibility_text(tmp_path) -> None:
    """Generated navigation, source listings, scripts, and styles stay out of the audit."""
    path = tmp_path / "page.html"
    source = """<!doctype html>
<html><body>
  <nav>Previous Next Search</nav>
  <main>
    <h1>Device model</h1>
    <p>The solver receives an explicit physics description.</p>
    <img src="response.png" alt="Conditional readout response">
    <button aria-label="Copy example">icon</button>
    <div class="highlight"><pre>print("not prose")</pre></div>
  </main>
  <script>const phrase = "not prose";</script>
  <style>.hidden { display: none; }</style>
</body></html>
"""

    passages = extract_html(path, source)
    combined = "\n".join(item.text for item in passages)

    assert "Device model" in combined
    assert "The solver receives an explicit physics description." in combined
    assert "Conditional readout response" in combined
    assert "Copy example" in combined
    assert "Previous Next Search" not in combined
    assert "print" not in combined
    assert "const phrase" not in combined
    assert "display: none" not in combined


def _candidate_ids(text: str) -> set[str]:
    passage = Passage("guide.md", 10, "markdown-prose", "<document>", text)
    return {candidate.rule_id for candidate in find_candidates(passage)}


def test_named_rules_detect_high_confidence_patterns() -> None:
    """The audit reports recognizable patterns by name instead of guessing authorship."""
    cases = {
        "Here's the thing: the engine is robust.": {"throat-clearing", "business-jargon"},
        "The question isn't speed. It's trust.": {"binary-contrast"},
        "What most people get wrong is the frame.": {"faux-insight"},
        "This marks a pivotal moment for the toolkit.": {"importance-puffery"},
        "Experts agree that this approach works.": {"weasel-attribution"},
        "This distinction matters more than it sounds.": {"interpretive-metadiscourse"},
        "The implications are significant.": {"vague-declaration"},
        "Speed. Quality. Cost. That's it.": {"dramatic-fragmentation"},
        "The best part: it learns.": {"colon-reveal"},
        "In conclusion, the model is complete.": {"recap-ending"},
    }

    for text, expected in cases.items():
        assert _candidate_ids(text) >= expected


def test_broad_stop_slop_checks_are_contextual_review_signals() -> None:
    """Categorical house-style bans cannot become confirmed findings on their own."""
    ids = _candidate_ids(
        "How is the state represented? The model is carefully tested — in simulation, hardware, and analysis."
    )

    assert ids >= {"wh-opener", "passive-voice", "adverb", "em-dash", "three-item-list"}


def test_scientific_and_api_prose_has_no_confirmed_pattern_by_default() -> None:
    """Technical vocabulary, uncertainty, and necessary passives remain protected content."""
    passage = Passage(
        "docs/physics.md",
        20,
        "markdown-prose",
        "<document>",
        "The state is represented in the dressed basis. chip.freq returns the 0→1 transition; "
        "the time-dependent Hamiltonian may include 2π factors at the engine boundary.",
    )

    candidates = find_candidates(passage)

    assert not [candidate for candidate in candidates if candidate.confidence == "high"]


def test_citations_and_compact_parameter_notes_are_not_dramatic_fragments() -> None:
    """Periods in citations and terse API constraints are not manufactured drama."""
    examples = (
        "Blais et al., Rev. Mod. Phys. 93, 025005 (2021), §II.B.",
        "Charging energy in GHz. Positive. JAX-traceable.",
        '"Git gud" grants none. Explain. Cite. Suggest.',
    )

    for text in examples:
        assert "dramatic-fragmentation" not in _candidate_ids(text)


def test_repository_report_reconciles_tracked_and_rendered_targets(tmp_path) -> None:
    """Every discovered target receives a disposition and stable machine-readable counts."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("generated/\nlocal.md\n", encoding="utf-8")
    (tmp_path / "sample.py").write_text('"""Here\'s the thing: module prose."""\n', encoding="utf-8")
    (tmp_path / "untracked.py").write_text('"""Untracked working-tree prose."""\n', encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Guide\n\nDirect authored prose.\n", encoding="utf-8")
    (tmp_path / "deleted.md").write_text("Tracked prose pending deletion.\n", encoding="utf-8")
    (tmp_path / "local.md").write_text("Local ignored prose.\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Not in the requested formats.\n", encoding="utf-8")
    rendered = tmp_path / "generated"
    rendered.mkdir()
    (rendered / "page.html").write_text("<main><p>Rendered prose.</p></main>\n", encoding="utf-8")
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "add",
            ".gitignore",
            "sample.py",
            "guide.md",
            "deleted.md",
            "notes.txt",
        ],
        check=True,
    )
    (tmp_path / "deleted.md").unlink()

    report = build_report(tmp_path, rendered, [tmp_path / "local.md"])

    assert report["inventory"] == {
        "python_files": 2,
        "markdown_files": 2,
        "html_files": 1,
        "docstrings": 2,
        "markdown_passages": 3,
        "html_passages": 1,
        "total_passages": 6,
        "high_confidence_candidates": 1,
        "contextual_candidates": 0,
        "failures": 0,
        "unaccounted_targets": 0,
    }
    assert [(target["path"], target["status"]) for target in report["targets"]] == [
        ("generated/page.html", "clean"),
        ("guide.md", "clean"),
        ("local.md", "clean"),
        ("sample.py", "candidate"),
        ("untracked.py", "clean"),
    ]


def test_cli_writes_stable_json_and_fails_on_extraction_errors(tmp_path) -> None:
    """Malformed targets are visible failures rather than silent omissions."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "broken.py"], check=True)
    output = tmp_path / "audit.json"

    exit_code = main(["--root", str(tmp_path), "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["inventory"]["failures"] == 1
    assert report["inventory"]["unaccounted_targets"] == 0
    assert report["failures"][0]["path"] == "broken.py"
    assert report["targets"][0]["status"] == "failed"
