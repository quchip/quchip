"""Write small HTML pages that send moved documentation pages to their new address."""

import os
from pathlib import Path

from sphinx.errors import ExtensionError

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Page moved</title>
<link rel="canonical" href="{target}">
<meta name="robots" content="noindex">
<meta http-equiv="refresh" content="0; url={target}">
<script>location.replace("{target}" + location.hash);</script>
</head>
<body>
<p>This page moved to <a href="{target}">{target}</a>.</p>
</body>
</html>
"""


def check_redirects(app, env):
    """Fail the build when a source is still a page or a target is not one."""
    for old, new in app.config.page_redirects.items():
        if old in env.found_docs:
            raise ExtensionError(f"Redirect source {old!r} is still a page.")
        if new not in env.found_docs:
            raise ExtensionError(f"Redirect target {new!r} is not a page.")


def write_redirects(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    outdir = Path(app.outdir)
    for old, new in app.config.page_redirects.items():
        page = outdir / f"{old}.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        target = Path(os.path.relpath(outdir / f"{new}.html", page.parent)).as_posix()
        page.write_text(TEMPLATE.format(target=target), encoding="utf-8")


def setup(app):
    app.add_config_value("page_redirects", {}, "html")
    app.connect("env-updated", check_redirects)
    app.connect("build-finished", write_redirects)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
