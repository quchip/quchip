#!/usr/bin/env bash
# Regenerate the CI constraint files from pyproject.toml + ci-constraints.in.
#
# CI installs with `-c ci-constraints-<python>.txt`, so test runs are
# reproducible regardless of upstream releases. Package metadata stays
# unpinned; these files affect CI only. The deps-canary workflow tests the
# unconstrained resolution weekly and flags drift.
#
# Run from the repo root after editing ci-constraints.in, or periodically to
# pick up new upstream releases (a green PR is the gate for adopting them).
set -euo pipefail
cd "$(dirname "$0")/.."

for py in 3.11 3.12; do
    uv pip compile pyproject.toml \
        --python-version "$py" \
        --python-platform linux \
        --extra dev --extra test --extra dynamiqs \
        -c ci-constraints.in \
        -o "ci-constraints-$py.txt" \
        --quiet
    echo "wrote ci-constraints-$py.txt"
done
