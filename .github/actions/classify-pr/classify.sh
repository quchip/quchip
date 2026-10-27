#!/usr/bin/env bash
# Fail closed if the compared commits cannot be read.
set -euo pipefail

if [[ "${EVENT_NAME:?}" != "pull_request" ]]; then
  echo "full_ci=true"
  exit 0
fi

changed_files=$(mktemp)
trap 'rm -f "$changed_files"' EXIT
git diff --no-renames --name-only -z "${BASE_SHA:?}" "${HEAD_SHA:?}" > "$changed_files"

full_ci=false
while IFS= read -r -d '' changed_file; do
  case "$changed_file" in
    .github/workflows/*|.github/actions/classify-pr/*)
      ;;
    examples/*|tests/*|quchip/*|tools/*|.github/actions/*|.github/rulesets/*)
      full_ci=true
      break
      ;;
    docs/*|*.md|CITATION.cff|LICENSE|.gitignore|.github/workflows/release.yml)
      ;;
    *)
      full_ci=true
      break
      ;;
  esac
done < "$changed_files"

echo "full_ci=$full_ci"
