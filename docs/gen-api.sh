#!/usr/bin/env bash
# Generate the Autodoc page (docs/api/index.md) from the header comments:
#   doxygen  -> XML (build/doxyxml)   ->   moxygen -> Markdown (docs/api/index.md)
# Requires `doxygen` and `npx` (Node) on PATH. Run from the repo root; the CI
# docs job runs this before `zensical build`. Safe to run locally to preview.
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> doxygen -> XML"
doxygen Doxyfile

echo ">> moxygen -> docs/api/index.md"
mkdir -p docs/api
npx --yes moxygen@2.1.12 \
    --output docs/api/index.md \
    build/doxyxml

# moxygen titles the page from the first symbol; prepend a stable H1 so the nav
# entry and page heading read "Autodoc".
tmp="$(mktemp)"
{ printf '# Autodoc\n\nGenerated from the header Doxygen comments (`doxygen` + `moxygen`). For a\ncurated view see the [Reference](../reference.md) and [Cheat sheet](../cheatsheet.md).\n\n'; cat docs/api/index.md; } > "$tmp"
mv "$tmp" docs/api/index.md
echo ">> wrote docs/api/index.md"
