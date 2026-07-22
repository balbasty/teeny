#!/usr/bin/env bash
# Generate the Autodoc page (docs/api/index.md) from the header comments:
#   doxygen  -> XML (build/doxyxml)   ->   moxygen -> Markdown (docs/api/index.md)
# Requires `doxygen` and `npx` (Node) on PATH. Run from the repo root; the CI
# docs job runs this before `zensical build`. Safe to run locally to preview.
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> doxygen -> XML"
# doxygen won't create a missing PARENT of XML_OUTPUT (build/doxyxml), and a
# clean CI checkout has no build/ dir — so make it first (mkdir -p is recursive).
mkdir -p build/doxyxml
doxygen Doxyfile

echo ">> moxygen -> docs/api/index.md"
mkdir -p docs/api
# --noindex: skip moxygen's summary index tables so the page is a single H1
# (`# tny`) with H2 category sections (Classes/Enumerations/Typedefs/Functions)
# and one H3 per symbol — that hierarchy is what drives a clean in-page TOC
# (toc_depth = 3 keeps the H4 members out of the TOC).
npx --yes moxygen@2.1.12 \
    --noindex \
    --output docs/api/index.md \
    build/doxyxml

# Post-process the generated page:
#  1. one H1 that reads "Autodoc" (matches the nav entry) + a short intro —
#     replacing moxygen's `# tny`, so the page has exactly one H1 and the
#     in-page TOC is well-formed.
#  2. drop the `cs::` namespace qualifier: every cuda::std name in teeny's public
#     surface is imported into `tny` (alias.h `using cs::…`), so within these docs
#     they read as their bare, tny-imported names (`extents`, `layout_right`, …).
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("docs/api/index.md")
s = p.read_text()
s = s.replace("cs::", "")                         # hide the cs:: qualifier (symbols stay; they're imported into tny)
intro = ("# Autodoc\n\nGenerated from the header Doxygen comments "
         "(`doxygen` + `moxygen`). For a curated view see the "
         "[Reference](../reference.md) and [Cheat sheet](../cheatsheet.md).\n")
s = s.lstrip("\n")                                # drop leading blank lines so ^ matches the heading
s = re.sub(r'^# .*\n', intro, s, count=1)         # retitle the single leading H1 (moxygen's `# tny`)
p.write_text(s)
PY
echo ">> wrote docs/api/index.md"
