#!/usr/bin/env bash
# Render main.tex to PDF (tectonic) and HTML (pandoc) from the same source.
# Usage: ./scripts/render.sh  (or call from anywhere - paths auto-resolve)
#
# Layout:
#   <repo>/main.tex            - LaTeX source
#   <repo>/references.bib      - BibTeX references
#   <repo>/scripts/render.sh   - this script
#   <repo>/docs/main.pdf      - generated PDF
#   <repo>/docs/index.html    - generated HTML (GitHub Pages compatible)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"

for tool in tectonic pandoc; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Error: '$tool' not found in PATH. Install with: brew install $tool" >&2
    exit 1
  fi
done

cd "$REPO_ROOT"
mkdir -p "$DOCS_DIR"

echo "[1/2] tectonic: main.tex -> docs/main.pdf"
tectonic main.tex --outdir="$DOCS_DIR"
echo "      -> $DOCS_DIR/main.pdf ($(ls -lh "$DOCS_DIR/main.pdf" | awk '{print $5}'))"

echo "[2/2] pandoc:   main.tex -> docs/index.html"
pandoc main.tex \
  --from=latex \
  --to=html5 \
  --standalone \
  --embed-resources \
  --mathjax \
  --citeproc \
  --bibliography=references.bib \
  --metadata=link-citations=true \
  --metadata=lang=en \
  --shift-heading-level-by=1 \
  --syntax-highlighting=tango \
  --output="$DOCS_DIR/index.html"
echo "      -> $DOCS_DIR/index.html ($(ls -lh "$DOCS_DIR/index.html" | awk '{print $5}'))"

echo "Done."
