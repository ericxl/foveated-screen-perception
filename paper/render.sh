#!/usr/bin/env bash
# Render main.tex to PDF using tectonic.
# Usage: ./render.sh
# Tectonic handles bibtex/references automatically in a single pass.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Rendering main.tex -> main.pdf ..."
tectonic main.tex

echo "Done: $(ls -lh main.pdf | awk '{print $5}') -> $SCRIPT_DIR/main.pdf"
