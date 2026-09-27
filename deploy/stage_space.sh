#!/usr/bin/env bash
# Stages the Hugging Face Space payload (app.py + dana/) into $1.
# Shared by build.yml's space-smoke job and deploy_hf.yml so CI imports
# exactly the file set that gets deployed — the 2026-09-24 Space outage was
# committed code importing modules that only existed in a local working tree.
set -euo pipefail
dest="${1:?usage: stage_space.sh <dest-dir>}"

rm -rf "$dest"
mkdir -p "$dest"
cp app.py "$dest/app.py"
cp packages.txt "$dest/packages.txt"
# deploy/requirements-space.txt doesn't exist (was never committed even
# though the deploy step once referenced it) — root requirements.txt is the
# proven-correct file already used for this exact codebase locally. It
# installs more than a minimal Space container strictly needs (the
# desktop/audio/Windows-only stack); trimming it requires verifying
# app.py's full transitive import graph, not guessing.
cp requirements.txt "$dest/requirements.txt"
cp deploy/space_README.md "$dest/README.md"

cp -r dana "$dest/dana"
find "$dest/dana" -name "__pycache__" -type d -prune -exec rm -rf {} +
