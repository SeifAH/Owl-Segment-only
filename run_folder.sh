#!/usr/bin/env bash
# Linux: process every image in a folder -> outputs/batch_<timestamp>/ (annotated images + summary.csv + objects.csv)
#   ./run_folder.sh /path/to/images [--objectness 0.15] [--out DIR]
[ -d "$1" ] || { echo "usage: $0 <folder> [--objectness X] [--out DIR]"; exit 1; }
FOLDER="$(realpath "$1")"; shift
cd "$(dirname "$0")"
source scripts/setup.sh
exec uv run --frozen python scripts/batch.py "$FOLDER" "$@"
