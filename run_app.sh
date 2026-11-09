#!/usr/bin/env bash
# Linux launcher: setup + start the web app at http://127.0.0.1:7862
cd "$(dirname "$0")"
source scripts/setup.sh
exec uv run --frozen python app.py
