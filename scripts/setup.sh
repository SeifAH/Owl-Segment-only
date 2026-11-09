#!/usr/bin/env bash
# Shared setup for run_app.sh and run_folder.sh (sourced from the project root): installs uv if needed, syncs the
# Python environment (CUDA PyTorch on Linux), downloads the OWLv2 and SAM models once.
set -e
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/owl-sam-segment}"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    echo "[setup] Installing uv - the Python package manager ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo "[setup] Checking the Python environment ..."
uv sync --frozen
if [ ! -f "$UV_PROJECT_ENVIRONMENT/.models_ok" ]; then
    echo "[setup] Downloading the OWLv2 and SAM models, about 1 GB - first run only ..."
    uv run --frozen python scripts/download_models.py
    echo ok > "$UV_PROJECT_ENVIRONMENT/.models_ok"
fi
# Models are cached locally - no need to contact the internet
export HF_HUB_OFFLINE=1
