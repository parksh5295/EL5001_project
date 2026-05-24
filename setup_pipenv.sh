#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v pipenv >/dev/null 2>&1; then
  echo "pipenv is not installed."
  echo "Install once on host Python: python -m pip install --user pipenv"
  exit 1
fi

export PIPENV_VENV_IN_PROJECT=1

echo "[1/2] Creating pipenv virtual environment"
pipenv --python 3.10 >/dev/null

echo "[2/2] Installing dependencies from requirements.txt"
pipenv install -r requirements.txt

echo "Done. Virtual env location: $ROOT_DIR/.venv"
