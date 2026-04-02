#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install build twine setuptools wheel
python3 -m build
python3 -m twine check dist/*

echo "Built and validated dist artifacts in ./dist"
