#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v vhs >/dev/null 2>&1; then
  echo "vhs is required to render demo/vectrace-demo.gif"
  echo "Install: brew install vhs"
  exit 1
fi

echo "Rendering demo/vectrace-demo.gif ..."
vhs demo/vectrace-demo.tape
echo "Done: demo/vectrace-demo.gif"
