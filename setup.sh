#!/usr/bin/env bash
# Create a local Python environment, install dependencies, and run import/syntax checks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python 3.10+ is required (set PYTHON=/path/to/python if needed)." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Error: Python 3.10 or newer is required.")
print(f"Using Python {sys.version.split()[0]}")
PY

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [[ -x .venv/bin/python ]]; then
  VENV_PYTHON=.venv/bin/python
  VENV_PIP=.venv/bin/pip
elif [[ -x .venv/Scripts/python.exe ]]; then
  VENV_PYTHON=.venv/Scripts/python.exe
  VENV_PIP=.venv/Scripts/pip.exe
else
  echo "Error: could not locate Python in the created .venv." >&2
  exit 1
fi

"$VENV_PIP" install --upgrade pip
"$VENV_PIP" install -r requirements.txt
"$VENV_PYTHON" scripts/verify_project.py

echo
printf 'Setup complete. Activate the environment with:\n'
if [[ -f .venv/bin/activate ]]; then
  echo "  source .venv/bin/activate"
else
  echo "  .venv\\Scripts\\activate"
fi
