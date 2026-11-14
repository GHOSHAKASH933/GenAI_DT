#!/usr/bin/env bash
# Create a local virtual environment, install this project, and verify imports.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python was not found. Install Python 3.10+ or set PYTHON=/path/to/python." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required.")
print("Using Python", sys.version.split()[0])
PY

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
if [[ -x .venv/bin/python ]]; then
  VENV_PYTHON=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then
  VENV_PYTHON=.venv/Scripts/python.exe
else
  echo "Could not locate Python in .venv." >&2
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -e '.[dev]'
"$VENV_PYTHON" scripts/verify_project.py

echo
printf 'Setup complete. Activate the environment with:\n'
if [[ -f .venv/bin/activate ]]; then
  echo '  source .venv/bin/activate'
else
  echo '  .venv\\Scripts\\Activate.ps1'
fi
