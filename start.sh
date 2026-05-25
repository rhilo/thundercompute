#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_DIR="${SCRIPT_DIR}/.venv"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[start] Creating minimal venv for the TUI at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python - <<'PY' >/dev/null 2>&1 || pip install -q "textual==0.47.0" "PyYAML==6.0.1"
import textual
import yaml
PY

exec python pipeline_tui.py "$@"
