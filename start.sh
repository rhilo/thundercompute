#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_DIR="${SCRIPT_DIR}/.venv"

install_rclone() {
  if command -v rclone >/dev/null 2>&1; then
    return 0
  fi

  echo "[start] Installing rclone"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq rclone
  else
    echo "[start] Error: rclone not found and apt-get unavailable. Install rclone manually." >&2
    exit 1
  fi
}

ensure_rclone_remote() {
  local remote_name="${RCLONE_REMOTE_NAME:-gdrive}"
  if rclone listremotes 2>/dev/null | grep -Fxq "${remote_name}:"; then
    return 0
  fi

  echo "[start] rclone is installed, but remote '${remote_name}' is not configured."
  echo "[start] Starting 'rclone config'. Create a Google Drive remote named '${remote_name}'."
  rclone config
}

install_rclone
ensure_rclone_remote

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
