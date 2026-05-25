#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_DIR="${SCRIPT_DIR}/.venv"
PIPELINE_CONFIG="${SCRIPT_DIR}/pipeline.yaml"
MIN_RCLONE_VERSION="1.68.0"

rclone_needs_install() {
  if ! command -v rclone >/dev/null 2>&1; then
    return 0
  fi
  local version
  version="$(rclone version | awk 'NR==1 {print $2}' | sed 's/^v//')"
  if [[ -z "${version}" ]]; then
    return 0
  fi
  [[ "$(printf '%s\n%s\n' "${MIN_RCLONE_VERSION}" "${version}" | sort -V | head -n 1)" != "${MIN_RCLONE_VERSION}" ]]
}

install_rclone() {
  if ! rclone_needs_install; then
    return 0
  fi

  echo "[start] Installing current rclone (Ubuntu apt can be too old for Google login)"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq curl unzip
    curl -fsSL https://rclone.org/install.sh | sudo bash
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL https://rclone.org/install.sh | sudo bash
  else
    echo "[start] Error: curl unavailable. Install current rclone manually from https://rclone.org/downloads/" >&2
    exit 1
  fi
}

drive_remote_name() {
  local remote_name="${RCLONE_REMOTE_NAME:-}"
  if [[ -z "${remote_name}" && -f "${PIPELINE_CONFIG}" ]]; then
    remote_name="$(
      awk -F: '
        /^[[:space:]]*rclone_remote:/ {
          value=$2
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
          gsub(/^["'\'']|["'\'']$/, "", value)
          print value
          exit
        }
      ' "${PIPELINE_CONFIG}"
    )"
  fi
  printf '%s\n' "${remote_name:-gdrive}"
}

warn_if_rclone_remote_missing() {
  local remote_name
  remote_name="$(drive_remote_name)"
  if rclone listremotes 2>/dev/null | grep -Fxq "${remote_name}:"; then
    return 0
  fi

  echo "[start] Google Drive remote '${remote_name}' is not configured yet."
  echo "[start] Opening the TUI anyway. Drive buttons will need rclone configured first."
  echo "[start] To configure Drive later, run: rclone config"
}

install_rclone
warn_if_rclone_remote_missing

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
