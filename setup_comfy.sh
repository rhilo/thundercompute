#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SYNC_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --sync-only) SYNC_ONLY=1 ;;
  esac
done

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
  echo "[setup-comfy] Installing current rclone (Ubuntu apt can be too old for Google login)"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq curl unzip
    curl -fsSL https://rclone.org/install.sh | sudo bash
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL https://rclone.org/install.sh | sudo bash
  else
    echo "[setup-comfy] Error: curl unavailable. Install current rclone manually from https://rclone.org/downloads/" >&2
    exit 1
  fi
}

install_rclone

if [[ ! -f "${SCRIPT_DIR}/drive_sync.py" ]]; then
  echo "[setup-comfy] Error: drive_sync.py not found in ${SCRIPT_DIR}" >&2
  exit 1
fi

echo "[setup-comfy] Pulling ComfyUI assets from Google Drive (thunder_compute/)"
python3 drive_sync.py check
python3 drive_sync.py pull --profile comfyui

if [[ "${SYNC_ONLY}" -eq 1 ]]; then
  echo "[setup-comfy] Sync complete (--sync-only)."
  exit 0
fi

for subdir in unet clip vae loras; do
  path="${HOME}/ComfyUI/models/${subdir}"
  if [[ -d "${path}" ]] && [[ -n "$(ls -A "${path}" 2>/dev/null || true)" ]]; then
    echo "[setup-comfy] OK: ${path}"
  else
    echo "[setup-comfy] Warning: ${path} is empty or missing — add weights on Drive under models/${subdir}/"
  fi
done

echo "[setup-comfy] Ready. Run ComfyUI, then push renders:"
echo "  python3 drive_sync.py push --profile comfyui"
