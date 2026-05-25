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

install_rclone() {
  if command -v rclone >/dev/null 2>&1; then
    return 0
  fi
  echo "[setup-comfy] Installing rclone"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq rclone
  else
    echo "[setup-comfy] Error: install rclone manually." >&2
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
