#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SYNC_DRIVE=0
for arg in "$@"; do
  case "${arg}" in
    --sync-drive) SYNC_DRIVE=1 ;;
    --no-sync) SYNC_DRIVE=0 ;;
  esac
done

PIPELINE_CONFIG="${SCRIPT_DIR}/pipeline.yaml"
PIPELINE_EXAMPLE="${SCRIPT_DIR}/pipeline.example.yaml"
if [[ ! -f "${PIPELINE_CONFIG}" && -f "${PIPELINE_EXAMPLE}" ]]; then
  cp "${PIPELINE_EXAMPLE}" "${PIPELINE_CONFIG}"
  echo "[setup] Created ${PIPELINE_CONFIG} from pipeline.example.yaml — edit paths and HF token."
fi

VENV_DIR="${SCRIPT_DIR}/.venv"
AI_TOOLKIT_DIR="${HOME}/ai-toolkit"
AI_TOOLKIT_REF="fbac1cb7f50f86b539f5c56a7847ef8878df012f"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
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
  echo "[setup] Installing current rclone (Ubuntu apt can be too old for Google login)"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq curl unzip
    curl -fsSL https://rclone.org/install.sh | sudo bash
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL https://rclone.org/install.sh | sudo bash
  else
    echo "[setup] Error: curl unavailable. Install current rclone manually from https://rclone.org/downloads/" >&2
    exit 1
  fi
}

drive_remote_name() {
  local remote_name=""
  if [[ -f "${PIPELINE_CONFIG}" ]]; then
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

print_rclone_config_help() {
  local remote_name="$1"
  cat <<EOF
[setup] Google Drive setup uses rclone. Create one remote for this project.
[setup] In the rclone wizard, use these answers:
[setup]   n                         # New remote
[setup]   name: ${remote_name}
[setup]   Storage: drive            # Google Drive; type 'drive' or choose its number
[setup]   client_id: <press Enter>
[setup]   client_secret: <press Enter>
[setup]   scope: 1                  # Full Drive access, needed for pull and push
[setup]   root_folder_id: <press Enter>
[setup]   service_account_file: <press Enter>
[setup]   Edit advanced config: n
[setup]   Use auto config: n        # Headless/cloud instance
[setup]   Open the printed URL in your browser, approve Google access, paste the code back here.
[setup]   Configure shared/team drive: n
[setup]   Keep this remote: y
[setup]   Quit config: q
EOF
}

ensure_rclone_remote() {
  local remote_name="$1"
  if rclone lsd "${remote_name}:" >/dev/null 2>&1; then
    return 0
  fi

  echo "[setup] rclone remote '${remote_name}' is not configured or is not accessible."
  echo "[setup] Starting 'rclone config'. Create a Google Drive remote named '${remote_name}'."
  print_rclone_config_help "${remote_name}"
  rclone config

  if ! rclone lsd "${remote_name}:" >/dev/null 2>&1; then
    echo "[setup] Error: rclone still cannot access '${remote_name}:'. Re-run 'rclone config' and try again." >&2
    exit 1
  fi
}

if [[ "${SYNC_DRIVE}" -eq 1 ]]; then
  install_rclone
  ensure_rclone_remote "$(drive_remote_name)"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] Creating Python virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

cleanup_invalid_pip_distributions() {
  python - <<'PY'
import site
import shutil
from pathlib import Path

for site_dir in site.getsitepackages():
    root = Path(site_dir)
    if not root.is_dir():
        continue
    for path in root.glob("~*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
PY
}

cleanup_invalid_pip_distributions

echo "[setup] Upgrading pip"
python -m pip install --upgrade pip -q

echo "[setup] Installing PyTorch (CUDA 12.8)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q

if [[ -f "${REQUIREMENTS_FILE}" ]]; then
  echo "[setup] Installing project requirements"
  pip install -r "${REQUIREMENTS_FILE}" -q
fi

sync_ai_toolkit_checkout() {
  if [[ -d "${AI_TOOLKIT_DIR}/.git" ]]; then
    echo "[setup] Updating pinned ai-toolkit checkout (${AI_TOOLKIT_REF})"
    git -C "${AI_TOOLKIT_DIR}" fetch origin --tags
  else
    echo "[setup] Cloning ai-toolkit into ${AI_TOOLKIT_DIR}"
    git clone https://github.com/ostris/ai-toolkit.git "${AI_TOOLKIT_DIR}"
    git -C "${AI_TOOLKIT_DIR}" fetch origin --tags
  fi

  git -C "${AI_TOOLKIT_DIR}" checkout --detach "${AI_TOOLKIT_REF}"
  git -C "${AI_TOOLKIT_DIR}" submodule sync --recursive
  git -C "${AI_TOOLKIT_DIR}" submodule update --init --recursive
}

sync_ai_toolkit_checkout

AI_TOOLKIT_REQUIREMENTS="${AI_TOOLKIT_DIR}/requirements.txt"
if [[ ! -f "${AI_TOOLKIT_REQUIREMENTS}" ]]; then
  echo "[setup] Error: ai-toolkit requirements not found at ${AI_TOOLKIT_REQUIREMENTS}" >&2
  exit 1
fi

echo "[setup] Installing ai-toolkit dependencies (may take several minutes)"
pip install -r "${AI_TOOLKIT_REQUIREMENTS}"

patch_ai_toolkit_hf_transfer_env() {
  local run_py="${AI_TOOLKIT_DIR}/run.py"
  local deprecated='os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "1")'
  local replacement='os.environ["HF_XET_HIGH_PERFORMANCE"] = os.getenv("HF_XET_HIGH_PERFORMANCE", "1")'

  if [[ ! -f "${run_py}" ]]; then
    return 0
  fi
  if grep -Fq "${deprecated}" "${run_py}"; then
    echo "[setup] Patching ai-toolkit run.py (HF_XET_HIGH_PERFORMANCE replaces deprecated hf_transfer env)"
    sed -i "s|$(printf '%s' "${deprecated}" | sed 's/[&/\|]/\\&/g')|$(printf '%s' "${replacement}" | sed 's/[&/\|]/\\&/g')|" "${run_py}"
  fi
}

patch_ai_toolkit_hf_transfer_env

if [[ "${SYNC_DRIVE}" -eq 1 ]]; then
  echo "[setup] Pulling training assets from Google Drive (thunder_compute/)"
  python3 drive_sync.py check
  python3 drive_sync.py pull --profile training --only input,flux,venv
fi

echo "[setup] Environment ready. Activate with: source ${VENV_DIR}/bin/activate"
echo "[setup] Interactive TUI: bash tui.sh"
echo "[setup] CLI commands (run separately, not with pipes):"
echo "[setup]   source ${VENV_DIR}/bin/activate"
echo "[setup]   python3 run_pipeline.py --from preprocess"
echo "[setup] Note: run_pipeline.py is the non-interactive CLI runner, not the TUI."
echo "[setup] Optional FA2 compile: python3 post-setup.py --max-jobs 3"
