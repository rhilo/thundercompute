#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_DIR="${SCRIPT_DIR}/.venv"
AI_TOOLKIT_DIR="${HOME}/ai-toolkit"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] Creating Python virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "[setup] Upgrading pip"
python -m pip install --upgrade pip -q

echo "[setup] Installing PyTorch (CUDA 12.8)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q

if [[ -f "${REQUIREMENTS_FILE}" ]]; then
  echo "[setup] Installing project requirements"
  pip install -r "${REQUIREMENTS_FILE}" -q
fi

if [[ -d "${AI_TOOLKIT_DIR}/.git" ]]; then
  echo "[setup] Updating existing ai-toolkit checkout"
  git -C "${AI_TOOLKIT_DIR}" pull --ff-only
  git -C "${AI_TOOLKIT_DIR}" submodule update --init --recursive
else
  echo "[setup] Cloning ai-toolkit into ${AI_TOOLKIT_DIR}"
  git clone https://github.com/ostris/ai-toolkit.git "${AI_TOOLKIT_DIR}"
  git -C "${AI_TOOLKIT_DIR}" submodule update --init --recursive
fi

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

echo "[setup] Environment ready. Activate with: source ${VENV_DIR}/bin/activate"
