#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash train.sh /path/to/config.yaml" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
AI_TOOLKIT_DIR="${HOME}/ai-toolkit"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Error: .venv not found. Run bash setup.sh first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER
# After FLUX weights are fully cached locally, set HF_HUB_OFFLINE=1 to block hub traffic.

cd "${AI_TOOLKIT_DIR}"
exec python run.py "$@"
