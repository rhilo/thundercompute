#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
AI_TOOLKIT_DIR="${HOME}/ai-toolkit"
PIPELINE_CONFIG="${SCRIPT_DIR}/pipeline.yaml"

AI_TOOLKIT_TRAIN_CONFIG=""
if [[ $# -ge 1 && "${1}" != --config ]]; then
  AI_TOOLKIT_TRAIN_CONFIG="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      PIPELINE_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Usage: bash train.sh [ai-toolkit-config.yaml] [--config pipeline.yaml]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Error: .venv not found. Run bash setup.sh first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if [[ -f "${PIPELINE_CONFIG}" ]]; then
  cd "${SCRIPT_DIR}"
  eval "$(python3 - "${PIPELINE_CONFIG}" <<'PY'
import sys
from pathlib import Path
from pipeline_config import apply_runtime_env, load_pipeline_settings

settings = load_pipeline_settings(Path(sys.argv[1]))
apply_runtime_env(settings)
config_out = str(settings.ai_toolkit_config_out)
print(f'export PIPELINE_AI_TOOLKIT_CONFIG="{config_out}"')
PY
)"
  if [[ -z "${AI_TOOLKIT_TRAIN_CONFIG}" ]]; then
    AI_TOOLKIT_TRAIN_CONFIG="${PIPELINE_AI_TOOLKIT_CONFIG}"
  fi
else
  export WANDB_MODE="${WANDB_MODE:-disabled}"
  unset HF_HUB_ENABLE_HF_TRANSFER
fi

if [[ -z "${AI_TOOLKIT_TRAIN_CONFIG}" ]]; then
  echo "Error: no ai-toolkit config path. Pass it as an argument or set paths.ai_toolkit_config_out in pipeline.yaml." >&2
  exit 1
fi

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

cd "${AI_TOOLKIT_DIR}"
exec python run.py "${AI_TOOLKIT_TRAIN_CONFIG}"
