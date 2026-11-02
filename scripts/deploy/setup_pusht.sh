#!/usr/bin/env bash
# Bootstrap a RunPod image for LeWM PushT training.
# Torch and CUDA are already in the image Python; this script does not replace them.
# HF_TOKEN must already be set. Training is not started; the command is printed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STABLEWM_HOME="/workspace/.stablewm"
export STABLEWM_HOME

if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  PY="$(command -v python3)"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set." >&2
  exit 1
fi

echo "Using Python: ${PY}"
"${PY}" - <<'PY'
import sys
import torch

print(f"torch {torch.__version__}  cuda {torch.version.cuda}  available {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available in this interpreter. Point PYTHON at the RunPod image Python.")
PY

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

EXCLUDES="$(mktemp)"
trap 'rm -f "${EXCLUDES}"' EXIT
printf '%s\n' torch torchvision > "${EXCLUDES}"

# PyPI stable-worldmodel 0.1.1, not the local checkout in tool.uv.sources.
# Leave the image torch/torchvision in place.
uv pip install \
  --python "${PY}" \
  --break-system-packages \
  --no-sources \
  --excludes "${EXCLUDES}" \
  "stable-worldmodel[env,format,train]==0.1.1" \
  "einops>=0.8.2" \
  "huggingface-hub>=2.2.0"

uv pip install \
  --python "${PY}" \
  --break-system-packages \
  --no-deps \
  -e "${ROOT}"

DATASET_DIR="${STABLEWM_HOME}/datasets"
mkdir -p "${DATASET_DIR}"
# Console script installed into this interpreter's environment, not whatever `hf` is first on PATH.
HF_BIN="$("${PY}" - <<'PY'
import sysconfig
from pathlib import Path

hf = Path(sysconfig.get_path("scripts")) / "hf"
print(hf if hf.is_file() else "")
PY
)"
if [[ -z "${HF_BIN}" ]]; then
  echo "hf download CLI was not installed next to ${PY}." >&2
  exit 1
fi

echo "Downloading PushT to ${DATASET_DIR}/pusht_expert_train.lance"
if [[ -d "${DATASET_DIR}/pusht_expert_train.lance" ]] && [[ -n "$(ls -A "${DATASET_DIR}/pusht_expert_train.lance")" ]]; then
  echo "Already present: ${DATASET_DIR}/pusht_expert_train.lance"
else
  "${HF_BIN}" download galilai-group/lewm-pusht pusht_expert_train.lance/ \
    --repo-type dataset \
    --local-dir "${DATASET_DIR}" \
    --token "${HF_TOKEN}"
fi

TRAIN="${ROOT}/src/mosaic_predictors/transformer_based/train/lewm.py"
echo
echo "Start training with:"
echo "STABLEWM_HOME=${STABLEWM_HOME} ${PY} ${TRAIN} data=pusht"
