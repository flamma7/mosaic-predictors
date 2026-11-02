#!/usr/bin/env bash
# Bootstrap a RunPod image for MoPWM training.
# Usage: setup_pusht.sh [pusht|tworoom]
# Torch and CUDA are already in the image Python; this script does not replace them.
# HF_TOKEN must already be set. Training is not started; the command is printed.
set -euo pipefail

TASK="${1:-pusht}"
case "${TASK}" in
  pusht|tworoom) ;;
  *)
    echo "Usage: $0 [pusht|tworoom]" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

# PyPI stable-worldmodel 0.1.1. This repo's pyproject.toml and uv.lock point
# that package at ../stable-worldmodel, which is the desktop checkout and is
# not on RunPod. --directory and --no-config keep uv from reading them.
# Leave the image torch/torchvision in place.
uv pip install \
  --directory /tmp \
  --python "${PY}" \
  --break-system-packages \
  --no-config \
  --no-sources \
  --excludes "${EXCLUDES}" \
  "stable-worldmodel[env,format,train]==0.1.1" \
  "einops>=0.8.2" \
  "huggingface-hub>=2.2.0"

uv pip install \
  --python "${PY}" \
  --break-system-packages \
  --no-deps \
  --no-sources \
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

if [[ "${TASK}" == pusht ]]; then
  DATASET_PATH="${DATASET_DIR}/pusht_expert_train.lance"
  echo "Downloading PushT to ${DATASET_PATH}"
  if [[ -d "${DATASET_PATH}" ]] && [[ -n "$(ls -A "${DATASET_PATH}")" ]]; then
    echo "Already present: ${DATASET_PATH}"
  else
    "${HF_BIN}" download galilai-group/lewm-pusht pusht_expert_train.lance/ \
      --repo-type dataset \
      --local-dir "${DATASET_DIR}" \
      --token "${HF_TOKEN}"
  fi
  RUN_NAME=mopwm-pusht4
else
  DATASET_PATH="${DATASET_DIR}/tworoom.h5"
  echo "Downloading TwoRoom to ${DATASET_PATH}"
  if [[ -f "${DATASET_PATH}" ]]; then
    echo "Already present: ${DATASET_PATH}"
  else
    if ! command -v zstd >/dev/null 2>&1; then
      echo "zstd is required to unpack tworoom.tar.zst." >&2
      exit 1
    fi
    "${HF_BIN}" download quentinll/lewm-tworooms tworoom.tar.zst \
      --repo-type dataset \
      --local-dir "${DATASET_DIR}" \
      --token "${HF_TOKEN}"
    tar --zstd -xf "${DATASET_DIR}/tworoom.tar.zst" -C "${DATASET_DIR}"
    rm -f "${DATASET_DIR}/tworoom.tar.zst"
  fi
  RUN_NAME=mopwm-tworoom
fi

echo
echo "Start training with:"
echo "STABLEWM_HOME=${STABLEWM_HOME} ${PY} ${ROOT}/scripts/train/mopwm.py data=${TASK} output_model_name=${RUN_NAME}"
