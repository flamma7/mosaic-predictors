#!/usr/bin/env bash
# Bootstrap a RunPod image for DINO-WM or MoPWM training.
# Usage: setup_train.sh [pusht|tworoom]
# Torch and CUDA are already in the image Python; this script does not replace them.
# HF_TOKEN must already be set. Training is not started; the command is printed.
#
# stable-worldmodel is cloned at SWM_REV and then patched from this repo.
# PyPI 0.1.1 does not contain the PushT visual-shift and Lance forkserver
# changes. Regenerate the patch from the desktop checkout (the second
# command exits 1 because the test file is untracked):
#   git -C ../stable-worldmodel diff > patches/stable-worldmodel.patch
#   git -C ../stable-worldmodel diff --no-index -- /dev/null tests/test_visual_perturbation.py >> patches/stable-worldmodel.patch
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
# Keep the DINOv2 download on the persistent volume, next to the dataset.
HF_HOME="${HF_HOME:-${STABLEWM_HOME}/huggingface}"
export STABLEWM_HOME HF_HOME

SWM_REV="21446f1ede6d5284e981bd7b47f432b994e6d812"
SWM_URL="https://github.com/galilai-group/stable-worldmodel.git"
SWM_SRC="${STABLEWM_HOME}/src/stable-worldmodel"
PATCH="${ROOT}/patches/stable-worldmodel.patch"

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
if [[ ! -f "${PATCH}" ]]; then
  echo "Missing ${PATCH}" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required to fetch stable-worldmodel ${SWM_REV}." >&2
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

mkdir -p "$(dirname "${SWM_SRC}")"
if [[ ! -d "${SWM_SRC}/.git" ]]; then
  git init -q "${SWM_SRC}"
  git -C "${SWM_SRC}" remote add origin "${SWM_URL}"
fi

synced=0
if git -C "${SWM_SRC}" rev-parse --verify HEAD >/dev/null 2>&1 \
  && [[ "$(git -C "${SWM_SRC}" rev-parse HEAD)" == "${SWM_REV}" ]] \
  && git -C "${SWM_SRC}" apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
  synced=1
  echo "stable-worldmodel already at ${SWM_REV} with patches/stable-worldmodel.patch."
fi
if [[ "${synced}" -eq 0 ]]; then
  echo "Fetching stable-worldmodel ${SWM_REV}"
  git -C "${SWM_SRC}" fetch --depth 1 origin "${SWM_REV}"
  git -C "${SWM_SRC}" checkout --force --detach FETCH_HEAD
  git -C "${SWM_SRC}" clean -fdq
  git -C "${SWM_SRC}" apply "${PATCH}"
fi

EXCLUDES="$(mktemp)"
trap 'rm -f "${EXCLUDES}"' EXIT
printf '%s\n' torch torchvision > "${EXCLUDES}"

# --directory and --no-config keep uv from reading this repo's pyproject,
# which points stable-worldmodel at the desktop checkout. Leave the image
# torch/torchvision in place. [env] is pygame, pymunk, and opencv for PushT
# MPC. [format] is the Lance dataset. [train] is transformers (DINOv2),
# stable-pretraining, and Lightning. scikit-learn is imported by the MPC
# eval and is otherwise only a transitive dependency.
uv pip install \
  --directory /tmp \
  --python "${PY}" \
  --break-system-packages \
  --no-config \
  --no-sources \
  --excludes "${EXCLUDES}" \
  -e "${SWM_SRC}[env,format,train]" \
  "einops>=0.8.2" \
  "huggingface-hub>=2.2.0" \
  "scikit-learn"

uv pip install \
  --python "${PY}" \
  --break-system-packages \
  --no-deps \
  --no-sources \
  -e "${ROOT}"

SDL_VIDEODRIVER=dummy "${PY}" - <<'PY'
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import cv2
import pygame
import pymunk
from sklearn.preprocessing import StandardScaler
from stable_worldmodel.envs.pusht.env import PushT
from stable_worldmodel.wrapper.visual import (
    apply_blur,
    apply_dark,
    apply_salt_pepper,
    make_adajepa_wrappers,
)

env = PushT()
try:
    env.reset(seed=0)
    frame = env.render()
finally:
    env.close()
if frame.ndim != 3 or frame.shape[-1] != 3:
    raise SystemExit(f"PushT render returned {getattr(frame, 'shape', None)}")
print(
    f"PushT inference ok  frame {tuple(frame.shape)}  "
    f"cv2 {cv2.__version__}  pygame {pygame.ver}  pymunk {pymunk.version}"
)
del StandardScaler, apply_blur, apply_dark, apply_salt_pepper, make_adajepa_wrappers
PY

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
  echo "Caching facebook/dinov2-small under ${HF_HOME}"
  HF_HOME="${HF_HOME}" "${PY}" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("facebook/dinov2-small")
PY
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
    tar --zstd --no-same-owner -xf "${DATASET_DIR}/tworoom.tar.zst" -C "${DATASET_DIR}"
    rm -f "${DATASET_DIR}/tworoom.tar.zst"
  fi
  RUN_NAME=mopwm-tworoom
fi

ENV_PREFIX="SDL_VIDEODRIVER=dummy HF_HOME=${HF_HOME} STABLEWM_HOME=${STABLEWM_HOME}"
echo
if [[ "${TASK}" == pusht ]]; then
  echo "Start DINO-WM training with:"
  echo "${ENV_PREFIX} ${PY} ${ROOT}/scripts/train/dinowm.py data=pusht output_model_name=dinowm-pusht"
  echo
  echo "MoPWM:"
  echo "${ENV_PREFIX} ${PY} ${ROOT}/scripts/train/mopwm.py data=pusht output_model_name=${RUN_NAME}"
else
  echo "Start training with:"
  echo "${ENV_PREFIX} ${PY} ${ROOT}/scripts/train/mopwm.py data=${TASK} output_model_name=${RUN_NAME}"
fi
