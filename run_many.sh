#!/usr/bin/env bash
# 150-episode LeWM planning evals at seed 0.
#
# Each run writes only to its own directory. The success rate is the
# success_rate field under "==== RESULTS ====" in results.txt.
#
#   outputs/run_many/lewm_published/pusht/seed_0/results.txt
#   outputs/run_many/lewm_published/avoidt/seed_0/results.txt
#   outputs/run_many/lewm_tworoom_epoch10/tworoom/seed_0/results.txt
#   outputs/run_many/lewm_tworoom_epoch10/tworoom_route/seed_0/results.txt
#   outputs/run_many/lewm_pusht3_epoch10/pusht/seed_0/results.txt
#   outputs/run_many/lewm_pusht3_epoch10/avoidt/seed_0/results.txt
#   outputs/run_many/mopwm-tworoom-local5/tworoom/seed_0/results.txt
#   outputs/run_many/mopwm-tworoom-local5/tworoom_route/seed_0/results.txt
#   outputs/run_many/mopwm-pusht5/pusht/seed_0/results.txt
#   outputs/run_many/mopwm-pusht5/avoidt/seed_0/results.txt
#   outputs/run_many/lewm_pusht3_epoch10/stackt/seed_0/results.txt
#   outputs/run_many/lewm_pusht3_epoch10/pusht-antigoal/seed_0/results.txt
#   outputs/run_many/mopwm-pusht5/stackt/seed_0/results.txt
#   outputs/run_many/mopwm-pusht5/pusht-antigoal/seed_0/results.txt

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ROOT}/outputs/run_many"
NUM_EVAL=150
SEED=0

TWOROOM_CKPT="/home/flamma/Projects/.stablewm/checkpoints/lewm-tworoom/weights_epoch_10.pt"
PUSHT3_CKPT="/home/flamma/Projects/.stablewm/checkpoints/lewm-pusht3/weights_epoch_10.pt"
MOPWM_TWOROOM_CKPT="/home/flamma/Projects/.stablewm/checkpoints/mopwm-tworoom-local5/weights_epoch_10.pt"
MOPWM_PUSHT_CKPT="/home/flamma/Projects/.stablewm/checkpoints/mopwm-pusht5/weights_epoch_10.pt"

run_eval() {
  local name="$1"
  local dataset="$2"
  local checkpoint="${3:-}"
  local model="${4:-lewm}"
  local out_dir="${OUT}/${name}/seed_${SEED}"

  mkdir -p "${out_dir}"
  echo "=== ${name}  dataset=${dataset}  seed=${SEED}  num_eval=${NUM_EVAL} ==="
  echo "    ${out_dir}/results.txt"

  local cmd=(
    uv run --project "${ROOT}" anygoal eval planning
    --model "${model}"
    --dataset "${dataset}"
    --seed "${SEED}"
    # --download
  )
  if [[ -n "${checkpoint}" ]]; then
    cmd+=(--checkpoint "${checkpoint}")
  fi
  cmd+=(
    --
    "eval.num_eval=${NUM_EVAL}"
    "output.directory=${out_dir}"
    "output.filename=results.txt"
  )
  "${cmd[@]}"
}

# run_eval "lewm_published/pusht" pusht
# run_eval "lewm_pusht3_epoch10/pusht" pusht "${PUSHT3_CKPT}"
# run_eval "lewm_published/avoidt" avoidt
# run_eval "lewm_pusht3_epoch10/avoidt" avoidt "${PUSHT3_CKPT}"
# run_eval "lewm_tworoom_epoch10/tworoom" tworoom "${TWOROOM_CKPT}"
# run_eval "lewm_tworoom_epoch10/tworoom_route" tworoom_route "${TWOROOM_CKPT}"

# run_eval "mopwm-tworoom-local5/tworoom" tworoom "${MOPWM_TWOROOM_CKPT}" mopwm
# run_eval "mopwm-tworoom-local5/tworoom_route" tworoom_route "${MOPWM_TWOROOM_CKPT}" mopwm
# run_eval "mopwm-pusht5/pusht" pusht "${MOPWM_PUSHT_CKPT}" mopwm
# run_eval "mopwm-pusht5/avoidt" avoidt "${MOPWM_PUSHT_CKPT}" mopwm

run_eval "lewm_pusht3_epoch10/stackt" stackt "${PUSHT3_CKPT}"
run_eval "lewm_pusht3_epoch10/pusht-antigoal" pusht-antigoal "${PUSHT3_CKPT}"
run_eval "mopwm-pusht5/stackt" stackt "${MOPWM_PUSHT_CKPT}" mopwm
run_eval "mopwm-pusht5/pusht-antigoal" pusht-antigoal "${MOPWM_PUSHT_CKPT}" mopwm
