#!/usr/bin/env bash
# PushT planning evals for each AdaJEPA visual shift, one at a time.
#
# LeWM and MoPWM each record one env_<i>.mp4 per scenario.
# Results land in outputs/run_many/<model>/pusht-<visual>/results.txt.
#
#   outputs/run_many/lewm/pusht-blur/results.txt
#   outputs/run_many/mopwm/pusht-red-agent/results.txt

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ROOT}/outputs/run_many"
NUM_EVAL=25
SEED=42

LEWM_CKPT="lewm-pusht3/weights_epoch_10.pt"
MOPWM_CKPT="mopwm-pusht5/weights_epoch_10.pt"

# name, then the hydra override that turns that shift on.
# An empty override is the unperturbed baseline.
PERTURBATIONS=(
  "none|"
  "blur|visual.blur=true"
  "snp1|visual.salt_pepper=true"
  "snp5|visual.salt_pepper=0.05"
  "dark|visual.dark=0.7"
  "red-agent|visual.agent_color=red"
  "red-block|visual.block_color=red"
  "red-anchor|visual.anchor_color=red"
)

run_eval() {
  local model="$1"
  local checkpoint="$2"
  local visual="$3"
  local override="$4"
  local record_video="$5"
  local out_dir="${OUT}/${model}/pusht-${visual}"

  mkdir -p "${out_dir}"
  rm -f "${out_dir}/results.txt"

  echo "=== ${model}  pusht-${visual}  num_eval=${NUM_EVAL}  video=${record_video} ==="
  echo "    ${out_dir}/results.txt"

  local cmd=(
    python "${ROOT}/scripts/plan/eval_wm.py"
    --config-name pusht
    "policy=${checkpoint}"
    "seed=${SEED}"
    "eval.num_eval=${NUM_EVAL}"
    "output.directory=${out_dir}"
    "output.filename=results.txt"
    "video=${record_video}"
  )
  if [[ -n "${override}" ]]; then
    cmd+=("${override}")
  fi
  if (($# > 5)); then
    cmd+=("${@:6}")
  fi
  (cd "${ROOT}" && "${cmd[@]}")
}

for spec in "${PERTURBATIONS[@]}"; do
  visual="${spec%%|*}"
  override="${spec#*|}"
  run_eval "lewm" "${LEWM_CKPT}" "${visual}" "${override}" true
  run_eval "mopwm" "${MOPWM_CKPT}" "${visual}" "${override}" true
done

# for spec in "${PERTURBATIONS[@]}"; do
#   visual="${spec%%|*}"
#   override="${spec#*|}"
#   run_eval "mopwm-full" "${MOPWM_CKPT}" "${visual}" "${override}" false "max_memory=null"
# done

python "${ROOT}/scripts/plan/visual_table.py" "${OUT}"
