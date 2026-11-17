#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TZ=Asia/Shanghai
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
if [[ -z "${TRANSREID_PYTHON:-}" ]]; then
    conda_root="$(conda info --base)"
    TRANSREID_PYTHON="$conda_root/envs/transreid/bin/python"
fi

output=logs/market1501_transreid_no_sie
mkdir -p "$output"
exec 9>logs/reproduction.lock
if ! flock -n 9; then
    echo 'Another training run holds logs/reproduction.lock.' >&2
    exit 1
fi
printf '%s\n' "$$" > "$output/run.pid"
trap 'printf "FAILED %s\n" "$(date -Is)" > "$output/status"' ERR

# Keep the supplied configuration for subsequent evaluation and restarts.
config="$output/source_config.yml"
if [[ ! -f "$config" ]]; then
    cp configs/Market/vit_transreid_no_sie.yml "$config"
fi
options=(
    MODEL.SIE_CAMERA False
    MODEL.SIE_VIEW False
    MODEL.SIE_COE 0.0
    SOLVER.CHECKPOINT_PERIOD 10
    TEST.IMS_PER_BATCH 64
    OUTPUT_DIR "$output"
)

if [[ ! -f "$output/metrics_epoch_120.json" ]]; then
    printf 'TRAINING %s\n' "$(date -Is)" > "$output/status"
    resume=()
    if [[ -f "$output/latest_training.pth" ]]; then
        resume=(SOLVER.RESUME "$output/latest_training.pth")
    fi
    "$TRANSREID_PYTHON" -u train.py --config_file "$config" \
        "${options[@]}" "${resume[@]}" >> "$output/console.log" 2>&1
fi

printf 'EVALUATING %s\n' "$(date -Is)" > "$output/status"
"$TRANSREID_PYTHON" -u test.py --config_file "$config" \
    "${options[@]}" MODEL.PRETRAIN_CHOICE self \
    TEST.WEIGHT "$output/transformer_120.pth" >> "$output/evaluation_console.log" 2>&1
printf 'COMPLETED %s\n' "$(date -Is)" > "$output/status"
