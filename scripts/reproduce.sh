#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TZ=Asia/Shanghai
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -z "${TRANSREID_PYTHON:-}" ]]; then
    conda_root="$(conda info --base)"
    TRANSREID_PYTHON="$conda_root/envs/transreid/bin/python"
fi
mkdir -p logs
exec 9>logs/reproduction.lock
if ! flock -n 9; then
    echo 'Another reproduction process is already running.' >&2
    exit 1
fi
printf '%s\n' "$$" > logs/reproduction.pid
trap 'printf "FAILED at %s\n" "$(date -Is)" > logs/reproduction.status' ERR
if [[ $# -eq 0 ]]; then
    set -- market1501 msmt17
fi
for dataset in "$@"; do
    case "$dataset" in market1501|msmt17) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2 ;; esac
    config="configs/local/${dataset}_transreid_stride.yml"
    output="logs/${dataset}_transreid_stride"
    mkdir -p "$output"
    if [[ ! -f "$output/metrics_epoch_120.json" ]]; then
        printf 'TRAINING %s %s\n' "$dataset" "$(date -Is)" > logs/reproduction.status
        resume=()
        if [[ -f "$output/latest_training.pth" ]]; then
            resume=(SOLVER.RESUME "$output/latest_training.pth")
        fi
        "$TRANSREID_PYTHON" -u train.py --config_file "$config" "${resume[@]}" >> "$output/console.log" 2>&1
    fi
    printf 'EVALUATING %s %s\n' "$dataset" "$(date -Is)" > logs/reproduction.status
    "$TRANSREID_PYTHON" -u test.py --config_file "$config" \
        MODEL.PRETRAIN_CHOICE self TEST.WEIGHT "$output/transformer_120.pth" \
        >> "$output/evaluation_console.log" 2>&1
done
"$TRANSREID_PYTHON" scripts/summarize_results.py > logs/results.md
printf 'COMPLETED %s\n' "$(date -Is)" > logs/reproduction.status
