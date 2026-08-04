#!/usr/bin/env bash
set -eu

if [ -z "${1:-}" ]; then
    echo "Usage: bash train.sh <gpu_ids> [args...]"
    echo "Example: bash train.sh 0,1 --dataset cifar100 --increment 10"
    exit 1
fi

GPUS=$1
NB_COMMA=$(echo "${GPUS}" | tr -cd , | wc -c)
NB_GPUS=$((NB_COMMA + 1))
PORT=$((9000 + RANDOM % 1000))

shift

echo "Launching exp on GPU(s): ${GPUS} (${NB_GPUS} processes), port ${PORT}..."
CUDA_VISIBLE_DEVICES=${GPUS} torchrun \
    --master_port ${PORT} \
    --nproc_per_node=${NB_GPUS} \
    -- main.py "$@"
