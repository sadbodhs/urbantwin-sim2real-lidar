#!/usr/bin/env bash
# Run a command inside the project container with the GPU, the repo and the data mounted.
# Builds the image first if it is missing.
#   UT_GPUS=device=0   GPU(s) to use (default: all)
#   UT_NOGPU=1         run without a GPU (data prep, CPU evaluation)
#   UT_CPUS=N          docker --cpus cap (Mac jobs next to other containers)
#   OMP_NUM_THREADS=N  passed into the container when set (one thread per dataloader worker)
#   scripts/docker_run.sh python -m utl.prepare ...
#   scripts/docker_run.sh            # interactive shell
set -euo pipefail

IMAGE="${UT_IMAGE:-urbantwin-lidar:latest}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${UT_DATA:-$REPO/data}"
mkdir -p "$DATA_DIR"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" "$REPO"

TTY_FLAGS="-i"
[ -t 0 ] && [ -t 1 ] && TTY_FLAGS="-it"
USER_ARGS=(--user "$(id -u):$(id -g)")
case "$(docker info -f '{{.SecurityOptions}}' 2>/dev/null || true)" in *rootless*) USER_ARGS=() ;; esac
GPU_ARGS=(--gpus "${UT_GPUS:-all}")
[ "${UT_NOGPU:-0}" = 1 ] && GPU_ARGS=()

if [ $# -eq 0 ]; then set -- bash; fi

exec docker run --rm $TTY_FLAGS \
    ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} \
    --ipc=host \
    ${USER_ARGS[@]+"${USER_ARGS[@]}"} \
    -e HOME=/tmp/home \
    -e NUMBA_CACHE_DIR=/tmp/numba \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ${OMP_NUM_THREADS:+-e OMP_NUM_THREADS=$OMP_NUM_THREADS} \
    ${UT_CPUS:+--cpus $UT_CPUS} \
    ${CUDA_LAUNCH_BLOCKING:+-e CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING} \
    -v "$REPO":/workspace \
    -v "$DATA_DIR":/workspace/data \
    -w /workspace \
    "$IMAGE" "$@"
