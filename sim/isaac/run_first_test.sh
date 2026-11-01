#!/usr/bin/env bash
# Original balance pipeline; generated data lives beside the repository.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${PRIMUS_WORKSPACE:-$(cd -- "$HERE/../../.." && pwd)}"
export PRIMUS_RUN_DIR="${PRIMUS_RUN_DIR:-$WORKSPACE/experiments/$(date +%Y-%m-%d_%H-%M-%S)-balance}"
mkdir -p "$PRIMUS_RUN_DIR"
DEVICE="${PRIMUS_DEVICE:-cuda:0}"
python3 "$HERE/launch.py" preflight.py --device "$DEVICE" --output "$PRIMUS_RUN_DIR/preflight.json" 2>&1 | tee "$PRIMUS_RUN_DIR/preflight.log"
python3 "$HERE/launch.py" convert_urdf_to_usd.py 2>&1 | tee "$PRIMUS_RUN_DIR/conversion.log"
python3 "$HERE/launch.py" smoke_test.py --device "$DEVICE" --num_envs 2 --steps 300 2>&1 | tee "$PRIMUS_RUN_DIR/smoke.log"
python3 "$HERE/launch.py" run_rl.py train --device "$DEVICE" --num_envs 32 --max_iterations 20 --visualizer none 2>&1 | tee "$PRIMUS_RUN_DIR/train.log"
