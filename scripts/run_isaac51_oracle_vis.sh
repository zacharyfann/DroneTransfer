#!/usr/bin/env bash
# Run visualized oracle training on Isaac Sim 5.1 + Isaac Lab v2.3.0.
#
# Run INSIDE isaac-sim-51 container:
#   bash /workspace/drone/scripts/run_isaac51_oracle_vis.sh
#
# Resume from checkpoint (visual, 1 env):
#   CHECKPOINT=/workspace/drone/checkpoints/oracle/ppo_oracle_80000_steps \
#   STEPS=200000 NUM_ENVS=1 bash /workspace/drone/scripts/run_isaac51_oracle_vis.sh

set -euo pipefail

case "${TERM:-}" in
  ""|dumb|unknown|ansi+tabs|ansi) export TERM=xterm ;;
esac

LAB_PATH="${ISAACLAB_MAIN_PATH:-/isaac-sim/IsaacLab-main}"
TRAIN_SCRIPT="${DRONETRANSFER_TRAIN:-/workspace/drone/train_ppo.py}"

TASK="${ISAAC_TASK:-Isaac-Quadcopter-Direct-v0}"
STEPS="${STEPS:-100000}"
NUM_ENVS="${NUM_ENVS:-1}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-}"

export PUBLIC_IP="${PUBLIC_IP:-127.0.0.1}"
export ISAAC_WEBRTC_STYLE="${ISAAC_WEBRTC_STYLE:-legacy}"
export ENABLE_CAMERAS=1

if [[ ! -f "${LAB_PATH}/isaaclab.sh" ]]; then
  echo "[ERROR] Missing ${LAB_PATH}/isaaclab.sh"
  exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] Missing train script: ${TRAIN_SCRIPT}"
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
  if [[ ! -f "${CHECKPOINT}.zip" && ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CHECKPOINT}(.zip)"
    echo "        List available: ls /workspace/drone/checkpoints/oracle/"
    exit 1
  fi
  EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
  echo "[INFO] Resuming from ${CHECKPOINT}"
fi

cd "${LAB_PATH}"
exec ./isaaclab.sh -p "${TRAIN_SCRIPT}" \
  --sim-backend isaac \
  --isaac-task "${TASK}" \
  --mode oracle \
  --steps "${STEPS}" \
  --num-envs "${NUM_ENVS}" \
  --sim-device "${SIM_DEVICE}" \
  --device "${POLICY_DEVICE}" \
  --isaac-vis \
  "${EXTRA_ARGS[@]}"
