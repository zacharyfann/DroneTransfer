#!/usr/bin/env bash
# Setup Isaac Lab v2.3.0 for Isaac Sim 5.1 (DroneTransfer training).
#
# IMPORTANT: Do NOT use Isaac Lab `main` on Isaac Sim 5.1 — main requires
# isaacsim.asset.importer.urdf 2.4.31 while 5.1.0 bundles 2.4.30.
#
# Run INSIDE isaac-sim-51 as root:
#   docker exec -u root -it -e TERM=xterm isaac-sim-51 bash
#   bash /workspace/drone/scripts/setup_isaaclab_51_main.sh
#
# If a previous run upgraded torch / missed deps, run the repair script instead:
#   bash /workspace/drone/scripts/fix_isaac51_pip.sh

set -euo pipefail

PYTHON="${ISAACSIM_PYTHON_EXE:-/isaac-sim/python.sh}"
DEFAULT_LAB="/isaac-sim/IsaacLab-main"
LAB_GIT_REF="${ISAACLAB_GIT_REF:-v2.3.0}"

find_lab_path() {
  local candidates=(
    "${ISAACLAB_MAIN_PATH:-}"
    "/isaac-sim/IsaacLab-main"
    "/isaac-sim/IsaacLab-v2.3.0"
    "/isaac-sim/IsaacLab"
    "/tmp/c-zfann/IsaacLab-v2.3.0"
    "/tmp/c-zfann/IsaacLab-main"
    "/tmp/IsaacLab-main"
  )
  for p in "${candidates[@]}"; do
    [[ -n "${p}" && -f "${p}/isaaclab.sh" ]] && { echo "${p}"; return 0; }
  done
  return 1
}

ensure_lab_repo() {
  if LAB_PATH="$(find_lab_path)"; then
    echo "[INFO] Using Isaac Lab at: ${LAB_PATH}"
    if [[ -d "${LAB_PATH}/.git" ]]; then
      local tag
      tag="$(git -C "${LAB_PATH}" describe --tags --exact-match 2>/dev/null || true)"
      if [[ -n "${tag}" && "${tag}" != "${LAB_GIT_REF}" ]]; then
        echo "[WARN] Repo tag is ${tag}, expected ${LAB_GIT_REF} for Isaac Sim 5.1"
      fi
    fi
    return 0
  fi

  echo "[INFO] Isaac Lab not found — cloning ${LAB_GIT_REF} to ${DEFAULT_LAB}"
  if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] git not found. Copy repo from host:"
    echo "  docker cp /tmp/c-zfann/IsaacLab-v2.3.0/. isaac-sim-51:${DEFAULT_LAB}/"
    exit 1
  fi

  mkdir -p "$(dirname "${DEFAULT_LAB}")"
  rm -rf "${DEFAULT_LAB}"
  git clone --branch "${LAB_GIT_REF}" --depth 1 https://github.com/isaac-sim/IsaacLab.git "${DEFAULT_LAB}"
  LAB_PATH="${DEFAULT_LAB}"
}

pip_install() {
  "${PYTHON}" -m pip install "$@" \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.nvidia.com \
    --trusted-host download.pytorch.org \
    --trusted-host download-r2.pytorch.org
}

bootstrap_pip_tooling() {
  echo "[INFO] Bootstrapping pip build tools (Isaac Sim setuptools 82+ breaks old sdist builds)"
  pip_install "setuptools<82" wheel
}

install_flatdict() {
  echo "[INFO] Installing flatdict==4.0.1 (sdist needs setuptools pkg_resources)"
  if ! pip_install "flatdict==4.0.1" --only-binary flatdict 2>/dev/null; then
    pip_install "flatdict==4.0.1" --no-build-isolation
  fi
}

restore_isaac_torch() {
  echo "[INFO] Restoring Isaac Sim torch 2.7.0+cu128 (never leave SB3 on torch 2.12+)"
  pip_install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
  "${PYTHON}" -c "import torch; v=torch.__version__; assert v.startswith('2.7'), v; print('torch OK:', v)"
}

install_editable_no_deps() {
  local pkg_dir="$1"
  if [[ -d "${pkg_dir}" ]]; then
    echo "  installing ${pkg_dir}"
    pip_install -e "${pkg_dir}" --no-deps
  else
    echo "  [skip] missing ${pkg_dir}"
  fi
}

install_isaaclab_python_deps() {
  echo "[INFO] Installing Isaac Lab v2.3.0 Python deps (from source/isaaclab/setup.py)"
  bootstrap_pip_tooling
  install_flatdict
  # Matches v2.3.0 install_requires — omit torch/numpy (bundled with Isaac Sim).
  pip_install \
    "gymnasium==1.2.0" \
    "prettytable==3.3.0" \
    "hidapi==0.14.0.post2" \
    "pillow==11.3.0" \
    "starlette==0.45.3" \
    "packaging<24" \
    "onnx>=1.18.0" \
    "pyglet<2" \
    warp-lang \
    einops \
    transformers \
    trimesh \
    toml \
    flaky \
    junitparser \
    pytest \
    pytest-mock \
    "pin-pink==3.1.0" \
    "dex-retargeting==0.4.6" \
    hydra-core \
    h5py \
    moviepy
}

install_drone_transfer_deps() {
  echo "[INFO] Installing DroneTransfer / SB3 deps (--no-deps on SB3 to protect torch)"
  pip_install cloudpickle pandas matplotlib tqdm rich tensorboard lazy_loader
  pip_install stable-baselines3 --no-deps
}

verify_install() {
  echo ""
  echo "=== Verify ==="
  "${PYTHON}" -c "import warp; import gymnasium; print('gymnasium', gymnasium.__version__, 'OK')"
  "${PYTHON}" -c "import prettytable; import stable_baselines3; print('sb3 OK')"
  "${PYTHON}" -c "import isaaclab, isaaclab_rl; print('isaaclab', isaaclab.__version__, 'OK')"
  if "${PYTHON}" -c "import isaaclab_tasks" 2>/dev/null; then
    echo "isaaclab_tasks OK"
  else
    echo "[WARN] isaaclab_tasks pre-AppLauncher import failed (pxr) — training should still work"
  fi
}

ensure_lab_repo
LAB_PATH="${LAB_PATH:-${DEFAULT_LAB}}"

echo ""
echo "=== Step 1: Link Isaac Lab to bundled Isaac Sim ==="
cd "${LAB_PATH}"
ln -sf /isaac-sim _isaac_sim 2>/dev/null || true
export ISAACSIM_PATH=/isaac-sim
export ISAACSIM_PYTHON_EXE="${PYTHON}"

echo ""
echo "=== Step 2: pip sanity ==="
"${PYTHON}" -m pip --version

echo ""
echo "=== Step 3: Editable install (--no-deps) ==="
rm -rf source/isaaclab*/isaaclab*.egg-info 2>/dev/null || true
install_editable_no_deps source/isaaclab
install_editable_no_deps source/isaaclab_assets
install_editable_no_deps source/isaaclab_tasks
install_editable_no_deps source/isaaclab_rl
install_editable_no_deps source/isaaclab_visualizers

echo ""
echo "=== Step 4: Python runtime deps ==="
install_isaaclab_python_deps
install_drone_transfer_deps

echo ""
echo "=== Step 5: Restore Isaac Sim torch (always last) ==="
restore_isaac_torch

verify_install

echo ""
echo "Setup complete. LAB_PATH=${LAB_PATH} (ref ${LAB_GIT_REF})"
echo "  export ISAACLAB_MAIN_PATH=${LAB_PATH}"
