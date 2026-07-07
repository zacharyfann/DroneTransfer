#!/usr/bin/env bash
# Repair pip / Isaac Lab deps in Isaac Sim 6.0 container WITHOUT isaaclab.sh -i
# (isaaclab.sh -i downgrades torch and fails on corp SSL proxies).
#
# Run as root inside isaac-sim:
#   docker exec -u root -it isaac-sim bash
#   bash /workspace/drone/scripts/fix_isaac_pip.sh

set -euo pipefail

PYTHON="/isaac-sim/kit/python/bin/python3"
PIP_SITE="/isaac-sim/kit/python/lib/python3.12/site-packages"
LAB="/isaac-sim/IsaacLab"

# Corporate proxy / self-signed cert workaround for pip
export PIP_TRUSTED_HOST="pypi.org,files.pythonhosted.org,download.pytorch.org,download-r2.pytorch.org,pypi.nvidia.com,py.mujoco.org"
export PIP_DISABLE_PIP_VERSION_CHECK=1

pip_install() {
  "$PYTHON" -m pip install "$@" \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host download.pytorch.org \
    --trusted-host download-r2.pytorch.org \
    --trusted-host pypi.nvidia.com \
    --trusted-host py.mujoco.org
}

echo "=== Step 1: Repair pip (if needed) ==="
if ! "$PYTHON" -m pip --version 2>/dev/null; then
  rm -rf "${PIP_SITE}/pip" "${PIP_SITE}/pip-"*.dist-info 2>/dev/null || true
  "$PYTHON" -m ensurepip --upgrade
fi
"$PYTHON" -m pip --version

echo ""
echo "=== Step 2: Restore torch (DO NOT run isaaclab.sh -i) ==="
if ! "$PYTHON" -c "import torch; print('torch', torch.__version__)" 2>/dev/null; then
  echo "torch missing — reinstalling (prefer Isaac Sim bundled version)..."
  # Try NVIDIA index first (often works on locked-down networks)
  if ! pip_install torch torchvision --index-url https://pypi.nvidia.com 2>/dev/null; then
    # Fallback: PyPI / PyTorch wheels with trusted-host
    pip_install torch torchvision || \
      pip_install "torch>=2.7" "torchvision>=0.20"
  fi
fi
"$PYTHON" -c "import torch; print('torch OK:', torch.__version__)"

echo ""
echo "=== Step 3: Hide IsaacLab-main (wrong branch) ==="
if [[ -d /isaac-sim/IsaacLab-main ]]; then
  mv /isaac-sim/IsaacLab-main /isaac-sim/IsaacLab-main.disabled 2>/dev/null || true
fi
pip_install --quiet -e "${LAB}/source/isaaclab" 2>/dev/null || true

echo ""
echo "=== Step 4: Manual editable install (--no-deps; Isaac Sim has warp/torch) ==="
cd "${LAB}"
rm -rf source/isaaclab_rl/isaaclab_rl.egg-info \
       source/isaaclab/isaaclab.egg-info \
       source/isaaclab_tasks/isaaclab_tasks.egg-info \
       source/isaaclab_ovphysx/isaaclab_ovphysx.egg-info \
       source/isaaclab_assets/isaaclab_assets.egg-info 2>/dev/null || true

# develop pins warp-lang==1.15.0.dev* which is NOT on public PyPI — skip deps
pip_install -e source/isaaclab --no-deps
pip_install -e source/isaaclab_assets --no-deps
pip_install -e source/isaaclab_tasks --no-deps
pip_install -e source/isaaclab_ovphysx --no-deps
pip_install -e source/isaaclab_rl --no-deps

echo ""
echo "=== Step 5: Training deps (install SB3 separately; skip rl[sb3] extra) ==="
pip_install lazy_loader stable-baselines3 tqdm rich tensorboard

echo ""
echo "=== Step 6: Verify ==="
"$PYTHON" -c "import isaaclab; import lazy_loader; import isaaclab_rl; print('isaaclab_rl:', isaaclab_rl.__file__)"
"$PYTHON" -c "from isaaclab_ovphysx.physics import OvPhysxCfg; print('OvPhysX import OK')" || \
  echo "[WARN] OvPhysxCfg import failed — ovphysx wheel may still be missing"

echo ""
echo "=== Done. Train with: ==="
cat <<'EOF'
cd /isaac-sim/IsaacLab
./isaaclab.sh -p /workspace/drone/train_ppo.py \
  --sim-backend isaac \
  --isaac-task Isaac-Cartpole-Direct \
  --mode oracle \
  --steps 50000 \
  --device cuda
EOF
