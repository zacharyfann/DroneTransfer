# Isaac Sim 5.1 Visual Drone Training (Oracle MLP)

This runbook sets up **Isaac Sim 5.1 + Isaac Lab v2.3.0** to run `train_ppo.py` with:

- `--sim-backend isaac`
- drone task (default: `Isaac-Quadcopter-Direct-v0`)
- `--mode oracle` (`[goal, vel] -> action` MLP)
- WebRTC livestream (`--isaac-vis`)

---

## 0) Why this path

On this server, Isaac Sim 6.0 is fine for headless Cartpole smoke tests but has unstable livestream behavior during training.
Isaac Sim 5.1 (`isaac-sim-51`) already proved WebRTC rendering works for drone scenes, so this is the best base for visualized drone training.

---

## 1) Copy DroneTransfer files into `isaac-sim-51`

The 5.1 container does **not** mount `/workspace/drone` by default. Copy via `/tmp`:

```bash
# on host (after scp from Windows to ~/)
docker start isaac-sim-51
docker cp ~/train_ppo.py isaac-sim-51:/tmp/train_ppo.py
docker cp ~/isaac_nav_env.py isaac-sim-51:/tmp/isaac_nav_env.py
docker cp ~/setup_isaaclab_51_main.sh isaac-sim-51:/tmp/setup_isaaclab_51_main.sh
docker cp ~/run_isaac51_oracle_vis.sh isaac-sim-51:/tmp/run_isaac51_oracle_vis.sh

docker exec -u root isaac-sim-51 bash -lc '
mkdir -p /workspace/drone/scripts
cp /tmp/train_ppo.py /workspace/drone/
cp /tmp/isaac_nav_env.py /workspace/drone/
cp /tmp/setup_isaaclab_51_main.sh /workspace/drone/scripts/
cp /tmp/run_isaac51_oracle_vis.sh /workspace/drone/scripts/
chmod +x /workspace/drone/scripts/*.sh
'
```

## 2) Install Isaac Lab v2.3.0 inside `isaac-sim-51`

**Do not use Isaac Lab `main` on Isaac Sim 5.1.** Main requires URDF importer 2.4.31; the 5.1 container bundles 2.4.30 and cannot reach NVIDIA extension registries.

`isaac-sim-51` also does **not** mount Isaac Lab. Either clone v2.3.0 inside the container or copy from host.

**Option A — clone v2.3.0 inside container (recommended):**

```bash
docker exec -u root isaac-sim-51 bash -lc '
rm -rf /isaac-sim/IsaacLab-main
git clone --branch v2.3.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git /isaac-sim/IsaacLab-main
'
docker exec -u root -it isaac-sim-51 bash
bash /workspace/drone/scripts/setup_isaaclab_51_main.sh
```

**Option B — copy from host:**

```bash
cd /tmp/c-zfann
rm -rf IsaacLab-v2.3.0
git clone --branch v2.3.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git IsaacLab-v2.3.0
docker exec -u root isaac-sim-51 rm -rf /isaac-sim/IsaacLab-main
docker cp IsaacLab-v2.3.0/. isaac-sim-51:/isaac-sim/IsaacLab-main/
docker exec -u root -it isaac-sim-51 bash -lc '
export ISAACLAB_MAIN_PATH=/isaac-sim/IsaacLab-main
bash /workspace/drone/scripts/setup_isaaclab_51_main.sh
'
```

This installs editable Isaac Lab packages plus SB3/TensorBoard deps.

---

## 3) Launch visualized training

Inside the same container shell:

```bash
bash /workspace/drone/scripts/run_isaac51_oracle_vis.sh
```

Default runtime values:

- `TASK=Isaac-Quadcopter-Direct-v0`
- `STEPS=100000`
- `NUM_ENVS=8`
- `SIM_DEVICE=cuda:0`
- `POLICY_DEVICE=cuda`

To override:

```bash
TASK=Isaac-Quadcopter-Direct-v0 STEPS=200000 NUM_ENVS=4 \
bash /workspace/drone/scripts/run_isaac51_oracle_vis.sh
```

---

## 4) Connect WebRTC client from Windows

Create tunnel:

```powershell
ssh -N -L 49100:127.0.0.1:49100 -L 47998:127.0.0.1:47998 c-zfann@ambus-algoq8000
```

In NVIDIA Isaac WebRTC client:

- Host: `127.0.0.1`
- Port: `49100`

If you see connection but black screen, reduce `NUM_ENVS` to `1-4`.

---

## 5) Important operational notes

- Run only **one** livestreaming sim container at a time.
  - Stop any conflicting container (for example, another sim using WebRTC).
- Keep `NUM_ENVS` low when visualizing (`1-16` recommended).
- `--steps` is in **environment timesteps** and PPO collects rollout chunks; expect some overshoot if rollout size > remaining target.
- For maximum training throughput, run without `--isaac-vis`.

---

## 6) Expected success signal

In training logs, you should see:

- simulation app startup complete
- PPO logging starts
- timesteps increasing

In WebRTC client, you should see:

- Isaac scene with quadcopter task viewport
- motion updates while training is running

---

## 7) Troubleshooting quick checks

Inside container:

```bash
/isaac-sim/python.sh -c "import isaaclab, isaaclab_tasks, isaaclab_rl, isaaclab_visualizers; print('imports OK')"
```

On host:

```bash
docker ps | grep isaac-sim
ss -tlnp | grep 49100
```

If 49100 is already occupied by another sim, stop the other one first.
