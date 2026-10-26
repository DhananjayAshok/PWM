# V-JEPA 2-AC in PWM

Code: `pwm/wm/vjepa2/` (env `vjepa2`, settings `configs/world_models.yaml: vjepa2`). Upstream:
facebookresearch/vjepa2 @ 204698b (src/ and app/ on PYTHONPATH, nothing patched).

## Model
- Meta's released V-JEPA 2-AC (`vjepa2-ac-vitg.pt`, 11.8 GB, downloaded once to `<model_dir>/vjepa2/`): ViT-g/16 encoder
  (frozen; 256 tokens x 1408 per frame, each frame encoded alone as a repeated 2-frame tubelet) and a 24-layer,
  1024-wide frame-causal predictor (~300M params, trained). Built exactly as `src/hub/backbones.py`
  `_make_vjepa2_ac_model` (the hub entry itself points at `http://localhost:8300` on main, so we load the file directly).
- Objective as the release (`app/vjepa_droid/train.py`): L1 on layer-normed features, one teacher-forced pass over the
  8-step window + a 2-step autoregressive rollout from its first frame.
- **Action / state projections** (release: 7-D DROID pose deltas + gripper; 7-D state). A model step stacks c commands:
  Franka/GR-1 c = 4 (2 commands per recorded frame x stride 2), WidowX/Google c = 1.
  - 7-D command robots: action projection = released weights tiled /c (the projection of the mean command) -> Franka
    28-D, WidowX/Google 7-D (unchanged). **GR-1 (29-D joints, 116-D stacked): fresh init.**
  - State projection: kept only if the robot's state is 7-D, else fresh (Franka's state is 8-D -> fresh).
- **State input = the start state of the window, repeated** (commands-only rule: no measured future state). The
  release fed the measured state of every frame.

## Data
- Model step = `frame_stride` recorded frames: 5 fps for Franka/GR-1 (stride 2 of 10 fps), 5 fps WidowX, 3 fps Google
  (release: 4 fps).
- Frame cache, not a feature cache: `<dataset>/derived/vjepa2-<hash>/frames/<split>/<episode>.npy`, uint8 256x256 per
  model step, built by claim-based workers (`python -m pwm.wm.vjepa2.data build-cache`, restart-safe; the training
  job runs 16 in parallel first). Encoder features are computed on the fly. Reason: a full feature cache would be
  ~390 GB per robot at 30 h (256 x 1408 fp16 per frame) and the training loader would read ~90 MB/s per GPU from CephFS;
  frames are ~100 GB per robot and the frozen encoder costs ~35% of a step.

## Training (`scripts/jobs/train_vjepa2.sh`, `cluster/sail/train_vjepa2.yaml`)
- Settings: `configs/world_models.yaml: vjepa2` (the robot's entry merged in); `--override key=value` for quick tests.
  One run per robot and dataset.
- AdamW lr 2e-4 (constant after 300 warm-up steps), wd 0.04, clip 1.0, bf16 autocast, activation checkpointing,
  batch 16 per GPU; DDP over all GPUs of the job.
- **Selection number: `val/loss_ar`** = L1 on layer-normed features of a 7-step free-running rollout from the first frame
  of fixed 8-step val windows (<= 512); `val/loss_tf` (teacher-forced) reported separately. Constants in
  `pwm/wm/vjepa2/train.py` (VAL_*), not configurable. Every 500 steps; stop after 3 evaluations without a 0.1%
  improvement (`STOPPED.json`, `best.pt` = best predictor). Safety cap 200k steps.
- Afterwards `rollout_eval.py` runs on best.pt (not used for selection): 4.8 s free-running rollout L1/L2 on 32 fixed val
  episodes from 1 context frame; action sensitivity = error with commands of another episode / error with the true ones.
- Checkpoints every 10 min (background write of a CPU copy, tmp + rename, newest 2 kept); resume from the newest
  readable one; the data stream is a deterministic function of (seed, rank, worker, resume step). Resume states are
  deleted once the run stops.
- Run directory `<storage_dir>/checkpoints/vjepa2/<robot>/<dataset>/` (on /persist, so the dev box and the panel can read
  it): `HEARTBEAT` (a second process on the same run exits), `progress.json` (history, readable without torch),
  `STOPPED.json`, `rollout_eval.json`, `best.pt`, `step_*.pt`; the job log ends with `STOPPED <dir>`.
- training-v1 telemetry from rank 0 (W&B project `pwm-vjepa2`).

## Sweep (archived)
The hyperparameter grid (lr, predictor init, rollout loss, depth, batch; 9 runs x 4 robots) and its orchestrator were
archived on 2026-10-26 in `archive/sweeps/`; results and the chosen settings: plans/sweep_report.md. The sweep's run directories stay in
`<storage_dir>/checkpoints/vjepa2/<robot>/nominal-<run>/`.

## Entry points (plans/predictions_format.md; `scripts/jobs/eval_vjepa2.sh`, `cluster/sail/eval_vjepa2.yaml`)
- `predict`: frame 0 + commands of the first 4.8 s, free-running; pred_features (K, 16, 1408) float16 (4x4-pooled
  token grid; `--pool none` keeps all 256), frame_index = recorded frames s, 2s, ... (s = frame_stride).
- `encode`: frozen-encoder features of real frames, same shape, tag `frozen-encoder` (identical for every predictor).

## Per-robot commands
```bash
# train (more GPUs: --gpus B300:8 --cpus 176 --memory 2000)
bash cluster/sail/launch.sh cluster/sail/train_vjepa2.yaml vj2-franka-libero-nominal --priority p2 \
    --env ROBOT=franka_libero --env EXPERIMENT_TAG=vj2-franka-libero-nominal --env WANDB_RUN_ID=vj2-franka-libero-nominal
# predictions on a test set / real features
bash cluster/sail/launch.sh cluster/sail/eval_vjepa2.yaml vj2-pred-franka-nominal --env ROBOT=franka_libero \
    --env DATASET=nominal --env SPLIT=test --env MODE=predict \
    --env CHECKPOINT=<storage_dir>/checkpoints/vjepa2/franka_libero/nominal/best.pt
bash cluster/sail/launch.sh cluster/sail/eval_vjepa2.yaml vj2-enc-franka-nominal --env ROBOT=franka_libero \
    --env DATASET=nominal --env SPLIT=train --env MODE=encode --env MAX_EPISODES=2000
```

## Log
- 2026-10-24: env built (4.9 GB tar), checkpoint downloaded, smoke dataset `franka_libero/smoke` (old Part-3 episodes
  converted; shared with the DreamerV3 worker, who added more episodes).
- Bugs found on GPU: telemetry config duplicate `robot` key (crash at start; the job then lingered ~11 h RUNNING, 2 GPUs
  wasted -> reaper added); training window past the cached frames (n_frames now capped by the cache); wandb `get_url`.
- 2026-10-24 ~19:00 UTC: 4 baselines (35330, 35332-35334) + 12 grid packs (35335-35346) queued at p2 on the 5 h data;
  orchestrator started.
