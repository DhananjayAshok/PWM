# DreamerV3 world model (third world model, no pretraining)

What it is here: the world-model half of DreamerV3, from the PyTorch port NM512/dreamerv3-torch (MIT, pinned in
`setup/envs/dreamerv3/sources.txt`): CNN encoder/decoder, RSSM (deterministic GRU state + 32x32 discrete stochastic
latents), reward and continue heads. No actor-critic. Trained from scratch per robot on our offline episodes.

## Pieces

| piece | where |
|---|---|
| environment (torch 2.9 cu130, the port on PYTHONPATH) | `setup/envs/dreamerv3/` |
| offline replay: decode once to low res into `derived/dreamerv3-<size>px-<hash>/<split>/shard_*.{npz,json}`, in-memory windows | `pwm/wm/dreamerv3/data.py` |
| config (port defaults + size presets + `configs/world_models.yaml` `dreamerv3`), loss without optimizer step, checkpoint I/O | `pwm/wm/dreamerv3/model.py` |
| trainer (val loss, plateau stop, checkpoints, training-v1 telemetry) | `pwm/wm/dreamerv3/train.py`, `scripts/jobs/train_dreamerv3.sh`, `cluster/sail/train_dreamerv3.yaml` |
| prediction (imagined latents + decoded frames; posterior latents of real frames) | `pwm/wm/dreamerv3/predict.py`, `scripts/jobs/eval_dreamerv3.sh`, `cluster/sail/eval_dreamerv3.yaml` |

## Choices (and why)

- **Inputs:** image only (downscaled whole recorded frame, area-averaged) + the commands into each frame, stacked
  (`action_cmd` of all control steps between two recorded frames: Franka/GR-1 2 x A, WidowX/Google 1 x A). No proprio:
  at evaluation only the start state and the commands exist, and the image carries the start state. Commands are
  standardised with train-split mean/std (fixed in `action_stats.json` at the first start of a run).
- **Reward head = per-step task success** (port's symlog-discrete head); continue head trained on "never terminal".
- **Resolution 64x64** (the original); 128 px (~4x compute) is supported.
- **Size `s100m`** (105.5M params; chosen 2026-10-26 from the grid, plans/sweep_report.md); s12m (13.0M), s25m
  (27.9M, the sweep baseline) and s50m (55.6M) are also defined.
- **Context: 1 real frame** (as DreamDojo's 1 conditioning frame), then open-loop imagination on commands only.
- **Training windows** 32 x 64 frames sampled uniformly over the concatenated split (cross episode boundaries with
  `is_first`, as the port's replay). Mixed precision (port's fp16 AMP). lr 1e-4, the port's other defaults.
- **Selection metric** (picks the best checkpoint and stops the run): open-loop MSE at 64 px of the 4.8 s
  prediction from 1 real frame on 128 fixed val episodes, every 500 steps; stop after 4 evaluations without a 0.2%
  relative improvement (cap 400k steps). The full training objective and its parts on 256 fixed val windows are
  reported alongside. Best weights exported to `<model_dir>/dreamerv3/<run_name>/` with a `meta_*.yaml`.
- **Checkpoints** every 5 min and after every evaluation (`step_*.pt`, tmp + os.replace, last 3 kept, `best.pt`);
  the `step_*.pt` resume states are deleted once the run has stopped.
- **effective MFU** for telemetry: FLOPs of one training step measured with `torch.utils.flop_counter` / wall step time
  / 2.25e15. It is a small model, so expect low MFU; one run per GPU.

## Sweep

Archived 2026-10-26 (sweeps cancelled by the user): the sweep code is in archive/sweeps/, its results stay in
`<storage_dir>/sweeps/dreamerv3/` and `<results_dir>/sweeps/dreamerv3/`. Results and the chosen settings: plans/sweep_report.md. Runs now
train with exactly the `configs/world_models.yaml` `dreamerv3` settings.

## Training runs

The run summary (settings, history, best, GPU-hours) is written to `<storage_dir>/runs/dreamerv3/<robot>/<run_name>.json`
(readable from the dev box); checkpoints go to `/checkpoints/<user>/<run_name>/` and the best weights are exported to
`<model_dir>/dreamerv3/<run_name>/`, where `scripts/jobs/eval_dreamerv3.sh` finds them by default.

```bash
for r in franka_libero gr1_robocasa widowx_bridge google_fractal; do
  bash cluster/sail/launch.sh cluster/sail/train_dreamerv3.yaml dv3-$r-nominal \
      --env ROBOT=$r --env EXPERIMENT_TAG=dv3-$r-nominal --env WANDB_RUN_ID=dv3-$r-nominal --priority p2
done
```

## Log

- 2026-10-24: env built (4.8 GB tar). CPU test of cache -> replay -> loss -> optimizer step -> imagine/encode passes.
- 2026-10-24 06:44-07:05: GPU smoke (job 35275, smoke data): sweep kill -9 at 240 s + resume OK, plateau stop, export,
  predict (contract layout) and encode OK, table written. Throughput (1 B300, s25m, 64 px, 64-frame windows):
  batch 32 0.60 s/step, batch 128 0.68, batch 256 0.76 (latency-bound Python RSSM loop, MFU < 1%); 4 runs packed on one
  GPU keep 0.6-0.8 s/step each, so packing ~4x the work per GPU.
- 2026-10-24 18:25: grid launched on nominal 5 h data, 4 robots x 11 runs packed 4 per GPU (jobs 35313-35325,
  orchestrator cluster/sail/sweep_dreamerv3.sh --group grid --assume_ready 1). Generation stopped at the hours target,
  far below the planned unit counts, so data readiness is asserted (--assume_ready). Val splits are small (Franka 51,
  GR-1 25, WidowX 7, Google 10 episodes): the selection metric uses all of them (not 128), so it is noisy on WidowX
  and Google. First evaluation, Franka step 500: openloop_mse64 s12m 0.0081, s25m 0.0065, s50m 0.0064, s100m 0.0062.
