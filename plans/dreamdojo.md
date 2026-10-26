# DreamDojo in PWM: notes

Native-action post-training of DreamDojo 2B (no LAM, no aligner), one model per robot.

## Pieces

| what | where |
|---|---|
| environment | `setup/envs/dreamdojo/` (DreamDojo @ 02f119b from source on PYTHONPATH; deps of cosmos-oss + its cu130_torch29 extra, locked in our uv.lock; natten / xformers left out) |
| settings | `configs/world_models.yaml`, `dreamdojo:` (assets, per-robot action slots + packing, training, prediction) |
| runtime glue | `pwm/wm/dreamdojo/runtime.py`: chdir into the source, `checkpoints/DreamDojo` -> `<model_dir>/dreamdojo`, Wan2.1 VAE redirect (the gated Cosmos-Predict2.5-2B tokenizer is the same file), dummy path for gated init weights, LAM replaced by a zero stub |
| weights | `pwm/wm/dreamdojo/assets.py` (`scripts/jobs/prepare_dreamdojo.sh`, CPU job `cluster/sail/prepare_dreamdojo.yaml`): nvidia/DreamDojo 2B_pretrain iter_000140000 into HF_HOME, converted to `<model_dir>/dreamdojo/2B_pretrain/iter_000140000/model_ema_bf16.pt`; Wan VAE symlinked |
| data | `pwm/wm/dreamdojo/data.py`: clips of 13 recorded frames, actions packed per transition and normalised with train-split stats (`<run_dir>/action_stats.json`) |
| trainer | `pwm/wm/dreamdojo/train.py` (`scripts/jobs/train_dreamdojo.sh`, `cluster/sail/train_dreamdojo.yaml`) |
| prediction | `pwm/wm/dreamdojo/predict.py` (`scripts/jobs/eval_dreamdojo.sh`, `cluster/sail/eval_dreamdojo.yaml`) |

## Action slots (configs/world_models.yaml)

| robot | slot | packing |
|---|---|---|
| gr1_robocasa | [0, 29) (NVIDIA's GR-1 slot) | net_delta: command at the end of each 2-step interval minus the one before (NVIDIA's GR-1 data subsamples absolute targets at stride 2 and takes deltas); the first "previous command" is the start state |
| franka_libero | [169, 183) | stack: the 2 EE-delta commands of each interval |
| widowx_bridge | [183, 190) | stack: 1 command |
| google_fractal | [190, 197) | stack: 1 command |

The latent-action slots [352, 384) are zero; DreamDojo's forward multiplies them by the LAM output, so the LAM is a stub.

## Run directory

`step_<n>.pt` (full state: fp32 net, EMA, optimiser, counters, val history; last 2 kept), `ema_bf16_latest.pt` and
`best_ema_bf16.pt` (inference format), `action_stats.json`, `train.log`, and `STOPPED` (json: reason, best step/val,
val history) once the val loss has not improved for 3 consecutive evaluations.

## Facts and gotchas

- fps conditioning is disabled in DreamDojo's action-conditioned nets (`rope_enable_fps_modulation=False`): the fps
  value passed does not matter. (DreamDojo's own inference passes a random 16-32; harmless for the same reason.)
- DreamDojo's sampler (`generate_samples_from_batch`) only handles batch 1 (`latents[0].unsqueeze(0)` in the loop):
  prediction runs one episode at a time per GPU.
- The training noise level is drawn with torch's CPU RNG: validation fixes it with `fork_rng` + `manual_seed` per clip.

## Sweep

Cancelled before any run started and archived 2026-10-26 (`archive/sweeps/`; no results; plans/sweep_report.md). Training uses exactly the
`dreamdojo.train` settings of `configs/world_models.yaml`. Each run's small status files (run_info.json, progress.json,
STOPPED) are mirrored to `<results_dir>/runs/dreamdojo/<run name>/`, readable from the dev box (/checkpoints is not
mounted there).

## Commands

```bash
bash cluster/sail/launch.sh cluster/sail/prepare_dreamdojo.yaml pwm-prepare-dreamdojo
bash cluster/sail/launch.sh cluster/sail/train_dreamdojo.yaml dd-<robot>-nominal --env ROBOT=<robot> \
    --env EXPERIMENT_TAG=dd-<robot>-nominal --env WANDB_RUN_ID=dd-<robot>-nominal --priority p2
bash cluster/sail/launch.sh cluster/sail/eval_dreamdojo.yaml dd-eval-<robot>-<dataset> --env ROBOT=<robot> \
    --env EXTRA_ARGS="--run_dir /checkpoints/<user>/dd-<robot>-nominal --dataset <dataset> --split test"
```

## Log
- 2026-10-24: pivot to 5 h data + ablations. Sweep = configs/sweeps/dreamdojo.yaml (grid lr x effective batch x
  scope on franka; later stages appended automatically by pwm/wm/dreamdojo/sweep_plan.py). Unattended
  orchestrator: `setsid nohup bash cluster/sail/sweep_dreamdojo.sh --gpus 2 --priority p2 >>
  <results_dir>/logs/sweep_dreamdojo/orchestrator.log 2>&1 < /dev/null &` (pid in orchestrator.pid). Run status
  was mirrored to <results_dir>/sweeps/dreamdojo/runs/<tag>/ (now <results_dir>/runs/dreamdojo/); table:
  <results_dir>/sweeps/dreamdojo/<robot>.md. (All of this is archived since 2026-10-26.)
- Env fixes: natten/xformers dropped (no cp312 cu130 wheels), decord from PyPI (cosmos cu130 build is aarch64-only),
  numpy pinned 2.3.4, DreamDojo's undeclared imports added (lightning, moviepy 2.2.1, jsonargparse, ...; static scan).
