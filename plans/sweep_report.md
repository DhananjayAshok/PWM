# Sweep report: how the standard training settings were chosen

Written 2026-10-26, when the sweeps were stopped. This is the single record of the hyperparameter sweeps and training
tricks tried on the three world models, and of the settings now in `configs/world_models.yaml`, which every final run
trains with. The sweep code is archived in `archive/sweeps/` (see its README); the raw results stay where the runs wrote
them (DreamerV3 summaries: `<storage_dir>/sweeps/dreamerv3/<robot>/`; V-JEPA run directories:
`<storage_dir>/checkpoints/vjepa2/<robot>/nominal-<run>/`).

## Chosen settings

| Model | Setting | Value | Why |
|---|---|---|---|
| DreamerV3 | `size` | **s100m** (105M params; was s25m) | Better than the baseline on 3 of 4 robots (-4% Franka, -6% WidowX (-8% as a 3-seed mean), -1% Google, +3% GR-1). It costs no more GPU time: every grid run used under about 1.1 GPU-hours, because the RSSM's time loop is launch-bound |
| DreamerV3 | everything else | unchanged: lr 1e-4, batch 32, sequence 64, 64 px, KL defaults | No other change beat the baseline on most robots. A lower lr, a smaller model and longer sequences were clearly worse |
| V-JEPA 2-AC | all | unchanged: the release's recipe (release predictor, depth 24, TF + 2-step rollout loss, lr 2e-4, batch 16) | A fresh predictor and teacher-forcing-only training were worse. Every other change was within about ±1% of the base on the robots where the base finished, with no seed replicates to show it is more than noise |
| DreamDojo | all | unchanged: lr 4e-5, effective batch 96, full fine-tune | The sweep was cancelled before any run started: these are untested defaults |

No training tricks are used (user decision, 2026-10-26).

## Protocol (common to all sweeps)

- Data: 5 h of nominal data per robot; fixed train / val split (val about 50 episodes per robot).
- Selection metric, fixed up front and identical for every run of a model:
  - DreamerV3: `val/openloop_mse64`, the 64 px image MSE of a 4.8 s open-loop rollout from 1 real frame, on 128 fixed val episodes.
  - V-JEPA: `val/loss_ar`, the L1 on layer-normed features of a 7-step free-running rollout from the first frame of up to 512 fixed val windows.
  - DreamDojo: the flow-matching val loss on 192 fixed val clips with fixed noise (never measured).
- Early stopping: a run stops after 4 (DreamerV3) or 3 (V-JEPA) evaluations without improving the metric by the minimum
  amount; its best value is reported. Lower is better everywhere.
- Grids changed one factor at a time around a baseline, at seed 0. The two best grid runs of a robot were to get seeds 1
  and 2 so differences could be compared with seed noise (done for DreamerV3 GR-1 and WidowX only).

## DreamerV3 grid

Best `val/openloop_mse64`, with the change against the baseline (s25m, 64 px, sequence 64, lr 1e-4, batch 32, default
KL). `*` = unfinished when the sweep stopped (Google's last pack).

| Run | Franka | WidowX | Google | GR-1 |
|---|---|---|---|---|
| baseline | 0.00345 | 0.01100 | 0.00631 | 0.05534 |
| size s12m | 0.00405 (+18%) | 0.01283 (+17%) | 0.00641 (+2%) | 0.05741 (+4%) |
| size s50m | 0.00344 (-0%) | 0.01112 (+1%) | 0.00655 (+4%) | 0.05667 (+2%) |
| **size s100m** | **0.00329 (-4%)** | **0.01034 (-6%)** | **0.00627 (-1%)** | 0.05674 (+3%) |
| 128 px input | 0.00351 (+2%) | 0.01114 (+1%) | 0.00682 (+8%)* | 0.05887 (+6%) |
| sequence 128 | 0.00392 (+14%) | 0.01195 (+9%) | 0.00715 (+13%)* | 0.05949 (+8%) |
| lr 3e-5 | 0.00449 (+30%) | 0.01223 (+11%) | 0.00759 (+20%)* | 0.06661 (+20%) |
| lr 3e-4 | 0.00411 (+19%) | 0.01079 (-2%) | 0.00651 (+3%)* | 0.05073 (-8%) |
| free bits 0.1 | 0.00384 (+11%) | 0.01127 (+2%) | 0.00649 (+3%) | 0.05534 (+0%) |
| KL balance 0.8 / 0.2 | 0.00371 (+8%) | 0.01163 (+6%) | 0.00725 (+15%) | 0.05590 (+1%) |
| batch 128 | 0.00321 (-7%) | 0.01099 (-0%) | 0.00622 (-1%) | 0.05920 (+7%) |

Seed replicates (seed 0 is the grid run itself):

| Robot | Run | Seeds 0 / 1 / 2 | Mean |
|---|---|---|---|
| GR-1 | lr 3e-4 (grid best) | 0.0507 / 0.0607 / 0.0580 | 0.0565 |
| GR-1 | baseline (grid second) | 0.0553 / 0.0541 / 0.0555 | 0.0550 |
| WidowX | size s100m (grid best) | 0.0103 / 0.0099 / 0.0103 | 0.0102 |
| WidowX | lr 3e-4 (grid second) | 0.0108 / 0.0099 / 0.0101 | 0.0103 |

Reading:
- GR-1's grid "winner" (lr 3e-4) was a lucky seed: over 3 seeds it is worse than the baseline, and its seeds spread
  ±9%, against ±1% for the baseline. A higher lr made results less repeatable.
- Seed noise is about ±1-2% for stable settings: differences of a few percent from a single seed are not evidence.
- On GR-1, free bits 0.1 gave exactly the baseline's numbers: its KL (9-23 nats) never reached either floor, so
  free bits never acted. It also shows the runs are deterministic for a seed.
- Franka and Google seed replicates were cancelled before they ran.

## V-JEPA 2-AC grid

Best `val/loss_ar`, with the change against `base` (release predictor, depth 24, TF + 2-step rollout loss, lr 2e-4,
batch 16 per GPU). `*` = cancelled before it plateaued (evictions and the cluster being full left half the grid
unfinished). Franka's base was evicted at step 2,252 and never finished, so Franka's percentages are against an
unfinished base: compare Franka's runs with each other instead.

| Run | Franka | WidowX | Google | GR-1 |
|---|---|---|---|---|
| base | 0.2289* | 0.2121 | 0.3110 | 0.4042 |
| lr 1e-4 | 0.2215 | 0.2175 (+2.6%)* | 0.3090 (-0.6%)* | 0.4058 (+0.4%)* |
| lr 4e-4 | 0.2218 | 0.2157 (+1.7%)* | 0.3124 (+0.5%) | 0.4093 (+1.3%)* |
| fresh predictor | 0.2327 | 0.2516 (+18.6%)* | 0.3277 (+5.4%)* | 0.4293 (+6.2%)* |
| TF only (no rollout loss) | 0.2267 | 0.2165 (+2.1%) | 0.3179 (+2.2%) | 0.4112 (+1.7%) |
| TF + 4-step rollout | 0.2203* | 0.2105 (-0.7%) | 0.3080 (-0.9%)* | 0.4068 (+0.6%)* |
| TF + 6-step rollout | 0.2219* | 0.2106 (-0.7%)* | 0.3088 (-0.7%)* | 0.4037 (-0.1%)* |
| depth 12 | 0.2285 | 0.2154 (+1.5%) | 0.3148 (+1.2%) | 0.4066 (+0.6%)* |
| batch 32 | 0.2210 | 0.2102 (-0.9%) | 0.3104 (-0.2%) | 0.4032 (-0.2%)* |

Reading:
- The released predictor matters most: a random one is 5-19% worse on every robot where it trained long enough.
- Training through a rollout helps a little: teacher forcing alone is about 2% worse on 3 of 4 robots. Longer rollouts
  (4 or 6 steps) give nothing reliable over 2.
- Learning rate, batch 32 and depth 12 are within about ±1-2%. Batch 32 is never worse, but by under 1% and from one
  seed. Depth 12 is about 1% worse at half the predictor's size, which matters only if compute becomes the limit.
- No V-JEPA seed replicates ran, so its seed noise is unknown.

## DreamDojo

A 12-point grid on Franka (lr {1e-5, 4e-5, 1e-4} × effective batch {32, 96} × {full, LoRA rank 64}) was planned, then
the best point on the other three robots. The cluster was full for most of that time and the user cancelled the sweep
(2026-10-26) before any run was admitted. No results.

## Training tricks (dropped)

A vetted list of 2025+ top-venue training tricks was compiled
(`archive/sweeps/plans/literature_tricks.md`); only DreamerV3 ran them, on GR-1 and WidowX, 3 seeds each:
- Input random-shift augmentation (±4 px) was the only clear effect: -13% on GR-1 (all 3 seeds below the best grid
  seed), no effect on WidowX.
- AC-CPC auxiliary loss, SOAP, AdEMAMix, LaProp and a WSD cooldown (against a matched constant-LR control) showed
  nothing reliable on either robot.
The user dropped all tricks on 2026-10-26 because none worked reliably; the code was deleted.

## What went wrong (and limits confidence)

- The shared Ceph pool filled on 2026-10-25 (~05:09 UTC). Checkpoint writes failed silently for hours, and the first
  V-JEPA base runs lost their checkpoints and were rerun.
- Deleting the managed Python interpreters during a storage cleanup broke the job environments for a while
  (`No module named timm`); the affected runs were relaunched.
- Duplicate job packs (an unreliable queue read) and evictions of borrowed GPUs cost time. One pair of duplicates
  wrote over each other's checkpoints, though the runs' results survived.
- The cluster stayed full (0 of 56 GPUs free) for most of 2026-10-26, so V-JEPA finished only half its grid and
  DreamDojo none of it.

## Open questions for later

- DreamDojo's settings are untested: if its final runs look poor, a small lr check (1e-5 / 4e-5 / 1e-4) is the first
  thing to try.
- V-JEPA batch 32 and a 4-step rollout are the only candidates worth a seed-replicated check.
- DreamerV3 s100m is slightly worse on GR-1 (+3%, above GR-1's baseline seed spread of about ±1%); GR-1 may prefer
  the s25m default.
