# PWM

World models for predicting robot policy behaviour under deployment shifts, across four robots in simulation:

| robot | simulator | policy |
|---|---|---|
| `franka_libero` | LIBERO (MuJoCo) | pi0.5 (LeRobot) |
| `gr1_robocasa` | RoboCasa GR-1 tabletop (MuJoCo) | GR00T N1.6 |
| `widowx_bridge` | SimplerEnv Bridge (SAPIEN) | GR00T N1.7 |
| `google_fractal` | SimplerEnv fractal (SAPIEN) | GR00T N1.7 |

Status: step 1 (environments, and checking that each policy reproduces its published success in them).

## Setup

See [setup/README.md](setup/README.md). In short: fill `configs/private_vars.yaml`, set `HF_HOME`, run
`bash setup/setup.sh all`. Each simulator/policy stack has its own environment ([setup/envs/README.md](setup/envs/README.md)).

## Acceptance

```bash
bash scripts/jobs/verify_robot.sh --robot widowx_bridge    # on a GPU machine
```
runs the robot's policy on its published tasks and writes `<results_dir>/verification/widowx_bridge.{md,json}`
(our success per task vs published, within 2 binomial standard errors). On the Sail cluster: [cluster/sail](cluster/sail/README.md).

## Layout

| path | contents |
|---|---|
| `configs/` | parameters; `robots.yaml` is the robot registry ([README](configs/README.md)) |
| `setup/` | core environment (`pyproject.toml`), `setup.sh`, per-environment definitions in `setup/envs/` |
| `pwm/sim/` | rollout clients that run inside a simulator environment; shared episode format (`episode.py`) |
| `pwm/policies/` | serving policies that run in their own environment (GR00T) |
| `pwm/verify.py` | acceptance record |
| `pwm/utils/` | parameters, logging, meta/hash records, plotting, statistics |
| `scripts/` | `utils.sh` (shared bash helpers and every script's defaults), `get_strings.py`, `jobs/` (runnable jobs) |
| `cluster/sail/` | Sail-specific job wrappers and node preparation |

Episode records (`pwm/sim/episode.py`): per episode an npz of per-step arrays (`action_env`, `action_policy`,
`state_policy`, `success`, plus robot-specific simulator state) and an mp4 of the policy camera; `episodes.jsonl` indexes
the finished episodes of a rollout directory.
