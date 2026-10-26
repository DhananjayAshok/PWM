# Simulator and policy environments

The simulators and policies pin incompatible stacks (Python 3.10 vs 3.12, numpy 1.24 vs 1.26, torch CPU vs cu128 vs
cu130), so each gets its own environment, separate from the core one in `setup/`. A simulator and a remote policy talk
over ZMQ (`pwm/sim/policy_client.py`), so they never share a process.

| env | Python | contents | used by |
|---|---|---|---|
| `libero` | 3.12 | LeRobot 0.6.1 (LIBERO env + pi0.5/SmolVLA), MuJoCo 3.2.3, torch 2.11 cu128 | franka_libero (sim and policy) |
| `robocasa_gr1` | 3.10 | robosuite 1.5.1, MuJoCo 3.2.6, RoboCasa GR-1 tabletop tasks + assets, gr00t client side | gr1_robocasa sim |
| `gr00t_n16` | 3.12 | Isaac-GR00T n1.6-release, torch 2.9 cu130, Eagle backbone patched to SDPA | gr1_robocasa policy |
| `simpler` | 3.10 | SimplerEnv + ManiSkill2_real2sim (SAPIEN 2.2.2) | widowx_bridge, google_fractal sim |
| `gr00t_n17` | 3.12 | Isaac-GR00T N1.7, torch 2.9 cu130 (SDPA, no flash-attn) | widowx_bridge, google_fractal policy |

Which robot uses which environment is set in `configs/robots.yaml`.

## Layout of a definition, `setup/envs/<env>/`

- `pyproject.toml`, `.python-version`, `uv.lock`: the Python dependencies, locked. Regenerate the lock after editing the
  pyproject: `uv lock --project setup/envs/<env>` (from a shell that sourced `scripts/utils.sh`, so uv's cache is under
  storage_dir).
- `sources.txt`: upstream repositories that are not pip-installed, one per line: `name url commit submodules`
  (submodules: comma-separated paths to initialise, or `none`). They are cloned at that commit and put on `PYTHONPATH`
  by `env.sh`.
- `patches/<name>/*.diff`: our changes to a source, applied in sorted order with `git apply`. Every deviation from
  upstream lives here as a diff with a comment saying why.
- `build.sh`: builds the environment; idempotent, `--force true` rebuilds.
- `env.sh`: runtime variables, sourced when the environment is activated.

## What a build produces

A build runs on node-local disk (`$PWM_BUILD_DIR`, default `/tmp/pwm_build/<env>/`): `venv/` (relocatable), `src/`
(patched sources), `assets/` (if any), `manifest.json` (hash of the definition, source commits, patches, build time).
Only two files are published to storage_dir: `$storage_dir/envs/<env>.tar` and `$storage_dir/envs/<env>/manifest.json`;
the local build is then removed. Shared filesystems are far too slow for venvs and asset trees of tens of thousands of
files, so nothing is ever unpacked there. A build is skipped while the published manifest's hash matches the
definition. Build everything with `bash setup/setup.sh all`, or some with
`bash scripts/jobs/build_envs.sh --envs "simpler gr00t_n17"` (cluster: `cluster/sail/build_envs.yaml`).

## Using one

In bash, after `source scripts/utils.sh`: `use_env <env>` unpacks the tar to node-local disk (`$PWM_LOCAL_ENV_DIR`,
default `/tmp/pwm_envs/<env>/`; skipped when the copy there has the same manifest), activates it, and sources its
`env.sh`; `use_core_env` returns. Run different environments in subshells, `( use_env simpler; python -m pwm.sim.simpler
... )`, so their variables do not mix.

## Deviations from upstream, and why

- `gr00t_n16`: torch 2.9.0+cu130 instead of 2.7.1+cu128 (native sm_103 kernels on B300); no flash-attn (no wheel for
  torch 2.9 / CUDA 13), so `patches/Isaac-GR00T/01_eagle_sdpa_fallback.diff` makes the Eagle backbone use SDPA, with
  per-image attention for SigLIP2's packed sequences (the semantics of flash-attn's varlen kernel).
- `gr00t_n17`: same torch; without flash-attn gr00t's Qwen3-VL backbone falls back to SDPA by itself. The backbone's
  base model (nvidia/Cosmos-Reason2-2B) is gated; `pwm/policies/gr00t_checkpoint.py` points checkpoints at the ungated
  Qwen/Qwen3-VL-2B-Instruct (identical config/processor files; the backbone weights come from the checkpoint).
- `libero`: MuJoCo pinned to 3.2.3 (lerobot[libero] pulls 3.8, which renders LIBERO visibly differently); the gated
  PaliGemma tokenizer that pi0.5 needs is rebuilt from the public big_vision file.
- `robocasa_gr1`, `simpler`: CPU-only / no torch; versions follow the environments that reproduced the published
  numbers.
