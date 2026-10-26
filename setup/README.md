# Setup

Two kinds of environments:

- **Core** (`setup/pyproject.toml`, Python 3.12, venv in `setup/.venv`): configs, orchestration, analysis, plotting.
  Every bash script activates it through `scripts/utils.sh`.
- **Simulator / policy environments** (`setup/envs/<env>/`), one per incompatible stack, built into storage_dir. See
  [envs/README.md](envs/README.md).

## Installation

1. Install [uv](https://docs.astral.sh/uv/): `pip install uv`.
2. Fill in `configs/private_vars.yaml` (absolute paths; on a cluster, storage_dir and results_dir must be on the volume
   that jobs mount).
3. Set `HF_HOME` to your Hugging Face cache.
4. From the project root: `bash setup/setup.sh all` (core venv, `configs/config.env`, all environments; rerunning skips
   what is up to date). `bash setup/setup.sh` does only the first two; `bash setup/setup.sh simpler gr00t_n17` builds those.

Then check that the policies reproduce their published success in our environments: `scripts/jobs/verify_robot.sh`
(needs a GPU; on the Sail cluster use `cluster/sail/`, see its README).

## Adding a core dependency

```bash
cd setup
uv add <name>
cd ..
```
and commit `setup/pyproject.toml` and `setup/uv.lock`.
