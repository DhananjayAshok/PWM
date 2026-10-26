# Configs

All `.yaml` files here are merged into one `parameters` dict (`pwm.utils.load_parameters`); a key may appear in only
one file. All paths must be absolute. `python configs/create_env_file.py` (run by `setup/setup.sh`) writes the scalar
values to `configs/config.env` for bash scripts (`scripts/utils.sh` sources it).

| file | contents |
|---|---|
| `private_vars.yaml` | machine/person-specific values, committed as `PLACEHOLDER`: `storage_dir` (heavy files: environments, data, models), `results_dir` (logs, verification records, figures), `huggingface_repo_namespace` / `huggingface_repo_name` (`sync_data.py`). Cluster accounts are not here: see `cluster/<cluster>/` |
| `project_vars.yaml` | `random_seed`, `figure_skip_show`, `WANDB_PROJECT` |
| `robots.yaml` | the robot registry: environments, policy (+ pinned revision), evaluation protocol, tasks and published success rates of each robot; the substitute for GR00T N1.7's gated backbone |

Derived automatically (do not set): `data_dir`, `model_dir`, `tmp_dir`, `sync_dir` under storage_dir; `log_dir`,
`figure_dir` under results_dir.
