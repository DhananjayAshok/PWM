# Archived: hyperparameter sweeps (2026-10-24 to 2026-10-26)

The sweep machinery for the three world models, moved here on 2026-10-26 when the sweeps were cancelled. The files keep
their original paths below this folder (e.g. `archive/sweeps/cluster/sail/sweep_vjepa2.py` was
`cluster/sail/sweep_vjepa2.py`). **None of it runs as is:** the trainers no longer take a sweep run name, and the paths
inside these files point at the old locations. It is kept as a record of how the sweeps were run.

- `configs/sweeps/<model>.yaml`: the named runs (overrides of `configs/world_models.yaml`) and their groups.
- `cluster/sail/sweep_dreamerv3.sh`, `sweep_vjepa2.py`, `sweep_dreamdojo.sh`: the dev-box orchestrators (launch missing
  runs, relaunch evicted ones, pick each robot's best grid run, schedule its seed replicates).
- `pwm/wm/dreamerv3/sweep.py`, `pwm/wm/dreamdojo/sweep_plan.py`: selection and planning helpers of those orchestrators.
- `scripts/jobs/sweep_dreamerv3.sh`, `scripts/jobs/train_vjepa2_pack.sh` and their `cluster/sail` wrappers: several runs
  packed on one GPU.
- `pwm/wm/*/sweep_table.py`, `pwm/wm/dreamdojo/sweep_report.py`: per-robot results tables.
- `pwm/wm/dreamdojo/rollout_eval.py` (+ job wrappers): post-training rollout evaluation used only by the DreamDojo sweep.
- `plans/literature_tricks.md`: the vetted list of 2025+ training tricks, and the results of the ones that ran
  (dropped 2026-10-26).

The write-up of what was tried and the settings chosen from it: `plans/sweep_report.md`. The results stay where the
runs wrote them: DreamerV3 summaries in `<storage_dir>/sweeps/dreamerv3/<robot>/`,
V-JEPA run directories in `<storage_dir>/checkpoints/vjepa2/<robot>/`, tables in `<results_dir>/sweeps/`.
