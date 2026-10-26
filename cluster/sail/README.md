# Running on the Sail cluster (SkyPilot managed jobs on B300 nodes)

Everything cluster-specific lives here; the scripts in `scripts/jobs/` run anywhere with a GPU. Each `*.yaml` is a
thin wrapper: its `setup:` prepares the node (`job_setup.sh`: EGL/Vulkan libraries for the simulators, core venv) and
its `run:` calls one `scripts/jobs/*.sh`.

Prerequisites: `cluster/sail/sail_vars.yaml` (gitignored; copy `sail_vars.example.yaml`) with your `cluster_user` and
`cluster_volume`; `configs/private_vars.yaml` with storage_dir and results_dir under `/persist` (the volume is mounted there in jobs, as on the dev box); `bash setup/setup.sh` run
once on the dev box (writes `configs/config.env`, which jobs read).

Submit from the project root with `launch.sh`, which fills in the user/volume, passes `HF_HOME` and detaches:

```bash
# build environments as a CPU job (instead of on the dev box)
bash cluster/sail/launch.sh cluster/sail/build_envs.yaml pwm-build-envs --env ENVS="simpler gr00t_n17"
# step-1 acceptance, one robot per job (1 GPU each)
bash cluster/sail/launch.sh cluster/sail/verify_robot.yaml pwm-verify-widowx_bridge --env ROBOT=widowx_bridge
bash cluster/sail/launch.sh cluster/sail/verify_robot.yaml pwm-verify-smoke --env ROBOT=gr1_robocasa --env EXTRA_ARGS="--episodes 2"
```

Then `sky jobs queue --skip-finished -a` (the default listing shows only the latest 50 jobs). Logs:
`<results_dir>/logs/verify_<robot>/`; record: `<results_dir>/verification/<robot>.md`.

Cluster facts the wrappers rely on:
- The image (`nvcr.io/nvidia/pytorch:26.05-py3`) has no GL/EGL/Vulkan user-space libraries: `prepare_node.sh`.
- Jobs can stay RUNNING after their script ends if a child process outlives it: `run:` kills leftovers, and finished
  jobs that still show RUNNING should be cancelled (`sky jobs cancel <id>`).
- Borrowed GPUs can be reclaimed without warning: jobs are restart-safe (finished episodes are kept) and use
  `job_recovery`.
