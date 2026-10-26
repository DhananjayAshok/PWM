# Data format (shared contract between generation, world models, shifts and evaluation)

## Layout

```
$storage_dir/data/<robot>/<dataset>/            dataset = nominal | shift-<shift>-L<level> (test) | calib-<shift>-L<level>
  meta_<hash>.yaml                              the expanded generation config (write_meta)
  units.jsonl                                   every planned unit (deterministic), one line each
  <split>/                                      split = train | val | test | calib   (a pwm.sim.episode.RolloutDir)
    episodes.jsonl                              one line per finished episode (see below)
    <episode_id>.npz                            per-step arrays
    <episode_id>.mp4                            world-model camera, native resolution, recording fps
$storage_dir/data/<robot>/<dataset>/derived/<model>-<hash>/   model-specific caches (clips, latents, features); rebuildable
```

`val` is a fixed held-out shard of the nominal train starts (by unit id hash, 3%), used only for the validation loss.

## Episode

`episodes.jsonl` line: `episode_id, robot, task, split, source (demo|policy|replay|idle), noise_level (0..4 or null),
replay_type (noise|grip|branch|null), group (counterfactual group id or null), start (start-state id), seed, success,
steps, fps, shift, shift_level`.

`<episode_id>.npz`:

| key | shape | meaning |
|---|---|---|
| `action_cmd` | (T, A) | **the command**: control input after the robot's fixed post-processing, before any shift wrapper. World models are conditioned on this |
| `action_env` | (T, A) | what the simulator executed (differs from `action_cmd` only under action-side shifts) |
| `state` | (T+1, S) | proprio the policy saw (start state is `state[0]`) |
| `sim_state` | (T+1, *) | full simulator state where cheap (qpos/qvel) |
| `probe_state` | (T+1, P) | the state-probe target vector: EE position (m, base frame) 3, gripper opening 1, task-object positions 3 each, grasped flag 1 |
| `success` | (T,) | task success per step |
| `instruction` | str | |

Recorded frame k is the observation before control step k*s (s = control steps per recorded frame), for every k*s <= T, so the final observation is included when s divides T.

## Recording rates and cameras (world-model camera)

| robot | control rate | recorded fps | actions per recorded frame | camera, resolution |
|---|---|---|---|---|
| franka_libero | 20 Hz | 10 (stride 2) | 2 (stacked) | agentview, 640x480 (as Parts 2-3) |
| gr1_robocasa | 20 Hz | 10 (stride 2) | 2 (stacked) | egoview rendered at 640x480 |
| widowx_bridge | 5 Hz | 5 | 1 | 3rd_view_camera, 640x480, photo overlay OFF |
| google_fractal | 3 Hz | 3 | 1 | overhead_camera, 640x512, photo overlay OFF |

The npz keeps every control step; mp4s keep only recorded frames (`steps` and `fps` let readers align them).
