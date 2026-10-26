# Deployment shifts (pwm/shifts, configs/shifts.yaml)

## Integration API (for the rollout clients in pwm/sim)

```python
from pwm.shifts import make_shift, refresh_obs
shift = make_shift(robot=robot, name=name, level=level)   # level 1..5; level 0 or name "none" -> NullShift (no-op)
env = <construct env>(..., **shift.make_kwargs())         # scene-build kwargs (only SimplerEnv shadows uses it)
for each episode:
    shift.before_reset(env)            # pre-build edits (distractors: LIBERO task file / RoboCasa distractor config)
    obs, info = env.reset(...)
    shift.on_reset(env)                # re-applies the shift (scenes are rebuilt/re-initialised on reset); restarts
                                       # per-episode state (action buffers, rng seeded per episode)
    if shift.reobserve:
        obs = refresh_obs(robot, env)  # first observation re-rendered from the shifted scene
    for t, control step:
        cmd = <policy command after the robot's fixed post-processing>   -> recorded as action_cmd
        shift.on_step(env, t)          # time-varying edits (light flicker, payload force)
        executed = shift.wrap_action(cmd, t)                              -> recorded as action_env
        obs, ... = env.step(executed)
    record shift.describe() in the episode's info line (shift, shift_level, shift_value, shift_kind)
```

What `env` is per robot (the object the hooks receive):

| robot | env | notes |
|---|---|---|
| franka_libero | lerobot `LiberoEnv` (one per episode, as pwm/sim/libero.py does) | `on_reset` runs after lerobot's own settle steps; shifts that move bodies settle 10 more steps themselves. Distractors: `before_reset` must be called on the fresh LiberoEnv before its first reset |
| gr1_robocasa | the RoboCasa GR-1 gym env (`env.unwrapped` = RoboCasaEnv); inside the GR00T rollout loop, call the hooks from the per-env recorder wrapper (reset / step), below the MultiStepWrapper | actions are the 29-D vector in ORDER (left_arm, right_arm, left_hand, right_hand, waist): convert the action dict to that vector, wrap, convert back |
| widowx_bridge, google_fractal | what `simpler_env.make(task, **shift.make_kwargs())` returns | overlay must be OFF (`env.unwrapped.rgb_overlay_img = None` after make) |

Action vectors given to `wrap_action` (plans/data_format.md `action_cmd`): Franka 7-D normalised OSC delta;
WidowX 7-D (m, rad, gripper +1/-1 after thresholding); Google 7-D (m, axis-angle rad, relative gripper after the sticky
filter); GR-1 29-D absolute joint targets.

Arbitrary values (no ladder): `pwm.shifts.registry.make_shift_value(robot=, name=, value=)`, value in the robot's units.

## Support matrix

`ok` = implemented and verified by `python -m pwm.shifts.selftest` (strongest candidate: pixel change of the first frame
within the same episode, 0-255 scale, and end-effector path change under a fixed command). `n/a` = does not apply.

| shift | franka_libero | gr1_robocasa | widowx_bridge | google_fractal | how |
|---|---|---|---|---|---|
| camera_orientation | ok | ok | ok | ok | rotate the WM/policy camera about its pan+tilt diagonal |
| camera_fov | ok | ok | ok | ok | + deg vertical FOV |
| camera_shift | ok | ok | ok | ok | cm right + half up, camera frame |
| light_flicker | ok | ok | ok | ok | per-step light intensity modulation (amplitude, Hz) |
| shadows | ok | ok | being fixed | being fixed | shadows on + lights tilted (SimplerEnv: shadow-casting light swapped in) |
| colour_temperature | ok | ok | ok | ok | light colour x (1+w, 1+w/3, 1-w) |
| light_direction | ok | ok | ok | ok | lights rotated about world y |
| surface_texture | ok | ok | being fixed | being fixed | MuJoCo: table texture blended to a checker; SimplerEnv: stage texture replaced by a checker of growing contrast |
| table_colour | ok | ok | being fixed | being fixed | table material tinted to green (Google: the stage is one mesh, room tinted too) |
| table_height | ok | ok | ok | ok | Franka: table+objects raised; others: robot lowered (cameras are robot-mounted, equivalent) |
| distractors | ok | ok | ok | ok | LIBERO: task-file objects + init-state remap; RoboCasa: distractor region in the task config; SimplerEnv: model-db actors placed around the task objects |
| control_frequency | ok | ok | ok | ok | commands executed at k x the trained rate |
| latency_jitter | ok | ok | ok | ok | bounded random-walk delay, never reorders |
| latency | ok | ok | ok | ok | fixed delay (whole control steps; robot idles at first) |
| frame_rotation | ok | n/a (joint control) | ok | ok | commanded xy translation rotated about z |
| base_pose | ok | ok | ok | ok | robot base moved along its diagonal + yawed |
| payload | ok | ok | ok | ok | uncompensated external force at the hand(s) |

Translation from the screening robot (franka_libero) to the others (`registry.translate`): same physical units for
camera, light, surface, table, distractor, rate, rotation, bias (cm/s) and base-pose shifts; latency and jitter in ms,
rounded to whole control steps of the robot (min 1, kept strictly increasing, so the 5 Hz / 3 Hz SimplerEnv robots
get coarser, longer delays); payload as the same fraction of the robot's rated payload (configs/shifts.yaml
robot_facts; GR-1 and Google payloads are assumptions).

## Dropped

- `action_bias` (constant drift velocity added to position commands): no effect on pi0.5 success at any screened value up to
  10 cm/s (12 episodes per value), likely a units problem in the conversion to normalised OSC commands; dropped from the
  suite by user decision (2026-10-24) rather than fixed.

## Level ladders and screening evidence

(filled in below after screening)

## Chosen 5-level ladders (2026-10-25, provisional)

Screened on franka_libero (pi0.5), 12 episodes per value, nominal 12/12. `action_bias` dropped (no effect even at the
top candidate). Rule: where the policy breaks, L1 = first noticeable value and L5 = breaking point; where it never
breaks (most visual shifts: pi0.5 is robust to them), five values spread over the screened range. n = 12 is thin (one
episode = 8 points): re-screen L1-L5 with n >= 20 when GPUs allow. Other robots: same physical values via `translate`.

| shift | L1 | L2 | L3 | L4 | L5 | success at L1 -> L5 | breaks policy? |
|---|---|---|---|---|---|---|---|
| camera_orientation (deg) | 4 | 10 | 14 | 18 | 24 | 100 100 83 58 25 % | yes |
| camera_fov (deg) | 3 | 6 | 12 | 20 | 25 | 92 92 92 92 92 % | no |
| camera_shift (cm) | 2 | 5 | 7 | 10 | 14 | 100 100 100 92 92 % | no |
| light_flicker [amp, Hz] | 0.1,1 | 0.3,1 | 0.5,1 | 0.7,1 | 0.9,3 | 100 % all | no |
| shadows (deg) | 10 | 20 | 45 | 60 | 75 | 100 100 83 100 100 % | no |
| colour_temperature | 0.2 | 0.45 | 0.6 | 0.8 | 1.0 | 100 % all | no |
| light_direction (deg) | 20 | 45 | 60 | 75 | 90 | 100 % all | no |
| surface_texture | 0.15 | 0.3 | 0.6 | 0.9 | 1.0 | 100 % all | no |
| table_colour | 0.15 | 0.3 | 0.6 | 0.9 | 1.0 | 100 % all | no |
| table_height (cm) | 2 | 4 | 6 | 8 | 10 | 92 92 83 75 67 % | partly |
| distractors (count) | 1 | 3 | 4 | 5 | 7 | 100 92 92 75 75 % | partly |
| control_frequency (x) | 0.8 | 0.6 | 0.5 | 0.4 | 0.3 | 92 100 83 67 42 % | yes |
| latency_jitter (ms) | 100 | 200 | 300 | 400 | 600 | 100 100 92 67 58 % | partly |
| latency (ms) | 100 | 150 | 200 | 300 | 600 | 100 83 50 33 17 % | yes |
| frame_rotation (deg) | 10 | 15 | 20 | 30 | 45 | 92 83 67 42 0 % | yes (not GR-1) |
| base_pose [cm, deg] | 1,2 | 4,8 | 6,10 | 8,15 | 12,20 | 92 83 100 75 75 % | partly |
| payload (kg) | 0.25 | 0.5 | 1.0 | 2.0 | 3.0 | 92 58 50 25 0 % | yes |
