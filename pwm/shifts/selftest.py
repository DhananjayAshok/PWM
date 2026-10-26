"""Self-test of every shift on one robot: apply it at its strongest candidate value, render the first frame and step a
fixed command, and report the first-frame pixel change and the end-effector path change against no shift. Writes one
JSON line per shift to <out>/selftest_<robot>.jsonl. The pixel change compares the first frame before and after the
shift is applied in the same episode (RoboCasa does not reproduce scenes across resets). Runs in the robot's simulator environment (GPU for SimplerEnv).

    python -m pwm.shifts.selftest --robot widowx_bridge --out <dir>
"""
import json
import os
import time
import traceback

import click
import numpy as np

from pwm.shifts.registry import make_shift_value, refresh_obs, unsupported_reason
from pwm.utils import load_parameters, log_info

TASK = {"franka_libero": ("libero_spatial", 0), "widowx_bridge": "widowx_carrot_on_plate",
        "google_fractal": "google_robot_pick_coke_can",
        "gr1_robocasa": "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"}
STEPS = 12


class Harness:
    """Robot-specific env construction, first frame, end-effector position and a fixed command."""

    def __init__(self, robot):
        self.robot = robot

    def make(self, shift):
        kw = shift.make_kwargs() if shift is not None else {}
        if self.robot == "franka_libero":
            from lerobot.envs.libero import LiberoEnv, _get_suite
            suite_name, task_id = TASK[self.robot]
            return LiberoEnv(_get_suite(suite_name), task_id, suite_name, obs_type="pixels_agent_pos", episode_index=0,
                             observation_height=128, observation_width=128)
        if self.robot == "gr1_robocasa":
            import gymnasium as gym
            import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401 (registers the envs)
            return gym.make(TASK[self.robot], enable_render=True, **kw)
        import simpler_env
        env = simpler_env.make(TASK[self.robot], **kw)
        env.unwrapped.rgb_overlay_img = None
        return env

    def reset(self, env):
        if self.robot == "franka_libero":
            return env.reset(seed=0)[0]
        if self.robot == "gr1_robocasa":
            np.random.seed(0)
            return env.reset(seed=0)[0]
        opts = {"obj_init_options": {"episode_id": 0}} if self.robot == "widowx_bridge" else None
        return env.reset(seed=0, options=opts)[0]

    def frame(self, env, obs):
        if self.robot == "franka_libero":
            return env._env.sim.render(camera_name="agentview", width=128, height=128)[::-1].copy()
        if self.robot == "gr1_robocasa":
            return np.asarray(obs["video.ego_view_pad_res256_freq20"])
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
        return get_image_from_maniskill2_obs_dict(env, obs)

    def ee(self, env, obs):
        if self.robot == "franka_libero":
            return np.asarray(obs["robot_state"]["eef"]["pos"], float)
        if self.robot == "gr1_robocasa":
            from pwm.shifts import mujoco_common as mc
            m, d = mc.raw(env.unwrapped.env.sim.model), mc.raw(env.unwrapped.env.sim.data)
            return d.xpos[mc.mj().mj_name2id(m, mc.mj().mjtObj.mjOBJ_BODY, "robot0_r_hand_pitch")].copy()
        return np.asarray(env.unwrapped.tcp.pose.p, float)

    def command(self, obs, t):
        if self.robot == "franka_libero":
            return np.array([0.3, 0, 0, 0, 0, 0, -1.0], np.float32)
        if self.robot == "gr1_robocasa":
            from pwm.sim.robocasa_gr1 import ORDER
            return np.concatenate([np.asarray(obs[f"state.{k}"], np.float32) for k in ORDER]) + 0.02 * min(t, 6)
        if self.robot == "widowx_bridge":
            return np.array([0.01, 0, 0, 0, 0, 0, 1.0])
        return np.array([0.01, 0, 0, 0, 0, 0, 0.0])

    def step(self, env, action):
        if self.robot == "gr1_robocasa":
            from pwm.sim.robocasa_gr1 import ORDER
            sizes = {"left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6, "waist": 3}
            out, i = {}, 0
            for k in ORDER:
                out[f"action.{k}"] = np.asarray(action[i:i + sizes[k]], np.float32)
                i += sizes[k]
            return env.step(out)[0]
        return env.step(action)[0]


def run(h, shift):
    env = h.make(shift)
    if shift is not None:
        shift.before_reset(env)
    obs = h.reset(env)
    f_pre = h.frame(env, obs)
    if shift is not None:
        shift.on_reset(env)
        if shift.reobserve:
            obs = refresh_obs(h.robot, env)
    f0 = h.frame(env, obs)
    run.f_pre = f_pre
    run.pix = float(np.abs(f0.astype(int) - f_pre.astype(int)).mean())  # same scene, before vs after the shift
    run.idem = None
    if shift is not None and shift.reobserve and run.factory is not None:
        # a second, fresh shift object applied again without a reset (what restore_state + on_reset does in the data
        # generator) must not change the scene further
        again = run.factory()
        again.on_reset(env)
        f1 = h.frame(env, refresh_obs(h.robot, env))
        run.idem = float(np.abs(f1.astype(int) - f0.astype(int)).mean())
    ees = []
    for t in range(STEPS):
        cmd = h.command(obs, t)
        a = cmd
        if shift is not None:
            shift.on_step(env, t)
            a = shift.wrap_action(cmd, t)
        obs = h.step(env, a)
        ees.append(h.ee(env, obs))
    env.close()
    return f0, np.array(ees)


@click.command()
@click.option("--robot", required=True)
@click.option("--out", required=True)
@click.option("--shifts", default="all")
def main(robot, out, shifts):
    parameters = load_parameters()
    os.makedirs(out, exist_ok=True)
    h = Harness(robot)
    run.factory = None
    f_base, ee_base = run(h, None)
    suite = parameters["shifts"]["suite"]
    names = list(suite) if shifts == "all" else shifts.split(",")
    path = os.path.join(out, f"selftest_{robot}.jsonl")
    for name in names:
        reason = unsupported_reason(robot=robot, name=name, parameters=parameters)
        row = {"robot": robot, "shift": name}
        if reason:
            row.update(status="unsupported", reason=reason)
        else:
            value = suite[name]["candidates"][-1]
            row["value"] = value
            t0 = time.time()
            try:
                run.factory = lambda: make_shift_value(robot=robot, name=name, value=value, parameters=parameters)
                f0, ee = run(h, run.factory())
                import imageio.v2 as iio
                iio.imwrite(os.path.join(out, f"selftest_{robot}_{name}.png"),
                            np.concatenate([run.f_pre, f0], axis=1).astype(np.uint8))
                n = min(len(ee), len(ee_base))
                row.update(status="ok", pix=round(run.pix, 2), idempotence_pix=None if run.idem is None else round(run.idem, 2),
                           ee_dev_cm=round(float(np.linalg.norm(ee[:n] - ee_base[:n], axis=1).mean() * 100), 2),
                           seconds=round(time.time() - t0, 1))
            except Exception as e:
                row.update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
        log_info(json.dumps({k: v for k, v in row.items() if k != "trace"}), parameters=parameters)
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
