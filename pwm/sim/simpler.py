"""SimplerEnv rollouts (WidowX / Bridge and Google Robot / fractal) against a GR00T N1.7 policy server, in the simpler
environment.

Visual-matching tasks made with simpler_env.make (prepackaged scenes with the real-photo background overlay), as in
Isaac-GR00T's examples/SimplerEnv. Observation and action conversions follow Isaac-GR00T's
gr00t/eval/sim/SimplerEnv/simpler_env.py (WidowXBridgeEnv, GoogleFractalEnv). Episode i is reset with seed i (and, for
Bridge, the prepackaged episode_id i); success = task success at any step, the episode stops at the first success or at
the step limit.

    python -m pwm.sim.simpler --robot widowx_bridge --task widowx_spoon_on_towel --out <dir> --episodes 24
"""
import time

import click
import cv2
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from transforms3d import euler, quaternions

from pwm.robots import get_robot
from pwm.sim.episode import RolloutDir
from pwm.sim.policy_client import PolicyClient
from pwm.utils import load_parameters, log_info

ARM_KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]


class BridgeAdapter:
    """WidowX: image_0, state = eef xyz + rpy (in Bridge's frame convention) + pad + gripper; gripper action [0,1]
    thresholded to -1/+1."""
    image_key = "video.image_0"
    control_hz = 5
    bridge_rotation = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

    def reset_kwargs(self, episode):
        return {"seed": episode, "options": {"obj_init_options": {"episode_id": episode}}}

    def start_episode(self):
        pass

    def state(self, eef_pos):
        roll, pitch, yaw = euler.mat2euler(quaternions.quat2mat(eef_pos[3:7]) @ self.bridge_rotation.T)
        return {"x": eef_pos[0], "y": eef_pos[1], "z": eef_pos[2], "roll": roll, "pitch": pitch, "yaw": yaw,
                "pad": 0.0, "gripper": eef_pos[7]}

    def gripper(self, g):
        return 2.0 * (g > 0.5) - 1.0


class FractalAdapter:
    """Google Robot: image, state = eef xyz + quat xyzw + gripper closedness; gripper action [0,1] -> relative [-1,1]
    with GR00T's "sticky gripper" (a gripper command is held for 15 steps)."""
    image_key = "video.image"
    control_hz = 3
    sticky_steps = 15

    def reset_kwargs(self, episode):
        return {"seed": episode}

    def start_episode(self):
        self._sticky_on, self._sticky_value, self._sticky_count = False, 0.0, 0

    def state(self, eef_pos):
        qx, qy, qz, qw = np.roll(eef_pos[3:7], -1)
        return {"x": eef_pos[0], "y": eef_pos[1], "z": eef_pos[2], "rx": qx, "ry": qy, "rz": qz, "rw": qw,
                "gripper": 1 - eef_pos[7]}

    def gripper(self, g):
        relative = -(g * 2 - 1)
        if abs(relative) > 0.5 and not self._sticky_on:
            self._sticky_on, self._sticky_value = True, relative
        if self._sticky_on:
            self._sticky_count += 1
            relative = self._sticky_value
        if self._sticky_count == self.sticky_steps:
            self._sticky_on, self._sticky_value, self._sticky_count = False, 0.0, 0
        return relative


ADAPTERS = {"SIMPLER_ENV_WIDOWX": BridgeAdapter, "SIMPLER_ENV_GOOGLE": FractalAdapter}


def policy_observation(*, env, obs, adapter, image_size):
    image = get_image_from_maniskill2_obs_dict(env, obs)
    state = adapter.state(obs["agent"]["eef_pos"])
    observation = {adapter.image_key: cv2.resize(image, (image_size[1], image_size[0]))[None, None].astype(np.uint8),
                   "annotation.human.action.task_description": [env.unwrapped.get_language_instruction()]}
    for k, v in state.items():
        observation["state." + k] = np.array([[[v]]], dtype=np.float32)
    return observation, image, np.array(list(state.values()), np.float32)


def run_episode(*, env, client, adapter, episode, protocol):
    obs, _ = env.reset(**adapter.reset_kwargs(episode))
    client.reset()
    adapter.start_episode()
    instruction = env.unwrapped.get_language_instruction()
    rec = {k: [] for k in ("action_env", "action_policy", "state_policy", "eef_pos", "qpos", "tcp_pose", "success")}
    frames, step, success_step = [], 0, None
    while step < protocol["max_steps"] and success_step is None:
        observation, image, state = policy_observation(env=env, obs=obs, adapter=adapter,
                                                       image_size=protocol["image_size"])
        action = client.get_action(observation=observation)
        for i in range(protocol["n_action_steps"]):
            if i > 0:
                observation, image, state = policy_observation(env=env, obs=obs, adapter=adapter,
                                                               image_size=protocol["image_size"])
            arm = [float(np.asarray(action["action." + k])[0, i, 0]) for k in ARM_KEYS]
            g = float(np.asarray(action["action.gripper"])[0, i, 0])
            env_action = np.array(arm + [adapter.gripper(g)], np.float64)
            frames.append(image)
            rec["action_policy"].append(np.array(arm + [g], np.float32)); rec["action_env"].append(env_action)
            rec["state_policy"].append(state)
            rec["eef_pos"].append(np.asarray(obs["agent"]["eef_pos"], np.float32))
            rec["qpos"].append(np.asarray(obs["agent"]["qpos"], np.float32))
            tcp = env.unwrapped.tcp.pose
            rec["tcp_pose"].append(np.concatenate([tcp.p, tcp.q]).astype(np.float32))
            obs, _, _, _, info = env.step(env_action)
            step += 1
            rec["success"].append(bool(info.get("success", False)))
            if rec["success"][-1]:
                success_step = step
            if success_step is not None or step >= protocol["max_steps"]:
                break
    frames.append(get_image_from_maniskill2_obs_dict(env, obs))
    return rec, frames, instruction, success_step


@click.command()
@click.option("--robot", required=True, help="widowx_bridge or google_fractal")
@click.option("--task", required=True, help="SimplerEnv visual-matching task")
@click.option("--out", required=True, help="Rollout directory for this task (absolute path)")
@click.option("--episodes", default=24, type=int)
@click.option("--episode_start", default=0, type=int)
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=5555, type=int)
def main(robot, task, out, episodes, episode_start, host, port):
    parameters = load_parameters()
    spec = get_robot(robot=robot, parameters=parameters)
    adapter = ADAPTERS[spec["embodiment_tag"]]()
    rollouts = RolloutDir(path=out)
    done = rollouts.finished()
    client = PolicyClient(host=host, port=port)
    client.wait_until_ready()
    env = simpler_env.make(task)
    env._max_episode_steps = 10 ** 6  # the protocol's step limit applies, not the env's
    for episode in range(episode_start, episode_start + episodes):
        episode_id = f"ep{episode:03d}"
        if episode_id in done:
            continue
        start = time.time()
        rec, frames, instruction, success_step = run_episode(env=env, client=client, adapter=adapter,
                                                             episode=episode, protocol=spec["protocol"])
        summary = {"task": task, "episode": episode, "success": success_step is not None,
                   "first_success_step": success_step, "steps": len(rec["action_env"]), "instruction": instruction,
                   "wall_s": round(time.time() - start, 1)}
        rollouts.write(episode_id=episode_id, frames=frames, fps=adapter.control_hz,
                       arrays={**rec, "instruction": instruction}, summary=summary)
        log_info(f"{task} {episode_id}: success={summary['success']} steps={summary['steps']}", parameters=parameters)
    env.close()


if __name__ == "__main__":
    main()
