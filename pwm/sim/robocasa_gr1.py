"""GR-1 / RoboCasa tabletop rollouts against a GR00T N1.6 policy server, in the robocasa_gr1 environment.

Runs Isaac-GR00T's own evaluation loop (gr00t/eval/rollout_policy.py: vectorised envs, MultiStepWrapper with
n_action_steps / max_steps, terminate on success), with an episode recorder inserted below the MultiStepWrapper so that
every executed 20 Hz step is recorded. An episode is written when it ends (success or step limit); episodes still
running when the loop has collected its n_episodes are dropped.

    python -m pwm.sim.robocasa_gr1 --task <gym id> --out <dir> --episodes 20 [--n_envs 5] [--port 5555]
"""
import os
import uuid

import click
import numpy as np

from pwm.robots import get_robot
from pwm.utils import load_parameters, log_info

ORDER = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]  # 7 + 7 + 6 + 6 + 3 = 29-D state and action
POLICY_VIEW = "video.ego_view_bg_crop_pad_res256_freq20"
FULL_VIEW = "video.ego_view_pad_res256_freq20"
THIRD_VIEW = "robot0_agentview_left"
HANDS = ("robot0_r_hand_pitch", "robot0_l_hand_pitch")


def make_recorder_class():
    import gymnasium as gym
    from pwm.sim.episode import RolloutDir

    class EpisodeRecorder(gym.Wrapper):
        """Records every executed step of the wrapped (single-step) env and writes each finished episode."""

        def __init__(self, env, *, out_dir: str, task: str, env_idx: int):
            super().__init__(env)
            self.rollouts, self.task, self.env_idx = RolloutDir(path=out_dir), task, env_idx
            self.run_id, self.count, self.buf = uuid.uuid4().hex[:6], 0, None

        @property
        def sim(self):
            return self.env.unwrapped.env.sim

        def _observe(self, obs):
            b, sim = self.buf, self.sim
            b["state_policy"].append(np.concatenate([np.asarray(obs[f"state.{k}"], np.float32) for k in ORDER]))
            b["frames"].append(obs[POLICY_VIEW]); b["full"].append(obs[FULL_VIEW])
            b["third"].append(sim.render(camera_name=THIRD_VIEW, width=256, height=256)[::-1].copy())
            b["qpos"].append(sim.data.qpos.copy()); b["qvel"].append(sim.data.qvel.copy())
            b["eef"].append(np.concatenate([np.concatenate([sim.data.xpos[sim.model.body(h).id],
                                                            sim.data.xquat[sim.model.body(h).id]]) for h in HANDS]))

        def _write(self):
            b = self.buf
            self.buf = None
            if b is None or not b["action_env"]:
                return
            episode_id = f"env{self.env_idx}_{self.run_id}_ep{self.count:03d}"
            self.count += 1
            success = bool(any(b["success"]))
            self.rollouts.write(
                episode_id=episode_id, frames=b["frames"], fps=20,
                extra_videos={"full": b["full"], "third": b["third"]},
                arrays={"action_env": b["action_env"], "action_policy": b["action_env"],
                        "state_policy": b["state_policy"], "success": b["success"], "qpos": b["qpos"],
                        "qvel": b["qvel"], "eef": b["eef"], "order": np.array(ORDER), "instruction": b["instruction"]},
                summary={"task": self.task, "success": success, "steps": len(b["action_env"]),
                         "instruction": b["instruction"]})

        def reset(self, **kwargs):
            self._write()  # the wrapper resets only after an episode ended, so the buffer holds a finished episode
            obs, info = self.env.reset(**kwargs)
            self.buf = {k: [] for k in ("action_env", "state_policy", "frames", "full", "third", "qpos", "qvel",
                                        "eef", "success")}
            self.buf["instruction"] = str(obs.get("annotation.human.coarse_action", ""))
            self._observe(obs)
            return obs, info

        def step(self, action):
            self.buf["action_env"].append(np.concatenate([np.asarray(action[f"action.{k}"], np.float32) for k in ORDER]))
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.buf["success"].append(bool(np.any(info.get("success", False))))
            self._observe(obs)
            return obs, reward, terminated, truncated, info

        def close(self):
            self.buf = None  # unfinished episode
            return self.env.close()

    return EpisodeRecorder


def create_eval_env(env_name, env_idx, total_n_envs, wrapper_configs):
    """Isaac-GR00T's rollout_policy.create_eval_env with the recorder below the MultiStepWrapper. Spawned env workers
    pickle this function by reference, so it must stay at module level; they read the output dir from the env."""
    from gr00t.eval import rollout_policy
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
    env = rollout_policy.get_gym_env(env_name, env_idx, total_n_envs)
    env = make_recorder_class()(env, out_dir=os.environ["PWM_ROLLOUT_DIR"], task=env_name, env_idx=env_idx)
    ms = wrapper_configs.multistep
    return MultiStepWrapper(env, video_delta_indices=ms.video_delta_indices, state_delta_indices=ms.state_delta_indices,
                            n_action_steps=ms.n_action_steps, max_episode_steps=ms.max_episode_steps,
                            terminate_on_success=ms.terminate_on_success)


@click.command()
@click.option("--robot", default="gr1_robocasa")
@click.option("--task", required=True, help="RoboCasa GR-1 gym id")
@click.option("--out", required=True, help="Rollout directory for this task (absolute path)")
@click.option("--episodes", default=20, type=int)
@click.option("--n_envs", default=5, type=int, help="Parallel simulator processes")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=5555, type=int)
def main(robot, task, out, episodes, n_envs, host, port):
    parameters = load_parameters()
    protocol = get_robot(robot=robot, parameters=parameters)["protocol"]
    os.makedirs(out, exist_ok=True)
    os.environ["PWM_ROLLOUT_DIR"] = out  # inherited by the spawned env workers
    from gr00t.eval import rollout_policy
    from gr00t.policy.server_client import PolicyClient
    rollout_policy.create_eval_env = create_eval_env
    wrapper_configs = rollout_policy.WrapperConfigs(
        video=rollout_policy.VideoConfig(video_dir=None, max_episode_steps=protocol["max_steps"]),
        multistep=rollout_policy.MultiStepConfig(n_action_steps=protocol["n_action_steps"],
                                                 max_episode_steps=protocol["max_steps"], terminate_on_success=True))
    policy = PolicyClient(host=host, port=port)
    _, successes, _ = rollout_policy.run_rollout_gymnasium_policy(task, policy, wrapper_configs, n_episodes=episodes,
                                                                 n_envs=n_envs)
    log_info(f"{task}: GR00T loop reports {int(np.sum(successes))}/{len(successes)} successes", parameters=parameters)


if __name__ == "__main__":
    main()
