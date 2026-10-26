"""Franka / LIBERO rollouts of a LeRobot policy, in the libero environment (policy and simulator in one process).

Follows lerobot-eval's protocol: LiberoEnv with its config defaults (360x360 agent + wrist views, 20 Hz, relative
control, hard resets), episode i of a task starts from LIBERO's fixed initial state i, an episode ends at success or at
lerobot's per-suite step limit. The episodes of one task run as one policy batch.

    python -m pwm.sim.libero --robot franka_libero --suite libero_spatial --out <dir> [--episodes_per_task 10]
"""
import os

import click
import numpy as np

from pwm.robots import get_robot
from pwm.sim.episode import RolloutDir
from pwm.utils import load_parameters, log_info


def load_policy(*, repo: str, revision: str, n_action_steps: int):
    import torch
    if os.environ.get("PWM_DISABLE_CUDNN") == "1":  # set by setup/envs/libero/env.sh (B300, cu128 cuDNN)
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.enable_cudnn_sdp(False)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    cfg = PreTrainedConfig.from_pretrained(repo, revision=revision)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.n_action_steps = n_action_steps
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = False
    policy = get_policy_class(cfg.type).from_pretrained(repo, config=cfg, revision=revision).to(cfg.device).eval()
    overrides = {"device_processor": {"device": cfg.device}}
    if cfg.type == "pi05":  # the gated PaliGemma tokenizer, rebuilt from the public file (setup/envs/libero/build.sh)
        overrides["tokenizer_processor"] = {"tokenizer_name": os.environ["PWM_PI05_TOKENIZER"]}
    pre, post = make_pre_post_processors(policy.config, repo, preprocessor_overrides=overrides)
    return policy, pre, post


def _stack(items):
    if isinstance(items[0], dict):
        return {k: _stack([d[k] for d in items]) for k in items[0]}
    return np.stack(items)


def run_task(*, policy, pre, post, suite_name: str, task_id: int, episodes: int, rollouts: RolloutDir, parameters):
    import torch
    from lerobot.envs.configs import LiberoEnv as LiberoConfig
    from lerobot.envs.libero import LiberoEnv, _get_suite
    from lerobot.envs.utils import preprocess_observation

    done = rollouts.finished()
    ids = [f"{suite_name}_t{task_id:02d}_i{i:02d}" for i in range(episodes)]
    todo = [i for i, eid in enumerate(ids) if eid not in done]
    if not todo:
        return
    config = LiberoConfig(task=suite_name)
    env_pre, env_post = config.get_env_processors()
    suite = _get_suite(suite_name)
    envs = [LiberoEnv(suite, task_id, suite_name, obs_type=config.obs_type, episode_index=i,
                      observation_height=config.observation_height, observation_width=config.observation_width)
            for i in todo]
    obs = [env.reset(seed=i)[0] for env, i in zip(envs, todo)]
    policy.reset()
    n, max_steps = len(envs), envs[0]._max_episode_steps
    instruction = envs[0].task_description
    rec = [{"frames": [], "wrist": [], "action_env": [], "state_policy": [], "success": []} for _ in envs]
    active = np.ones(n, bool)
    for _ in range(max_steps):
        batch = preprocess_observation({"pixels": _stack([o["pixels"] for o in obs]),
                                        "robot_state": _stack([o["robot_state"] for o in obs])})
        batch["task"] = [instruction] * n
        batch = pre(env_pre(batch))
        with torch.inference_mode():
            action = post(policy.select_action(batch))
        action = env_post({"action": action})["action"].float().cpu().numpy()
        for k, env in enumerate(envs):
            if not active[k]:
                continue
            r, o = rec[k], obs[k]
            r["frames"].append(o["pixels"]["image"]); r["wrist"].append(o["pixels"]["image2"])
            rs = o["robot_state"]
            r["state_policy"].append(np.concatenate([rs["eef"]["pos"], rs["eef"]["quat"], rs["gripper"]["qpos"]]))
            r["action_env"].append(action[k])
            obs[k], _, terminated, _, info = env.step(action[k])
            r["success"].append(bool(info["is_success"]))
            if terminated:
                active[k] = False
        if not active.any():
            break
    for k, (env, i) in enumerate(zip(envs, todo)):
        r = rec[k]
        r["frames"].append(obs[k]["pixels"]["image"]); r["wrist"].append(obs[k]["pixels"]["image2"])
        success = any(r["success"])
        rollouts.write(episode_id=ids[i], frames=r["frames"], extra_videos={"wrist": r["wrist"]}, fps=config.fps,
                       arrays={"action_env": r["action_env"], "action_policy": r["action_env"],
                               "state_policy": r["state_policy"], "success": r["success"], "instruction": instruction},
                       summary={"task": suite_name, "task_id": task_id, "init_state": i, "success": success,
                                "steps": len(r["action_env"]), "instruction": instruction})
        env.close()
    log_info(f"{suite_name} task {task_id}: {sum(any(r['success']) for r in rec)}/{n} successes", parameters=parameters)


@click.command()
@click.option("--robot", default="franka_libero")
@click.option("--suite", required=True, help="LIBERO suite, e.g. libero_spatial")
@click.option("--out", required=True, help="Rollout directory (absolute path)")
@click.option("--episodes_per_task", default=None, type=int, help="Defaults to the robot's protocol")
def main(robot, suite, out, episodes_per_task):
    parameters = load_parameters()
    spec = get_robot(robot=robot, parameters=parameters)
    episodes = episodes_per_task or spec["protocol"]["episodes_per_task"]
    policy, pre, post = load_policy(repo=spec["policy"], revision=spec["policy_revision"],
                                    n_action_steps=spec["protocol"]["n_action_steps"])
    from lerobot.envs.libero import _get_suite
    rollouts = RolloutDir(path=out)
    for task_id in range(len(_get_suite(suite).tasks)):
        run_task(policy=policy, pre=pre, post=post, suite_name=suite, task_id=task_id, episodes=episodes,
                 rollouts=rollouts, parameters=parameters)


if __name__ == "__main__":
    main()
