"""Franka / LIBERO backend (libero environment): LeRobot's LiberoEnv (the policy's exact observation pipeline) around
LIBERO's robosuite env, pi0.5 in-process.

Tasks are "<suite>/<task_id>"; a start is a LIBERO benchmark initial-state index. The world-model camera is the
agent view rendered at 640x480 (separately from the policy's 360x360 observation).
"""

import numpy as np

from pwm.sim.backends.base import Backend

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
WM_W, WM_H = 640, 480
N_OBJECTS = 2  # task objects in the probe vector (padded with zeros)


class LiberoBackend(Backend):
    control_hz = 20
    record_stride = 2
    action_dim = 7

    def __init__(self, *, robot, spec, parameters, port=5555):
        super().__init__(robot=robot, spec=spec, parameters=parameters, port=port)
        self._env, self._env_task, self._policy = None, None, None
        self.held_out = {tuple(x) for x in spec["data"]["held_out_tasks"]}

    # --- tasks and env -----------------------------------------------------------------------------------------------
    def tasks(self, *, split):
        from lerobot.envs.libero import _get_suite
        out = []
        for s in SUITES:
            for t in range(len(_get_suite(s).tasks)):
                held = (s, t) in self.held_out
                if (split == "test") or not held:
                    out.append(f"{s}/{t}")
        return out

    def _make_env(self, task):
        if self._env_task == task:
            return
        if self._env is not None:
            self._env.close()
        from lerobot.envs.configs import LiberoEnv as LiberoConfig
        from lerobot.envs.libero import LiberoEnv, _get_suite
        suite, task_id = task.split("/")
        self._config = LiberoConfig(task=suite)
        self._env = LiberoEnv(_get_suite(suite), int(task_id), suite, obs_type=self._config.obs_type, episode_index=0,
                              observation_height=self._config.observation_height,
                              observation_width=self._config.observation_width)
        self._env_task, self._suite = task, suite

    @property
    def raw(self):
        return self._env._env       # LIBERO OffScreenRenderEnv

    @property
    def rs(self):
        return self._env._env.env   # robosuite env

    def reset(self, *, task, start):
        self._make_env(task)
        self._env.init_state_id = int(start)
        obs, _ = self._env.reset(seed=int(start))
        self._obs = obs
        return obs

    def save_state(self):
        return self.raw.sim.get_state().flatten().copy()

    def restore_state(self, state):
        raw_obs = self.raw.set_init_state(state)
        self._obs = self._env._format_raw_obs(raw_obs)
        return self._obs

    def reset_to_state(self, *, task, state):
        """Start from an arbitrary flattened sim state (demo replays)."""
        self._make_env(task)
        self._env.reset(seed=0)
        return self.restore_state(state)

    # --- policy ------------------------------------------------------------------------------------------------------
    def _load_policy(self):
        if self._policy is None:
            from pwm.sim.libero import load_policy
            self._policy = load_policy(repo=self.spec["policy"], revision=self.spec["policy_revision"],
                                       n_action_steps=self.spec["protocol"]["n_action_steps"])
            self._env_pre, self._env_post = self._config.get_env_processors()
        return self._policy

    def policy_reset(self):
        if self._policy is not None:
            self._policy[0].reset()

    def policy_action(self, obs):
        import torch
        from lerobot.envs.utils import preprocess_observation
        policy, pre, post = self._load_policy()
        stack = lambda x: {k: stack(v) for k, v in x.items()} if isinstance(x, dict) else np.asarray(x)[None]
        batch = preprocess_observation({"pixels": stack(obs["pixels"]), "robot_state": stack(obs["robot_state"])})
        batch["task"] = [self._env.task_description]
        batch = pre(self._env_pre(batch))
        with torch.inference_mode():
            action = post(policy.select_action(batch))
        return self._env_post({"action": action})["action"].float().cpu().numpy()[0]

    # --- stepping and recording --------------------------------------------------------------------------------------
    def step(self, action):
        obs, _, _, _, info = self._env.step(np.asarray(action, np.float32))
        self._obs = obs
        return obs, bool(info["is_success"])

    def wm_frame(self):
        return self.raw.sim.render(camera_name="agentview", width=WM_W, height=WM_H)[::-1].copy()

    def policy_state(self, obs):
        rs = obs["robot_state"]
        return np.concatenate([rs["eef"]["pos"], rs["eef"]["quat"], rs["gripper"]["qpos"]])

    def _base_pos(self):
        sim = self.raw.sim
        return sim.data.body_xpos[sim.model.body_name2id(self.rs.robots[0].robot_model.root_body)]

    def probe_state(self):
        env, sim = self.rs, self.raw.sim
        base = self._base_pos()
        robot = env.robots[0]
        eef = sim.data.site_xpos[robot.eef_site_id] - base
        grip = float(np.sum(np.abs(sim.data.qpos[robot._ref_gripper_joint_pos_indexes])))
        objs, grasped = [], 0.0
        for name in list(env.obj_of_interest)[:N_OBJECTS]:
            if name in env.obj_body_id:
                objs.append(sim.data.body_xpos[env.obj_body_id[name]] - base)
            if name in env.objects_dict and not grasped:
                try:
                    grasped = float(env._check_grasp(gripper=robot.gripper, object_geoms=env.objects_dict[name]))
                except Exception:
                    pass
        objs += [np.zeros(3)] * (N_OBJECTS - len(objs))
        return np.concatenate([eef, [grip], *objs, [grasped]])

    def sim_state(self):
        return self.save_state()

    def idle_action(self):
        return np.array([0, 0, 0, 0, 0, 0, -1.0])

    def noise_scale(self):
        return np.array([1, 1, 1, 1, 1, 1, 0.0])

    def max_steps(self, *, task):
        return self._env._max_episode_steps

    def instruction(self):
        return self._env.task_description if self._env is not None else ""

    def close(self):
        if self._env is not None:
            self._env.close()
