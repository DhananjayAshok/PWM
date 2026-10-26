"""SimplerEnv backend (simpler environment) for WidowX / Bridge and Google Robot / fractal; GR00T N1.7 runs in its own
environment as a ZMQ server, queried every n_action_steps.

Visual-matching scenes with the real-photo overlay OFF (the full simulated scene is rendered, so camera, lighting and
table shifts are consistent). A start is a seed; for Bridge it also places the two objects at seeded random positions
around the task's prepackaged layouts, so there are more distinct starts than the 24 prepackaged ones. The command is
the 7-D env action after the robot's fixed gripper post-processing (Bridge: thresholded to -1/+1; fractal: GR00T's
sticky gripper); noise goes on the 6 arm dimensions only.
"""
import numpy as np

from pwm.sim.backends.base import Backend
from pwm.sim.policy_client import PolicyClient
from pwm.sim.simpler import ADAPTERS, ARM_KEYS

JITTER_M = 0.03       # Bridge: seeded object-position jitter around a prepackaged layout
NOISE_SCALE = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.0])   # typical per-step EE delta (m, rad)
N_OBJECTS = 2


class SimplerBackend(Backend):
    record_stride = 1
    action_dim = 7

    def __init__(self, *, robot, spec, parameters, port=5555):
        super().__init__(robot=robot, spec=spec, parameters=parameters, port=port)
        self.adapter = ADAPTERS[spec["embodiment_tag"]]()
        self.control_hz = self.adapter.control_hz
        self._env, self._env_task, self._client, self._queue, self._info = None, None, None, [], {}

    def tasks(self, *, split):
        return list(self.spec["tasks"])

    def _make_env(self, task):
        if self._env_task == task:
            return
        if self._env is not None:
            self._env.close()
        import simpler_env
        self._env = simpler_env.make(task)
        self._env._max_episode_steps = 10 ** 6
        u = self._env.unwrapped
        self._orig_layouts = (list(getattr(u, "_xy_configs", [])), list(getattr(u, "_quat_configs", [])))
        self._env_task = task

    @property
    def u(self):
        return self._env.unwrapped

    def _fresh_obs(self):
        from mani_skill2_real2sim.utils.wrappers.observation import RGBDObservationWrapper
        e = self._env
        while not isinstance(e, RGBDObservationWrapper):
            e = e.env
        return e.observation(self.u.get_obs())

    def reset(self, *, task, start):
        self._make_env(task)
        start = int(start)
        xys, quats = self._orig_layouts
        if xys:   # Bridge: one seeded layout, jittered
            rng = np.random.default_rng(start)
            xy = np.asarray(xys[rng.integers(len(xys))], float)
            self.u._xy_configs = [xy + rng.uniform(-JITTER_M, JITTER_M, xy.shape)]
            self.u._quat_configs = [quats[rng.integers(len(quats))]]
            self._env.reset(seed=start, options={"obj_init_options": {"episode_id": 0}})
        else:
            self._env.reset(seed=start)
        self.u.rgb_overlay_img = None          # photo overlay off (checked at every get_obs)
        self._obs, self._queue, self._info = self._fresh_obs(), [], {}
        return self._obs

    def save_state(self):
        return np.asarray(self.u.get_state()).copy()

    def restore_state(self, state):
        self.u.set_state(state)
        self._obs, self._queue, self._info = self._fresh_obs(), [], {}
        return self._obs

    def policy_reset(self):
        self._queue = []
        self.adapter.start_episode()
        if self._client is not None:
            self._client.reset()

    def policy_action(self, obs):
        from pwm.sim.simpler import policy_observation
        if self._client is None:
            self._client = PolicyClient(port=self.port)
            self._client.wait_until_ready()
            self._client.reset()
        if not self._queue:
            observation, _, _ = policy_observation(env=self._env, obs=obs, adapter=self.adapter,
                                                   image_size=self.spec["protocol"]["image_size"])
            action = self._client.get_action(observation=observation)
            n = self.spec["protocol"]["n_action_steps"]
            self._queue = [([float(np.asarray(action["action." + k])[0, i, 0]) for k in ARM_KEYS],
                            float(np.asarray(action["action.gripper"])[0, i, 0])) for i in range(n)]
        arm, g = self._queue.pop(0)
        return np.array(arm + [self.adapter.gripper(g)], np.float64)

    def step(self, action):
        obs, _, _, _, info = self._env.step(np.asarray(action, np.float64))
        self._obs, self._info = obs, info
        return obs, bool(info.get("success", False))

    def wm_frame(self):
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
        return np.asarray(get_image_from_maniskill2_obs_dict(self._env, self._obs), np.uint8).copy()

    def policy_state(self, obs):
        return np.asarray(list(self.adapter.state(obs["agent"]["eef_pos"]).values()), np.float32)

    def _task_objects(self):
        u = self.u
        objs = [getattr(u, a) for a in ("episode_source_obj", "episode_target_obj", "obj") if getattr(u, a, None) is not None]
        return objs[:N_OBJECTS]

    def probe_state(self):
        u = self.u
        base = u.agent.robot.pose.p
        tcp = u.tcp.pose.p - base
        opening = float(self._obs["agent"]["eef_pos"][7])
        objs = [np.asarray(o.pose.p) - base for o in self._task_objects()]
        objs += [np.zeros(3)] * (N_OBJECTS - len(objs))
        grasped = float(self._info.get("is_src_obj_grasped", False))
        if not self._info and self._task_objects():
            try:
                grasped = float(u.agent.check_grasp(self._task_objects()[0]))
            except Exception:
                pass
        return np.concatenate([tcp, [opening], *objs, [grasped]])

    def sim_state(self):
        return np.asarray(self._obs["agent"]["qpos"], np.float32)

    def idle_action(self):
        g = 1.0 if self.spec["embodiment_tag"] == "SIMPLER_ENV_WIDOWX" else 0.0   # Bridge: keep open; fractal: no change
        return np.array([0, 0, 0, 0, 0, 0, g])

    def noise_scale(self):
        return NOISE_SCALE

    def clip_action(self, action):
        return action

    def max_steps(self, *, task):
        return self.spec["protocol"]["max_steps"]

    def instruction(self):
        return self.u.get_language_instruction() if self._env is not None else ""

    def close(self):
        if self._env is not None:
            self._env.close()
