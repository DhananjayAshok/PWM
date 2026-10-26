"""GR-1 / RoboCasa tabletop backend (robocasa_gr1 environment); GR00T N1.6 runs in its own environment as a ZMQ
server (pwm.policies.gr00t_server), queried every n_action_steps through gr00t's own PolicyClient (the N1.6 server
uses gr00t's array encoding, not msgpack_numpy's).

Tasks are RoboCasa GR-1 gym ids; a start is a reset seed. RoboCasa rebuilds the scene on every reset and a seed does
not reproduce it across processes, so everything that must share a start (counterfactual groups, replays of a rollout)
is run inside one unit, from a saved state, in one env instance. The world-model camera is the ego view rendered at
640x480.
"""
import numpy as np

from pwm.sim.backends.base import Backend

ORDER = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]   # 7 + 7 + 6 + 6 + 3 = 29
DIMS = {"left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6, "waist": 3}
HANDS = ("robot0_r_hand_pitch", "robot0_l_hand_pitch")
WM_CAMERA, WM_W, WM_H = "egoview", 640, 480
N_OBJECTS = 2


class RoboCasaGR1Backend(Backend):
    control_hz = 20
    record_stride = 2
    action_dim = 29

    def __init__(self, *, robot, spec, parameters, port=5555):
        super().__init__(robot=robot, spec=spec, parameters=parameters, port=port)
        self._env, self._env_task, self._client, self._queue = None, None, None, []
        self.held_out = set(spec["data"]["held_out_tasks"])

    def tasks(self, *, split):
        tasks = list(self.spec["data"]["tasks"])
        return tasks if split == "test" else [t for t in tasks if t not in self.held_out]

    def _make_env(self, task):
        if self._env_task == task:
            return
        if self._env is not None:
            self._env.close()
        import gymnasium as gym
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401  (registers the gr1_unified/* ids)
        self._env = gym.make(task, enable_render=True)
        self._env_task = task

    @property
    def u(self):
        return self._env.unwrapped          # GrootRoboCasaEnv

    @property
    def sim(self):
        return self.u.env.sim

    def reset(self, *, task, start):
        self._make_env(task)
        obs, _ = self._env.reset(seed=int(start))
        self._obs, self._queue = obs, []
        return obs

    def save_state(self):
        return self.sim.get_state().flatten().copy()

    def restore_state(self, state):
        self.sim.set_state_from_flattened(state)
        self.sim.forward()
        raw = self.u.env._get_observations(force_update=True)
        self._obs, self._queue = self.u.get_groot_observation(self.u.get_basic_observation(raw)), []
        return self._obs

    def policy_reset(self):
        self._queue = []
        if self._client is not None:
            self._client.reset()

    def policy_action(self, obs):
        if self._client is None:
            import time
            from gr00t.policy.server_client import PolicyClient
            self._client = PolicyClient(port=self.port, timeout_ms=120000)
            start = time.time()
            while not self._client.ping():
                if time.time() - start > 1800:
                    raise TimeoutError("GR00T N1.6 server not up")
                time.sleep(5)
            self._client.reset()
        if not self._queue:
            batch = {}
            for k, v in obs.items():
                if k.startswith("video."):
                    batch[k] = np.asarray(v, np.uint8)[None, None]
                elif k.startswith("state."):
                    batch[k] = np.asarray(v, np.float32)[None, None]
                else:
                    batch[k] = [str(v)]
            action, _ = self._client.get_action(batch)
            n = self.spec["protocol"]["n_action_steps"]
            chunk = np.concatenate([np.asarray(action[f"action.{k}"])[0, :n] for k in ORDER], axis=-1)
            self._queue = list(chunk)
        return self._queue.pop(0)

    def _to_dict(self, action):
        out, i = {}, 0
        for k in ORDER:
            out[k] = np.asarray(action[i:i + DIMS[k]], np.float64)
            i += DIMS[k]
        return {f"action.{k}": v for k, v in out.items()}

    def step(self, action):
        obs, _, _, _, info = self._env.step(self._to_dict(action))
        self._obs = obs
        return obs, bool(np.any(info.get("success", False)))

    def wm_frame(self):
        return self.sim.render(camera_name=WM_CAMERA, width=WM_W, height=WM_H)[::-1].copy()

    def policy_state(self, obs):
        return np.concatenate([np.asarray(obs[f"state.{k}"], np.float32) for k in ORDER])

    def probe_state(self):
        sim, env = self.sim, self.u.env
        base = sim.data.body_xpos[sim.model.body_name2id(env.robots[0].robot_model.root_body)]
        hands = [sim.data.xpos[sim.model.body(h).id] - base for h in HANDS]
        state = self.policy_state(self._obs)
        opening = [float(np.mean(state[14:20])), float(np.mean(state[20:26]))]   # left, right hand joint targets
        objs, grasped = [], 0.0
        for name, obj in list(getattr(env, "objects", {}).items())[:N_OBJECTS]:
            objs.append(sim.data.body_xpos[sim.model.body_name2id(obj.root_body)] - base)
            if not grasped:
                for side in ("right", "left"):
                    try:
                        grasped = max(grasped, float(env._check_grasp(gripper=env.robots[0].gripper[side],
                                                                      object_geoms=obj)))
                    except Exception:
                        pass
        objs += [np.zeros(3)] * (N_OBJECTS - len(objs))
        return np.concatenate([*hands, opening, *objs, [grasped]])

    def sim_state(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).astype(np.float32)

    def idle_action(self):
        return self.policy_state(self._obs).astype(np.float64)   # absolute joint targets: hold the current pose

    def noise_scale(self):
        scale = np.zeros(self.action_dim)
        scale[0:14] = 1.0      # arm joints (rad)
        scale[26:29] = 0.5     # waist
        return scale

    def clip_action(self, action):
        return action          # absolute joint targets; the controller clips to joint limits

    def max_steps(self, *, task):
        return self.spec["protocol"]["max_steps"]

    def instruction(self):
        return str(self._obs.get("annotation.human.coarse_action", "")) if self._env is not None else ""

    def close(self):
        if self._env is not None:
            self._env.close()
