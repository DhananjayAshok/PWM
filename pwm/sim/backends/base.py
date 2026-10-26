"""The backend interface and the one episode loop every data source uses.

A backend wraps one robot's simulator and policy. The loop (run_episode) is the same for every robot:

    reset(task, start) -> obs                        start = start-state id (LIBERO init index, or a seed)
    loop: cmd = policy(obs) [+ noise] | demo/replay action | idle action
          executed = shift.wrap_action(cmd, t)       (identity without a shift)
          obs, success = step(executed)
    record per control step: action_cmd, action_env, state, probe_state, success; and every `record_stride`-th frame of
    the world-model camera (plans/data_format.md).

Shifts (pwm/shifts) plug in through the optional `shift` object: on_reset(backend), wrap_action(cmd, t),
on_step(backend, t), obs(obs).
"""
import numpy as np


class Backend:
    control_hz = None       # control rate of the simulator
    record_stride = None    # world-model frames are recorded every `record_stride` control steps
    action_dim = None

    def __init__(self, *, robot, spec, parameters, port=5555):
        self.robot, self.spec, self.parameters, self.port = robot, spec, parameters, port

    # --- to implement -----------------------------------------------------------------------------------------------
    def tasks(self, *, split):
        """Task names available for a split (train | test | calib)."""
        raise NotImplementedError

    def reset(self, *, task, start):
        """Reset to a reproducible start state; returns the policy observation."""
        raise NotImplementedError

    def save_state(self):
        raise NotImplementedError

    def restore_state(self, state):
        """Restore a saved state; returns the policy observation re-rendered from it."""
        raise NotImplementedError

    def policy_reset(self):
        pass

    def policy_action(self, obs):
        """The policy's command for the next control step (after the robot's fixed post-processing)."""
        raise NotImplementedError

    def step(self, action):
        """Execute one control step; returns (obs, success)."""
        raise NotImplementedError

    def wm_frame(self):
        """The world-model camera image, uint8 (H, W, 3)."""
        raise NotImplementedError

    def policy_state(self, obs):
        """Proprio vector the policy saw."""
        raise NotImplementedError

    def probe_state(self):
        """State-probe target vector (plans/data_format.md)."""
        raise NotImplementedError

    def sim_state(self):
        return np.zeros(0, np.float32)

    def idle_action(self):
        """Command that holds the robot still."""
        raise NotImplementedError

    def max_steps(self, *, task):
        raise NotImplementedError

    def instruction(self):
        return ""

    def noise_scale(self):
        """Per-dimension scale of action noise (noise_level sigma multiplies it); zeros where noise must not go."""
        raise NotImplementedError

    def clip_action(self, action):
        """Clip a (noisy) command to the valid action range."""
        return np.clip(action, -1.0, 1.0)

    def close(self):
        pass


def ou_noise(*, rng, steps, dim, sigma, theta=0.15):
    """Ornstein-Uhlenbeck noise: smooth, mean-reverting; sigma is the stationary standard deviation."""
    x, out = np.zeros(dim), np.zeros((steps, dim))
    for t in range(steps):
        x = x - theta * x + sigma * np.sqrt(2 * theta) * rng.standard_normal(dim)
        out[t] = x
    return out


class EpisodeBuffer:
    def __init__(self, *, backend):
        self.backend = backend
        self.rows = {k: [] for k in ("action_cmd", "action_env", "state", "probe_state", "sim_state", "success")}
        self.frames = []

    def observe(self, obs, t):
        b = self.backend
        self.rows["state"].append(np.asarray(b.policy_state(obs), np.float32))
        self.rows["probe_state"].append(np.asarray(b.probe_state(), np.float32))
        self.rows["sim_state"].append(np.asarray(b.sim_state(), np.float32))
        if t % b.record_stride == 0:
            self.frames.append(b.wm_frame())

    def act(self, cmd, executed, success):
        self.rows["action_cmd"].append(np.asarray(cmd, np.float32))
        self.rows["action_env"].append(np.asarray(executed, np.float32))
        self.rows["success"].append(bool(success))

    def arrays(self):
        out = {k: np.stack(v) if len(v) else np.zeros((0,), np.float32) for k, v in self.rows.items()}
        return out


def run_episode(*, backend, unit, rng, shift=None, actions=None, start_obs=None, stop_after_success=10):
    """
    Run one episode and return (arrays, frames, summary).

    :param unit: the unit dict (task, start, source, noise_level, ...).
    :param rng: numpy Generator for this unit (noise, perturbations).
    :param shift: optional shift object (pwm/shifts).
    :param actions: fixed command sequence (demo / replay / idle); None = run the policy.
    :param start_obs: observation to start from (when the caller already reset or restored the state).
    :param stop_after_success: steps to keep recording after the first success (policy episodes); None = never stop.
    """
    obs = start_obs if start_obs is not None else backend.reset(task=unit["task"], start=unit["start"])
    if shift is not None:
        shift.on_reset(backend)
        obs = shift.obs(obs)
    backend.policy_reset()
    buf = EpisodeBuffer(backend=backend)
    horizon = len(actions) if actions is not None else backend.max_steps(task=unit["task"])
    noise = None
    sigma = unit.get("noise_sigma") or 0.0
    if actions is None and sigma > 0:
        noise = ou_noise(rng=rng, steps=horizon, dim=backend.action_dim, sigma=1.0) * sigma * backend.noise_scale()
    first_success, success = None, False
    for t in range(horizon):
        buf.observe(obs, t)
        if actions is not None:
            cmd = np.asarray(actions[t], np.float64)
        else:
            cmd = np.asarray(backend.policy_action(obs), np.float64)
            if noise is not None:
                cmd = backend.clip_action(cmd + noise[t])
        executed = shift.wrap_action(cmd, t) if shift is not None else cmd
        obs, success = backend.step(executed)
        if shift is not None:
            shift.on_step(backend, t)
            obs = shift.obs(obs)
        buf.act(cmd, executed, success)
        if success and first_success is None:
            first_success = t
        if actions is None and first_success is not None and stop_after_success is not None \
                and t - first_success >= stop_after_success:
            break
    t = len(buf.rows["action_cmd"])
    buf.rows["state"].append(np.asarray(backend.policy_state(obs), np.float32))
    buf.rows["probe_state"].append(np.asarray(backend.probe_state(), np.float32))
    buf.rows["sim_state"].append(np.asarray(backend.sim_state(), np.float32))
    if t % backend.record_stride == 0:  # recorded frames are those at control steps 0, s, 2s, ... <= T
        buf.frames.append(backend.wm_frame())
    summary = {"success": first_success is not None, "first_success_step": first_success, "steps": t,
               "fps": backend.control_hz / backend.record_stride, "instruction": backend.instruction()}
    return buf.arrays(), buf.frames, summary
