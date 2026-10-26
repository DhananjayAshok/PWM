"""The Shift interface. A shift is created per (robot, name, level) and reused across episodes of one environment
instance; `on_reset` re-applies it after every reset (all three simulators rebuild or re-initialise the scene) and
restarts its per-episode state (action buffers, random streams seeded per episode)."""
from typing import Any, Optional

import numpy as np


class Shift:
    """Base class: every hook is a no-op, so a shift overrides only the hooks of its kind."""

    #: one of: model (sim edit after reset), action (wrapper between action_cmd and action_env), step (per-step hook),
    #: scene (needs env-construction kwargs or a pre-reset edit), mixed
    kind = "model"
    #: True if the first observation after reset must be re-rendered (anything that changes what the camera sees)
    reobserve = False

    def __init__(self, *, robot: str, name: str, level: int, value: Any, facts: dict):
        self.robot, self.name, self.level, self.value, self.facts = robot, name, level, value, facts
        self.episode = 0
        self.rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- hooks called by the rollout client
    def make_kwargs(self) -> dict:
        """Extra keyword arguments for constructing the environment (scene-build shifts)."""
        return {}

    def before_reset(self, handle, task: Optional[str] = None) -> None:
        """Called before env/backend reset (scene-build shifts that need new objects: distractors). `task` lets a
        backend create its env first."""

    def on_reset(self, handle) -> None:
        """Called after every reset or state restore, before the first observation is used. `handle` is a pwm.sim
        backend (or the raw env). Subclasses call super().on_reset. Edits are idempotent: they start from the
        scene's nominal values, so a restore without a rebuild followed by on_reset does not compound them."""
        self.episode += 1
        self.rng = np.random.default_rng(1000 + self.episode)
        self._handle, self._fresh = handle, self.reobserve
        self.start_episode()

    def start_episode(self) -> None:
        """Restart per-episode state (action buffers)."""

    def wrap_action(self, cmd: np.ndarray, t: int) -> np.ndarray:
        """Map the command at control step t to the action the simulator executes."""
        return cmd

    def on_step(self, handle, t: int) -> None:
        """Called once per control step t (pwm.sim.backends.base.run_episode calls it right after step t, i.e. it
        prepares step t+1): time-varying edits, external forces."""

    def obs(self, obs: Any) -> Any:
        """The observation to use. Right after on_reset of a shift that changes what the camera sees, the observation
        returned by reset/restore predates the edit, so it is re-rendered once (and the backend's cached observation
        updated); otherwise unchanged."""
        if getattr(self, "_fresh", False):
            self._fresh = False
            from pwm.shifts.registry import refresh_obs
            return refresh_obs(self.robot, self._handle)
        return obs

    # ---------------------------------------------------------------- bookkeeping
    def describe(self) -> dict:
        return {"shift": self.name, "shift_level": self.level, "shift_value": _jsonable(self.value),
                "shift_kind": self.kind}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.robot}, {self.name}, L{self.level}={self.value})"


class NullShift(Shift):
    """No shift (level 0, or name 'none')."""

    def __init__(self, *, robot: str, facts: Optional[dict] = None):
        super().__init__(robot=robot, name="none", level=0, value=None, facts=facts or {})


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v
