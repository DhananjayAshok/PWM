"""Action-side shifts: wrappers between the command (`action_cmd`) and what the simulator executes (`action_env`).

Command layouts (plans/data_format.md):
  franka_libero  7: delta position (normalised; x 0.05 m), delta rotation (normalised; x 0.5 rad), gripper (-1 open, +1 close)
  widowx_bridge  7: delta position (m), delta rotation (rpy, rad), gripper (+1 open, -1 close) after Bridge's thresholding
  google_fractal 7: delta position (m), delta rotation (axis-angle, rad), gripper relative command after the sticky filter
                    (0 = no change)
  gr1_robocasa  29: absolute joint targets, order left_arm 7, right_arm 7, left_hand 6, right_hand 6, waist 3
"""
from collections import deque

import numpy as np

from pwm.shifts.base import Shift

DELTA_ROBOTS = ("franka_libero", "widowx_bridge", "google_fractal")


def idle_action(robot: str, previous: np.ndarray, first: np.ndarray) -> np.ndarray:
    """What the robot executes when no command has arrived yet: hold still, keep the gripper as it is."""
    if robot == "gr1_robocasa":  # absolute targets: hold the last target (or the first one)
        return (previous if previous is not None else first).copy()
    a = np.zeros_like(first)
    if robot == "google_fractal":  # relative gripper: 0 = no change
        return a
    open_value = -1.0 if robot == "franka_libero" else 1.0
    a[6] = previous[6] if previous is not None else open_value
    return a


class _BufferedShift(Shift):
    kind = "action"

    def start_episode(self):
        self.history, self.previous = deque(), None

    def _emit(self, a):
        self.previous = a
        return a


class Latency(_BufferedShift):
    """Every command is executed `steps` control steps late; before that the robot idles."""

    def wrap_action(self, cmd, t):
        steps = int(self.value["steps"])
        self.history.append(np.asarray(cmd, float).copy())
        if len(self.history) > steps:
            return self._emit(self.history.popleft())
        return self._emit(idle_action(self.robot, self.previous, self.history[0]))


class LatencyJitter(_BufferedShift):
    """Variable delay: each step the executed command is 0..max_steps old (bounded random walk, never reorders)."""

    def start_episode(self):
        super().start_episode()
        self.delay, self.last_index = 0, -1

    def wrap_action(self, cmd, t):
        max_steps = int(self.value["max_steps"])
        self.history.append(np.asarray(cmd, float).copy())
        self.delay = int(np.clip(self.delay + self.rng.integers(-1, 2), 0, max_steps))
        index = max(len(self.history) - 1 - self.delay, self.last_index)  # monotone: never go back in time
        if index < 0:
            return self._emit(idle_action(self.robot, self.previous, self.history[0]))
        self.last_index = index
        return self._emit(self.history[index].copy())


class FrameRotation(Shift):
    """Miscalibrated base frame: commanded xy translation rotated about the vertical axis."""
    kind = "action"

    def wrap_action(self, cmd, t):
        c, s = np.cos(np.radians(self.value)), np.sin(np.radians(self.value))
        a = np.asarray(cmd, float).copy()
        a[:2] = [c * a[0] - s * a[1], s * a[0] + c * a[1]]
        return a
