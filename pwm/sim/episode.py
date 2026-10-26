"""Episode records, the same layout for every robot.

A rollout directory holds, per episode ``<episode_id>``:
  <episode_id>.npz    per-step arrays: ``action_env`` (T, A) the action the simulator executed, ``action_policy`` (T, A')
                      the raw policy output, ``state_policy`` (T, S) the state the policy saw, ``success`` (T,), plus
                      robot-specific arrays (full simulator state, end-effector pose, ...)
  <episode_id>.mp4    the policy camera, frame t = observation before action t, last frame = final observation
and one line per finished episode in ``episodes.jsonl`` (episode_id, task, success, steps, ...). An episode is written
atomically (temporary names, then os.replace), so a listed episode is complete; reruns skip listed episodes.
"""
import json
import os
from typing import Any, Optional

import imageio.v2 as iio
import numpy as np


class RolloutDir:
    def __init__(self, *, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self._index = os.path.join(path, "episodes.jsonl")

    def finished(self) -> dict[str, dict[str, Any]]:
        """Episodes already written, by episode_id."""
        if not os.path.exists(self._index):
            return {}
        with open(self._index) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return {r["episode_id"]: r for r in rows}

    def write(self, *, episode_id: str, arrays: dict[str, Any], frames: list, fps: float, summary: dict[str, Any],
              extra_videos: Optional[dict[str, list]] = None) -> None:
        """
        Write one episode (npz + mp4 + index line).

        :param episode_id: Unique name within this directory.
        :param arrays: Per-step arrays and scalars for the npz.
        :param frames: Policy-camera frames (H, W, 3) uint8.
        :param fps: Video frame rate (the control rate).
        :param summary: Fields of the episodes.jsonl line (episode_id is added).
        :param extra_videos: Further camera views, saved as ``<episode_id>_<name>.mp4``.
        """
        base = os.path.join(self.path, episode_id)
        np.savez_compressed(base + ".tmp.npz", **{k: np.asarray(v) for k, v in arrays.items()})
        os.replace(base + ".tmp.npz", base + ".npz")
        videos = {"": frames, **{"_" + k: v for k, v in (extra_videos or {}).items()}}
        for suffix, vid in videos.items():
            if len(vid) == 0:
                continue
            iio.mimwrite(base + suffix + ".tmp.mp4", [np.asarray(f, np.uint8) for f in vid], fps=fps, macro_block_size=8)
            os.replace(base + suffix + ".tmp.mp4", base + suffix + ".mp4")
        with open(self._index, "a") as f:
            f.write(json.dumps({"episode_id": episode_id, **summary}) + "\n")
