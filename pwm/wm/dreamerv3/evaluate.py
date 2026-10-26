"""Open-loop evaluation on fixed validation episodes, the same protocol as the eval panel
(plans/predictions_format.md): from recorded frame 0 (1 real context frame), imagine the first 4.8 s on the commands
alone, seed 0, sample 0.

- ``openloop_mse64``: mean squared error per pixel in [0, 1] at 64x64 (predictions of 128 px models are 2x2
  average-pooled; ground truth is the 64 px cache), averaged over frames, then over episodes. This is the training
  **selection and stopping number**, comparable across image resolutions.
- report (per finished run): MSE and LPIPS (AlexNet) at the model's native resolution, and **action sensitivity**:
  the same rollouts with every episode's commands replaced by another episode's (a fixed cyclic shift of the episode
  list), ``mse_shuffled / mse_true`` (1 = actions ignored; higher = the model follows its actions).
"""
import zlib
from typing import Optional

import numpy as np
import torch

from pwm.wm.dreamerv3.predict import imagine


def fixed_episodes(*, episodes: list[str], count: int) -> list[str]:
    """Deterministic episode sample (by crc32 of the id), the same for every run on a dataset."""
    return sorted(episodes, key=lambda e: zlib.crc32(e.encode()))[:count]


def _pool64(frames: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) uint8 at 64 * f -> (N, 64, 64, 3) float in [0, 1]."""
    x = frames.astype(np.float32) / 255.0
    f = x.shape[1] // 64
    if f > 1:
        n, h, w, c = x.shape
        x = x.reshape(n, 64, f, 64, f, c).mean((2, 4))
    return x


def _rollouts(*, wm, stats, replay, episodes, fps, horizon_s, context, action_source=None):
    out = []
    for i, e in enumerate(episodes):
        ep = replay.episode(e)
        n = max(0, min(int(round(horizon_s * fps.get(e, 10.0))), len(ep["image"]) - context))
        if n == 0:
            continue
        actions = ep["action"][:context + n].copy()
        if action_source is not None:  # commands of another episode (same length, padded with its last command)
            other = replay.episode(action_source[i])["action"][context:context + n]
            if len(other) < n:
                other = np.concatenate([other, np.repeat(other[-1:], n - len(other), 0)])
            actions[context:] = other
        pred = imagine(wm=wm, context=ep["image"][:context], actions=actions, stats=stats, seed=0)["frames"]
        out.append((e, pred, slice(context, context + n)))
    return out


@torch.no_grad()
def openloop_mse64(*, wm, stats, replay, replay64, episodes, fps, horizon_s: float = 4.8, context: int = 1) -> float:
    errs = []
    for e, pred, sl in _rollouts(wm=wm, stats=stats, replay=replay, episodes=episodes, fps=fps, horizon_s=horizon_s,
                                 context=context):
        truth = replay64.episode(e)["image"][sl].astype(np.float32) / 255.0
        errs.append(float(np.mean((_pool64(pred) - truth) ** 2)))
    return float(np.mean(errs))


@torch.no_grad()
def report(*, wm, stats, replay, episodes, fps, horizon_s: float = 4.8, context: int = 1,
           lpips_net: Optional[torch.nn.Module] = None) -> dict[str, float]:
    """Native-resolution open-loop MSE / LPIPS and action sensitivity on ``episodes``."""
    if lpips_net is None:
        import lpips
        lpips_net = lpips.LPIPS(net="alex", verbose=False).to(wm._config.device).eval()
    shuffled = episodes[1:] + episodes[:1]

    def score(rolls):
        mse, lp = [], []
        for e, pred, sl in rolls:
            truth = replay.episode(e)["image"][sl]
            p = torch.from_numpy(pred).to(wm._config.device).permute(0, 3, 1, 2).float() / 127.5 - 1
            t = torch.from_numpy(truth).to(wm._config.device).permute(0, 3, 1, 2).float() / 127.5 - 1
            mse.append(float(((p - t) / 2).pow(2).mean()))
            lp.append(float(lpips_net(p, t).mean()))
        return float(np.mean(mse)), float(np.mean(lp))

    mse, lp = score(_rollouts(wm=wm, stats=stats, replay=replay, episodes=episodes, fps=fps, horizon_s=horizon_s,
                              context=context))
    mse_s, lp_s = score(_rollouts(wm=wm, stats=stats, replay=replay, episodes=episodes, fps=fps, horizon_s=horizon_s,
                                  context=context, action_source=shuffled))
    return {"openloop_mse_native": mse, "openloop_lpips_native": lp, "shuffled_mse_native": mse_s,
            "shuffled_lpips_native": lp_s, "action_sensitivity_mse": mse_s / max(mse, 1e-12),
            "action_sensitivity_lpips": lp_s / max(lp, 1e-12), "report_episodes": len(episodes)}
