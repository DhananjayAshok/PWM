"""Pixel metrics between a predicted and a real clip: PSNR, SSIM, LPIPS (whole frame), moving-region PSNR and motion IoU
(only where the real video changes), per frame and per 1.2 s chunk.

Resolution modes (plans: DreamDojo once at its native resolution; DreamerV3 twice):
  "pred": the real clip is resized to the prediction's resolution
  "real": the prediction is resized (bilinear) to the real clip's resolution
"""
import cv2
import numpy as np
import torch

_LPIPS = {}
CHANGE_THRESHOLD = 0.08      # |frame_t - frame_0| (grayscale, [0,1]) above which a pixel counts as moving


def _lpips(device):
    if device not in _LPIPS:
        import piq
        _LPIPS[device] = piq.LPIPS(reduction="none").to(device)
    return _LPIPS[device]


def resize(frames, hw):
    if frames.shape[1:3] == tuple(hw):
        return frames
    return np.stack([cv2.resize(f, (hw[1], hw[0]), interpolation=cv2.INTER_AREA if f.shape[0] > hw[0] else cv2.INTER_LINEAR)
                     for f in frames])


def _gray(x):
    return x.astype(np.float32).mean(-1) / 255.0


def frame_metrics(*, pred, real, device="cuda"):
    """pred, real: (T, H, W, 3) uint8 of equal shape; frame 0 = the shared context frame. Metrics on frames 1..T-1."""
    import piq
    p = torch.from_numpy(pred[1:]).permute(0, 3, 1, 2).float().div(255).to(device)
    r = torch.from_numpy(real[1:]).permute(0, 3, 1, 2).float().div(255).to(device)
    with torch.no_grad():
        mse = ((p - r) ** 2).flatten(1).mean(1)
        psnr = (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).cpu().numpy()
        ssim = piq.ssim(p, r, data_range=1.0, reduction="none").cpu().numpy()
        lpips = _lpips(device)(p, r).flatten().cpu().numpy()
    g0, gr, gp = _gray(real[0]), _gray(real[1:]), _gray(pred[1:])
    real_move = np.abs(gr - g0[None]) > CHANGE_THRESHOLD
    pred_move = np.abs(gp - _gray(pred[0])[None]) > CHANGE_THRESHOLD
    sq = ((pred[1:].astype(np.float32) - real[1:].astype(np.float32)) / 255.0) ** 2
    mr_psnr, iou = [], []
    for t in range(len(gr)):
        m = real_move[t]
        mr_psnr.append(float(10 * np.log10(1.0 / max(sq[t][m].mean(), 1e-10))) if m.any() else np.nan)
        union = (m | pred_move[t]).sum()
        iou.append(float((m & pred_move[t]).sum() / union) if union else np.nan)
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips, "moving_psnr": np.array(mr_psnr), "motion_iou": np.array(iou)}


def chunked(values, *, fps, chunk_s=1.2):
    """Mean of each metric over consecutive chunk_s windows (frames 1..T-1)."""
    n = max(1, int(round(chunk_s * fps)))
    return {k: [float(np.nanmean(v[i:i + n])) for i in range(0, len(v), n)] for k, v in values.items()}


def clip_metrics(*, pred, real, mode, fps, device="cuda"):
    if mode == "pred":
        real = resize(real, pred.shape[1:3])
    else:
        pred = resize(pred, real.shape[1:3])
    n = min(len(pred), len(real))
    per_frame = frame_metrics(pred=pred[:n], real=real[:n], device=device)
    return {"mean": {k: float(np.nanmean(v)) for k, v in per_frame.items()}, "chunks": chunked(per_frame, fps=fps)}
