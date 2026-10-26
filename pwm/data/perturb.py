"""Perturbed replays of a command sequence (configs/data.yaml replay_types): noise | grip | branch.

Commands are perturbed in the robot's own action space; `scale` (backend.noise_scale) says which dimensions carry arm
motion and how large a typical step is, `gripper` which dimensions are the gripper/hands.
"""
import numpy as np

from pwm.sim.backends.base import ou_noise


def gripper_dims(*, robot):
    return {"gr1_robocasa": list(range(14, 26))}.get(robot, [6])


def perturb(*, actions, kind, rng, robot, scale, sigmas, control_hz):
    """
    Return (perturbed actions, metadata).

    noise:  smooth OU noise on the arm dimensions, sigma drawn between the 2nd and 4th noise level.
    grip:   the gripper command inverted for a short window in the middle 20-80% of the sequence (drops, regrasps).
    branch: from a random point in the middle, the rest replaced by 2-5 s of smooth random motion around the last
            command (absolute-target robots) or of random small deltas (delta-action robots).
    """
    A = np.asarray(actions, np.float64).copy()
    T = len(A)
    meta = {"replay_type": kind}
    if kind == "noise":
        sigma = float(rng.uniform(sigmas[1], sigmas[3]))
        A += ou_noise(rng=rng, steps=T, dim=A.shape[1], sigma=1.0) * sigma * scale
        meta["sigma"] = sigma
    elif kind == "grip":
        at = int(rng.integers(int(0.2 * T), max(int(0.2 * T) + 1, int(0.8 * T))))
        length = int(rng.integers(control_hz // 4 + 1, control_hz + 2))
        dims = gripper_dims(robot=robot)
        if robot == "gr1_robocasa":
            lo, hi = A[:, dims].min(0), A[:, dims].max(0)
            A[at:at + length, dims] = lo + hi - A[at:at + length, dims]   # open <-> closed within the episode's range
        else:
            A[at:at + length, dims] *= -1
        meta.update(at=at, length=length)
    elif kind == "branch":
        at = int(rng.integers(int(0.2 * T), max(int(0.2 * T) + 1, int(0.8 * T))))
        length = int(rng.integers(2 * control_hz, 5 * control_hz + 1))
        wander = ou_noise(rng=rng, steps=length, dim=A.shape[1], sigma=1.0) * sigmas[3] * scale
        if robot == "gr1_robocasa":      # absolute joint targets: wander around the last command
            B = A[at - 1][None] + wander
        else:                            # EE deltas: random small motion, gripper held
            B = wander
            B[:, gripper_dims(robot=robot)] = A[at - 1, gripper_dims(robot=robot)]
        A = np.concatenate([A[:at], B])
        meta.update(at=at, length=length)
    else:
        raise ValueError(f"unknown replay type {kind}")
    return A, meta
