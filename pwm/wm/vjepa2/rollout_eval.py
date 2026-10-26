"""Rollout metrics of a trained V-JEPA 2-AC run (reported next to the selection number, not used for selection).

On 32 fixed val episodes (evenly spaced in episode-id order), from 1 real context frame:
  (a) free-running rollout error over the first 4.8 s (pwm.wm.vjepa2.predict.horizon_steps): mean L1 and L2 between
      predicted and real layer-normed tokens (all 256 per frame), averaged over steps;
  (b) action sensitivity: the same error when each episode is rolled out with the commands of the next episode in the
      list (horizon = the shorter of the two), and the ratio shuffled / true (1 = the model ignores its actions).
Written to <run dir>/rollout_eval.json.

    python -m pwm.wm.vjepa2.rollout_eval --robot franka_libero --checkpoint <run dir>/best.pt
"""
import json
import os

import click
import numpy as np
import torch

from pwm.utils import load_parameters, log_info
from pwm.wm.vjepa2 import data, model
from pwm.wm.vjepa2.predict import CONTEXT_FRAMES, horizon_steps

N_EPISODES = 32


@torch.no_grad()
def _errors(predictor, h_real, actions, states, horizon):
    pred = model.rollout(predictor, h_real[:, : CONTEXT_FRAMES * model.TOKENS_PER_FRAME], actions, states, horizon)
    real = h_real[:, CONTEXT_FRAMES * model.TOKENS_PER_FRAME :].view_as(pred)
    d = (pred.float() - real.float())
    return d.abs().mean().item(), d.pow(2).mean().item()


@torch.no_grad()
def run(*, parameters, robot: str, dataset: str, checkpoint: str, encoder=None, device: str = "cuda") -> dict:
    cfg = data.robot_cfg(parameters=parameters, robot=robot)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run_cfg = ck.get("cfg", {})
    if encoder is None:
        encoder, predictor = model.build(parameters=parameters, action_dim=ck["action_dim"], state_dim=ck["state_dim"],
                                         device=device, predictor_init="fresh",
                                         predictor_depth=run_cfg.get("predictor_depth", 24))
    else:
        from src.models import ac_predictor
        predictor = ac_predictor.__dict__["vit_ac_predictor"](
            img_size=(256, 256), patch_size=16, num_frames=64, tubelet_size=2, embed_dim=encoder.embed_dim,
            depth=run_cfg.get("predictor_depth", 24))
        predictor.action_encoder = torch.nn.Linear(ck["action_dim"], predictor.action_encoder.out_features)
        predictor.state_encoder = torch.nn.Linear(ck["state_dim"], predictor.state_encoder.out_features)
    predictor.load_state_dict(ck["predictor"])
    predictor = predictor.to(device).eval()
    eps = sorted(data.load_episodes(parameters=parameters, robot=robot, dataset=dataset, split="val", min_frames=2),
                 key=lambda e: e.row["episode_id"])
    eps = [eps[i] for i in np.linspace(0, len(eps) - 1, min(N_EPISODES, len(eps))).round().astype(int)]
    rows = []
    for i, ep in enumerate(eps):
        other = eps[(i + 1) % len(eps)]
        hz = min(horizon_steps(ep=ep, cfg=cfg), horizon_steps(ep=other, cfg=cfg))
        frames, actions, states = ep.window(start=0, length=CONTEXT_FRAMES + hz)
        _, actions_other, _ = other.window(start=0, length=CONTEXT_FRAMES + hz)
        x = data.to_model_input(frames).unsqueeze(0).to(device)
        s = torch.from_numpy(states).unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(encoder, x)
            true = _errors(predictor, h, torch.from_numpy(actions).unsqueeze(0).to(device), s, hz)
            shuf = _errors(predictor, h, torch.from_numpy(actions_other).unsqueeze(0).to(device), s, hz)
        rows.append({"episode_id": ep.row["episode_id"], "steps": hz, "true_l1": true[0], "true_l2": true[1],
                     "shuffled_l1": shuf[0], "shuffled_l2": shuf[1]})
    mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    out = {"n_episodes": len(rows), "horizon_s": 4.8, "context_frames": CONTEXT_FRAMES,
           "rollout_l1": mean("true_l1"), "rollout_l2": mean("true_l2"),
           "shuffled_l1": mean("shuffled_l1"), "shuffled_l2": mean("shuffled_l2"),
           "action_sensitivity_l1": mean("shuffled_l1") / mean("true_l1"),
           "action_sensitivity_l2": mean("shuffled_l2") / mean("true_l2"),
           "checkpoint": os.path.abspath(checkpoint), "episodes": rows}
    path = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "rollout_eval.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(path + ".tmp", path)
    log_info(f"[vjepa2] rollout eval {path}: L1 {out['rollout_l1']:.4f}, shuffled {out['shuffled_l1']:.4f} "
             f"(x{out['action_sensitivity_l1']:.3f})", parameters=parameters)
    return out


@click.command()
@click.option("--robot", required=True)
@click.option("--dataset", default="nominal")
@click.option("--checkpoint", required=True)
def main(robot, dataset, checkpoint):
    run(parameters=load_parameters(), robot=robot, dataset=dataset, checkpoint=checkpoint)


if __name__ == "__main__":
    main()
