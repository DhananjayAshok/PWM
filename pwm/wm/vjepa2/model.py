"""V-JEPA 2-AC model: Meta's frozen V-JEPA 2 ViT-g encoder and action-conditioned predictor (src/vjepa2), built
exactly as the released checkpoint expects (src/hub/backbones.py, _make_vjepa2_ac_model), with the predictor's
action and state projections resized to the robot's dimensions.

Projections: the release maps 7-D DROID actions (end-effector pose deltas + gripper) and 7-D states. For a robot whose
command is 7-D (Franka, WidowX, Google), a model step stacks c commands; the action projection is initialised by tiling
the released weights /c (the projection of the mean command). Any other dimension (GR-1's 29-D joints, non-7-D states)
starts fresh (truncated normal, std 0.02, as the predictor's own init).
"""
import os
from typing import Any

import torch
import torch.nn.functional as F

from pwm.utils import log_info

TOKENS_PER_FRAME = 256  # (256 / 16) ** 2
MAX_FRAMES = 32  # the predictor's causal mask is built for num_frames=64 at tubelet 2


def checkpoint_path(*, parameters: dict[str, Any]) -> str:
    """The released V-JEPA 2-AC checkpoint, downloaded once into model_dir."""
    path = os.path.join(parameters["model_dir"], "vjepa2", os.path.basename(parameters["vjepa2"]["checkpoint_url"]))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.hub.download_url_to_file(parameters["vjepa2"]["checkpoint_url"], path + ".part", progress=False)
        os.replace(path + ".part", path)
    return path


def _clean(state_dict: dict) -> dict:
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}


def _resize_linear(layer: torch.nn.Linear, new_in: int, tile: bool) -> torch.nn.Linear:
    if layer.in_features == new_in:
        return layer
    new = torch.nn.Linear(new_in, layer.out_features, bias=True)
    with torch.no_grad():
        if tile and new_in % layer.in_features == 0:
            c = new_in // layer.in_features
            new.weight.copy_(layer.weight.repeat(1, c) / c)
            new.bias.copy_(layer.bias)
        else:
            torch.nn.init.trunc_normal_(new.weight, std=0.02)
            new.bias.zero_()
    return new


def build(*, parameters: dict[str, Any], action_dim: int, state_dim: int, device: str = "cuda",
          predictor_init: str = "release", predictor_depth: int = 24):
    """Encoder (frozen, eval; always the release's) and predictor (trainable), projections resized.
    predictor_init: "release" loads Meta's V-JEPA 2-AC predictor (with predictor_depth < 24, its first blocks);
    "fresh" keeps the predictor's own random init."""
    from src.models import ac_predictor, vision_transformer

    encoder = vision_transformer.__dict__["vit_giant_xformers"](
        patch_size=16, img_size=(256, 256), num_frames=64, tubelet_size=2, use_sdpa=True, use_SiLU=False,
        wide_SiLU=True, uniform_power=False, use_rope=True)
    predictor = ac_predictor.__dict__["vit_ac_predictor"](
        img_size=(256, 256), patch_size=16, num_frames=64, tubelet_size=2, embed_dim=encoder.embed_dim,
        depth=predictor_depth)
    state = torch.load(checkpoint_path(parameters=parameters), map_location="cpu", weights_only=False)
    missing, unexpected = encoder.load_state_dict(_clean(state["encoder"]), strict=False)
    if missing or unexpected:
        log_info(f"[vjepa2] encoder load: {len(missing)} missing, {len(unexpected)} unexpected keys", parameters=parameters)
    if predictor_init == "release":
        released = _clean(state["predictor"])
        keep = {k: v for k, v in released.items()
                if not k.startswith("predictor_blocks.") or int(k.split(".")[1]) < predictor_depth}
        predictor.load_state_dict(keep, strict=True)
    else:
        assert predictor_init == "fresh", predictor_init
    del state
    base_dim = predictor.action_encoder.in_features
    predictor.action_encoder = _resize_linear(predictor.action_encoder, action_dim, tile=predictor_init == "release")
    predictor.state_encoder = _resize_linear(predictor.state_encoder, state_dim, tile=False)
    log_info(f"[vjepa2] predictor {predictor_init}, depth {predictor_depth}; action projection {base_dim}->{action_dim}, "
             f"state projection {base_dim}->{state_dim}", parameters=parameters)
    encoder.requires_grad_(False).eval()
    return encoder.to(device), predictor.to(device)


@torch.no_grad()
def encode(encoder, frames: torch.Tensor) -> torch.Tensor:
    """frames (B, T, 3, 256, 256) normalised -> layer-normed tokens (B, T * 256, D). Each frame is encoded on its own
    as a 2-frame tubelet (the frame repeated), as in V-JEPA 2-AC's training."""
    b, t = frames.shape[:2]
    x = frames.flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)  # (B*T, 3, 2, H, W)
    h = encoder(x)
    h = h.view(b, t, -1, h.size(-1)).flatten(1, 2)
    return F.layer_norm(h, (h.size(-1),))


def step(predictor, z: torch.Tensor, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """One predictor pass: tokens of frames 0..n-1 + n actions/states -> layer-normed predictions of frames 1..n."""
    out = predictor(z, actions, states)
    return F.layer_norm(out, (out.size(-1),))


def losses(predictor, h: torch.Tensor, actions: torch.Tensor, states: torch.Tensor, rollout_steps: int):
    """V-JEPA 2-AC's objective (app/vjepa_droid/train.py): L1 of one teacher-forced pass over the clip (every frame
    predicted from the real earlier frames), and L1 of a `rollout_steps` free-running rollout from the first frame
    (each further frame from the model's own earlier predictions; 0 = none; the release uses 2 steps).
    Returns (tf, rollout); the rollout loss is 0 when rollout_steps = 0."""
    tpf = TOKENS_PER_FRAME
    z_tf = step(predictor, h[:, :-tpf], actions, states[:, :-1])
    tf = torch.mean(torch.abs(z_tf - h[:, tpf:]))
    if rollout_steps == 0:
        return tf, torch.zeros((), device=h.device)
    assert 1 + rollout_steps <= h.size(1) // tpf, "window too short for the rollout"
    z = h[:, :tpf]
    for n in range(1, 1 + rollout_steps):  # n frames available, predict frame n
        nxt = step(predictor, z, actions[:, :n], states[:, :n])[:, -tpf:]
        z = torch.cat([z, nxt], dim=1)
    return tf, torch.mean(torch.abs(z[:, tpf:] - h[:, tpf : (1 + rollout_steps) * tpf]))


@torch.no_grad()
def rollout(predictor, h_ctx: torch.Tensor, actions: torch.Tensor, states: torch.Tensor, horizon: int) -> torch.Tensor:
    """Autoregressive prediction in feature space. h_ctx: encoded context frames (B, C*256, D); actions/states for
    C - 1 + horizon steps (states: the start state repeated). Returns predicted tokens of the next `horizon` frames,
    (B, horizon, 256, D). Each new frame is predicted from the real context and the model's own earlier predictions."""
    tpf = TOKENS_PER_FRAME
    n_ctx = h_ctx.size(1) // tpf
    assert n_ctx + horizon <= MAX_FRAMES, f"context + horizon must be <= {MAX_FRAMES} frames"
    z = h_ctx
    for k in range(horizon):
        n = n_ctx + k  # frames available
        nxt = step(predictor, z, actions[:, :n], states[:, :n])[:, -tpf:]
        z = torch.cat([z, nxt], dim=1)
    return z[:, n_ctx * tpf :].view(z.size(0), horizon, tpf, -1)
