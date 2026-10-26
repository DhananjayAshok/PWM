"""DreamerV3's world model (the PyTorch port's models.WorldModel: CNN encoder/decoder, RSSM, reward and continue heads)
built from the port's configs.yaml defaults plus our overrides (configs/world_models.yaml, ``dreamerv3``).

Only the world model is used; there is no actor-critic. The reward head is trained on our per-step task success, and
the continue head sees no terminal steps (offline windows never end an episode).
"""
import os
import types
from typing import Any

import numpy as np
import torch
import yaml

import models   # dreamerv3-torch (on PYTHONPATH via setup/envs/dreamerv3/env.sh)
import tools

# Model sizes. small/medium/large: first presets; sNNm: sizes named by their parameter count at 64 px
# (13.0M / 27.9M / 55.6M / 105.5M; at 128 px the CNNs add more). Only the keys that change from the port's defaults.
def _size(hidden, deter, units, depth, mlp_units=None):
    mlp_units = mlp_units or units
    return {"dyn_hidden": hidden, "dyn_deter": deter, "units": units,
            "encoder": {"cnn_depth": depth, "mlp_units": mlp_units}, "decoder": {"cnn_depth": depth, "mlp_units": mlp_units}}


SIZES = {
    "small": {},
    "medium": _size(768, 1024, 768, 48),
    "large": _size(1024, 2048, 1024, 64),
    "s12m": {"dyn_hidden": 512, "dyn_deter": 512, "units": 512, "encoder": {"cnn_depth": 24}, "decoder": {"cnn_depth": 24}},
    "s25m": {"dyn_hidden": 640, "dyn_deter": 1024, "units": 640, "encoder": {"cnn_depth": 40}, "decoder": {"cnn_depth": 40}},
    "s50m": {"dyn_hidden": 768, "dyn_deter": 2048, "units": 768, "encoder": {"cnn_depth": 48}, "decoder": {"cnn_depth": 48}},
    "s100m": {"dyn_hidden": 1024, "dyn_deter": 3072, "units": 1024, "encoder": {"cnn_depth": 64}, "decoder": {"cnn_depth": 64}},
}


def _update(base: dict, update: dict) -> None:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _update(base[k], v)
        else:
            base[k] = v


def port_defaults() -> dict[str, Any]:
    """The port's ``defaults`` block, with its string-typed numbers ("1e-4") converted as its dreamer.py does."""
    path = os.path.join(os.path.dirname(models.__file__), "configs.yaml")
    with open(path) as f:
        raw = yaml.safe_load(f)["defaults"]
    out = {}
    for k, v in raw.items():
        out[k] = tools.args_type(v)(v) if not isinstance(v, dict) else v
    return out


def make_config(*, settings: dict[str, Any], action_dim: int, device: str = "cuda:0") -> types.SimpleNamespace:
    """``settings`` = configs/world_models.yaml ``dreamerv3`` block (size, image_size, precision, model overrides)."""
    cfg = port_defaults()
    _update(cfg, SIZES[settings["size"]])
    _update(cfg, settings.get("overrides") or {})
    cfg["size"] = (settings["image_size"], settings["image_size"])
    cfg["precision"] = settings["precision"]
    cfg["num_actions"] = action_dim
    cfg["device"] = device
    cfg["grad_heads"] = ["decoder", "reward", "cont"]
    for k in ("model_lr", "opt_eps", "grad_clip", "weight_decay", "kl_free", "dyn_scale", "rep_scale", "discount",
              "unimix_ratio", "dyn_min_std"):
        cfg[k] = float(cfg[k])
    if settings.get("model_lr") is not None:
        cfg["model_lr"] = float(settings["model_lr"])
    return types.SimpleNamespace(**cfg)


def build_world_model(*, config: types.SimpleNamespace, image_size: int):
    """The port's WorldModel (its own Adam optimizer over every parameter)."""
    spaces = {"image": types.SimpleNamespace(shape=(image_size, image_size, 3))}
    obs_space = types.SimpleNamespace(spaces=spaces)
    act_space = types.SimpleNamespace(shape=(config.num_actions,))
    return models.WorldModel(obs_space, act_space, 0, config).to(config.device)


def normalize_actions(batch: dict[str, np.ndarray], stats: dict[str, list[float]]) -> dict[str, np.ndarray]:
    mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
    out = dict(batch)
    a = (batch["action"] - mean) / std
    a[batch["is_first"]] = 0.0
    out["action"] = np.clip(a, -10.0, 10.0).astype(np.float32)
    return out


def to_device(batch: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    """The port's WorldModel.preprocess for numpy batches, moving images as uint8 (4x less host-to-device traffic)."""
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True)
        out[k] = t.float() / 255.0 if k == "image" else t.float()
    out["cont"] = (1.0 - out["is_terminal"]).unsqueeze(-1)
    return out


def world_model_loss(wm, batch: dict[str, np.ndarray]) -> tuple[torch.Tensor, dict[str, float]]:
    """The port's WorldModel._train objective without the optimizer step: KL (dynamics + representation) plus
    decoder, reward (success) and continue log-likelihoods, averaged over batch and time ("base objective"), also
    returned as the ``base_loss`` metric."""
    data = to_device(batch, wm._config.device)
    cfg = wm._config
    with torch.amp.autocast("cuda", enabled=wm._use_amp, dtype=torch.float16):
        embed = wm.encoder(data)
        post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
        kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(post, prior, cfg.kl_free, cfg.dyn_scale,
                                                                    cfg.rep_scale)
        feat = wm.dynamics.get_feat(post)
        preds = {}
        for name, head in wm.heads.items():
            pred = head(feat)
            preds.update(pred) if isinstance(pred, dict) else preds.__setitem__(name, pred)
        losses = {name: -pred.log_prob(data[name]) for name, pred in preds.items()}
        base = (sum(v * wm._scales.get(k, 1.0) for k, v in losses.items()) + kl_loss).mean()
    metrics = {f"{k}_loss": float(v.detach().mean()) for k, v in losses.items()}
    metrics.update(kl=float(kl_value.detach().mean()), dyn_loss=float(dyn_loss.detach().mean()),
                   rep_loss=float(rep_loss.detach().mean()), kl_term=float(kl_loss.detach().mean()),
                   base_loss=float(base.detach()))
    return base, metrics


def save_checkpoint(*, path: str, wm, step: int, extra: dict[str, Any]) -> None:
    state = {"model": wm.state_dict(), "optimizer": wm._model_opt._opt.state_dict(),
             "scaler": wm._model_opt._scaler.state_dict(), "step": step, **extra}
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(*, path: str, wm, with_optimizer: bool = True) -> dict[str, Any]:
    state = torch.load(path, map_location=wm._config.device, weights_only=False)
    wm.load_state_dict(state["model"])
    if with_optimizer:
        wm._model_opt._opt.load_state_dict(state["optimizer"])
        wm._model_opt._scaler.load_state_dict(state["scaler"])
    return state
