"""Download and prepare DreamDojo's released weights (configs/world_models.yaml, dreamdojo.assets) into
<model_dir>/dreamdojo:
  <init_checkpoint>/model_ema_bf16.pt   EMA weights of the released checkpoint in inference format (net.*, bf16)
  Wan2.1_VAE.pth                         the video VAE (symlink into the Hugging Face cache)
The raw downloads stay in the Hugging Face cache (HF_HOME). Idempotent. CPU only; the conversion needs ~40 GB RAM.

    python -m pwm.wm.dreamdojo.assets
"""
import os

import click
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from pwm.utils import load_parameters, log_info
from pwm.wm.dreamdojo.runtime import asset_dir


def init_weights_path(parameters) -> str:
    return os.path.join(asset_dir(parameters), parameters["dreamdojo"]["assets"]["init_checkpoint"], "model_ema_bf16.pt")


def convert_distcp(*, distcp_dir: str, out: str, parameters) -> None:
    """distcp checkpoint -> EMA weights only, renamed net_ema.* -> net.*, floats in bf16 (written atomically)."""
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    full = os.path.join(parameters["tmp_dir"], f"dreamdojo_full_{os.getpid()}.pt")
    dcp_to_torch_save(distcp_dir, full)
    state = torch.load(full, map_location="cpu", weights_only=False)
    os.remove(full)
    ema = {k.replace("net_ema.", "net.", 1): (v.bfloat16() if torch.is_tensor(v) and v.dtype == torch.float32 else v)
           for k, v in state.items() if k.startswith("net_ema.")}
    if not ema:
        raise RuntimeError(f"no EMA weights in {distcp_dir}")
    n = sum(v.numel() for v in ema.values() if torch.is_tensor(v)) / 1e9
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(ema, out + ".tmp")
    os.replace(out + ".tmp", out)
    log_info(f"converted {distcp_dir} -> {out} ({len(ema)} tensors, {n:.2f}B params)", parameters=parameters)


def prepare(parameters) -> None:
    cfg = parameters["dreamdojo"]["assets"]
    target = asset_dir(parameters)
    os.makedirs(target, exist_ok=True)
    out = init_weights_path(parameters)
    if not os.path.exists(out):
        root = snapshot_download(cfg["repo"], revision=cfg["revision"], max_workers=16,
                                 allow_patterns=[f"{cfg['init_checkpoint']}/model/*"])
        convert_distcp(distcp_dir=os.path.join(root, cfg["init_checkpoint"], "model"), out=out, parameters=parameters)
    vae = os.path.join(target, cfg["vae_file"])
    if not os.path.exists(vae):
        src = hf_hub_download(cfg["vae_repo"], cfg["vae_file"], revision=cfg["vae_revision"])
        os.symlink(os.path.realpath(src), vae)
    log_info(f"DreamDojo assets ready in {target}", parameters=parameters)


@click.command()
def main():
    prepare(load_parameters())


if __name__ == "__main__":
    main()
