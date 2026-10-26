"""Make DreamDojo importable and runnable from the dreamdojo environment. Import before anything from cosmos_predict2.

- chdirs into the DreamDojo source (its code uses repo-relative paths) and links checkpoints/DreamDojo to PWM's asset
  directory (<model_dir>/dreamdojo, filled by pwm.wm.dreamdojo.assets);
- redirects the Wan2.1 VAE to the public Wan-AI copy (nvidia/Cosmos-Predict2.5-2B, which DreamDojo references, is
  gated; its tokenizer.pth is the same file);
- returns a dummy path for checkpoints it cannot resolve (gated Cosmos-Predict2.5 base weights that the configs name as
  a training init path but nothing loads);
- replaces DreamDojo's latent-action model (LAM, 710M parameters) with a stub that returns zeros: with native actions
  the latent-action slots are zero, and the model multiplies them by the LAM output, so the LAM never matters.
"""
import os

import torch

DD_SRC = os.environ.get("PWM_DREAMDOJO_SRC")
if not DD_SRC or not os.path.isdir(DD_SRC):
    raise RuntimeError("PWM_DREAMDOJO_SRC is not set: activate the dreamdojo environment (use_env dreamdojo)")
ACTION_CONFIG = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
LATENT_SLOTS = slice(352, 384)
ACTION_DIM = 384


def asset_dir(parameters) -> str:
    return os.path.join(parameters["model_dir"], "dreamdojo")


def setup(parameters) -> None:
    """Idempotent: paths, symlinks and patches. Call once before importing cosmos_predict2 model code."""
    if getattr(setup, "_done", False):
        return
    assets = asset_dir(parameters)
    os.makedirs(assets, exist_ok=True)
    os.chdir(DD_SRC)
    os.makedirs("checkpoints", exist_ok=True)
    link = "checkpoints/DreamDojo"
    if os.path.islink(link) and os.readlink(link) != assets:
        os.remove(link)
    if not os.path.lexists(link):
        try:
            os.symlink(assets, link)
        except FileExistsError:  # another rank of the same job created it first
            pass

    from cosmos_predict2._src.imaginaire.utils import checkpoint_db
    original = checkpoint_db.get_checkpoint_path
    vae = os.path.join(assets, parameters["dreamdojo"]["assets"]["vae_file"])

    def get_checkpoint_path(uri: str) -> str:
        if "Wan2.1_VAE.pth" in uri or uri.endswith("tokenizer.pth"):
            return vae
        try:
            return original(uri)
        except Exception as e:
            print(f"[dreamdojo.runtime] cannot resolve {uri} ({type(e).__name__}); dummy path (never loaded)", flush=True)
            return "/tmp"

    checkpoint_db.get_checkpoint_path = get_checkpoint_path

    import external.lam.model as lam_module

    class ZeroLatentActions(torch.nn.Module):
        """Stands in for DreamDojo's LAM wrapper: `.lam(batch)["z_rep"]` is zeros of the shape forward() expects."""

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.lam = self

        def forward(self, batch):
            videos = batch["videos"]  # ((b p), 2, H, W, C)
            return {"z_rep": torch.zeros(videos.shape[0], 1, 1, 32, device=videos.device)}

    lam_module.LAM = ZeroLatentActions
    setup._done = True
