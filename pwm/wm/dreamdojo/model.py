"""Loading DreamDojo for training and prediction, and building its input batches.

The model is built exactly as for inference (DreamDojo's model loader + action-conditioned config), from an
inference-format checkpoint (`net.*` weights): the released 2B pretrained EMA weights to start, or a PWM training
export. The prompt is "" everywhere (as DreamDojo's action-conditioned post-training): its text embedding is computed
once with the text encoder, cached in <model_dir>/dreamdojo, and the 7B encoder is then dropped from the GPU.
"""
import os

import torch

from pwm.wm.dreamdojo import runtime

N_CHUNK = 12  # actions (= new frames) per generation chunk
FPS = 4.0     # value DreamDojo's datasets report; fps modulation is disabled in the action-conditioned nets


def load(*, parameters, weights: str):
    """Returns the DreamDojo model (tokenizer, conditioner, net on GPU; text encoder replaced by the cached "" embedding)."""
    runtime.setup(parameters)
    from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(
        experiment_name=parameters["dreamdojo"]["assets"]["experiment"], s3_checkpoint_dir=weights,
        config_file=runtime.ACTION_CONFIG, load_ema_to_reg=True, experiment_opts=["~data_train"], to_device="cuda")
    cache = os.path.join(runtime.asset_dir(parameters), "empty_prompt_embedding.pt")
    if os.path.exists(cache):
        emb = torch.load(cache, map_location="cuda")
    else:
        with torch.no_grad():
            emb = model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [""], "images": None}, input_caption_key="ai_caption")
        if int(os.environ.get("RANK", "0")) == 0:
            torch.save(emb.cpu(), cache + f".tmp{os.getpid()}")
            os.replace(cache + f".tmp{os.getpid()}", cache)
    model.pwm_empty_prompt = emb
    model.text_encoder.compute_text_embeddings_online = lambda data_batch, input_caption_key: emb.expand(
        len(data_batch[input_caption_key]), *emb.shape[1:])
    if hasattr(model.text_encoder, "model"):
        model.text_encoder.model = None
    torch.cuda.empty_cache()
    return model


def batch(*, frames: torch.Tensor, actions: torch.Tensor, num_conditional_frames: int | None = None) -> dict:
    """frames (B, 13, H, W, 3) uint8 on GPU, actions (B, 12, 384) -> the data batch DreamDojo's forward / sampler take."""
    b, n, h, w, _ = frames.shape
    out = {
        "dataset_name": "video_data",
        "video": frames.permute(0, 4, 1, 2, 3).contiguous(),  # B C T H W uint8
        "action": actions,
        "lam_video": torch.zeros(b, 2 * (n - 1), 1, 1, 3, device=frames.device),  # shape only (LAM is a zero stub)
        "fps": torch.full((b,), FPS, device=frames.device),
        "padding_mask": torch.zeros(b, 1, h, w, device=frames.device),
        "ai_caption": [""] * b,
        "num_frames": n,
    }
    if num_conditional_frames is not None:
        out["num_conditional_frames"] = num_conditional_frames
    return out


def inference_batch(*, model, frames: torch.Tensor, actions: torch.Tensor) -> dict:
    """As `batch`, for the sampler: one conditioning frame, the cached "" embedding, floats in bf16 (as DreamDojo's
    own inference pipeline prepares them)."""
    out = batch(frames=frames, actions=actions, num_conditional_frames=1)
    out["t5_text_embeddings"] = model.pwm_empty_prompt.expand(frames.shape[0], *model.pwm_empty_prompt.shape[1:])
    for k, v in out.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            out[k] = v.cuda().to(torch.bfloat16)
    return out


def export_inference_weights(*, net: torch.nn.Module, ema_params: list | None) -> dict:
    """Inference-format state dict (`net.*`, floats in bf16), with EMA values for the parameters if given."""
    by_name = {n: e for (n, _), e in zip(net.named_parameters(), ema_params)} if ema_params is not None else {}
    out = {}
    for n, v in net.state_dict().items():
        v = by_name.get(n, v)
        if torch.is_tensor(v):
            v = v.detach().cpu()
            v = v.to(torch.bfloat16) if v.is_floating_point() else v
        out[f"net.{n}"] = v
    return out


def _inference_condition(model, data_batch):
    """The conditioned Condition of a data batch, prepared as DreamDojo's get_velocity_fn_from_batch does for inference."""
    from cosmos_predict2._src.predict2.conditioner import DataType
    n_cond = data_batch.get("num_conditional_frames", 1)
    condition, _ = model.conditioner.get_condition_uncondition(data_batch)
    condition = condition.edit_data_type(DataType.VIDEO)
    _, x0, _ = model.get_data_and_condition(data_batch)
    condition = condition.set_video_condition(
        gt_frames=x0, random_min_num_conditional_frames=model.config.min_num_conditional_frames,
        random_max_num_conditional_frames=model.config.max_num_conditional_frames, num_conditional_frames=n_cond,
        conditional_frames_probs=model.config.conditional_frames_probs)
    condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=n_cond)
    return condition, x0


@torch.no_grad()
def sample(*, model, batch: dict, seed: int, num_steps: int, shift: float = 5.0) -> torch.Tensor:
    """Generated latents (1, C, T, h, w) for one clip (an `inference_batch` of size 1): DreamDojo's rectified-flow Euler
    sampler at guidance 0 (same scheduler, shift and seeding as generate_samples_from_batch)."""
    from cosmos_predict2._src.imaginaire.utils import misc
    model._normalize_video_databatch_inplace(batch)
    model._augment_image_dim_inplace(batch)
    cond, x0 = _inference_condition(model, batch)
    device = model.tensor_kwargs["device"]
    noise = misc.arch_invariant_rand((1,) + tuple(x0.shape[1:]), torch.float32, device, seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model.sample_scheduler.set_timesteps(num_steps, device=device, shift=shift,
                                         use_kerras_sigma=model.config.use_kerras_sigma_at_inference)
    latents = noise
    for t in model.sample_scheduler.timesteps:
        v = model.denoise(noise, latents, torch.stack([t]).unsqueeze(0), cond)
        latents = model.sample_scheduler.step(v.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False,
                                              generator=gen)[0].squeeze(0)
    return latents
