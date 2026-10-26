"""LoRA for DreamDojo's DiT (fine-tuning scope "lora"): every attention / MLP projection of the transformer blocks
(q_proj, k_proj, v_proj, output_proj, mlp.layer1, mlp.layer2; plain nn.Linear in DreamDojo's DiT) gets a trainable
low-rank update W + (alpha / r) B A with B initialised to zero; the action embedders train fully; everything else is
frozen. Exports merge the update into W, so inference-format weights are the same as for a full fine-tune.
"""
import math

import torch
from torch import nn

TARGETS = ("q_proj", "k_proj", "v_proj", "output_proj", "layer1", "layer2")
FULLY_TRAINED = ("action_embedder",)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.to(x.dtype).t() @ self.lora_b.to(x.dtype).t()) * self.scale

    def merged_weight(self, lora_a=None, lora_b=None) -> torch.Tensor:
        a = self.lora_a if lora_a is None else lora_a
        b = self.lora_b if lora_b is None else lora_b
        return self.base.weight.detach().float().cpu() + self.scale * (b.float().cpu() @ a.float().cpu())


def apply(*, net: nn.Module, rank: int, alpha: float) -> list[nn.Parameter]:
    """Wrap the targets in the transformer blocks; returns the trainable parameters (LoRA factors + action embedders),
    in a deterministic order. Everything else gets requires_grad False."""
    for p in net.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, module in list(net.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if child_name in TARGETS and isinstance(child, nn.Linear) and ".blocks." in f".{full}":
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                wrapped.append(full)
    if not wrapped:
        raise RuntimeError("LoRA: no target layers found in the DiT")
    trainable = []
    for name, p in net.named_parameters():
        if "lora_" in name or any(t in name for t in FULLY_TRAINED):
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def export_state_dict(*, net: nn.Module, params: list[nn.Parameter], values: list[torch.Tensor] | None) -> dict:
    """Inference-format state dict (`net.*`, floats bf16) with LoRA merged; `values` (e.g. EMA) replace the trainable
    parameters' values when given."""
    value_of = {id(p): v for p, v in zip(params, values)} if values is not None else {}

    def val(p):
        return value_of.get(id(p), p.detach())

    out = {}
    named = dict(net.named_parameters())
    lora_prefixes = {}
    for name, module in net.named_modules():
        if isinstance(module, LoRALinear):
            lora_prefixes[name] = module
    for name, v in net.state_dict().items():
        if ".lora_" in name:
            continue
        prefix = next((p for p in lora_prefixes if name.startswith(p + ".base.")), None)
        if prefix is not None:
            module = lora_prefixes[prefix]
            if name == prefix + ".base.weight":
                v = module.merged_weight(val(module.lora_a), val(module.lora_b))
            name = prefix + name[len(prefix) + len(".base"):]
        else:
            p = named.get(name)
            if p is not None and id(p) in value_of:
                v = value_of[id(p)]
        if torch.is_tensor(v):
            v = v.detach().cpu()
            v = v.to(torch.bfloat16) if v.is_floating_point() else v
        out[f"net.{name}"] = v
    return out
