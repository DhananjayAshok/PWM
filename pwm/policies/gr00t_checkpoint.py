"""Prepare a GR00T checkpoint for serving: download it (pinned revision) into the Hugging Face cache and, if its
backbone is gated, build a local view of it that points at the ungated substitute (configs/robots.yaml,
``gr00t_backbone_substitutes``).

The local view <model_dir>/policies/<repo>@<revision> holds symlinks to the cached files, except config.json and
processor_config.json, whose ``model_name`` is replaced by a local copy of the substitute. GR00T picks the backbone class
by matching "Qwen/Qwen3-VL" in that path, so the substitute keeps its <org>/<name> layout.

    python -m pwm.policies.gr00t_checkpoint --robot widowx_bridge     # prints the path to serve
"""
import json
import os
from typing import Any

import click
from huggingface_hub import snapshot_download

from pwm.robots import get_robot
from pwm.utils import load_parameters, log_info

IGNORE = ["global_step*", "*.pt", "optimizer*"]  # training state shipped with some checkpoints, not needed to serve


def _link_tree(*, src: str, dst: str) -> None:
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            target = os.path.join(dst, rel, f)
            if not os.path.lexists(target):
                try:
                    os.symlink(os.path.realpath(os.path.join(root, f)), target)
                except FileExistsError:   # created by a concurrent job
                    pass


def prepare(*, repo: str, revision: str, parameters: dict[str, Any]) -> str:
    """Return a local checkpoint directory that GR00T can load without access to gated repos."""
    cached = snapshot_download(repo, revision=revision, ignore_patterns=IGNORE)
    with open(os.path.join(cached, "config.json")) as f:
        backbone = json.load(f).get("model_name")
    substitute = parameters["gr00t_backbone_substitutes"].get(backbone)
    if substitute is None:
        return cached
    sub_dir = os.path.join(parameters["model_dir"], "policies", substitute["repo"])
    _link_tree(src=snapshot_download(substitute["repo"], revision=substitute["revision"]), dst=sub_dir)
    view = os.path.join(parameters["model_dir"], "policies", f"{repo}@{revision[:7]}")
    _link_tree(src=cached, dst=view)
    for name, section in (("config.json", None), ("processor_config.json", "processor_kwargs")):
        path = os.path.join(view, name)
        with open(os.path.realpath(path)) as f:
            config = json.load(f)
        (config[section] if section else config)["model_name"] = sub_dir
        tmp = f"{path}.tmp{os.getpid()}"   # several jobs may prepare the same view at once: replace atomically
        with open(tmp, "w") as f:
            json.dump(config, f, indent=2)
        os.replace(tmp, path)               # replaces the symlink into the cache (or an earlier copy)
    log_info(f"{repo}: backbone {backbone} -> {substitute['repo']} ({view})", parameters=parameters)
    return view


@click.command()
@click.option("--robot", required=True)
def main(robot):
    parameters = load_parameters()
    spec = get_robot(robot=robot, parameters=parameters)
    print(prepare(repo=spec["policy"], revision=spec["policy_revision"], parameters=parameters))


if __name__ == "__main__":
    main()
