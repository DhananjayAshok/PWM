"""Build the eval panel tables from <results_dir>/panel/<model>/<robot>/<tag>/<dataset>.json (core environment).

    python -m pwm.eval.tables [--tag best]

Writes <results_dir>/panel/panel.md: one table per robot for the no-shift test set (all models, all metrics), and
per robot x shift tables of each metric across levels L1-L5 next to the no-shift value.
"""
import glob
import json
import os
import re
from collections import defaultdict

import click

from pwm.utils import load_parameters, log_info

COLUMNS = [("counterfactual_acc", None, "counterfactual"), ("probe", "ee_cm", "EE err cm"),
           ("probe", "objects_cm", "object err cm"), ("probe", "grasped_acc", "grasp acc"),
           ("probe_floor", "ee_cm", "EE floor cm"), ("pixel_pred", "lpips", "LPIPS"), ("pixel_pred", "psnr", "PSNR"),
           ("pixel_pred", "moving_psnr", "moving PSNR"), ("pixel_pred", "motion_iou", "motion IoU"),
           ("pixel_real", "lpips", "LPIPS (up)"), ("pixel_real", "psnr", "PSNR (up)")]


def value(summary, key, sub):
    v = summary.get(key)
    if sub is not None:
        v = v.get(sub) if isinstance(v, dict) else None
    return "-" if v is None else (f"{v:.1%}" if key == "counterfactual_acc" or sub == "grasped_acc" else f"{v:.3f}")


@click.command()
@click.option("--tag", default="best")
def main(tag):
    parameters = load_parameters()
    root = os.path.join(parameters["results_dir"], "panel")
    res = defaultdict(dict)   # (robot, dataset) -> model -> summary
    for path in glob.glob(os.path.join(root, "*", "*", tag, "*.json")):
        model, robot = path.split(os.sep)[-4], path.split(os.sep)[-3]
        with open(path) as f:
            res[(robot, os.path.basename(path)[:-5])][model] = json.load(f)["summary"]
    lines = [f"# Eval panel (checkpoint tag `{tag}`)", ""]
    for robot in sorted({r for r, _ in res}):
        lines += [f"## {robot}: no-shift test set", "", "| model | " + " | ".join(c[2] for c in COLUMNS) + " |",
                  "|---" * (len(COLUMNS) + 1) + "|"]
        for model, s in sorted(res.get((robot, "nominal-test"), {}).items()):
            lines.append(f"| {model} | " + " | ".join(value(s, k, sub) for k, sub, _ in COLUMNS) + " |")
        lines.append("")
        shifts = sorted({re.sub(r"-L\d$", "", d) for r, d in res if r == robot and d.startswith("shift-")})
        for shift in shifts:
            lines += [f"### {robot}: {shift[len('shift-'):]}", "",
                      "| model | metric | no shift | L1 | L2 | L3 | L4 | L5 |", "|---|---|---|---|---|---|---|---|"]
            models = sorted({m for (r, d), ms in res.items() if r == robot and d.startswith(shift) for m in ms})
            for model in models:
                for key, sub, name in COLUMNS[:4]:
                    row = [value(res.get((robot, "nominal-test"), {}).get(model, {}), key, sub)]
                    row += [value(res.get((robot, f"{shift}-L{l}"), {}).get(model, {}), key, sub) for l in range(1, 6)]
                    lines.append(f"| {model} | {name} | " + " | ".join(row) + " |")
            lines.append("")
    out = os.path.join(root, "panel.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    log_info(f"panel written to {out}", parameters=parameters)


if __name__ == "__main__":
    main()
