"""Results table of the V-JEPA 2-AC sweep for one robot: <results_dir>/sweeps/vjepa2/<robot>.md.

Reads every run directory <storage_dir>/checkpoints/vjepa2/<robot>/<dataset>-<run>/ (STOPPED.json when
finished, else the newest checkpoint's history; rollout_eval.json for the 4.8 s rollout and action-sensitivity columns).
Selection number: val/loss_ar (pwm/wm/vjepa2/train.py).

    python -m pwm.wm.vjepa2.sweep_table --robot franka_libero [--dataset nominal]
"""
import json
import os
import re

import click
import yaml

from pwm.utils import load_parameters, log_info

SELECT = "val/loss_ar"  # = pwm.wm.vjepa2.train.SELECT (not imported: the table runs in the core env, without torch)
SWEEPS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "sweeps", "vjepa2.yaml")


def sweep_runs() -> dict:
    with open(SWEEPS) as f:
        return yaml.safe_load(f)["runs"]


def run_status(ckpt_dir: str) -> dict:
    stopped = os.path.join(ckpt_dir, "STOPPED.json")
    if os.path.exists(stopped):
        with open(stopped) as f:
            s = json.load(f)
        return {"status": f"stopped ({s['reason']})", "history": s["history"], "gpu_h": s["gpu_h"]}
    progress = os.path.join(ckpt_dir, "progress.json")
    if os.path.exists(progress):
        with open(progress) as f:
            s = json.load(f)
        return {"status": f"running (step {s['step']})", "history": s["history"], "gpu_h": s["gpu_h"]}
    return {"status": "queued" if os.path.isdir(ckpt_dir) else "not started", "history": [], "gpu_h": 0.0}


def seed_section(*, rows: list) -> list[str]:
    """Mean / sd / worst of the selection number over seeds, for every run that has more than one seed."""
    by = {}
    for name, _, st, best in rows:
        base = re.sub(r"-seed\d+$", "", name)
        by.setdefault(base, []).append(best[SELECT] if best and st["status"].startswith("stopped") else None)
    out = ["", "## Over seeds", "", "Seed 0 plus the extra seeds (cluster/sail/sweep_vjepa2.py SEEDS); only finished "
           "runs count.", "", "| run | seeds done | mean | sd | worst seed |", "|---|---|---|---|---|"]
    for base, vals in sorted(by.items()):
        if len(vals) < 2:
            continue
        done = [v for v in vals if v is not None]
        if len(done) < 2:
            out.append(f"| {base} | {len(done)}/{len(vals)} | - | - | - |")
            continue
        mean = sum(done) / len(done)
        sd = (sum((v - mean) ** 2 for v in done) / (len(done) - 1)) ** 0.5
        out.append(f"| {base} | {len(done)}/{len(vals)} | {mean:.5f} | {sd:.5f} | {max(done):.5f} |")
    return out


@click.command()
@click.option("--robot", required=True)
@click.option("--dataset", default="nominal")
def main(robot, dataset):
    parameters = load_parameters()
    root = os.path.join(parameters["storage_dir"], "checkpoints", "vjepa2", robot)
    runs = sweep_runs()
    names = list(runs)  # grid runs always listed; their extra seeds once launched
    if os.path.isdir(root):
        names += sorted(d[len(dataset) + 1 :] for d in os.listdir(root)
                        if d.startswith(dataset + "-") and re.search(r"-seed\d+$", d))
    rows = []
    for name in names:
        m = re.fullmatch(r"(.+)-seed(\d+)", name)
        base_run, seed = (m[1], int(m[2])) if m else (name, 0)
        changes = {**runs.get(base_run, {}), **({"seed": seed} if seed else {})}
        st = run_status(os.path.join(root, f"{dataset}-{name}"))
        ev = os.path.join(root, f"{dataset}-{name}", "rollout_eval.json")
        st["rollout"] = json.load(open(ev)) if os.path.exists(ev) else None
        best = min(st["history"], key=lambda h: h[SELECT]) if st["history"] else None
        rows.append((name, changes, st, best))
    ranked = sorted([r for r in rows if r[3]], key=lambda r: r[3][SELECT])
    lines = [f"# V-JEPA 2-AC sweep: {robot} / {dataset}", "",
             f"Selection: **{SELECT}** (7-step free-running L1 on fixed val windows; lower is better). "
             f"val/loss_tf = teacher-forced. Rollout 4.8 s: free-running L1 / L2 on 32 fixed val episodes from 1 context "
             f"frame (best.pt). Action sens.: error with shuffled commands / error with true commands (1 = actions "
             f"ignored). GPU-h = GPU time held (incl. start-up and evaluations).", "",
             "| rank | run | changes | best val/loss_ar | val/loss_tf at best | best step | evals | rollout 4.8 s L1 / L2 "
             "| action sens. L1 / L2 | status | GPU-h |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    rank = {r[0]: i + 1 for i, r in enumerate(ranked)}
    for run, changes, st, best in sorted(rows, key=lambda r: rank.get(r[0], 999)):
        ch = ", ".join(f"{k}={v}" for k, v in changes.items()) or "(baseline)"
        ro = st["rollout"]
        ro_s = f"{ro['rollout_l1']:.4f} / {ro['rollout_l2']:.4f}" if ro else "-"
        se_s = f"{ro['action_sensitivity_l1']:.3f} / {ro['action_sensitivity_l2']:.3f}" if ro else "-"
        if best:
            lines.append(f"| {rank[run]} | {run} | {ch} | {best[SELECT]:.5f} | {best['val/loss_tf']:.5f} | {best['step']} | "
                         f"{len(st['history'])} | {ro_s} | {se_s} | {st['status']} | {st['gpu_h']:.1f} |")
        else:
            lines.append(f"| - | {run} | {ch} | - | - | - | 0 | - | - | {st['status']} | {st['gpu_h']:.1f} |")
    lines += seed_section(rows=rows)
    out = os.path.join(parameters["results_dir"], "sweeps", "vjepa2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{robot}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    log_info(f"[vjepa2] sweep table {out}/{robot}.md", parameters=parameters)


if __name__ == "__main__":
    main()
