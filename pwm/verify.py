"""Step-1 acceptance: compare a robot's rollout success with its published success rates (configs/robots.yaml) and write
a verification record to <results_dir>/verification/<robot>.{json,md}.

A task passes when our rate is within ``--tolerance_se`` binomial standard errors of the published rate (the standard
error of our estimate, sqrt(p(1-p)/n) at the published p, floored at 1/n so p = 0 or 1 still allows one miss).

    python -m pwm.verify --robot widowx_bridge
"""
import datetime
import json
import os

import click
import numpy as np

from pwm.robots import get_robot
from pwm.utils import load_parameters, log_info, log_warn
from scripts.get_strings import depathify


def task_result(*, rollout_dir: str, reference: float, max_episodes: int, tolerance_se: float) -> dict:
    index = os.path.join(rollout_dir, "episodes.jsonl")
    rows = []
    if os.path.exists(index):
        with open(index) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[:max_episodes]
    n = len(rows)
    if n == 0:
        return {"n": 0, "success": None, "reference": reference, "passed": False}
    rate = float(np.mean([r["success"] for r in rows]))
    se = max(np.sqrt(reference * (1 - reference) / n), 1.0 / n)
    return {"n": n, "success": rate, "reference": reference, "difference": rate - reference,
            "standard_error": float(se), "passed": bool(abs(rate - reference) <= tolerance_se * se)}


@click.command()
@click.option("--robot", required=True)
@click.option("--max_episodes", default=1000, type=int, help="Use at most this many episodes per task")
@click.option("--tolerance_se", default=2.0, type=float)
def main(robot, max_episodes, tolerance_se):
    parameters = load_parameters()
    spec = get_robot(robot=robot, parameters=parameters)
    root = os.path.join(parameters["data_dir"], "verification", robot)
    tasks = {t: task_result(rollout_dir=os.path.join(root, depathify(t)), reference=spec["reference"]["success"][t],
                            max_episodes=max_episodes, tolerance_se=tolerance_se) for t in spec["tasks"]}
    n_total = sum(r["n"] for r in tasks.values())
    pooled = sum((r["success"] or 0) * r["n"] for r in tasks.values()) / max(n_total, 1)
    record = {"robot": robot, "policy": spec["policy"], "policy_revision": spec.get("policy_revision"),
              "sim_env": spec["sim_env"], "policy_env": spec["policy_env"], "reference_source": spec["reference"]["source"],
              "tolerance_se": tolerance_se, "tasks": tasks, "pooled_success": pooled,
              "pooled_reference": float(np.mean([r["reference"] for r in tasks.values()])),
              "passed": all(r["passed"] for r in tasks.values()),
              "rollouts": root, "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    for env in {spec["sim_env"], spec["policy_env"]}:
        manifest = os.path.join(os.environ.get("PWM_ENV_DIR", ""), env, "manifest.json")
        if os.path.exists(manifest):
            with open(manifest) as f:
                record.setdefault("environments", {})[env] = json.load(f)
    out = os.path.join(parameters["results_dir"], "verification")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{robot}.json"), "w") as f:
        json.dump(record, f, indent=1)
    lines = [f"# {robot}: {'PASSED' if record['passed'] else 'NOT PASSED'}", "",
             f"Policy `{spec['policy']}`; reference: {spec['reference']['source']}; tolerance {tolerance_se} SE.", "",
             "| task | n | ours | published | diff (SE) | pass |", "|---|---|---|---|---|---|"]
    for t, r in tasks.items():
        if r["n"] == 0:
            lines.append(f"| {t} | 0 | - | {r['reference']:.1%} | - | no |")
            continue
        lines.append(f"| {t} | {r['n']} | {r['success']:.1%} | {r['reference']:.1%} | "
                     f"{r['difference'] / r['standard_error']:+.1f} | {'yes' if r['passed'] else 'no'} |")
    lines.append(f"| pooled | {n_total} | {pooled:.1%} | {record['pooled_reference']:.1%} | | |")
    with open(os.path.join(out, f"{robot}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    (log_info if record["passed"] else log_warn)(f"{robot} verification written to {out}", parameters=parameters)


if __name__ == "__main__":
    main()
