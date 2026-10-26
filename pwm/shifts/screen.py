"""Level screening on franka_libero: the strong policy (pi0.5) on a fixed set of train starts, every shift at every
candidate value in configs/shifts.yaml (and no shift), in the libero environment.

Work units are (shift, value, suite, task); each runs the task's fixed initial states as one policy batch and appends one
line per episode to <out>/results.jsonl, plus the first observed agent-view frame (128 px) for the visual-effect
measure. Units already in results.jsonl are skipped (restart-safe); --worker / --num_workers shard the deterministic
unit list.

    python -m pwm.shifts.screen run --out <dir> [--worker 0 --num_workers 1] [--shifts all]
    python -m pwm.shifts.screen summarize --out <dir>
"""
import json
import os

import click
import numpy as np

from pwm.shifts.registry import make_shift_value
from pwm.utils import load_parameters, log_info

ROBOT = "franka_libero"
TASKS = [("libero_spatial", 0), ("libero_object", 0), ("libero_goal", 0), ("libero_10", 0)]  # one per suite
INITS = [0, 20, 40]  # train-split initial states (test uses 45-49)
CONFIRM_INITS = [5, 10, 15, 25, 30]  # second set for confirming chosen levels (n = 4 tasks x 5 = 20 more per value)


def unit_list(*, parameters, shifts, key="candidates"):
    """key = candidates (first round, plus the no-shift baseline) or extension_candidates (second round)."""
    suite = parameters["shifts"]["suite"]
    names = list(suite) if shifts == "all" else shifts.split(",")
    units = [("none", None, s, t) for s, t in TASKS] if key == "candidates" else []
    for name in names:
        for value in suite[name].get(key) or []:
            units += [(name, value, s, t) for s, t in TASKS]
    return units


def unit_key(name, value, suite_name, task_id, inits=None):
    tag = "" if inits is None or inits == INITS else "|confirm"
    return f"{name}|{json.dumps(value)}|{suite_name}|{task_id}{tag}"


def _stack(items):
    if isinstance(items[0], dict):
        return {k: _stack([d[k] for d in items]) for k in items[0]}
    return np.stack(items)


def run_unit(*, policy, pre, post, name, value, suite_name, task_id, out, parameters, inits=None):
    inits = inits or INITS
    import torch
    from lerobot.envs.configs import LiberoEnv as LiberoConfig
    from lerobot.envs.libero import LiberoEnv, _get_suite
    from lerobot.envs.utils import preprocess_observation
    from pwm.shifts.registry import refresh_obs

    torch.manual_seed(0)  # same flow-matching noise stream for every shift value
    config = LiberoConfig(task=suite_name)
    env_pre, env_post = config.get_env_processors()
    suite = _get_suite(suite_name)
    envs, shifts, obs = [], [], []
    for i in inits:
        env = LiberoEnv(suite, task_id, suite_name, obs_type=config.obs_type, episode_index=i,
                        observation_height=config.observation_height, observation_width=config.observation_width)
        shift = None if name == "none" else make_shift_value(robot=ROBOT, name=name, value=value, parameters=parameters)
        if shift is not None:
            shift.before_reset(env)
        o, _ = env.reset(seed=i)
        if shift is not None:
            shift.on_reset(env)
            if shift.reobserve:
                o = refresh_obs(ROBOT, env)
        envs.append(env), shifts.append(shift), obs.append(o)
    frames0 = [env._env.sim.render(camera_name="agentview", width=128, height=128)[::-1].copy() for env in envs]
    policy.reset()
    n, max_steps = len(envs), envs[0]._max_episode_steps
    max_steps = min(max_steps, int(os.environ.get("PWM_SCREEN_MAX_STEPS", max_steps)))  # debugging only
    instruction = envs[0].task_description
    success, steps = [False] * n, [0] * n
    active = np.ones(n, bool)
    for t in range(max_steps):
        batch = preprocess_observation({"pixels": _stack([o["pixels"] for o in obs]),
                                        "robot_state": _stack([o["robot_state"] for o in obs])})
        batch["task"] = [instruction] * n
        batch = pre(env_pre(batch))
        with torch.inference_mode():
            action = post(policy.select_action(batch))
        action = env_post({"action": action})["action"].float().cpu().numpy()
        for k, env in enumerate(envs):
            if not active[k]:
                continue
            cmd = action[k]
            executed = cmd
            if shifts[k] is not None:
                shifts[k].on_step(env, t)
                executed = np.asarray(shifts[k].wrap_action(cmd, t), np.float32)
            obs[k], _, terminated, _, info = env.step(executed)
            steps[k] += 1
            success[k] = success[k] or bool(info["is_success"])
            if terminated:
                active[k] = False
        if not active.any():
            break
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    rows = []
    for k, i in enumerate(inits):
        tag = f"{name}_{json.dumps(value).replace(' ', '')}_{suite_name}_t{task_id}_i{i}"
        np.save(os.path.join(out, "frames", tag + ".npy"), frames0[k])
        rows.append({"unit": unit_key(name, value, suite_name, task_id, inits), "shift": name, "value": value,
                     "suite": suite_name, "task": task_id, "init": i, "success": bool(success[k]), "steps": steps[k],
                     "frame": tag})
        envs[k].close()
    with open(os.path.join(out, "results.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log_info(f"{name}={value} {suite_name} t{task_id}: {sum(success)}/{n}", parameters=parameters)


@click.group()
def main():
    pass


@main.command()
@click.option("--out", required=True)
@click.option("--worker", default=0, type=int)
@click.option("--num_workers", default=1, type=int)
@click.option("--shifts", default="all")
@click.option("--candidates_key", default="candidates", help="candidates | extension_candidates")
@click.option("--confirm", default=None, help="JSON file {shift: [values]}: run those values (and no shift) on the confirmation starts")
def run(out, worker, num_workers, shifts, candidates_key, confirm):
    parameters = load_parameters()
    os.makedirs(out, exist_ok=True)
    inits = INITS
    if confirm:
        with open(confirm) as f:
            chosen = json.load(f)
        inits = CONFIRM_INITS
        units = [("none", None, s, t) for s, t in TASKS]
        for name, values in chosen.items():
            units += [(name, v, s, t) for v in values for s, t in TASKS]
        units = units[worker::num_workers]
    else:
        units = unit_list(parameters=parameters, shifts=shifts, key=candidates_key)[worker::num_workers]
    done = set()
    path = os.path.join(out, "results.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            done = {json.loads(l)["unit"] for l in f if l.strip()}
    todo = [u for u in units if unit_key(*u, inits) not in done]
    log_info(f"screen worker {worker}/{num_workers}: {len(todo)} of {len(units)} units to run", parameters=parameters)
    if not todo:
        return
    from pwm.robots import get_robot
    from pwm.sim.libero import load_policy
    spec = get_robot(robot=ROBOT, parameters=parameters)
    policy, pre, post = load_policy(repo=spec["policy"], revision=spec["policy_revision"],
                                    n_action_steps=spec["protocol"]["n_action_steps"])
    for name, value, suite_name, task_id in todo:
        try:
            run_unit(policy=policy, pre=pre, post=post, name=name, value=value, suite_name=suite_name,
                     task_id=task_id, out=out, parameters=parameters, inits=inits)
        except Exception as e:  # one broken shift must not stop the screen; recorded and retried on restart
            log_info(f"FAILED {name}={value} {suite_name} t{task_id}: {type(e).__name__}: {e}", parameters=parameters)
            with open(os.path.join(out, "errors.jsonl"), "a") as f:
                f.write(json.dumps({"unit": unit_key(name, value, suite_name, task_id, inits),
                                    "error": f"{type(e).__name__}: {e}"}) + "\n")


@main.command()
@click.option("--out", required=True)
def summarize(out):
    """Per shift and value: success rate, change vs no shift, first-frame pixel change vs no shift."""
    with open(os.path.join(out, "results.jsonl")) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    base = {(r["suite"], r["task"], r["init"]): r for r in rows if r["shift"] == "none"}
    groups = {}
    for r in rows:
        groups.setdefault((r["shift"], json.dumps(r["value"])), []).append(r)
    nominal = np.mean([r["success"] for r in base.values()]) if base else float("nan")
    lines = [f"nominal success {nominal:.0%} (n={len(base)})", "",
             "| shift | value | n | success | delta vs nominal | first-frame pixel change |", "|---|---|---|---|---|---|"]
    summary = []
    for (name, value), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if name == "none":
            continue
        pix = []
        for r in rs:
            b = base.get((r["suite"], r["task"], r["init"]))
            if b is None:
                continue
            f1 = np.load(os.path.join(out, "frames", r["frame"] + ".npy")).astype(int)
            f0 = np.load(os.path.join(out, "frames", b["frame"] + ".npy")).astype(int)
            pix.append(np.abs(f1 - f0).mean())
        s = np.mean([r["success"] for r in rs])
        p = float(np.mean(pix)) if pix else float("nan")
        lines.append(f"| {name} | {value} | {len(rs)} | {s:.0%} | {s - nominal:+.0%} | {p:.1f} |")
        summary.append({"shift": name, "value": json.loads(value), "n": len(rs), "success": float(s), "pix": p})
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"nominal": float(nominal), "rows": summary}, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
