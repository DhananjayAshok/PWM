"""Propose 5-level ladders from a screening summary (python -m pwm.shifts.screen summarize).

Rule, per shift, over its candidates in increasing severity:
  L5 = the first candidate where success <= `broken` (policy essentially broken); the strongest candidate otherwise.
  L1 = the first candidate that is noticeable: first-frame pixel change >= `min_pix` for shifts that change the image,
       or success drop >= `min_drop` for shifts that do not; the weakest candidate otherwise.
  L2..L4 = candidates spread evenly (by index) between L1 and L5; scalar values are interpolated when fewer than 5
       candidates lie in that range (marked "interpolated": not screened directly).

    python -m pwm.shifts.ladders --summary <out>/summary.json
"""
import json

import click
import numpy as np

from pwm.utils import load_parameters

IMAGE_SHIFTS = {"camera_orientation", "camera_fov", "camera_shift", "shadows", "colour_temperature", "light_direction",
                "surface_texture", "table_colour", "table_height", "distractors", "base_pose"}


def propose(*, candidates, rows, nominal, image, broken=0.2, min_pix=3.0, min_drop=0.05):
    by_value = {json.dumps(r["value"]): r for r in rows}
    stats = [by_value.get(json.dumps(c)) for c in candidates]
    idx = [i for i, s in enumerate(stats) if s is not None]
    if not idx:
        return None
    i5 = next((i for i in idx if stats[i]["success"] <= broken), idx[-1])
    if image:
        i1 = next((i for i in idx if np.isfinite(stats[i]["pix"]) and stats[i]["pix"] >= min_pix), idx[0])
    else:
        i1 = next((i for i in idx if nominal - stats[i]["success"] >= min_drop), idx[0])
    i1 = min(i1, i5)
    span = [i for i in idx if i1 <= i <= i5]
    if len(span) >= 5:
        picks = [span[int(round(k * (len(span) - 1) / 4))] for k in range(5)]
        return {"levels": [candidates[i] for i in picks], "interpolated": False,
                "success": [stats[i]["success"] for i in picks], "pix": [stats[i]["pix"] for i in picks]}
    lo, hi = candidates[i1], candidates[i5]
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and hi != lo:
        vals = [float(lo + (hi - lo) * k / 4) for k in range(5)]
        vals = [int(round(v)) if isinstance(lo, int) and isinstance(hi, int) else round(v, 3) for v in vals]
    else:  # pairs or a too-short range: take neighbouring candidates
        start = max(0, min(i1, len(candidates) - 5))
        vals = candidates[start:start + 5]
    return {"levels": vals, "interpolated": True,
            "success": [by_value[json.dumps(v)]["success"] if json.dumps(v) in by_value else None for v in vals],
            "pix": [by_value[json.dumps(v)]["pix"] if json.dumps(v) in by_value else None for v in vals]}


DECREASING = {"control_frequency"}  # smaller value = stronger shift


def ordered_candidates(*, name, spec):
    """Both screening rounds' values, weakest to strongest."""
    values = list(spec["candidates"]) + list(spec.get("extension_candidates") or [])
    unique = []
    for v in values:
        if v not in unique:
            unique.append(v)
    return sorted(unique, key=lambda v: (tuple(v) if isinstance(v, list) else v), reverse=name in DECREASING)


@click.command()
@click.option("--summary", required=True)
@click.option("--out", default=None, help="Write the proposal as JSON here (default: next to the summary)")
def main(summary, out):
    parameters = load_parameters()
    with open(summary) as f:
        s = json.load(f)
    suite = parameters["shifts"]["suite"]
    proposal = {}
    for name, spec in suite.items():
        rows = [r for r in s["rows"] if r["shift"] == name]
        proposal[name] = propose(candidates=ordered_candidates(name=name, spec=spec), rows=rows, nominal=s["nominal"],
                                 image=name in IMAGE_SHIFTS)
        print(name, json.dumps(proposal[name]))
    with open(out or summary.replace("summary.json", "ladders_proposed.json"), "w") as f:
        json.dump(proposal, f, indent=1)


if __name__ == "__main__":
    main()
