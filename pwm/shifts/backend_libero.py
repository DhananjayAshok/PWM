"""franka_libero shifts. The hooks take a pwm.sim.backends.libero backend (its `_env` is lerobot's LiberoEnv) or a
LiberoEnv directly. Hard resets rebuild the MuJoCo model, but state restores (counterfactual groups, replays) do not, so
edits start from the model's nominal values (mujoco_common.restore_nominal). lerobot runs its settle steps before
`on_reset`, so shifts that move bodies settle again here."""
import os
import re

import numpy as np

from pwm.shifts import mujoco_common as mc
from pwm.shifts.base import Shift

CAMERA = ["agentview"]
DUMMY = [0, 0, 0, 0, 0, 0, -1]
SETTLE = 10


def _lenv(handle):
    """lerobot LiberoEnv from a backend or the env itself."""
    return handle._env if hasattr(handle, "wm_frame") else handle


def _sim(handle):
    raw = _lenv(handle)._env
    return raw, mc.raw(raw.sim.model), mc.raw(raw.sim.data)


def _forward(env):
    _, m, d = _sim(env)
    mc.mj().mj_forward(m, d)


def _settle(env, n=SETTLE):
    raw, _, _ = _sim(env)
    for _ in range(n):
        raw.step(list(map(float, DUMMY)))


def _table_top_geoms(m):
    g = mc.ids(m, mc.mj().mjtObj.mjOBJ_GEOM, lambda s: "table" in s and "visual" in s and "leg" not in s)
    return g or mc.ids(m, mc.mj().mjtObj.mjOBJ_GEOM, lambda s: s == "floor")


def _render_context(env):
    raw, _, _ = _sim(env)
    return getattr(getattr(raw.sim, "_render_context_offscreen", None), "con", None)


class Model(Shift):
    """A sim edit applied once after reset (from the model's nominal values)."""
    kind = "model"
    reobserve = True

    def on_reset(self, handle):
        super().on_reset(handle)
        env = _lenv(handle)
        _, m, d = _sim(env)
        mc.restore_nominal(m)
        self.apply(env, m, d)
        _forward(env)

    def apply(self, env, m, d):
        raise NotImplementedError


class CameraOrientation(Model):
    def apply(self, env, m, d):
        mc.camera_orientation(m, mc.camera_id(m, CAMERA), self.value)


class CameraFov(Model):
    def apply(self, env, m, d):
        mc.camera_fov(m, mc.camera_id(m, CAMERA), self.value)


class CameraShift(Model):
    def apply(self, env, m, d):
        mc.camera_shift_local(m, mc.camera_id(m, CAMERA), self.value)


class Shadows(Model):
    """Shadows on (LIBERO renders without them) and the lights tilted by `value` degrees, so shadows get longer."""

    def apply(self, env, m, d):
        mc.keep(m, "light_castshadow")
        m.light_castshadow[:] = 1
        mc.rotate_lights(m, self.value)


class ColourTemperature(Model):
    def apply(self, env, m, d):
        mc.scale_lights(m, mc.warm_cool(self.value))


class LightDirection(Model):
    def apply(self, env, m, d):
        mc.rotate_lights(m, self.value)


class SurfaceTexture(Model):
    """The work surface's texture blended towards a high-contrast checker by `value` (0..1)."""

    def apply(self, env, m, d):
        textures = {mc.mat_rgb_texture(m, int(m.geom_matid[g])) for g in _table_top_geoms(m) if m.geom_matid[g] >= 0}
        for tex in sorted(t for t in textures if t >= 0):
            mc.blend_texture(m, tex, self.value, _render_context(env))


class TableColour(Model):
    """The work surface tinted towards dark green by `value` (0..1)."""
    target = (0.2, 0.45, 0.2)

    def apply(self, env, m, d):
        geoms = _table_top_geoms(m)
        mats = sorted({int(m.geom_matid[g]) for g in geoms if m.geom_matid[g] >= 0})
        mc.tint_materials(m, mats, self.target, self.value)
        mc.keep(m, "geom_rgba")
        for g in geoms:
            if m.geom_matid[g] < 0:
                m.geom_rgba[g, :3] = (1 - self.value) * m.geom_rgba[g, :3] + self.value * np.asarray(self.target)


class TableHeight(Model):
    """Table, fixtures and objects raised by `value` cm; robot and camera stay where they are."""

    def apply(self, env, m, d):
        dz = self.value / 100
        mc.keep(m, "body_pos")
        for b in range(1, m.nbody):
            if m.body_parentid[b] != 0 or mc.name_of(m, mc.mj().mjtObj.mjOBJ_BODY, b).startswith("robot0"):
                continue
            if m.body_jntnum[b] == 0:
                m.body_pos[b, 2] += dz
            else:
                j = m.body_jntadr[b]
                if m.jnt_type[j] == mc.mj().mjtJoint.mjJNT_FREE:
                    d.qpos[m.jnt_qposadr[j] + 2] += dz
        _forward(env)
        _settle(env)


class BasePose(Model):
    """Robot mounted off its calibrated spot: value = (cm along the base diagonal, deg of yaw)."""

    def apply(self, env, m, d):
        cm, deg = self.value
        mc.shift_base(m, mc.mj().mj_name2id(m, mc.mj().mjtObj.mjOBJ_BODY, "robot0_base"), cm, deg)
        _forward(env)
        _settle(env)


class ControlFrequency(Shift):
    """Commands executed at `value` x the trained 20 Hz (each command lasts 1 / (20 value) s)."""
    kind = "model"

    def on_reset(self, handle):
        super().on_reset(handle)
        rs = _lenv(handle)._env.env
        hz = self.facts["control_hz"] * self.value
        rs.control_freq, rs.control_timestep = hz, 1.0 / hz


class LightFlicker(Shift):
    """Light intensity modulated over time: value = (amplitude, Hz)."""
    kind = "step"

    def on_reset(self, handle):
        super().on_reset(handle)
        _, m, _ = _sim(handle)
        self.base = (mc.nominal(m, "light_diffuse"), mc.nominal(m, "light_ambient"))

    def on_step(self, handle, t):
        amp, hz = self.value
        _, m, _ = _sim(handle)
        k = 1 + amp * np.sin(2 * np.pi * hz * t / self.facts["control_hz"])
        mc.scale_lights(m, [k, k, k], *self.base)


class Payload(Shift):
    """Uncompensated load at the hand (the OSC compensates model mass, so a mass edit would be invisible)."""
    kind = "step"

    def on_reset(self, handle):
        super().on_reset(handle)
        self.on_step(handle, 0)

    def on_step(self, handle, t):
        _, m, d = _sim(handle)
        b = mc.mj().mj_name2id(m, mc.mj().mjtObj.mjOBJ_BODY, "robot0_right_hand")
        d.xfrc_applied[b] = [0, 0, -9.81 * self.value, 0, 0, 0]


# ---------------------------------------------------------------- distractors (task-file edit + init-state remap)
SCENE_POOL = ["butter", "chocolate_pudding", "cream_cheese", "alphabet_soup", "ketchup", "milk", "orange_juice",
              "bbq_sauce", "tomato_sauce", "salad_dressing", "macaroni_and_cheese", "popcorn"]


def scene_bddl(text, n):
    """n distractor objects of classes not in the task, placed at the free grid points of the work surface farthest from
    every existing region (kept within view)."""
    objs = re.search(r"\(:objects\n(.*?)\)", text, re.S).group(1)
    classes = {l.split(" - ")[1].strip() for l in objs.strip().split("\n") if " - " in l}
    surf = re.search(r"\(:fixtures\n\s*(\w+) -", text).group(1)
    cent = [((a + c) / 2, (b + d) / 2) for a, b, c, d in
            [tuple(map(float, m.split())) for m in re.findall(rf"\(:target {surf}\)\s*\(:ranges \(\s*\(([-\d. e]+)\)", text)]]
    grid = [(x, y) for x in np.arange(-0.15, 0.16, 0.05) for y in np.arange(-0.15, 0.16, 0.05)]
    picks = []
    for _ in range(n):
        score = lambda p: min([np.hypot(p[0] - q[0], p[1] - q[1]) for q in cent + picks] or [1])
        picks.append(max(grid, key=score))
    cls = [c for c in SCENE_POOL if c not in classes][:n]
    reg = "".join(f"      (pwm_distractor_region_{i}\n          (:target {surf})\n          (:ranges (({x - .01:.3f} {y - .01:.3f} {x + .01:.3f} {y + .01:.3f})))\n      )\n"
                  for i, (x, y) in enumerate(picks))
    text = text.replace("    (:regions\n", "    (:regions\n" + reg, 1)
    text = re.sub(r"\(:objects\n", "(:objects\n" + "".join(f"    pwm_distractor_{i}_1 - {c}\n" for i, c in enumerate(cls)), text, 1)
    return text.replace("  (:init\n", "  (:init\n" + "".join(
        f"    (On pwm_distractor_{i}_1 {surf}_pwm_distractor_region_{i})\n" for i in range(len(cls))), 1)


def joint_layout(m):
    out = {}
    for j in range(m.njnt):
        nq = {0: 7, 1: 4}.get(int(m.jnt_type[j]), 1)
        nv = {0: 6, 1: 3}.get(int(m.jnt_type[j]), 1)
        out[mc.name_of(m, mc.mj().mjtObj.mjOBJ_JOINT, j)] = (int(m.jnt_qposadr[j]), nq, int(m.jnt_dofadr[j]), nv)
    return out, m.nq, m.nv


class Distractors(Shift):
    """`value` extra objects on the work surface. The task file gets new objects and regions and the LiberoEnv is rebuilt
    from it (LIBERO parses the task file only when the env is built); LIBERO's fixed initial states belong to the
    original joint layout, so they are copied joint by joint (by name) after the reset. The env stays on the edited task
    file afterwards (one dataset per worker)."""
    kind = "scene"
    reobserve = True
    _layouts = {}

    def _edit(self, env):
        original = getattr(env, "_pwm_original_bddl", None) or env._task_bddl_file
        edited = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"pwm_distractors{self.value}_{os.path.basename(original)}")
        if env._task_bddl_file == edited:
            return False
        if original not in Distractors._layouts:
            from libero.libero.envs import OffScreenRenderEnv
            ref = OffScreenRenderEnv(bddl_file_name=original, camera_heights=16, camera_widths=16)
            ref.reset()
            Distractors._layouts[original] = joint_layout(mc.raw(ref.sim.model))
            ref.close()
        with open(original) as f:
            text = scene_bddl(f.read(), int(self.value))
        with open(edited, "w") as f:
            f.write(text)
        if env._env is not None:
            env._env.close()
            env._env = None  # rebuilt from the edited task file at the next reset
        env._pwm_original_bddl, env._task_bddl_file = original, edited
        env._pwm_init_states, env._init_states = env._init_states, None  # loaded by name in on_reset
        return True

    def before_reset(self, handle, task=None):
        if hasattr(handle, "wm_frame") and task is not None:
            handle._make_env(task)
        self._edit(_lenv(handle))

    def on_reset(self, handle):
        super().on_reset(handle)
        env = _lenv(handle)
        if self._edit(env):  # before_reset was not called: rebuild and reset again to the same initial state
            env.reset(seed=int(env.init_state_id))
        _, m, d = _sim(env)
        init_states = getattr(env, "_pwm_init_states", None)
        if init_states is not None:
            flat = init_states[int(env.init_state_id) % len(init_states)]
            (old, nq, nv) = Distractors._layouts[env._pwm_original_bddl]
            new, _, _ = joint_layout(m)
            qpos, qvel = flat[1:1 + nq], flat[1 + nq:1 + nq + nv]
            for n, (qa, k, da, kv) in old.items():
                if n in new and new[n][1] == k:
                    d.qpos[new[n][0]:new[n][0] + k] = qpos[qa:qa + k]
                    d.qvel[new[n][2]:new[n][2] + kv] = qvel[da:da + kv]
            _forward(env)
            _settle(env)


def refresh_obs(handle):
    env = _lenv(handle)
    obs = env._format_raw_obs(env._env.env._get_observations(force_update=True))
    if hasattr(handle, "wm_frame"):
        handle._obs = obs
    return obs


SHIFTS = {
    "camera_orientation": CameraOrientation, "camera_fov": CameraFov, "camera_shift": CameraShift,
    "light_flicker": LightFlicker, "shadows": Shadows, "colour_temperature": ColourTemperature,
    "light_direction": LightDirection, "surface_texture": SurfaceTexture, "table_colour": TableColour,
    "table_height": TableHeight, "distractors": Distractors,
    "control_frequency": ControlFrequency, "base_pose": BasePose, "payload": Payload,
}
