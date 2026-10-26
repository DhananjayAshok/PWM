"""gr1_robocasa shifts. The hooks take a pwm.sim.backends.robocasa_gr1 backend (its `_env` is the RoboCasa GR-1 gym
env) or the gym env itself (`env.unwrapped` is RoboCasaEnv, `env.unwrapped.env` the robosuite 1.5 env). robosuite
rebuilds the MuJoCo model on every reset, but state restores do not, so edits start from the model's nominal values.
Scene changes that need new bodies are made in `before_reset` through the task's own configuration (the next reset
rebuilds the model with them).

The GR-1 is joint-position controlled (29-D absolute targets), so the end-effector-frame shifts (frame_rotation)
does not apply. Its ego camera (egoview) is mounted on robot0_head_pitch: lowering the robot is the same, as seen by
the camera and the robot, as raising the table, which is how table_height is done here."""
import numpy as np

from pwm.shifts import mujoco_common as mc
from pwm.shifts.base import Shift

CAMERA = ["egoview"]
HANDS = ("robot0_r_hand_pitch", "robot0_l_hand_pitch")


def _genv(handle):
    """The gym env from a backend or the env itself."""
    return handle._env if hasattr(handle, "wm_frame") else handle


def _rs(handle):
    return _genv(handle).unwrapped.env


def _sim(env):
    sim = _rs(env).sim
    return mc.raw(sim.model), mc.raw(sim.data)


def _forward(env):
    m, d = _sim(env)
    mc.mj().mj_forward(m, d)


def _render_context(env):
    return getattr(getattr(_rs(env).sim, "_render_context_offscreen", None), "con", None)


def _base_body(m):
    for name in ("robot0_base", "robot0_root", "base"):
        i = mc.mj().mj_name2id(m, mc.mj().mjtObj.mjOBJ_BODY, name)
        if i >= 0:
            return i
    raise KeyError("robot base body not found")


def _counter_top_geoms(m):
    """Visual geoms of the work surface: the tabletop scenes' table fixture (table_main_group_top_visual)."""
    g = mc.ids(m, mc.mj().mjtObj.mjOBJ_GEOM, lambda s: s.startswith("table") and "top" in s and s.endswith("visual"))
    return g or mc.ids(m, mc.mj().mjtObj.mjOBJ_GEOM, lambda s: "counter" in s and "top" in s and "visual" in s)


class Model(Shift):
    kind = "model"
    reobserve = True

    def on_reset(self, handle):
        super().on_reset(handle)
        m, d = _sim(handle)
        mc.restore_nominal(m)
        self.apply(handle, m, d)
        _forward(handle)

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
    def apply(self, env, m, d):
        textures = {mc.mat_rgb_texture(m, int(m.geom_matid[g])) for g in _counter_top_geoms(m) if m.geom_matid[g] >= 0}
        for tex in sorted(t for t in textures if t >= 0):
            mc.blend_texture(m, tex, self.value, _render_context(env))


class TableColour(Model):
    target = (0.2, 0.45, 0.2)

    def apply(self, env, m, d):
        geoms = _counter_top_geoms(m)
        mats = sorted({int(m.geom_matid[g]) for g in geoms if m.geom_matid[g] >= 0})
        mc.tint_materials(m, mats, self.target, self.value)
        mc.keep(m, "geom_rgba")
        for g in geoms:
            if m.geom_matid[g] < 0:
                m.geom_rgba[g, :3] = (1 - self.value) * m.geom_rgba[g, :3] + self.value * np.asarray(self.target)


class TableHeight(Model):
    """Table raised by `value` cm relative to robot and (head-mounted) camera: the robot base is lowered instead."""

    def apply(self, env, m, d):
        mc.keep(m, "body_pos")
        m.body_pos[_base_body(m), 2] -= self.value / 100


class BasePose(Model):
    def apply(self, env, m, d):
        cm, deg = self.value
        mc.shift_base(m, _base_body(m), cm, deg)


class ControlFrequency(Shift):
    kind = "model"

    def on_reset(self, handle):
        super().on_reset(handle)
        rs = _rs(handle)
        hz = self.facts["control_hz"] * self.value
        rs.control_freq, rs.control_timestep = hz, 1.0 / hz


class LightFlicker(Shift):
    kind = "step"

    def on_reset(self, handle):
        super().on_reset(handle)
        m, _ = _sim(handle)
        self.base = (mc.nominal(m, "light_diffuse"), mc.nominal(m, "light_ambient"))

    def on_step(self, handle, t):
        amp, hz = self.value
        m, _ = _sim(handle)
        k = 1 + amp * np.sin(2 * np.pi * hz * t / self.facts["control_hz"])
        mc.scale_lights(m, [k, k, k], *self.base)


class Payload(Shift):
    """Uncompensated load of `value` kg at each hand (external force, held through robosuite's substeps)."""
    kind = "step"

    def on_reset(self, handle):
        super().on_reset(handle)
        self.on_step(handle, 0)

    def on_step(self, handle, t):
        m, d = _sim(handle)
        for h in HANDS:
            b = mc.mj().mj_name2id(m, mc.mj().mjtObj.mjOBJ_BODY, h)
            if b >= 0:
                d.xfrc_applied[b] = [0, 0, -9.81 * self.value, 0, 0, 0]


class Distractors(Shift):
    """`value` extra distractor objects (categories other than the task's), placed on the counter by RoboCasa's own
    sampler: a region added to the task's distractor configuration, used by the model rebuild of the next reset."""
    kind = "scene"
    reobserve = True
    region = "pwm_distractors"

    def before_reset(self, handle, task=None):
        if hasattr(handle, "wm_frame") and task is not None:
            handle._make_env(task)
        rs = _rs(handle)
        if rs.distractor_config is None:
            rs.distractor_config = {"regions": {}}
        rs.distractor_config.setdefault("regions", {})[self.region] = {
            "placement": {"size": (1.1, 0.5), "pos": (0, -0.55), "ensure_object_boundary_in_range": False,
                          "try_not_to_place_in": ["container", "obj_container"]},
            "objects": [{"type": self._distractor_categories(rs), "count": (int(self.value), int(self.value)),
                         "exclude_obj_cat": True}],
        }

    def on_reset(self, handle):
        super().on_reset(handle)
        rs = _rs(handle)
        regions = (rs.distractor_config or {}).get("regions", {})
        if self.region not in regions:  # before_reset was not called: configure and rebuild with a new reset
            self.before_reset(handle)
            np.random.seed(int(self.rng.integers(2 ** 31)))
            _genv(handle).reset(seed=int(self.rng.integers(2 ** 31)))

    @staticmethod
    def _distractor_categories(rs):
        """Object categories other than the task object's (the task's distractor pool resolves the same way)."""
        from robocasa.environments.tabletop import tabletop_24dc as t24
        pool = list(t24.TASK_CONFIG.get("obj_groups", []))
        focus = rs.obj_groups if isinstance(rs.obj_groups, (list, tuple)) else [rs.obj_groups]
        return [c for c in pool if c not in focus] or "all"


def refresh_obs(handle):
    u = _genv(handle).unwrapped
    raw = u.env._get_observations(force_update=True)
    obs = u.get_groot_observation(u.get_basic_observation(raw))
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
