"""widowx_bridge / google_fractal shifts (SimplerEnv on SAPIEN 2, photo overlay OFF). The hooks take a
pwm.sim.backends.simpler backend (its `_env` is what simpler_env.make returns) or that env itself; `env.unwrapped` is
the ManiSkill2 CustomSceneEnv. Scenes persist across resets and state restores, and rollout loops create a new shift
object per episode, so every edit starts from nominal values kept at module level per scene object.

Both cameras are mounted on the robot (WidowX: 3rd_view_camera on the base, Google: overhead_camera on the head), so
table_height lowers the robot (the same, as seen by camera and robot, as raising the table) and base_pose moves the
camera with the robot."""
import numpy as np

from pwm.shifts.base import Shift

CAMERA = {"widowx_bridge": "3rd_view_camera", "google_fractal": "overhead_camera"}
_NOMINAL = {}


def _genv(handle):
    return handle._env if hasattr(handle, "wm_frame") else handle


def _u(handle):
    return _genv(handle).unwrapped


def nominal(obj, key, read):
    """Nominal value of `key` of a scene object, recorded the first time it is seen (object identity checked)."""
    k = (id(obj), key)
    entry = _NOMINAL.get(k)
    if entry is None or entry[0] is not obj:
        entry = (obj, read(obj))
        _NOMINAL[k] = entry
    return entry[1]


def _sapien():
    import sapien.core as sapien
    return sapien


def _quat_axis_angle(axis, deg):
    from transforms3d.quaternions import axangle2quat
    return axangle2quat(np.asarray(axis, float), np.radians(deg))


def _rotate_y(v, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    x, y, z = v
    return np.array([c * x - s * z, y, s * x + c * z])


class Model(Shift):
    kind = "model"
    reobserve = True

    def on_reset(self, handle):
        super().on_reset(handle)
        self.apply(_u(handle))

    def apply(self, u):
        raise NotImplementedError


class _CameraEdit(Model):
    def apply(self, u):
        cam = u._cameras[CAMERA[self.robot]]
        base = cam.camera_cfg.pose  # the configured mount pose (nominal by construction)
        cam.camera.set_local_pose(base * self.delta())

    def delta(self):
        raise NotImplementedError


class CameraOrientation(_CameraEdit):
    """Rotate about the camera's pan + tilt diagonal (SAPIEN camera frame: x forward, y left, z up)."""

    def delta(self):
        return _sapien().Pose([0, 0, 0], _quat_axis_angle([0.0, 1.0, 1.0], self.value))


class CameraShift(_CameraEdit):
    """value cm to the right and value/2 cm up, in the camera frame."""

    def delta(self):
        return _sapien().Pose([0, -self.value / 100, self.value / 200], [1, 0, 0, 0])


class CameraFov(Model):
    def apply(self, u):
        cam = u._cameras[CAMERA[self.robot]].camera
        fovy = nominal(u._scene, "fovy", lambda _: float(cam.fovy))
        cam.set_fovy(fovy + np.radians(self.value), compute_x=True)


def _lights(u):
    return list(u._scene.get_all_lights())


def _set_light_scale(u, rgb):
    for i, light in enumerate(_lights(u)):  # wrappers are re-created per call: key nominal values by scene + index
        light.set_color(nominal(u._scene, f"light{i}_color", lambda _: np.array(light.color, float)) * np.asarray(rgb))
    u._scene.set_ambient_light(nominal(u._scene, "ambient", lambda s: np.array(s.ambient_light, float)) * np.asarray(rgb))


class ColourTemperature(Model):
    def apply(self, u):
        _set_light_scale(u, [1 + self.value, 1 + self.value / 3, 1 - self.value])


def _set_light_dirs(u, deg):
    for i, light in enumerate(_lights(u)):
        if hasattr(light, "direction") and hasattr(light, "set_direction"):
            d0 = nominal(u._scene, f"light{i}_direction", lambda _: np.array(light.direction, float))
            light.set_direction(_rotate_y(d0, deg))


class LightDirection(Model):
    """Every directional light's direction rotated about the world y axis by `value` degrees."""

    def apply(self, u):
        _set_light_dirs(u, self.value)


class Shadows(Model):
    """Shadows on and the lights tilted by `value` degrees. SimplerEnv creates its lights without shadows (an env
    construction flag simpler_env.make does not pass on), so the main directional light is replaced, once per scene, by
    an identical one that casts shadows."""

    def apply(self, u):
        scene = u._scene
        if not nominal(scene, "pwm_shadow_light", lambda s: False):
            lights = _lights(u)
            first = next((i for i, l in enumerate(lights) if hasattr(l, "direction")), None)
            if first is not None:
                old = lights[first]
                d0 = nominal(scene, f"light{first}_direction", lambda _: np.array(old.direction, float))
                c0 = nominal(scene, f"light{first}_color", lambda _: np.array(old.color, float))
                scene.remove_light(old)
                scene.add_directional_light(d0, c0, shadow=True, scale=5, shadow_map_size=2048)
                # the new light is appended last: carry the nominal values over to its index
                lights = _lights(u)
                for i, l in enumerate(lights):
                    _NOMINAL[(id(scene), f"light{i}_direction")] = (scene, np.array(l.direction, float))
                    _NOMINAL[(id(scene), f"light{i}_color")] = (scene, np.array(l.color, float))
            _NOMINAL[(id(scene), "pwm_shadow_light")] = (scene, True)
        _set_light_dirs(u, self.value)


class LightFlicker(Model):
    kind = "step"
    reobserve = False

    def apply(self, u):
        _set_light_scale(u, [1, 1, 1])

    def on_step(self, handle, t):
        amp, hz = self.value
        k = 1 + amp * np.sin(2 * np.pi * hz * (t + 1) / self.facts["control_hz"])  # prepares step t + 1
        _set_light_scale(_u(handle), [k, k, k])


def _stage_shapes(u):
    """Render shapes of the scene stage (u.arena: Bridge = the table; Google = table and the room around it, one mesh)."""
    shapes = []
    for body in u.arena.get_visual_bodies():
        shapes.extend(body.get_render_shapes())
    return shapes


def _edit_material(shape, edit):
    """shape.material is a copy: edit it and set it back."""
    mat = shape.material
    edit(mat)
    shape.set_material(mat)


class TableColour(Model):
    """Stage tinted towards dark green by `value` (base colour multiplies the texture). On the Google scenes the
    stage is one mesh, so the room is tinted with the table."""
    target = (0.2, 0.45, 0.2, 1.0)

    def apply(self, u):
        for i, shape in enumerate(_stage_shapes(u)):
            c0 = nominal(u._scene, f"stage{i}_base_color", lambda _: np.array(shape.material.base_color, float))
            _edit_material(shape, lambda m: m.set_base_color(list((1 - self.value) * c0 + self.value * np.asarray(self.target))))


class SurfaceTexture(Model):
    """Stage texture replaced by a checker whose contrast grows with `value` (0..1), around a neutral wood colour.
    SAPIEN textures cannot be read back, so unlike the MuJoCo robots this is not a blend with the original texture."""
    wood = np.array([140, 115, 90], np.float32)

    def apply(self, u):
        from pwm.shifts.mujoco_common import checker
        size = 512
        img = (1 - self.value) * self.wood[None, None] + self.value * checker(size, size, 3)
        rgba = np.concatenate([np.clip(img, 0, 255), np.full((size, size, 1), 255, np.float32)], -1) / 255.0
        texture = nominal(u._scene, f"pwm_checker_{self.value}",
                          lambda s: u._renderer.create_texture_from_array(rgba.astype(np.float32)))
        for shape in _stage_shapes(u):
            _edit_material(shape, lambda m: m.set_diffuse_texture(texture))


class _RobotPoseEdit(Model):
    """Robot root moved relative to its pose after reset / restore (both nominal)."""

    def apply(self, u):
        robot = u.agent.robot
        robot.set_pose(robot.get_pose() * self.delta())

    def delta(self):
        raise NotImplementedError


class TableHeight(_RobotPoseEdit):
    def delta(self):
        return _sapien().Pose([0, 0, -self.value / 100], [1, 0, 0, 0])


class BasePose(_RobotPoseEdit):
    def delta(self):
        cm, deg = self.value
        return _sapien().Pose(np.array([cm, cm, 0]) / 100 / np.sqrt(2), _quat_axis_angle([0, 0, 1.0], deg))


class ControlFrequency(Shift):
    """Commands executed at `value` x the trained rate: fewer/more physics substeps per command."""
    kind = "model"

    def on_reset(self, handle):
        super().on_reset(handle)
        u = _u(handle)
        hz = nominal(u._scene, "control_freq", lambda _: u._control_freq) * self.value
        u._control_freq = hz
        u._sim_steps_per_control = max(1, int(round(u._sim_freq / hz)))
        ctrl = u.agent.controller
        for c in getattr(ctrl, "controllers", {"_": ctrl}).values():
            c._control_freq = hz


class Payload(Shift):
    """Uncompensated load of `value` kg at the tool frame, applied before every physics substep (SAPIEN clears forces
    after each step). The agent's substep hook is wrapped once; the force is read from the agent each substep."""
    kind = "step"

    def on_reset(self, handle):
        super().on_reset(handle)
        u = _u(handle)
        agent = u.agent
        if not getattr(agent, "_pwm_payload_hook", False):
            original = type(agent).before_simulation_step

            def before_simulation_step(_self=agent):
                original(_self)
                f = getattr(_self, "_pwm_payload_force", None)
                if f is not None:
                    u.tcp.add_force_torque(f, np.zeros(3))
            agent.before_simulation_step = before_simulation_step
            agent._pwm_payload_hook = True
        agent._pwm_payload_force = np.array([0, 0, -9.81 * self.value])


class Distractors(Shift):
    """`value` extra objects from SimplerEnv's model database. They are added to the scene in before_reset (so states
    saved after the reset include them) and placed on free spots around the task objects in on_reset."""
    kind = "scene"
    reobserve = True
    MODELS = ["green_cube_3cm", "yellow_cube_3cm", "eggplant", "bridge_spoon_generated_modified",
              "bridge_carrot_generated_modified", "opened_coke_can", "opened_pepsi_can", "apple", "orange", "sponge"]

    def _actors(self, u):
        return nominal(u._scene, f"pwm_distractors_{self.value}", lambda s: [])

    def before_reset(self, handle, task=None):
        if hasattr(handle, "wm_frame") and task is not None:
            handle._make_env(task)
        u = _u(handle)
        actors = self._actors(u)
        if actors:
            return
        existing = [a.name for a in u._scene.get_all_actors()]
        models = [m for m in self.MODELS if not any(m in n for n in existing)]
        for i in range(int(self.value)):
            actor = u._build_actor_helper(models[i % len(models)], u._scene, scale=1.0, root_dir=u.asset_root)
            actor.name = f"pwm_distractor_{i}"
            actor.set_pose(_sapien().Pose([10 + i, 10, 0]))  # parked far away until placed
            actors.append(actor)

    def on_reset(self, handle):
        super().on_reset(handle)
        u = _u(handle)
        if not self._actors(u):  # before_reset was not called (no saved state includes them, so this is safe here)
            self.before_reset(handle)
        sapien = _sapien()
        task_objs = [o for o in u.get_actors() if o.name not in ("ground", "goal_site", "", "arena")
                     and not o.name.startswith("pwm_distractor")]
        centre = np.mean([o.pose.p for o in task_objs], axis=0) if task_objs else np.zeros(3)
        z = max((o.pose.p[2] for o in task_objs), default=0.9)
        n = len(self._actors(u))
        for i, actor in enumerate(self._actors(u)):
            ang = 2 * np.pi * (i + 0.5) / max(1, n)
            r = 0.12 + 0.03 * (i // 6)
            actor.set_pose(sapien.Pose([centre[0] + r * np.cos(ang), centre[1] + r * np.sin(ang), z + 0.05]))
            if hasattr(actor, "set_velocity"):
                actor.set_velocity(np.zeros(3)), actor.set_angular_velocity(np.zeros(3))
        for _ in range(int(u._sim_freq * 0.5)):  # let them settle for 0.5 s
            u._scene.step()


def refresh_obs(handle):
    from mani_skill2_real2sim.utils.wrappers.observation import RGBDObservationWrapper
    e = _genv(handle)
    while not isinstance(e, RGBDObservationWrapper):
        e = e.env
    obs = e.observation(_u(handle).get_obs())
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
