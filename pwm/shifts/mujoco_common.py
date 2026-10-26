"""MuJoCo edit primitives shared by the LIBERO (robosuite 1.4) and RoboCasa (robosuite 1.5) backends. They take the raw
mujoco MjModel / MjData. Camera and light poses reach the renderer only through mj_forward (mj_camlight), so every pose
edit is followed by one."""
import numpy as np


def mj():
    import mujoco
    return mujoco


def raw(x):
    """robosuite wraps MjModel / MjData; return the mujoco object (robosuite 1.5's data wrapper also carries a _model,
    so _data is checked first)."""
    if hasattr(x, "_data"):
        return x._data
    return getattr(x, "_model", x)


def name_of(m, obj, i):
    return mj().mj_id2name(m, obj, i) or ""


def ids(m, obj, pred):
    count = {mj().mjtObj.mjOBJ_BODY: m.nbody, mj().mjtObj.mjOBJ_GEOM: m.ngeom, mj().mjtObj.mjOBJ_CAMERA: m.ncam,
             mj().mjtObj.mjOBJ_LIGHT: m.nlight, mj().mjtObj.mjOBJ_MATERIAL: m.nmat}[obj]
    return [i for i in range(count) if pred(name_of(m, obj, i))]


def camera_id(m, candidates):
    for c in candidates:
        i = mj().mj_name2id(m, mj().mjtObj.mjOBJ_CAMERA, c)
        if i >= 0:
            return i
    raise KeyError(f"none of the cameras {candidates} in model (have {ids(m, mj().mjtObj.mjOBJ_CAMERA, lambda s: True)})")


def quat_axis_angle(axis, deg):
    q = np.zeros(4)
    mj().mju_axisAngle2Quat(q, np.asarray(axis, float) / np.linalg.norm(axis), np.radians(deg))
    return q


def quat_mul(a, b):
    out = np.zeros(4)
    mj().mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


# ---------------------------------------------------------------- nominal values
# Rollout loops create a new shift object per episode and may restore a saved state without rebuilding the model
# (counterfactual groups, replays), so every edit is made from the model's nominal values, kept here per model object
# (only the few most recent models: one env per process).
_NOMINAL = {}


def _snapshot(m):
    entry = _NOMINAL.get(id(m))
    if entry is None or entry[0] is not m:
        if len(_NOMINAL) >= 4:
            _NOMINAL.clear()
        entry = (m, {})
        _NOMINAL[id(m)] = entry
    return entry[1]


def keep(m, *fields):
    """Remember the nominal value of model fields before they are edited for the first time."""
    snap = _snapshot(m)
    for f in fields:
        if f not in snap:
            snap[f] = getattr(m, f).copy()


def nominal(m, field):
    keep(m, field)
    return _snapshot(m)[field]


def restore_nominal(m):
    """Undo every edit made to this model (called before a shift is applied)."""
    for f, v in _snapshot(m).items():
        getattr(m, f)[...] = v


# ---------------------------------------------------------------- camera
def camera_orientation(m, cam, deg):
    """Rotate a camera about its own pan + tilt diagonal (MuJoCo cameras look along -z, y up: pan = y, tilt = x)."""
    keep(m, "cam_quat")
    m.cam_quat[cam] = quat_mul(m.cam_quat[cam].copy(), quat_axis_angle([1.0, 1.0, 0.0], deg))


def camera_fov(m, cam, deg):
    keep(m, "cam_fovy")
    m.cam_fovy[cam] += deg


def camera_shift_local(m, cam, cm):
    """Translate a camera in its own frame: cm to the right and cm/2 up (MuJoCo camera frame: x right, y up)."""
    keep(m, "cam_pos")
    rot = np.zeros(9)
    mj().mju_quat2Mat(rot, m.cam_quat[cam])
    m.cam_pos[cam] += rot.reshape(3, 3) @ np.array([cm / 100, cm / 200, 0.0])


# ---------------------------------------------------------------- lights
def rotate_lights(m, deg, move_pos=True):
    """Rotate every light's direction about the world y axis (sun moves across the scene)."""
    keep(m, "light_dir", "light_pos")
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    for i in range(m.nlight):
        x, y, z = m.light_dir[i]
        m.light_dir[i] = [c * x - s * z, y, s * x + c * z]
        if move_pos:
            m.light_pos[i, 0] += 2 * s


def scale_lights(m, rgb, base_diffuse=None, base_ambient=None):
    d0 = nominal(m, "light_diffuse") if base_diffuse is None else base_diffuse
    a0 = nominal(m, "light_ambient") if base_ambient is None else base_ambient
    m.light_diffuse[:] = d0 * np.asarray(rgb)
    m.light_ambient[:] = a0 * np.asarray(rgb)


def warm_cool(w):
    """Colour-temperature cast: w > 0 warm, w < 0 cool."""
    return np.array([1 + w, 1 + w / 3, 1 - w])


# ---------------------------------------------------------------- surfaces
def mat_rgb_texture(m, mat):
    """Texture id of a material's RGB role (MuJoCo >= 3.1 stores one texture per role), or -1."""
    t = m.mat_texid[mat]
    if np.ndim(t) == 0:
        return int(t)
    return int(t[1]) if t[1] >= 0 else int(t[0])


def checker(h, w, c, tiles=8):
    yy, xx = np.mgrid[0:h, 0:w]
    board = ((yy * tiles // h + xx * tiles // w) % 2).astype(np.float32)
    dark, light = np.array([60, 45, 35], np.float32), np.array([215, 205, 180], np.float32)
    img = dark[None, None] * (1 - board[..., None]) + light[None, None] * board[..., None]
    if c == 4:
        img = np.concatenate([img, np.full((h, w, 1), 255, np.float32)], -1)
    return img[..., :c]


def blend_texture(m, tex, alpha, render_context=None):
    """Blend texture `tex` towards a high-contrast checker by alpha; upload it if a render context already exists."""
    keep(m, "tex_data")
    h, w, c = int(m.tex_height[tex]), int(m.tex_width[tex]), int(m.tex_nchannel[tex])
    adr = int(m.tex_adr[tex])
    data = m.tex_data[adr:adr + h * w * c].reshape(h, w, c).astype(np.float32)
    out = (1 - alpha) * data + alpha * checker(h, w, c)
    m.tex_data[adr:adr + h * w * c] = np.clip(out, 0, 255).astype(np.uint8).reshape(-1)
    if render_context is not None:
        mj().mjr_uploadTexture(m, render_context, tex)


def tint_materials(m, mats, target, strength):
    """Multiply materials' colour (which multiplies their texture) towards target by strength."""
    keep(m, "mat_rgba")
    t = np.asarray(target, float)
    for mat in mats:
        m.mat_rgba[mat, :3] = (1 - strength) * m.mat_rgba[mat, :3] + strength * t


def shift_base(m, body, cm, deg):
    """Move a (fixed) base body by cm along the base diagonal in xy and yaw it by deg."""
    keep(m, "body_pos", "body_quat")
    m.body_pos[body, :2] += np.array([cm, cm]) / 100 / np.sqrt(2)
    m.body_quat[body] = quat_mul(quat_axis_angle([0, 0, 1.0], deg), m.body_quat[body].copy())
