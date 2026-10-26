"""State probes: one per world model per robot (and per shift level, trained on that level's calibration pool). A small
MLP from the model's own features (real frames) to the simulator state vector (plans/data_format.md probe_state),
trained to convergence (early stopping on held-out episodes), then applied to predicted features.

Features: spatial maps (C, H, W) are average-pooled to (C, g, g) with g = min(8, sqrt(8192 / C)) so the flattened
size stays near 8k (16-channel VAE latents keep an 8x8 grid, 1408-d V-JEPA tokens a 2x2 grid); token grids (N, D)
with square N are reshaped to (D, sqrt N, sqrt N) and pooled the same way; vectors are used as they are.
"""
import numpy as np
import torch

POOL, MAX_DIM = 8, 8192
LAYOUT = {   # probe_state components: name -> (slice, kind)
    "franka_libero": {"ee": (slice(0, 3), "pos"), "gripper": (slice(3, 4), "scalar"), "objects": (slice(4, 10), "pos"),
                      "grasped": (slice(10, 11), "flag")},
    "widowx_bridge": None, "google_fractal": None,
    "gr1_robocasa": {"ee": (slice(0, 6), "pos"), "gripper": (slice(6, 8), "scalar"), "objects": (slice(8, 14), "pos"),
                     "grasped": (slice(14, 15), "flag")},
}
LAYOUT["widowx_bridge"] = LAYOUT["google_fractal"] = LAYOUT["franka_libero"]


def flatten_features(x):
    x = torch.as_tensor(np.asarray(x, np.float32))
    if x.ndim == 2:                         # (K, D)
        return x
    if x.ndim == 3:                         # (K, N, D) tokens
        k, n, d = x.shape
        s = int(round(n ** 0.5))
        if s * s == n:
            x = x.reshape(k, s, s, d).permute(0, 3, 1, 2)
        else:
            return x.mean(1)
    if x.ndim == 4:                         # (K, C, H, W)
        g = max(1, min(POOL, int((MAX_DIM / x.shape[1]) ** 0.5)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (min(g, x.shape[2]), min(g, x.shape[3])))
        return x.flatten(1)
    return x.flatten(1)


def targets_at(*, probe_state, frame_index, stride):
    steps = np.minimum(np.asarray(frame_index) * stride, len(probe_state) - 1)
    return np.asarray(probe_state, np.float32)[steps]


class Probe(torch.nn.Module):
    def __init__(self, *, d_in, d_out, hidden=512):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.GELU(), torch.nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def fit_probe(*, X, Y, groups, robot, device="cuda", max_epochs=500, patience=20, seed=0):
    """
    Train to convergence. X (N, D) features, Y (N, P) targets, groups (N,) episode ids (the held-out 10% is split by
    episode). Returns a dict with the model and normalisation, and the held-out loss curve.
    """
    rng = np.random.default_rng(seed)
    eps = np.unique(groups)
    held = set(rng.choice(eps, size=max(1, len(eps) // 10), replace=False))
    val = np.array([g in held for g in groups])
    X, Y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)
    mu, sd = X[~val].mean(0), X[~val].std(0) + 1e-6
    ymu, ysd = Y[~val].mean(0), Y[~val].std(0) + 1e-6
    flag = LAYOUT[robot]["grasped"][0]
    ymu[flag], ysd[flag] = 0.0, 1.0
    Xn, Yn = ((X - mu) / sd).to(device), ((Y - ymu) / ysd).to(device)
    torch.manual_seed(seed)
    model = Probe(d_in=X.shape[1], d_out=Y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tr, va = torch.where(torch.as_tensor(~val))[0], torch.where(torch.as_tensor(val))[0]
    best, best_state, bad, curve = np.inf, None, 0, []

    def loss_fn(pred, y):
        cont = torch.ones(y.shape[1], dtype=torch.bool, device=y.device)
        cont[flag] = False
        return torch.nn.functional.mse_loss(pred[:, cont], y[:, cont]) + \
            torch.nn.functional.binary_cross_entropy_with_logits(pred[:, flag], y[:, flag])

    for epoch in range(max_epochs):
        model.train()
        for idx in tr[torch.randperm(len(tr))].split(1024):
            opt.zero_grad()
            loss_fn(model(Xn[idx]), Yn[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(Xn[va]), Yn[va]))
        curve.append(v)
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"model": model.cpu(), "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd, "curve": curve, "epochs": len(curve),
            "best_val": best}


def apply_probe(probe, X):
    X = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        out = probe["model"]((X - probe["mu"]) / probe["sd"]) * probe["ysd"] + probe["ymu"]
    return out.numpy()


def probe_errors(*, pred, true, robot):
    """Per-component errors: positions in cm (mean Euclidean over 3-D points), scalars abs, grasped accuracy."""
    out = {}
    for name, (sl, kind) in LAYOUT[robot].items():
        p, t = pred[:, sl], true[:, sl]
        if kind == "pos":
            out[name + "_cm"] = float(np.mean(np.linalg.norm((p - t).reshape(len(p), -1, 3), axis=-1)) * 100)
        elif kind == "scalar":
            out[name + "_abs"] = float(np.mean(np.abs(p - t)))
        else:
            out[name + "_acc"] = float(np.mean((p[:, 0] > 0) == (t[:, 0] > 0.5)))
    return out
