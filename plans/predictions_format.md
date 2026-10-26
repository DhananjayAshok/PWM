# Predictions format (contract between the world models and the eval panel, pwm/eval)

Every world model provides two entry points (click commands in its own environment) that read episodes in the
plans/data_format.md layout and write here:

```
$storage_dir/predictions/<model>/<robot>/<dataset>/<checkpoint_tag>/
  <episode_id>/
    pred.mp4              pixel models only: predicted frames at the model's native resolution, recording fps
    pred_features.npy     (K, ...) model features of the predicted future, one per entry of frame_index
    frame_index.npy       (K,) recorded-frame index (within the episode) each feature describes
    meta.json             context_frames, horizon_frames, seed, checkpoint, wall_s
$storage_dir/predictions/<model>/<robot>/features/<checkpoint_tag>/<dataset>/<split>/   (encode entry)
  <episode_id>.npz        real_features (K, ...), frame_index (K,)   -- features of REAL frames, same space
```

Protocol (identical for all models):
- **Window:** the first `horizon_s = 4.8 s` of the episode (counterfactual groups share their first frame, so windows
  start at recorded frame 0). horizon_frames = round(4.8 * fps): 48 at 10 fps (Franka, GR-1), 24 at 5 fps (WidowX),
  14 at 3 fps (Google). Episodes shorter than the window are evaluated on their full length.
- **Context:** `context_frames = 1` real frame (frame 0) for every model; actions = `action_cmd` for the window, grouped
  per recorded frame (plans/data_format.md). Free-running: each chunk continues from the model's own previous output.
- **Seeds:** fixed (seed 0); stochastic models write sample 0; optional extra samples as pred_s<k>.mp4 / features.
- **Features:** the end of the model's encoder side. DreamDojo: video-VAE latents (generated latents before decoding
  for predictions; VAE-encoded real frames for real features). V-JEPA 2-AC: encoder features (real) / predictor
  output (predicted). DreamerV3: posterior latent state (real) / imagined prior state (predicted). frame_index maps each
  feature to the last recorded frame it covers (e.g. DreamDojo latent k covers frames 4k-3..4k).
- **Real features:** the encode entry runs on the nominal train split (a sample of >= 2,000 episodes, all recorded
  frames; probe training), on each calibration pool (shift probes), and on every test set (probe floor and
  counterfactual accuracy); same space as pred_features.
- **Feature shapes:** spatial maps as (K, C, H, W) (DreamDojo VAE latents), token grids as (K, N, D) with square N
  (V-JEPA), vectors as (K, D) (DreamerV3). The panel pools them (pwm/eval/probe.py flatten_features).
