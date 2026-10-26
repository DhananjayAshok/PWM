# Training tricks for our three world models: a vetted literature list

Compiled 2026-10-24. Research only: no code was changed and no jobs were run.

> **Dropped 2026-10-26 (user decision): no tricks for any model.** All trick code was deleted and the trick jobs were
> cancelled; this file is kept as the record. Only DreamerV3 ran tricks, on GR-1 and WidowX (3 seeds each, selection
> metric val/openloop_mse64, lower is better; the grid winner's 3 seeds in brackets):
> - GR-1 [lr3e-4: 0.0507 / 0.0607 / 0.0580]: aug (C2) 0.0497 / 0.0500 / 0.0469 (the only clear effect, about -13%),
>   LaProp 0.0542 / 0.0560 / 0.0549, CPC (C1) 0.0571 / 0.0554 / 0.0529, SOAP 0.0526 / 0.0534 / 0.0587, AdEMAMix
>   0.0545 / 0.0575 / 0.0572. WSD cooldown: no gain over its constant-LR control, both worse than the parent run.
> - WidowX [size_s100m: 0.0103 / 0.0099 / 0.0103]: every trick within about ±5% (aug 0.0108 / 0.0102 / 0.0097,
>   AdEMAMix 0.0101 / 0.0096 / 0.0096, CPC 0.0102 / 0.0103 / 0.0102, SOAP 0.0106 / 0.0104 / 0.0106, LaProp
>   0.0103 / 0.0103 / 0.0102 with seed 0 unfinished).
> - V-JEPA 2-AC and DreamDojo never ran a trick. Franka and Google trick runs were cancelled before they produced results.

**How this was vetted.** For every paper I confirmed the title, venue, year and arXiv id on the proceedings page,
the conference virtual site (iclr.cc / neurips.cc / icml.cc / PMLR / CVF open access) or the arXiv record. Citation
counts come from one Semantic Scholar batch query on 2026-10-24 and are rounded. They are approximate and lower than
Google Scholar's. Where the numbers are quoted, they come from each paper's own abstract or tables (the arXiv HTML
versions I read) and are the authors' claims, not reproductions. Allowed venues: ICLR, NeurIPS, ICML, ACL and EMNLP,
2025 or later. Anything else is marked **other venue**.

**What "val loss" means for each model, and why it matters here.** Our selection criterion (val loss on a fixed shard)
is **teacher-forced** for (A) and mostly so for (B) and (C). Several of the best long-horizon tricks below (context
noise, self-rollout training, guidance) target **autoregressive rollout error**. They can leave the teacher-forced
val loss unchanged or make it slightly *worse*, while improving the 4.8 s rollouts we actually care about. Guidance
tricks are inference-only and cannot move the val loss at all. Before testing these, add a cheap **rollout metric**
next to val loss: for example, latent MSE / PSNR / LPIPS of a free-running 4.8 s prediction on ~32 fixed val
episodes, plus an **action-sensitivity score** (rollout error with true actions vs actions shuffled across episodes).
Without that metric, tricks 4, 5, 7 and 9 of the shortlist will look neutral or harmful.

Things I checked in our code while writing this:
- DreamDojo / Cosmos-Predict2.5 (`/persist/dreamdojo_franka/ext/DreamDojo/cosmos_predict2/_src/predict2/models/video2world_model_rectified_flow.py`)
  conditions with `FRAME_REPLACE`. Context latents are replaced by **clean** ground truth. `conditional_frame_timestep`
  defaults to -1 (off), and there is a `use_video_condition=False` path that zeroes the context. So there is currently
  no context-noise augmentation, and there is a hook for it.
- The text2world RF base samples `train_time_distribution="logitnormal"` with `shift` (SD3-style). Timestep sampling is
  already the standard choice.
- DreamDojo ships `interactive/models/action_video2world_self_forcing.py` (action-conditioned Self-Forcing + DMD2
  distillation with KV-cache rollout). Self Forcing is therefore an existing code path, not a from-scratch build.
- V-JEPA 2-AC (`plans/vjepa2.md`): L1 teacher forcing + a 2-step rollout, constant LR after warm-up.
- DreamerV3 port (`plans/dreamerv3.md`): lr 1e-4 constant, 1 context frame, the port's defaults. The DreamerV3
  robustness tricks (symlog, free bits, KL balancing, unimix, LayerNorm) are already in the port.

Difficulty: **S** = config or <50 lines; **M** = a new loss or model input, a day of work; **L** = new training stage
or architecture change.

---

## (A) DreamDojo 2B (Cosmos-Predict2.5 video DiT, rectified flow, 12-frame AR chunks)

| # | Trick (one line) | Paper · venue · year · arXiv · ~cites | Claims (authors' numbers) | Effect on our val loss | Diff. | Cheap test |
|---|---|---|---|---|---|---|
| A1 | **Context-frame noise augmentation:** during training, add Gaussian noise of a random level to the context latents and tell the model the level (a learned embedding of 10 buckets, or our existing `conditional_frame_timestep`); at inference use a small fixed level | *Diffusion Models Are Real-Time Game Engines* (GameNGen) · ICLR 2025 · 2408.14837 · ~360 | Max noise 0.7, uniform, 10 buckets. Without it, LPIPS/PSNR degrade after ~10-20 AR frames; with it, stable multi-minute rollouts. Next-frame PSNR 29.4 | Teacher-forced loss may rise slightly (noisier context); rollout error should fall | S | One robot, 2 short fine-tunes from the same init (e.g. 2-3k steps): noise max ∈ {0, 0.3}. Compare teacher-forced val loss **and** 4.8 s rollout error (3 chunks) on fixed val episodes |
| A2 | **Action dropout + action classifier-free guidance:** drop each frame's action to a null token with p≈0.2 in training; at sampling use ε = (1+λ)ε(a) − λε(∅) | *Vid2World: Crafting Video Diffusion Models to Interactive World Models* · ICLR 2026 · 2505.14357 · ~85 | On RT-1 (AR, 30k steps), guidance improves every metric: FVD 28.6 → 22.4, FID 7.52 → 6.16, PSNR 23.4 → 23.9. p=0.2, λ=2.5, rescale 0.7 | Dropout costs a few % of teacher-forced loss; guidance is inference-only | S (train) + S (sampler: 2 forward passes) | Fine-tune with p=0.1-0.2 (null = zero vector after normalisation, or a learned token); sweep λ ∈ {0, 1, 2.5} at eval. Main metric: action-sensitivity score + rollout error. GameNGen found action CFG did **not** help in Doom, so this needs testing, not assuming |
| A3 | **History guidance / per-frame independent noise (Diffusion Forcing Transformer):** train with an independent noise level per latent frame, so "context" is just low-noise frames; at sampling, apply CFG between the true history and a fully noised history (optionally a partially noised one: "fractional" HG) | *History-Guided Video Diffusion* (DFoT) · ICML 2025 (PMLR 267) · 2502.06764 · ~185 | Kinetics-600 FVD 4.3 vs 4.8 for a standard conditional baseline. 64-frame rollouts: FVD 247.5 (baseline) → 208.0 (DFoT) → 170.4 (fractional HG). Converting a pretrained model took ~12.5% of the original training cost | Per-frame noise changes the training distribution, so val loss is not comparable with the baseline's. Judge by rollouts | M (train objective) / S (guidance if context dropout already exists) | Cheapest first step: history CFG alone. Cosmos already supports zeroing the context (`use_video_condition=False`); train with context dropout 0.1 and sample with history guidance 1.5 (GameNGen used 1.5 on history and warned that larger values add drifting artefacts). Full per-frame noise is a second step |
| A4 | **Representation alignment (REPA) to a frozen video encoder, fine-tuning variant:** add a loss aligning a mid-layer of the DiT (through a small MLP) to features of a frozen SSL encoder. For fine-tuning, align token-to-token *relations* (VideoREPA's "TRD" loss) rather than the features themselves | *REPA* · ICLR 2025 · 2410.06940 · ~810. *VideoREPA* · NeurIPS 2025 · 2505.23656 · ~110. *HASTE: "REPA Works Until It Doesn't"* · NeurIPS 2025 · 2505.16792 · ~40 | REPA: >17.5× faster SiT training (ImageNet, from scratch). VideoREPA: substantially better physics commonsense when fine-tuning CogVideoX, where plain REPA helps less. HASTE: switch the alignment off after an early phase, because REPA plateaus or hurts later; 28× fewer steps | Adds an auxiliary loss; report the flow loss term alone. Evidence is mostly from-scratch training and image generation. The benefit for fine-tuning a pretrained 2B model on 5 h of data is uncertain | M | Teacher = the frozen V-JEPA 2 ViT-g we already run for (B) (or DINOv2). Align block ~8 of 28 (REPA found early-to-mid layers best), λ=0.5, with a HASTE-style cutoff (off after 50% of steps). Compare flow-loss val and rollout error at equal steps |
| A5 | **Train on its own rollouts (error recycling, cheap):** inject the model's own past prediction errors into the clean context/inputs during fine-tuning, with errors stored in a replay bank | *Stable Video Infinity: Infinite-Length Video Generation with Error Recycling* · ICLR 2026 (oral) · 2510.09212 · ~75 | Videos from seconds to "infinite" length with no extra inference cost (authors' claim; benchmark numbers not checked) | Targets rollout error, not teacher-forced loss | M | After A1, if drift remains: generate 1-step predictions for training clips, store the residuals, and add them to the context in 50% of batches. Compare 3-chunk rollout error against A1 |
| A6 | **Self Forcing:** train by autoregressively unrolling the model's own chunks with KV cache, using a sequence-level distribution-matching loss (DMD) and few-step sampling | *Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion* · NeurIPS 2025 (spotlight) · 2506.08009 · ~710. Predecessor: *CausVid* · CVPR 2025 (**other venue**) · 2412.07772 · ~500 | Real-time streaming with sub-second latency on one GPU; quality matches or beats slower non-causal models | Produces a few-step distilled student; the denoising val loss no longer applies | L (although DreamDojo ships `action_video2world_self_forcing.py`) | Only after the base fine-tune is chosen: run DreamDojo's self-forcing stage on the best checkpoint and compare 4.8 s rollouts with the multi-step teacher. Costs a teacher, a critic and a student in memory |
| A7 | **Shorter-LR-decay finish (WSD / linear decay to zero):** see X1 below; for DreamDojo it is the cheapest gain, because a cooldown branch can start from an existing checkpoint | see X1 | | lowers val loss directly | S | see X1 |

## (B) V-JEPA 2-AC (frozen ViT-g, ~300M frame-causal predictor, L1 TF + rollout)

The most directly relevant ablation study, *What Drives Success in Physical Planning with Joint-Embedding Predictive
World Models?* (JEPA-WMs, arXiv 2512.24497, ~60 cites; Semantic Scholar lists TMLR, which I did not verify), is
**other venue**. I still use it as the main evidence because it ablates exactly the V-JEPA 2-AC recipe. Top-venue
support comes from DINO-WM (ICML 2025) and PLDM (NeurIPS 2025). Note that V-JEPA 2 itself (2506.09985, ~970 cites)
has **no confirmed venue** (arXiv only).

| # | Trick | Paper · venue · year · arXiv · ~cites | Claims | Effect on our val loss | Diff. | Cheap test |
|---|---|---|---|---|---|---|
| B1 | **Tune the rollout-loss horizon** (number of autoregressive steps in the training loss) | JEPA-WMs · **other venue** (TMLR?) · 2512.24497 · ~60; V-JEPA 2 · arXiv only · 2506.09985 · ~970 | Teacher forcing → +2-step rollout loss helps everywhere. More steps hurt in simple simulated environments, but **6 steps was best on DROID** (real Franka) | Directly lowers multi-step val loss if val includes rollout steps; make sure val reports TF and rollout parts separately | S (config) | Sweep rollout steps ∈ {2, 4, 6} on one robot at equal wall-clock; select on the rollout part of val loss at horizon 4.8 s |
| B2 | **Tune the training context length W** (frames of real history the predictor sees in the TF pass) | JEPA-WMs (above); DINO-WM · ICML 2025 · 2411.04983 · ~530 (frame-history predictor on frozen DINOv2 patches) | W=1 much worse than W=2 (the predictor needs two frames to infer velocity). Best W=3 in simulation and W=5 on DROID. **Inference context must not exceed training context** (predictions degrade quickly) | Our eval starts from frame 0 only (1 context frame). Make sure training includes W=1 windows, or we test outside the trained distribution | S | Mix W ∈ {1..W_max} per batch vs fixed; evaluate free-running from 1 frame |
| B3 | **Action-conditioning architecture: AdaLN (with RoPE) vs action tokens** | JEPA-WMs (above) | AdaLN had the best average, but by a small margin that varied by environment. AdaLN-zero was worse on DROID/Push-T | Can lower val loss | M (predictor surgery; loses released action weights) | Low priority. Only if the action-sensitivity score is poor: replace the action token with AdaLN modulation in each block, fine-tune, compare |
| B4 | **Proprioceptive input** | JEPA-WMs (above) | Models with proprio input were consistently better than visual-only | Our commands-only rule allows only the start state, which we already feed (repeated) | — | Already done within our rules; listed so nobody re-adds future states |
| B5 | **Optimizer change (SOAP or AdEMAMix) for the predictor** | *SOAP* · ICLR 2025 · 2409.11321 · ~240; *AdEMAMix* · ICLR 2025 · 2409.03137 · ~60 | SOAP: >40% fewer iterations and >35% less wall-clock than AdamW (360M/660M LMs, large batch). AdEMAMix: a 1.3B LM on 101B tokens ≈ AdamW on 197B, and slower forgetting | Lowers val loss at fixed steps, if it transfers (evidence is from LMs only) | S (drop-in packages) | Same seed and data, 1 robot, 3 runs: AdamW / SOAP (precond. freq 10) / AdEMAMix; compare val-loss curves at equal wall-clock |
| B6 | **LR cooldown (WSD / decay to zero):** see X1. Our current schedule is constant after warm-up, which is exactly the "stable" phase that WSD expects to be followed by a decay | see X1 | | lowers val loss directly | S | see X1 |
| B7 | **Context-feature noise augmentation** (A1 transplanted to feature space): add Gaussian noise to the context features, or recycle the predictor's own 1-step predictions as context | GameNGen · ICLR 2025 (above); idea of training on own outputs: Self Forcing · NeurIPS 2025 (above). No JEPA-specific 2025 venue paper found | — (no feature-space evidence) | Rollout loss already covers part of this | S | σ ∈ {0, 0.1, 0.2}·std of layer-normed features on context tokens; compare rollout val |

## (C) DreamerV3-style RSSM (offline world model only, 64-128 px)

The base recipe is *Mastering diverse control tasks through world models* (DreamerV3), Nature 2025, arXiv 2301.04104
(**other venue**; ~1,540 cites on the arXiv record). Its symlog, free bits (1 nat), KL balancing, unimix and
LayerNorm are already in the port.

| # | Trick | Paper · venue · year · arXiv · ~cites | Claims | Effect on our val loss | Diff. | Cheap test |
|---|---|---|---|---|---|---|
| C1 | **Action-conditioned contrastive predictive coding (AC-CPC) auxiliary loss:** from state s_t and the future actions a_{t:t+k}, predict (InfoNCE, batch negatives) the representation of the augmented future observation, for k=1..10 | *Learning Transformer-based World Models with Contrastive Predictive Coding* (TWISTER) · ICLR 2025 · 2503.04416 · ~35 | Atari 100k mean HNS 162% vs 112% without AC-CPC, and 111% without the action conditioning. K=10 was best (15 worse). Random crop-resize augmentation matters (120% without it). The same loss on a DreamerV3 RSSM gave 121% | Adds a loss term, so compare on image NLL + KL alone and on open-loop prediction error. Purpose: action-following and long-horizon latents (good for state probes) | M | Add the AC-CPC head (one MLP pair per k, K ∈ {5, 10}) to the port, weight 1. Compare open-loop 4.8 s image MSE, action-sensitivity score and the linear state probe vs baseline. RSSM runs are cheap, so use 3 seeds |
| C2 | **Data augmentation + frame/action stacking for robustness to the model's own errors** | *Simple, Good, Fast: Self-Supervised World Models Free of Baggage* (SGF) · ICLR 2025 · 2506.02612 · ~7 | Competitive on Atari 100k without RNN, transformer, discrete latents or reconstruction; augmentation is used to make the model robust to its own errors (I did not verify the numeric ablations) | Augmentation usually raises train loss and may lower val loss on 5 h of data (overfitting regime) | S | Random-shift (±4 px at 64 px) or crop-resize on encoder inputs only, with the decoder target un-augmented. Compare val loss and open-loop error |
| C3 | **More real context before imagination** (burn-in frames) | DreamerV3 (baseline); context-length evidence from GameNGen (ICLR 2025): 1 history frame PSNR 20.94 vs 2 frames 22.03 vs 8 frames 22.26 | 1→2 context frames is the big step; beyond ~4 it saturates | Our eval gives 1 frame, so this applies only if the eval protocol changes. Otherwise skip | S | Only if the protocol allows 2+ frames |
| C4 | **LR cooldown (WSD / linear decay to zero)**: see X1 | see X1 | | lowers val loss | S | see X1 |
| C5 | **SimBa-style network blocks** (input running-stat normalisation, pre-LN residual MLP blocks) in the RSSM MLPs / heads | *SimBa: Simplicity Bias for Scaling Up Parameters in Deep RL* · ICLR 2025 · 2410.09754 · ~90 | Better sample efficiency across off/on-policy and unsupervised RL when scaling parameters (SAC on DMC, MyoSuite, HumanoidBench) | Evidence is from actor/critic networks, not world models; transfer is speculative | S-M | Only if we scale the RSSM to `large` and it underfits; swap MLP blocks, compare val loss at medium/large |
| C6 | **Optimizer swap (SOAP / AdEMAMix)**: see B5 | ICLR 2025 ×2 | | | S | same protocol as B5, 3 seeds |

## Cross-cutting

| # | Trick | Paper · venue · year · arXiv · ~cites | Claims | Diff. | Cheap test |
|---|---|---|---|---|---|
| X1 | **Warmup-stable-decay (WSD) / linear decay to zero:** keep a constant LR as long as useful, then branch a short decay phase (~10-20% of steps) to 0 from a checkpoint | *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape View* · ICLR 2025 · 2410.05192 · ~90. *Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs* · ICLR 2025 · 2502.15938 · ~40 | WSD: a high-LR phase makes progress "along the river", the decay phase removes the oscillation; WSD-S beats WSD and cyclic cosine across budgets. D2Z: linear decay to zero beats cosine-to-10%; a 610M model at 80 tokens/param with D2Z beat 200 tokens/param with 10× decay (~60% compute saved) | S | All three models: take the current best (constant-LR) checkpoint and run a linear decay to 0 over 10-20% of the steps already run. Compare val loss with continuing at constant LR for the same steps. Our early-stopping-on-plateau rule fits WSD well (plateau → decay branch) |

---

## Ranked shortlist: what to try first (expected value per GPU-hour)

1. **X1 LR cooldown branch (all three models).** S; ~10-20% extra steps from an existing checkpoint. Two ICLR 2025
   papers. It lowers exactly the metric we select on, and our constant-LR + plateau-stop setup currently leaves this
   gain unused.
2. **B1 rollout-loss horizon sweep {2, 4, 6} for V-JEPA 2-AC.** S; cheap runs. Directly ablated (6 best on real
   DROID). Other-venue evidence (JEPA-WMs), but the setting is the closest to ours of any paper on this list.
3. **B2 mixed context length incl. W=1 for V-JEPA 2-AC.** S. Our eval rolls out from 1 frame; the paper shows
   inference context > training context degrades quickly, and W=1 behaves very differently.
4. **A1 context-frame noise augmentation for DreamDojo.** S; the hook exists (`conditional_frame_timestep` +
   adding noise to the context latents). ICLR 2025, ~360 cites. Needs the rollout metric.
5. **C1 AC-CPC auxiliary loss for the RSSM.** M; very cheap runs. ICLR 2025, a large ablation effect (162 vs 112
   HNS), and should help action-following and state probes too.
6. **A2 action dropout (p≈0.1-0.2) + action CFG for DreamDojo.** S. ICLR 2026, consistent gains on RT-1 robot
   video. Judge by the action-sensitivity score; contradictory evidence in GameNGen (no gain in Doom).
7. **C2 input augmentation for the RSSM.** S. ICLR 2025 (SGF) and TWISTER both find augmentation important. On 5 h
   of data per robot, overfitting is likely the binding constraint.
8. **A3 history guidance via context dropout for DreamDojo.** S for the guidance-only version, M for full per-frame
   noise (DFoT). ICML 2025, ~185 cites, large long-rollout FVD gains.
9. **A4 REPA / VideoREPA alignment to the frozen V-JEPA 2 encoder, with a HASTE early cutoff.** M; costs extra
   encoder forwards. Strong venue evidence (ICLR + 2× NeurIPS 2025), but mostly from-scratch training, so the gain
   for a short fine-tune is uncertain.
10. **B5/C6 SOAP or AdEMAMix in place of AdamW (B and C only).** S; ICLR 2025 ×2, but evidence comes from language
    models only.

Later or only if needed: A5 error recycling (ICLR 2026) if drift remains after A1; A6 Self Forcing (NeurIPS 2025;
expensive, and it changes the model into a few-step student); B3 AdaLN action conditioning; C5 SimBa blocks.

---

## Considered and excluded (and why)

| Item | Why excluded |
|---|---|
| Diffusion Forcing (NeurIPS 2024, 2407.01392, ~830) | Before 2025. Its 2025 follow-up DFoT (A3) is listed instead |
| Schedule-free AdamW, *The Road Less Scheduled* (NeurIPS 2024, 2405.15682) | Before 2025, as instructed; WSD/D2Z (ICLR 2025) cover the same need |
| Logit-normal / shifted timestep sampling, SD3 *Scaling Rectified Flow Transformers* (ICML 2024, 2403.03206, ~5,000) | Before 2025, and Cosmos already uses logit-normal + shift. I found no 2025 top-venue paper with a clearly better flow-matching timestep sampler or loss weighting (several 2025-26 arXiv-only analyses). A shift sweep is cheap but has no new evidence behind it |
| Muon (*Muon is Scalable for LLM Training*, 2502.16982, ~460) | arXiv only. The ICLR 2026 Muon papers I found are convergence theory / communication efficiency, not empirical training-quality evidence |
| LeJEPA / SIGReg (2511.08544, ~300) | arXiv only (no venue found), and it prevents collapse when training the *encoder*. Our V-JEPA encoder is frozen, so collapse is not our problem |
| Dispersive loss, *Diffuse and Disperse* (2506.09027, ~85) | No venue confirmed (arXiv only) |
| REG, *Representation Entanglement for Generation* (NeurIPS 2025 oral, 2507.01467, ~115) | Entangles a class token from a foundation model with the latents; designed for class-conditional image generation, with no obvious analogue for action-conditioned video. Revisit only if A4 works |
| REPA-E (ICCV 2025, 2504.10483, ~250) | Other venue; requires end-to-end tuning of the VAE. We keep the Wan VAE frozen (shared tokenizer), so not worth it |
| *Improving Transformer World Models for Data-Efficient RL* (ICML 2025, 2502.01591, ~17) | Its gains come from Dyna-with-warmup (RL-only), a nearest-neighbour patch tokenizer (for discrete-token transformer WMs on 2D Craftax) and block teacher forcing. V-JEPA 2-AC's frame-causal predictor already predicts all tokens of the next frame in parallel, i.e. it is block teacher forcing already. Nothing transfers cleanly |
| Drama (ICLR 2025, 2410.08893, ~9) | An architecture swap (Mamba for the RSSM) with competitive but not better Atari 100k scores; low citations |
| Long-Context State-Space Video World Models (ICCV 2025, 2505.20171, ~80) | Other venue; targets memory over long horizons. Our windows are 4.8 s |
| FramePack / *Frame Context Packing and Drift Prevention* (NeurIPS 2025, 2504.12626, ~155) | Anti-drift via early-established endpoints and inverted sampling order requires knowing the end frame (not available in action-conditioned forward prediction); context packing targets thousands of frames |
| Navigation World Models (CVPR 2025, 2412.03572, ~375) | Other venue; an architecture (CDiT) for training from scratch, not a fine-tuning trick |
| AdaWorld (ICML 2025, 2503.18938, ~135) | Latent-action pretraining for when action labels are missing; we have native actions |
| Ctrl-World (ICLR 2026, 2510.10125, ~210) | Contributions are multi-view input, pose-conditioned memory retrieval and frame-level action conditioning. DreamDojo already conditions per frame; memory retrieval matters beyond 4.8 s |
| DINO-Foresight (NeurIPS 2025, 2412.11673, ~55): multi-layer + PCA feature targets | Changing the prediction target would make (B)'s val loss and probe features incomparable with the released V-JEPA 2-AC setup |
| PLDM (NeurIPS 2025, 2502.14819, ~125) | Main lesson concerns training the encoder with JEPA/VICReg-style objectives on reward-free data; our encoder is frozen. It does support multi-step latent rollout losses (B1) |
| EMA of weights for (B)/(C) | Sensible and nearly free (DreamDojo already validates its EMA), but I found no 2025 top-venue paper to cite; try it as engineering hygiene, not as a literature trick |

---

## Protocol as run (2026-10-26)

Applied to all three sweeps (configs/sweeps/*.yaml; orchestrators in cluster/sail/):

- **Seeds.** Grid runs have one seed. The two best grid runs get seeds 1 and 2 (`<run>-seed<N>`), and every trick,
  control and stacked run is trained at seeds 0, 1 and 2. Only the training-data order (and the model init where it is
  random) changes; validation is identical for every seed.
- **Matched controls.** Every LR cooldown is paired with a control that trains the same extra steps at constant LR
  (DreamerV3 `trick_wsd_control`, V-JEPA `t_cooldown_control`, DreamDojo `trick_cooldown_control`).
- **Judging.** DreamerV3 and V-JEPA tricks: the sweep's own selection number (open-loop MSE@64 / val loss_ar).
  DreamDojo cooldown: validation loss, against its control. DreamDojo context noise, action dropout and history
  dropout: rollout latent MSE (32 fixed val episodes, 4.8 s), each at its best sampling variant, because their
  benefit only shows when sampling with guidance / a context-noise level (a teacher-forced denoising loss cannot see it).
- **Stack.** A trick enters the stacked run when its worst seed beats the best grid run's 3-seed mean (V-JEPA's
  cooldown must also beat its control's mean). When two winners set the same setting (two optimizers), the better mean
  keeps it. Stacks are trained from scratch, so cooldown branches are reported but not stacked (DreamerV3, DreamDojo).
  Decisions: DreamerV3 <storage_dir>/sweeps/dreamerv3/<robot>/stack.json, V-JEPA
  <storage_dir>/checkpoints/vjepa2/<robot>/stack.json, DreamDojo <results_dir>/sweeps/dreamdojo/<robot>_stack.json.
- **Evidence caveats.** SGF (~7 cites), SOAP / AdEMAMix (language-model evidence only) and JEPA-WMs (not an allowed
  venue) are kept because they are cheap; results that rest on them are labelled exploratory.
