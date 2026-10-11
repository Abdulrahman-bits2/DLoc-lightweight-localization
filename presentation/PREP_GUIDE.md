# Preparation Guide — Concepts & Q&A, by Section

How to use this: each section below is one block of the talk. For your block, (1) make sure you can
explain every item under **Own these concepts** *in one breath, without the slide*; (2) rehearse the
**likely questions**; (3) memorize the **deflection lines** so a cross-section question never stalls
you. The deeper conceptual treatment of everything here is in `../guide/` (the concept guide) — go
there when an answer feels shaky.

**Golden rules for Q&A**
- If you know it, answer in 2 sentences, then stop. Over-answering invites follow-ups.
- If you don't know it, say "we didn't test that, but our expectation is X because Y" — reasoning beats
  bluffing. Never invent a number.
- If it belongs to a teammate's section, hand off: *"[Name] covered the setup — [Name], do you want to
  take the cross-session split?"*
- Numbers to never get wrong: **0.707 / 0.640 m**, **16.5M → 2.41M (6.8×)**, **3.4× noise (KD)**,
  **5.5× noise (skips)**, **56× FLOPs / 10× CPU**, **0.930 vs 0.86 m** cross-session, **~2M** capacity
  threshold.

---

## Section 1 — Introduction (problem + the WiFi signal)  · Presenter 1

**Own these concepts**
- Why GPS fails indoors: walls reflect/attenuate the satellite signals; timing breaks.
- What CSI is: complex channel per subcarrier × antenna. Amplitude → distance; phase across antennas →
  angle of arrival (AoA); phase across subcarriers → time of flight (ToF).
- Multipath & NLOS: signal arrives via many reflected paths; the strongest is often *not* the direct
  one — this defeats classical geometry (e.g. SpotFi).
- The clock offset: unsynced clocks add an unknown delay → each AP's heatmap shifts radially 10–15 m →
  the 4 APs disagree. Reconciling them is what learning buys you.

**Likely questions**
- *Why a heatmap, not direct (x,y) regression?* Heatmaps keep spatial structure and represent
  uncertainty/ambiguity (two plausible spots = two bumps); the argmax still decides. Direct regression
  throws that away and can't use convolutional locality.
- *Why deep learning over SpotFi/MUSIC?* Those estimate AoA/ToF analytically and fail under NLOS and
  rich multipath. A network learns to pick the direct path by cross-checking APs and to undo the offset.
- *How many APs / subcarriers?* 4 APs in Jacobs Hall; CSI is per-subcarrier per-antenna (Quantenna
  card). The input image is 4 × 161 × 361.

**Deflection lines**
- "How exactly is CSI turned into a heatmap?" → *"There's a MATLAB AoA–ToF step — it's in our dataset
  backup slide, I can show it after."*

---

## Section 2 — DLoc  · Presenter 1

**Own these concepts**
- Encoder + **two** decoders: shared ResNet encoder → (a) location decoder = the heatmap we use,
  (b) consistency/offset decoder = training-only.
- What the consistency decoder does: forces the 4 AP maps to agree → learns to (1) remove the ToF
  offset, (2) resolve multipath (pick direct path), (3) super-resolve low-bandwidth inputs. One AP
  can't disambiguate itself; four mutually-consistent APs can.
- Why it's heavy: three ResNet stacks at 64 filters → 16.5M params, 81 GFLOPs, 271 ms CPU.
- The thesis question: is that cost inherent, or over-parameterization?

**Likely questions**
- *Your DLoc is 0.707 m but the paper says 0.64 — why?* An implementation gap in our reproduction. It
  doesn't undermine us: cross-session comparisons use the *published* DLoc numbers (fair), and in-dist
  comparisons are all against our own consistent reproduction.
- *Do your students keep the consistency decoder?* No — they're single-decoder. That's exactly why we
  trail DLoc slightly on the reflector-fold median (where that decoder helps most) — but we still beat
  its reflector P90.
- *What's InstanceNorm / why not BatchNorm?* InstanceNorm normalizes per-sample, which is standard for
  image-translation generators; it's what DLoc uses.

**Deflection lines**
- "Show me the reflector effect" → *"That's in the cross-session and qualitative backup slides."*

---

## Section 3 — Our method / Knowledge distillation  · Presenter 1

**Own these concepts**
- Standalone vs distilled = two training modes (ground-truth only vs ground-truth + teacher).
- The loss: L = 0.7·MSE(ŷ, y) + 0.3·MSE(ŷ, y_T). Ground truth dominant; teacher is a gentle regularizer.
- "Dark knowledge" / soft targets: a teacher's full output carries structure a hard label omits. Here
  the teacher's heatmap is *already soft* (mass spread around the location) → **spatial label smoothing**.
- Teacher precomputed (~4 GB train / 1 GB test) → never run during student training → cheap to distill
  many students/seeds.
- The dual nature (preview): helps robustness above ~2M params; *harms* below it.

**Likely questions**
- *Why distill if it doesn't improve accuracy?* Because accuracy was never the goal here — in clean
  conditions the teacher ≈ ground truth, so it's redundant; the value shows up off-distribution (noise,
  new sessions), which is the realistic deployment case. It's a robustness technique.
- *Why 0.7/0.3?* Keep ground truth dominant; it's a convex blend, not heavily tuned. Distillation
  weighting is standard.
- *Is this real distillation without a softmax/temperature?* Yes — the temperature's job (smoothing the
  target) is already done for us by the teacher's spread-out heatmap.

**Deflection lines**
- "Where's the proof it helps robustness?" → *"That's the model results — [Presenter 3] will show the
  3.4× and 5.5× numbers."*

---

## Section 4 — Dataset (WILD)  · Presenter 2

**Own these concepts**
- WILD = public dataset released with DLoc (UC San Diego).
- MapFind = TurtleBot2 with Hokuyo LIDAR + Orbbec RGB-D + Quantenna WiFi, driving 15 cm/s.
- Ground truth from SLAM (RTAB-Map), median ≈ 5.7 cm, validated vs an HTC Vive tracker; CSI every 50 ms;
  ~100k points total.
- Two spaces: Atkinson (3-AP, simple, unused) and **Jacobs Hall (4-AP, complex, used)** — AP4 behind a
  wall = NLOS, plasma screens behind AP3 = multipath.
- 5 sessions; in-dist = sess 1+2 (17,574 train / 4,260 test); sess 3–5 = cross-session; reflector
  (sess 5) hardest.
- Feature internals (backup): AoA–ToF profile, grid 161×361 @ 0.1 m, Gaussian label σ=0.3 m, with/
  without-offset versions, HDF5 with reversed dims.

**Likely questions**
- *How accurate is the ground truth?* ~5.7 cm median (SLAM), well below our ~0.6 m localization error,
  so labels aren't the bottleneck.
- *Why only one building?* The public data is Jacobs Hall; cross-*building* data isn't available —
  collecting an E-JUST dataset is our future work.
- *Why is the reflector hardest?* An aluminium panel creates specular multipath — convincing phantom
  reflections — so the heatmaps gain spurious peaks.
- *What's the with/without-offset thing?* Two versions of the per-AP heatmaps; "with offset" is the
  network input, "without" is the consistency-decoder target.

**Deflection lines**
- "Which split did model X use?" → *"That's the next slide — the setup table covers every run."*

---

## Section 5 — Experimental setup / datasplits  · Presenter 2

**Own these concepts (this is the "why each split" slide — own it cold)**
- **In-distribution:** train sess1 + sess2-train (17,574); test sess2 held-out (4,260). Used for
  accuracy, scheduler, multi-seed. *Why:* same-conditions generalization, matches DLoc's "Fig. 10b".
- **Robustness:** same split; corrupt the CSI at **test time only**, no retraining. *Why:* measures how
  a *deployed* model copes with unseen degradation (not how well it can be trained for it).
- **Cross-session LOSO (fair):** 3 folds, each leaves out one August session, trains on the rest, tests
  on the held-out one; 3 seeds; **standalone only**. *Why standalone:* distilling would need a fresh
  teacher per fold, breaking comparability — this way it matches the *published* DLoc protocol exactly.
- **Zero-shot:** train on the 2 July sessions only, test on August. *Why:* an internal architecture/
  distillation ablation under a *harder* shift — explicitly **not** a DLoc comparison.
- Hardware: H100 (Vast.ai) for training; local RTX 3060 for benchmarking/figures.

**Likely questions**
- *Why is cross-session standalone and not distilled?* (See above — teacher-per-fold; keeps it fair.)
- *Isn't zero-shot a weaker result than DLoc?* It's not meant to beat DLoc — it uses *less* training
  data on purpose, to isolate which architecture/training choice helps under a bigger shift.
- *Why test-time-only corruption?* To measure deployed robustness honestly; training-time augmentation
  is future work and would likely help.
- *How many runs total?* In-dist (all models, ×modes), robustness (6 conditions × models), LOSO (3
  folds × 3 seeds), zero-shot, plus baseline multi-seed (3).

**Deflection lines**
- "Why did the U-Net use cosine vs step LR?" → *"Scheduler barely moved the median — [Presenter 3] has
  the comparison; architecture dominates."*

---

## Section 6 — MobileNetV2  · Presenter 2

**Own these concepts**
- Depthwise-separable conv = depthwise (space, per-channel) + pointwise (1×1, channels) → ~8–9× fewer
  MACs. Inverted residual + linear bottleneck = MobileNetV2's block.
- 2.3M params, 7.2× smaller; matches DLoc's 0.707 m **losslessly** → DLoc was over-parameterized.
- Noise-fragile standalone (16.4 m @ σ=0.1) because the bottleneck keeps no shallow shortcut → noise
  scrambles the latent → peak jumps. Distillation → 4.8 m (**3.4×**).

**Likely questions**
- *How can 2.3M match 16.5M?* Depthwise-separable convs preserve representational power cheaply; DLoc's
  extra ~14M was redundant for the clean mapping.
- *Why does it fail under noise but DLoc/U-Net don't as badly?* No skip path around the bottleneck; a
  single corrupted latent dooms the prediction. (Distillation or skips fix it.)
- *Distilled vs standalone in-distribution?* Identical median — distillation only shows up off-dist.

**Deflection lines**
- "Show the noise numbers for all models" → the robustness results slide (slide 23).

---

## Section 7 — TinyCNN  · Presenter 2

**Own these concepts**
- 3 conv + 2 FC, 47K params (351× smaller) — a deliberate *floor probe*.
- 3.9 m (≈5× baseline): below the ~2M capacity floor; can't learn per-AP features **and** spatial
  fusion together.
- Distillation **harms** it (dropout 7.9 → 13.6 m): no spare capacity; mimicking the teacher's AP
  interactions backfires when an AP drops.

**Likely questions**
- *So distillation isn't always good?* Correct — that's a finding. It needs a student rich enough
  (~>2M) to absorb the teacher's hints without sacrificing its own fallback behavior.
- *Could more training fix TinyCNN?* No — it's a capacity ceiling, not an optimization issue; error
  plateaus.

**Deflection lines**
- "What's the exact capacity threshold?" → *"Empirically ~2M — MobileNetV2 (2.3M) helps, TinyCNN (47K)
  hurts; we didn't sweep the exact boundary."*

---

## Section 8 — MobileNetV2-UNet (the hero)  · Presenter 3

**Own these concepts**
- = MobileNetV2 encoder + **U-Net skip connections** (concatenate encoder features into the decoder at
  matching resolution, bypassing the bottleneck). 2.41M params, 6.8× smaller.
- Beats the teacher: 0.640 m (9.5% better), and **6.5σ** beyond the baseline's seed spread → real.
- Robustness 5.5× from skips alone (shallow features bypass the noisy bottleneck) — *independent* of
  distillation.
- Cross-session 0.930 m vs DLoc 0.86; matches different-furniture (0.822 vs 0.82); beats reflector P90
  (2.49 vs 2.77). Efficiency 56× FLOPs / 10× CPU (27 ms).
- Two mechanisms complementary, not additive: skips→noise, distillation→AP dropout.

**Likely questions**
- *How does a student beat its teacher?* It's not more capacity — it's better *structure*. Skips
  restore the spatial detail the bottleneck discards, and spatial detail is the answer.
- *Is 0.640 vs 0.707 within noise / quantization?* No: 9.5%, 6.5σ, and it holds across the full CDF and
  the P90/P99 tails — not just the median. (Quantization detail in backup slide 30.)
- *Why behind DLoc on the reflector median?* No consistency decoder; that mechanism helps most under
  specular multipath. We still win its P90, and the gap is ~0.07 m on the hardest fold only.
- *Cosine vs step LR?* Cosine has slightly tighter tails; the median is identical — architecture
  dominates, not the schedule.

**Deflection lines**
- "Per-seed cross-session numbers?" → slide 24. "Full robustness?" → slide 23.

---

## Section 9 — Mamba (the negative result)  · Presenter 3

**Own these concepts**
- Selective state-space model; processes a **1-D sequence**, not a 2-D image. 0.65M params.
- Worst model (10.56 m). Best at epoch 1, then degrades; **train and test error rise together**.
- Why it fails: architectural mismatch — flattening the 2-D grid destroys spatial locality. NOT
  capacity (bigger than TinyCNN), NOT overfitting (train+test rise together).
- The point: confirms the **image framing** is load-bearing.

**Likely questions**
- *Did you tune it / give it a fair chance?* It diverges after epoch 1 with train and test moving
  together — that's a fitting/inductive-bias problem, not under-training; more tuning won't fix a
  framing mismatch.
- *Could a 2-D vision-Mamba work?* Plausibly — that's future work (Mamba *layers* inside a 2-D model,
  not a 1-D flatten). Our experiment was the 1-D sequence variant.
- *Why keep a failed model in the thesis?* A clean negative result is evidence — it isolates *why* the
  others work.

**Deflection lines**
- "Mamba's datasplit?" → *"Same in-distribution split as the others — the setup table."*

---

## Section 10 — Comparison, conclusion, future  · Presenter 3

**Own these concepts (the 6 findings, in one line each)**
1. DLoc over-parameterized → 7.2× lossless (MobileNetV2).
2. Architecture beats parameters → U-Net 0.640 m, 6.8× smaller, 6.5σ.
3. Distillation = robustness not accuracy → 3.4× noise.
4. Distillation capacity threshold ~2M (TinyCNN harmed).
5. Two robustness mechanisms complementary, not additive (skips→noise, KD→dropout).
6. Competitive cross-session at 6.8× compression + 56× cheaper compute.
- One-sentence contribution: *an order-of-magnitude-smaller model matches/beats DLoc by adding the
  right structure (skips), not parameters, plus distillation for robustness — with a capacity threshold.*

**Likely questions**
- *Is this an accuracy win or an efficiency win?* Honestly: an **efficiency** result (in-dist we beat
  DLoc; cross-session we're competitive, not ahead). We state it that way — that's rigor.
- *What's the single most important result?* The U-Net: better than the teacher in-distribution at
  6.8× fewer parameters and 56× less compute.
- *What would you do with more time?* Deploy on E-JUST edge hardware; collect a cross-building dataset;
  multi-seed the students; training-time augmentation.
- *Real-world impact?* 27 ms CPU inference makes on-device / on-AP localization realistic — no GPU, no
  cloud round-trip.

**Deflection lines**
- "Statistical significance for the students?" → *"Baseline is 3-seeded (0.875 ± 0.036); the U-Net's
  cross-session is 3-seeded; per-student multi-seed was de-scoped for compute — it's in future work."*

---

## Master slide map (know which to jump to)

The deck now presents the key result tables in the main flow (slides 21–26); only three backups remain.

| If asked about… | Go to slide |
|---|---|
| In-distribution accuracy, all models | 21 (Results 1) |
| Cosine vs step scheduler | 22 (Results 2) |
| Full robustness numbers | 23 (Results 3) |
| Per-seed cross-session / reflector gap | 24 (Results 4) |
| Multi-seed stability / is it luck? | 25 (Results 5) |
| FLOPs / latency for all models | 26 (Results 6) |
| Metric, identical medians, quantization | 30 (backup B3) |
| CSI→heatmap, SLAM accuracy, grids | 31 (backup B4) |
| What a prediction looks like | 32 (backup B6) |

Press the slide number + Enter (or use the URL `#21`) to jump straight there during Q&A.
