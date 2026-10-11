# Speaker Script — 20-Minute Defense

**Talk length:** ~18–20 min of speaking + buffer, leaving the rest of the slot for Q&A.
**Slides:** 29 main + 3 backup (32 total). Flow: Overview → **Part 1** DLoc & method (4–10) →
**Part 2** the four model architectures (11–15) → **Part 3** data, setup, then the **Experiments &
Results** block (slides 21–26); close on 27–29. Backups (30–32) answer questions.
The same script is embedded in the deck as **speaker notes** — press **S** to see the current slide's.

> **⚠️ Restructured 2026-06-23 — this script's per-slide narrative is partly stale.** The deck is now
> **architecture-first**: the four models (Mamba → TinyCNN → MobileNetV2 → MobileNetV2-UNet) come
> **before** the data section, and each model slide is now **architecture-only** — all result numbers
> moved into the Results block (21–26). The model/data narrative entries below still describe the OLD
> results-on-model-slides layout and order. **The deck (press S for per-slide notes) is the source of
> truth.** The structural tables here (presenter split, timing, Results block 21–26) are current. Ask
> for a full narrative rewrite.

> **Pacing note for the Results block (21–26):** the model slides and the comparison already framed the
> story; 21–26 are the *evidence*. Walk them briskly. Slides **22 (scheduler)** and **25 (multi-seed)**
> are **fast / optional** — a 15–20 s mention each, or skip and jump only if asked. Cut these first.

## Suggested 3-presenter split (hand off at the Part dividers — slides 4, 11, 16)

| Presenter | Slides | Owns | ~time |
|---|---|---|---|
| **1** | 1–10 | Overview, the WiFi signal, DLoc, our method | ~6 min |
| **2** | 11–15 | The four model architectures (Mamba → TinyCNN → MobileNetV2 → MobileNetV2-UNet) | ~4 min |
| **3** | 16–29 | Data & setup, comparison, the Results block (21–26), conclusion, future | ~8 min |

Whoever owns a section also fields its questions; see `PREP_GUIDE.md`.

## Delivery rules (read once, internalize)

- **Don't read the slides.** The slide is the headline; you say the sentence. Eye contact > screen.
- **Lead with the number, then the meaning** ("0.640 meters — that *beats* the teacher").
- **One idea per slide.** If you're explaining a second idea, you've gone too long.
- **Deflection is fine.** If asked about something a later section covers: *"Great question — that's
  exactly the next section, can I come back to it in two slides?"* Then actually come back.
- **Pause after the key line.** Let "an order of magnitude smaller" land before moving on.

---

# PART 1 — The Problem & DLoc  (Presenter 1)

### Slide 1 — Title  *(~20 s)*
"Good morning. We're presenting our graduation project: compressing **DLoc**, a deep-learning WiFi
indoor-localization network, into a model small enough to run on edge devices — without losing
accuracy. I'm [name], with [names]. Over the next twenty minutes we'll cover the problem, our method
and data, then the results."

### Slide 2 — Agenda  *(~20 s)*
"Three parts: first the problem and the DLoc baseline; then our method and the dataset; then the four
models we built, the results, and our conclusions. We've kept detailed tables in backup slides for
your questions."

### Slide 3 — Part 1 divider  *(~5 s)*
"Let's start with the problem."

### Slide 4 — Where GPS can't go  *(~45 s)*
"Indoor location matters for a lot of things — tracking assets in a warehouse, patients in a hospital,
AR, robots. Outdoors GPS handles it, but indoors GPS fails: walls and ceilings reflect and weaken the
satellite signals. So we need an alternative — and WiFi is the obvious one, because the access points
are already installed and they already measure the signal, with no new hardware on the device."

### Slide 5 — From a WiFi signal to a location  *(~55 s)*
"How do you get a position from WiFi? Each access point measures **Channel State Information** — the
complex channel per subcarrier and antenna. Amplitude hints at distance; the phase across antennas
gives the angle of arrival; the phase across frequencies gives the time of flight. The hard part is
that indoors the signal bounces — multipath and non-line-of-sight — which breaks classical geometry.
And because the clocks aren't synchronized, each access point's estimate is shifted by an unknown 10
to 15 meters. So the four APs disagree, and a *learned* model reconciles them where geometry can't."

### Slide 6 — DLoc: localization as image translation  *(~45 s)*
"DLoc's key idea is to treat this as image-to-image translation. [CLICK through the pipeline.] Each
AP's signal becomes a heatmap over the floor; stack four of them into an image; a network translates
that into one clean heatmap; and the brightest pixel, at 0.1-meter resolution, is the location. Hold
on to this framing — four of our five models accept it, and the one that doesn't, fails."

### Slide 7 — One encoder, two decoders  *(~50 s)*
"Architecturally, DLoc is an encoder with two decoders. A shared ResNet encoder feeds a **location
decoder** — that's the heatmap we read — and a **consistency decoder**, which is DLoc's clever part:
it forces the four AP maps to *agree*, and in doing so it learns to remove the clock offset and pick
the direct path through the multipath. Remember the consistency decoder — it comes back when we get to
the reflector session."

### Slide 8 — Accurate, but too heavy for the edge  *(~45 s)*
"And it works — sub-meter accuracy. But look at the cost: 16.5 million parameters, 81 gigaFLOPs, and
271 milliseconds per prediction on a CPU. That's a non-starter for the gateways, phones, and access
points where indoor localization is actually useful. So our central question: **is that cost inherent
to the task, or is DLoc just over-parameterized?** The rest of the talk answers that."

### Slide 9 — Two levers: architecture & distillation  *(~55 s)*
"We attack it with two levers: architecture and knowledge distillation. We train each small student in
two modes — standalone, on ground truth only, and distilled, where it also matches the DLoc teacher,
with a 0.7/0.3 weighting. We precompute the teacher once, so it never runs in the training loop. The
key insight: unlike a classifier, DLoc's output heatmap is already *soft* — probability spread around
the location — so distillation here acts as **spatial label smoothing**. As you'll see, that buys
robustness, not clean accuracy." *(Hand off to Presenter 2.)*

---

# PART 2 — Data & Setup  (Presenter 2)

### Slide 10 — Part 2 divider  *(~5 s)*
"I'll take the data and how we ran every experiment."

### Slide 11 — WILD: real measurements with ground truth  *(~50 s)*
"The data is **WILD**, released publicly with DLoc. It was collected by a robot called **MapFind** — a
TurtleBot with LIDAR, a camera, and a WiFi card — that drives the space slowly while **SLAM** gives it
ground-truth position to about 5.7 centimeters. So every example is real CSI paired with an accurate
label, about 100,000 points in total. We use the harder four-AP **Jacobs Hall** space, where one AP is
hidden behind a wall — strong non-line-of-sight — and plasma screens create heavy multipath."

### Slide 12 — Five sessions, increasing difficulty  *(~45 s)*
"There are five sessions. Sessions 1 and 2 are the same room an hour apart — that's our
in-distribution train/test split, about 17,500 training and 4,260 test samples. Sessions 3 to 5 were
collected three weeks later with furniture changes and, in session 5, an aluminium reflector — these
test cross-session generalization, and the reflector, which injects phantom reflections, is the
hardest case."

### Slide 13 — The runs, and why each split  *(~60 s)*
"This slide is the one that answers *which split for which run, and why* — please stop me here if you
have a setup question. In-distribution and robustness use the standard session-1-plus-2 split;
robustness just corrupts the input at test time, no retraining, to measure *deployed* behavior.
Cross-session uses **leave-one-session-out**, which matches DLoc's published protocol exactly, so the
comparison is fair — and it's standalone-only, because distilling would need a fresh teacher per fold.
Zero-shot trains on July only — that's an internal ablation, not a DLoc comparison. Training was on
H100 GPUs; benchmarking on a local RTX 3060."
*(If asked about a model's split now: "We'll see that model in a moment — its split is the one on this
table; I'll point back here.")* *(Hand off to Presenter 3 after TinyCNN, or continue.)*

### Slide 15 — MobileNetV2: lossless compression  *(~55 s)*
"Our first student is plain MobileNetV2 — 2.3 million parameters, 7.2 times smaller, built from
depthwise-separable convolutions. It hits *exactly* the same 0.707-meter median as DLoc: the
compression is lossless, which already shows DLoc was over-parameterized. But standalone it's fragile —
under noise it collapses to 16 meters, basically guessing, because its bottleneck keeps no shallow
shortcut. Distillation rescues it to under 5 meters — a 3.4-times gain. That's our first evidence that
distillation is a *robustness* lever."

### Slide 16 — TinyCNN: below the capacity floor  *(~50 s)*
"Second student, TinyCNN — just 47 thousand parameters, a deliberate 'how small is too small' probe.
It lands at 3.9 meters, five times worse, because it's below the roughly 2-million-parameter capacity
floor for this task. And here distillation actually *hurts*: with no spare capacity, mimicking the
teacher's AP interactions backfires the moment an AP drops. So distillation has a capacity threshold —
keep that in mind for the conclusions." *(Hand off to Presenter 3.)*

---

# PART 3 — Models, Results & Findings  (Presenter 3)

### Slide 14 — Part 3 divider  *(~5 s — if Presenter 3 starts here)*
"Now the results — our four students and what they tell us."

### Slide 17 — MobileNetV2-UNet: beating the teacher  *(~60 s)*
"This is the hero. Take that MobileNetV2 encoder and add **U-Net skip connections** — the dashed paths
that carry high-resolution encoder features straight into the decoder, bypassing the bottleneck. The
result: **0.640 meters**, which actually *beats* the 16.5-million teacher by nine and a half percent,
at 6.8 times smaller. And that gain is six and a half standard deviations beyond the baseline's
seed-to-seed noise, so it's real, not luck. The reason is simple: the skips restore the spatial detail
the bottleneck throws away — and spatial detail *is* the answer."

### Slide 18 — U-Net: robust and edge-ready  *(~60 s)*
"It's not just accurate — it's robust and cheap. [POINT to chart.] The skips alone give a
five-and-a-half-times improvement under noise. Cross-session, it averages 0.93 meters against DLoc's
0.86 — it *matches* DLoc on the different-furniture fold and even beats its P90 on the hardest
reflector fold, at 6.8 times fewer parameters. And it's 56 times fewer FLOPs and ten times faster on
CPU. One nuance for the record: skips fight *noise*, distillation fights *AP dropout* — they're
complementary, not additive."

### Slide 19 — Mamba: an instructive failure  *(~50 s)*
"Our fourth student, Mamba, is a negative result — and a useful one. It's a state-space model that
treats the CSI as a one-dimensional *sequence* instead of an image. It's the worst model, 10.5 meters.
Crucially, it's *bigger* than TinyCNN yet far worse, so it isn't capacity; and training and test error
rise *together*, so it isn't overfitting. Flattening a 2-D heatmap into a sequence destroys spatial
locality — a genuine architectural mismatch. The lesson: the image framing is what makes everything
else work."

### Slide 20 — All five, side by side  *(~45 s)*
"Here they are together. On the accuracy-versus-size plot, lower-left is better — and the U-Net sits
right in the sweet spot. The table is the whole story along the size axis: MobileNetV2 strips capacity
losslessly; TinyCNN strips too much; the U-Net adds the *right* structure and wins; Mamba changes the
framing and breaks."

---

## Experiments & Results block (21–26) — the evidence  *(Presenter 3; brisk)*

### Slide 21 — In-distribution accuracy  *(~40 s)*
"Here's the evidence behind the headline. On the held-out session: MobileNetV2 matches DLoc *exactly*
at 0.707 meters — lossless. TinyCNN at 47K is five times worse. And the U-Net beats the teacher at
0.640 with the tightest tail — P99 of 2.59 meters. The CDF on the right is from the *real* per-sample
errors, not summary statistics."

### Slide 22 — Learning-rate scheduler study  *(~20 s — FAST / optional)*
"Quick one: cosine versus step scheduling. Both hit the exact same 0.640 median — same optimum —
cosine just trims the tail a little. The scheduler is a second-order knob; architecture is what moves
the needle." *(Skip or compress if short on time.)*

### Slide 23 — Robustness under signal degradation  *(~45 s)*
"Robustness — we corrupt the input at test time, no retraining. Under noise, plain MobileNetV2
collapses to 16 meters but distillation pulls it back under 5; the U-Net's skips hold it at 3 on their
own. On AP dropout, it's distillation that helps the U-Net. Two different mechanisms — skips for
noise, distillation for dropout."

### Slide 24 — Cross-session generalization (LOSO)  *(~45 s)*
"The fair generalization result: leave-one-session-out, standalone-only, three seeds — exactly DLoc's
published protocol. We average 0.93 against DLoc's 0.86, *match* on different-furniture, and beat
DLoc's P90 on the hardest reflector fold — at 6.8 times fewer parameters."

### Slide 25 — Multi-seed statistical stability  *(~20 s — FAST / optional)*
"Is it luck? Across three seeds the baseline median varies only plus-or-minus 0.036 — a four-percent
coefficient of variation. The U-Net's 0.640 sits six-and-a-half sigma below that band, so the gain is
real." *(Skip or compress if short on time.)*

### Slide 26 — Computational efficiency  *(~35 s)*
"And the cost. U-Net versus DLoc: 56 times fewer FLOPs and ten times less CPU latency — 27
milliseconds — at *better* accuracy. That 27 milliseconds is exactly what makes edge deployment
realistic, and it sets up the future work."

---

### Slide 21 — What we found  *(~55 s)*
"Five findings. One: DLoc is over-parameterized — 7.2-times compression is lossless. Two: architecture
beats parameters — the U-Net's skips beat the teacher at 6.8 times smaller. Three: distillation buys
robustness, not accuracy, and only above a 2-million-parameter threshold. Four: the two robustness
mechanisms are complementary. Five: competitive cross-session at 56 times lower compute. The bottom
line — **DLoc's cost is not inherent**; a model an order of magnitude smaller can meet or beat it,
ready for the edge."

### Slide 22 — Where this goes next  *(~35 s)*
"Future work: the immediate step is actually deploying the U-Net on edge hardware at E-JUST — 27
milliseconds on CPU makes that realistic. Then collecting our own CSI dataset in a different building
to test transfer, multi-seeding the students for significance, training-time augmentation, and
exploring attention on the skips."

### Slide 23 — Thank you / Questions  *(~15 s)*
"That's our work — a 2.4-million-parameter model that matches DLoc at 6.8 times smaller and 56 times
cheaper. Thank you — we're happy to take your questions."

---

## Timing summary

| Part | Slides | Target |
|---|---|---|
| 1 — Overview, DLoc & method | 1–10 | 5.5–6 min |
| 2 — The four model architectures | 11–15 | 3.5–4 min |
| 3 — Data & experimental setup | 16–19 | 2.5–3 min |
| Comparison + Results block (evidence) | 20–26 | 3–4 min |
| Close — conclusion, future, thanks | 27–29 | 1.5 min |
| **Talk total** | | **~18–20 min** |
| Q&A (use backups) | 30–32 | remainder |

If you're running long, the safe cuts are: **slides 22 and 25** (scheduler + multi-seed — skip
entirely, they're optional), then slide 18 (summarize sessions verbally) and slide 28 (future work to
two bullets). Never cut slides 9, 10, 15, 27 — they carry the argument.
