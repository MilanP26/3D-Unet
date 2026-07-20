# Orientation for an AI agent picking this up

Read this first. Full rationale lives in [PLAN.md](PLAN.md); usage instructions live in
[README.md](README.md). This file is the "what's actually done, what to do next" summary —
if it ever disagrees with PLAN.md on a *design* question, PLAN.md wins; update this file to
match rather than the reverse.

## One-paragraph summary

Two model architectures now exist for the same underlying problem (given an EM volume and a
neuron's identity, produce that neuron's mask), because the simpler one has a real, measured
flaw:
- **Seeded per-instance model** (`model.py`/`dataset.py`/`train.py`/`infer.py`): given
  `[raw_EM, seed_heatmap]`, predicts one neuron's binary mask directly. Fully built, trained to
  convergence on real data. Flaw: each neuron is predicted independently, so touching/nearby
  neurons' masks can physically overlap — measured directly (PLAN.md §9) at 2-9% mutual overlap
  on real Phase B data, and confirmed visually to sometimes fully swallow one neuron into
  another's identity.
- **Affinity+LSD model** (`affinity_*.py`, added 2026-07-16): given raw EM only (no seed input),
  predicts dense per-voxel affinities (+ auxiliary LSDs, Sheridan et al. 2023), then identity
  comes from **seeded mutex watershed** (`mwatershed` package) at inference time using real
  seed points as hard markers — producing one hard partition where overlap between different
  neurons is structurally impossible, not just unlikely. This is the currently-recommended
  path forward; see "Affinity+LSD model" section below for full results.

Both share the same `Training Data/` (Phase A) and full-hard-drive-stack (Phase B) data
pipelines.

## Status checklist — seeded per-instance model

- [x] Data pipeline (`src/seeded_unet/stack_io.py`): discovers `Training Data/<Stack>/`
      folders automatically, decodes raw EM (PNG **or TIFF**, added 2026-07-16 for
      `Elle_Stack1`) + WKW annotation masks, caches to disk (invalidates correctly on content
      change, not just filename/count), dedups stacks that share identical raw EM, isolates
      per-stack failures so one bad folder doesn't block the rest. `EXCLUDE_FROM_TRAINING`
      marker file drops a stack from training persistently (alternative to `--exclude-stacks`).
- [x] Synthetic seed sampling + Gaussian heatmap channel (`seeds.py`) — training uses points
      sampled from the interior of ground-truth masks, not real VAST coordinates.
- [x] Anisotropic 3D U-Net (`model.py`), Dice+BCE loss + auxiliary LSD loss (`losses.py`,
      `lsd.py` — see "Local shape descriptors" below).
- [x] Training CLI with progress bars, live ETA, checkpointing, CSV logging, `--resume-from`
      to continue training instead of restarting (`train.py`, `scripts/train.py`).
- [x] Real training runs completed on the lab GPU machine (RTX 3090; device auto-detects, no
      code changes needed between the CPU laptop and the GPU machine):
    - 2026-07-08: 30 epochs, `Helena_Stack1` excluded (placeholder annotation), best val dice
      0.583, not converged.
    - 2026-07-15 (v2): 60 epochs, `Juliet_Stack3` added (a real second annotation of new raw
      EM, tightened seed jitter — see `dataset.py`), LSD auxiliary task added. This is the most
      recent seeded-model checkpoint; it predates `Elle_Stack1`.
- [x] `scripts/inspect_data.py` — decodes every stack and cross-checks against webKnossos's
      own recorded segment anchor points as a correctness check.
- [x] **Touching-neuron leakage quantified** (2026-07-15/16, PLAN.md §9): both on Phase A
      ground truth (a verified-adjacent trio in `Catherine_Stack1`, instances 71/73/74) and on
      real Phase B data (real trees 89/100/134, mutual-overlap proxy since no Phase B ground
      truth exists) — 2-9% of predicted mask overlaps a different real neuron. Confidence-based
      arbitration (keep whichever neuron's raw probability was higher at each contested voxel)
      was tried as a stopgap: removes overlap by construction but doesn't fix *why* the two
      predictions disagreed, and can favor the wrong neuron. This result is what motivated the
      affinity+LSD model below.

## Status checklist — affinity+LSD model (added 2026-07-16)

- [x] `affinity_targets.py`: dense per-voxel affinity targets (short-range adjacent-voxel +
      long-range offsets, anisotropic — see `DEFAULT_OFFSETS`) and a generalized dense
      multi-instance LSD target (`compute_lsd_target_dense`, each voxel described w.r.t. its
      *own* instance, matching Sheridan et al.'s actual per-segment formulation, vs. the
      seeded model's single-fixed-instance version in `lsd.py`).
- [x] `affinity_model.py`: same U-Net trunk as `model.py` (literally imports its
      `DoubleConv`/`Down`/`Up`), 1-channel input (no seed heatmap — this model is never told
      which neuron it's looking at), affinity head instead of a mask head.
- [x] `affinity_dataset.py`: dense random-crop patches (not seed-centered) with dense
      multi-instance targets, scaled by stack count not instance count (see "epoch" note in
      training runs below).
- [x] `affinity_train.py` / `scripts/affinity_train.py`: training CLI mirroring `train.py`.
- [x] `affinity_infer.py`: runs the model, then **seeded mutex watershed**
      (`mwatershed.agglom` — confirmed installs cleanly on Windows, calling convention verified
      against a synthetic two-blob test before trusting it: signed affinities in [-1,1]
      where positive=merge/negative=split, a `seeds` array whose nonzero voxels are guaranteed
      to keep their given id in the output). Real per-neuron nodes (however many are available
      — one or hundreds) are passed directly as seeds; multi-seeding (several real nodes per
      neuron) was confirmed necessary — a single seed point per neuron left large fractions of
      that neuron's true extent stranded as unmerged "orphan" fragments never connected back to
      any seed.
- [x] **Training run comparison, same touching trio (Catherine_Stack1 71/73/74/72/70)**:
    - Prototype (2026-07-16, 20 epochs, ~9x less training exposure than the seeded model,
      and — a real bug — accidentally included `Helena_Stack1`'s placeholder annotation):
      instances 73 and 74 almost entirely absorbed into instance 72's identity (97.8%/99.0%
      leaked).
    - Full-match (2026-07-16/17, 45 epochs, 4 train stacks x 400 samples/epoch ≈ 72k total
      samples matching the seeded model's total exposure, `Helena_Stack1` correctly excluded
      this time, `Elle_Stack1` included): **0.0% leaked into a wrong neuron for all 5
      instances.** Best checkpoint `outputs_affinity_full/checkpoints/best.pt` (epoch 29, val
      dice 0.983). Confirms the absorption problem was an undertrained-model issue, not a
      ceiling on the approach. Report:
      https://claude.ai/code/artifact/de204fb8-6448-4dbb-b619-a56a1e2d04c5
- [x] **Real Phase B validation** (2026-07-17, real full stack + real corrected VAST skeleton
      nodes, not synthetic Training Data seeds): searching all 208 real trees for sustained
      cross-tree proximity found trees 50 & 77 within 300nm of each other for 132 consecutive
      z-slices; their single closest same-z approach (113nm, z=493) turned out to sit inside a
      dense pocket of **14 distinct real neurons within ~1 micron**. Seeded 5 of them (50, 77,
      89, 87, 108) with their real nodes — all five held correct, stable, non-overlapping
      identities across all 12 slices shown. One dense cluster, not yet a survey across many;
      treat as a strong example, not a general result yet.
- [x] **Full single-neuron segmentation + review workflow** (2026-07-17,
      `affinity_phase_b_infer.py` / `scripts/affinity_phase_b_infer.py`): mirrors
      `phase_b_infer.py`'s walk-the-subsampled-trace structure, but each patch is seeded with
      *every* real tree that has a node inside it (not just the target tree — the multi-seed
      practice above, applied per-patch), and identity comes from seeded mutex watershed
      instead of a seed-conditioned model. Ran end-to-end on tree 1 (680 real nodes, 176
      subsampled seeds at 500nm spacing, ~26 micron path) in ~17.5 minutes. Output:
      `outputs/phase_b_affinity/tree_1/predictions.npz` (per-patch packed masks + placement)
      and a **176-page scrollable PDF** (`tree_1_review.pdf` — one page per seed, tightly
      cropped around the predicted mask, real node marked) as a practical stopgap for "how do I
      look at Phase B output" that sidesteps the still-unresolved VAST-import-format question
      (see step 3 below). Spot-checked first/middle/last pages: clean, plausible, membrane-
      tracking masks throughout.

## Local shape descriptors (LSD), added 2026-07-13

Auxiliary training task (Sheridan et al. 2023) predicting local shape statistics (size,
center-of-mass offset, coordinate covariance) alongside the main target, forcing the network to
use broader context instead of just local per-voxel evidence — motivated by real annotation
gaps (dust obscuring a neuron in one section during VAST tracing, where local evidence alone
can't tell the neuron obviously continues through the gap). Used by both the seeded model
(`lsd.py`, one fixed instance per patch) and the affinity model (`affinity_targets.py`'s dense
multi-instance version, closer to the paper's actual per-segment formulation, and closer to
LSD's real role: normally paired with affinities, used for both an auxiliary loss and a
merge-scoring feature during agglomeration — the affinity model is where LSD is finally used the
way the paper intends, not just as an auxiliary loss on a mask model). Performance-optimized via
block-average downsampling in x/y + nearest-neighbor upsampling (~25-30x speedup over full-res).

## Critical fact that changed the Phase B design (2026-07-08)

VAST skeletons are **not** one seed point per neuron. Each real neuron has its own skeleton made
of *many* nodes (often hundreds) — one placed on roughly every serial slice while manually
tracing that neuron through the stack, with more than one node in a slice where the neuron
branches or is too large for one click. Design implication (detailed in PLAN.md §3/§10): group
nodes by skeleton/neuron ID, subsample each neuron's trace to a spaced-out set of seeds (a
patch already spans many consecutive slices, so running inference at literally every node would
be redundant), run inference at each sampled seed. For the seeded model this meant a
same-neuron union/merge step (not built). For the affinity model, real nodes are used directly
as seeded-watershed markers instead — no separate merge step needed, since watershed IS the
merge step, and the "many nodes per neuron" redundancy becomes a free confidence/consistency
signal (do all of a neuron's nodes end up in the same connected component?) rather than
something to work around.

## Building Phase B: what to find out first, then what to build

Steps 1 and 2 below are resolved (2026-07-09) — see PLAN.md §13 for the details. Step 3 (output
format) is still genuinely open, but no longer blocking — a practical PDF-review workaround
exists (see "Full single-neuron segmentation + review workflow" above).

1. ~~Look at the actual `.vsanno` file~~ **Resolved via a different route**: the raw binary
   `.vsanno` (magic `VSA0`) turned out to be undocumented and risky to reverse-engineer blind.
   Instead, VAST can export skeleton data to CSV directly, checked in at
   `Data/VAST_skeleton_data.csv`. Column semantics verified against the data itself (see
   `vast_skeleton.py`'s docstring): tree id, per-tree local node id (contiguous 0..N-1), x/y/z
   in the full stack's mip0 voxel frame, parent local id (-1 = root, always resolves to another
   node in the same tree), a branch-child local id, and a free-text annotator tag (e.g.
   `risky_merge`, `potential_merge` — worth surfacing later as a confidence signal, not yet
   used). **Updated 2026-07-17**: the researcher corrected a few tracing errors and merged
   subtrees in VAST, re-exported the CSV (pulled via `git pull` — the file had already been
   pushed to GitHub before being mentioned in conversation, worth checking `git fetch`/`git log`
   before assuming a described data update isn't present yet). Now **208 distinct trees**
   (down from 269), 213,283 rows (up slightly from 209,892 — subtree merging doesn't reduce
   node count, just reassigns some nodes' tree ids).
2. ~~Figure out the full EM stack's file format~~ **Resolved**: VAST's own tiled
   multi-resolution-pyramid format (`volume.vsvi` config + `mip0/<section>/*_tr<r>-tc<c>.png`
   tiles, self-documenting). `phase_b_stack.py` reads windowed regions without loading the full
   102400x36864x1060 volume into memory; tile-stitching is verified byte-exact.
3. **Still open: what format VAST accepts to import a segmentation/mask volume back in.**
   Don't guess at this. Not currently blocking: the PDF-review workflow (see status checklist
   above) gives a usable way to inspect Phase B output without solving this first.
4. Built for both models (see status checklists above). Remaining for the affinity model: this
   has only been run end-to-end on one neuron (tree 1) and one dense cluster (5 of 14 nearby
   real neurons) — not yet run at full-worm scale across all 208 trees, and no attempt yet at
   solving item 3.
5. Coordinate alignment: **not actually needed** — real seeds come with absolute mip0
   coordinates already in the same frame `phase_b_stack.py` reads.

## Things already resolved — don't re-litigate these

- Discovery of new `Training Data/<Stack>/` folders is fully automatic; no code changes needed
  to add a stack, whether it's PNG or TIFF raw EM.
- The decode cache is content-hashed (not filename/count-based), so replacing a stack's
  annotation zip or raw slices (same count) is picked up automatically.
- PNG/mask voxel alignment within a `Training Data/` stack is verified (not just assumed) via
  matching webKnossos's own recorded segment anchor points to decoded label values.
- Training does not need VAST coordinates or hard-drive access at all — only Phase B does.
- A single seed point per neuron is fragile for the affinity model's seeded watershed (leaves
  large orphaned fragments); several real seed points per neuron fixes this, and real VAST data
  already provides many nodes per neuron, so this isn't a practical limitation for real usage.
- Mutex watershed (`mwatershed` PyPI package) installs cleanly on Windows with a prebuilt wheel
  — no build toolchain needed, unlike `waterz` which was not attempted given this worked.

## Known unresolved questions (see PLAN.md "Open questions" for the full list)

Segment-validity rule (which labeled IDs are trustworthy), per-stack annotation-completeness
extent, and whether stacks are from the same or different worm individuals are all still open
and affect training quality/splitting, independent of Phase B. VAST's importable segmentation
format (Phase B step 3 above) is also still open.
