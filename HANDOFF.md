# Project handoff — Pristionchus neuron segmentation (affinity + seeded watershed)

**Read this first.** It is the current state-of-truth as of 2026-07-23, written to hand the
project to a fresh session. `CLAUDE.md` and `PLAN.md` have older background; where they
disagree with this file, **this file wins** for anything from mid-July 2026 onward.

---

## 1. What the project is

Segment individual neurons in a serial-section EM volume of a *Pristionchus pacificus* worm,
using the complete manual VAST skeletons (one traced skeleton per neuron, hundreds of nodes
each) as seeds. Two phases:

- **Phase A — training** on small hand-annotated crops in `Training Data/<Stack>/`.
- **Phase B — inference** on the full worm: a huge tiled EM stack on an external drive
  (`F:\ppa_b4v5s13\aligned_stack\volume.vsvi`, 102400×36864×1060 voxels, scale 2/2/30 nm x/y/z),
  seeded by the real VAST skeleton nodes (`Data/VAST_skeleton_data.csv`).

There are **two model architectures** in the repo. **The affinity model is the current one**;
the older seeded per-instance model is superseded (kept for reference).

---

## 2. THE CURRENT PIPELINE (affinity + LSD + seeded mutex watershed)

Files under `src/seeded_unet/`:
- `affinity_model.py` — `AffinityLSDUNet3D`. Same anisotropic 3D U-Net trunk as `model.py`,
  input = raw EM only (1 channel, **no seed heatmap**), outputs dense per-voxel **affinities**
  (6 offsets, `affinity_targets.DEFAULT_OFFSETS`) + auxiliary **LSD** channels.
- `affinity_targets.py` — affinity + dense multi-instance LSD *targets* from label volumes.
- `affinity_dataset.py` — `DenseAffinityPatchDataset`: random dense patches (not seed-centered),
  with **augmentation** (rotation/flip/translation/artifact/misalignment — `augmentations.py`,
  per Sheridan et al. 2023).
- `affinity_train.py` / `scripts/affinity_train.py` — training CLI (progress, checkpoint,
  CSV log, `--resume-from`). Mirrors `train.py`.
- `affinity_infer.py` — `run_affinity_inference()` (GPU forward pass → affinity probs).
- `watershed.py` — `seeded_agglomerate()`: **seeded mutex watershed** via the `mwatershed`
  package. Signed affinities in [-1,1] (`2*p-1`), positive=merge/negative=split; real skeleton
  nodes passed as `seeds` keep their (global tree-id) label. **This IS the agglomeration step**
  — mutex watershed merges/splits in one pass. There is no separate supervoxel+agglomerate
  stage (that was tried in `experiments/` and dropped — see §5).
- `full_stack_export.py` — `process_region()`: tiles a region in XY, runs inference + seeded
  watershed per tile with overlap ("halo"), composites each tile's trusted core, writes
  VAST-importable PNGs (`write_region_pngs`). Has `max_tile_xy`/`overlap`/`keep_orphans`/
  `progress_label` params.
- `slice_regions.py` — finds where neurons are per z-slice from the skeleton nodes (clusters of
  nodes → padded polygons), so inference is scoped to real neuron locations, not the whole slice.
- `affinity_phase_b_infer.py` / `scripts/affinity_phase_b_infer.py` — walk one tree's
  subsampled skeleton, run per-patch, save per-patch masks to `predictions.npz` (older
  per-patch route; the region/tiled route in `full_stack_export.py` is preferred now).

**How inference works end to end:** pick a region (bbox from `slice_regions` node clumps) →
`process_region` tiles it (1024px tiles, 256 overlap) → each tile: read EM from `F:` →
`run_affinity_inference` → seed with every real tree that has a node in the tile →
`seeded_agglomerate` → keep trusted core → composite → `write_region_pngs` (16-bit PNG,
pixel value = global tree id, 0 = background).

---

## 3. CURRENT MODEL — v3 (use this)

- **Checkpoint: `outputs_affinity_v3/checkpoints/best.pt` (epoch 18, val dice 0.9849).**
  `last.pt` is epoch 27. Trained with Helena included, 500 samples/stack, augmentation on.
- Stopped at epoch 27 of a planned 30 because it had **converged/plateaued** (val dice bounces
  in a ~1% band after ~epoch 6; best was epoch 18). Epochs 28–30 skipped deliberately.
- Beats v2 (`outputs_affinity_v2/checkpoints/best.pt`, epoch 29, 0.9837) — v3 has consistently
  lower val loss (clearer signal than the tiny dice gain). Helena helped modestly.
- Older: `outputs_affinity_full/` (pre-augmentation full-match run), `outputs/checkpoints/`
  (the old *seeded per-instance* model, superseded).
- Train/val split: 5 train stacks (Catherine, Elle, Helena, Juliet2, Juliet3) / 1 val (Juliet1).
  Note the single val stack under-measures generalization; real quality judged by eyeballing
  inference on fresh regions.

To re-read training curves cleanly, parse `outputs_affinity_v3/training_log.csv` (NOT the
`.log` — see §7 tqdm gotcha).

---

## 4. WHAT WORKS AND WHAT DOESN'T (the key findings — read this)

**Works well: the nerve ring** (densely packed neurites, z≈500). Affinity diagnostics
(`outputs/affinity_seed_diagnostic_nervering*.png`) show the model detects the membrane lattice
sharply and continuously there, it's densely seeded, and seeded watershed cleanly separates
~28 neurons per slice with **0 cross-neuron overlap** (structural). See
`outputs/nervering_segmentation_v3.png`. This is the connectome-critical region and it's in
good shape.

**The remaining weakness: "bleed" in cell-body / sparsely-seeded regions.** Where a neuron has
no traced neighbor competing in a patch AND the local membranes are faint, the affinities are
saturated high (a cell-body diagnostic showed 81% of affinities >0.9) and the lone seed floods
across undetected membranes into neighbors. This is an **affinity-quality / under-detection-of-
membranes** problem, i.e. model-level. It is NOT over-segmentation and NOT fixable by
agglomeration or by watershed parameters.

**Tile size matters a lot (inference-side fix, no retrain):** 512px tiles produced boxy,
seam-cut masks (a neuron spanning tiles got cut at the core boundary and flooded its tile). **1024px
tiles fixed this** — large cell bodies fill their true rounded shape. Confirmed by direct 512-vs-1024
comparison. **Use `max_tile_xy=1024, overlap=256`.**

---

## 5. DEAD ENDS — do not revisit these (already tried, don't help)

- **Watershed merge/split bias tuning** (subtracting a bias from signed affinities): does NOT
  stop bleed. A lone seed with saturated-high affinities and no detected boundary still floods
  (98% → 97% at bias 0.4). Proven negative.
- **`skeleton_clean`** (`experiments/skeleton_priors/consistency.py`): drops mask components
  with no traced node. Only removes *disconnected* leak blobs; the actual bleed is *connected*
  to the neuron, so it survives. Metrics looked great (orphan voxels → 0) but the visible
  problem is untouched.
- **Distance bound** (trim mask voxels far from the neuron's own skeleton): geometrically cuts
  bleed and keeps 100% node coverage, but it's a blunt "tube around the skeleton" heuristic that
  doesn't follow membranes, and applied to old boxy per-patch output it looks blobby. A band-aid,
  not a fix.
- **Constrained supervoxel agglomeration** (the classic NN→fragments→agglomerate route): tried
  in `experiments/` and dropped — affinities are high nearly everywhere, so fragments came out
  huge and distinct neurons collapsed into one blob.

**The real fix for bleed is affinity quality (model-level):** more/diverse data (Helena, done in
v3) and, if pursued, a **boundary-weighted loss** (up-weight the thin membrane voxels so the
model stops under-detecting faint membranes). Not yet tried — this is the recommended v4 lever.

---

## 6. DATA STATE

- `Training Data/` stacks: **Catherine_Stack1, Elle_Stack1 (TIFF, not PNG — `stack_io.py`
  handles both), Helena_Stack1 (2048×2048; its annotation was a placeholder earlier but is now
  REAL data — no longer excluded), Juliet_Stack1/2/3.** No `EXCLUDE_FROM_TRAINING` markers
  currently active. Do NOT pass `--exclude-stacks Helena_Stack1` anymore.
- `Data/VAST_skeleton_data.csv`: **208 trees** (the researcher merged subtrees + fixed tracing
  errors; pulled via `git pull` on 2026-07-17 — check `git fetch`/`git log` if a data update is
  mentioned, it may already be on the remote). 213,283 nodes. Columns verified in
  `vast_skeleton.py` docstring.
- Full EM stack: external drive `F:\ppa_b4v5s13\aligned_stack\volume.vsvi` (must be attached for
  any Phase B work). VAST tiled format, read windowed by `phase_b_stack.py`.

---

## 7. ENVIRONMENT GOTCHAS (will bite you)

- **Use `py` (the launcher), NOT `python`** — `python` hits a Windows Store shim and fails.
- **Never run watershed via a subprocess / ProcessPoolExecutor.** Windows `spawn` pickling the
  large affinity array hangs indefinitely — this masqueraded as "watershed is mysteriously slow"
  for hours. `watershed.py` is deliberately torch-free, but the lesson is: **call
  `seeded_agglomerate` directly in-process.** It's fast (~4s for a 12×512×512 seeded tile).
  `process_region` now does this (the parallel/timeout machinery was removed).
- **Training logs are tqdm-polluted** (carriage returns make one giant "line"). For per-epoch
  numbers, read `outputs_affinity_v3/training_log.csv`, not the `.log`.
- **GPU memory:** 24 GB card. 1024×1024×25 seeded watershed peaks ~23 GB (just fits). A
  1024×1024×**50** tile OOMs → for >~25 z-slices, **process z in ≤25-slice chunks** and stitch
  (safe: seeds are global tree ids, so a neuron spanning the z-boundary keeps its id/color — no
  reconciliation needed).
- Watershed (`mwatershed`, PyPI) installs cleanly on Windows; it's CPU-only and single-threaded.
  Per-tile watershed is the dominant cost (~30–70s for a 1024²×25 seeded tile), not GPU inference
  (~1s).

---

## 8. VAST IMPORT (how to view results) — resolved from the VAST Lite 1.5.0 manual §4.7

Import via **File / Import / Import Segmentation from Images**. Fields:
- **Basic filename string**: a C-printf template, e.g. `region1_z%05d.png` (VAST fills the Z number).
- **No of first/last slice (Z)**: the actual absolute Z range of the files.
- **No of first/last column & row (X/Y)**: `1`/`1` (one image per section, no sub-tiling).
- **Start coordinates X/Y/Z**: where the tile's corner sits in the full volume (per-region!).
- **Tile size X/Y**: the PNG width/height.
- **Import at mip level**: 0 (full res required — no mip flexibility for this import path).
- **Force 8-bit Graylevel**: OFF (our PNGs are genuine 16-bit; forcing 8-bit corrupts labels).
- Rotate/Flip: all off.

Format: **16-bit grayscale PNG, pixel value = segment id (we use the global VAST tree id), 0 =
background.** Each region is a SEPARATE import pass (start coords differ per region). The Z
range must be contiguous — fill any interior gap slices with all-zero PNGs. To view multiple
regions together in one layer, either import each into the same layer (they share global tree
ids, so a neuron keeps one color across regions) or composite them onto one shared canvas first
(see the `combined_*` scripts). `write_region_pngs` emits a `manifest.json` with the exact
per-file placement.

---

## 9. WHERE THE OUTPUTS ARE

- `outputs/phase_b_z450_500/` — earlier nerve-ring export, z450-500, **512 tiles** (3 regions;
  `combined_r0r1/`, `combined_all3/` are single-import composites). ~229 neurons. Has the boxy
  512-tile artifacts.
- `outputs/phase_b_bigtile_test/` — region 1, z475-499, **1024 tiles** (the tile-size fix demo).
- `outputs/phase_b_region1_v3_1024/` — **DONE (current clean deliverable):** v3, region 1,
  z450-499, 1024 tiles, no post-processing, real neurons only. 50 PNGs (`region1_z%05d.png`),
  ~121 neurons/slice. VAST import: first/last slice Z = 450/499, Start coords X=41871 Y=17481
  Z=450, tile 8794x3717, mip 0, Force-8-bit OFF. Pixel value = global tree id.
- `outputs/phase_b_affinity/tree_1/predictions.npz` — old per-patch tree-1 run (pre-tile-fix;
  the `experiments/` oracle consumes this).
- Diagnostics: `outputs/affinity_seed_diagnostic_nervering_wide.png`,
  `outputs/nervering_segmentation_v3.png`, `outputs/v2_vs_v3_curves.png`.
- Scratch/experiment scripts live in the session scratchpad (temp dir), not the repo. The
  reusable pipeline is all under `src/seeded_unet/` and `experiments/`.

---

## 10. SUGGESTED NEXT STEPS

1. Once `phase_b_region1_v3_1024/` finishes, import into VAST and judge v3 quality on the nerve
   ring at 1024 tiles.
2. If bleed in cell-body/sparse regions still matters, try a **boundary-weighted affinity loss**
   (v4) — the one model-level lever not yet tried. Don't re-try the §5 dead ends.
3. Scale the region/tiled export (`full_stack_export.process_region`, 1024 tiles, z-chunked ≤25)
   to more regions / more of the stack. It's the working path.
4. Open question still: is a fully automatic (unseeded) segmentation wanted anywhere, or is
   everything seeded by the 208 real skeletons? So far everything is seeded.
