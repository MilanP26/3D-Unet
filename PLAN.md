# Seeded 3D U-Net for Pristionchus Neuron Segmentation — Development Plan

Status: **implemented and verified against real data (2026-07-07); paused before a real
training run, which will happen on a lab GPU machine.** See section 13 for what was built,
what was measured, and what to do on the GPU machine. Sections 0-12 below are the original
design plan and are kept as-is since they're still the rationale behind the implementation.

## 0. What inspection actually found (ground truth as of 2026-07-07)

### 0.1 Repo layout, current state

```
3D-Unet/
├── README.md
├── PLAN.md
└── Training Data/
    └── Juliet_Stack2/
        ├── README.md                                    # currently empty
        ├── Juliet_stack2.zip                             # webKnossos annotation export
        └── ppa_b4v5s13_head_volume_export_s0573.png ... s0632.png   # 60 raw EM slices
```

The intended convention going forward, per the researcher: `Training Data/<StackName>/` holds
one self-contained annotated training example — a sequence of raw EM slice PNGs plus the
webKnossos annotation zip for that same crop. More folders (`Juliet_Stack1`, `Juliet_Stack3`,
others) will be added the same way. This is a deliberate, useful split:

- **Phase A — Training (lives in this repo).** Small, cropped, fully self-contained
  (image + mask) examples, each easily a few dozen–hundred MB. This is what the model trains on.
- **Phase B — Skeletonizing / inference on the full worm (lives on the external hard drive).**
  The full EM stack is far too large for GitHub and will never be checked in here; VAST and its
  `.vsanno` skeleton files are how seed points get placed across the *whole* worm once a trained
  model exists. This only needs to happen where the hard drive is attached.

This distinction changes the plan meaningfully from the first pass: **training no longer
depends on resolving VAST↔webKnossos coordinate alignment at all** (see §3) — that alignment
problem only matters for Phase B, later. Details below.

### 0.2 `Juliet_stack2.zip` (the annotation)

Unchanged from the first inspection pass:
- `Juliet_stack2.nml` is a webKnossos *volume* annotation (colored masks), not a skeleton file.
  `<scale x="2.0" y="2.0" z="30.0" unit="nanometer"/>` — voxels are highly anisotropic (2 nm
  in-plane, 30 nm in z). `<offset x="0" y="0" z="0"/>`.
- `<volume id="0" name="Volume" location="data_Volume.zip" format="wkw" largestSegmentId="70">`,
  with a `<segments>` list of 70 IDs, only ~15 of which have an explicit name/color — the rest
  are unnamed/default, so not all 70 IDs are necessarily finished, trustworthy neuron labels.
- `data_Volume.zip` decodes to a WKW dataset: mag-1 only, sharded into `z0`/`z1` folders of
  32×32 buckets (32³ voxels each) → bounding box up to ~1024×1024×2048 voxels, but most buckets
  are near-empty (~555 bytes compressed) with only a handful around 4.3 KB of real labeled
  content — the actual painted region is a small, sparse sub-volume of that grid.
- No `datasource-properties.json`, no raw/color layer referenced in the `.nml` — confirmed
  again: **the annotation zip by itself still contains mask-only, no image data.**

### 0.3 New finding: `Training Data/Juliet_Stack2/*.png` (the raw EM)

60 PNGs, named `ppa_b4v5s13_head_volume_export_s0573.png` through `..._s0632.png`
(573→632 inclusive = 60 slices). Checked the PNG header directly (IHDR chunk):
**1024 × 1024 pixels, 8-bit, grayscale.**

This is a strong, checkable alignment signal, not a coincidence:
- 1024×1024 exactly matches the WKW bucket grid width computed above (32 buckets × 32 voxels).
- 60 slices matches the z-extent implied by the segment anchor positions in the `.nml`
  (z values observed in the 0–59 range).
- `<offset x="0" y="0" z="0"/>` plus slice `s0573` being the first file is consistent with local
  z=0 in the annotation corresponding to PNG `s0573`, i.e. **global slice index = local z + 573**.

**This is strongly suggestive that the PNG stack and the WKW mask are the same crop, pixel-for-
pixel, at offset (0,0)** — which would mean the raw-EM-missing problem from the first plan pass
is solved for training purposes: this folder alone is enough to build (image, mask) training
pairs without needing webKnossos API/raw-layer access at all. **This still needs an explicit
verification step once decoding is implemented** (load the WKW label array, overlay a labeled
slice's footprint on the corresponding PNG, and visually/numerically confirm the neuron outlines
actually land on the right EM structures) before being trusted — matching dimensions and offsets
is strong circumstantial evidence, not proof.

The `s0573`-style numbering also looks like a **global** slice index into a larger source stack
(`b4v5s13` reads like a block/volume/stack identifier), which — if confirmed — gives a
ready-made z-anchor for later relating this crop back to the full hard-drive stack, even though
the x/y crop origin within that larger stack is still unknown (not needed for training, but
will matter for Phase B).

### 0.4 Environment

Still no Python interpreter, no `wkw`/`webknossos`/`torch`/`numpy`, no GPU set up in this
workspace. Held as an action item, not done yet, per "don't implement yet."

### 0.5 Updated picture of the two annotation/data systems

| Source | Tool | Location | Used for | Status |
|---|---|---|---|---|
| Cropped raw EM + colored mask pairs | webKnossos export | pushed into `Training Data/<Stack>/` in this repo | **Training (Phase A)** | 1 example (`Juliet_Stack2`) in; 2–3 more coming |
| Full EM stack | VAST | USB hard drive | **Inference / skeletonizing whole worm (Phase B)** | not accessible from this machine |
| Skeleton traces (~206 skeletons, one per neuron, each made of *many* nodes — see §3 update) | VAST, `.vsanno` | USB hard drive | **Phase B seeds**, and optionally later as a *validation* check on Phase A seed realism | not accessible from this machine |

---

## 1. Inspecting and understanding the data formats

For **Phase A (training data, this repo)**, once Python is set up:
1. Decode each `Training Data/<Stack>/*.png` slice sequence into a dense `(Z, Y, X)` uint8
   raw-intensity volume (plain image I/O — no webKnossos API needed since the PNGs are already
   exported raw pixels).
2. Decode the matching `<Stack>.zip` → `.nml` + WKW volume into a dense instance-label array
   using the `wkw` Python package, over the same bounding box as the PNG stack.
3. **Verify alignment** (§0.3) — this is the first real coding task once implementation starts,
   and everything else depends on it being true. If it's *not* aligned, the plan needs to
   revisit how to get a matching raw layer per annotation.
4. Cross-reference the WKW array's actual unique label IDs against the 70 IDs listed in the
   `.nml`; compute a per-ID voxel-count histogram (validates data integrity, informs patch size
   in §4/§6).
5. Repeat 1–4 for each new `Training Data/<Stack>/` folder as it's added, and record per-stack
   facts (scale, slice range, which segment IDs are trustworthy, annotation-completeness extent)
   in a small manifest file per stack (e.g. `Training Data/<Stack>/manifest.yaml`) rather than
   re-deriving by hand each time.

For **Phase B (full-stack inference)**, later, not needed to start training:
6. Parse the VAST `.vsanno` file (on the USB drive) to get `(neuron_id, x, y, z)` skeleton nodes
   in VAST's native voxel space. **Important, corrected understanding (see §3 for the full
   design implication): this is not one node per neuron.** Each of the ~206 neurons has its own
   skeleton made of *many* nodes (often hundreds) — one was placed on (roughly) every serial
   slice while tracing that neuron through the stack, and a slice can have more than one node
   for the same neuron if it splits into multiple profiles there or is large enough to need more
   than one click to represent its cross-section. So parsing needs to preserve the grouping of
   nodes by skeleton/neuron identity, not just flatten everything into a single seed-point list.
7. Establish the mapping from VAST's full-stack coordinate space to whatever coordinate frame
   the trained model's inference script expects (crop origin, scale/mag) — this is where the
   VAST↔webKnossos alignment question that dominated the first plan pass actually matters, and
   it can be deferred until there's a trained model ready to run on the full stack.

## 2. Manual colored annotation → 3D label masks

- Decode the WKW volume layer into a dense instance-label numpy array per stack (background = 0,
  integers = instance IDs), sized to the populated region rather than the full sparse shard grid.
- Filter the `.nml`-listed IDs down to those with non-trivial voxel counts in the decoded array;
  apply a minimum-voxel-count threshold to drop specks/accidental clicks.
- Convert each retained instance ID into its own binary mask (`label == id`) on demand.
- Assign each instance a **globally unique** neuron ID across all stacks (local IDs are only
  unique within one stack) — tie to actual anatomical neuron identity if/when that's known,
  since the eventual scientific goal is per-neuron identity, not just arbitrary instance
  separation.
- Decide and document a convention for which segments count as "finished, trustworthy" labels
  (see Open Questions).

## 3. Seed channel construction

**Training (Phase A) does not need real VAST coordinates at all.** Since ground-truth instance
masks are already available for every training stack, seeds can be **synthetically sampled**
from the interior of each labeled mask — this is the standard approach for training seed/point-
conditioned segmentation models (used in interactive segmentation literature) and sidesteps the
VAST↔webKnossos registration problem entirely for training:
- For each training example, pick one or more interior points of a given instance mask (e.g.
  uniformly random voxel inside the mask, or biased toward the medial/central region away from
  the boundary, to mimic where a person tends to click) and use that as the seed for that patch.
- Vary seed position per epoch (don't always use the same synthetic point for a given instance)
  so the model doesn't overfit to one exact seed location per neuron, and so it learns to be
  robust to seeds that land near an edge, not just dead-center.
- Seed encoding options (same as before): binary point/dot vs. a 3D Gaussian heatmap with sigma
  set in physical nm then converted per-axis for the 2/2/30 nm anisotropy. Recommend starting
  with the Gaussian heatmap; treat as an ablation, not a fixed decision.

**Inference (Phase B) uses real seeds — but "one seed per neuron" is the wrong mental model.**
Each of the ~206 neurons has a skeleton made of many nodes (often hundreds), roughly one per
serial slice along the neuron's length, sometimes several in one slice (branch point, or a
cross-section too large for one click). This is actually a good match for how this model
works, once the inference procedure accounts for it correctly:

- Group `.vsanno` nodes by skeleton/neuron ID first. Each group is effectively a rough 3D
  trace of that neuron through the stack, ordered along z.
- **Don't run inference at every single node.** A training/inference patch already spans many
  consecutive z-slices (e.g. a 32-voxel-deep patch covers ~32 slices at this dataset's 30nm z
  spacing) — running inference at every node on a dense per-slice trace would be enormously
  redundant (adjacent-slice patches overlap almost completely) for ~hundreds of nodes per
  neuron x ~206 neurons. Instead, subsample each neuron's trace to a spaced-out set of seeds
  (roughly every `patch_depth/2` slices along z, so consecutive inference windows still overlap
  enough to stitch cleanly) before running the model.
- Run inference at each subsampled seed, then **union all resulting local masks belonging to
  the same neuron** into one combined mask spanning that neuron's full traced extent — this
  naturally handles branch points and multi-node slices too (just more seeds feeding the same
  union), no special-casing needed.
- Seed realism and coordinate alignment (§1, steps 6-7) still matter as described below, but
  the "one seed per neuron" framing anywhere else in this document (including §10) should be
  read as "one seed per *sampled point along* a neuron's trace."

## 4. Building training samples (patches)

- **Measure, don't guess, patch size**: once §1 decodes real instance masks from the available
  stacks, compute the bounding-box size distribution of actual neuron instances (in voxels and
  physical nm, respecting anisotropy) and use that to drive patch size.
- Center each training patch on a synthetic seed point (§3), sized to comfortably contain the
  full extent of most instances, with margin; jitter seed position per epoch as an augmentation.
- **Only-partially-annotated regions**: don't assume unlabeled voxels are guaranteed background.
  Determine, per stack, what region was *exhaustively* colored vs. merely "colored so far," and
  mask/ignore loss outside the exhaustively-annotated region if that extent isn't the whole crop.
  Needs a direct answer per stack (Open Questions).
- Consider deliberate negative samples (seed just outside a neuron, or in background) later, once
  the core positive-seed task works — not required for v1 given the current task framing.

## 5. Model input / output tensors

- **Input**: 2-channel volume `[raw_EM_patch, seed_channel]`, shape `(2, Dz, Hy, Wx)`, raw
  intensity normalized (per-stack, since EM contrast can vary between stacks/imaging sessions).
- **Output**: 1-channel binary mask, same spatial shape, sigmoid activation.
- Optional (not v1): an auxiliary distance-transform/boundary-map output to help separate
  touching neurons — revisit once that failure mode is actually observed empirically.

## 6. 3D U-Net architecture

- Standard encoder-decoder 3D U-Net with skip connections, **anisotropy-aware**: given 2/2/30 nm
  voxels, pool in-plane before pooling z (e.g. stride `(1,2,2)` early on), and/or use anisotropic
  kernels (mixing `(1,3,3)` and `(3,3,3)`) — standard practice for anisotropic EM connectomics
  data rather than treating the volume as isotropic.
- Prefer GroupNorm/InstanceNorm over BatchNorm given small 3D batch sizes under memory limits.
- Start from an established implementation (e.g. MONAI's `UNet`, or an nnU-Net-style anisotropic
  config) rather than writing the architecture from scratch, given how little labeled data
  currently exists.
- Plan explicitly for: heavy augmentation, a relatively shallow/small network to limit
  overfitting, and considering self-supervised pretraining on unlabeled raw EM (there will be
  plenty of unlabeled raw EM even within just the Phase A PNG stacks, let alone the full hard
  drive) or transfer from an existing permissively-licensed EM segmentation model.

## 7. Loss functions

- Baseline: combined **Dice + BCE**.
- Alternative if class imbalance or small-neuron performance is poor: **focal loss** or **focal
  Tversky loss** (lets FP/FN be weighted differently — relevant since under-segmenting vs.
  leaking into a neighboring neuron have different costs).
- Boundary-aware loss as a later refinement once touching-neuron leakage is observed empirically.

## 8. Train/validation/test split (no leakage)

- Split at the **instance (neuron) level** first — the same neuron instance must never appear
  (even via a different synthetic seed or jittered crop) in more than one split.
- Split at the **stack level** once there are enough stacks (whole `Training Data/<Stack>/`
  folders held out entirely) — the only way to test generalization to new tissue regions/imaging
  sessions, and the natural unit now that data arrives one stack at a time.
- With currently one stack: grouped k-fold / leave-some-neurons-out cross-validation within it
  for a first read on signal, explicitly treated as a **weak, interim** estimate until a second
  independent stack is available (see Open Questions — same worm or different worm matters here).

## 9. Evaluation metrics

- Per-instance volumetric **Dice** and **IoU**.
- Voxelwise and instance-level **precision/recall**.
- **Boundary accuracy**: average symmetric surface distance or boundary-F1.
- **Leakage metric**: fraction of predicted foreground voxels landing inside a *different*
  ground-truth instance than the seeded one — the touching-neuron failure mode this task is
  most exposed to.
- **Seed-sensitivity**: perturb the seed by a few voxels and measure output stability — a proxy
  for robustness to imprecise real clicks at Phase B inference time.

## 10. Inference workflow

Two distinct inference contexts now that Phase A/B are separated:

- **Validation-style inference (within this repo's data)**: given a held-out stack and a
  synthetic or real seed inside it, crop a patch, run the model, paste the thresholded
  prediction back into that stack's coordinate frame. No tiling/stitching problem since this is
  inherently local per seed, not whole-volume dense segmentation.
- **Phase B — full-worm inference (on the hard drive)**: for each neuron's skeleton, subsample
  its many nodes down to a spaced-out set of seeds (§3), map each into the trained model's
  expected frame (§1 step 7), crop a patch from the full EM stack around it, run the model, and
  union all of that neuron's local predictions into one full-length output mask. Running
  multiple seeds per neuron also gives a free consistency check (do independent seeds along the
  same neuron agree in their overlap regions?). This phase runs wherever the hard drive is
  attached (the trained model weights travel there; the raw stack does not travel here). **This
  entire phase is unimplemented as of 2026-07-07** — see §13 for what's built vs. not, and
  [CLAUDE.md](CLAUDE.md) for the concrete next-steps checklist.

## 11. Risks / open issues

- **Alignment assumption needs verification, not just trust**: the PNG-stack/WKW-mask match in
  §0.3 is strong circumstantial evidence (dimensions, offsets, slice count all line up) but has
  not been pixel-verified yet — do this before building on top of it.
- **Ambiguous segment validity**: most of the 70 listed segment IDs in `Juliet_Stack2` are
  unnamed; need a rule for which count as real, trustworthy neuron labels.
- **Partial annotation extent**: unlabeled ≠ guaranteed background unless the exhaustively-
  colored region is explicitly known per stack.
- **Extreme anisotropy** (2/2/30 nm) affects patch shape, kernel choice, and augmentation.
- **Small dataset currently** (effectively one stack) → high overfitting risk; mitigated somewhat
  by synthetic multi-seed sampling per instance, but still a real constraint.
- **Class imbalance** within any given patch (neuron voxels vs. background).
- **Touching/adjacent neurons** — likely the hardest part of this task.
- **Annotation noise**: hand-drawn boundaries have inherent human inconsistency.
- **Memory/compute limits**: no GPU currently set up in this workspace; patch size needs to be
  checked against whatever GPU/VRAM is actually available for training (Open Questions).
- **WKW block-boundary effects**: data stored in compressed 32³ buckets; the decoder must handle
  patches straddling bucket boundaries correctly.
- **Repo size growth**: each stack folder is PNGs (tens of MB) + an annotation zip; fine for now
  at one stack, but with several more coming this could approach GitHub's comfort limits —
  worth watching, and moving to Git LFS if it becomes a problem, but not urgent yet.
- **Phase B coordinate alignment** (VAST full-stack ↔ trained-model frame) is real but now
  deferred — it blocks full-worm inference later, not training now.

## 12. Future human-in-the-loop extension

- **v1**: train on Phase A stacks using synthetic seeds sampled from ground-truth masks, as
  scoped above.
- **v2 (active learning loop)**: run the trained model — either on held-out Phase A regions or,
  once ready, on real VAST seeds in Phase B — to propose masks → a human (in VAST or webKnossos)
  reviews/corrects them → each corrected (seed, mask) pair becomes a new training example (a new
  `Training Data/<Stack>/`-style folder, or an addition to an existing one) → periodic
  retraining/fine-tuning.
- Prioritize which proposals get reviewed first using the uncertainty signals from §9 (low
  confidence, high seed-sensitivity, high predicted leakage-into-neighbor risk) so human
  correction time is spent where it improves the model most.

---

## Open questions to resolve before coding begins

1. **Data convention going forward**: will every future `Training Data/<Stack>/` folder always
   contain both the raw EM PNG sequence *and* the annotation zip (as `Juliet_Stack2` does), so
   training never needs direct webKnossos/API access to raw data? Worth confirming since it
   simplifies the environment/tooling needed considerably.
2. **Segment validity rule**: of the 70 segment IDs in `Juliet_Stack2`, which count as finished,
   trustworthy neuron labels vs. WIP/scaffolding? Same question will apply to each new stack.
3. **Annotation completeness extent, per stack**: what region of each stack was *exhaustively*
   colored (every neuron present painted) vs. partially colored? Determines whether unlabeled
   voxels are safe as negative/background training signal.
4. **Relationship between stacks**: are `Juliet_Stack1/2/3` (and any others) from the same
   individual worm (different regions) or different worms? Affects how train/val/test splitting
   should be grouped to avoid leakage (§8).
5. **Compute**: what GPU/VRAM is available for training? Training itself can happen anywhere
   once `Training Data/` is populated (it no longer needs the hard drive) — but Phase B
   full-worm inference will need to happen on/near the hard drive.
6. **Typical neuron size**: once decoding is implemented I can measure this directly from the
   masks, but a rough existing sense (voxels or nm) would help sanity-check patch size early.
7. **Slice numbering**: is the `s0573`–`s0632` numbering in the PNG filenames a *global* slice
   index into the full hard-drive stack (as it appears), and if so, is the x/y crop origin of
   this 1024×1024 region within that full stack recorded anywhere? Not needed for training, but
   needed later to map Phase A regions and Phase B predictions into the same frame.
8. **VAST `.vsanno` schema**: for when Phase B planning starts in earnest — can you share a
   sample of the actual file content/export so I can confirm its coordinate convention, units,
   and (now known to be essential, not optional) how nodes are grouped by skeleton/neuron
   identity, given each neuron has many nodes rather than one?
9. **Full hard-drive EM stack format**: folder of slice images (like `Training Data/`, just
   much bigger), a single volume file, or a VAST-native format? Determines how Phase B's reader
   needs to work.
10. **VAST's import format for results**: what format does VAST accept to load a segmentation/
    mask volume back in, so Phase B's output is actually usable rather than a guess?

Setting up a local Python environment (`numpy`, `wkw`, `Pillow`/`tifffile`, `torch`) is the
natural next practical step to start answering #2, #3, #6, and to run the alignment-verification
check in §11 — held until you confirm you want to move past planning.

---

## 13. Implementation status, real measurements, and GPU handoff (2026-07-07)

The plan above became code in `src/seeded_unet/` (see [README.md](README.md) for usage). This
section records what actually got verified/measured, superseding the corresponding guesses in
sections 0-12 where they differ.

### What's built

`stack_io.py` (discover/decode/cache stacks + dedup shared raw EM) · `seeds.py` (synthetic
interior-biased seed sampling + anisotropic Gaussian heatmap) · `dataset.py` (instance list,
group-aware train/val split, patch sampling with disk-cached seed distributions) · `model.py`
(anisotropic 3D U-Net, in-plane pooling before z-pooling) · `losses.py` (Dice+BCE, Dice/IoU
metrics) · `train.py` / `infer.py` (CLIs with `tqdm` progress bars and a live per-epoch ETA) ·
`scripts/inspect_data.py` (alignment verification + size stats) ·`scripts/train.py` /
`scripts/infer.py` (entry points).

### Resolved since sections 0-12 were written

- **A third stack, `Catherine_Stack1`, arrived** and follows the same PNG+zip convention (70
  slices, same 1024×1024, same 2/2/30 nm scale) — answers open question #1: yes, the convention
  is holding.
- **PNG/mask alignment is now directly verified, not just circumstantial**: decoding
  `Juliet_Stack2`'s WKW mask and reading the voxel at webKnossos's own recorded anchor point for
  segment 3 (x=778, y=29, z=46) returns label `3` exactly. Across all three stacks,
  `scripts/inspect_data.py`'s anchor-point check passes for 85-97% of segments; the handful of
  mismatches read as normal post-anchor edits (segment merges, or a region getting erased/
  relabeled after the anchor was recorded), not a decoding bug.
- **`Juliet_Stack2` and `Juliet_Stack3` share byte-identical raw EM** (confirmed by hashing the
  PNGs) — two independent annotation passes over the same crop (Stack3 has far fewer segments
  done, 20 vs. 70, i.e. it looks like an earlier/partial pass). `stack_io.group_stacks_by_raw_hash`
  now assigns them the same `scene_group` automatically so a split can never separate them.
- **Instance sizes vary far more than "one compact soma" implies**: some labeled segments span
  nearly the full 1024×1024×Z crop (e.g. up to ~860×724×60 voxels) — almost certainly long
  neurites, not somas. This confirmed the plan's §4 revision was right: a patch only needs to
  show local context around the seed, and ground truth is just the mask intersected with the
  patch window, not the whole instance.
- **Segment validity** (open question #2) still has no researcher-provided rule, so the code
  defaults to a voxel-count threshold (`--min-instance-voxels`, default 500) and otherwise trusts
  every labeled ID equally, named or not. `scripts/inspect_data.py` reports how many segments per
  stack are explicitly named/colored (4-24 of them) so this can be revisited.
- **Compute** (open question #5): resolved — real training will run on a GPU machine in the lab,
  not this laptop (which has no CUDA GPU, only integrated AMD graphics).

### Real CPU timing (this laptop, no GPU) — why full-size training needs the GPU machine

| Config | patch (Z,Y,X) | base_channels | measured cost |
|---|---|---|---|
| toy smoke test | 16×64×64 | 8 | ~159s/epoch (train+val), 260+33 batches |
| timing probe | 24×128×128 | 16 | ~4.7s/batch (train), ~23 min/epoch extrapolated |
| **default** (`train.py` as shipped) | 32×256×256 | 24 | **not run to completion on CPU** — extrapolated ~12x the timing-probe's per-batch cost (more voxels + wider channels), i.e. very roughly **hours per epoch** |

There is also a one-time, patch-size-independent setup cost the first time seed distributions
are computed for a given `--min-instance-voxels`/`--interior-bias`: a distance transform per
instance (~4 minutes for the current 163 instances across 3 stacks). This is now cached to disk
under `.cache/seed_dist/`, so it only pays once ever, not once per run.

**Conclusion: iterate on this laptop with a small patch/model (like the timing-probe config) if
you want to sanity-check changes quickly; do the real training run on the lab GPU machine with
the defaults (or larger).**

### Two more stacks arrived, plus automation/robustness hardening (2026-07-07, later)

`Juliet_Stack1` (real, 53 slices, 25 instances) and `Helena_Stack1` arrived. Helena's PNGs
are 2048×2048 (double Catherine's/Juliet's 1024×1024), and its annotation zip is a deliberate
**placeholder** — internally it's literally a copy of `Catherine_stack1.nml` + Catherine's
`data_Volume.zip`, standing in until the real Helena annotation is sent over. The plan is for
the real one to replace it before the GPU training run.

This was a good real-world test of whether "drop in a new stack, no code changes" actually
holds, and it surfaced two robustness gaps that got fixed:

- **Cache staleness on annotation swap**: the on-disk decode cache only checked PNG count
  before, so replacing Helena's placeholder zip with the real one (same PNG count) would have
  silently kept serving the *old* decoded mask. Fixed: the cache now also hashes the annotation
  zip's actual bytes and invalidates on any change. The nested extraction directories are also
  now wiped and re-extracted from scratch every time a re-decode happens, instead of trusting a
  possibly-stale leftover extraction (this matters concretely here: the placeholder's internal
  `.nml` is named `Catherine_stack1.nml`, and the real Helena annotation will almost certainly
  use a different name — without this fix, the old file would have lingered and made the
  "exactly one .nml" check fail confusingly).
- **No fault isolation**: `load_all_stacks` previously aborted entirely if any single stack
  failed to decode. Fixed: each stack now loads independently; a failure prints a clear warning
  and excludes just that stack, so one bad/incomplete folder can't block training on the rest.

Verified: all 5 current stacks (`Catherine_Stack1`, `Helena_Stack1`, `Juliet_Stack1`,
`Juliet_Stack2`, `Juliet_Stack3`) decode cleanly and reproducibly through
`scripts/inspect_data.py` with these fixes in place. **Discovery genuinely requires no code
changes to add new stacks** — confirmed by `Juliet_Stack1` being picked up with zero edits.

### Correction: VAST skeletons are dense per-slice traces, not one seed per neuron (2026-07-08)

Sections 0, 1, 3, and 10 originally assumed roughly one seed point per neuron. That was wrong,
corrected in place above: each of the ~206 neurons has its own skeleton made of *many* nodes
(often hundreds) — one placed on (roughly) every serial slice while tracing the neuron through
the stack, with more than one node in a slice where the neuron branches or is too large for a
single click to represent. Phase B's design (§3, §10) was updated accordingly: group nodes by
skeleton ID, subsample each neuron's trace to spaced-out seeds instead of running inference at
every node, then union same-neuron predictions into one full-length mask.

**Phase B (turning a trained model + the real hard-drive data into masks VAST can load back
in) is still entirely unbuilt.** [CLAUDE.md](CLAUDE.md) is the concrete, checklist-style
next-steps brief for picking this up on the GPU machine (or any future session) — read that
first when resuming; it points back here for rationale.

### Real training run completed on the GPU machine (2026-07-08)

30 epochs, defaults (`32x256x256` patch, `base_channels=24`), `Helena_Stack1` excluded via
the new `--exclude-stacks` flag since its annotation is still the Catherine placeholder (not
replaced in time). ~400s/epoch on an RTX 3090. Best val dice **0.583** (epoch 30) — train
loss and val dice were both still improving at the last epoch with no sign of plateauing, so
this is a working first checkpoint, not a converged one; more epochs would likely help before
trusting it for anything beyond a Phase B pipeline smoke test.

### Phase B built and verified end-to-end on real data (2026-07-09)

With the hard drive (`E:\ppa_b4v5s13\aligned_stack`, `E:\Milan_files\`) attached to the GPU
machine, the two big unknowns CLAUDE.md flagged got resolved:

- **Full EM stack format**: it's VAST's own tiled multi-resolution pyramid, not flat PNGs —
  a `volume.vsvi` config (JSON-*like*, but with unescaped backslashes in its path templates,
  so needs a lenient parse, not `json.loads` directly) plus
  `mip0/<section-folder>/*_tr<row>-tc<col>.png` tiles, 4096x4096 each, 9 rows x 25 cols,
  102400x36864x1060 voxels total, 2/2/30nm scale (same scale as `Training Data/`).
  `MissingImagePolicy: black` in the vsvi confirms missing edge tiles (the tissue doesn't
  fill the full rectangular grid) should read as zero, not error. `src/seeded_unet/
  phase_b_stack.py` implements a windowed reader against this — verified byte-exact against
  raw tiles directly (including a read straddling a tile boundary) without ever loading the
  full volume into memory. Decoded tiles and glob lookups are LRU-cached, which cut real
  per-seed inference time from ~12s to ~3.5s average (seeds along the same trace usually
  revisit the same or a neighboring tile).
- **`.vsanno` was not reverse-engineered in the end, and that turned out to be the right
  call.** The raw binary (magic `VSA0`) has a header matching the stack's dimensions, a
  table of standard structure tags (Axon, Dendrite, Cell Body, Spine, etc.), and a per-node
  name table — enough to confirm the file really does contain hundreds of nodes per neuron,
  but a brute-force scan for the actual (x,y,z) coordinate array turned up a candidate that
  looked plausible (values in-range) but didn't hold up (z didn't move smoothly slice-to-
  slice the way a real trace should) — i.e. guessing further risked silently wrong
  coordinates. Asked the researcher instead: VAST can export skeleton/node data directly to
  CSV, avoiding the binary format entirely. That export is checked in at
  `Data/VAST_skeleton_data.csv` (209,892 rows, no header). Column semantics were verified
  against the data itself, not assumed:
  - col 0 tree/skeleton id (269 distinct), col 1 local node id (contiguous 0..N-1 per tree,
    confirmed), cols 3/4/5 = x/y/z in the full stack's mip0 voxel frame (range
    x:24682-68015, y:10506-23948, z:27-948 — a sub-region of the full 102400x36864x1060
    volume, consistent with "a semi-random coordinate in the nerve ring" rather than the
    whole worm), col 6 parent local id (-1 for the tree's root; for every other row, checked
    to resolve to another real local id in the same tree with **zero exceptions** across all
    209,892 rows), col 8 a branch-child local id (same verification, also zero exceptions),
    col 16 a free-text annotator tag, non-empty on 273 rows (values like `risky_merge`,
    `potential_merge`, `cell_body_and_nerve_ring_exit`, `merge_with_vnc` — exactly the
    touching/merging-neuron failure mode this whole task is most exposed to (§11); not used
    yet, but a natural confidence/priority signal for §12's active-learning loop later).
    Columns 2, 7, 9-15, and the second string field have no confirmed meaning and aren't
    used. `src/seeded_unet/vast_skeleton.py` implements this parser plus `subsample_seeds()`
    (walks each tree from its root, iteratively — trees run up to ~8000 nodes deep, past
    Python's default recursion limit — picking a seed every ~500nm of physical path
    distance, always including leaves so neurite tips aren't missed).
  - This also resolved the coordinate-alignment question in §1/§11/§10 a different way than
    expected: real seeds already come with absolute mip0 coordinates in the same frame
    `phase_b_stack.py` reads, so there's no crop-relative-frame translation to solve at all.
    The researcher separately confirmed `aligned_stack` and the `Training Data/` crops are
    the same underlying export (crops were made by picking a coordinate in VAST and exporting
    a small `Training Data`-shaped region for webKnossos annotation, with webKnossos then
    resetting that crop's origin to (0,0,0) — hence no shared origin to look up, by
    construction, not by omission). An attempted independent check (template-matching a known
    crop's pixels against `aligned_stack` at the corresponding slice) did not turn up a
    confident match — worth another look someday, but not blocking, since Phase B doesn't
    depend on that alignment either way.
- `src/seeded_unet/phase_b_infer.py` (+ `scripts/phase_b_infer.py`) ties it together: given a
  tree id, subsamples seeds, reads a real patch per seed from the real full stack, runs the
  existing trained model (reuses `infer.run_inference` completely unchanged — the only new
  code is *getting real patches and seeds in front of it*), and saves packed per-seed masks
  plus placement to `outputs/phase_b/tree_<id>/predictions.npz`. Verified on tree 1: real,
  non-degenerate predictions (~58-60% foreground fraction per patch, not 0% or 100%).
- **Not built yet**: merging a tree's per-seed patch predictions into one full-length neuron
  mask (§10's union/majority-vote step), and a writer for whatever format VAST needs to
  import a segmentation back in — genuinely unknown still, needs the researcher to check
  VAST's import options before this is worth guessing at. A full run across all 269 trees
  (~51,000 subsampled seeds at current settings) is estimated at very roughly two days even
  with tile caching — worth deciding scope deliberately rather than kicking that off blind,
  especially since the underlying checkpoint (above) hadn't converged yet either.

### Running this on the lab GPU machine

1. Copy the whole repo (or `git clone`/`git pull` it there) — `Training Data/` and the code are
   both small enough to live in git; `.cache/` and `outputs/` are gitignored and will just
   regenerate locally.
2. `py -m pip install -r requirements.txt` (this installs the CPU build of `torch` by default on
   most platforms via plain `pip install torch` — if the lab machine's CUDA version needs a
   specific wheel, install torch first from https://pytorch.org/get-started/locally/ for that
   machine's CUDA version, *then* `pip install -r requirements.txt` for the rest).
3. `py scripts/train.py` — device is auto-detected; it will print `Using device: cuda` if torch
   sees a GPU. No code changes needed to switch machines.
4. Expect the ~4 minute one-time seed-distribution setup cost to repeat once on that machine
   (its own `.cache/` starts empty), then real per-epoch timing that should be dramatically
   faster than the CPU numbers above — worth re-measuring a couple of epochs before committing to
   a long run, the same way the timing probe was used here.
