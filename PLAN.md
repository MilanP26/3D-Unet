# Seeded 3D U-Net for Pristionchus Neuron Segmentation — Development Plan

Status: **planning only, no model code yet.** This document is the result of inspecting the
repo and the one annotation package currently checked in, plus the data layout described by
the researcher. It is meant to be updated as more datasets and answers to open questions come in.

## 0. What inspection actually found (ground truth as of 2026-07-07)

The repo (`3D-Unet`) contained only a `README.md` until `Juliet_stack2.zip` (2.6 MB) was pushed.
Extracting it:

```
Juliet_stack2.zip
├── Juliet_stack2.nml        # webKnossos annotation metadata (XML)
└── data_Volume.zip          # webKnossos volume-annotation data, WKW format
```

**`Juliet_stack2.nml`** is a *webKnossos volume annotation* export, not a skeleton file:
- `<scale x="2.0" y="2.0" z="30.0" unit="nanometer"/>` — voxels are **highly anisotropic**:
  2 nm × 2 nm in-plane, 30 nm in z (15× coarser in z). This is typical serial-section EM and
  materially affects patch shape, convolution kernels, and augmentation choices (see §6).
- `<volume id="0" name="Volume" location="data_Volume.zip" format="wkw" largestSegmentId="70">`
  — one volume/segmentation layer, 70 segment IDs.
- A `<segments>` list of 70 entries, each with an `anchorPosition` (voxel coords) and creation
  timestamp; only ~15 of the 70 have an explicit `color.*` and/or a `name="Segment N"` attribute
  — the rest are default/unnamed. This means **not all 70 IDs necessarily represent a
  deliberately finished, identified neuron** — some may be scaffolding, accidental clicks, or
  WIP. This needs a resolved convention before labels are trusted (see Open Questions).
- There is **no `<trees>`/skeleton section** in this file — confirms the researcher's
  description that VAST skeleton nodes and webKnossos coloring are two independent systems,
  not two views of the same annotation file.
- There is **no raw EM / "color" layer referenced anywhere in this file** — only the "Volume"
  (label) layer.

**`data_Volume.zip`** decodes to a WKW (webKnossos-wrapper) dataset:
- Only a `1/` (mag-1, i.e. full resolution) folder — no downsampled mags, no
  `datasource-properties.json` (which is where the layer's global bounding box / voxel dtype
  / offset normally live at the dataset level).
- Grid: `z0`, `z1` shard folders, each containing 32×32 `x*.wkw`/`y*.wkw` block files → standard
  32³-voxel WKW buckets, so the annotated bounding box spans up to ~1024×1024×2048 voxels at
  mag 1, but block file sizes are mostly ~555 bytes (near-empty/background) with only a handful
  around 4.3 KB (actual labeled content) — **the real painted region is a small, sparse
  sub-volume of that bounding box**, not the whole thing.
- Header bytes (`57 4b 57 01 05 02 03 04...`) decode to WKW magic + version 1, and are
  consistent with 32³ buckets — exact voxel dtype/channel count should be confirmed with the
  `wkw`/`webknossos` Python libraries rather than assumed from a manual byte read.
- **Confirms the researcher's suspicion: this annotation download contains only the
  segmentation mask, not the raw EM image.** The raw EM must be fetched separately (same
  `datasetId="6a3be4a7010000c801198b2b"`, same bounding box) via the webKnossos web UI/API,
  or must already exist as a saved crop somewhere, or must be re-derived from the USB EM stack
  if that stack is the same source data as what was uploaded to webKnossos.

**Environment**: no Python interpreter, no `wkw`/`webknossos`/`torch`/`numpy` packages, and no
GPU are currently set up in this workspace. Actual voxel-level inspection (label sizes, class
balance, dtype confirmation) requires setting this up — noted as an action item, not done yet
per the "don't implement yet" instruction.

**Two independent annotation sources, confirmed:**
| Source | Tool | Location | Contains | Content confirmed today |
|---|---|---|---|---|
| Skeleton seed points | VAST | USB hard drive, `.vsanno` | ~206 skeleton nodes, one skeleton per neuron | not inspected (not accessible from this machine) |
| Colored/manual masks | webKnossos | downloaded per-dataset zip, 1 pushed so far (+2-3 more to come) | per-voxel instance labels (`data_Volume.zip`, WKW) + metadata (`.nml`) | inspected above — mask only, no raw EM bundled |
| Raw EM stack | — | USB hard drive (loaded into VAST); *may or may not* be same crop as webKnossos raw layer | full EM volume | not yet located/confirmed |

This means the single biggest structural risk to this project is **coordinate alignment
across three independently-addressed spaces** (VAST voxel space, webKnossos dataset space, and
whatever the final training array uses) — flagged throughout below and called out first in
Open Questions.

---

## 1. Inspecting and understanding the data formats

Concrete, tool-level steps (in order), to run once Python is set up:

1. **webKnossos volume annotation** (`data_Volume.zip` + `.nml`): use the `wkw` Python package
   (or the higher-level `webknossos` client) to open the mag-1 layer and read it into a dense
   `numpy` array over its bounding box. Cross-reference the resulting unique label IDs actually
   present in the array against the 70 IDs listed in the `.nml` — some listed IDs may have zero
   voxels (dead/empty segments), and some may be much smaller than others (specks vs. real
   somas). Compute a size histogram per ID immediately; this both validates data integrity and
   informs patch size (§4/§6).
2. **webKnossos raw EM layer**: use the `webknossos` Python client with account credentials to
   download the "color"/raw layer for the same `datasetId`, same bounding box as the volume
   annotation, at the same mag. If the client can't reach a self-hosted instance
   (`wkUrl="http://localhost:9000"` in the `.nml` suggests a local/self-hosted webKnossos, not
   the public webknossos.org), determine correct host/credentials with the researcher first.
3. **VAST `.vsanno` file** (on the USB drive, not accessible from here): open with VAST itself
   or parse as text — VAST's native tracing export is a readable node list. Need to confirm
   per node: neuron/skeleton identity, x/y/z coordinates, and the coordinate convention (raw
   full-stack voxel indices? Some crop-relative offset? Physical units?). This needs to happen
   on/near the machine with the USB drive.
4. **Registration between VAST and webKnossos coordinate spaces**: once both are read, take a
   handful of unambiguous shared landmarks (e.g., same neuron's centroid) if any exist, or use
   the known EM stack geometry (crop offset, voxel size) written down when the crop was
   uploaded to webKnossos, to derive the affine/offset mapping. This is the crux of the
   pipeline — see Open Questions #1–3.
5. Repeat steps 1–2 for each additional webKnossos dataset as it's pushed, and track per-dataset
   scale/offset/datasetId in one manifest file (e.g. `datasets.yaml`) rather than re-deriving
   each time by hand.

## 2. Manual colored annotation → 3D label masks

- Decode the WKW volume layer into a dense instance-label numpy array per snippet (background =
  0, other integers = instance IDs). Use the actual bounding box implied by populated buckets,
  not the full shard grid, to avoid allocating a mostly-empty 1024×1024×2048 array.
- Filter the 70 nml-listed IDs down to the set that actually has non-trivial voxel counts in the
  decoded array; apply a minimum-voxel-count threshold to drop specks/accidental clicks (exact
  threshold to be set once real size histogram is available).
- For training, convert each retained instance ID into its own binary mask (`label == id`) on
  demand rather than storing N separate dense volumes.
- Assign each instance a **globally unique** neuron ID across all snippets/datasets (local IDs
  1–70 are only unique within one snippet) — ideally tied to actual neuron identity/name if
  known from the VAST skeleton labeling, since the eventual scientific goal is per-neuron
  identity, not just arbitrary instance separation.
- Decide and document a convention for which segments count as "finished, trustworthy" labels
  (e.g., only those with an explicit name, or only those the researcher explicitly confirms) —
  see Open Question #4.

## 3. VAST skeleton nodes → model input channel

- Parse `.vsanno` into `(neuron_id, x, y, z)` tuples in VAST's native voxel space.
- Transform into each webKnossos snippet's local voxel grid using the offset/scale mapping from
  §1.4. Only nodes that land inside a given snippet's bounding box are usable with that snippet.
- Discard (or separately log) nodes that land outside any painted instance mask after transform
  — these indicate either a registration error or a neuron VAST marked that wasn't colored in
  webKnossos, both worth knowing about explicitly rather than silently training on them.
- Seed channel encoding: build a single-channel volume the same shape as the image patch,
  zero everywhere except a small blob at the seed location. Two options:
  - **Binary point/dot** (simplest, but very sparse — extreme class imbalance in that channel
    itself, and brittle to exact voxel-level jitter).
  - **3D Gaussian heatmap**, sigma chosen in physical nm then converted to per-axis voxel sigma
    given the 2/2/30 nm anisotropy (so the blob is physically round, not voxel-round) — this is
    the standard choice in interactive/prompted segmentation literature and is the recommended
    starting point.
  - Treat the choice as an empirical hyperparameter to ablate once a baseline trains, not a
    decision to over-engineer up front.

## 4. Building training samples (patches)

- **First measure, don't guess, patch size**: once §1 decodes real instance masks, compute the
  bounding-box size distribution of actual neuron instances (in voxels and in physical nm,
  respecting anisotropy) — this should directly drive patch size rather than picking one
  arbitrarily.
- Center each training patch on a (transformed, validated) seed coordinate, sized to comfortably
  contain the full extent of the great majority of instances found in step above, with margin.
- Augment seed position with small random jitter per training epoch (not exact-center every
  time) so the model doesn't overfit to pixel-perfect seed placement — real usage will have
  human-click imprecision at inference time.
- **Only-partially-annotated regions**: do not assume "unlabeled voxel = true background."
  Determine, per snippet, the sub-region that was *exhaustively* colored (all neurons present
  were painted) vs. merely "colored so far" — training loss should probably be masked/ignored
  outside the exhaustively-annotated region, otherwise the model is taught false negatives.
  This needs a direct answer from the researcher (Open Question #5) since it isn't recoverable
  from the files alone.
- Consider a modest number of deliberate negative samples (seed placed just outside a neuron,
  or in background) later, once the core positive-seed task works, to teach precise boundary
  respect — not required for the v1 pipeline given the current task framing (seed is always
  inside a neuron).

## 5. Model input / output tensors

- **Input**: 2-channel volume `[raw_EM_patch, seed_channel]`, shape `(2, Dz, Hy, Wx)`, raw
  intensity normalized (e.g. per-dataset percentile normalization, since EM contrast can vary
  between snippets/sessions).
- **Output**: 1-channel binary mask, same spatial shape, sigmoid activation (probability that
  voxel belongs to the seeded neuron).
- Optional (not v1): an auxiliary distance-transform or boundary-map output channel as extra
  supervision to help separate touching neurons — worth revisiting once touching-neuron
  failures are actually observed empirically (§9), not designed in blind.

## 6. 3D U-Net architecture

- Standard encoder-decoder 3D U-Net with skip connections, but **anisotropy-aware**: given
  2/2/30 nm voxels, early pooling/strides should be in-plane-only (e.g. stride `(1,2,2)`) before
  any pooling touches z, and/or use anisotropic kernels (e.g. `(1,3,3)` mixed with `(3,3,3)`) —
  this mirrors standard practice in anisotropic EM connectomics networks rather than treating
  the volume as isotropic.
- Prefer GroupNorm/InstanceNorm over BatchNorm given small batch sizes typical of 3D volumes
  under memory constraints.
- Start from an established, tested 3D U-Net implementation (e.g. MONAI's `UNet`, or an
  nnU-Net-style anisotropic config) rather than writing the architecture from scratch — data
  volume is currently small, so implementation risk should be minimized in favor of getting a
  correct baseline running.
- Given how little labeled data exists right now (one snippet, on the order of a few dozen
  usable instances), plan explicitly for: heavy augmentation, a relatively shallow/small network
  to limit overfitting, and considering self-supervised pretraining on unlabeled raw EM (once
  available) or transfer from an existing permissively-licensed EM segmentation model.

## 7. Loss functions

- Baseline: combined **Dice + BCE** — standard robust default for imbalanced binary volumetric
  segmentation and a reasonable v1 choice.
- Alternative to try if class imbalance or small-neuron performance is poor: **focal loss** or
  **focal Tversky loss** (lets FP/FN be weighted differently — relevant because under-segmenting
  a neuron and leaking into a neighboring one have different costs).
- Boundary-aware loss (e.g. distance-weighted or explicit boundary term) as a later refinement
  once touching-neuron leakage is observed as an actual failure mode, not designed preemptively.

## 8. Train/validation/test split (no leakage)

- Split at the **instance (neuron) level**, not the patch level — the same neuron instance
  (even via a different seed node or a jittered crop) must never appear in more than one split.
- Ideally split at the **snippet/dataset level** once there are enough snippets (i.e., whole
  datasets held out entirely), since that's the only way to test true generalization to new
  tissue regions/imaging sessions.
- With currently only one snippet: use grouped k-fold / leave-some-neurons-out cross-validation
  within it for a first read on signal, but treat this explicitly as a **weak, interim**
  estimate — real held-out generalization needs a second, independent snippet at minimum
  (Open Question #8 — same worm or different worm matters here too).

## 9. Evaluation metrics

- Per-instance volumetric **Dice** and **IoU**.
- Voxelwise and instance-level **precision/recall**.
- **Boundary accuracy**: average symmetric surface distance or boundary-F1 between predicted
  and ground-truth surfaces.
- **Leakage metric** (specific to this task): fraction of predicted foreground voxels that fall
  inside a *different* ground-truth instance than the seeded one — directly measures the
  touching-neuron failure mode this architecture is most exposed to.
- **Seed-sensitivity**: re-run inference with the seed perturbed by a few voxels and measure
  output stability — a practical proxy for how robust the model will be to imprecise human
  clicks at real inference time.

## 10. Inference workflow

- Given a raw volume region and one seed point: crop a patch around the seed at the same
  physical scale used in training, run the model, and paste the (thresholded, e.g. 0.5 initially
  then tuned on validation data) prediction back into the full-volume coordinate frame.
- Because this is seed-conditioned rather than whole-volume segmentation, there's no
  tile-and-stitch merging problem the way there would be for standard dense segmentation —
  inference is inherently local per seed.
- If a neuron has multiple VAST skeleton nodes along its length, running inference from each and
  taking the union (or majority vote) is a natural way to get a more complete mask and a
  built-in consistency check (do independent seeds on the same neuron agree?).

## 11. Risks / open issues

- **Coordinate alignment between VAST and webKnossos** is the largest concrete risk — confirmed
  today that this annotation download has no raw EM and no bounding-box/offset metadata file,
  so pixel-aligned (image, seed, mask) triples do not yet exist and must be actively constructed.
- **Ambiguous segment validity**: only a fraction of the 70 listed segment IDs are named/colored;
  a hard rule for "real neuron label" vs. "WIP/noise" is needed before trusting any ID blindly.
- **Partial annotation extent**: unlabeled ≠ guaranteed background unless the exhaustively-
  colored region is explicitly known per snippet.
- **Extreme anisotropy** (2/2/30 nm) affects patch shape, kernel choice, and augmentation
  (rotations must respect true physical proportions, not treat the volume as isotropic).
- **Very small dataset currently** (effectively one snippet) → high overfitting risk.
- **Class imbalance** (neuron voxels vs. background within any given patch).
- **Touching/adjacent neurons**: the model must learn to stop at the true instance boundary from
  a single seed rather than bleeding into a neighbor — likely the hardest part of this task.
- **Annotation noise**: hand-drawn boundaries have some inherent human inconsistency; expect
  soft/noisy ground truth, not pixel-perfect truth.
- **Memory/compute limits**: 3D patches at 2 nm in-plane resolution can get large fast; concrete
  patch size needs to be checked against available GPU VRAM once that's known (Open Question
  #6) — no GPU is currently set up in this workspace.
- **WKW block-boundary effects**: data is stored in compressed 32³ voxel buckets; whatever
  reader is used must correctly handle patches that straddle bucket boundaries.
- **Data locality**: the full raw EM stack (many GB, on a USB drive) cannot be pushed to GitHub;
  training will need to happen on/near that drive, or with a deliberately extracted, appropriately
  small subset — plan the repo to hold code/config/manifests only, never raw EM or full-resolution
  masks.

## 12. Future human-in-the-loop extension

- **v1**: train on the existing hand-colored-mask + VAST-seed pairs, as scoped above.
- **v2 (active learning loop)**: run the trained model on new seeds → propose masks → a human
  (in VAST or webKnossos) reviews and corrects them → each corrected (seed, mask) pair is added
  back into the training set → periodic retraining/fine-tuning.
- Prioritize which model proposals get reviewed first using the uncertainty signals already
  defined in §9 (low confidence, high seed-sensitivity, or high predicted-leakage-into-neighbor
  risk) so human correction time is spent where it improves the model most.

---

## Open questions to resolve before coding begins

1. **Raw EM access**: how do I get the raw EM image data matching this webKnossos annotation
   (same dataset/crop)? Via the `webknossos` Python client with your login against your
   (apparently self-hosted, `localhost:9000`) instance, a "download with volume data" export
   option, or do you already have that exact crop saved separately from an annotation-only
   download?
2. **VAST file access/schema**: can you share a sample of the actual `.vsanno` content (or a
   VAST-exported text/CSV of the skeleton nodes) so I can confirm its coordinate convention,
   units, and per-node neuron identity fields?
3. **Same source stack?**: is the EM stack loaded into VAST literally the same raw stack/crop
   that was uploaded to webKnossos as "Juliet_stack2," or a larger/different stack? If different,
   what's known about the crop offset between them (even approximately)?
4. **Segment validity rule**: of the 70 segment IDs in this annotation, which count as finished,
   trustworthy neuron labels vs. WIP/scaffolding — is it "only named," "only colored," or
   something else, or do you need to go back and clean this up in webKnossos first?
5. **Annotation completeness extent**: what region of this snippet was *exhaustively* colored
   (every neuron present painted) versus partially colored ("as far as I've gotten")? This
   determines whether unlabeled voxels are safe to use as negative/background training signal.
6. **Compute**: what GPU/VRAM do you have available for training, and will training happen on
   the machine with the USB drive attached, or will a subset of data be copied elsewhere?
7. **Typical neuron size**: once I can decode actual voxel data, I can measure this directly —
   but if you already know roughly how large a Pristionchus neuron soma/process is in this
   dataset (voxels or nm), that would help sanity-check patch size early.
8. **Relationship between the 2–3 additional webKnossos datasets**: are they from the same
   worm/individual (different regions of the same animal) or different individual worms? This
   affects how train/val/test splitting should be grouped to avoid leakage (§8).

Setting up a local Python environment (`numpy`, `wkw`, `webknossos`, `torch`) is the natural
next practical step to start answering #1, #4, #5, and #7 quantitatively — but is being held
until you confirm you want to move past planning.
