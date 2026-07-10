# Orientation for an AI agent picking this up

Read this first. Full rationale lives in [PLAN.md](PLAN.md); usage instructions live in
[README.md](README.md). This file is the "what's actually done, what to do next" summary —
if it ever disagrees with PLAN.md on a *design* question, PLAN.md wins; update this file to
match rather than the reverse.

## One-paragraph summary

Seeded (prompt-conditioned) 3D U-Net that, given an EM volume and a point inside a neuron,
predicts a binary mask of that neuron. Two separate phases:
- **Phase A — training** (this repo, small cropped stacks under `Training Data/`): fully
  built, verified against real data, currently only run at toy scale on a CPU-only laptop.
- **Phase B — full-worm inference** (a much bigger EM stack + real VAST skeleton seeds, on an
  external hard drive): core pipeline built and verified end-to-end on real data
  (2026-07-09) — reads real seeds, reads real patches from the full stack, runs the trained
  model. Not yet done: per-neuron mask merging and a VAST-importable output writer (format
  still unknown). See the status checklist and "Building Phase B" below.

## Status checklist

- [x] Data pipeline (`src/seeded_unet/stack_io.py`): discovers `Training Data/<Stack>/`
      folders automatically, decodes PNG raw EM + WKW annotation masks, caches to disk
      (invalidates correctly on content change, not just filename/count), dedups stacks that
      share identical raw EM, isolates per-stack failures so one bad folder doesn't block the
      rest.
- [x] Synthetic seed sampling + Gaussian heatmap channel (`seeds.py`) — training uses points
      sampled from the interior of ground-truth masks, not real VAST coordinates.
- [x] Anisotropic 3D U-Net (`model.py`), Dice+BCE loss (`losses.py`).
- [x] Training CLI with progress bars, live ETA, checkpointing, CSV logging (`train.py`,
      `scripts/train.py`).
- [x] Verified end-to-end on real data on a CPU-only laptop (toy-scale config only — see
      PLAN.md §13 for measured timing). Real full-scale training run is expected to happen on
      a lab GPU machine; no code changes should be needed to switch machines (device
      auto-detects).
- [x] `scripts/inspect_data.py` — decodes every stack and cross-checks against webKnossos's
      own recorded segment anchor points as a correctness check.
- [x] Real training run completed on the lab GPU machine (2026-07-08, RTX 3090, defaults,
      `Helena_Stack1` excluded via `--exclude-stacks` since its annotation is still a
      placeholder): 30 epochs, best val dice 0.583, val loss still falling at epoch 30 --
      not fully converged, worth more epochs before trusting it as final.
- [x] **Phase B core pipeline built and running end-to-end on real data** (2026-07-09):
    - `phase_b_stack.py`: windowed reader for the full hard-drive stack's native tiled
      format (`volume.vsvi` + `mip0/<section>/*_tr<r>-tc<c>.png`). Tile-stitching verified
      byte-exact against raw tiles directly, including a read straddling a tile boundary.
      LRU-caches decoded tiles and section-dir/tile-path glob lookups (~3.5x speedup for
      seeds that revisit nearby tiles, which is the common case along a skeleton trace).
    - `vast_skeleton.py`: parser for `Data/VAST_skeleton_data.csv` (a VAST export of the
      real skeleton annotations -- see "vsanno resolved" below). Column semantics verified
      against the raw data itself (parent references, contiguous local ids, z-per-node
      smoothness), not guessed. `subsample_seeds()` walks each tree from its root and picks
      spaced-out seeds (default 500nm) instead of running inference at every traced node.
    - `phase_b_infer.py` / `scripts/phase_b_infer.py`: for one tree id, subsamples seeds,
      reads a real patch per seed from the real full stack, runs the existing trained model
      (reuses `infer.run_inference` unchanged), saves packed per-seed masks + placement to
      `outputs/phase_b/tree_<id>/predictions.npz`. Verified working on tree 1 (real
      predictions, plausible non-degenerate foreground fractions ~58-60%).
    - **Still open, blocking a full-scale run**: (a) what format VAST needs to import a
      segmentation back in -- current output is a generic intermediate, not that format
      yet; (b) a full run across all 269 trees / ~51k subsampled seeds is estimated at
      very roughly 2 days at the current ~3.5s/seed even with tile caching -- worth
      deciding scope (all trees now vs. a subset) before committing to that.

## Critical fact that changed the Phase B design (2026-07-08)

VAST skeletons are **not** one seed point per neuron. Each of the ~206 neurons has its own
skeleton made of *many* nodes (often hundreds) — one placed on roughly every serial slice
while manually tracing that neuron through the stack, with more than one node in a slice
where the neuron branches or is too large for one click. Design implication (detailed in
PLAN.md §3/§10): group nodes by skeleton/neuron ID, subsample each neuron's trace to a
spaced-out set of seeds (a training/inference patch already spans many consecutive slices, so
running inference at literally every node would be extremely redundant), run inference at
each sampled seed, then union same-neuron predictions into one full-length mask.

## Building Phase B: what to find out first, then what to build

Steps 1 and 2 below are now resolved (2026-07-09) -- see PLAN.md §13 for the details.
Step 3 (output format) is still open.

1. ~~Look at the actual `.vsanno` file~~ **Resolved via a different route**: the raw binary
   `.vsanno` (magic `VSA0`) turned out to be undocumented and risky to reverse-engineer blind
   (a brute-force scan for coordinate arrays found a false-positive candidate that didn't
   hold up). Instead, VAST can export skeleton data to CSV directly -- the researcher did
   this and it's checked in at `Data/VAST_skeleton_data.csv`. Column semantics were verified
   against the data itself (see `vast_skeleton.py`'s docstring): tree id, per-tree local node
   id (contiguous 0..N-1), x/y/z in the full stack's mip0 voxel frame, parent local id (-1 =
   root, always resolves to another node in the same tree, verified with zero exceptions
   across all 209,892 rows), a branch-child local id, and a free-text annotator tag (e.g.
   `risky_merge`, `potential_merge`, `cell_body_and_nerve_ring_exit` -- worth surfacing later
   as a confidence signal, not yet used). 269 distinct trees.
2. ~~Figure out the full EM stack's file format~~ **Resolved**: VAST's own tiled
   multi-resolution-pyramid format (`volume.vsvi` config + `mip0/<section>/*_tr<r>-tc<c>.png`
   tiles, self-documenting, no reverse-engineering needed). `phase_b_stack.py` reads windowed
   regions without loading the full 102400x36864x1060 volume into memory; tile-stitching is
   verified byte-exact.
3. **Still open: what format VAST accepts to import a segmentation/mask volume back in.**
   Don't guess at this — building output in the wrong format wastes the rest of the work.
   `phase_b_infer.py` currently writes a generic intermediate (packed per-seed masks +
   absolute placement, not yet a VAST-importable file) so the rest of the pipeline could be
   built and verified without waiting on this answer.
4. ~~Once those three are known, build...~~ **Built** (`phase_b_stack.py`, `vast_skeleton.py`,
   `phase_b_infer.py`), verified end-to-end on tree 1. What's NOT built yet: the
   same-neuron union/merge step (currently each seed's patch prediction is saved
   independently, not merged into one full-length neuron mask) and the final writer for
   whatever step 3's answer turns out to be.
5. Coordinate alignment: **not actually needed** — real seeds come with absolute mip0
   coordinates already in the same frame `phase_b_stack.py` reads, so there's no
   crop-relative-frame translation to solve. (Separately, the pixel-level question of
   whether `aligned_stack` matches the Training Data crops' source export is confirmed by
   the researcher to be the same underlying export; an attempted independent pixel-match
   verification via template matching did not succeed, but isn't blocking since Phase B
   doesn't rely on that alignment either.)

## Things already resolved — don't re-litigate these

- Discovery of new `Training Data/<Stack>/` folders is fully automatic; no code changes are
  needed to add a stack. Verified concretely when `Juliet_Stack1` was added.
- The decode cache is content-hashed (not filename/count-based), so replacing a stack's
  annotation zip (e.g. `Helena_Stack1`'s placeholder becoming real) is picked up automatically.
- PNG/mask voxel alignment within a `Training Data/` stack is verified (not just assumed) via
  matching webKnossos's own recorded segment anchor points to decoded label values.
- Training does not need VAST coordinates or hard-drive access at all — only Phase B does.

## Known unresolved questions (see PLAN.md "Open questions" for the full list)

Segment-validity rule (which labeled IDs are trustworthy), per-stack annotation-completeness
extent, and whether stacks are from the same or different worm individuals are all still open
and affect training quality/splitting, independent of Phase B.
