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
  external hard drive): **not built yet.** This is the main thing to build next.

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
- [ ] **Phase B is unimplemented**: nothing in this repo reads a `.vsanno` file, reads the
      full hard-drive EM stack, runs the model across a whole neuron's traced length, or
      writes output back into a format VAST can import. See "Building Phase B" below.

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

This cannot be built blind. In order:

1. **Look at the actual `.vsanno` file** (only exists on the USB hard drive — inaccessible
   from a laptop with the drive unplugged). Figure out: how nodes are grouped by skeleton/
   neuron ID, the coordinate convention/units, and whether it's XML/text (parseable similarly
   to the `.nml` files already handled in `stack_io.py`) or something else.
2. **Figure out the full EM stack's file format** on the hard drive — a folder of slice
   images like `Training Data/` (just much larger), a single volume file, or a VAST-native
   format. This determines how the Phase B reader needs to work, and whether it can reuse
   `stack_io._load_raw_png_stack`-style logic or needs something new (e.g. windowed reads
   without loading the whole thing into memory — the full stack will not fit in RAM the way
   the small training crops do).
3. **Figure out what format VAST accepts to import a segmentation/mask volume back in.**
   Don't guess at this — building output in the wrong format wastes the rest of the work.
4. Once those three are known, build (roughly, as a new `src/seeded_unet/phase_b.py` or
   similar): a `.vsanno` parser grouping nodes by skeleton → a trace-subsampling step → a
   windowed full-stack reader → per-seed inference reusing `infer.run_inference` → a
   same-neuron union/merge step → a writer for whatever format step 3 requires.
5. Coordinate alignment (VAST's full-stack coordinate space vs. whatever crop-relative frame
   the trained model was built around) needs to be resolved for real seeds to land in the
   right place — see PLAN.md §1 steps 6-7 and §11.

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
