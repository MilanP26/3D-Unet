# experiments/ — structural-prior experiments (Milan's side project)

A sandbox, deliberately kept **separate from the production `src/seeded_unet/` pipeline**,
for trying ideas that use the *structure* of this specific problem (a stereotyped nematode
nervous system + complete manual VAST skeletons) rather than just a bigger/deeper network.
Nothing here changes or depends on the state of the main training/inference code; it only
*reads* what that pipeline already produces.

Motivation: the affinity+LSD+mutex-watershed pipeline is already close to the published state
of the art, and its per-patch accuracy is high (0% leak / 0.983 val dice on the touching trio).
The real cost in connectomics isn't per-voxel accuracy — it's that one error anywhere along a
hundreds-of-microns-long neurite breaks the whole reconstruction and forces human proofreading.
So these experiments target the *actual* bottleneck (proofreading time + merge/split errors)
and exploit two priors generic EM methods throw away:

1. the organism is **stereotyped** (Pristionchus pacificus, ~300 neurons in near-identical
   positions; a reference connectome now exists), and
2. we hold **complete manual skeletons** for every tree — hundreds of nodes each — which are
   *topological ground truth for connectivity*, not merely seed points.

## Does any of this need the model to be retrained?

**No.** Everything here consumes the *outputs* of the already-trained affinity model:
`skeleton_priors.consistency` (#1) runs on a saved `predictions.npz` plus the skeleton CSV
(no GPU, no torch), and `skeleton_clean` cleans the production watershed output.

## Layout

```
experiments/
  skeleton_priors/
    io_utils.py       # load a phase-B predictions.npz + skeletons, map to the global frame
    consistency.py    # #1: skeleton-consistency error oracle + skeleton_clean cleaning pass
  scripts/
    check_consistency.py   # run #1 over a tree's predictions.npz -> ranked error worklist CSV
    render_report.py       # render an HTML report of the oracle's flagged patches
  outputs/                 # experiment outputs (worklists, reports); safe to delete
```

Run from the repo root:

```
py experiments/scripts/check_consistency.py --predictions outputs/phase_b_affinity/tree_1/predictions.npz
py experiments/scripts/render_report.py
```

## Findings so far (2026-07-22, real tree-1 EM)

Ran on the lab GPU machine with the F: hard drive attached, using the existing
`outputs_affinity_full/checkpoints/best.pt` (val dice 0.983), inference on CPU so the
concurrent training run was left untouched.

- **#1 oracle works and is useful.** On tree 1 it found 0 merge-leaks but flagged real
  over-extension errors: e.g. patch 18's production mask floods an 817k-voxel blob with
  *no* traced node of tree 1 — a leak, caught automatically.
- **`consistency.skeleton_clean` works.** Applying the oracle as a cleaning pass on the
  production output cut orphan/leak voxels to 0 while keeping 100% node coverage. It is a
  heuristic cleaning of production output, not a new segmenter.
- **A constrained-supervoxel agglomeration (#2) was tried and dropped.** It failed on real
  affinities (they're high nearly everywhere, so any threshold-based supervoxel step
  under-segments and different neurons collapse into one blob). Removed 2026-07-22; not a
  tuning issue, so not worth revisiting on these affinities.
