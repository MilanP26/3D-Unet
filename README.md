# 3D-Unet

Seeded (prompt-conditioned) 3D U-Net for segmenting individual *Pristionchus pacificus*
neurons from EM volumes, given a point placed inside the neuron. See [PLAN.md](PLAN.md) for
the full design rationale, data findings, and open questions.

## Setup

```
py -m pip install -r requirements.txt
```

Python 3.11 was used for development. GPU (CUDA) is auto-detected and used if available;
otherwise everything falls back to CPU.

## Data layout

Each folder under `Training Data/<StackName>/` is one self-contained training example:
a sequence of raw EM slice PNGs plus one webKnossos annotation `.zip` (mask). New stacks
just need to be dropped in following that same convention -- the code discovers them
automatically at the start of every training run, no config or code changes needed. This
also covers **replacing** an existing stack's annotation zip later (e.g. a placeholder
swapped for the real thing): the decode cache is keyed on the annotation zip's actual
content, not just its filename or PNG count, so a changed zip is automatically re-decoded
rather than silently reusing the old cached mask.

If a stack folder fails to decode (corrupt zip, unexpected internal structure, etc.), it's
skipped with a clear warning printed at the top of the run rather than crashing training for
every other stack -- worth actually reading that warning output once in a while, since a
silently-skipped stack is not the same as "no stacks have problems."

## Inspecting data

```
py scripts/inspect_data.py
```

Decodes every stack, cross-checks decoded labels against webKnossos's own recorded
segment anchor points (a correctness check on the WKW decode + PNG/mask alignment), and
prints per-stack instance-size statistics.

## Training

```
py scripts/train.py --epochs 30
```

Useful flags (see `py scripts/train.py --help` for all of them):

| Flag | Meaning |
|---|---|
| `--patch-size Z Y X` | training patch size in voxels (default `32 256 256`) |
| `--base-channels N` | U-Net width (default 24) |
| `--samples-per-instance N` | synthetic seed samples drawn per neuron per epoch |
| `--batch-size N` | |
| `--device cuda\|cpu` | default: auto-detect |
| `--exclude-stacks NAME [NAME ...]` | skip these `Training Data/<Stack>` folders entirely (e.g. a stack still using a placeholder annotation) |

**What you'll see:** a one-time "Precomputing seed distributions" progress bar (this cost
depends only on how many neurons exist, not on patch size or model size -- it doesn't
repeat during training), then a `tqdm` progress bar per epoch for train and validation
batches with a running loss, then a one-line summary per epoch:

```
epoch   3/30 | train_loss 0.4622 | val_loss 0.4726 | val_dice 0.6969 | val_iou 0.5895 | 158.5s (avg 158.9s/epoch, ETA 42.3 min)
```

The ETA is a live running average of actual epoch time on your machine, recalculated every
epoch -- it becomes more accurate after the first couple of epochs. All of this is also
written to `outputs/training_log.csv` as it goes, and checkpoints are saved to
`outputs/checkpoints/{last,best}.pt` after every epoch, so you can kill training early and
still have a usable model.

**On CPU-only machines**: real 3D U-Net training is slow without a GPU. Measured on a
CPU-only laptop: a small config (24x128x128 patch, base_channels=16) ran ~4.7s/batch
(~23 min/epoch); the *default* config above (32x256x256, base_channels=24) extrapolates to
roughly hours per epoch on the same hardware -- see [PLAN.md](PLAN.md) section 13 for the
full numbers. Nothing about the code requires a GPU (device is auto-detected either way),
but for a real training run, use a CUDA machine; for quick local iteration/debugging on a
laptop, shrink `--patch-size` and `--base-channels` (and re-time it -- see section 13's
"timing probe" approach) rather than running the defaults.

There's also a one-time setup cost the first time seed distributions are computed for a given
`--min-instance-voxels`/seed-bias combo (a distance transform per neuron instance, ~4 minutes
for the current 3 stacks) -- this is cached to `.cache/seed_dist/` so it's paid once ever per
machine, not once per run.

**Checkpoint loading note**: newer PyTorch (2.6+) defaults `torch.load` to `weights_only=True`,
which used to reject these checkpoints (they store a few `Path` objects in `args`). Fixed on
both ends -- `train.py` now stores plain strings, and `infer.py`/`phase_b_infer.py` explicitly
load with `weights_only=False` (safe: these are checkpoints this codebase produces itself, not
third-party files) -- so this affects neither old nor new checkpoints going forward.

## Plotting loss curves

```
py scripts/plot_training_log.py outputs/training_log.csv
```

Reads any `training_log.csv` written by `train.py` and saves `loss_curves.png` (train/val
loss + val Dice/IoU) next to it. No GPU or training data needed -- just the CSV file, so
this can be run anywhere, including a different machine than the one that trained.

## Inference (validation-style, within a Training Data stack)

```
py scripts/infer.py --checkpoint outputs/checkpoints/best.pt --stack-name Catherine_Stack1 --seed-zyx 30 400 400
```

Crops a patch around the given seed voxel in that stack, runs the model, and saves the
predicted binary mask patch to `outputs/inference_mask.npy` **and** a viewable PNG to
`outputs/inference_visualization.png` (raw EM with the predicted mask overlaid across a few
slices -- this, not the raw `.npy` array, is what to hand someone who asks "what did the
model predict"). Pass `--visualization ''` to skip the image. This only needs the checkpoint
file and this repo's `Training Data/` -- no GPU or hard drive required, so it runs fine on a
laptop as long as the checkpoint has been copied over from wherever training happened.

## Phase B: inference on the full worm with real VAST seeds

Requires the external hard drive attached (the full EM stack + `Data/VAST_skeleton_data.csv`
are far too large/sensitive to live in this repo's `Training Data/`; see PLAN.md/CLAUDE.md
for how this was reverse-engineered and verified). Real skeleton seeds already come with
absolute coordinates in the full stack's own frame, so no coordinate-alignment step is
needed -- see CLAUDE.md "Building Phase B" for the full story.

```
py scripts/phase_b_infer.py --checkpoint outputs/checkpoints/best.pt --tree-id 1 --vsvi "E:\ppa_b4v5s13\aligned_stack\volume.vsvi" --max-seeds 20
```

For one skeleton (`--tree-id`, see `Data/VAST_skeleton_data.csv`), subsamples spaced-out
seeds along its traced length (default every 500nm, `--target-spacing-nm`), reads a real
patch per seed straight from the full tiled stack, runs the trained model, and saves packed
per-seed masks + placement to `outputs/phase_b/tree_<id>/predictions.npz`. `--max-seeds` caps
how many seeds run, for a quick test instead of a full tree. `--save-example-visualizations N`
(default 3) also saves `example_node<id>.png` overlay images for a spread of seeds along the
trace -- worth keeping on, since `predictions.npz` itself only stores packed masks, not the
raw EM, and the raw EM is only readable while the hard drive is attached; without these PNGs
saved during the run, producing a viewable image later would mean reading the hard drive
again. **Not yet built**: merging a tree's per-seed predictions into one full-length mask, and
a writer for whatever format VAST needs to import a segmentation back in (still unknown --
see CLAUDE.md).

## Code layout

```
src/seeded_unet/
  stack_io.py       discovers/decodes/caches Training Data/<Stack>/ folders; dedups stacks
                    that share identical raw EM so they never get split across train/val
  seeds.py          synthetic seed-point sampling (interior-biased) + Gaussian heatmap channel
  dataset.py        builds the per-neuron instance list, train/val split, patch sampling
  model.py          anisotropy-aware 3D U-Net (in-plane pooling before z-pooling)
  losses.py         Dice + BCE loss, Dice/IoU metrics
  train.py          training CLI
  infer.py          seed -> mask inference CLI (Phase A, within a Training Data stack)
  visualize.py      turns a (raw patch, predicted mask) pair into a viewable PNG overlay
  phase_b_stack.py  windowed reader for the full hard-drive EM stack (VAST's tiled format)
  vast_skeleton.py  parses Data/VAST_skeleton_data.csv, subsamples seeds along each trace
  phase_b_infer.py  Phase B inference CLI (real seeds, real full-stack patches)
scripts/            thin entry points (`train.py`, `infer.py`, `inspect_data.py`,
                    `phase_b_infer.py`, `plot_training_log.py`)
```
