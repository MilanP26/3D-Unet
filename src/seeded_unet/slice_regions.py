"""Finds where the neurons actually are, per EM z-slice, from the real VAST skeleton
nodes -- so affinity inference only has to run inside those regions instead of scanning
the entire (102400 x 36864-voxel) slice blindly.

At a given z, nodes from different trees that traveled together (or a single neuron that
wandered off alone) form a natural clump. This clusters same-z nodes by physical proximity
(connected components under an eps-radius graph -- a lone node forms its own singleton
cluster), then fits each cluster to whatever shape actually wraps it: a circle for one node,
a capsule for two, a padded convex hull for three or more -- a fixed box would either waste
space around a round clump or clip the corners off an elongated one. `shapely`'s buffer()
gives all three for free from the same call, regardless of point count.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

DEFAULT_EPS_NM = 300.0  # same proximity radius used elsewhere in this project for "close"
DEFAULT_PAD_NM = 600.0  # generous margin so a real neuron's boundary is never clipped


def _cluster_indices(xy_nm: np.ndarray, eps_nm: float) -> list[list[int]]:
    """Connected components under the eps-radius graph -- an isolated point becomes its
    own single-element cluster rather than being dropped as noise (unlike DBSCAN's default
    "noise" label), which is exactly the "a single neuron traveled away alone" case."""
    n = len(xy_nm)
    if n == 1:
        return [[0]]
    kd = cKDTree(xy_nm)
    pairs = list(kd.query_pairs(r=eps_nm))
    if not pairs:
        return [[i] for i in range(n)]
    rows = [p[0] for p in pairs] + [p[1] for p in pairs]
    cols = [p[1] for p in pairs] + [p[0] for p in pairs]
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(adj, directed=False)
    groups: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)
    return list(groups.values())


def compute_slice_regions(
    trees: dict[int, list],
    scale_nm_xyz: tuple[float, float, float],
    eps_nm: float = DEFAULT_EPS_NM,
    pad_nm: float = DEFAULT_PAD_NM,
) -> dict[int, list]:
    """Returns {z: [shapely Polygon, ...]} -- one or more padded regions per z-slice that
    actually has real nodes, each in VOXEL (x, y) coordinates (not nm), ready to intersect
    directly against stack coordinates. Every polygon traces back to a real clump of nodes;
    slices with no real nodes at all are simply absent from the dict."""
    sx, sy, _sz = scale_nm_xyz
    by_z: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for tid, nodes in trees.items():
        for n in nodes:
            by_z[n.z].append((tid, n.x * sx, n.y * sy, n.x, n.y))

    pad_vox = pad_nm / ((sx + sy) / 2)  # pad in nm -> voxels using the (near-isotropic) xy scale

    regions: dict[int, list] = {}
    for z, entries in by_z.items():
        xy_nm = np.array([(e[1], e[2]) for e in entries])
        xy_vox = np.array([(e[3], e[4]) for e in entries])
        clusters = _cluster_indices(xy_nm, eps_nm)
        polys = []
        for idxs in clusters:
            pts = xy_vox[idxs]
            shape = MultiPoint([tuple(p) for p in pts]).convex_hull.buffer(pad_vox)
            polys.append(shape)
        regions[z] = polys
    return regions


def merge_regions_over_z_range(
    regions: dict[int, list], z0: int, z1: int
) -> list:
    """Unions every per-slice polygon across z in [z0, z1) into as few merged 2D regions as
    possible -- polygons that overlap or touch (the same clump seen at consecutive z's, or two
    clumps that happen to converge) collapse into one; genuinely separate clumps stay separate.
    This is what actually gets used to size one inference window per z-chunk, not the raw
    per-slice polygons."""
    all_polys = []
    for z in range(z0, z1):
        all_polys.extend(regions.get(z, []))
    if not all_polys:
        return []
    merged = unary_union(all_polys)
    return list(merged.geoms) if hasattr(merged, "geoms") else [merged]
