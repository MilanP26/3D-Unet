"""Parser for the VAST skeleton export CSV (Phase B real seeds).

`Data/VAST_skeleton_data.csv` has no header row. Column meaning was reverse
engineered and *verified* on 2026-07-09 (see PLAN.md) against the raw binary
`.vsanno` this was exported from -- specifically: every `parent` value was
confirmed to reference another row's local node id within the same tree
(zero exceptions across all 209,892 rows), local node ids are confirmed
contiguous 0..N-1 per tree, and z advances by ~1 per node in the common
case, consistent with "one node per traced slice" (CLAUDE.md). Columns not
listed below (2, 9-15, and the second string field) have no confirmed
meaning yet and are not used.

  0  tree_id           skeleton/neuron id (not necessarily 1..N contiguous
                        globally -- 269 distinct ids observed)
  1  local_id          node index within this tree, contiguous 0..N-1
  3  x, 4 y, 5 z        absolute voxel position in the full EM stack's mip0
                        frame (same frame as `volume.vsvi` / phase_b_stack.py)
  6  parent_local_id   -1 for the tree's root, else another local_id in the
                        same tree
  8  branch_child_local_id  -1 if none, else another local_id in the same
                        tree (a second child at a branch point)
  16 tag               free-text annotator note (e.g. "risky_merge",
                        "potential_merge", "cell_body_and_nerve_ring_exit"),
                        empty for most nodes
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKELETON_CSV = REPO_ROOT / "Data" / "VAST_skeleton_data.csv"


@dataclass
class SkeletonNode:
    tree_id: int
    local_id: int
    x: int
    y: int
    z: int
    parent_local_id: int
    branch_child_local_id: int
    tag: str


def load_skeletons(csv_path: Path = DEFAULT_SKELETON_CSV) -> dict[int, list[SkeletonNode]]:
    """Returns {tree_id: nodes}, each tree's nodes indexed by local_id (i.e.
    `nodes[i].local_id == i` -- relies on the verified contiguity above)."""
    by_tree: dict[int, list[SkeletonNode]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            node = SkeletonNode(
                tree_id=int(row[0]),
                local_id=int(row[1]),
                x=int(row[3]),
                y=int(row[4]),
                z=int(row[5]),
                parent_local_id=int(row[6]),
                branch_child_local_id=int(row[8]),
                tag=row[16],
            )
            by_tree[node.tree_id].append(node)

    for tree_id, nodes in by_tree.items():
        nodes.sort(key=lambda n: n.local_id)
        if [n.local_id for n in nodes] != list(range(len(nodes))):
            raise ValueError(f"Tree {tree_id}: local_ids are not contiguous 0..N-1")
    return dict(by_tree)


def _physical_dist(a: SkeletonNode, b: SkeletonNode, scale_nm_xyz: tuple[float, float, float]) -> float:
    sx, sy, sz = scale_nm_xyz
    dx, dy, dz = (a.x - b.x) * sx, (a.y - b.y) * sy, (a.z - b.z) * sz
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def subsample_seeds(
    nodes: list[SkeletonNode],
    scale_nm_xyz: tuple[float, float, float],
    target_spacing_nm: float = 500.0,
) -> list[SkeletonNode]:
    """Walks the tree from its root (iterative DFS -- trees can have
    thousands of nodes in one branch, deeper than Python's default recursion
    limit) and picks a spaced-out subset of nodes to run inference at,
    instead of every node (CLAUDE.md: running inference at every node would
    be extremely redundant since one patch already spans many consecutive
    slices). A node is picked once the accumulated physical path distance
    since the last pick exceeds `target_spacing_nm`; leaves are always
    picked so neurite tips aren't missed even if short of the threshold."""
    if not nodes:
        return []

    children: dict[int, list[int]] = defaultdict(list)
    for n in nodes:
        if n.parent_local_id != -1:
            children[n.parent_local_id].append(n.local_id)

    roots = [n.local_id for n in nodes if n.parent_local_id == -1]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root (parent_local_id == -1), found {len(roots)}")

    selected: list[SkeletonNode] = [nodes[roots[0]]]
    # Stack entries: (local_id, last_selected_local_id, dist_since_last_selected)
    stack: list[tuple[int, int, float]] = [(roots[0], roots[0], 0.0)]
    while stack:
        local_id, last_selected, dist = stack.pop()
        for child_id in children.get(local_id, []):
            step = _physical_dist(nodes[child_id], nodes[local_id], scale_nm_xyz)
            new_dist = dist + step
            is_leaf = not children.get(child_id)
            if new_dist >= target_spacing_nm or is_leaf:
                selected.append(nodes[child_id])
                stack.append((child_id, child_id, 0.0))
            else:
                stack.append((child_id, last_selected, new_dist))
    return selected
