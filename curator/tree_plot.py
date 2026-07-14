"""Lineage tree plot (extracted, uses the shared lineage classifier).

Performance notes
-----------------
The nodes are drawn with a SINGLE vectorized ``scatter`` call rather than one
``scatter`` per node: on a dense dataset (thousands of cells) the per-node
version took minutes (each call builds a separate matplotlib artist), which
showed up as the viewer hanging when "Show lineage tree plot" was clicked.

A tree with thousands of nodes is also illegible, so by default only the
largest families (those with the most members, i.e. real division trees) are
drawn, up to ``max_families`` / ``max_nodes``. Single-cell "families" (flagged
but unlinked) are excluded by default since they carry no genealogy; pass
``include_singles=True`` to show them too.
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt

from . import lineage
from .config import COL_TRACK, COL_FRAME


def lineage_tree_figure(df, max_families=60, max_nodes=400,
                        include_singles=False):
    """Draw cell lineage trees, biggest families first.

    Parameters
    ----------
    max_families : int
        Cap on how many families (distinct roots) to draw, largest first.
    max_nodes : int
        Hard cap on total nodes drawn; families are added largest-first until
        this is reached, so the figure stays legible and fast.
    include_singles : bool
        Include one-cell families (flagged but with no parent/children).
    Returns the matplotlib Figure, or None if there is nothing to draw.
    """
    groups = lineage.classify_lineage(df)
    if not groups.all_nodes:
        return None
    children_of = groups.children_of

    # Build each family (root -> set of member nodes) so we can rank by size.
    p_of = lineage.parent_of(df)
    root_cache: dict[int, int] = {}
    members_by_root: dict[int, set] = {}
    for node in groups.all_nodes:
        r = lineage._root_in_map(p_of, int(node), root_cache)
        members_by_root.setdefault(r, set()).add(int(node))

    # Optionally drop single-cell families (no genealogy to show).
    families = [(r, m) for r, m in members_by_root.items()
                if include_singles or len(m) > 1]
    if not families:
        # Everything is a single; fall back to showing singles so the user sees
        # something rather than an empty plot.
        families = list(members_by_root.items())
    # Largest families first.
    families.sort(key=lambda rm: len(rm[1]), reverse=True)

    # Select families up to the node/family caps.
    selected_roots = []
    total = 0
    for r, m in families:
        if len(selected_roots) >= max_families or total + len(m) > max_nodes:
            break
        selected_roots.append(r)
        total += len(m)
    if not selected_roots:  # the very first family already exceeds max_nodes
        selected_roots = [families[0][0]]
    n_families_total = len(families)
    truncated = len(selected_roots) < n_families_total

    # Layout only the selected families.
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    pos_x, pos_y, visited = {}, {}, set()
    leaf_x = [0]

    def layout_node(node, depth):
        if node in visited:
            return
        visited.add(node)
        pos_y[node] = -depth
        children = children_of.get(node, [])
        if not children:
            pos_x[node] = leaf_x[0]
            leaf_x[0] += 1
        else:
            for c in children:
                layout_node(c, depth + 1)
            xs = [pos_x[c] for c in children if c in pos_x]
            pos_x[node] = sum(xs) / len(xs) if xs else leaf_x[0]
            if not xs:
                leaf_x[0] += 1

    for r in selected_roots:
        layout_node(r, 0)
        leaf_x[0] += 1

    drawn = [n for n in pos_x]  # nodes actually positioned
    fig, ax = plt.subplots(figsize=(12, 8))
    title = "Cell lineages"
    if truncated:
        title += f" (showing {len(selected_roots)} of {n_families_total} families, biggest first)"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    # Edges: collect all segments, draw with one LineCollection (fast).
    from matplotlib.collections import LineCollection
    segments = []
    for p, children in children_of.items():
        if p not in pos_x:
            continue
        for c in children:
            if c in pos_x:
                segments.append([(pos_x[p], pos_y[p]), (pos_x[c], pos_y[c])])
    if segments:
        ax.add_collection(LineCollection(segments, colors="gray",
                                         linewidths=2.0, zorder=1))

    # Nodes: ONE vectorized scatter for all of them (the key speedup).
    if drawn:
        xs = np.array([pos_x[n] for n in drawn], dtype=float)
        ys = np.array([pos_y[n] for n in drawn], dtype=float)
        # Scale marker size down as the node count grows, so a big tree stays
        # readable instead of a wall of overlapping blobs.
        s = 1200 if len(drawn) <= 60 else (500 if len(drawn) <= 200 else 150)
        ax.scatter(xs, ys, s=s, c="#ADD8E6", edgecolors="#2C3E50",
                   linewidth=2.0, zorder=2)
        # Labels: the genealogy path (1 -> 1.1, 1.2 -> 1.1.1 ...) derived from
        # parent_id, NOT the raw track_id. This is purely a display translation;
        # IDs and the mask are never rewritten. The real track_id is shown small
        # underneath (when the tree is small enough to read) so cells can still be
        # located in the curator.
        hlabels = lineage.hierarchical_labels(df)
        if len(drawn) <= 200:
            fs = 12 if len(drawn) <= 60 else 8
            show_ids = len(drawn) <= 60
            for n in drawn:
                ax.text(pos_x[n], pos_y[n], hlabels.get(n, str(n)),
                        ha="center", va="center", fontsize=fs,
                        fontweight="bold", color="#1A252F", zorder=3)
                if show_ids:
                    ax.text(pos_x[n], pos_y[n] - 0.28, f"id {n}", ha="center",
                            va="center", fontsize=6, color="#5D6D7E", zorder=3)

    # Set limits explicitly (LineCollection/scatter don't autoscale reliably).
    if drawn:
        ax.set_xlim(min(xs) - 1, max(xs) + 1)
        ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.axis("off")
    fig.tight_layout()
    return fig
