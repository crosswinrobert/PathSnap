"""Candidate road-segment generation.

For every GPS observation we want the set of arcs it could plausibly belong to.
Method: sample points along every arc (every `sample_spacing_m`), index the samples in a KD-tree,
query a ball around each GPS point, then compute the EXACT point-to-segment distance for the arcs
that were hit. Because consecutive samples are at most `spacing` apart, any point on an arc is within
spacing/2 of a sample, so querying radius + spacing/2 and filtering exactly never misses an arc.

Only arcs near the track bounding box are indexed (networks can have >300k arcs).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import point_segment_dist


class CandidateGenerator:
    def __init__(self, net, track_xy, spacing=15.0, bbox_margin=500.0):
        self.net, self.spacing = net, float(spacing)
        x, y = track_xy[:, 0], track_xy[:, 1]
        x0, x1 = x.min() - bbox_margin, x.max() + bbox_margin
        y0, y1 = y.min() - bbox_margin, y.max() + bbox_margin
        ax, ay, bx, by = net.ax, net.ay, net.bx, net.by
        keep = ~((np.maximum(ax, bx) < x0) | (np.minimum(ax, bx) > x1)
                 | (np.maximum(ay, by) < y0) | (np.minimum(ay, by) > y1))
        self.arc_ids = np.flatnonzero(keep)
        if len(self.arc_ids) == 0:
            self.tree = None
            return
        a = self.arc_ids
        n = np.maximum(np.ceil(net.length[a] / self.spacing), 1).astype(np.int64) + 1   # samples per arc (>=2, also for zero-length self-loops)
        arc_rep = np.repeat(a, n)
        start = np.repeat(np.cumsum(n) - n, n)
        frac = (np.arange(n.sum()) - start) / np.repeat(n - 1, n)
        sx = net.ax[arc_rep] + frac * (net.bx[arc_rep] - net.ax[arc_rep])
        sy = net.ay[arc_rep] + frac * (net.by[arc_rep] - net.ay[arc_rep])
        self.sample_arc = arc_rep
        self.tree = cKDTree(np.column_stack([sx, sy]))

    def _pairs(self, pts, pt_idx, radius, chunk=1500):
        """All (point, arc) pairs with exact distance <= radius."""
        net = self.net
        out_p, out_a, out_d, out_f = [], [], [], []
        for s in range(0, len(pt_idx), chunk):
            idx = pt_idx[s:s + chunk]
            lists = self.tree.query_ball_point(pts[idx], r=radius + self.spacing / 2.0)
            cnt = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
            if cnt.sum() == 0:
                continue
            p = np.repeat(idx, cnt)
            a = self.sample_arc[np.concatenate([np.asarray(l, dtype=np.int64) for l in lists])]
            key = np.unique(p.astype(np.int64) * net.n_arcs + a)          # one row per (point, arc)
            p, a = key // net.n_arcs, key % net.n_arcs
            d, f = point_segment_dist(pts[p, 0], pts[p, 1], net.ax[a], net.ay[a], net.bx[a], net.by[a])
            m = d <= radius
            out_p.append(p[m]); out_a.append(a[m]); out_d.append(d[m]); out_f.append(f[m])
        if not out_p:
            e = np.empty(0)
            return e.astype(np.int64), e.astype(np.int64), e, e
        return (np.concatenate(out_p), np.concatenate(out_a),
                np.concatenate(out_d), np.concatenate(out_f))

    def generate(self, pts, radius, max_candidates=None, fallback_radius=None):
        """Return a DataFrame (pt, arc, dist, frac, rank, is_fallback) sorted by (pt, dist).
        `pt` is the row position in `pts`; `rank` is 1 for the nearest arc of each point."""
        n = len(pts)
        cols = ["pt", "arc", "dist", "frac", "is_fallback"]
        if self.tree is None or n == 0:
            return pd.DataFrame(columns=cols + ["rank"])
        p, a, d, f = self._pairs(pts, np.arange(n), radius)
        df = pd.DataFrame({"pt": p, "arc": a, "dist": d, "frac": f, "is_fallback": 0})
        if fallback_radius and fallback_radius > radius:
            empty = np.setdiff1d(np.arange(n), df["pt"].unique())
            if len(empty):
                p2, a2, d2, f2 = self._pairs(pts, empty, fallback_radius)
                df = pd.concat([df, pd.DataFrame({"pt": p2, "arc": a2, "dist": d2, "frac": f2,
                                                  "is_fallback": 1})], ignore_index=True)
        df = df.sort_values(["pt", "dist", "arc"], kind="stable").reset_index(drop=True)
        df["rank"] = df.groupby("pt").cumcount() + 1
        if max_candidates:
            df = df[df["rank"] <= max_candidates].reset_index(drop=True)
        return df
