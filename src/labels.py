"""Per-observation reference labels from the supplied reference route.

The .route file lists ARC IDS in travel order but there is NO per-GPS-point label (a trip has ~2 600
GPS points and ~390 route arcs on average). We derive the label by a monotone alignment:

    choose, for every GPS point i, a route position j(i) with j(1) <= j(2) <= ... <= j(N)
    minimising   sum_i  min(dist(point_i, arc[route[j(i)]]), clip)  +  penalty * (number of forward jumps)

solved exactly by dynamic programming in O(N*M) (prefix minima give the best earlier position).
Monotonicity matters: the vehicle cannot go back along its own route, which stops a point near a
junction from being assigned to a later/earlier arc that merely happens to be close.
These labels are used for TRAINING and EVALUATION ONLY; they are never an input at inference time.
"""
import numpy as np

from .geo import point_segment_dist


def align_labels(net, route, track_xy, jump_penalty=10.0, clip=200.0):
    n = len(track_xy)
    if route is None or len(route) == 0 or n == 0:
        return (np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64),
                np.full(n, np.nan))
    route = np.asarray(route, dtype=np.int64)
    m = len(route)
    ax, ay, bx, by = net.ax[route], net.ay[route], net.bx[route], net.by[route]
    idx = np.arange(m)

    def row(i):
        return np.minimum(point_segment_dist(track_xy[i, 0], track_xy[i, 1], ax, ay, bx, by)[0], clip)

    back = np.zeros((n, m), dtype=np.int16 if m < 32000 else np.int32)
    dp = row(0)
    for i in range(1, n):
        pm = np.minimum.accumulate(dp)                       # best dp[k], k <= j
        rec = np.where(dp <= pm, idx, 0)
        arg = np.maximum.accumulate(rec)                     # argmin of that prefix
        jump_cost = np.concatenate(([np.inf], pm[:-1])) + jump_penalty   # best dp[k], k < j, plus penalty
        jump_from = np.concatenate(([0], arg[:-1]))
        stay = dp <= jump_cost
        back[i] = np.where(stay, idx, jump_from)
        dp = row(i) + np.where(stay, dp, jump_cost)
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(dp))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    arc = route[pos]
    d = point_segment_dist(track_xy[:, 0], track_xy[:, 1], net.ax[arc], net.ay[arc],
                           net.bx[arc], net.by[arc])[0]
    return pos, arc, d
