"""Road network geometry in a trip-local projected frame (metres)."""
import numpy as np

from .geo import LocalProjection, bearing_deg


class RoadNetwork:
    def __init__(self, nodes, arcs, proj: LocalProjection):
        self.proj = proj
        nx_, ny_ = proj.to_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
        u = arcs["u"].to_numpy(dtype=np.int64)
        v = arcs["v"].to_numpy(dtype=np.int64)
        self.n_arcs, self.n_nodes = len(arcs), len(nodes)
        self.ax, self.ay, self.bx, self.by = nx_[u], ny_[u], nx_[v], ny_[v]
        dx, dy = self.bx - self.ax, self.by - self.ay
        self.length = np.hypot(dx, dy)
        self.bearing = bearing_deg(dx, dy)
        out_deg = np.bincount(u, minlength=self.n_nodes)
        in_deg = np.bincount(v, minlength=self.n_nodes)
        self.head_out_deg = out_deg[v]      # choices available after leaving this arc
        self.tail_in_deg = in_deg[u]        # ways of entering this arc
        self.is_dup = arcs.duplicated(keep=False).to_numpy()   # another arc with same (u, v)
