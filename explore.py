"""Autonomous exploration helpers: an occupancy grid built from the LiDAR while driving,
frontier picking, a planner grid for GridMap's A*, and a rough camera 'ledge ahead' check.

Frontier = a known-free cell next to a never-seen cell. Driving to one and looking again
reveals what is behind it; when no reachable frontier is left, the area is mapped."""
import io
import math

import numpy as np
from scipy import ndimage

from navigate import FREE, OCC, RES, UNKNOWN, GridMap

FREE_RANGE = 3.0         # rays are carved as free up to this far (beam spacing stays below one cell)
PREFER_CLEAR = 0.50      # metres: paths are pushed away from walls within this distance
MIN_FRONTIER = 4         # smaller frontier clusters (cells) are noise
LEDGE_THRESHOLD = 35.0   # row-to-row brightness jump in the floor ahead; carpet scenes measured about 15
_CROSS = ndimage.generate_binary_structure(2, 1)
_BOX = np.ones((3, 3), bool)


class Occupancy:
    """Fixed-size grid centred on the start, filled from successive scans."""

    def __init__(self, size=30.0):
        self.n = int(size / RES)
        self.min = np.array([-size / 2, -size / 2])
        self.free = np.zeros((self.n, self.n), np.uint8)
        self.hit = np.zeros((self.n, self.n), np.uint8)

    def _idx(self, xy):
        ij = np.floor((xy - self.min) / RES).astype(int)
        ok = (ij >= 0).all(-1) & (ij < self.n).all(-1)
        return ij[..., 0], ij[..., 1], ok

    def update(self, pose_xy, pts_world):
        """pts_world: (N, 2) scan points in map coordinates, taken from pose_xy."""
        p0 = np.asarray(pose_xy, float)
        d = pts_world - p0
        r = np.hypot(d[:, 0], d[:, 1])
        keep = r > 0.05
        d, r = d[keep], r[keep]
        hit = r <= FREE_RANGE
        reach = np.minimum(r, FREE_RANGE) - np.where(hit, RES, 0.0)       # stop a cell short of a wall
        reach = np.maximum(reach, 0.0)
        steps = int(FREE_RANGE / (RES * 0.7)) + 1
        t = np.linspace(0.0, 1.0, steps)
        samples = p0 + (d / r[:, None])[:, None, :] * (reach[:, None] * t[None, :])[..., None]
        ix, iy, ok = self._idx(samples)
        ix, iy = ix[ok], iy[ok]
        self.free[iy, ix] = np.minimum(self.free[iy, ix] + 1, 250)
        hx, hy, hok = self._idx(p0 + d[hit])
        hx, hy = hx[hok], hy[hok]
        self.hit[hy, hx] = np.minimum(self.hit[hy, hx] + 1, 250)

    def classify(self):
        """-> (known_free, occupied, unknown) boolean grids."""
        occ = (self.hit >= 2) & (self.hit * 4 >= self.free)
        occ = GridMap._drop_specks(occ)                                   # stray single dots
        free = ndimage.binary_closing(self.free > 0, _BOX) & ~occ        # fill ray gaps
        wall_skin = ndimage.binary_dilation(self.hit > 0, _BOX)          # a wall's own cells and their neighbours were seen
        return free, occ, ~(free | occ | wall_skin)


class ExploreGrid(GridMap):
    """GridMap built from an Occupancy: only cells the LiDAR has seen free are drivable."""

    def __init__(self, occ, polys, radius, car_xy):
        self.min = occ.min.copy()
        self.w = self.h = occ.n
        self.radius = radius
        free, hit, unknown = occ.classify()
        self.known_free, self.unknown = free, unknown
        self.occ_raw = hit
        inflated = self.inflate(hit)
        self.static_blocked = inflated | ~free
        self.grid = np.full((self.h, self.w), UNKNOWN, dtype=np.uint8)
        self.grid[free] = FREE
        self.grid[inflated] = OCC
        self.zones_raw = np.zeros_like(hit)
        self.zones = np.zeros_like(hit)
        self.blocked = self.static_blocked.copy()
        self.car_cell = self.to_cell(*car_xy)
        # soft cost: keep to the middle of corridors instead of hugging walls
        dist = ndimage.distance_transform_edt(~hit) * RES
        self.penalty = (np.clip((PREFER_CLEAR - dist) / PREFER_CLEAR, 0, 1) * 6.0).astype(np.float32)
        self.set_zones(polys)                                             # also labels the regions

    def _label_regions(self):
        lab, n = ndimage.label(~self.blocked)
        self.main = np.zeros_like(self.blocked)
        if n == 0:
            return
        cx, cy = self.car_cell
        win = lab[max(0, cy - 8):cy + 9, max(0, cx - 8):cx + 9]
        ids = win[win > 0]
        pick = np.bincount(ids).argmax() if len(ids) else np.bincount(lab[lab > 0]).argmax()
        self.main = lab == pick

    def frontiers(self, max_dist=None, origin=(0.0, 0.0)):
        """[(cluster_size, (x, y))] for reachable frontier clusters, representative = member nearest the centroid."""
        edge = self.known_free & ndimage.binary_dilation(self.unknown, _CROSS)
        edge &= self.main
        if max_dist is not None:
            ys, xs = np.nonzero(edge)
            far = np.hypot(self.min[0] + (xs + .5) * RES - origin[0], self.min[1] + (ys + .5) * RES - origin[1]) > max_dist
            edge[ys[far], xs[far]] = False
        lab, n = ndimage.label(edge, structure=_BOX)
        out = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if len(xs) < MIN_FRONTIER:
                continue
            k = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
            out.append((len(xs), self.to_world(xs[k], ys[k])))
        return out


def truncate(path, max_len):
    """Keep the start of a path up to max_len metres (cutting inside a segment if needed)."""
    out, total = [path[0]], 0.0
    for a, b in zip(path, path[1:]):
        seg = math.dist(a, b)
        if total + seg >= max_len:
            f = (max_len - total) / seg if seg else 0.0
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
            return out
        out.append(b)
        total += seg
    return out


def ledge_score(jpg):
    """Biggest row-to-row brightness jump in the floor strip straight ahead of the camera.
    A floor that suddenly ends (stairs down, a step) produces one sharp horizontal edge there.
    Rough: patterned rugs and door thresholds can trip it too, which only costs a no-go box."""
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(jpg)).convert("L").resize((160, 120)), float)
    rows = a[66:120, 40:120].mean(1)
    rows = np.convolve(rows, [1 / 3] * 3, "valid")
    return float(np.abs(np.diff(rows)).max())
