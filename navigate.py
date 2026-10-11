"""Path planning on the LiDAR map (home_map_clean.pcd).

Builds a 2D occupancy grid from the point cloud, inflates obstacles by the
robot radius, and plans with A*. Map frame: origin = where mapping started.

Usage:
    uv run python navigate.py --start 0 0 --goal -3 -2      # coordinates in metres
    uv run python navigate.py                               # click start, then goal
    uv run python navigate.py --map home_map.pcd --robot-radius 0.2

Output: waypoints printed (x, y in metres) + saved to nav_path.png / nav_path.csv.
"""
import argparse
import heapq
import math

import matplotlib
import numpy as np
import open3d as o3d

RES = 0.05  # grid cell size, metres

FREE, UNKNOWN, OCC = 0, 1, 2


class GridMap:
    def __init__(self, pcd_file, robot_radius=0.15, origin=(0.0, 0.0), min_range=0.35, points=None):
        """Build from a .pcd file, or straight from an (N, 2/3) array of map points (live map)."""
        pts = (np.asarray(points)[:, :2] if points is not None
               else np.asarray(o3d.io.read_point_cloud(pcd_file).points)[:, :2])
        # points hugging the origin are the robot's own body, not obstacles
        pts = pts[np.hypot(*(pts - origin).T) > min_range]
        pad = 0.5
        self.min = pts.min(0) - pad
        size = np.ceil((pts.max(0) + pad - self.min) / RES).astype(int)
        self.w, self.h = int(size[0]), int(size[1])

        self.grid = np.full((self.h, self.w), UNKNOWN, dtype=np.uint8)

        # cells between the start position and each hit point are known free
        o = self.to_cell(*origin)
        for p in pts:
            c = self.to_cell(*p)
            for cx, cy in self._line(o, c)[:-1]:
                if self.inside(cx, cy):
                    self.grid[cy, cx] = FREE

        occ = np.zeros((self.h, self.w), dtype=bool)
        for p in pts:
            cx, cy = self.to_cell(*p)
            if self.inside(cx, cy):
                occ[cy, cx] = True
        occ = self._drop_specks(occ)
        self.occ_raw = occ

        # inflate obstacles by the robot radius (disc dilation, numpy only)
        self.radius = robot_radius
        inflated = self.inflate(occ)
        self.static_blocked = inflated          # walls etc., inflated by the robot radius
        self.zones_raw = np.zeros_like(occ)     # user-drawn no-go zones
        self.zones = np.zeros_like(occ)         # ...inflated by the robot radius
        self.blocked = inflated.copy()          # what the planner may not enter
        self.grid[inflated] = OCC
        self._label_regions()

    @staticmethod
    def _drop_specks(occ, min_support=3, win=2):
        """Remove isolated occupied cells (stray scan-matching dots). A real wall cell has
        neighbours within a (2*win+1)^2 window; an isolated dot does not."""
        h, w = occ.shape
        pad = np.pad(occ.astype(np.uint8), win)
        cnt = sum(pad[dy:dy + h, dx:dx + w] for dy in range(2 * win + 1) for dx in range(2 * win + 1))
        return occ & (cnt >= min_support)       # cnt includes the cell itself

    def set_zones(self, polys):
        """Rasterise no-go polygons (lists of (x, y) metres) and make the planner avoid them."""
        from matplotlib.path import Path
        raw = np.zeros((self.h, self.w), dtype=bool)
        if polys:
            ys, xs = np.mgrid[0:self.h, 0:self.w]
            centres = np.column_stack([(self.min[0] + (xs + .5) * RES).ravel(), (self.min[1] + (ys + .5) * RES).ravel()])
            for poly in polys:
                if len(poly) >= 3:
                    raw |= Path(poly).contains_points(centres).reshape(self.h, self.w)
        self.zones_raw = raw
        self.zones = self.inflate(raw) if raw.any() else raw
        self.blocked = self.static_blocked | self.zones
        self._label_regions()

    def hits_zone(self, a, b):
        """True if the straight line a->b (world coords) touches a no-go zone (incl. robot margin)."""
        if not self.zones.any():
            return False
        return any(self.zones[y, x] for x, y in self._line(self.to_cell(*a), self.to_cell(*b))
                   if self.inside(x, y))

    def in_zone(self, p):
        cx, cy = self.to_cell(*p)
        return self.inside(cx, cy) and bool(self.zones_raw[cy, cx])

    def _label_regions(self):
        """Label connected free regions; keep the largest as the navigable one."""
        label = np.zeros((self.h, self.w), dtype=np.int32)
        sizes, n = {}, 0
        for sy in range(self.h):
            for sx in range(self.w):
                if self.blocked[sy, sx] or label[sy, sx]:
                    continue
                n += 1
                label[sy, sx] = n
                stack, cnt = [(sx, sy)], 0
                while stack:
                    x, y = stack.pop()
                    cnt += 1
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if self.inside(nx, ny) and not self.blocked[ny, nx] and not label[ny, nx]:
                            label[ny, nx] = n
                            stack.append((nx, ny))
                sizes[n] = cnt
        self.main = label == max(sizes, key=sizes.get)

    def inflate(self, mask, extra=0.0):
        """Grow True cells of `mask` by the robot radius (+ extra metres)."""
        r = int(math.ceil((self.radius + extra) / RES))
        out = np.zeros_like(mask)
        ys, xs = np.nonzero(mask)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    yy, xx = ys + dy, xs + dx
                    ok = (yy >= 0) & (yy < self.h) & (xx >= 0) & (xx < self.w)
                    out[yy[ok], xx[ok]] = True
        return out

    # ── coordinate helpers ──
    def to_cell(self, x, y):
        return int((x - self.min[0]) / RES), int((y - self.min[1]) / RES)

    def to_world(self, cx, cy):
        return self.min[0] + (cx + 0.5) * RES, self.min[1] + (cy + 0.5) * RES

    def inside(self, cx, cy):
        return 0 <= cx < self.w and 0 <= cy < self.h

    @staticmethod
    def _line(a, b):
        """Bresenham line between two cells."""
        (x0, y0), (x1, y1) = a, b
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err, out = dx + dy, []
        while True:
            out.append((x0, y0))
            if (x0, y0) == (x1, y1):
                return out
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def line_clear(self, a, b, blk=None):
        blk = self.blocked if blk is None else blk
        return all(not blk[y, x] for x, y in self._line(a, b))

    # ── planning ──
    def nearest_free(self, cell, max_r=40):
        """If a cell is blocked (e.g. start hugging a wall), find the closest usable one."""
        if self.inside(*cell) and self.main[cell[1], cell[0]]:
            return cell
        for r in range(1, max_r):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    c = (cell[0] + dx, cell[1] + dy)
                    if self.inside(*c) and self.main[c[1], c[0]]:
                        return c
        return None

    def plan(self, start, goal, extra=None):
        """A* in world coords -> list of (x, y) waypoints, or None.
        `extra` = optional boolean mask of temporary obstacles (already inflated)."""
        s = self.nearest_free(self.to_cell(*start))
        g = self.nearest_free(self.to_cell(*goal))
        if s is None or g is None:
            return None
        blk = self.blocked
        if extra is not None:
            blk = self.blocked | extra
            # never let a temporary obstacle swallow the cell the car is standing on
            blk = blk.copy()
            blk[max(0, s[1] - 6):s[1] + 7, max(0, s[0] - 6):s[0] + 7] = self.blocked[max(0, s[1] - 6):s[1] + 7, max(0, s[0] - 6):s[0] + 7]

        pen = getattr(self, "penalty", None)     # optional extra cost, e.g. "too close to a wall"

        def cost(c):  # prefer known-free space over unexplored space
            base = 1.0 if self.grid[c[1], c[0]] == FREE else 4.0
            return base if pen is None else base + pen[c[1], c[0]]

        def h(c):
            return math.hypot(c[0] - g[0], c[1] - g[1])

        moves = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        openq = [(h(s), 0.0, s)]
        best, came = {s: 0.0}, {}
        while openq:
            _, gc, cur = heapq.heappop(openq)
            if cur == g:
                break
            if gc > best.get(cur, math.inf):
                continue
            for dx, dy, step in moves:
                n = (cur[0] + dx, cur[1] + dy)
                if not self.inside(*n) or blk[n[1], n[0]]:
                    continue
                if dx and dy and (blk[cur[1], n[0]] or blk[n[1], cur[0]]):
                    continue  # no cutting corners
                ng = gc + step * cost(n)
                if ng < best.get(n, math.inf):
                    best[n], came[n] = ng, cur
                    heapq.heappush(openq, (ng + h(n), ng, n))
        else:
            return None
        if g not in came and g != s:
            return None

        path = [g]
        while path[-1] != s:
            path.append(came[path[-1]])
        path.reverse()
        return [self.to_world(*c) for c in self._smooth(path, blk)]

    def _smooth(self, path, blk=None):
        """String-pulling: keep only waypoints needed to avoid obstacles."""
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.line_clear(path[i], path[j], blk):
                j -= 1
            out.append(path[j])
            i = j
        return out


def pick_points(gm, pcd_pts):
    """Click start then goal on a matplotlib window."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_map(ax, gm)
    ax.set_title("Click START, then GOAL")
    clicks = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(clicks) < 2:
        raise SystemExit("Need two clicks")
    return clicks[0], clicks[1]


def draw_map(ax, gm):
    ext = [gm.min[0], gm.min[0] + gm.w * RES, gm.min[1], gm.min[1] + gm.h * RES]
    img = np.full((gm.h, gm.w), 0.6)
    img[gm.grid == FREE] = 1.0
    img[gm.blocked] = 0.82
    img[gm.occ_raw] = 0.0
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", extent=ext)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.plot(0, 0, "r+", ms=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="home_map_clean.pcd")
    ap.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--robot-radius", type=float, default=0.15, help="metres (default 0.15)")
    args = ap.parse_args()

    gm = GridMap(args.map, args.robot_radius)
    print(f"Grid {gm.w}x{gm.h} cells @ {RES*100:.0f} cm, "
          f"x∈[{gm.min[0]:.1f},{gm.min[0]+gm.w*RES:.1f}] y∈[{gm.min[1]:.1f},{gm.min[1]+gm.h*RES:.1f}]")

    if args.start and args.goal:
        start, goal = args.start, args.goal
    else:
        start, goal = pick_points(gm, None)

    path = gm.plan(start, goal)
    if path is None:
        raise SystemExit("No path found (goal enclosed or outside the map?)")

    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    print(f"Path: {len(path)} waypoints, {length:.2f} m")
    for i, (x, y) in enumerate(path):
        print(f"  {i:2d}: ({x:6.2f}, {y:6.2f})")
    np.savetxt("nav_path.csv", path, delimiter=",", header="x_m,y_m", comments="")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_map(ax, gm)
    px, py = zip(*path)
    ax.plot(px, py, "c-o", ms=4, lw=2, label="path")
    ax.plot(*start, "go", ms=10, label="start")
    ax.plot(*goal, "r*", ms=14, label="goal")
    ax.legend()
    fig.savefig("nav_path.png", dpi=120, bbox_inches="tight")
    print("Saved nav_path.png and nav_path.csv")


if __name__ == "__main__":
    main()
