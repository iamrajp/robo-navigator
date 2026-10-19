"""Web UI: click a point on the LiDAR map, press Go, the car drives there.

    uv run python nav_server.py              # real car + LiDAR localisation
    uv run python nav_server.py --sim        # no hardware, simulated robot

Drives the motors directly via Robo.py (gpiozero) - no separate car-server needed.
Place the car at the map origin (where mapping started, facing the way you
started) before launching: the pose starts at (0,0,0) and is then tracked by
matching each LiDAR scan against the map (ICP).

The car has no encoders, so driving is closed-loop through the LiDAR pose:
short pulse -> stop -> re-measure -> re-aim.  Tune the PULSE_* constants.
"""
import argparse
import base64
import json
import glob
import math
import os
import re
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


def _load_env(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Load KEY=VALUE lines from .env into os.environ (real env vars win, blanks are skipped)."""
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip("\"'")
        if k and v:
            os.environ.setdefault(k, v)


_load_env()  # before llm_brain is imported: it reads LLM_* settings at import time

from navigate import RES, GridMap
import explore as ex
import experiment
import objects as obj_mod
import push

ARRIVE_M = 0.15          # waypoint reached within this distance
AIM_TOL = math.radians(12)
MAP_FIT_MIN = 0.55      # mapping is stricter than navigating
MAP_FIT_STRICT = 0.70   # ...and stricter still while exploring on its own
EXPLORE_MOVE_T = 0.25    # shorter pushes / turns while exploring: smaller steps keep the scan matcher on track
EXPLORE_TURN_T = 0.20
SETTLE_S = 1.5           # after a back-up or recovery turn, hold off adding scans to the map this long
NAV_TURN_SPEED = 0.30    # turn speed set at start-up (0.4 is brisk); the page sliders override it
TURN_RATE0 = math.radians(45)   # first guess of turn rate (rad/s); learned from measurements
BODY_R = 0.22            # LiDAR points closer than this are the car itself (log showed chassis hits out to ~0.17 m)
OBST_AHEAD = 0.50        # an obstacle this close in front triggers a reroute
DETOUR_MARGIN = 0.07     # detours keep this much more clearance than the planner's robot radius
LEARN_FIT = 0.80         # only learn from scans matched at least this well
LEARN_PAUSE = 1.0        # the car must sit still this long before the map is updated (moving scans smear walls)
LEARN_EVERY = 0.5        # seconds between map-learning steps
LEARN_ADD_N = 3          # a new spot must be seen in this many learning steps before it joins the map
LEARN_DEL_N = 4          # ...and an old point must be contradicted this many times before it leaves
LEARN_NEW_DIST = 0.12    # scan points farther than this from every map point count as new
NAV_MIN_POINTS = 30       # fewest usable scan points to try localising
LOST_SCANS = 3           # this many bad matches in a row = tracking lost
RELOC_FIT_MIN = 0.5      # a global-search result must match at least this well
FIT_MIN = 0.4            # ICP fitness below this = pose not trusted
SIM_SPEED = 0.25
EXPLORE_CLEARANCE = 0.12 # extra gap to walls while exploring (added to the robot radius)
EXPLORE_LEG = 1.5        # replan after at most this much driving, so the map and hazard checks stay fresh
EXPLORE_FAILS = 5        # consecutive unreachable / failed targets before giving up
MAX_HAZARDS = 6          # ledge alarms in one run before exploring stops itself
SEEN_R = 0.7            # a frontier this close to somewhere the car already stood has been looked at: skip it
BAD_TARGET_R = 0.6       # a failed frontier blocks other targets this close to it

state = {"pose": [0.0, 0.0], "yaw": 0.0, "path": [], "moving": False,
         "stop": False, "arrived": False, "zone_hit": False, "sim_speed": [0.3, 0.3], "wp": 1, "msg": "Click the map to choose a destination", "lost": False}
gm = None
gm_radius = 0.15
MAP_DIR = "maps"
current_map = [""]


def list_maps():
    """Saved maps live in maps/; any .pcd in the project folder is offered too."""
    found = sorted(glob.glob(f"{MAP_DIR}/*.pcd")) + sorted(glob.glob("*.pcd"))
    return found


loc = None
car = None


drive_t = [0.0]


def watchdog():
    """Manual driving must be refreshed by the browser; if it goes silent, stop the car."""
    while True:
        time.sleep(0.1)
        if drive_t[0] and time.time() - drive_t[0] > 0.4:
            drive_t[0] = 0.0
            if car:
                car.send("S")


def log(msg):
    """Print and append to nav.log so a failed run can be examined afterwards."""
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    try:
        with open("nav.log", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def parse_scan_packets(buf):
    """Decode RPLidar standard-scan packets from a bytearray, consuming what it uses.
    Every packet is validated (start bit and its inverse differ, check bit set, angle <= 360),
    and on a bad packet we slide forward ONE byte, so a single dropped byte re-syncs by itself
    instead of turning everything after it into garbage."""
    out = []
    while len(buf) >= 5:
        b0, b1, b2, b3, b4 = buf[:5]
        angle = ((b1 >> 1) + (b2 << 7)) / 64.0
        if ((b0 & 1) ^ ((b0 >> 1) & 1)) == 1 and (b1 & 1) == 1 and angle <= 360.0:
            out.append((bool(b0 & 1), b0 >> 2, angle, (b3 + (b4 << 8)) / 4.0))
            del buf[:5]
        else:
            del buf[0]
    return out


grid_ver = [0]      # bumped each time the planner grid is rebuilt from the live map


# ── no-go zones (saved next to the map: maps/foo.pcd -> maps/foo.zones.json) ──
zones = []          # [{"id": int, "pts": [[x, y], ...]}]
zones_file = [""]


def zones_path(map_file):
    return os.path.splitext(map_file)[0] + ".zones.json"


def load_zones(map_file):
    global zones
    zones_file[0] = zones_path(map_file)
    try:
        with open(zones_file[0]) as f:
            zones = json.load(f)
    except (OSError, ValueError):
        zones = []
    load_photos(map_file)
    load_objects(map_file)
    load_places(map_file)


# ── named objects (bed, desk, stairs... placed by camera + LiDAR: maps/foo.pcd -> foo.objects.json) ──
map_objects = []    # [{"id", "label", "x", "y", "seen", "t"}] for the loaded map; while mapping they live in explorer.objects
obj_lock = threading.Lock()
obj_busy = threading.Lock()
obj_last = [None]   # (x, y, heading) of the last auto-scan


def load_objects(map_file):
    try:
        with open(os.path.splitext(map_file)[0] + ".objects.json") as f:
            map_objects[:] = json.load(f)
    except (OSError, ValueError):
        map_objects[:] = []


def save_objects():
    try:
        with open(photos_base[0] + ".objects.json", "w") as f:
            json.dump(map_objects, f)
    except OSError as e:
        log(f"[objects] could not save: {e}")


def object_pool():
    return explorer.objects if loc and loc.mode == "map" else map_objects


def scan_objects(force=False):
    """Snapshot camera frame + pose + LiDAR sweep now (call while the car is stopped); the vision call runs in the background.
    Returns "" when started, otherwise why not."""
    if not loc or not camera or camera.frame is None:
        return "camera or LiDAR not available"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY not set"
    if nav.fwd_off is None:
        return "drive forward once first so the car learns which way is front"
    x, y, yaw, _ = loc.pose()
    h = nav.heading(yaw)
    last = obj_last[0]
    if not force and last and math.dist(last[:2], (x, y)) < 0.4 and abs(wrap(h - last[2])) < 0.6:
        return "already looked from here"
    if not obj_busy.acquire(blocking=False):
        return "still identifying the previous view"
    obj_last[0] = (x, y, h)
    threading.Thread(target=_scan_worker, args=(camera.frame, loc.live_points(), (x, y, yaw), nav.fwd_off, nav.turn_sign,
                                                   object_pool()), daemon=True).start()
    return ""


def _scan_worker(frame, pts, pose, fwd_off, left_sign, pool):
    try:
        found = obj_mod.detect(frame)
        added = []
        for o in found:
            hit = obj_mod.place(o["x_frac"], pts, pose, fwd_off, left_sign)
            if hit is None:
                continue                          # no LiDAR surface that way: can't say where it is
            with obj_lock:
                m = obj_mod.merge(pool, o["label"], hit[0], hit[1], time.strftime("%H:%M:%S %d %b"))
            if m:
                added.append(f"{o['label']}@({hit[0]:.1f},{hit[1]:.1f})")
        if pool is map_objects:
            save_objects()
        log(f"[objects] saw {len(found)}, placed: {', '.join(added) or 'none'}")
    except Exception as e:
        log(f"[objects] identify failed: {e!r}")
    finally:
        obj_busy.release()


def objects_line(pool):
    return "Objects on map: " + ("; ".join(f"{o['label']} at ({o['x']:.1f},{o['y']:.1f})" for o in pool[-15:]) or "none") + "."


# ── named places (pins you drop on the map with a name + description; the LLM can drive to them: foo.pcd -> foo.places.json) ──
places = []         # [{"id": int, "name": str, "description": str, "instructions": str, "icon": str, "img": int|None, "x": m, "y": m, "t": "HH:MM:SS date"}]


def load_places(map_file):
    try:
        with open(os.path.splitext(map_file)[0] + ".places.json") as f:
            places[:] = json.load(f)
    except (OSError, ValueError):
        places[:] = []


def save_places():
    try:
        with open(photos_base[0] + ".places.json", "w") as f:
            json.dump(places, f, indent=1)
    except OSError as e:
        log(f"[places] could not save: {e}")


def place_img_file(pid):
    return f"{photos_base[0]}.places/{int(pid)}.jpg"


def set_place_image(pl, data):
    """Apply an uploaded picture (data-URL jpeg from the browser) or a removal request to a place dict."""
    img = data.get("image")
    if img:
        try:
            raw = base64.b64decode(str(img).split(",", 1)[-1], validate=False)
            if len(raw) > 2_000_000 or raw[:2] != b"\xff\xd8":
                return "Picture must be a JPEG under 2 MB"
            os.makedirs(photos_base[0] + ".places", exist_ok=True)
            with open(place_img_file(pl["id"]), "wb") as f:
                f.write(raw)
            pl["img"] = int(time.time())
        except (OSError, ValueError) as e:
            return f"Could not save the picture: {e}"
    elif data.get("clear_image"):
        pl["img"] = None
        try:
            os.remove(place_img_file(pl["id"]))
        except OSError:
            pass


def find_place(name):
    """Case-insensitive exact match first, then a unique substring match. -> place dict or None."""
    n = str(name).strip().lower()
    hit = [p for p in places if p["name"].lower() == n] or [p for p in places if n and (n in p["name"].lower() or p["name"].lower() in n)]
    return hit[0] if len(hit) == 1 or (hit and hit[0]["name"].lower() == n) else None


def places_line():
    return "Named places (go there with go_to_place; follow any [on arrival] instructions once you get there): " + (
        "; ".join(f"\"{p['name']}\"" + (f" ({p['description']})" if p.get("description") else "") + f" at ({p['x']:.1f},{p['y']:.1f})" + (f" [on arrival: {p['instructions']}]" if p.get("instructions") else "")
                  for p in places[-20:]) or "none") + "."


# ── photo pins (camera snapshots pinned on the map: maps/foo.pcd -> foo.photos.json + foo.photos/<id>.jpg) ──
photos = []         # [{"id": int, "x": m, "y": m, "yaw": rad, "t": "HH:MM:SS date"}]
photos_base = [""]  # maps/foo (no extension)


def load_photos(map_file):
    global photos
    photos_base[0] = os.path.splitext(map_file)[0]
    try:
        with open(photos_base[0] + ".photos.json") as f:
            photos = json.load(f)
    except (OSError, ValueError):
        photos = []


def save_photos():
    try:
        with open(photos_base[0] + ".photos.json", "w") as f:
            json.dump(photos, f)
    except OSError as e:
        log(f"[photos] could not save: {e}")


def photo_file(pid):
    return f"{photos_base[0]}.photos/{int(pid)}.jpg"


def save_zones():
    try:
        with open(zones_file[0], "w") as f:
            json.dump(zones, f)
    except OSError as e:
        log(f"[zones] could not save: {e}")


def build_gm(file=None, points=None):
    """A planner grid with the current no-go zones applied."""
    g = GridMap(file, gm_radius, points=points)
    g.set_zones([z["pts"] for z in zones])
    return g


def grid_refresher():
    """Rebuild the planner's grid from the live map a few seconds after it changes."""
    global gm
    built = -1
    while True:
        time.sleep(3)
        if loc is None or loc.mode != "nav" or loc.map_version == built:
            continue
        built = loc.map_version
        try:
            gm = build_gm(points=loc.live_map_points())
            grid_ver[0] += 1
        except Exception as e:
            log(f"[learn] grid rebuild failed: {e!r}")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ── LiDAR localisation (ICP against the saved map) ───────────────────────────

class Localizer(threading.Thread):
    def __init__(self, map_file):
        super().__init__(daemon=True)
        import open3d as o3d
        import build_map as bm
        self.o3d, self.bm = o3d, bm
        self.map = o3d.io.read_point_cloud(map_file).voxel_down_sample(bm.VOXEL_SIZE)
        self.T = np.eye(4)
        self.mode = "nav"            # "nav" = localise on saved map, "map" = build a new one
        self.mmap = o3d.geometry.PointCloud()
        self.skipped = 0
        self.key_T, self.merges = np.eye(4), 0
        self.bad = 0
        self.last_diag = 0.0
        self.learn, self.last_learn = True, 0.0
        self.strict, self.still_n, self.hold_until = False, 0, 0.0     # strict = explore mode: only merge scans taken at a standstill
        self.still_ref, self.still_since = None, time.time()
        self._reset_live_map()
        self._dt, self._dt_key = None, None
        self.scan_id = 0
        self.fitness = 0.0
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.sweep, self.sweep_id, self.reader_err = None, 0, None

    def live_points(self):
        """Latest full sweep as (x, y) metres in the car's LiDAR frame."""
        with self.cond:
            raw = self.sweep
        if not raw:
            return np.zeros((0, 2))
        a = np.radians([p[0] for p in raw])
        d = np.array([p[1] for p in raw]) / 1000.0
        ok = d > BODY_R
        return np.stack([d[ok] * np.cos(a[ok]), d[ok] * np.sin(a[ok])], 1)

    def pose(self):
        with self.lock:
            return self.T[0, 3], self.T[1, 3], math.atan2(self.T[1, 0], self.T[0, 0]), self.scan_id

    def wait_scans(self, n=None, timeout=4.0):
        """Block until n fresh scans have been matched (call after the car stopped).
        Exploring waits one scan longer, so the standstill scans needed for map merging happen."""
        if n is None:
            n = 3 if self.strict else 2
        target = self.scan_id + n
        t0 = time.time()
        while self.scan_id < target and time.time() - t0 < timeout:
            time.sleep(0.05)
        return self.pose()

    # ── mapping ──
    def start_mapping(self):
        with self.lock:
            self.mmap = self.o3d.geometry.PointCloud()
            self.T, self.skipped, self.mode = np.eye(4), 0, "map"

    def _reset_live_map(self):
        """(Re)start the editable copy of the map from self.map."""
        self.mp = np.asarray(self.map.points).copy()
        self.mmiss = np.zeros(len(self.mp), np.int16)
        self.cand = {}
        self.map_version, self.added, self.removed = 0, 0, 0

    def live_map_points(self):
        with self.lock:
            return self.mp.copy()

    def save_current(self, path):
        pts = self.live_map_points()
        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(pts)
        if not self.o3d.io.write_point_cloud(path, pcd):
            raise RuntimeError("could not write " + path)

    def _learn(self, pcd, T):
        """Keep the map current while driving: add what is newly there (after repeated sightings),
        remove what the LiDAR beam now passes straight through. Runs only on well-matched scans."""
        from scipy.spatial import cKDTree
        P = np.asarray(pcd.points)[:, :2]
        r = np.hypot(*P.T)
        W = P[(r > 0.30) & (r < 3.5)] @ T[:2, :2].T + T[:2, 3]
        if len(W) < 30 or len(self.mp) == 0:
            return
        changed = False
        # ---- additions
        d, _ = cKDTree(self.mp[:, :2]).query(W)
        keys = {tuple(k) for k in np.floor(W[d > LEARN_NEW_DIST] / 0.05).astype(int)}
        for k in keys:
            self.cand[k] = self.cand.get(k, 0) + 1
        for k in [k for k in self.cand if k not in keys]:          # not seen again: fade out
            self.cand[k] -= 1
            if self.cand[k] <= 0:
                del self.cand[k]
        done = [k for k, c in self.cand.items() if c >= LEARN_ADD_N]
        new_pts = None
        if done:
            new_pts = np.array([[(k[0] + .5) * 0.05, (k[1] + .5) * 0.05, 0.0] for k in done])
            for k in done:
                del self.cand[k]
        # ---- removals: map points the beam went past
        S = (self.mp[:, :2] - T[:2, 3]) @ T[:2, :2]                  # map points in the car's scan frame
        rng = np.hypot(*S.T)
        b = (np.degrees(np.arctan2(S[:, 1], S[:, 0])) % 360).astype(int)
        meas = np.zeros(360)                                         # nearest measured range per degree (0 = no data)
        sb = (np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360).astype(int)
        for bi, ri in zip(sb, r):
            if meas[bi] == 0 or ri < meas[bi]:
                meas[bi] = ri
        has = (meas[(b - 1) % 360] > 0) & (meas[b] > 0) & (meas[(b + 1) % 360] > 0)
        nearest3 = np.minimum(np.minimum(meas[(b - 1) % 360], meas[b]), meas[(b + 1) % 360])
        gone = has & (rng > 0.30) & (rng < 3.5) & (nearest3 > rng + 0.25)
        seen = has & (np.abs(meas[b] - rng) < 0.10)
        self.mmiss[gone] += 1
        self.mmiss[seen & ~gone] = np.maximum(0, self.mmiss[seen & ~gone] - 1)
        drop = self.mmiss >= LEARN_DEL_N
        with self.lock:
            if drop.any():
                self.mp, self.mmiss = self.mp[~drop], self.mmiss[~drop]
                self.removed += int(drop.sum())
                changed = True
            if new_pts is not None:
                self.mp = np.vstack([self.mp, new_pts])
                self.mmiss = np.concatenate([self.mmiss, np.zeros(len(new_pts), np.int16)])
                self.added += len(new_pts)
                changed = True
            if changed:
                cloud = self.o3d.geometry.PointCloud()
                cloud.points = self.o3d.utility.Vector3dVector(self.mp)
                self.map = cloud                                     # ICP uses the updated map from now on
                self.map_version += 1

    def set_map(self, path):
        pcd = self.o3d.io.read_point_cloud(path).voxel_down_sample(self.bm.VOXEL_SIZE)
        with self.lock:
            self.map, self.mode, self.T, self.bad = pcd, "nav", np.eye(4), 0
        self._reset_live_map()

    def cancel_mapping(self):
        with self.lock:
            self.mode, self.T = "nav", np.eye(4)

    def map_points(self, limit=15000):
        with self.lock:
            p = np.asarray(self.mmap.points)[:, :2].astype(np.float32)
        if len(p) > limit:
            p = p[:: len(p) // limit + 1]
        return p, self.skipped

    def save_map(self, path):
        """Clean and write the map being built. Does NOT leave mapping mode - the caller
        does that once everything else (grid, switch) has worked."""
        o3d = self.o3d
        with self.lock:
            pcd = self.mmap.voxel_down_sample(0.05)
        if len(pcd.points) < 100:
            raise RuntimeError("map is still empty - drive around first")
        clean, _ = pcd.remove_radius_outlier(nb_points=3, radius=0.15)      # isolated streak points
        clean, _ = clean.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(clean.points) < 0.5 * len(pcd.points):                       # filter ate the walls: don't trust it
            clean = pcd
        if not o3d.io.write_point_cloud(path, clean):
            raise RuntimeError("could not write " + path)

    def _icp(self, src, dst, init, fine=0.10):
        o3d = self.o3d
        reg, est = o3d.pipelines.registration, o3d.pipelines.registration.TransformationEstimationPointToPoint()
        coarse = reg.registration_icp(src, dst, 0.30, init, est, reg.ICPConvergenceCriteria(max_iteration=30))
        return reg.registration_icp(src, dst, fine, coarse.transformation, est,
                                    reg.ICPConvergenceCriteria(max_iteration=50))

    def _distance_field(self, g):
        """Distance (m) from every grid cell to the nearest mapped wall point; cached per map."""
        if self._dt_key is not id(g):
            from scipy.ndimage import distance_transform_edt
            self._dt, self._dt_key = distance_transform_edt(~g.occ_raw) * RES, id(g)
        return self._dt

    def _relocalize(self, pcd):
        """Global search for the car's pose: score many (x, y, heading) guesses by how well the
        scan lands on mapped walls, then refine the best few with ICP."""
        g = gm
        t0 = time.time()
        dt = self._distance_field(g)
        P = np.asarray(pcd.points)[:, :2]
        P = P[:: len(P) // 150 + 1]
        cells = np.argwhere(g.main)
        cells = cells[(cells[:, 0] % 4 == 0) & (cells[:, 1] % 4 == 0)]          # every 20 cm
        pos = np.stack([g.min[0] + (cells[:, 1] + .5) * RES, g.min[1] + (cells[:, 0] + .5) * RES], 1)
        best = []                                                                # (score, x, y, yaw)
        for yaw in np.arange(0, 2 * np.pi, np.radians(6)):
            c, s_ = np.cos(yaw), np.sin(yaw)
            Q = P @ np.array([[c, s_], [-s_, c]])                                # rotate by yaw
            W = pos[:, None, :] + Q[None]
            ix = ((W[..., 0] - g.min[0]) / RES).astype(int)
            iy = ((W[..., 1] - g.min[1]) / RES).astype(int)
            ok = (ix >= 0) & (ix < g.w) & (iy >= 0) & (iy < g.h)
            d = np.where(ok, dt[np.clip(iy, 0, g.h - 1), np.clip(ix, 0, g.w - 1)], 1.0)
            score = np.exp(-(d / 0.12) ** 2).mean(1)
            for k in np.argsort(score)[-3:]:
                best.append((score[k], pos[k, 0], pos[k, 1], yaw))
        best.sort(reverse=True)
        found, found_fit = None, 0.0
        for sc, x, y, yaw in best[:8]:
            T0 = np.eye(4)
            T0[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
            T0[0, 3], T0[1, 3] = x, y
            r = self._icp(pcd, self.map, T0, fine=0.15)
            if r.fitness > found_fit:
                found, found_fit = r.transformation, r.fitness
        log(f"[reloc] best fitness {found_fit:.2f} in {time.time() - t0:.1f}s")
        if found is not None and found_fit >= RELOC_FIT_MIN:
            with self.lock:
                self.T, self.bad = found, 0
            log(f"[reloc] found pose x={found[0, 3]:.2f} y={found[1, 3]:.2f}")

    def _map_step(self, pcd):
        """Track the car against the map built so far, and only add a scan to the map when
        the car has really moved (a stationary car must not thicken the walls)."""
        bm = self.bm
        with self.lock:
            if len(self.mmap.points) == 0:
                self.mmap += pcd
                self.key_T, self.merges = np.eye(4), 0
                return True
            prev, key = self.T.copy(), self.key_T
        res = self._icp(pcd, self.mmap.voxel_down_sample(bm.VOXEL_SIZE), prev)
        T = res.transformation
        d = np.linalg.inv(prev) @ T
        jump, turn = math.hypot(d[0, 3], d[1, 3]), abs(math.atan2(d[1, 0], d[0, 0]))
        if res.fitness < (MAP_FIT_STRICT if self.strict else MAP_FIT_MIN) or jump > 0.35 or turn > math.radians(30):
            with self.lock:
                self.skipped += 1
            return False                     # bad match: ignore it, never merge it
        k = np.linalg.inv(key) @ T
        moved = math.hypot(k[0, 3], k[1, 3]) > 0.08 or abs(math.atan2(k[1, 0], k[0, 0])) > math.radians(4)
        steady = jump < 0.06 and turn < math.radians(2)   # merging mid-turn smears walls into streaks
        if self.strict:                                   # exploring: add scans only after the car has paused
            self.still_n = self.still_n + 1 if (jump < 0.012 and turn < math.radians(0.8)) else 0
            steady = self.still_n >= 2 and time.time() >= self.hold_until
        with self.lock:
            self.T = T
            if moved and steady:
                pcd.transform(T)
                self.mmap += pcd
                self.key_T, self.merges = T, self.merges + 1
                if self.merges % 10 == 0:
                    self.mmap = self.mmap.voxel_down_sample(bm.VOXEL_SIZE)
        return True

    def run(self):
        """Keep the LiDAR alive: any failure restarts the whole session."""
        while True:
            try:
                self._session()
            except Exception as e:
                log(f"[lidar] session failed: {e!r} - restarting in 2 s")
                state["msg"] = "LiDAR error - restarting…"
                time.sleep(2)

    def _reader(self, ser, stop):
        """Read the raw serial stream, validate every packet, and publish each full revolution
        (a packet with the start flag marks the beginning of a new one)."""
        bm = self.bm
        buf, pts, got_start, bad_bytes = bytearray(), [], False, 0
        try:
            while not stop.is_set():
                chunk = ser.read(max(1, ser.in_waiting))
                if not chunk:
                    raise RuntimeError("LiDAR sent no data")
                buf += chunk
                before = len(buf)
                for start, quality, angle, dist in parse_scan_packets(buf):
                    if start:
                        if got_start and len(pts) >= bm.MIN_POINTS:
                            with self.cond:
                                self.sweep, self.sweep_id = pts, self.sweep_id + 1
                                self.cond.notify_all()
                        pts, got_start = [], True
                    if got_start and dist > 0 and quality > 0:
                        pts.append((angle, dist))
        except Exception as e:
            self.reader_err = e
            with self.cond:
                self.cond.notify_all()

    def _session(self):
        from pyrplidar import PyRPlidar
        bm, o3d = self.bm, self.o3d
        lidar = PyRPlidar()
        lidar.connect(port=bm.PORT, baudrate=bm.BAUDRATE, timeout=3)
        stop = threading.Event()
        try:
            lidar.stop()
            time.sleep(0.5)
            lidar.lidar_serial._serial.reset_input_buffer()   # stale bytes from a killed run
            lidar.get_info()
            lidar.set_motor_pwm(660)
            time.sleep(2)
            lidar.start_scan()                       # sends the scan command, reads the 7-byte reply
            self.reader_err, self.sweep, self.sweep_id = None, None, 0
            threading.Thread(target=self._reader, args=(lidar.lidar_serial._serial, stop), daemon=True).start()
            seen = 0
            while True:
                with self.cond:
                    self.cond.wait_for(lambda: self.sweep_id > seen or self.reader_err, timeout=5)
                    if self.reader_err:
                        raise self.reader_err
                    if self.sweep_id == seen:
                        raise RuntimeError("no scans for 5 s")
                    seen, raw = self.sweep_id, self.sweep
                raw_pcd = bm.scan_to_pointcloud(raw)
                pcd = bm.preprocess(raw_pcd)
                n = len(pcd.points)
                if time.time() - self.last_diag > 5:
                    self.last_diag = time.time()
                    log(f"[loc] mode={self.mode} sweeps={self.sweep_id} raw={len(raw)} in_range={len(raw_pcd.points)} "
                        f"filtered={n} fit={self.fitness:.2f} bad={self.bad} pose=({self.T[0,3]:.2f},{self.T[1,3]:.2f})")
                if n < (bm.MIN_POINTS if self.mode == "map" else NAV_MIN_POINTS):
                    # too few usable points (everything beyond range / filtered out): say so, don't stay silent
                    with self.lock:
                        self.bad += 1
                        self.scan_id += 1
                    state["lost"] = self.bad >= LOST_SCANS
                    state["msg"] = f"Too few LiDAR points in range ({n}) - move the car closer to walls"
                    continue
                if self.mode == "map":
                    ok = self._map_step(pcd)
                    with self.lock:
                        self.scan_id += 1
                    state["lost"] = not ok
                    continue
                res = self._icp(pcd, self.map, self.T, fine=0.15)
                good = res.fitness >= FIT_MIN
                with self.lock:
                    self.fitness = res.fitness
                    if good:
                        self.T = res.transformation
                    self.bad = 0 if good else self.bad + 1
                    self.scan_id += 1
                state["lost"] = self.bad >= LOST_SCANS
                # learn only after the car has paused: any real movement restarts the wait
                Tn = self.T
                moved = (self.still_ref is None or math.hypot(*(Tn[:2, 3] - self.still_ref[:2, 3])) > 0.03
                         or abs(wrap(math.atan2(Tn[1, 0], Tn[0, 0]) - math.atan2(self.still_ref[1, 0], self.still_ref[0, 0]))) > math.radians(3))
                if moved:
                    self.still_ref, self.still_since = Tn.copy(), time.time()
                if (self.learn and good and res.fitness >= LEARN_FIT and self.bad == 0
                        and time.time() - self.still_since >= LEARN_PAUSE
                        and time.time() - self.last_learn >= LEARN_EVERY):
                    self.last_learn = time.time()
                    try:
                        self._learn(pcd, res.transformation)
                    except Exception as e:
                        log(f"[learn] skipped: {e!r}")
                if self.bad >= LOST_SCANS and (self.bad - LOST_SCANS) % 3 == 0:
                    # car was moved by hand (or tracking failed): search the whole map
                    self._relocalize(pcd)
                    state["lost"] = self.bad >= LOST_SCANS
        finally:
            stop.set()
            try:
                lidar.stop()
                lidar.set_motor_pwm(0)
            except Exception:
                pass
            lidar.disconnect()


# ── Car (drives the motors directly through Robo.py / gpiozero) ──────────────

class Car:
    """Same interface the navigator always used (send / pulse / set_speeds), minus the network hop."""
    _METHOD = {"F": "forward", "B": "backward", "L": "left", "R": "right", "S": "stop"}

    def __init__(self):
        from Robo import Robot
        self.robo = Robot()
        self.lock = threading.Lock()
        self.move_speed, self.turn_speed = 0.3, NAV_TURN_SPEED

    def set_speeds(self, move=None, turn=None):
        """Forward / turn speed (0..1), applied to every command from now on."""
        if move is not None:
            self.move_speed = round(max(0.0, min(1.0, float(move))), 2)
        if turn is not None:
            self.turn_speed = round(max(0.0, min(1.0, float(turn))), 2)

    def send(self, d):
        if d not in self._METHOD:
            return
        with self.lock:
            fn = getattr(self.robo, self._METHOD[d])
            if d == "S":
                fn()
            else:
                fn(self.move_speed if d in "FB" else self.turn_speed)

    def pulse(self, d, secs):
        self.send(d)
        time.sleep(secs)
        self.send("S")


# ── Camera (camera_stream.py under .camvenv -> MJPEG at /video) ──────

class Camera:
    def __init__(self):
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        self.frame, self.cond = None, threading.Condition()
        self.proc = subprocess.Popen([os.path.join(here, ".camvenv", "bin", "python"), os.path.join(here, "camera_stream.py")],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        import struct
        f = self.proc.stdout
        while True:
            head = f.read(4)
            if len(head) < 4:
                log("[camera] stream ended")
                return
            data = f.read(struct.unpack(">I", head)[0])
            with self.cond:
                self.frame = data
                self.cond.notify_all()

    def frames(self):
        while True:
            with self.cond:
                self.cond.wait(timeout=2.0)
                fr = self.frame
            if fr:
                yield fr


camera = None


# ── Navigator ────────────────────────────────────────────────────────────────

class Hazard(RuntimeError):
    """The camera thinks the floor ends just ahead."""


class Navigator:
    """Pulse-and-measure controller. Self-calibrates two unknowns of the LiDAR frame:
    which way 'forward' points (fwd_off) and whether 'L' raises or lowers the yaw (turn_sign).
    Turns are proportional (short pulse for a small error) and the turn rate is learned.
    The live LiDAR scan is checked before every forward pulse; an obstacle triggers a local reroute."""
    def __init__(self):
        self.fwd_off = None
        self.turn_sign = 1
        self.turn_rate = TURN_RATE0      # rad/s at the current turn speed, learned
        self.rate_speed = NAV_TURN_SPEED
        self.move_t = 0.4
        self.max_turn = 0.35
        self.hazard_check = None         # callable -> truthy when the floor ahead looks unsafe

    def heading(self, yaw):
        return yaw + (self.fwd_off or 0.0)

    # ── obstacles ──
    def blocked_ahead(self, limit=OBST_AHEAD):
        """Nearest obstacle distance in the corridor in front of the car (None = clear).
        The corridor is as wide as the planner's clearance, so a planned path never trips it."""
        P = loc.live_points()
        if not len(P):
            return None
        if self.fwd_off is None:                       # front unknown: any close point blocks
            r = np.hypot(*P.T)
            return float(r.min()) if (r < 0.30).any() else None
        c, s_ = math.cos(self.fwd_off), math.sin(self.fwd_off)
        f = P[:, 0] * c + P[:, 1] * s_
        lat = -P[:, 0] * s_ + P[:, 1] * c
        m = (f > 0) & (f < limit) & (np.abs(lat) < gm.radius)
        return float(f[m].min()) if m.any() else None

    def clear_dir(self, rel, limit):
        """Nearest obstacle in a corridor `rel` radians off the car's front (pi = straight behind); None = clear."""
        P = loc.live_points()
        if not len(P) or self.fwd_off is None:
            return None
        a = self.fwd_off + rel
        c, s_ = math.cos(a), math.sin(a)
        f = P[:, 0] * c + P[:, 1] * s_
        lat = -P[:, 0] * s_ + P[:, 1] * c
        m = (f > 0) & (f < limit) & (np.abs(lat) < gm.radius)
        return float(f[m].min()) if m.any() else None

    def back_off(self, want=0.30):
        """Get unstuck: reverse a little while the LiDAR says the way behind is clear; if it isn't,
        turn away instead. Returns True if the car moved or turned."""
        car.send("S")
        x0, y0, yaw0, _ = loc.wait_scans()
        moved = 0.0
        for _ in range(5):
            if state["stop"] or moved >= want:
                break
            if self.clear_dir(math.pi, 0.40) is not None:
                break                                         # something right behind: don't reverse into it
            state["msg"] = "Stuck - backing up…"
            car.pulse("B", 0.4)
            x, y, _, _ = loc.wait_scans()
            moved = math.hypot(x - x0, y - y0)
        if moved >= 0.08:
            log(f"[nav] backed up {moved:.2f} m")
            loc.hold_until = time.time() + SETTLE_S
            loc.wait_scans(3)
            return True
        # could not reverse: turn toward whichever side is more open
        left, right = self.clear_dir(math.pi / 2, 1.2), self.clear_dir(-math.pi / 2, 1.2)
        sign = 1 if (right is not None and (left is None or left > right)) else -1
        state["msg"] = "Stuck - turning away…"
        _, _, yaw, _ = loc.wait_scans()
        goal = yaw + sign * math.radians(100)
        for _ in range(10):
            if state["stop"]:
                break
            _, _, yaw, _ = loc.wait_scans()
            err = wrap(goal - yaw)
            if abs(err) < AIM_TOL:
                break
            self.turn_step(err, yaw)
        log("[nav] turned away to get unstuck")
        loc.hold_until = time.time() + SETTLE_S
        return True

    def obstacle_mask(self):
        """Everything the LiDAR sees within 2.5 m, as map cells, inflated a bit more than the planner
        does. Using all points (not just ones missing from the map) keeps the detour consistent with
        the corridor check that triggered it, so the same obstacle can't be 'planned through' again."""
        P = loc.live_points()
        P = P[np.hypot(*P.T) < 2.5]
        with loc.lock:
            T = loc.T.copy()
        W = P @ T[:2, :2].T + T[:2, 3]
        mask = np.zeros((gm.h, gm.w), dtype=bool)
        for x, y in W:
            cx, cy = gm.to_cell(x, y)
            if gm.inside(cx, cy):
                mask[cy, cx] = True
        return gm.inflate(mask, DETOUR_MARGIN)

    def reroute(self, path, k):
        """Replace waypoint k by a detour around what the LiDAR sees now. False = no way round."""
        x, y, _, _ = loc.pose()
        sub = gm.plan((x, y), path[k], extra=self.obstacle_mask())
        if sub is None or len(sub) < 2:
            return False
        path[k:k + 1] = sub[1:]                   # sub ends at the original waypoint
        state["path"] = [tuple(p) for p in path]
        log(f"[nav] rerouted around obstacle: {len(sub) - 1} new waypoint(s)")
        return True

    # ── driving ──
    def wait_tracking(self, secs=8.0):
        """Car stays stopped until the LiDAR pose is trustworthy again (it relocalises on its own)."""
        car.send("S")
        t0 = time.time()
        while loc.bad >= LOST_SCANS and time.time() - t0 < secs:
            if state["stop"]:
                return False
            state["msg"] = "Lost my position - searching the map…"
            time.sleep(0.3)
        loc.wait_scans()
        return loc.bad < LOST_SCANS

    def calibrate_forward(self):
        """First forward pulse: learn which LiDAR direction is the car's front. No return trip."""
        x0, y0, yaw0, _ = loc.wait_scans()
        for secs in (0.5, 0.9):
            if self.blocked_ahead() is not None:
                raise RuntimeError("something is right next to the car - give it some room")
            car.pulse("F", secs)
            x1, y1, _, _ = loc.wait_scans(3)
            if math.hypot(x1 - x0, y1 - y0) >= 0.04:
                self.fwd_off = wrap(math.atan2(y1 - y0, x1 - x0) - yaw0)
                log(f"[nav] forward offset {math.degrees(self.fwd_off):.0f} deg")
                return
        raise RuntimeError("car did not move (clear path? speed too low?)")

    def turn_step(self, err, yaw):
        """One short turn pulse sized to the remaining error; learns the turn rate."""
        d = "L" if err * self.turn_sign > 0 else "R"
        ts = car.turn_speed or NAV_TURN_SPEED
        if abs(ts - self.rate_speed) > 0.01:                 # turn speed changed: old rate no longer valid
            self.turn_rate, self.rate_speed = TURN_RATE0 * ts / NAV_TURN_SPEED, ts
        secs = min(self.max_turn, max(0.12, 0.7 * abs(err) / self.turn_rate))     # <0.12 s is too short for the motors to start
        car.pulse(d, secs)
        _, _, yaw2, _ = loc.wait_scans()
        dyaw = wrap(yaw2 - yaw)
        want = self.turn_sign if d == "L" else -self.turn_sign
        if abs(dyaw) < math.radians(1):
            self.turn_rate = max(math.radians(8), self.turn_rate * 0.5)     # barely moved: pulses must be longer
        elif dyaw * want < 0 and abs(dyaw) > math.radians(3):
            self.turn_sign = -self.turn_sign                                # wrong way: flip
        else:
            self.turn_rate = 0.6 * self.turn_rate + 0.4 * (abs(dyaw) / secs)

    def face(self, goal, tries=14):
        """Turn in place until the car's front points along world heading `goal` (radians)."""
        for _ in range(tries):
            if state["stop"]:
                break
            _, _, yw, _ = loc.wait_scans()
            err = wrap(goal - self.heading(yw))
            if abs(err) < AIM_TOL:
                break
            self.turn_step(err, yw)
        loc.hold_until = time.time() + SETTLE_S

    def approach(self, path, k):
        """Drive to waypoint k. Returns "arrived", "rerouted" (path changed, call again) or None (stopped)."""
        wx, wy = path[k]
        best, pulses = math.inf, 0              # stuck detector: forward pulses that do not bring us closer
        while True:
            if state["stop"]:
                return None
            if loc.bad >= LOST_SCANS and not self.wait_tracking():
                raise RuntimeError("lost my position - put the car somewhere with walls in view and press Go")
            x, y, yaw, _ = loc.wait_scans()
            dist = math.hypot(wx - x, wy - y)
            if dist < ARRIVE_M:
                return "arrived"
            if gm.in_zone((x, y)):
                car.send("S")
                raise RuntimeError("the car is inside a no-go zone - stopped")
            if gm.hits_zone((x, y), (wx, wy)):
                # waypoint is reachable only through a zone (zone added mid-run / position drift): plan around it
                sub = gm.plan((x, y), (wx, wy))
                if sub is None or len(sub) < 2:
                    raise RuntimeError("no way to the next waypoint without entering a no-go zone")
                path[k:k + 1] = sub[1:]
                state["path"] = [tuple(p) for p in path]
                log("[nav] re-planned around a no-go zone")
                return "rerouted"
            err = wrap(math.atan2(wy - y, wx - x) - self.heading(yaw))
            state["msg"] = f"Driving… waypoint {k}/{len(path) - 1}, {dist:.2f} m to go"
            log(f"[nav] wp{k} pos=({x:.2f},{y:.2f}) dist={dist:.2f} err={math.degrees(err):.0f}deg fit={loc.fitness:.2f}")
            if abs(err) > AIM_TOL:
                self.turn_step(err, yaw)
                continue
            if self.hazard_check and self.hazard_check():
                car.send("S")
                raise Hazard("floor ahead looks like a ledge")
            ob = self.blocked_ahead(min(OBST_AHEAD, dist + 0.05))     # stuff beyond the waypoint doesn't matter
            if ob is None:
                if dist < best - 0.03:
                    best, pulses = dist, 0
                else:
                    pulses += 1
                    if pulses >= 6:
                        raise RuntimeError("no progress after 6 pushes - the car is stuck")
                car.pulse("F", min(self.move_t, max(0.15, dist / (0.25 * (car.move_speed or 0.3) / 0.3))))
                continue
            # something is in the way
            car.send("S")
            state["msg"] = f"Obstacle {ob:.2f} m ahead - going around"
            log(f"[nav] obstacle at {ob:.2f} m")
            self.detours += 1
            if self.detours <= 8 and self.reroute(path, k):
                return "rerouted"
            t0 = time.time()                       # no way round: maybe a person walking past
            while self.blocked_ahead() is not None and time.time() - t0 < 5 and not state["stop"]:
                state["msg"] = "Path blocked - waiting…"
                time.sleep(0.3)
            if self.blocked_ahead() is not None:
                raise RuntimeError("path blocked by an obstacle with no way round")
            self.detours = 0

    def go(self, path):
        path = list(path)
        if self.fwd_off is None:
            state["msg"] = "Calibrating…"
            self.calibrate_forward()
        self.detours = 0
        while state["wp"] < len(path):
            r = self.approach(path, state["wp"])
            if r is None:
                return False
            if r == "arrived":
                state["wp"] += 1
                self.detours = 0
        state["msg"] = "Arrived"
        return True


def follow(path):
    state["moving"], state["stop"], state["arrived"] = True, False, False
    try:
        state["arrived"] = bool(sim_follow(path) if car is None else nav.go(path))
        if not state["arrived"] and state["stop"]:
            state["msg"] = "Stopped - press Go to resume"
    except Exception as e:
        state["msg"] = f"Stopped: {e}"
        log(f"[nav] aborted: {e!r}")
    finally:
        if car:
            car.send("S")
        state["moving"] = False
        if state["arrived"]:
            state["path"] = []


def sim_follow(path):
    for x, y in path[state["wp"]:]:
        dist = math.hypot(x - state["pose"][0], y - state["pose"][1])
        steps = max(1, int(dist / SIM_SPEED / 0.1))
        x0, y0 = state["pose"]
        for i in range(1, steps + 1):
            if state["stop"]:
                return
            time.sleep(0.1)
            state["pose"] = [x0 + (x - x0) * i / steps, y0 + (y - y0) * i / steps]
        state["wp"] += 1
    state["msg"] = "Arrived"
    return True


nav = Navigator()


# ── Auto-explore ─────────────────────────────────────────────────────────────

def set_zones_list(new):
    global zones
    zones = new


def store_pin(entry, frame):
    """Save a photo pin (metadata + jpeg) next to the currently loaded map."""
    os.makedirs(photos_base[0] + ".photos", exist_ok=True)
    with open(photo_file(entry["id"]), "wb") as f:
        f.write(frame)
    photos.append({k: v for k, v in entry.items() if k != "frame"})
    save_photos()


class Explorer:
    """Drives to unexplored edges while mapping (frontier exploration), or - with an LLM brain attached -
    lets the LLM choose the next leg (explore / go / look / pin / no-go) in mapping or navigation mode.
    Floor that looks like it ends (camera) becomes a no-go box and is never driven into."""

    def __init__(self):
        self.on = False
        self.occ = None
        self.zones = []              # no-go boxes drawn automatically; they become the new map's zones on save
        self.pins = []               # photo pins taken while mapping (written with the map on save)
        self.objects = []            # named objects found while mapping (written with the map on save)
        self.target = None
        self.bad = []                # frontier spots that failed
        self.trail = []              # places the car has stopped and looked from
        self.prev_gm = None
        self.radius = 5.0
        self.hazards = 0
        self.fails = 0
        self.brain, self.task, self.ai_log, self.ai_n, self.finished = None, "", [], 0, None
        self.runlog = None

    def reset(self):
        self.occ, self.zones, self.pins, self.bad, self.trail, self.target, self.hazards = None, [], [], [], [], None, 0
        self.objects = []
        obj_last[0] = None

    def start(self, radius, brain=None, task=""):
        self.radius, self.on, self.brain, self.task = radius, True, brain, task
        self.ai_log, self.ai_n, self.finished, self.fails = [], 0, None, 0
        threading.Thread(target=self.run, daemon=True).start()

    def snap(self, frame, tag):
        """Keep a camera frame with the run's log (unless EXP_LOG=0). Returns the file name or None."""
        if self.runlog and frame:
            return self.runlog.snap(frame, tag)

    def hazard_now(self):
        fr = camera.frame if camera else None
        if fr is None:
            return False
        try:
            sc = ex.ledge_score(fr)
        except Exception:
            return False
        if sc > ex.LEDGE_THRESHOLD:
            log(f"[explore] ledge check tripped (score {sc:.0f} > {ex.LEDGE_THRESHOLD:.0f})")
            return True
        return False

    def add_zone(self, pts, note):
        """A no-go polygon: kept with the map being built, or applied straight away on a loaded map."""
        global gm
        pts = [[float(x), float(y)] for x, y in pts]
        if loc.mode == "map":
            self.zones.append({"id": max([z["id"] for z in self.zones] + [0]) + 1, "pts": pts, "auto": note})
        else:
            zones.append({"id": max([z["id"] for z in zones] + [0]) + 1, "pts": pts, "auto": note})
            gm.set_zones([z["pts"] for z in zones])
            save_zones()

    def add_hazard_zone(self):
        x, y, yaw, _ = loc.pose()
        h = nav.heading(yaw)
        u, v = (math.cos(h), math.sin(h)), (-math.sin(h), math.cos(h))
        pts = [[x + u[0] * f + v[0] * w, y + u[1] * f + v[1] * w] for f, w in ((0.15, -0.3), (0.85, -0.3), (0.85, 0.3), (0.15, 0.3))]
        self.add_zone(pts, "ledge?")
        self.hazards += 1
        if camera and camera.frame:                       # keep the frame so you can judge it afterwards
            os.makedirs("explore_hazards", exist_ok=True)
            with open(f"explore_hazards/{time.strftime('%Y%m%d-%H%M%S')}.jpg", "wb") as f:
                f.write(camera.frame)
        log(f"[explore] no-go box added ahead of ({x:.2f},{y:.2f}) - possible ledge ({self.hazards})")
        if self.runlog:
            self.runlog.event("ledge_alarm", x=round(x, 3), y=round(y, 3), n=self.hazards)

    def candidates(self, g, car_xy):
        """Reachable, not-yet-seen frontiers as [(dist, size, cell, path)], nearest first."""
        raw = []
        for size, c in g.frontiers(self.radius):
            if any(math.dist(c, b) < BAD_TARGET_R for b in self.bad):
                continue
            if any(math.dist(c, t) < SEEN_R for t in self.trail):     # already been (and looked) there
                continue
            raw.append((math.dist(c, car_xy) - 0.02 * min(size, 30), size, c))
        raw.sort()
        out = []
        for _, size, c in raw[:8]:
            path = g.plan(car_xy, c)
            if path and len(path) >= 2:
                out.append((sum(math.dist(p, q) for p, q in zip(path, path[1:])), size, c, path))
        return out

    def drive_leg(self, target, path):
        """Run one short leg. None = fine, otherwise a sentence about what went wrong."""
        leg = ex.truncate(path, EXPLORE_LEG)
        state["path"], state["wp"] = leg, 1
        try:
            nav.go(leg)
            self.fails = 0
            return None
        except Hazard:
            self.add_hazard_zone()
            self.bad.append(target)
            nav.back_off(0.35)                  # step away from the edge, then choose another target
            if self.hazards >= MAX_HAZARDS:
                raise RuntimeError(f"{self.hazards} possible ledges found - stopping, please check the car")
            return "the camera ledge check stopped the car; a no-go box was added ahead of it"
        except RuntimeError as e:
            log(f"[explore] leg failed: {e}")
            self.bad.append(target)
            self.fails += 1
            nav.back_off()                      # reverse (or turn away), then try a different target
            if self.fails >= EXPLORE_FAILS:
                raise RuntimeError(f"stuck - {self.fails} targets in a row failed ({e})")
            return f"the leg failed ({e}); the car backed off"

    # ── AI decision step ──
    def ai_step(self, g, car_xy, yaw, cands):
        """Ask the LLM what to do next and carry it out. Returns False when the run should end."""
        import llm_brain
        nav_mode = loc.mode != "map"
        h = nav.heading(yaw)
        cells = g.grid.copy()
        cells[g.occ_raw] = 3
        zl = zones if nav_mode else self.zones
        pins = photos if nav_mode else self.pins
        fr = [(i + 1, c[2]) for i, c in enumerate(cands)]
        png = llm_brain.render_map(cells, g.min, car_xy, h, fr, [z["pts"] for z in zl],
                                   [(p["id"], (p["x"], p["y"])) for p in pins], self.trail[-60:])
        lines = [f"Robot at ({car_xy[0]:.2f}, {car_xy[1]:.2f}) m, facing {math.degrees(wrap(h)):.0f} deg (0 = +x, 90 = +y).",
                 f"No-go boxes: {len(zl)}. Ledge alarms this run: {self.hazards}.",
                 f"Decisions used: {self.brain.decisions}/{llm_brain.MAX_DECISIONS}."]
        lines.append(objects_line(object_pool()))
        if nav_mode:
            lines.append(places_line())
        if pins:
            lines.append("Photo pins: " + "; ".join(f"{p['id']}={p.get('note') or 'photo'} at ({p['x']:.1f},{p['y']:.1f})" for p in pins[-12:]))
        if nav_mode:
            lines.append("Mode: navigating a finished map (no exploration).")
        elif cands:
            lines.append("Frontiers (route length m, size): " + "; ".join(
                f"F{i + 1} at ({c[2][0]:.1f},{c[2][1]:.1f}) {c[0]:.1f} m, {c[1]} cells" for i, c in enumerate(cands)))
        else:
            lines.append("There are no reachable frontiers left.")
        allowed = set(llm_brain.allowed_tools())
        allowed.discard("explore_to" if nav_mode else "push_object")
        if not (nav_mode and places):
            allowed.discard("go_to_place")
        n_next = self.brain.decisions + 1
        imgs = []
        if self.runlog:
            imgs = [self.runlog.snap_png(png, f"d{n_next}_map"), self.snap(camera.frame if camera else None, f"d{n_next}_cam")]
        t_think = time.time()
        try:
            name, args = self.brain.decide(self.task, png, camera.frame if camera else None, "\n".join(lines), allowed)
        except llm_brain.DecisionLimit:
            raise
        except llm_brain.BrainError as e:
            if nav_mode:
                raise RuntimeError(f"AI unavailable: {e}")
            log(f"[ai] {e} - using the built-in explorer for this step")
            self.ai_log.append(f"AI error ({e}) - built-in explorer used")
            return self.heuristic_step(cands)
        why = args.get("reason") or args.get("note") or args.get("summary") or ""
        t_exec = time.time()
        think_s = t_exec - t_think
        res = self.execute(name, args, g, car_xy, yaw, cands)
        if self.runlog:
            after = self.snap(camera.frame if camera else None, f"d{n_next}_after_{name}")
            self.runlog.decision(name, args, res, self.brain, time.time() - t_exec,
                                 pose=[round(car_xy[0], 3), round(car_xy[1], 3), round(math.degrees(wrap(h)), 1)],
                                 imgs=[i for i in imgs + [after] if i], think_s=think_s)
        self.brain.note_result(f"{self.brain.decisions}. {name}({json.dumps(args)[:120]}) -> {res}")
        self.ai_log.append(f"#{self.brain.decisions} {name}: {why} -> {res}")
        self.ai_n += 1
        del self.ai_log[:-40]
        state["msg"] = f"AI: {why}" if why else f"AI: {name}"
        return name != "finish"

    def execute(self, name, args, g, car_xy, yaw, cands):
        global gm
        try:
            if name == "explore_to":
                i = int(args["frontier_id"]) - 1
                if not 0 <= i < len(cands):
                    return f"error: there is no frontier F{i + 1}"
                _, _, c, path = cands[i]
                self.target = c
                return self.drive_leg(c, path) or f"ok, drove toward F{i + 1}"
            if name == "go_to":
                tgt = (float(args["x"]), float(args["y"]))
                cx, cy = g.to_cell(*tgt)
                if not g.inside(cx, cy) or g.grid[cy, cx] != 0 or g.blocked[cy, cx]:
                    return "error: that point is not known free floor (unknown, wall margin or no-go)"
                path = g.plan(car_xy, tgt)
                if not path or len(path) < 2:
                    return "error: no path there"
                self.target = tgt
                return self.drive_leg(tgt, path) or "ok, drove toward the point"
            if name == "go_to_place":
                if loc.mode == "map":
                    return "error: named places only work on a finished map"
                pl = find_place(args.get("name", ""))
                if pl is None:
                    return "error: no place called %r. Known places: %s" % (args.get("name", ""), ", ".join(p["name"] for p in places) or "none")
                todo = f" Owner's instructions for this place - do them now: {pl['instructions']}" if pl.get("instructions") else ""
                if math.dist(car_xy, (pl["x"], pl["y"])) < 0.4:
                    return f"arrived at \"{pl['name']}\"." + todo
                res = self.execute("go_to", {"x": pl["x"], "y": pl["y"]}, g, car_xy, yaw, cands)
                if str(res).startswith("error"):
                    return res.replace("that point", f"the place \"{pl['name']}\"")
                left = math.dist(cur_pose()[0], (pl["x"], pl["y"]))
                return (f"arrived at \"{pl['name']}\"." + todo if left < 0.4 else f"ok, drove toward \"{pl['name']}\", {left:.1f} m left - call go_to_place again")
            if name == "turn_to":
                nav.face(math.radians(float(args["heading_deg"])))
                return "ok, turned"
            if name in ("drive_forward", "drive_back"):
                if not car:
                    return "error: needs the real car"
                dist = float(args["meters"])
                if not 0.05 <= dist <= 0.5:
                    return "error: meters must be 0.05 to 0.5"
                robot = CarRobot()
                if name == "drive_back":
                    return f"ok, backed up {robot.back(dist):.2f} m"
                return robot.advance(dist)
            if name == "push_object":
                if not car or not camera:
                    return "error: pushing needs the real car and camera"
                if loc.mode == "map":
                    return "error: pushing needs a finished map (navigation mode)"
                return push.push_object(CarRobot(), str(args["label"])[:30], (float(args["x"]), float(args["y"])))
            if name == "pin_photo":
                fr = camera.frame if camera else None
                if fr is None:
                    return "error: camera not available"
                pool = self.pins if loc.mode == "map" else photos
                entry = {"id": max([p["id"] for p in pool] + [0]) + 1, "x": float(car_xy[0]), "y": float(car_xy[1]),
                         "yaw": float(nav.heading(yaw)), "t": time.strftime("%H:%M:%S %d %b"), "note": str(args.get("note", ""))[:80]}
                if loc.mode == "map":
                    self.pins.append({**entry, "frame": fr})
                else:
                    store_pin(entry, fr)
                return "ok, photo pinned"
            if name == "add_no_go":
                return "error: the LLM may not create no-go boxes; avoid that area and respect the existing ones"
            if name == "finish":
                self.finished = str(args.get("summary", ""))
                return "finished"
            return f"error: unknown tool {name}"
        except (KeyError, ValueError, TypeError) as e:
            return f"error: bad arguments ({e})"

    def heuristic_step(self, cands):
        """Built-in explorer: nearest reachable frontier. Returns False when none is left."""
        if not cands:
            self.target = None
            state["msg"] = "Exploration complete - name the map and press Save & navigate"
            log("[explore] no reachable frontier left - done")
            return False
        _, _, c, path = cands[0]
        self.target = c
        state["msg"] = f"Exploring - heading to ({c[0]:.1f}, {c[1]:.1f})"
        self.drive_leg(c, path)
        return True

    def run(self):
        global gm
        nav_mode = loc.mode != "map"
        state["moving"], state["stop"] = True, False
        nav.hazard_check = self.hazard_now
        loc.strict = not nav_mode
        old_move_t, old_max_turn = nav.move_t, nav.max_turn
        nav.move_t, nav.max_turn = EXPLORE_MOVE_T, EXPLORE_TURN_T
        runlog = self.runlog = experiment.RunLog("ai" if self.brain else "heuristic", task=self.task, map=current_map[0],
                                                  places=[p["name"] for p in places], pose=[round(v, 3) for v in cur_pose()[0][:2]])
        try:
            self.snap(camera.frame if camera else None, "start")
            if nav.fwd_off is None:
                state["msg"] = "Calibrating…"
                nav.calibrate_forward()
            if not nav_mode and self.occ is None:
                self.occ = ex.Occupancy()
            while not state["stop"]:
                if loc.bad >= LOST_SCANS and not nav.wait_tracking():
                    raise RuntimeError("lost my position")
                x, y, yaw, _ = loc.wait_scans(2)             # the car is stopped here
                if not self.trail or math.dist(self.trail[-1], (x, y)) > 0.3:
                    self.trail.append((x, y))
                if nav_mode:
                    g, cands = gm, []
                else:
                    P = loc.live_points()
                    with loc.lock:
                        T = loc.T.copy()
                    self.occ.update((x, y), P @ T[:2, :2].T + T[:2, 3])
                    gm = g = ex.ExploreGrid(self.occ, [z["pts"] for z in self.zones], gm_radius + EXPLORE_CLEARANCE, (x, y))
                    cands = self.candidates(g, (x, y))
                runlog.step((x, y), g, self)
                if self.brain or os.environ.get("OBJECT_SCAN") == "1":
                    scan_objects()
                if self.brain:
                    state["msg"] = "AI is thinking…"
                    keep_going = self.ai_step(g, (x, y), yaw, cands)
                else:
                    keep_going = self.heuristic_step(cands)
                self.target = None if not keep_going else self.target
                if not keep_going:
                    if self.finished is not None:
                        state["msg"] = "AI finished: " + self.finished
                    break
            else:
                state["msg"] = "Stopped"
        except Exception as e:
            state["msg"] = f"Stopped: {e}"
            log(f"[explore] aborted: {e!r}")
        finally:
            self.snap(camera.frame if camera else None, "end")
            runlog.end(state["msg"], self)
            self.runlog = None
            if self.brain and self.brain.decisions:
                import llm_brain
                threading.Thread(target=llm_brain.reflect, args=(self.task, list(self.ai_log), state["msg"], log), daemon=True).start()
            nav.hazard_check = None
            loc.strict = False
            nav.move_t, nav.max_turn = old_move_t, old_max_turn
            car.send("S")
            state["moving"], state["path"], self.on, self.target = False, [], False, None


explorer = Explorer()


class CarRobot:
    """push.py's view of the real car. All motion goes through the existing LiDAR pose feedback, no-go zones,
    obstacle detours and the camera ledge check - the skill adds no new way to move."""
    @staticmethod
    def log(msg):
        log(msg)
        if explorer.runlog:
            explorer.runlog.event("push", msg=str(msg)[:300])

    @property
    def left_sign(self):
        return nav.turn_sign

    def pose(self):
        x, y, yaw, _ = loc.wait_scans()
        return x, y, wrap(nav.heading(yaw))

    def stopped(self):
        return state["stop"]

    def see(self, label):
        time.sleep(0.4)                                   # let the camera image settle after a move
        fr = camera.frame if camera else None
        if fr is None:
            return None
        explorer.snap(fr, "look_" + label.replace(" ", "_"))
        try:
            return obj_mod.find(fr, label)
        except Exception as e:
            log(f"[push] vision call failed: {e!r}")
            return None

    def turn_to(self, heading):
        nav.face(heading)

    def forward(self, secs):
        if nav.fwd_off is None:
            nav.calibrate_forward()
        car.pulse("F", secs)

    def back(self, metres):
        x0, y0, _ = self.pose()
        moved = 0.0
        for _ in range(8):
            if state["stop"] or moved >= metres or nav.clear_dir(math.pi, 0.40) is not None:
                break
            car.pulse("B", 0.4)
            x, y, _ = self.pose()
            moved = math.hypot(x - x0, y - y0)
        return moved

    def advance(self, metres):
        """Drive forward about `metres` with every safety check; -> result text."""
        if nav.fwd_off is None:
            nav.calibrate_forward()
        x0, y0, _ = self.pose()
        for _ in range(12):
            x, y, _ = self.pose()
            got = math.hypot(x - x0, y - y0)
            if state["stop"]:
                return "stopped"
            if got >= metres - 0.03:
                return f"ok, drove {got:.2f} m"
            if explorer.hazard_now():
                explorer.add_hazard_zone()
                self.back(0.3)
                return f"error: the ledge check stopped the car after {got:.2f} m; a no-go box was added ahead"
            if gm.in_zone((x, y)):
                self.back(0.3)
                return "error: the car is inside a no-go zone"
            ob = nav.blocked_ahead(0.25)
            if ob is not None:
                return f"error: something is {ob:.2f} m ahead; drove {got:.2f} m"
            car.pulse("F", min(0.35, max(0.15, (metres - got) / 0.25)))
        x, y, _ = self.pose()
        return f"ok, drove {math.hypot(x - x0, y - y0):.2f} m"

    def drive_to(self, x, y):
        """Plan and drive to a map point in legs; -> None when there, else why not."""
        for _ in range(8):
            if state["stop"]:
                return "stopped"
            px, py, _ = self.pose()
            if math.dist((px, py), (x, y)) < 0.15:
                return None
            path = gm.plan((px, py), (x, y))
            if not path or len(path) < 2:
                return "no path"
            leg = ex.truncate(path, EXPLORE_LEG)
            state["path"], state["wp"] = leg, 1
            try:
                nav.go(leg)
            except Hazard:
                explorer.add_hazard_zone()
                nav.back_off(0.35)
                return "the ledge check stopped the car; a no-go box was added"
            except RuntimeError as e:
                nav.back_off()
                return str(e)
        return "did not arrive"

    def free(self, x, y):
        cx, cy = gm.to_cell(x, y)
        return bool(gm.inside(cx, cy) and gm.grid[cy, cx] == 0 and not gm.blocked[cy, cx])

    def line_ok(self, a, b):
        return not gm.hits_zone(a, b)

    def in_zone(self, x, y):
        return bool(gm.in_zone((x, y)))

    def hazard(self):
        return explorer.hazard_now()

    def obstacle_ahead(self, limit):
        return nav.blocked_ahead(limit)


chat = None
chat_lock = threading.Lock()


def chat_turn(text, on_text=None):
    """One chat turn (shared by the web page and Telegram): -> (reply, actions)."""
    global chat
    import llm_brain
    with chat_lock:
        if chat is None:
            chat = llm_brain.Chat(log)
        status, png, cam = chat_context()
        actions = []
        reply = chat.reply(text, png, cam, status, lambda n, a: chat_tool(n, a, actions), on_text=on_text)
    log_chat(text, reply, status, actions)
    return reply, actions


def log_chat(question, reply, status, actions):
    """Every chat / status question with the robot state it was answered from (evidence for the status-update claim)."""
    if os.environ.get("EXP_LOG", "1") == "0":
        return
    try:
        os.makedirs("runs", exist_ok=True)
        with open(f"runs/chat_{time.strftime('%Y%m%d')}.jsonl", "a") as f:
            f.write(json.dumps({"time": time.strftime("%H:%M:%S"), "question": question, "reply": reply, "status": status,
                                "actions": actions}) + "\n")
    except OSError as e:
        log(f"[chat] could not log: {e}")


def ai_start(task, radius=5.0):
    """Start an AI run. -> (ok, error)."""
    if not car or not loc:
        return False, "AI control needs the real car"
    if state["moving"]:
        return False, "Already busy"
    if loc.mode == "nav" and loc.bad >= LOST_SCANS:
        return False, "Localisation is lost - put the car where it can see walls first"
    try:
        import llm_brain
        brain = llm_brain.Brain(log)
    except Exception as e:
        return False, f"Cannot start the AI: {e} (set ANTHROPIC_API_KEY before starting the server)"
    explorer.start(max(1.0, min(12.0, float(radius))), brain, str(task)[:500])
    return True, ""


def robot_stop():
    if state["moving"] and explorer.runlog:
        explorer.runlog.event("user_stop")
    state["stop"] = True
    if car:
        car.send("S")


def chat_context():
    """Status text, map image (or None) and camera frame for a chat turn."""
    pose, yaw = cur_pose()
    h = nav.heading(yaw) if loc else yaw
    mode = "mapping a new map" if loc and loc.mode == "map" else "navigating a saved map"
    st_ = [f"Mode: {mode}. Robot at ({pose[0]:.2f}, {pose[1]:.2f}) m facing {math.degrees(wrap(h)):.0f} deg (0 = +x, 90 = +y).",
           f"Robot is {'moving (' + state['msg'] + ')' if state['moving'] else 'stopped'}.",
           f"Localisation: {'LOST' if state['lost'] else 'ok'}."]
    png = None
    mapping = loc and loc.mode == "map"
    zl = explorer.zones if mapping else zones
    pins = explorer.pins if mapping else photos
    st_.append(objects_line(object_pool()))
    if not mapping:
        st_.append(places_line())
    st_.append(f"No-go boxes: {len(zl)}. Photo pins: " + ("; ".join(f"{p['id']}={p.get('note') or 'photo'} at ({p['x']:.1f},{p['y']:.1f})" for p in pins[-10:]) or "none") + ".")
    if gm is not None and (not mapping or explorer.occ is not None):
        try:
            import llm_brain
            cells = gm.grid.copy()
            cells[gm.occ_raw] = 3
            png = llm_brain.render_map(cells, gm.min, pose, h, (), [z["pts"] for z in zl],
                                       [(p["id"], (p["x"], p["y"])) for p in pins], explorer.trail[-60:] if mapping else ())
        except Exception as e:
            log(f"[chat] map render failed: {e!r}")
    return "\n".join(st_), png, (camera.frame if camera else None)


def chat_tool(name, args, actions):
    pose, yaw = cur_pose()
    if name == "propose_task":
        actions.append({"type": "propose", "task": str(args.get("task", ""))[:500]})
        return "Shown to the owner with a Start button; the robot will not move until they press it."
    if name == "stop_robot":
        state["stop"] = True
        if car:
            car.send("S")
        actions.append({"type": "stopped"})
        return "Stopped."
    if name == "pin_photo":
        fr = camera.frame if camera else None
        if fr is None or not loc:
            return "error: camera not available"
        mapping = loc.mode == "map"
        pool = explorer.pins if mapping else photos
        entry = {"id": max([p["id"] for p in pool] + [0]) + 1, "x": float(pose[0]), "y": float(pose[1]), "yaw": float(nav.heading(yaw)),
                 "t": time.strftime("%H:%M:%S %d %b"), "note": str(args.get("note", ""))[:80]}
        if mapping:
            explorer.pins.append({**entry, "frame": fr})
        else:
            store_pin(entry, fr)
        actions.append({"type": "pinned", "note": entry["note"]})
        return "Photo pinned."
    return "error: unknown tool"


def cur_pose():
    if loc:
        x, y, yaw, _ = loc.pose()
        return [x, y], yaw
    return state["pose"], state["yaw"]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/map":
            cells = gm.grid.copy()
            cells[gm.occ_raw] = 3
            self._send({"w": gm.w, "h": gm.h, "res": RES, "min": list(gm.min),
                        "cells": base64.b64encode(cells.tobytes()).decode()})
        elif self.path == "/video":
            if camera is None:
                self.send_error(503, "camera unavailable")
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for fr in camera.frames():
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(fr) + fr + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/photos":
            self._send({"photos": photos})
        elif self.path == "/places":
            self._send({"places": places})
        elif self.path.startswith("/placeimg/"):
            try:
                with open(place_img_file(self.path[10:].split(".")[0]), "rb") as f:
                    self._send(f.read(), "image/jpeg")
            except (OSError, ValueError):
                self.send_error(404)
        elif self.path.startswith("/photo/"):
            try:
                with open(photo_file(self.path[7:].split(".")[0]), "rb") as f:
                    self._send(f.read(), "image/jpeg")
            except (OSError, ValueError):
                self.send_error(404)
        elif self.path == "/objects":
            self._send({"objects": object_pool()})
        elif self.path == "/zones":
            self._send({"zones": zones})
        elif self.path == "/speed":
            self._send({"move": car.move_speed if car else state["sim_speed"][0],
                        "turn": car.turn_speed if car else state["sim_speed"][1]})
        elif self.path == "/maps":
            self._send({"maps": list_maps(), "current": current_map[0]})
        elif self.path == "/mappts":
            p, skipped = loc.map_points() if loc else (np.zeros((0, 2), np.float32), 0)
            pose, yaw = cur_pose()
            self._send({"n": len(p), "pts": base64.b64encode(p.tobytes()).decode(),
                        "pose": pose, "yaw": nav.heading(yaw), "skipped": skipped, "lost": state["lost"],
                        "explore": explorer.on, "msg": state["msg"], "zones": [z["pts"] for z in explorer.zones],
                        "target": explorer.target, "objects": explorer.objects})
        elif self.path == "/state":
            pose, yaw = cur_pose()
            self._send({"objects": object_pool(), "explore": explorer.on, "ai": {"on": bool(explorer.on and explorer.brain), "log": explorer.ai_log[-8:], "n": explorer.ai_n,
                        "tokens": [explorer.brain.tokens_in, explorer.brain.tokens_out, explorer.brain.tokens_cached] if explorer.brain else [0, 0, 0]},
                        "learn": loc.learn if loc else False,
                        "added": loc.added if loc else 0, "removed": loc.removed if loc else 0,
                        "map_version": grid_ver[0], "pose": pose, "yaw": nav.heading(yaw) if loc else yaw, "path": state["path"],
                        "moving": state["moving"], "arrived": state["arrived"], "mode": loc.mode if loc else "nav", "msg": state["msg"], "lost": state["lost"]})
        else:
            self.send_error(404)

    def do_POST(self):
        global gm
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/plan":
            pose, _ = cur_pose()
            path = gm.plan(pose, data["goal"])
            if path is None:
                return self._send({"ok": False, "error": "No path to that point"})
            path[0] = tuple(pose)
            length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
            state["path"], state["wp"] = path, 1
            self._send({"ok": True, "path": path, "length": length})
        elif self.path == "/setpath":
            pose, _ = cur_pose()
            pts = [tuple(p) for p in data["points"]]
            path = [tuple(pose)] + pts
            # blocked cells right at the car's own position don't count
            bad = [i for i, (a, b) in enumerate(zip(path, path[1:]))
                   if any(gm.blocked[y, x] and math.dist(gm.to_world(x, y), path[0]) > 0.4
                          for x, y in gm._line(gm.to_cell(*a), gm.to_cell(*b)))]
            length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
            in_zone = [i for i, (a, b) in enumerate(zip(path, path[1:])) if gm.hits_zone(a, b)]
            state["path"], state["wp"] = path, 1
            state["zone_hit"] = bool(in_zone)
            self._send({"ok": True, "path": path, "length": length, "bad": bad, "zone": in_zone})
        elif self.path == "/mode":
            if not loc:
                return self._send({"ok": False, "error": "Mapping needs the real LiDAR (not --sim)"})
            if state["moving"]:
                return self._send({"ok": False, "error": "Stop navigation first"})
            if data["mode"] == "map":
                loc.start_mapping()
                explorer.reset()
                explorer.prev_gm = gm
            else:
                loc.cancel_mapping()
                if explorer.prev_gm is not None:
                    gm, explorer.prev_gm = explorer.prev_gm, None
                explorer.reset()
            self._send({"ok": True})
        elif self.path == "/speed":
            def clamp(v):
                return None if v is None else round(max(0.1, min(1.0, float(v))), 2)
            mv, tn = clamp(data.get("move")), clamp(data.get("turn"))
            if car:
                car.set_speeds(mv, tn)
            else:
                state["sim_speed"] = [mv or state["sim_speed"][0], tn or state["sim_speed"][1]]
            self._send({"ok": True})
        elif self.path == "/zones":
            global zones
            if state["moving"]:
                return self._send({"ok": False, "error": "Stop the car before editing zones"})
            if data.get("action") == "add":
                pts = [[float(x), float(y)] for x, y in data.get("pts", [])]
                if len(pts) < 3:
                    return self._send({"ok": False, "error": "A zone needs at least 3 corners"})
                zones.append({"id": max([z["id"] for z in zones] + [0]) + 1, "pts": pts})
            elif data.get("action") == "delete":
                zones = [z for z in zones if z["id"] != data.get("id")]
            else:
                return self._send({"ok": False, "error": "unknown action"})
            gm.set_zones([z["pts"] for z in zones])
            save_zones()
            state["path"], state["wp"] = [], 1          # an old path may now cross the zone
            self._send({"ok": True, "zones": zones})
        elif self.path == "/objects":
            pool = object_pool()
            if data.get("action") == "delete":
                with obj_lock:
                    pool[:] = [o for o in pool if o["id"] != data.get("id")]
                if pool is map_objects:
                    save_objects()
                return self._send({"ok": True, "objects": pool})
            why = scan_objects(force=True)
            self._send({"ok": not why, "error": why, "objects": pool})
        elif self.path == "/place":
            if data.get("action") == "delete":
                places[:] = [p for p in places if p["id"] != data.get("id")]
                try:
                    os.remove(place_img_file(data.get("id")))
                except (OSError, ValueError, TypeError):
                    pass
                save_places()
                return self._send({"ok": True, "places": places})
            name = str(data.get("name", "")).strip()[:40]
            if not name:
                return self._send({"ok": False, "error": "A place needs a name"})
            desc = str(data.get("description", "")).strip()[:200]
            instr = str(data.get("instructions", "")).strip()[:500]
            icon = str(data.get("icon", "")).strip()[:4]
            if data.get("action") == "update":
                pl = next((p for p in places if p["id"] == data.get("id")), None)
                if pl is None:
                    return self._send({"ok": False, "error": "Unknown place"})
                if any(p is not pl and p["name"].lower() == name.lower() for p in places):
                    return self._send({"ok": False, "error": f"There is already a place called {name}"})
                pl["name"], pl["description"], pl["instructions"], pl["icon"] = name, desc, instr, icon
                bad = set_place_image(pl, data)
                if bad:
                    return self._send({"ok": False, "error": bad})
                save_places()
                return self._send({"ok": True, "places": places})
            if loc is None or loc.mode != "nav" or gm is None:
                return self._send({"ok": False, "error": "Places need a loaded map (not while mapping)"})
            if any(p["name"].lower() == name.lower() for p in places):
                return self._send({"ok": False, "error": f"There is already a place called {name}"})
            try:
                x, y = float(data["x"]), float(data["y"])
            except (KeyError, TypeError, ValueError):
                return self._send({"ok": False, "error": "Click a point on the map"})
            cx, cy = gm.to_cell(x, y)
            if not gm.inside(cx, cy) or gm.grid[cy, cx] != 0 or gm.blocked[cy, cx]:
                return self._send({"ok": False, "error": "That spot is not known free floor (unknown, wall margin or no-go) - pick a white area"})
            places.append({"id": max([p["id"] for p in places] + [0]) + 1, "name": name, "description": desc, "instructions": instr, "icon": icon, "img": None,
                           "x": round(x, 3), "y": round(y, 3), "t": time.strftime("%H:%M:%S %d %b")})
            bad = set_place_image(places[-1], data)
            save_places()
            self._send({"ok": True, "places": places})
        elif self.path == "/photo":
            global photos
            if data.get("action") == "delete":
                photos = [p for p in photos if p["id"] != data.get("id")]
                try:
                    os.remove(photo_file(data.get("id")))
                except (OSError, ValueError):
                    pass
                save_photos()
                return self._send({"ok": True, "photos": photos})
            frame = camera.frame if camera else None
            if frame is None:
                return self._send({"ok": False, "error": "Camera not available"})
            if loc is None or loc.mode != "nav":
                return self._send({"ok": False, "error": "Photo pins need a loaded map (not while mapping)"})
            pose, yaw = cur_pose()
            pid = max([p["id"] for p in photos] + [0]) + 1
            os.makedirs(photos_base[0] + ".photos", exist_ok=True)
            with open(photo_file(pid), "wb") as f:
                f.write(frame)
            photos.append({"id": pid, "x": float(pose[0]), "y": float(pose[1]), "yaw": float(nav.heading(yaw)),
                           "t": time.strftime("%H:%M:%S %d %b"), "note": str(data.get("note", ""))[:80]})
            save_photos()
            self._send({"ok": True, "photos": photos})
        elif self.path == "/explore":
            if not car or not loc or loc.mode != "map":
                return self._send({"ok": False, "error": "Auto-explore needs the real car and mapping mode"})
            if data.get("on"):
                if state["moving"]:
                    return self._send({"ok": False, "error": "Already busy"})
                explorer.start(max(1.0, min(12.0, float(data.get("radius", 5.0)))))
            else:
                state["stop"] = True
                car.send("S")
            self._send({"ok": True})
        elif self.path == "/chat":
            global chat
            if data.get("reset"):
                if chat:
                    chat.reset()
                return self._send({"ok": True})
            text = str(data.get("message", "")).strip()[:2000]
            if not text:
                return self._send({"ok": False, "error": "Empty message"})
            # streamed reply: one JSON object per line, text deltas first, then a final {"done": true, ...}
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def emit(obj):
                self.wfile.write((json.dumps(obj) + "\n").encode())
                self.wfile.flush()
            try:
                _, actions = chat_turn(text, on_text=lambda d: emit({"t": d}))
                emit({"done": True, "actions": actions})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                hint = " (set ANTHROPIC_API_KEY before starting the server)" if "api_key" in str(e).lower() or "auth" in str(e).lower() else ""
                try:
                    emit({"error": f"{e}{hint}"})
                except (BrokenPipeError, ConnectionResetError):
                    pass
        elif self.path == "/ai":
            if not data.get("on"):
                robot_stop()
                return self._send({"ok": True})
            ok, err = ai_start(data.get("task", ""), data.get("radius", 5.0))
            self._send({"ok": ok, "error": err})
        elif self.path == "/learn":
            if loc:
                loc.learn = bool(data.get("on"))
            self._send({"ok": True})
        elif self.path == "/saveupdated":
            name = str(data.get("name", ""))
            if not loc or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
                return self._send({"ok": False, "error": "Use letters, digits, - or _ only"})
            os.makedirs(MAP_DIR, exist_ok=True)
            path = f"{MAP_DIR}/{name}.pcd"
            if os.path.exists(path):
                return self._send({"ok": False, "error": f"{path} already exists - pick another name"})
            try:
                loc.save_current(path)
            except Exception as e:
                return self._send({"ok": False, "error": str(e)})
            current_map[0] = path
            old_base = photos_base[0]
            zones_file[0] = zones_path(path)          # the saved copy keeps its zones
            save_zones()
            photos_base[0] = os.path.splitext(path)[0]    # ...and its photo pins
            if photos and os.path.isdir(old_base + ".photos"):
                shutil.copytree(old_base + ".photos", photos_base[0] + ".photos", dirs_exist_ok=True)
            save_photos()
            save_objects()
            save_places()
            self._send({"ok": True, "file": path})
        elif self.path == "/deletemap":
            f = data.get("file")
            if f not in list_maps():
                return self._send({"ok": False, "error": "Unknown map"})
            if f == current_map[0]:
                return self._send({"ok": False, "error": "That map is loaded - switch to another map first"})
            if state["moving"] or (loc and loc.mode == "map"):
                return self._send({"ok": False, "error": "Stop navigating / mapping first"})
            base = os.path.splitext(f)[0]
            trash = f"{MAP_DIR}/trash/{time.strftime('%Y%m%d-%H%M%S')}"     # moved, not erased - recoverable
            os.makedirs(trash, exist_ok=True)
            for p in [f] + glob.glob(glob.escape(base) + ".*"):     # .zones/.photos/.photos.json/.objects/.places...
                if os.path.exists(p):
                    shutil.move(p, trash)
            log(f"[maps] deleted {f} (moved to {trash})")
            self._send({"ok": True, "maps": list_maps(), "current": current_map[0]})
        elif self.path == "/selectmap":
            f = data.get("file")
            if f not in list_maps():
                return self._send({"ok": False, "error": "Unknown map"})
            if state["moving"] or (loc and loc.mode == "map"):
                return self._send({"ok": False, "error": "Stop navigating / mapping first"})
            try:
                load_zones(f)
                new = build_gm(f)
                if loc:
                    loc.set_map(f)
            except Exception as e:
                return self._send({"ok": False, "error": str(e)})
            gm, current_map[0] = new, f
            state["path"] = []
            self._send({"ok": True})
        elif self.path == "/drive":
            if not car or state["moving"] or not loc or loc.mode != "map":
                return self._send({"ok": False})
            d = data.get("dir", "S")
            if d in ("F", "B", "L", "R"):
                drive_t[0] = time.time()
            else:
                drive_t[0], d = 0.0, "S"
            car.send(d)
            self._send({"ok": True})
        elif self.path == "/savemap":
            name = str(data.get("name", ""))
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
                return self._send({"ok": False, "error": "Use letters, digits, - or _ only"})
            os.makedirs(MAP_DIR, exist_ok=True)
            path = f"{MAP_DIR}/{name}.pcd"
            if os.path.exists(path):
                return self._send({"ok": False, "error": f"{path} already exists - pick another name"})
            try:
                loc.save_map(path)
                load_zones(path)                  # a freshly mapped map has no zones yet...
                if explorer.zones:                # ...except the ledge boxes auto-explore drew
                    set_zones_list(list(explorer.zones))
                    zones_file[0] = zones_path(path)
                    save_zones()
                for pin in explorer.pins:         # ...and the photos pinned during the run
                    store_pin(pin, pin["frame"])
                map_objects[:] = explorer.objects     # ...and the furniture it named
                photos_base[0] = os.path.splitext(path)[0]
                save_objects()
                new = build_gm(path)
                loc.set_map(path)                 # switches back to navigation
            except Exception as e:
                if os.path.exists(path):
                    os.remove(path)               # no half-saved map blocking a retry
                return self._send({"ok": False, "error": "Save failed: " + str(e)})
            gm = new
            explorer.prev_gm, explorer.zones, explorer.pins, explorer.objects = None, [], [], []
            state["path"] = []
            current_map[0] = path
            self._send({"ok": True, "file": path})
        elif self.path == "/go":
            if state["moving"] or not state["path"]:
                return self._send({"ok": False, "error": "Busy or no planned path"})
            p = state["path"]
            if any(gm.hits_zone(a, b) for a, b in zip(p[state["wp"] - 1:], p[state["wp"]:])):
                return self._send({"ok": False, "error": "That path goes through a no-go zone - edit it first"})
            threading.Thread(target=follow, args=(list(state["path"]),), daemon=True).start()
            self._send({"ok": True})
        elif self.path == "/stop":
            state["stop"] = True
            if car:
                car.send("S")
            self._send({"ok": True})
        else:
            self.send_error(404)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Navigator</title>
<style>
 body{margin:0;font:15px system-ui,sans-serif;background:#12151a;color:#e6e8eb;display:flex;flex-direction:column;height:100vh}
 header{display:flex;gap:10px;align-items:center;padding:10px 14px;background:#1b2028;flex-wrap:wrap}
 h1{font-size:16px;margin:0 auto 0 0}
 button{border:0;border-radius:6px;padding:9px 18px;font-size:15px;cursor:pointer;color:#fff}
 #speedbar{display:flex;gap:26px;padding:6px 14px;background:#161a20;flex-wrap:wrap;color:#9aa4b2;font-size:14px}
 #speedbar input{vertical-align:middle;width:170px;margin:0 8px}#speedbar b{color:#e6e8eb;display:inline-block;width:34px}
 #zbtn{background:#c92a2a}#zbtn.on{background:#fa5252}
 #zbar{display:none;gap:8px;align-items:center;flex-wrap:wrap;padding:6px 14px;background:#2a1517;font-size:14px}
 #zbar button{padding:5px 12px;font-size:13px;background:#3b4252}
 #zbar .tool.on{background:#fa5252}
 .chip{background:#5c1f23;border-radius:14px;padding:3px 6px 3px 12px;display:inline-flex;gap:6px;align-items:center}
 .chip i{cursor:pointer;font-style:normal;background:#8b2c31;border-radius:50%;width:18px;height:18px;text-align:center;line-height:18px}
 #maps{padding:8px;border-radius:6px;border:0;background:#3b4252;color:#fff;font-size:14px;max-width:210px}
 #mode{background:#a5521f}
 #chatbtn{position:fixed;left:16px;bottom:16px;z-index:5;background:#7048e8;border-radius:24px;padding:12px 18px;box-shadow:0 2px 10px #0008}
 #chat{display:none;position:fixed;left:16px;bottom:70px;width:min(400px,calc(100vw - 32px));height:min(60vh,520px);z-index:6;background:#1b2028;border:1px solid #2e3644;border-radius:12px;flex-direction:column;box-shadow:0 4px 24px #000a}
 #chat header{padding:8px 12px;border-radius:12px 12px 0 0;font-size:14px}#chat header button{padding:3px 10px;font-size:12px;background:#3b4252}
 #chatlog{flex:1;overflow:auto;padding:10px;display:flex;flex-direction:column;gap:8px}
 .m{max-width:88%;padding:8px 11px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word;font-size:14px;line-height:1.35}
 .m.u{align-self:flex-end;background:#3b5bdb}.m.a{align-self:flex-start;background:#2b3241}.m.s{align-self:center;background:none;color:#9aa4b2;font-size:12px}
 .m button{margin-top:6px;background:#1f9d55;padding:6px 12px;font-size:13px}
 #chatform{display:flex;gap:6px;padding:8px;border-top:1px solid #2e3644}
 #chatin{flex:1;padding:9px;border-radius:8px;border:0;background:#12151a;color:#e6e8eb;font-size:14px}
 #pad{display:none;position:fixed;right:18px;bottom:18px;grid-template-columns:repeat(3,64px);gap:6px}
 #pad button{height:56px;background:#3b4252;font-size:20px;padding:0}
 #savebar{display:none;gap:8px;align-items:center}
 #savebar input{padding:8px;border-radius:6px;border:0;width:130px}
 #savebtn{background:#1f9d55}
 #draw,#undo,#clear{background:#3b4252}#draw.on{background:#5c7cfa}
 #go{background:#1f9d55}#stop{background:#d64545}button:disabled{opacity:.4;cursor:default}
 #status{color:#9aa4b2}
 main{flex:1;display:flex;justify-content:center;align-items:center;overflow:hidden;padding:8px;gap:8px}
 #cam{display:none;width:640px;max-width:45vw;border-radius:8px;align-self:flex-start;background:#000}
 canvas{max-width:100%;max-height:100%;border-radius:8px;cursor:crosshair;background:#999}
</style></head><body>
<header><h1>Robot Navigator</h1><span id="status">Click the map to choose a destination</span>
<select id="maps" title="Choose map"></select><button id="delmap" title="See all maps and delete the ones you do not need" style="background:#3b4252">🗂 Manage maps</button><button id="mode">Mode: Navigate</button><button id="zbtn">No-go zones: off</button><button id="draw">Draw path: off</button><button id="undo" disabled>Undo</button><button id="clear" disabled>Clear</button><button id="go" disabled>Go</button><button id="stop">Stop</button><button id="snap" style="background:#3b4252">📷 Pin photo</button><button id="placebtn" style="background:#3b4252" title="Click the map to drop a named place the AI can drive to">📍 Pin place: off</button><button id="objscan" style="background:#3b4252" title="Look at the camera view, name the furniture and place it on the map">🔎 Identify objects</button></header>
<div id="speedbar"><label>Forward <input type="range" id="mv" min="0.1" max="1" step="0.05" value="0.3"><b id="mvv">0.30</b></label>
 <label style="margin-left:auto">Map: <b id="mstat" style="width:auto">–</b></label>
 <button id="learn" style="padding:4px 12px;font-size:13px;background:#2f9e44">Learn changes: on</button>
 <button id="saveupd" style="padding:4px 12px;font-size:13px;background:#3b4252">Save updated map</button>
 <label>Turn <input type="range" id="tn" min="0.1" max="1" step="0.05" value="0.3"><b id="tnv">0.30</b></label></div>
<div id="zbar"><button id="ztbox" class="tool on">▭ Box</button><button id="ztpoly" class="tool">⬠ Polygon</button>
 <span id="zhint">Drag on the map to draw a box the car must NOT enter.</span>
 <button id="zclose" style="display:none">Close zone</button><button id="zundo" style="display:none">Undo corner</button><span id="zlist"></span></div>
<div id="aibar" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:6px 14px;background:#1d2433">
 <button id="aibtn" style="background:#7048e8">🧠 AI</button>
 <input id="aitask" placeholder="Task for the AI, e.g. map the living room first, skip the garage, pin a photo in each room" style="flex:1;min-width:260px;padding:8px;border-radius:6px;border:0">
 <span id="aistat" style="color:#9aa4b2;font-size:13px"></span></div>
<div id="ailog" style="display:none;padding:6px 14px;background:#161b27;color:#b8c1d1;font-size:13px;max-height:110px;overflow:auto;white-space:pre-wrap"></div>
<div id="savebar" style="padding:6px 14px;background:#1b2028"><button id="explore" style="background:#1c7ed6">🧭 Auto-explore</button>
 <label style="color:#9aa4b2">radius <input id="xr" type="number" min="1" max="12" step="0.5" value="5" style="width:56px;padding:6px;border-radius:6px;border:0"> m</label>
 <span>…or drive the car around every room yourself, then:</span>
 <input id="mname" placeholder="map name" value="my_map"><button id="savebtn">Save &amp; navigate</button></div>
<div id="pad"><span></span><button data-d="F">▲</button><span></span><button data-d="L">◀</button><button data-d="S">■</button><button data-d="R">▶</button><span></span><button data-d="B">▼</button><span></span></div>
<main><canvas id="c"></canvas><img id="cam" alt="camera"></main>
<button id="chatbtn">💬 Chat</button>
<div id="chat"><header style="display:flex;gap:8px;align-items:center;background:#242b38"><b style="flex:1">Chat with the robot</b><button id="chatnew">New</button><button id="chatx">×</button></header>
 <div id="chatlog"></div>
 <form id="chatform"><input id="chatin" placeholder="Ask what it sees, or give it a task…" autocomplete="off"><button id="chatsend" style="background:#7048e8">Send</button></form></div>
<div id="pv" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9;align-items:center;justify-content:center;flex-direction:column;gap:10px">
 <img id="pvimg" style="max-width:90vw;max-height:78vh;border-radius:8px"><div id="pvt" style="color:#9aa4b2"></div>
 <div style="display:flex;gap:10px"><button id="pvdel" style="background:#d64545">Delete pin</button><button id="pvclose" style="background:#3b4252">Close</button></div></div>
<div id="mm" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9;align-items:center;justify-content:center">
 <div style="background:#2e3440;padding:16px;border-radius:8px;display:flex;flex-direction:column;gap:8px;min-width:320px;max-width:90vw;max-height:80vh">
  <div style="color:#9aa4b2">Saved maps (deleted maps go to maps/trash, not erased)</div><div id="mmlist" style="overflow:auto;display:flex;flex-direction:column;gap:6px"></div>
  <button id="mmclose" style="background:#3b4252">Close</button></div></div>
<div id="plm" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9;align-items:center;justify-content:center">
 <div style="background:#2e3440;padding:16px;border-radius:8px;display:flex;flex-direction:column;gap:8px;min-width:280px">
  <div style="color:#9aa4b2">Edit place</div><input id="plname" placeholder="name" maxlength="40"><input id="pldesc" placeholder="description" maxlength="200"><div style="display:flex;gap:6px;align-items:center"><input id="plicon" placeholder="icon" maxlength="4" style="width:60px;text-align:center;font-size:20px"><span id="plpal" style="display:flex;flex-wrap:wrap;gap:2px"></span></div>
  <div style="display:flex;gap:8px;align-items:center"><img id="plthumb" style="display:none;width:56px;height:56px;object-fit:cover;border-radius:50%"><input id="plfile" type="file" accept="image/*" style="max-width:190px"><button id="plnoimg" type="button" style="background:#3b4252">Remove picture</button></div>
  <textarea id="plinstr" rows="4" maxlength="500" placeholder="Instructions for the AI at this pin, e.g. take a photo, turn slowly all round, then finish" style="padding:8px;border-radius:6px;border:0"></textarea>
  <div style="display:flex;gap:10px"><button id="plsave">Save</button><button id="pldel" style="background:#d64545">Delete pin</button><button id="plclose" style="background:#3b4252">Cancel</button></div></div></div>
<script>
{const im=document.getElementById('cam');im.onload=()=>im.style.display='block';im.onerror=()=>im.style.display='none';im.src='/video';}
const S=4, cv=document.getElementById('c'), ctx=cv.getContext('2d');
const st=document.getElementById('status'), go=document.getElementById('go');
let mapping=false, mpts=null, mpose=[0,0], myaw=0, drawMode=false, pts=[], bad=[], M, base, goal=null, path=[], pose=[0,0], yaw=0, moving=false;
const COL=[[255,255,255],[153,153,153],[210,210,210],[0,0,0]];
const w2p=(x,y)=>[(x-M.min[0])/M.res*S,(M.h-(y-M.min[1])/M.res)*S];
const p2w=(px,py)=>[M.min[0]+px/S*M.res,M.min[1]+(M.h-py/S)*M.res];
let flashUntil=0;
const flash=m=>{st.textContent=m;flashUntil=Date.now()+8000};
const post=(u,b)=>fetch(u,{method:'POST',body:JSON.stringify(b||{})}).then(r=>r.json());

async function loadMap(){
  M=await (await fetch('/map')).json();
  const raw=Uint8Array.from(atob(M.cells),c=>c.charCodeAt(0));
  cv.width=M.w*S; cv.height=M.h*S;
  const o=document.createElement('canvas'); o.width=M.w; o.height=M.h;
  const oc=o.getContext('2d'), im=oc.createImageData(M.w,M.h);
  for(let y=0;y<M.h;y++)for(let x=0;x<M.w;x++){
    const c=COL[raw[y*M.w+x]], i=((M.h-1-y)*M.w+x)*4;
    im.data.set([c[0],c[1],c[2],255],i);
  }
  oc.putImageData(im,0,0); base=o; draw();
}
const sl={move:document.getElementById('mv'),turn:document.getElementById('tn')};
const sv={move:document.getElementById('mvv'),turn:document.getElementById('tnv')};
let dragging=false, speedTimer=null;
for(const k of ['move','turn']){
  sl[k].oninput=()=>{dragging=true;sv[k].textContent=(+sl[k].value).toFixed(2);
    clearTimeout(speedTimer);speedTimer=setTimeout(async()=>{await post('/speed',{[k]:+sl[k].value});dragging=false},150)};
}
async function pollSpeed(){
  if(dragging)return;
  const r=await (await fetch('/speed')).json();
  for(const k of ['move','turn'])if(r[k]!=null&&document.activeElement!==sl[k]){sl[k].value=r[k];sv[k].textContent=(+r[k]).toFixed(2)}
}
let lastVer=-1, learnOn=true;
const learnBtn=document.getElementById('learn');
learnBtn.onclick=async()=>{learnOn=!learnOn;await post('/learn',{on:learnOn});showLearn()};
function showLearn(){learnBtn.textContent='Learn changes: '+(learnOn?'on':'off');learnBtn.style.background=learnOn?'#2f9e44':'#6b7280'}
document.getElementById('saveupd').onclick=async()=>{
  const cur=mapSel.value.replace(/^maps\//,'').replace(/\.pcd$/,'');
  const name=prompt('Save the updated map as (letters, digits, - _):',cur+'_updated');
  if(!name)return;
  const r=await post('/saveupdated',{name});
  if(!r.ok){flash(r.error);return}
  await loadMapList();flash('Saved '+r.file);
};
let photos=[], pvId=null, mzones=[], mtarget=null;
const snapBtn=document.getElementById('snap'), pv=document.getElementById('pv');
async function loadPhotos(){photos=(await (await fetch('/photos')).json()).photos;draw()}
snapBtn.onclick=async()=>{const r=await post('/photo',{});if(!r.ok)return flash(r.error);photos=r.photos;flash('Photo pinned at the car\'s position');draw()};
function showPhoto(p){pvId=p.id;document.getElementById('pvimg').src='/photo/'+p.id+'.jpg?'+Date.now();
  document.getElementById('pvt').textContent=`Pin ${p.id}${p.note?' \u201c'+p.note+'\u201d':''} - ${p.t} - at (${p.x.toFixed(2)}, ${p.y.toFixed(2)}) m`;pv.style.display='flex'}
document.getElementById('pvclose').onclick=()=>{pv.style.display='none'};
document.getElementById('pvdel').onclick=async()=>{const r=await post('/photo',{action:'delete',id:pvId});photos=r.photos;pv.style.display='none';draw()};
function drawPhotos(){photos.forEach(p=>{const q=w2p(p.x,p.y);
  ctx.fillStyle='rgba(255,212,59,.35)';ctx.beginPath();ctx.moveTo(q[0],q[1]);
  ctx.arc(q[0],q[1],34,-p.yaw-.45,-p.yaw+.45);ctx.closePath();ctx.fill();      // the way the camera was facing
  ctx.fillStyle='#fab005';ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(q[0],q[1],9,0,7);ctx.fill();ctx.stroke();
  ctx.fillStyle='#000';ctx.font='bold 10px sans-serif';ctx.textAlign='center';ctx.fillText('📷',q[0],q[1]+4)})}
let places=[], placeMode=false;
const placeBtn=document.getElementById('placebtn');
async function loadPlaces(){places=(await (await fetch('/places')).json()).places;draw()}
placeBtn.onclick=()=>{placeMode=!placeMode;placeBtn.textContent='📍 Pin place: '+(placeMode?'ON - click the map':'off');placeBtn.style.background=placeMode?'#2f9e44':'#3b4252';
  if(placeMode)flash('Click a white (free) spot on the map to name it')};
function drawPlaces(){places.forEach(p=>{const q=w2p(p.x,p.y),im=placeImg(p),R=im?15:p.icon?13:8;
  ctx.fillStyle='#e8590c';ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(q[0],q[1],R,0,7);ctx.fill();
  if(im){ctx.save();ctx.clip();const m=Math.min(im.naturalWidth,im.naturalHeight);ctx.drawImage(im,(im.naturalWidth-m)/2,(im.naturalHeight-m)/2,m,m,q[0]-R,q[1]-R,2*R,2*R);ctx.restore()}
  ctx.beginPath();ctx.arc(q[0],q[1],R,0,7);ctx.stroke();
  if(!im&&p.icon){ctx.font='16px sans-serif';ctx.textAlign='center';ctx.fillStyle='#fff';ctx.fillText(p.icon,q[0],q[1]+6)}
  ctx.font='bold 12px sans-serif';ctx.textAlign='center';ctx.lineWidth=3;ctx.strokeStyle='#000';ctx.strokeText(p.name,q[0],q[1]-R-6);ctx.fillStyle='#ffd8a8';ctx.fillText(p.name,q[0],q[1]-R-6)})}
async function placeClick(e,b){
  const cx=(e.clientX-b.left)/b.width*cv.width, cy=(e.clientY-b.top)/b.height*cv.height;
  const hit=places.find(p=>{const q=w2p(p.x,p.y);return Math.hypot(q[0]-cx,q[1]-cy)<20});
  if(hit){plId=hit.id;plname.value=hit.name;pldesc.value=hit.description||'';plinstr.value=hit.instructions||'';plicon.value=hit.icon||'';plImg=undefined;plfile.value='';showThumb(hit);plm.style.display='flex';plname.focus();return}
  if(!placeMode)return false;
  const w=p2w(cx,cy);const nm=prompt('Name this place (e.g. kitchen, charging dock):');if(!nm||!nm.trim())return;
  const ds=prompt('Description (optional):','')||'';const ins=prompt('Instructions for the AI at this pin (optional), e.g. take a photo then turn all round:','')||'';
  const r=await post('/place',{name:nm,description:ds,instructions:ins,x:w[0],y:w[1]});if(!r.ok)return flash(r.error);
  places=r.places;flash('Pinned "'+nm.trim()+'"');draw()}
let objs=[];
const ICONS={bed:'🛏️',chair:'🪑','office chair':'💺',desk:'🖥️',table:'🍽️',sofa:'🛋️',couch:'🛋️',tv:'📺',stairs:'🪜',door:'🚪',window:'🪟',plant:'🪴',
  bookshelf:'📚',shelf:'📚',wardrobe:'🚪',cabinet:'🗄️',fridge:'🧊',sink:'🚰',toilet:'🚽',lamp:'💡',bin:'🗑️',monitor:'🖥️',laptop:'💻',box:'📦',rug:'🟫',stool:'🪑',bench:'🪑',piano:'🎹'};
function drawObjects(P){objs.forEach(o=>{const q=P(o.x,o.y);
  ctx.font='22px sans-serif';ctx.textAlign='center';ctx.fillStyle='#000';ctx.fillText(ICONS[o.label]||'📍',q[0],q[1]+7);
  ctx.font='bold 11px sans-serif';ctx.lineWidth=3;ctx.strokeStyle='#000';ctx.strokeText(o.label,q[0],q[1]+24);ctx.fillStyle='#fff';ctx.fillText(o.label,q[0],q[1]+24)})}
document.getElementById('objscan').onclick=async()=>{const r=await post('/objects',{});objs=r.objects||objs;
  flash(r.ok?'Identifying objects in the camera view… they appear on the map in a few seconds':r.error);draw()};
let zoneMode=false, zones=[], zpts=[], ztool='box', zbox=null;
const zbar=document.getElementById('zbar'), zbtn=document.getElementById('zbtn');
async function loadZones(){zones=(await (await fetch('/zones')).json()).zones;renderZones()}
function renderZones(){
  document.getElementById('zlist').innerHTML=zones.map(z=>`<span class="chip">Zone ${z.id}<i data-id="${z.id}" title="Delete">×</i></span>`).join(' ');
  document.querySelectorAll('#zlist i').forEach(i=>i.onclick=async()=>{
    const r=await post('/zones',{action:'delete',id:+i.dataset.id});
    if(!r.ok)return flash(r.error);zones=r.zones;renderZones();path=[];draw()});
  draw();
}
zbtn.onclick=()=>{
  if(moving||mapping)return;
  zoneMode=!zoneMode;zpts=[];zbox=null;zbtn.classList.toggle('on',zoneMode);
  zbtn.textContent='No-go zones: '+(zoneMode?'editing':'off');
  zbar.style.display=zoneMode?'flex':'none';
  if(zoneMode&&drawMode)drawBtn.onclick();
  setTool(ztool);
  st.textContent=zoneMode?'Zone editing: '+(ztool==='box'?'drag a box on the map':'click corners, then Close zone'):'Click the map to choose a destination';
};
function setTool(t){
  ztool=t;zpts=[];zbox=null;
  document.getElementById('ztbox').classList.toggle('on',t==='box');
  document.getElementById('ztpoly').classList.toggle('on',t==='poly');
  for(const id of ['zclose','zundo'])document.getElementById(id).style.display=t==='poly'?'':'none';
  document.getElementById('zhint').textContent=t==='box'?'Drag on the map to draw a box the car must NOT enter.':'Click the corners of the area the car must NOT enter, then Close zone.';
  cv.style.touchAction=(zoneMode&&t==='box')?'none':'';draw();
}
document.getElementById('ztbox').onclick=()=>setTool('box');
document.getElementById('ztpoly').onclick=()=>setTool('poly');
const evW=e=>{const b=cv.getBoundingClientRect();return p2w((e.clientX-b.left)/b.width*cv.width,(e.clientY-b.top)/b.height*cv.height)};
cv.addEventListener('pointerdown',e=>{
  if(!zoneMode||ztool!=='box'||moving||mapping)return;
  e.preventDefault();cv.setPointerCapture(e.pointerId);const w=evW(e);zbox={a:w,b:w};draw();
});
cv.addEventListener('pointermove',e=>{if(zbox){zbox.b=evW(e);draw()}});
cv.addEventListener('pointerup',async e=>{
  if(!zbox)return;
  zbox.b=evW(e);const {a,b}=zbox;zbox=null;
  const x0=Math.min(a[0],b[0]),x1=Math.max(a[0],b[0]),y0=Math.min(a[1],b[1]),y1=Math.max(a[1],b[1]);
  if(x1-x0<0.15||y1-y0<0.15){draw();return flash('Box too small - drag a larger area')}
  const r=await post('/zones',{action:'add',pts:[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]});
  if(!r.ok){draw();return flash(r.error)}
  zones=r.zones;path=[];renderZones();flash('No-go box added - the planner now avoids it');
});
cv.addEventListener('pointercancel',()=>{zbox=null;draw()});
document.getElementById('zundo').onclick=()=>{zpts.pop();draw()};
document.getElementById('zclose').onclick=async()=>{
  if(zpts.length<3)return flash('A zone needs at least 3 corners');
  const r=await post('/zones',{action:'add',pts:zpts});
  if(!r.ok)return flash(r.error);
  zones=r.zones;zpts=[];path=[];renderZones();flash('No-go zone added - the planner now avoids it');
};
function drawZones(){
  const poly=(p,fill)=>{ctx.beginPath();p.forEach((q,i)=>{const w=w2p(...q);i?ctx.lineTo(...w):ctx.moveTo(...w)});ctx.closePath();
    if(fill){ctx.fillStyle='rgba(250,60,60,.38)';ctx.fill()}ctx.strokeStyle='#ff6b6b';ctx.lineWidth=2;ctx.stroke()};
  zones.forEach(z=>poly(z.pts,true));
  if(zbox){const p=w2p(zbox.a[0],zbox.a[1]),q=w2p(zbox.b[0],zbox.b[1]);
    ctx.fillStyle='rgba(250,60,60,.25)';ctx.fillRect(p[0],p[1],q[0]-p[0],q[1]-p[1]);
    ctx.setLineDash([6,4]);ctx.strokeStyle='#ffd43b';ctx.lineWidth=2;ctx.strokeRect(p[0],p[1],q[0]-p[0],q[1]-p[1]);ctx.setLineDash([])}
  if(zpts.length){ctx.strokeStyle='#ffd43b';ctx.lineWidth=2;ctx.beginPath();
    zpts.forEach((q,i)=>{const w=w2p(...q);i?ctx.lineTo(...w):ctx.moveTo(...w)});ctx.stroke();
    zpts.forEach(q=>{const w=w2p(...q);ctx.fillStyle='#ffd43b';ctx.beginPath();ctx.arc(w[0],w[1],5,0,7);ctx.fill()})}
}
const mapSel=document.getElementById('maps');
async function loadMapList(){
  const r=await (await fetch('/maps')).json();
  mapSel.innerHTML=r.maps.map(m=>`<option value="${m}"${m===r.current?' selected':''}>${m.replace(/^maps\//,'').replace(/\.pcd$/,'')}</option>`).join('');
}
const mm=document.getElementById('mm'),mmlist=document.getElementById('mmlist');
const mapName=f=>f.replace(/^maps\//,'').replace(/\.pcd$/,'');
async function renderManage(){
  const r=await (await fetch('/maps')).json();
  mmlist.innerHTML=r.maps.map(m=>`<div style="display:flex;gap:10px;align-items:center;justify-content:space-between"><span>${mapName(m)}${m===r.current?' <i style="color:#69db7c">(loaded)</i>':''}</span>`+
    `<button data-f="${m}" ${m===r.current?'disabled title="Switch to another map first"':''} style="background:#d64545">Delete</button></div>`).join('')||'No maps';
  mmlist.querySelectorAll('button[data-f]').forEach(bt=>bt.onclick=async()=>{
    const f=bt.dataset.f;if(!confirm('Delete map "'+mapName(f)+'" with its zones, places, objects and photo pins?'))return;
    const d=await post('/deletemap',{file:f});if(!d.ok)return flash(d.error);
    flash('Deleted "'+mapName(f)+'"');await renderManage();await loadMapList()})}
document.getElementById('delmap').onclick=async()=>{await renderManage();mm.style.display='flex'};
document.getElementById('mmclose').onclick=()=>{mm.style.display='none'};
mapSel.onchange=async()=>{
  if(moving){await loadMapList();return}
  st.textContent='Loading map…';
  const r=await post('/selectmap',{file:mapSel.value});
  if(!r.ok){st.textContent=r.error;await loadMapList();return}
  path=[];goal=[];goal=null;pts=[];bad=[];undo.disabled=clear.disabled=go.disabled=true;
  await loadMap();await loadZones();await loadPhotos();await loadPlaces();
  st.textContent='Map loaded — put the car at its origin (where that map was started), then click a destination';
};
async function init(){await loadMapList();await loadMap();await loadZones();await loadPhotos();await loadPlaces(); setInterval(poll,300); setInterval(pollMap,700);setInterval(pollSpeed,1500);pollSpeed();}
function draw(){
  if(mapping)return drawMapping();
  drawBase();drawZones();drawPhotos();drawPlaces();drawObjects(w2p);drawRest();
}
function drawBase(){
  ctx.imageSmoothingEnabled=false; ctx.drawImage(base,0,0,cv.width,cv.height);
}
function drawRest(){
  for(let i=1;i<path.length;i++){ctx.strokeStyle=bad.includes(i-1)?'#ff6b6b':'#00c8d7';ctx.lineWidth=3;
    ctx.beginPath();ctx.moveTo(...w2p(...path[i-1]));ctx.lineTo(...w2p(...path[i]));ctx.stroke();}
  if(drawMode)path.slice(1).forEach((p,i)=>{const q=w2p(...p);ctx.fillStyle='#ffd43b';ctx.beginPath();
    ctx.arc(q[0],q[1],6,0,7);ctx.fill();ctx.fillStyle='#000';ctx.font='bold 9px sans-serif';ctx.textAlign='center';
    ctx.fillText(i+1,q[0],q[1]+3)});
  if(goal){const q=w2p(...goal);ctx.fillStyle='#e03131';ctx.beginPath();ctx.arc(q[0],q[1],8,0,7);ctx.fill();}
  drawCar(...w2p(...pose),yaw);
}
function drawCar(x,y,a){      // top-down car, nose pointing the way it faces
  ctx.save();ctx.translate(x,y);ctx.rotate(-a);
  const rr=(px,py,w,h,r)=>{ctx.beginPath();ctx.roundRect(px,py,w,h,r)};
  ctx.fillStyle='#111';rr(-12,-12,9,5,2);ctx.fill();rr(-12,7,9,5,2);ctx.fill();rr(5,-12,9,5,2);ctx.fill();rr(5,7,9,5,2);ctx.fill();   // wheels
  ctx.fillStyle='#2b8a3e';ctx.strokeStyle='#fff';ctx.lineWidth=1.5;rr(-15,-8,30,16,5);ctx.fill();ctx.stroke();                     // body
  ctx.fillStyle='#a5d8ff';rr(1,-6,8,12,2);ctx.fill();                                                                             // windscreen (front)
  ctx.fillStyle='#ffd43b';ctx.beginPath();ctx.moveTo(15,-4);ctx.lineTo(21,0);ctx.lineTo(15,4);ctx.closePath();ctx.fill();         // nose arrow
  ctx.restore();
}
function drawMapping(){
  ctx.fillStyle='#0d0f12';ctx.fillRect(0,0,cv.width,cv.height);
  let xs=[mpose[0]],ys=[mpose[1]];
  if(mpts)for(let i=0;i<mpts.length;i+=2){xs.push(mpts[i]);ys.push(mpts[i+1])}
  const x0=Math.min(...xs)-.5,x1=Math.max(...xs)+.5,y0=Math.min(...ys)-.5,y1=Math.max(...ys)+.5;
  const sc=Math.min(cv.width/(x1-x0),cv.height/(y1-y0));
  const P=(x,y)=>[(x-x0)*sc,cv.height-(y-y0)*sc];
  ctx.fillStyle='#4dd9e8';
  if(mpts)for(let i=0;i<mpts.length;i+=2){const q=P(mpts[i],mpts[i+1]);ctx.fillRect(q[0]-1,q[1]-1,2,2)}
  mzones.forEach(z=>{ctx.beginPath();z.forEach((q,i)=>{const w=P(...q);i?ctx.lineTo(...w):ctx.moveTo(...w)});ctx.closePath();
    ctx.fillStyle='rgba(250,60,60,.38)';ctx.fill();ctx.strokeStyle='#ff6b6b';ctx.lineWidth=2;ctx.stroke()});
  if(mtarget){const q=P(...mtarget);ctx.strokeStyle='#ffd43b';ctx.lineWidth=2;ctx.beginPath();ctx.arc(q[0],q[1],9,0,7);ctx.stroke()}
  drawObjects(P);
  drawCar(...P(...mpose),myaw);
}
async function pollMap(){
  if(!mapping)return;
  const s=await (await fetch('/mappts')).json();
  mpts=new Float32Array(Uint8Array.from(atob(s.pts),c=>c.charCodeAt(0)).buffer);
  mpose=s.pose;myaw=s.yaw;mzones=s.zones||[];mtarget=s.target;objs=s.objects||[];
  if(!mapping||Date.now()<flashUntil)return;
  if(s.explore||/xplor/.test(s.msg||''))st.textContent=s.msg;else
  st.textContent='Mapping — '+(mpts.length/2)+' pts'+(s.skipped?', '+s.skipped+' scans skipped (drive slower)':'');
  drawMapping();
}
const chatEl=document.getElementById('chat'), chatLog=document.getElementById('chatlog'), chatIn=document.getElementById('chatin');
function addMsg(cls,text,btn){const d=document.createElement('div');d.className='m '+cls;d.textContent=text;
  if(btn){const b=document.createElement('button');b.textContent='▶ Start: '+btn;b.onclick=async()=>{
    if(!confirm('The AI will drive the car by itself and send map + camera images to Anthropic. Watch it, and keep it away from stairs. Start?'))return;
    aiTask.value=btn;const r=await post('/ai',{on:true,task:btn,radius:+xr.value});
    addMsg('s',r.ok?'Task started - watch the AI log.':'Could not start: '+r.error);if(r.ok)b.disabled=true};d.appendChild(b)}
  chatLog.appendChild(d);chatLog.scrollTop=chatLog.scrollHeight;return d}
document.getElementById('chatbtn').onclick=()=>{chatEl.style.display=chatEl.style.display==='flex'?'none':'flex';
  if(chatEl.style.display==='flex'){if(!chatLog.children.length)addMsg('s','Ask me what I see, where I am, or tell me what to do.');chatIn.focus()}};
document.getElementById('chatx').onclick=()=>{chatEl.style.display='none'};
document.getElementById('chatnew').onclick=async()=>{await post('/chat',{reset:true});chatLog.innerHTML='';addMsg('s','New conversation.')};
document.getElementById('chatform').onsubmit=async e=>{e.preventDefault();const t=chatIn.value.trim();if(!t)return;
  chatIn.value='';addMsg('u',t);let bub=addMsg('s','thinking…');chatIn.disabled=true;let got=false;
  const handle=o=>{
    if(o.t!==undefined){if(!got){got=true;bub.className='m a';bub.textContent=''}bub.textContent+=o.t;chatLog.scrollTop=chatLog.scrollHeight}
    else if(o.error){if(!got)bub.remove();addMsg('s','⚠ '+o.error);got=true}
    else if(o.done){if(!got)bub.remove();(o.actions||[]).forEach(a=>{if(a.type==='propose')addMsg('s','Suggested task',a.task);
      else if(a.type==='pinned')addMsg('s','📷 Photo pinned: '+a.note);else if(a.type==='stopped')addMsg('s','■ Robot stopped')})}};
  try{const resp=await fetch('/chat',{method:'POST',body:JSON.stringify({message:t})});
    const rd=resp.body.getReader(),dec=new TextDecoder();let buf='';
    for(;;){const {done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
      let i;while((i=buf.indexOf('\n'))>=0){const line=buf.slice(0,i);buf=buf.slice(i+1);if(line.trim())handle(JSON.parse(line))}}
  }catch(err){if(!got)bub.remove();addMsg('s','⚠ '+err)}
  chatIn.disabled=false;chatIn.focus()};
let aiShown=0;
function mirrorAi(ai){      // the AI's own decisions show up in the chat box as they happen
  if(ai.n<aiShown)aiShown=ai.n;
  if(ai.n>aiShown){const fresh=ai.log.slice(-Math.min(ai.n-aiShown,ai.log.length));aiShown=ai.n;
    fresh.forEach(l=>addMsg('a','🤖 '+l))}}
const exBtn=document.getElementById('explore'), xr=document.getElementById('xr');
let exploring=false, aiOn=false;
const aiBtn=document.getElementById('aibtn'), aiTask=document.getElementById('aitask'), aiLog=document.getElementById('ailog'), aiStat=document.getElementById('aistat');
aiBtn.onclick=async()=>{
  if(!aiOn&&!confirm('The AI will drive the car by itself and send map + camera images to Anthropic. Keep it away from stairs you cannot block off, and watch it. Start?'))return;
  const r=await post('/ai',{on:!aiOn,task:aiTask.value,radius:+xr.value});
  if(!r.ok)flash(r.error);
};
exBtn.onclick=async()=>{
  if(!exploring&&!confirm('The car will drive by itself. Keep it away from stairs or drops you cannot block off, and watch it. Start?'))return;
  const r=await post('/explore',{on:!exploring,radius:+xr.value});
  if(!r.ok)flash(r.error);
};
const modeBtn=document.getElementById('mode'), pad=document.getElementById('pad'), savebar=document.getElementById('savebar');
function setUI(){
  modeBtn.textContent='Mode: '+(mapping?'Mapping':'Navigate');
  pad.style.display=mapping?'grid':'none'; savebar.style.display=mapping?'flex':'none';
  mapSel.style.display=mapping?'none':'';zbtn.style.display=mapping?'none':'';if(mapping&&zoneMode)zbtn.onclick();
  for(const id of ['draw','go','undo','clear'])document.getElementById(id).style.display=mapping?'none':'';
}
modeBtn.onclick=async()=>{
  if(moving)return;
  if(!mapping&&!confirm('Start a NEW map? Place the car at the spot that will be the map origin.'))return;
  const r=await post('/mode',{mode:mapping?'nav':'map'});
  if(!r.ok){flash(r.error);return}
  mapping=!mapping;mpts=null;path=[];goal=null;pts=[];setUI();
  flash(mapping?'Mapping — drive with the pad or arrow keys':'Back to navigation — put the car at the map origin');draw();
};
let held=null,hb=null;
const drive=d=>post('/drive',{dir:d});
function hold(d){if(held===d)return;held=d;drive(d);clearInterval(hb);if(d!=='S')hb=setInterval(()=>drive(d),150)}
function release(){held=null;clearInterval(hb);drive('S')}
pad.querySelectorAll('button').forEach(b=>{
  b.onpointerdown=e=>{e.preventDefault();hold(b.dataset.d)};
  b.onpointerup=b.onpointerleave=release});
const KM={ArrowUp:'F',ArrowDown:'B',ArrowLeft:'L',ArrowRight:'R'};
addEventListener('keydown',e=>{if(!mapping||e.target.tagName==='INPUT')return;const d=KM[e.key]||(e.key===' '?'S':null);if(d){e.preventDefault();hold(d)}});
addEventListener('keyup',e=>{if(KM[e.key])release()});
document.getElementById('savebtn').onclick=async()=>{
  const r=await post('/savemap',{name:document.getElementById('mname').value.trim()});
  if(!r.ok){flash(r.error);return}
  mapping=false;setUI();await loadMapList();await loadMap();
  flash('Saved '+r.file+' and loaded it — put the car at the map origin to navigate');
};
async function poll(){
  const s=await (await fetch('/state')).json();
  learnOn=s.learn;showLearn();if(!mapping)objs=s.objects||[];
  document.getElementById('mstat').textContent='+'+s.added+' / −'+s.removed+' pts';
  if(s.map_version!==lastVer&&!mapping){const first=lastVer<0;lastVer=s.map_version;if(!first)loadMap()}
  aiOn=!!(s.ai&&s.ai.on);aiBtn.textContent=aiOn?'■ Stop AI':'🧠 AI';aiBtn.style.background=aiOn?'#d64545':'#7048e8';
  if(s.ai)mirrorAi(s.ai);
  if(s.ai){aiLog.style.display=s.ai.log.length?'block':'none';aiLog.textContent=s.ai.log.join('\n');aiLog.scrollTop=aiLog.scrollHeight;
    aiStat.textContent=s.ai.tokens[0]?`tokens in ${s.ai.tokens[0]} / out ${s.ai.tokens[1]} / cached ${s.ai.tokens[2]}`:''}
  exploring=s.explore;exBtn.textContent=exploring?'■ Stop exploring':'🧭 Auto-explore';exBtn.style.background=exploring?'#d64545':'#1c7ed6';
  pose=s.pose; yaw=s.yaw; const was=moving; moving=s.moving;
  if(moving){path=s.path}
  else if(was){if(s.arrived){path=[];goal=null;pts=[];bad=[];undo.disabled=clear.disabled=true}else{path=s.path;flash(s.msg)}}
  if((moving||was||s.lost)&&Date.now()>flashUntil&&!mapping)st.textContent=(s.lost?'⚠ localisation lost — ':'')+s.msg;
  go.disabled=moving||!path.length; draw();
}
cv.addEventListener('click',async e=>{
  if(moving||mapping)return;
  const b=cv.getBoundingClientRect();
  if(!zoneMode&&!drawMode&&M){const cx=(e.clientX-b.left)/b.width*cv.width,cy=(e.clientY-b.top)/b.height*cv.height;
    if(placeMode||places.some(p=>{const q=w2p(p.x,p.y);return Math.hypot(q[0]-cx,q[1]-cy)<20})){if(await placeClick(e,b)!==false)return}
    const hit=photos.find(p=>{const q=w2p(p.x,p.y);return Math.hypot(q[0]-cx,q[1]-cy)<14});if(hit)return showPhoto(hit);
    const ob=objs.find(o=>{const q=w2p(o.x,o.y);return Math.hypot(q[0]-cx,q[1]-cy)<16});
    if(ob){if(confirm('Remove "'+ob.label+'" from the map?')){const r=await post('/objects',{action:'delete',id:ob.id});objs=r.objects;draw()}return}}
  if(zoneMode){if(ztool!=='poly')return;zpts.push(p2w((e.clientX-b.left)/b.width*cv.width,(e.clientY-b.top)/b.height*cv.height));return draw()}
  if(drawMode){
    pts.push(p2w((e.clientX-b.left)/b.width*cv.width,(e.clientY-b.top)/b.height*cv.height));
    return sendPts();
  }
  goal=p2w((e.clientX-b.left)/b.width*cv.width,(e.clientY-b.top)/b.height*cv.height);
  const r=await post('/plan',{goal});
  if(r.ok){path=r.path;st.textContent='Path '+r.length.toFixed(2)+' m — press Go'}
  else{path=[];st.textContent=r.error}
  go.disabled=!r.ok; draw();
});
async function sendPts(){
  const r=pts.length?await post('/setpath',{points:pts}):(await post('/stop'),{ok:true,path:[],bad:[]});
  path=r.path; bad=r.bad||[];
  st.textContent=pts.length?(pts.length+' point(s), '+r.length.toFixed(2)+' m'+(r.zone&&r.zone.length?' — ⛔ crosses a no-go zone, cannot start':(bad.length?' — ⚠ red segment(s) cross an obstacle':''))+' — press Go'):'Click points to draw a path';
  go.disabled=!pts.length||!!(r.zone&&r.zone.length); undo.disabled=clear.disabled=!pts.length; draw();
}
const undo=document.getElementById('undo'), clear=document.getElementById('clear'), drawBtn=document.getElementById('draw');
drawBtn.onclick=()=>{if(zoneMode&&!drawMode)zbtn.onclick();drawMode=!drawMode;drawBtn.classList.toggle('on',drawMode);
  drawBtn.textContent='Draw path: '+(drawMode?'on':'off');pts=[];bad=[];path=[];goal=null;
  go.disabled=undo.disabled=clear.disabled=true;
  st.textContent=drawMode?'Click points to draw a path':'Click the map to choose a destination';draw();};
undo.onclick=()=>{pts.pop();sendPts()};
clear.onclick=()=>{pts=[];sendPts()};
go.onclick=async()=>{const r=await post('/go'); if(!r.ok)st.textContent=r.error};
document.getElementById('stop').onclick=()=>post('/stop');
init();
</script></body></html>"""


def main():
    global gm, loc, car, gm_radius, camera
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="home_map_clean.pcd")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--robot-radius", type=float, default=0.15)
    ap.add_argument("--sim", action="store_true", help="simulate the robot, no hardware")
    a = ap.parse_args()
    gm_radius = a.robot_radius
    load_zones(a.map)
    gm = build_gm(a.map)
    current_map[0] = a.map
    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=grid_refresher, daemon=True).start()
    if not a.sim:
        car = Car()
        try:
            camera = Camera()
        except Exception as e:
            log(f"[camera] unavailable: {e!r}")
        loc = Localizer(a.map)
        loc.start()
        print("Waiting for first LiDAR scan…")
        loc.wait_scans(2, timeout=15)
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    if tok and loc:
        import telegram_bot
        allowed = [int(x) for x in os.environ.get("TELEGRAM_ALLOWED_IDS", "").replace(" ", "").split(",") if x]
        telegram_bot.TelegramBot(tok, allowed, {
            "context": chat_context, "turn": chat_turn,
            "new_chat": lambda: chat and chat.reset(),
            "start_task": ai_start, "stop": robot_stop,
            "ai_state": lambda: (explorer.ai_n, list(explorer.ai_log))}, log).start()
    print(f"Open http://localhost:{a.port}  (or http://<pi-ip>:{a.port})")
    try:
        ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
    finally:
        if car:
            car.send("S")
        if camera:
            camera.proc.terminate()


if __name__ == "__main__":
    main()
