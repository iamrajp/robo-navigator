"""Simulated world for push.py: noisy kinematic robot, a ball that is pushed on contact, a camera model that is the inverse of
push.ball_world (plus pixel noise), a room with walls and a no-go box.  This tests the LOGIC of the skill, not the real camera
calibration, so a pass here says nothing about the physical success rate.   Run:  uv run python test_push_sim.py"""
import math
import random

import push

ROOM = (0.0, 0.0, 5.0, 4.0)            # x0, y0, x1, y1
WALL_MARGIN = 0.25
CONTACT = 0.22                          # ball centre ahead of the car's centre when touching


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SimRobot:
    def __init__(self, pose, ball, zones=(), seed=0, pix_noise=0.01, drive_noise=0.08, left_sign=1):
        self.x, self.y, self.h = pose
        self.ball = list(ball) if ball else None
        self.zones = list(zones)                       # axis-aligned boxes (x0, y0, x1, y1)
        self.rng = random.Random(seed)
        self.pix, self.dn = pix_noise, drive_noise
        self.left_sign = left_sign
        self.cfg = push.Config()
        self.t = 0.0
        self.calls = 0
        self.lines = []

    # ── interface used by push.py ──
    def pose(self):
        return self.x, self.y, self.h

    def log(self, msg):
        self.lines.append(msg)

    def stopped(self):
        return False

    def see(self, label):
        self.calls += 1
        if self.ball is None:
            return None
        dx, dy = self.ball[0] - self.x, self.ball[1] - self.y
        r = math.hypot(dx, dy)
        rel = wrap(math.atan2(dy, dx) - self.h)
        if abs(rel) > self.cfg.cam_hfov / 2 or r < 0.05:
            return None
        ang = math.atan(self.cfg.cam_height / r)
        yf = 0.5 + (ang - self.cfg.cam_tilt) / self.cfg.cam_vfov
        xf = 0.5 - self.left_sign * rel / self.cfg.cam_hfov
        if yf > 1.0 or not 0 <= xf <= 1:
            return None                                # too close: below the bottom of the image
        n = self.rng.gauss
        return {"x_frac": min(1, max(0, xf + n(0, self.pix))), "y_frac": min(1, max(0, yf + n(0, self.pix))), "conf": 0.9}

    def turn_to(self, heading):
        self.h = wrap(heading + self.rng.gauss(0, math.radians(3)))

    def _move(self, dist):
        d = dist * (1 + self.rng.gauss(0, self.dn))
        self.h = wrap(self.h + self.rng.gauss(0, math.radians(1.5)) * abs(dist) / 0.1)
        nx, ny = self.x + d * math.cos(self.h), self.y + d * math.sin(self.h)
        if self.ball is not None:
            bx, by = self.ball
            if math.hypot(bx - nx, by - ny) < CONTACT and (bx - self.x) * math.cos(self.h) + (by - self.y) * math.sin(self.h) > 0:
                a = math.atan2(by - ny, bx - nx) + self.rng.gauss(0, math.radians(2.5))
                bx, by = nx + CONTACT * math.cos(a), ny + CONTACT * math.sin(a)
                if self._inside(bx, by, WALL_MARGIN * 0.5):
                    self.ball = [bx, by]
                else:
                    return                              # ball against the wall: car can't advance
        if self._inside(nx, ny, WALL_MARGIN):
            self.x, self.y = nx, ny

    def forward(self, secs):
        self._move(0.25 * secs)

    def back(self, metres):
        for _ in range(max(1, round(metres / 0.0875))):
            self._move(-0.0875)

    def drive_to(self, x, y):
        if not self.free(x, y):
            return "no path"
        self.x, self.y = x + self.rng.gauss(0, 0.03), y + self.rng.gauss(0, 0.03)
        self.h = wrap(math.atan2(y - self.y, x - self.x))
        return None

    def _inside(self, x, y, m):
        return ROOM[0] + m < x < ROOM[2] - m and ROOM[1] + m < y < ROOM[3] - m

    def free(self, x, y):
        return self._inside(x, y, WALL_MARGIN) and not self.in_zone(x, y)

    def in_zone(self, x, y):
        return any(z[0] <= x <= z[2] and z[1] <= y <= z[3] for z in self.zones)

    def line_ok(self, a, b):
        for i in range(21):
            if self.in_zone(a[0] + (b[0] - a[0]) * i / 20, a[1] + (b[1] - a[1]) * i / 20):
                return False
        return True

    def hazard(self):
        return False

    def obstacle_ahead(self, limit):
        d = [(wall, 0) for wall in ()]
        # distance to the nearest wall straight ahead
        best = None
        for t in [i * 0.05 for i in range(1, int(limit / 0.05) + 1)]:
            if not self._inside(self.x + t * math.cos(self.h), self.y + t * math.sin(self.h), 0.0):
                best = t
                break
        return best


def run(seed, **kw):
    rng = random.Random(seed)
    ball = (rng.uniform(1.5, 3.0), rng.uniform(1.2, 2.8))
    target = (rng.uniform(3.2, 4.2), rng.uniform(1.0, 3.0))
    robot = SimRobot((0.8, 2.0, rng.uniform(-0.6, 0.6)), ball, seed=seed, **kw)
    res = push.push_object(robot, "ball", target)
    d = math.hypot(robot.ball[0] - target[0], robot.ball[1] - target[1])
    return res, d, robot


def main():
    # 1. the basic cases
    r, d, _ = run(1)
    assert r.startswith("ok") and d < 0.35, (r, d)

    robot = SimRobot((1.0, 2.0, 0.0), None)
    assert push.push_object(robot, "ball", (3.5, 2.0)).startswith("error: could not find"), "no ball -> error"

    robot = SimRobot((1.0, 2.0, 0.0), (2.0, 2.0), zones=[(3.4, 1.5, 3.9, 2.5)])
    assert push.push_object(robot, "ball", (3.6, 2.0)).startswith("error: the target is not known free"), "target in zone"

    robot = SimRobot((1.0, 2.0, 0.0), (2.0, 2.0), zones=[(2.8, 1.7, 3.1, 2.3)])
    assert "no-go" in push.push_object(robot, "ball", (3.8, 2.0)), "line through zone"

    robot = SimRobot((1.0, 2.0, math.pi), (2.0, 2.4))                    # facing away: must search
    r = push.push_object(robot, "ball", (4.0, 2.4))
    assert r.startswith("ok"), r

    robot = SimRobot((1.0, 2.0, 0.0), (2.0, 2.0), left_sign=-1)          # mirrored yaw convention
    assert push.push_object(robot, "ball", (3.6, 2.0)).startswith("ok")

    # 2. repeatability over random worlds
    n, ok, errs = 60, 0, []
    for s in range(100, 100 + n):
        res, d, _ = run(s)
        ok += res.startswith("ok") and d < 0.35
        errs.append(d)
    errs.sort()
    print(f"sim: {ok}/{n} pushes ended within 0.35 m of the target; median final error {errs[n // 2]:.2f} m")
    assert ok >= 0.8 * n, "success rate in simulation is too low"
    print("all push simulation checks passed")


if __name__ == "__main__":
    main()
