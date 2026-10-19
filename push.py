"""push_object skill: find a ball with the camera, stand behind it on the line to a target, and push it there.

The module is hardware-free: everything it needs from the robot comes through a `robot` object (see Robot below),
so it runs unchanged against the real car (nav_server.CarRobot) and against the simulator in test_push_sim.py.

Geometry: the camera is low (CAM_HEIGHT above the floor) and level (CAM_TILT_DEG), so a ball's floor-contact point at image row
y_frac lies at floor range  CAM_HEIGHT / tan(tilt + (y_frac - 0.5) * VFOV).  The bearing comes from x_frac and the horizontal FOV.
Once the ball touches the car it leaves the camera's view, so the push is done in short chunks: push, back off, look again.

Robot interface (all positions in map metres, headings in radians, 0 = +x):
    pose() -> (x, y, heading)         see(label) -> {"x_frac", "y_frac"} or None
    turn_to(heading)                  forward(secs) pulse / back(metres)
    drive_to(x, y) -> None | error    free(x, y) -> bool        line_ok(a, b) -> bool
    in_zone(x, y) -> bool             hazard() -> bool          obstacle_ahead(limit) -> metres | None
    stopped() -> bool                 left_sign -> +1/-1        log(msg)
"""
import math
import os
from dataclasses import dataclass

import objects


def _env(name, default):
    return float(os.environ.get(name, default))


@dataclass
class Config:
    cam_height: float = _env("CAM_HEIGHT_M", 0.10)
    cam_vfov: float = math.radians(_env("CAM_VFOV_DEG", 48))
    cam_tilt: float = math.radians(_env("CAM_TILT_DEG", 0))      # + = camera pitched down
    cam_hfov: float = objects.CAM_HFOV
    max_range: float = 3.0               # beyond this the row->range estimate is too coarse
    stage_dist: float = 0.40             # stand this far behind the ball (ball centre to car centre)
    tol: float = 0.25                    # ball within this of the target = success
    lateral_tol: float = 0.12            # ball further than this off the push line = re-stage
    chunk: float = 0.35                  # push this far, then back off and look
    backoff: float = 0.35
    pulse_s: float = 0.35
    contact: float = 0.22                # car centre to ball centre when touching
    max_pulses_per_chunk: int = 16
    max_chunks: int = 14
    max_restage: int = 10
    search_step: float = math.radians(45)
    search_steps: int = 8
    wall_stop: float = 0.30              # stop pushing if the LiDAR sees something this close ahead


def ball_world(look, pose, left_sign, cfg):
    """Camera detection -> (x, y, range) in the map, or None if the row gives no usable range."""
    x, y, h = pose
    ang = cfg.cam_tilt + (look["y_frac"] - 0.5) * cfg.cam_vfov
    if ang < math.radians(1.0):
        return None                                           # at/above the horizon: far or not on the floor
    rng = cfg.cam_height / math.tan(ang)
    if rng > cfg.max_range:
        return None
    b = h + left_sign * (0.5 - look["x_frac"]) * cfg.cam_hfov
    return x + rng * math.cos(b), y + rng * math.sin(b), rng


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def locate(robot, label, cfg):
    look = robot.see(label)
    return ball_world(look, robot.pose(), robot.left_sign, cfg) if look else None


def search(robot, label, cfg):
    """Look here, then turn in steps until the object shows up. Returns (x, y, range) or None."""
    _, _, h0 = robot.pose()
    for i in range(cfg.search_steps):
        if robot.stopped():
            return None
        b = locate(robot, label, cfg)
        if b:
            return b
        robot.turn_to(wrap(h0 + (i + 1) * cfg.search_step))
    return locate(robot, label, cfg)


def push_object(robot, label, target, cfg=None):
    """-> result text for the LLM. Starts with 'ok' on success, 'partial' if the ball moved but missed, else 'error'."""
    cfg = cfg or Config()
    tx, ty = float(target[0]), float(target[1])
    if not robot.free(tx, ty):
        return "error: the target is not known free floor (unknown, wall margin or no-go)"
    b = search(robot, label, cfg)
    if b is None:
        return f"error: could not find the {label} (not visible, or farther than {cfg.max_range:.0f} m). Move closer or turn, then try again"
    robot.log(f"[push] {label} seen at ({b[0]:.2f},{b[1]:.2f}), {b[2]:.2f} m away; target ({tx:.2f},{ty:.2f})")
    first = b
    restage = 0
    chunks = 0
    while chunks < cfg.max_chunks and not robot.stopped():
        bx, by = b[0], b[1]
        d = math.hypot(tx - bx, ty - by)
        if d <= cfg.tol:
            return f"ok: the {label} is {d:.2f} m from the target"
        ux, uy = (tx - bx) / d, (ty - by) / d
        if not robot.line_ok((bx, by), (tx, ty)):
            return f"error: pushing the {label} to the target would cross a no-go zone"
        rx, ry, rh = robot.pose()
        lateral = abs(-uy * (rx - bx) + ux * (ry - by))          # car's offset from the ball->target line
        behind = (rx - bx) * ux + (ry - by) * uy                  # < 0 = car is on the near side of the ball
        if behind > -0.6 * cfg.stage_dist or behind < -2.0 * cfg.stage_dist or lateral > cfg.lateral_tol:
            if restage > cfg.max_restage:
                return f"partial: gave up re-staging; the {label} is {d:.2f} m from the target"
            # floor-plane range error grows with distance (~r^2): from far away stop short, look again, then line up
            sd = cfg.stage_dist + (0.35 if b[2] > 0.9 else 0.0)
            sx, sy = bx - ux * sd, by - uy * sd
            if not robot.free(sx, sy):
                return f"error: cannot stand behind the {label} (wall, unknown or no-go there)"
            err = robot.drive_to(sx, sy)
            if err:
                return f"error: could not reach the spot behind the {label}: {err}"
            restage += 1
            robot.turn_to(math.atan2(uy, ux))
            nb = locate(robot, label, cfg) or search(robot, label, cfg)
            if nb is None:
                return f"error: lost sight of the {label} while lining up"
            b = nb
            continue                                               # recompute the line from the fresh position
        # ── push one chunk ──
        robot.turn_to(math.atan2(uy, ux))
        x0, y0, _ = robot.pose()
        gap = max(0.0, math.hypot(bx - x0, by - y0) - cfg.contact)      # free run before the ball is touched
        want = gap + min(cfg.chunk, d)
        for _ in range(cfg.max_pulses_per_chunk):
            if robot.stopped():
                return "stopped"
            if robot.hazard():
                robot.back(0.4)
                return f"error: the ledge check stopped the push; the {label} is probably near {d:.1f} m from the target"
            x, y, h = robot.pose()
            if robot.in_zone(x, y):
                robot.back(0.4)
                return "error: the car reached a no-go zone - stopped"
            if (x - x0) * ux + (y - y0) * uy >= want:
                break
            ob = robot.obstacle_ahead(cfg.wall_stop + 0.2)
            if ob is not None and ob < cfg.wall_stop:
                break                                              # something hard right ahead
            if abs(wrap(math.atan2(uy, ux) - h)) > math.radians(10):
                robot.turn_to(math.atan2(uy, ux))
            robot.forward(cfg.pulse_s)
        chunks += 1
        robot.back(cfg.backoff)
        nb = locate(robot, label, cfg) or search(robot, label, cfg)
        if nb is None:
            return f"error: lost sight of the {label} after pushing ({chunks} chunk(s)); it may have rolled away"
        moved = math.hypot(nb[0] - first[0], nb[1] - first[1])
        b = nb
        robot.log(f"[push] chunk {chunks}: {label} now ({b[0]:.2f},{b[1]:.2f}), moved {moved:.2f} m in total")
    d = math.hypot(tx - b[0], ty - b[1])
    moved = math.hypot(b[0] - first[0], b[1] - first[1])
    if robot.stopped():
        return "stopped"
    if d <= cfg.tol:
        return f"ok: the {label} is {d:.2f} m from the target"
    return f"partial: the {label} moved {moved:.2f} m but is still {d:.2f} m from the target after {chunks} chunk(s)"
