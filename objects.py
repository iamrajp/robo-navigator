"""Semantic objects on the map: look at a camera frame, name the furniture in it (bed, desk, chair, stairs...),
and put each one where the LiDAR sees a surface in that direction.

detect(jpg)  -> [{"label", "x_frac", "conf"}]   one vision call (x_frac 0 = left edge of the image, 1 = right edge)
place(...)   -> world (x, y) of the nearest LiDAR hit along the object's bearing, or None
merge(...)   -> add to / refresh the list, de-duplicating the same label seen again nearby"""
import base64
import json
import math
import os

import numpy as np

import llm_brain

CAM_HFOV = math.radians(float(os.environ.get("CAM_HFOV_DEG", "62")))   # camera horizontal field of view
MAX_RANGE = 4.0          # objects farther than this are ignored (LiDAR/camera bearing gets unreliable)
MIN_RANGE = 0.20
BEAM = math.radians(4.0)  # half-width of the LiDAR wedge searched around the bearing
MERGE_DIST = 0.8         # same label closer than this = same object
MAX_OBJECTS = 80
MIN_CONF = 0.5

ICONS = {"bed": "🛏️", "chair": "🪑", "office chair": "💺", "desk": "🖥️", "table": "🍽️", "sofa": "🛋️", "couch": "🛋️",
         "tv": "📺", "stairs": "🪜", "door": "🚪", "window": "🪟", "plant": "🪴", "bookshelf": "📚", "shelf": "📚",
         "wardrobe": "🚪", "cabinet": "🗄️", "fridge": "🧊", "sink": "🚰", "toilet": "🚽", "lamp": "💡", "bin": "🗑️",
         "monitor": "🖥️", "laptop": "💻", "box": "📦", "rug": "🟫", "stool": "🪑", "bench": "🪑", "piano": "🎹"}

PROMPT = ("This is a frame from a small robot's low, forward-facing camera (about 10 cm above the floor, so furniture is seen "
          "from below). List the large, fixed or semi-fixed objects clearly visible that matter on a floor plan: bed, desk, table, "
          "chair, office chair, sofa, tv, bookshelf, wardrobe, cabinet, fridge, door, window, plant, bin, and especially STAIRS or a step/drop. "
          "Ignore small items, the floor and walls. Use short lowercase singular names. For each give x_frac: the horizontal position of the "
          "object's centre in the image (0 = left edge, 1 = right edge), and conf 0-1. Report only what you can actually see; an empty list is fine.")

TOOL = {"name": "report_objects", "description": "Objects visible in the camera frame.",
        "input_schema": {"type": "object", "properties": {"objects": {"type": "array", "items": {
            "type": "object", "properties": {"label": {"type": "string"}, "x_frac": {"type": "number"}, "conf": {"type": "number"}},
            "required": ["label", "x_frac", "conf"], "additionalProperties": False}}},
            "required": ["objects"], "additionalProperties": False}}

_client = None


def detect(jpg):
    """One vision call. Raises on API trouble (callers log and carry on)."""
    global _client
    import anthropic
    if _client is None:
        _client = anthropic.Anthropic(timeout=llm_brain.API_TIMEOUT, max_retries=1)
    resp = _client.messages.create(
        model=llm_brain.MODEL, max_tokens=1000, tools=[TOOL], tool_choice={"type": "tool", "name": "report_objects"},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(jpg).decode()}},
            {"type": "text", "text": PROMPT}]}])
    call = next((b for b in resp.content if b.type == "tool_use"), None)
    if call is None:
        return []
    out = []
    for o in (call.input or {}).get("objects", []):
        try:
            label = str(o["label"]).strip().lower()[:30]
            xf, conf = float(o["x_frac"]), float(o["conf"])
        except (KeyError, TypeError, ValueError):
            continue
        if label and 0.0 <= xf <= 1.0 and conf >= MIN_CONF:
            out.append({"label": label, "x_frac": xf, "conf": conf})
    return out


FIND_TOOL = {"name": "report_target", "description": "Where the named object is in the camera frame, if visible.",
             "input_schema": {"type": "object", "properties": {
                 "visible": {"type": "boolean"},
                 "x_frac": {"type": "number", "description": "horizontal centre of the object, 0 = left edge, 1 = right edge"},
                 "y_frac": {"type": "number", "description": "vertical position of the object's LOWEST point (where it touches the floor), 0 = top edge, 1 = bottom edge"},
                 "conf": {"type": "number"}},
                 "required": ["visible"], "additionalProperties": False}}


def find(jpg, label):
    """One vision call: locate `label` (e.g. 'ball') in the frame -> {"x_frac", "y_frac", "conf"} or None if not visible."""
    global _client
    import anthropic
    if _client is None:
        _client = anthropic.Anthropic(timeout=llm_brain.API_TIMEOUT, max_retries=1)
    resp = _client.messages.create(
        model=llm_brain.MODEL, max_tokens=300, tools=[FIND_TOOL], tool_choice={"type": "tool", "name": "report_target"},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(jpg).decode()}},
            {"type": "text", "text": f"This is a frame from a small robot's low, forward-facing camera (about 10 cm above the floor). "
                                     f"Is there a {label} on the floor in view? If yes give where it is: x_frac of its centre and y_frac of the "
                                     f"point where it touches the floor (0 = top of image, 1 = bottom). Report only what you can actually see."}]}])
    call = next((b for b in resp.content if b.type == "tool_use"), None)
    d = (call.input if call else None) or {}
    try:
        if not d.get("visible"):
            return None
        xf, yf, conf = float(d["x_frac"]), float(d["y_frac"]), float(d.get("conf", 1.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not (0.0 <= xf <= 1.0 and 0.0 <= yf <= 1.0) or conf < MIN_CONF:
        return None
    return {"x_frac": xf, "y_frac": yf, "conf": conf}


def place(x_frac, pts, pose, fwd_off, left_sign):
    """pts: (N, 2) LiDAR points in the car's LiDAR frame; pose = (x, y, yaw) of that frame in the map.
    left_sign: +1 if turning left raises the yaw. Returns (x, y, range_m) of the nearest hit in the object's direction, or None."""
    if not len(pts):
        return None
    rel = left_sign * (0.5 - x_frac) * CAM_HFOV                  # + = to the car's left
    a = fwd_off + rel
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    diff = np.abs((ang - a + math.pi) % (2 * math.pi) - math.pi)
    r = np.hypot(pts[:, 0], pts[:, 1])
    m = (diff < BEAM) & (r > MIN_RANGE) & (r < MAX_RANGE)
    if not m.any():
        return None
    k = np.argmin(np.where(m, r, np.inf))
    px, py = pts[k]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return float(pose[0] + c * px - s * py), float(pose[1] + s * px + c * py), float(r[k])


def merge(objs, label, x, y, t):
    """Refresh an existing object of this label nearby (running average), else append. Returns the object."""
    best = None
    for o in objs:
        d = math.dist((o["x"], o["y"]), (x, y))
        if o["label"] == label and d < MERGE_DIST and (best is None or d < best[0]):
            best = (d, o)
    if best:
        o = best[1]
        n = o.get("seen", 1)
        o["x"], o["y"] = (o["x"] * n + x) / (n + 1), (o["y"] * n + y) / (n + 1)
        o["seen"], o["t"] = n + 1, t
        return o
    if len(objs) >= MAX_OBJECTS:
        return None
    o = {"id": max([q["id"] for q in objs] + [0]) + 1, "label": label, "x": x, "y": y, "seen": 1, "t": t}
    objs.append(o)
    return o
