"""LLM 'brain': looks at the map image + camera frame and picks the robot's next high-level action.

The LLM never drives. It returns ONE tool call per decision (explore_to / go_to / turn_to /
pin_photo / add_no_go / finish); nav_server validates it and runs it through the normal planner,
no-go zones, obstacle detours, stuck handling and ledge check."""
import base64
import io
import json
import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw

from navigate import RES

MODEL = os.environ.get("LLM_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("LLM_EFFORT", "low")
MAX_DECISIONS = int(os.environ.get("LLM_MAX_DECISIONS", "60"))
API_TIMEOUT = 60.0
HISTORY_LINES = 10

SYSTEM = """You are the navigation brain of a small wheeled indoor robot (about 30 cm wide) with a 2D LiDAR and a low, forward-facing camera mounted near the floor.
Each turn you get: the user's task, a top-down map image, the latest camera frame, and a status text. Choose exactly ONE tool call. Another turn follows after the robot has carried it out.

Map image: white = floor the LiDAR has seen free, dark gray = never seen (unknown), black = walls/obstacles, light gray = keep-out margin around obstacles, red = no-go boxes, yellow numbered discs = photo pins, blue dots = where the robot has been, the blue arrow = the robot (pointing the way it faces), green discs labelled F1, F2... = frontiers (reachable edges of unexplored space). Faint grid lines and the axis labels are in metres. World coordinates: +x to the right, +y UP in the image.

The robot's low camera and 2D LiDAR cannot see drops. STAIRS, STEPS DOWN OR LEDGES ARE THE BIGGEST DANGER: if the camera shows stairs, a step, a balcony edge, or the floor seems to end, call add_no_go with a box covering the area in front of the robot before doing anything else, and do not drive toward it. Also avoid spots where the floor looks unlike the floor behind you.

Rules:
- Obey the user's task. Prefer finishing nearby areas before long transits. Do not go back to places already covered.
- explore_to(frontier_id) is the main way to explore; go_to(x, y) only to already-seen free floor.
- Each move is short (at most about 1.5 m); you will be asked again.
- Use pin_photo (once per room or notable spot) with a short note naming it, e.g. "kitchen door".
- Use turn_to to look around when the camera view is unhelpful.
- Call finish when the task is done or nothing useful is left (e.g. no frontiers).
- Give a "reason" of at most 25 words with every call. Be decisive; do not ask questions."""

TOOLS = [
    {"name": "explore_to", "description": "Drive one short leg toward the numbered frontier (e.g. 3 for F3).",
     "input_schema": {"type": "object", "properties": {"frontier_id": {"type": "integer"}, "reason": {"type": "string"}},
                      "required": ["frontier_id", "reason"], "additionalProperties": False}},
    {"name": "go_to", "description": "Drive one short leg toward a world point (metres) on already-seen free floor.",
     "input_schema": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "reason": {"type": "string"}},
                      "required": ["x", "y", "reason"], "additionalProperties": False}},
    {"name": "turn_to", "description": "Rotate in place to face a world heading (degrees; 0 = +x, 90 = +y) to look around.",
     "input_schema": {"type": "object", "properties": {"heading_deg": {"type": "number"}, "reason": {"type": "string"}},
                      "required": ["heading_deg", "reason"], "additionalProperties": False}},
    {"name": "pin_photo", "description": "Save the current camera frame as a labelled photo pin at the robot's position.",
     "input_schema": {"type": "object", "properties": {"note": {"type": "string"}, "reason": {"type": "string"}},
                      "required": ["note", "reason"], "additionalProperties": False}},
    {"name": "add_no_go", "description": "Mark an axis-aligned no-go box (centre x,y and size w,h in metres, max 3 m each) the robot must never enter, e.g. stairs. Cannot be removed by you.",
     "input_schema": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "w": {"type": "number"},
                                                       "h": {"type": "number"}, "reason": {"type": "string"}},
                      "required": ["x", "y", "w", "h", "reason"], "additionalProperties": False}},
    {"name": "finish", "description": "End the run: the task is complete or there is nothing useful left.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"], "additionalProperties": False}},
]


class BrainError(RuntimeError):
    pass


class DecisionLimit(BrainError):
    pass


# ── map image ────────────────────────────────────────────────────────────────

def render_map(cells, gmin, car, heading, frontiers=(), zones=(), pins=(), trail=(), max_px=900):
    """cells: uint8 grid (0 free, 1 unknown, 2 inflated obstacle, 3 wall), row = +y. Returns PNG bytes.
    frontiers: [(id, (x, y))]; zones: [[(x, y), ...]]; pins: [(id, (x, y))]; trail: [(x, y)]."""
    h, w = cells.shape
    seen = np.argwhere(cells != 1)
    ys, xs = (seen[:, 0], seen[:, 1]) if len(seen) else (np.array([h // 2]), np.array([w // 2]))
    cx, cy = int((car[0] - gmin[0]) / RES), int((car[1] - gmin[1]) / RES)
    x0, x1 = max(0, min(xs.min(), cx) - 20), min(w, max(xs.max(), cx) + 21)
    y0, y1 = max(0, min(ys.min(), cy) - 20), min(h, max(ys.max(), cy) + 21)
    sub = cells[y0:y1, x0:x1]
    sh, sw = sub.shape
    px = max(2, min(8, max_px // max(sh, sw)))
    colour = np.array([[255, 255, 255], [105, 105, 105], [205, 205, 205], [0, 0, 0]], np.uint8)
    img = Image.fromarray(colour[sub[::-1]]).resize((sw * px, sh * px), Image.NEAREST)
    d = ImageDraw.Draw(img)

    def P(x, y):                         # world metres -> image pixels
        return ((x - gmin[0]) / RES - x0) * px, (sh - ((y - gmin[1]) / RES - y0)) * px

    wx0, wy0 = gmin[0] + x0 * RES, gmin[1] + y0 * RES
    for gx in range(math.ceil(wx0), math.floor(wx0 + sw * RES) + 1):          # 1 m grid + labels
        a, _ = P(gx, 0)
        d.line([(a, 0), (a, sh * px)], fill=(150, 190, 255), width=1)
        d.text((a + 2, 2), f"{gx}", fill=(0, 70, 200))
    for gy in range(math.ceil(wy0), math.floor(wy0 + sh * RES) + 1):
        _, b = P(0, gy)
        d.line([(0, b), (sw * px, b)], fill=(150, 190, 255), width=1)
        d.text((2, b + 2), f"{gy}", fill=(0, 70, 200))
    for z in zones:
        d.polygon([P(*q) for q in z], fill=(255, 90, 90), outline=(200, 0, 0))
    for t in trail:
        a, b = P(*t)
        d.ellipse([a - 2, b - 2, a + 2, b + 2], fill=(30, 100, 230))
    for pid, q in pins:
        a, b = P(*q)
        d.ellipse([a - 8, b - 8, a + 8, b + 8], fill=(255, 200, 0), outline=(0, 0, 0))
        d.text((a - 3, b - 5), str(pid), fill=(0, 0, 0))
    for fid, q in frontiers:
        a, b = P(*q)
        d.ellipse([a - 11, b - 11, a + 11, b + 11], fill=(40, 170, 70), outline=(0, 70, 0), width=2)
        d.text((a - 8, b - 5), f"F{fid}", fill=(255, 255, 255))
    a, b = P(*car)                        # robot: arrow in its heading
    ux, uy = math.cos(heading), -math.sin(heading)
    tip, base = (a + ux * 16, b + uy * 16), (a - ux * 8, b - uy * 8)
    left, right = (base[0] - uy * 9, base[1] + ux * 9), (base[0] + uy * 9, base[1] - ux * 9)
    d.polygon([tip, left, right], fill=(20, 80, 255), outline=(255, 255, 255))
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


# ── LLM client ───────────────────────────────────────────────────────────────

class Brain:
    def __init__(self, log=print):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=API_TIMEOUT, max_retries=1)
        self.log = log
        self.history = []                   # one text line per past decision + its result
        self.decisions = 0
        self.tokens_in = self.tokens_out = self.tokens_cached = 0
        self.use_fallbacks = True

    def note_result(self, line):
        self.history.append(line)
        del self.history[:-HISTORY_LINES]

    def _call(self, content):
        kw = dict(model=MODEL, max_tokens=4000, system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                  tools=TOOLS, tool_choice={"type": "auto"}, messages=[{"role": "user", "content": content}],
                  output_config={"effort": EFFORT})
        if self.use_fallbacks:
            try:
                return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                        extra_body={"fallbacks": "default"}, **kw)
            except self.anthropic.BadRequestError as e:
                self.log(f"[ai] fallbacks not accepted ({e}); continuing without")
                self.use_fallbacks = False
        return self.client.messages.create(**kw)

    def decide(self, task, map_png, cam_jpg, status, allowed):
        """-> (tool_name, args). Raises BrainError on any API / parsing problem."""
        if self.decisions >= MAX_DECISIONS:
            raise DecisionLimit(f"decision limit ({MAX_DECISIONS}) reached")
        content = [{"type": "text", "text": "TASK: " + (task or "Explore and map everything reachable.")}]
        if self.history:
            content.append({"type": "text", "text": "RECENT ACTIONS (oldest first):\n" + "\n".join(self.history)})
        content.append({"type": "text", "text": "Map:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                    "data": base64.standard_b64encode(map_png).decode()}})
        if cam_jpg:
            content.append({"type": "text", "text": "Camera (what the robot sees right now):"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.standard_b64encode(cam_jpg).decode()}})
        content.append({"type": "text", "text": status + "\nAllowed tools this turn: " + ", ".join(sorted(allowed)) +
                        ".\nCall exactly one tool."})
        try:
            resp = self._call(content)
        except self.anthropic.APIError as e:
            raise BrainError(f"{type(e).__name__}: {e}") from e
        self.decisions += 1
        u = resp.usage
        self.tokens_in += u.input_tokens
        self.tokens_out += u.output_tokens
        self.tokens_cached += getattr(u, "cache_read_input_tokens", 0) or 0
        if resp.stop_reason == "refusal":
            raise BrainError("the model declined this request")
        call = next((b for b in resp.content if b.type == "tool_use"), None)
        if call is None:
            raise BrainError(f"no tool call (stop_reason={resp.stop_reason})")
        if call.name not in allowed:
            raise BrainError(f"tool {call.name!r} is not allowed now")
        args = call.input if isinstance(call.input, dict) else json.loads(call.input)
        self.log(f"[ai] #{self.decisions} {call.name} {json.dumps(args)[:160]} "
                 f"(in={u.input_tokens} out={u.output_tokens} cached={getattr(u, 'cache_read_input_tokens', 0) or 0})")
        return call.name, args


# ── chat ─────────────────────────────────────────────────────────────────────

CHAT_SYSTEM = """You are the voice of a small wheeled indoor robot (about 30 cm wide) with a 2D LiDAR and a low, forward-facing camera. You chat with its owner in a web page next to the robot's live map.
Every user message comes with the robot's current status, the top-down map image (when there is one; legend: white = seen free floor, dark gray = unseen, black = walls, light gray = keep-out margin, red = no-go boxes, yellow discs = photo pins, blue arrow = the robot; grid labels are metres, +x right, +y up) and what the camera sees right now.
Answer questions about what the robot sees, where it is, what the map shows, and what it could do. Be concise and friendly; say plainly when you cannot tell something from the image (the camera is very low, so the view is a floor-level one).
You can act, but you never drive directly:
- propose_task(task): offer the owner a task for the robot's autonomous mode (e.g. "map the living room, skip the garage", "go to the dining table and take a photo"). The robot only moves after the owner presses a Start button, so propose when they ask for something to be done and say so in your reply.
- pin_photo(note): save the current camera frame as a labelled photo pin where the robot is.
- stop_robot(): stop all motion immediately (use it at once if asked to stop).
Stairs, steps and ledges are the main danger to this robot; mention any you notice."""

CHAT_TOOLS = [
    {"name": "propose_task", "description": "Offer the owner a task for the robot's autonomous mode; it starts only when they press Start.",
     "input_schema": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"], "additionalProperties": False}},
    {"name": "pin_photo", "description": "Save the current camera frame as a labelled photo pin at the robot's position.",
     "input_schema": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"], "additionalProperties": False}},
    {"name": "stop_robot", "description": "Stop all robot motion immediately.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]
CHAT_TURNS = 16


class Chat:
    """Conversation with the robot's brain. History keeps text only; images go with the newest message."""

    def __init__(self, log=print):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=90.0, max_retries=1)
        self.log = log
        self.history = []                      # [{"role", "content": str}]
        self.model = os.environ.get("LLM_CHAT_MODEL", MODEL)

    def reset(self):
        self.history = []

    def reply(self, text, map_png, cam_jpg, status, handler, on_text=None):
        """-> reply text. handler(name, args) -> str result for each tool call; on_text(delta) gets the reply as it streams."""
        content = [{"type": "text", "text": f"[Robot status]\n{status}"}]
        if map_png:
            content += [{"type": "text", "text": "[Map]"}, {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(map_png).decode()}}]
        if cam_jpg:
            content += [{"type": "text", "text": "[Camera now]"}, {"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(cam_jpg).decode()}}]
        content.append({"type": "text", "text": text})
        messages = self.history[-CHAT_TURNS:] + [{"role": "user", "content": content}]
        out = ""
        try:
            for _ in range(4):                                  # answer -> tools -> answer
                if out and on_text:
                    on_text("\n\n")
                with self.client.messages.stream(
                        model=self.model, max_tokens=3000, tools=CHAT_TOOLS, messages=messages,
                        system=[{"type": "text", "text": CHAT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                        output_config={"effort": EFFORT}) as stream:
                    for delta in stream.text_stream:
                        if on_text:
                            on_text(delta)
                    resp = stream.get_final_message()
                self.log(f"[chat] in={resp.usage.input_tokens} out={resp.usage.output_tokens} stop={resp.stop_reason}")
                if resp.stop_reason == "refusal":
                    raise BrainError("the model declined this request")
                out += "".join(b.text for b in resp.content if b.type == "text")
                calls = [b for b in resp.content if b.type == "tool_use"]
                if not calls:
                    break
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": c.id, "content": handler(c.name, c.input)} for c in calls]})
        except self.anthropic.APIError as e:
            raise BrainError(f"{type(e).__name__}: {e}") from e
        out = out.strip() or "(no reply)"
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": out}]
        del self.history[:-2 * CHAT_TURNS]
        return out
