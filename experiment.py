"""Per-run metrics for exploration experiments: one JSON line per decision step in runs/<label>_<time>.jsonl.

Every run is logged automatically (set EXP_LOG=0 to turn this off). Name a run with the EXP_LABEL env var
(e.g. EXP_LABEL=full_push_roomA_t3); without it the label is auto-made from the skill level and the task text.
Images (map the LLM saw, camera frames before/after each decision) go in runs/<label>_<time>/ next to the .jsonl.
analyze_runs.py / analyze_tasks.py turn the files into the tables used in the paper."""
import json
import math
import os
import re
import subprocess
import time

from navigate import RES


class RunLog:
    def __init__(self, mode, env_label=None, task="", **extra):
        self.f = None
        if os.environ.get("EXP_LOG", "1") == "0":
            return
        slug = re.sub(r"[^a-z0-9]+", "-", (task or "explore").lower()).strip("-")[:30] or "explore"
        auto = f"{os.environ.get('LLM_SKILLS', 'full') if mode == 'ai' else mode}_{slug}"
        self.label = env_label or os.environ.get("EXP_LABEL") or auto
        os.makedirs("runs", exist_ok=True)
        self.path = f"runs/{self.label}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self.f = open(self.path, "w")
        self.t0, self.last, self.dist, self.steps = time.time(), None, 0.0, 0
        self.photo_dir, self.photos = self.path[:-6], 0
        try:
            import llm_brain
            model, effort = llm_brain.MODEL, llm_brain.EFFORT
        except Exception:
            model = effort = None
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=3).stdout.strip()
            dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "*.py"], capture_output=True, text=True, timeout=3).stdout.strip())
        except Exception:
            commit, dirty = "", None
        self.write({"event": "start", "mode": mode, "label": self.label, "task": task, "model": model, "effort": effort,
                    "git_commit": commit, "code_modified": dirty, "map": extra.get("map"), "places": extra.get("places"),
                    "start_pose": extra.get("pose"), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "skills": os.environ.get("LLM_SKILLS", "full"), "memory": os.environ.get("LLM_MEMORY") == "1",
                    "prompt_commit": os.environ.get("PROMPT_COMMIT", "")})

    def write(self, d):
        if self.f:
            d["t"] = round(time.time() - self.t0, 2) if hasattr(self, "t0") else 0
            self.f.write(json.dumps(d) + "\n")
            self.f.flush()

    def event(self, kind, **kw):
        """Free-form event (ledge alarm, push progress, user stop...)."""
        self.write({"event": kind, **kw})

    def snap(self, jpg, tag):
        """Save a camera frame next to the log: runs/<label>_<time>/<n>_<tag>.jpg (at most 600 per run)."""
        if not self.f or self.photos >= 600:
            return
        os.makedirs(self.photo_dir, exist_ok=True)
        self.photos += 1
        with open(f"{self.photo_dir}/{self.photos:03d}_{tag}.jpg", "wb") as fh:
            fh.write(jpg)
        name = f"{self.photos:03d}_{tag}.jpg"
        self.write({"event": "photo", "n": self.photos, "tag": tag, "file": name})
        return name

    def snap_png(self, png, tag):
        """Save the map image the LLM was shown. Returns the file name (relative to the run folder) or None."""
        if not self.f or not png:
            return None
        os.makedirs(self.photo_dir, exist_ok=True)
        self.photos += 1
        name = f"{self.photos:03d}_{tag}.png"
        with open(f"{self.photo_dir}/{name}", "wb") as fh:
            fh.write(png)
        return name

    def decision(self, name, args, result, brain, secs, pose=None, imgs=None, think_s=None):
        """One LLM decision and what happened (the raw material for success / failure analysis)."""
        self.write({"event": "decision", "name": name, "args": args, "result": str(result)[:300], "exec_s": round(secs, 1),
                    "think_s": None if think_s is None else round(think_s, 1), "pose": pose, "imgs": imgs or [],
                    "tokens_in": getattr(brain, "tokens_in", 0), "tokens_out": getattr(brain, "tokens_out", 0),
                    "n": getattr(brain, "decisions", 0)})

    def step(self, xy, grid, ex):
        """Call when the car is stopped, before the next decision."""
        if not self.f:
            return
        if self.last is not None:
            self.dist += math.dist(self.last, xy)
        self.last = xy
        self.steps += 1
        try:
            area = float(grid.known_free.sum()) * RES * RES           # m^2 of floor seen free
        except AttributeError:
            area = None
        brain = ex.brain
        self.write({"event": "step", "n": self.steps, "x": round(xy[0], 3), "y": round(xy[1], 3),
                    "dist_m": round(self.dist, 2), "free_area_m2": None if area is None else round(area, 2),
                    "bad_targets": len(ex.bad), "fails": ex.fails, "hazards": ex.hazards,
                    "decisions": brain.decisions if brain else 0,
                    "tokens_in": getattr(brain, "tokens_in", 0), "tokens_out": getattr(brain, "tokens_out", 0)})

    def end(self, reason, ex):
        if not self.f:
            return
        self.write({"event": "end", "reason": reason, "dist_m": round(self.dist, 2), "steps": self.steps,
                    "hazards": ex.hazards, "fails": ex.fails, "ai_calls": ex.ai_n})
        self.f.close()
        self.f = None
