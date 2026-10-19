"""Create/extend runs/outcomes.csv with one blank row per logged run, ready to fill in by hand.
Usage: uv run python make_outcomes.py [runs_dir]      (safe to re-run: existing rows are never touched)
Fill in after each trial, using a tape measure / your eyes - not the robot's own log:
  success         1 or 0 (task criterion from paper/PROTOCOL.md)
  final_error_m   distance of final pose / ball from the tape mark, metres (blank if not applicable)
  interventions   times you had to touch/stop the car
  collisions      bumps or falls
  notes           why it failed, what happened"""
import csv
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "runs"
path = os.path.join(d, "outcomes.csv")
cols = ["label", "success", "final_error_m", "interventions", "collisions", "notes"]
have = set()
if os.path.exists(path):
    have = {r["label"].strip() for r in csv.DictReader(open(path))}
new = []
for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    try:
        ev = json.loads(open(p).readline())
    except ValueError:
        continue
    if ev.get("event") == "start" and ev.get("label") and ev["label"] not in have:
        have.add(ev["label"])
        new.append(ev["label"])
write_header = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(cols)
    for lab in new:
        w.writerow([lab, "", "", "", "", ""])
print(f"{path}: added {len(new)} blank row(s)" + "".join(f"\n  {l}" for l in new))
