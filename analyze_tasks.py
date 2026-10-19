"""Task-trial analysis for the paper.  Usage:  uv run python analyze_tasks.py [runs_dir]

Inputs
  runs/<cond>_<task>_<env>_t<k>_<time>.jsonl   written by the server when EXP_LABEL is set   (cond e.g. full, mid, primitive, full-mem)
  runs/outcomes.csv                            YOU fill this in after each trial, with a tape measure:
                                               label,success,final_error_m,interventions,collisions,notes
                                               (label = the EXP_LABEL, success = 1 or 0)
Outputs
  printed tables, paper/results_table.md, paper/fig_learning.png, paper/fig_success.png
Only trials that have a row in outcomes.csv are counted, so nothing unverified reaches the paper."""
import csv
import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "runs"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def parse(label):
    parts = label.split("_")
    cond = parts[0]
    task = parts[1] if len(parts) > 1 else "?"
    k = next((int(p[1:]) for p in parts if p.startswith("t") and p[1:].isdigit()), 0)
    return cond, task, k


outcomes = {}
try:
    for r in csv.DictReader(open(os.path.join(d, "outcomes.csv"))):
        outcomes[r["label"].strip()] = r
except OSError:
    print("no runs/outcomes.csv yet - nothing to analyse")
    sys.exit(0)

trials = []
for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    ev = [json.loads(l) for l in open(p)]
    if not ev or ev[0].get("event") != "start":
        continue
    label = ev[0]["label"]
    if label not in outcomes:
        continue
    dec = [e for e in ev if e["event"] == "decision"]
    end = next((e for e in ev if e["event"] == "end"), {})
    o = outcomes[label]
    cond, task, k = parse(label)
    errs = o.get("final_error_m", "").strip()
    trials.append(dict(label=label, cond=cond, task=task, k=k, ok=int(o["success"]), time=end.get("t", ev[-1]["t"]),
                       decisions=len(dec), tokens=(dec[-1]["tokens_in"] + dec[-1]["tokens_out"]) if dec else 0,
                       err=float(errs) if errs else None, interv=int(o.get("interventions") or 0),
                       coll=int(o.get("collisions") or 0), bad=sum(1 for e in dec if str(e["result"]).startswith("error")),
                       memory=ev[0].get("memory"), skills=ev[0].get("skills")))

if not trials:
    print("outcomes.csv has no row matching a runs/*.jsonl label")
    sys.exit(0)

groups = defaultdict(list)
for t in trials:
    groups[(t["cond"], t["task"])].append(t)


def med(v):
    v = [x for x in v if x is not None]
    return f"{st.median(v):.1f}" if v else "-"


lines = ["| condition | task | n | success | 95% CI | median time (s) | median LLM calls | median tokens | median final error (m) | rejected calls | interventions |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for (c, tk), r in sorted(groups.items()):
    k = sum(x["ok"] for x in r)
    lo, hi = wilson(k, len(r))
    lines.append(f"| {c} | {tk} | {len(r)} | {k}/{len(r)} | {lo:.2f}-{hi:.2f} | {med([x['time'] for x in r])} | {med([x['decisions'] for x in r])} | "
                 f"{med([x['tokens'] for x in r])} | {med([x['err'] for x in r])} | {sum(x['bad'] for x in r)} | {sum(x['interv'] for x in r)} |")
table = "\n".join(lines)
print(table)
os.makedirs("paper", exist_ok=True)
open("paper/results_table.md", "w").write(table + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)

# success by condition and task
conds = sorted({t["cond"] for t in trials})
tasks = sorted({t["task"] for t in trials})
fig, ax = plt.subplots(figsize=(5, 3))
w = 0.8 / max(1, len(conds))
for i, c in enumerate(conds):
    xs, ys, lo, hi = [], [], [], []
    for j, tk in enumerate(tasks):
        r = groups.get((c, tk), [])
        if r:
            k = sum(x["ok"] for x in r)
            a, b = wilson(k, len(r))
            xs.append(j + i * w)
            ys.append(k / len(r))
            lo.append(k / len(r) - a)
            hi.append(b - k / len(r))
    ax.bar(xs, ys, w, yerr=[lo, hi], capsize=2, label=c)
ax.set_xticks([j + 0.4 - w / 2 for j in range(len(tasks))])
ax.set_xticklabels(tasks)
ax.set_ylabel("success rate")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("paper/fig_success.png", dpi=200)

# learning curve: running success across trial order for each condition (pooled over tasks)
fig, ax = plt.subplots(figsize=(5, 3))
for c in conds:
    r = sorted([t for t in trials if t["cond"] == c], key=lambda t: t["k"])
    if len(r) < 3:
        continue
    run = [sum(x["ok"] for x in r[max(0, i - 2):i + 1]) / len(r[max(0, i - 2):i + 1]) for i in range(len(r))]
    ax.plot([x["k"] for x in r], run, marker="o", label=c)
ax.set_xlabel("trial index")
ax.set_ylabel("success (moving average of 3)")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("paper/fig_learning.png", dpi=200)
print("wrote paper/results_table.md, fig_success.png, fig_learning.png")
