"""Summarise runs/*.jsonl: per-condition mean/std of coverage, time, distance, failures, LLM cost.
Run names are <condition>_<env>_<trial>, e.g. ai_roomA_t1 -> condition 'ai'. Usage: python analyze_runs.py [runs_dir]"""
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "runs"
rows = defaultdict(list)
for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    ev = [json.loads(l) for l in open(p)]
    steps = [e for e in ev if e["event"] == "step" and e.get("free_area_m2") is not None]   # nav-mode steps have no area
    end = next((e for e in ev if e["event"] == "end"), None)
    if not steps or not end:
        continue
    label = ev[0]["label"].split("_")
    cond, env = label[0], label[1] if len(label) > 1 else "?"
    last = steps[-1]
    t90 = next((s["t"] for s in steps if s["free_area_m2"] >= 0.9 * last["free_area_m2"]), last["t"])
    rows[(cond, env)].append(dict(area=last["free_area_m2"], time=end["t"], t90=t90, dist=end["dist_m"],
                                  steps=end["steps"], fails=end["fails"], hazards=end["hazards"],
                                  tok=last["tokens_in"] + last["tokens_out"]))

def ms(v):
    return f"{st.mean(v):.1f}±{st.stdev(v):.1f}" if len(v) > 1 else f"{v[0]:.1f}"

print(f"{'cond':10}{'env':8}{'n':>3}{'area m2':>14}{'time s':>14}{'t90 s':>14}{'dist m':>14}{'fails':>10}{'tokens':>12}")
for (c, e), r in sorted(rows.items()):
    print(f"{c:10}{e:8}{len(r):>3}" + "".join(f"{ms([x[k] for x in r]):>14}" for k in ("area", "time", "t90", "dist"))
          + f"{ms([x['fails'] for x in r]):>10}{ms([x['tok'] for x in r]):>12}")
