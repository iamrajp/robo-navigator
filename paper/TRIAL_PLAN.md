# Trial plan (focus: skill-level ablation, hypotheses H1-H3 in paper.tex)

Write the hypotheses down before running. Freeze the prompt + commit hash (`git rev-parse --short HEAD`) per condition. Randomise trial order. Cap 15 min.

## Core cells (label = `<cond>_<task>_<env>_t<k>`)
| Task | Conditions | Trials each | Total |
|---|---|---|---|
| push ball to tape mark | full, mid, primitive | 10 | 30 |
| push, memory (run in order t1..t10, `LLM_MEMORY=1`) | primitive-mem | 10 | 10 |
| push, scripted baseline (ball position given) | script | 10 | 10 |
| route to a named place | full, mid, primitive | 5 | 15 |
| route, plain A* baseline | script | 5 | 5 |
| **Total** | | | **70** |

Use 2 rooms (alternate), tape-mark the target and a fixed start; the ball start varies over 3 marked positions.
Optional if time: mapping coverage (3 conds x 3 runs) and status questions (20 questions, check vs logged state).

## Per trial
1. `EXP_LABEL=<label> LLM_SKILLS=<full|mid|primitive> [LLM_MEMORY=1] uv run python nav_server.py`, give the task, press Start.
2. Measure with tape: ball-to-mark distance (push) or final pose-to-mark (route). Success: push <= 0.25 m, route <= 0.5 m (paper.tex).
3. `uv run python make_outcomes.py`, then fill success / final_error_m / interventions / collisions / notes.

## Safety evidence (H3)
- Count falls + collisions + interventions over all 70 trials (outcomes.csv).
- Count validator-rejected calls (`error:` results in the jsonl; analyze_tasks prints `bad`).
- Ledge detector TP/FP: label the images in `explore_hazards/` (real drop yes/no) and report precision/recall.

## Honest reporting
Report every trial including failures, N per cell, Wilson 95% CIs. If H1 or H2 is false, say so. Pilot runs in runs/ (unlabelled, n=1) stay as anecdotes only.
