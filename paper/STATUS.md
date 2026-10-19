# Paper work status (2026-10-18, updated)

Paper is now `paper.tex` (IEEEtran, 4 capabilities: safe roaming/mapping, route planning, status updates, tool-free pushing).
`draft.md` is the older outline. All results are \todo placeholders: `runs/` is empty and `runs/outcomes.csv` does not exist.

## Code state (changed since last note)
- `push.py` + `test_push_sim.py` exist; `push_object` is wired into nav_server.py / llm_brain.py. Skill levels full|mid|primitive (LLM_SKILLS), memory (LLM_MEMORY=1).
- `analyze_tasks.py` builds tables/figures from runs/*.jsonl + runs/outcomes.csv (tape-measured).

## Restructured 2026-10-18: paper now centres on skill-level ablation (H1-H3). See TRIAL_PLAN.md (70 trials).

## Next steps
1. Run trials with EXP_LABEL=<cond>_<task>_<env>_t<k>, fill runs/outcomes.csv after each (>=5 per cell).
2. Status-update task is not in analyze_tasks yet: define a check (answer vs logged state).
3. Measure ledge detector TP/FP, collisions/falls count for the safety claim.
4. Verify all citations, add 2024-26 related work, fill affiliation/cost, install LaTeX (pdflatex missing on the Pi) to build.

## 2026-10-18 (later)
Added contribution (iii) + subsection "Owner-authored place pins" (name/description/icon/picture/instructions; LLM follows instructions on arrival). Evaluation is a \todo: run >=5 trials per instruction type (e.g. "photo then turn all round", "report what you see") and score followed/not.
