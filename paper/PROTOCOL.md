# Experiment protocol (v2, 2026-10-18)

**Claim to test:** an LLM that sees a map image and camera frame, and can only call validated skills, can carry out free-form
tasks (navigate, explore, photograph, avoid areas, push a ball to a target) on a Pi 5 LiDAR car, at acceptable cost and safety.

## Prerequisite: implement `push_object` (and ball detection). Nothing in Section 4 of the paper can be run for pushing until this exists.

## Task suite (aim for 5 task types x 5 variants, each repeated >= 5 times)
| Type | Example instruction | Success criterion |
|---|---|---|
| A navigate | "go to the sofa" | final pose within 0.5 m of ground-truth marker |
| B photo | "take a photo of the door" | pin exists, door visible in frame (human-judged) |
| C avoid | "explore but stay out of the kitchen" | coverage >= X%, zero entry into region |
| D push | "push the ball to the tape mark" | ball centre within 0.3 m of mark, <= 15 min |
| E compound | "find the ball, push it into the kitchen, then photograph it" | all sub-goals met |

## Conditions
- `llm` - full system
- `llm_nomap` / `llm_nocam` / `llm_nohist` - ablations (env var switches to add in `llm_brain.py`)
- `script` - hand-written controller for the same tasks (no LLM); for push this is a fixed stage-and-push routine with ball position given
- optional: smaller/cheaper model

## Procedure
1. Fixed start pose (tape), same battery state, lights on, 2-3 rooms. Measure ground-truth marks with a tape.
2. `EXP_LABEL=<cond>_<task>_<env>_t<k> uv run python nav_server.py`; start via chat/Telegram. Randomise trial order; do not tune prompts between trials of one condition (freeze prompt and commit hash).
3. Cap each trial at 15 min / `LLM_MAX_DECISIONS`. Log: success, time, steps, LLM calls, tokens, collisions, human interventions, failure reason.
4. `uv run python analyze_runs.py`; report success with Wilson 95% CI, time/cost as median and IQR.

## Extra measurements
- Object placement error (grounded x,y vs tape), 20 objects.
- Localization RMSE at 10 tape checkpoints per room.
- Ledge detector true/false positives (`explore_hazards/`).
- Count of LLM calls rejected by the validator (safety evidence).

## Honesty rules
No number goes into the paper unless it is in `runs/`. Report failed trials. State N for every cell.
