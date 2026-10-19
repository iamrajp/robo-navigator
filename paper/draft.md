# LLM Task Planning and Execution on a Low-Cost LiDAR Robot: From Route Planning to Object Pushing

*(Draft v2, 2026-10-18. Everything marked **[RESULT]** or **[TODO]** is a placeholder. No experimental numbers exist yet and none
may be written in until they come from `runs/` via `analyze_runs.py`. See `PROTOCOL.md`.)*

## Abstract
We present a low-cost indoor robot (Raspberry Pi 5, 2D LiDAR, low monocular camera, differential-drive chassis) in which a
vision-language model (LLM) turns an open-ended natural-language instruction (e.g. "push the ball into the kitchen",
"take a photo of the sofa", "map the room but stay away from the stairs") into a sequence of executable actions. The LLM
does not drive. At every step it sees a rendered top-down map, the live camera frame and a status text, and emits one call to
a small, typed skill library (navigate, explore, turn, pin photo, add no-go zone, and **push object to target**). A classical
layer (ICP localization, A* planning, obstacle detours, a camera ledge detector, stuck detection) validates and executes each
call, so a wrong plan cannot make the robot unsafe. We evaluate on **[TODO N]** tasks in **[TODO M]** rooms against a scripted
nearest-frontier/hard-coded baseline and report task success, steps, wall-clock time, LLM calls and cost. **[RESULT]**

## 1. Introduction
Affordable robots execute fixed behaviours; adding a new task means writing new code. LLMs can decompose instructions, but
published systems mostly use expensive manipulators or simulation. We ask: *how far can an LLM planner go on a ~$150 wheeled
platform whose only "manipulator" is its own body?* Pushing a ball is the simplest non-trivial instance: the robot must find the
object, reason about where to stand, move it to a goal location, and recover when the ball rolls away.

Contributions:
1. A safety-layered architecture in which the LLM chooses among planner-validated skills; invalid calls return an error string
   the model sees and can correct (Section 3.5).
2. A `push_object` skill (stage behind object, push along the object-to-goal line, verify) and an object-grounding step that
   places camera detections on the LiDAR map (Section 3.6).
3. An on-device LiDAR ICP localizer/mapper with relocalization and a camera ledge detector, all on a Pi 5 (Section 3.2-3.4).
4. An evaluation of instruction-following success across navigation, mapping and pushing tasks, with ablations of the
   LLM's inputs (map image, camera, history) **[RESULT]**.

## 2. Related Work (all citations to be verified before submission)
- **LLM planning with robots:** SayCan (Ahn et al., 2022), Inner Monologue (Huang et al., 2022), Code as Policies (Liang et al., ICRA 2023), ReAct (Yao et al., ICLR 2023).
- **Language + navigation:** LM-Nav (Shah et al., CoRL 2022), VLMaps (Huang et al., ICRA 2023), NavGPT (Zhou et al., AAAI 2024), ESC (Zhou et al., ICML 2023), L3MVN (Yu et al., IROS 2023).
- **Classical building blocks:** frontier exploration (Yamauchi, CIRA 1997); ICP (Besl & McKay, 1992); 2D LiDAR SLAM (Hector, Gmapping, Cartographer).
- **Non-prehensile pushing:** Mason (1986), Lynch & Mason, "Stable pushing" (1996); learned pushing with RL.
- **[TODO]** add 2024-2026 work on LLM agents for low-cost / mobile robots and on VLM-based object-goal navigation; state clearly what is new here (the combination on commodity hardware plus an open, reproducible evaluation, not the individual parts).

## 3. System
### 3.1 Hardware
Raspberry Pi 5, serial 2D LiDAR, low forward camera (~10 cm above floor), differential-drive car, total cost **[TODO]**.
Compute is on the Pi; only LLM calls leave the device.

### 3.2 Localization and mapping (`Localizer`, `nav_server.py`)
Voxel downsampling, scan-to-map ICP, tracking-quality counter, relocalization with a distance field, maps stored as PCD.

### 3.3 Planning and control (`explore.py`, `navigate.py`)
Probabilistic occupancy grid, inflated obstacles, A* with wall-clearance penalty, user and LLM-authored no-go zones, short legs (<= 1.5 m).

### 3.4 Safety layer
Camera ledge score, automatic no-go box plus back-off, obstacle detour, stuck detection, per-run decision cap, API-failure fallback.
The LLM cannot remove a no-go box it created.

### 3.5 LLM decision layer (`llm_brain.py`)
Input per step: task text, last 10 action/result lines, rendered map PNG (free/unknown/walls/margins/zones/pins/frontiers/robot arrow, metre grid),
camera JPEG, status text, list of allowed tools. Output: exactly one tool call with a <= 25-word reason. Skills implemented today:
`explore_to`, `go_to`, `turn_to`, `pin_photo`, `add_no_go`, `finish`. Invalid arguments come back as `error: ...` text in the history.
Chat/Telegram front-end can only *propose* a task; the robot moves after the owner confirms.

### 3.6 Object grounding and pushing  **[TODO: not implemented yet]**
- *Grounding* (`objects.py`, implemented): the VLM returns a label and horizontal image fraction; the nearest LiDAR return inside a +/-4 deg wedge at that bearing gives world (x, y). Needs calibration of camera FOV and left/right sign (measure placement error vs tape, **[RESULT]**).
- *Ball detection for pushing* **[TODO]**: a ball lies below the 2D LiDAR plane or is small, so it may not appear in the scan. Options: VLM bearing plus floor-plane range from image row (camera height known), or lower the LiDAR hit threshold. Choose one after a pilot.
- *`push_object(label, target_x, target_y)`* **[TODO]**: (1) compute the push line from ball to target; (2) plan to a staging pose 0.35 m behind the ball on that line (ball treated as a temporary non-obstacle); (3) align heading; (4) drive forward in short pulses, re-detecting the ball after each pulse; (5) stop within a tolerance of the target or after a retry/step cap; (6) return success/failure text (distance remaining, ball lost) to the LLM, which can re-stage and retry.
- Failure handling: ball lost from view, ball stuck against a wall, ledge ahead (ledge check stays active during pushing).

## 4. Experiments
Protocol in `PROTOCOL.md`. Research questions:
- **RQ1** Can the LLM turn free-form instructions into successful skill sequences? (success rate by task type)
- **RQ2** Does it beat a scripted baseline and what does it cost (steps, time, tokens, dollars)?
- **RQ3** Which inputs matter? (ablate map image, camera frame, history)
- **RQ4** Pushing: success rate and final error to target vs ball start distance, and number of re-stages.
- **RQ5** Safety: ledge false positives/negatives, collisions, unsafe LLM proposals blocked by the validator.

Table 1 task suite **[TODO]**; Table 2 success by task and condition **[RESULT]**; Table 3 cost **[RESULT]**; Fig. map + trajectory for one pushing trial **[RESULT]**.

## 5. Discussion and Limitations
Latency per decision, cloud dependency, cost, non-determinism, 2D LiDAR blind to low objects, no gripper (pushing only), small number of rooms and a single robot,
camera/LiDAR extrinsics from tape measurements. Failure taxonomy to be filled from logs **[RESULT]**.

## 6. Conclusion
**[TODO after results]**

## References
**[TODO]** IEEE style; verify every entry listed in Section 2 against the original source before use.
