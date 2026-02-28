# Handoff: Argument Analyzer Scoring Recalibration

## Current Goal
Recalibrate scoring to match target spread:
- **Low quality** near ~10
- **Medium quality** in ~30–70
- **Scientific/logical high quality** in ~70–90

User requirement: scoring should be mostly linear arithmetic around a reasonable starting point, with penalties stronger than rewards.

---

## Important Constraint
Do **not** change metadata extraction / prompt-injected meta hints (these are accepted as correct).
Only scoring calibration and UX labels/interpretation should be adjusted.

---

## Files Changed In This Session
- `analyzer.py`
- `templates/index.html`
- `static/style.css`
- `templates/share.html`
- `templates/public_analysis.html`
- `engagement.py`
- `README.md`
- `SYSTEM_LOGIC.txt`

New calibration scripts:
- `calibration/test_fiction_logic.py`
- `calibration/test_simulation_21.py`
- `calibration/test_simulation_tail_18_21.py`

---

## Current Scoring Status
`analyzer.py` has been partially moved toward a linear model in `_compute_score_artifacts_linear()` with:
- base anchored at 75
- linear penalties/rewards
- fiction mode bonus/protection
- low-word penalty gated by logic integrity

However, full calibration validation has not yet completed due long-running API calls.

---

## Known Calibration Friction
1. Long batch simulations can stall due API latency.
2. Need final verification that updated linear model actually achieves:
   - weak near ~10
   - medium ~30–70
   - strong/scientific ~70–90
3. Some earlier runs showed too many cases pegged around low logic bands; needs full post-change verification.

---

## UX/Label Decisions Already Requested
- Keep label wording:
  - low logic label: `illogical/logic absent`
- Logic panel title:
  - `Logic Analysis` (not Logic Snapshot)
- Bias/testability copy simplified for user readability.

---

## What To Do First In New Chat
1. **Read current `analyzer.py` scoring block** and confirm linear model is coherent.
2. Run simulation in **3 smaller batches** to avoid stalls.
3. Consolidate results into one 21-row table.
4. Compare against target bands and apply minimal coefficient tuning.

---

## Suggested Commands (batched)
Run from project root:

```bash
python -u "calibration/test_simulation_21.py"
```

If full run stalls, run targeted batches by creating temporary small scripts or using inline Python:
- Batch A: IDs 1–7
- Batch B: IDs 8–14
- Batch C: IDs 15–21

Then merge outputs into one table:
- `ID, quality, words, score, bias, testability, logic, issues, status`

---

## Regression Spot Checks To Always Re-run
Short weak phrase:
- `"this app makes no sense"`

Should remain very low and not present as logically strong.

Fiction structured case:
- Ensure fiction internal coherence can score reasonably even with low evidence cues.

Scientific long case:
- Should be capable of reaching 70–90 after calibration.

---

## Success Criteria Before Finalizing
1. 21-case matrix completes.
2. Band alignment is approximately:
   - weak cluster near 0–20 (center near 10)
   - medium mostly 30–70
   - scientific/logical high mostly 70–90
3. No contradictions between numeric dimensions and status wording.
4. No regressions in app run / rendering.

