"""
Hint-based scoring: 3 dimensions from 0/1/2 hints.
- Bias: high = more issues (reduces score)
- Testability: high = razors passed (improves score)
- Logic integrity: high = logic hints present (improves score)
- Argument strength: composite 0-100
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "phase1_artifacts" / "hint_scoring_config.json"


def _load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _hint_val(hints, key, default=0):
    v = hints.get(key, default)
    if v in ("?", None):
        return 0
    return int(v)


def _penalty_bias_bonus(detected_issues, all_razors_passed=False):
    """
    Add bias from detected issues (penalties). Scaled down so hints remain primary.
    Use top 6 by impact (|penalty| * confidence) to avoid inflation from many weak matches.
    When all razors pass, halve the bonus so epistemically strong arguments (incl. fictional) score higher.
    """
    if not detected_issues:
        return 0.0
    impacts = []
    for cat in ("logical_fallacies", "cognitive_biases", "cognitive_distortions"):
        for item in (detected_issues.get(cat) or []):
            if isinstance(item, dict):
                p = abs(int(item.get("penalty", 0) or 0))
                c = float(item.get("confidence", 1.0) or 1.0)
                impacts.append(p * c)
    if not impacts:
        return 0.0
    impacts.sort(reverse=True)
    total = sum(impacts[:6])
    # Scale: ~60 for top 6 typical; add up to 30 bias points
    bonus = min(30.0, total / 60.0 * 30.0)
    if all_razors_passed:
        bonus *= 0.5
    return bonus


def _vacuous_bias_floor(argument_text, logic_score):
    """
    Very short, low-logic arguments (e.g. "everything sucks") are inherently
    biased — absolute language, affective, no structure. Add bias floor.
    """
    if not argument_text:
        return 0.0
    words = len(argument_text.strip().split())
    if words < 8 and logic_score < 55:
        return 25.0
    if words < 15 and logic_score < 45:
        return 15.0
    return 0.0


def compute_scores(hint_vector_012, detected_issues=None, argument_text=None):
    """
    Compute bias, testability, logic integrity, and argument strength from hints.
    Optional detected_issues: add bias bonus from penalties (scaled down).
    Optional argument_text: add vacuous-content bias floor for short, weak arguments.
    Returns dict with bias_score, testability_score, logic_score, argument_strength.
    """
    cfg = _load_config()
    hints = hint_vector_012 or {}

    # Bias: 0-100, higher = more issues. Average of 0/1/2 normalized.
    bias_hints = cfg.get("bias_hints", [])
    if bias_hints:
        bias_sum = sum(_hint_val(hints, k) for k in bias_hints)
        bias_score = min(100.0, 100.0 * bias_sum / (len(bias_hints) * 2))
    else:
        bias_score = 0.0

    # Razor alignment: % of razors that passed (needed before bias bonus)
    razors = (detected_issues or {}).get("philosophical_razors", [])
    num_razors = len(razors)
    passed_razors = sum(
        1 for r in razors
        if isinstance(r, dict) and r.get("pass", False)
    )
    razor_alignment = 100.0 * (passed_razors / num_razors) if num_razors else 0.0
    all_razors_passed = num_razors > 0 and passed_razors == num_razors

    # Bias bonus from detected issues (penalties) — catches vacuous/high-bias text
    # When all razors pass, halve penalty so epistemically strong content (incl. fictional) scores higher
    bias_score = min(100.0, bias_score + _penalty_bias_bonus(detected_issues or {}, all_razors_passed))

    # Testability: 0-100, higher = better. Razor hints: 0=good, 2=bad.
    # When all razors pass, use razor_alignment so testability reflects epistemic strength.
    razor_hints = cfg.get("razor_hints", [])
    if razor_hints:
        good_scores = []

        def _good(v):
            return (2 - v) / 2.0

        for k in razor_hints:
            v = _hint_val(hints, k)
            good_scores.append(_good(v))
        testability_from_hints = 100.0 * (sum(good_scores) / len(razor_hints))
    else:
        testability_from_hints = 50.0

    # When all razors pass, testability should be high; blend hint-based with razor alignment
    testability_score = max(testability_from_hints, razor_alignment)

    # Logic integrity: 0-100. Positive hints: high=good. Negative hints: high=bad.
    pos_hints = cfg.get("logic_hints_positive", [])
    neg_hints = cfg.get("logic_hints_negative", [])

    pos_score = 0.0
    if pos_hints:
        pos_score = sum(_hint_val(hints, k) for k in pos_hints) / (len(pos_hints) * 2)

    neg_score = 0.0
    if neg_hints:
        neg_score = sum((2 - _hint_val(hints, k)) / 2.0 for k in neg_hints) / len(neg_hints)

    if pos_hints and neg_hints:
        logic_score = 100.0 * (pos_score + neg_score) / 2
    elif pos_hints:
        logic_score = 100.0 * pos_score
    elif neg_hints:
        logic_score = 100.0 * neg_score
    else:
        logic_score = 50.0

    # Vacuous-content bias floor: short + weak logic = inherently biased
    bias_score = min(100.0, bias_score + _vacuous_bias_floor(argument_text or "", logic_score))

    # Composite: multiplicative — hard to score high, easy to score low.
    # One weak dimension pulls the score down; all three must be strong to score high.
    bias_quality = (100.0 - bias_score) / 100.0
    test_quality = testability_score / 100.0
    logic_quality = logic_score / 100.0
    argument_strength = bias_quality * test_quality * logic_quality * 100.0

    # Low-issues reward: when total issues below expected norm (4), push unbiased statements up
    di = detected_issues or {}
    total_issues = sum(
        len(di.get(cat, []))
        for cat in ("logical_fallacies", "cognitive_biases", "cognitive_distortions")
    )
    if total_issues <= 4:
        low_issues_bonus = (4 - total_issues) * 3.0  # up to +12 when 0 issues
        argument_strength = min(100.0, argument_strength + low_issues_bonus)

    argument_strength = max(0.0, min(100.0, argument_strength))

    return {
        "bias_score": round(bias_score, 2),
        "testability_score": round(testability_score, 2),
        "logic_score": round(logic_score, 2),
        "argument_strength": round(argument_strength, 2),
        "razor_alignment": round(razor_alignment, 2),
    }
