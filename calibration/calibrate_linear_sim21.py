"""
Constrained calibration for the current linear scoring model on the 21-case matrix.

Usage:
  python calibration/calibrate_linear_sim21.py collect
  python calibration/calibrate_linear_sim21.py sweep
  python calibration/calibrate_linear_sim21.py both
"""

import itertools
import json
import os
import random
import sys
from statistics import mean, pstdev
from typing import Dict, List

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyzer import ArgumentAnalyzer
from calibration.test_simulation_21 import build_cases

load_dotenv()

CACHE_PATH = os.path.join(os.path.dirname(__file__), "sim21_cache.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "sim21_sweep_results.json")


def _quality_group(quality: str) -> str:
    if quality in {"weak", "fiction_low"}:
        return "weak"
    if quality in {"medium", "fiction_medium"}:
        return "medium"
    return "high"


def _collect() -> None:
    analyzer = ArgumentAnalyzer()
    cases = build_cases()
    cache: List[Dict] = []

    print(f"Collecting detections for {len(cases)} simulation cases...")
    for case in cases:
        metadata = analyzer._extract_metadata(case["text"])
        prompt = analyzer._create_analysis_prompt(case["text"], metadata)
        response = analyzer.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in logic, cognitive psychology, and philosophy. Analyze arguments objectively.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        analysis_result = response.choices[0].message.content
        detected_issues = analyzer._parse_openai_response(analysis_result)
        detected_issues = analyzer._normalize_detected_issues(detected_issues)
        cache.append(
            {
                "id": case["id"],
                "quality": case["quality"],
                "group": _quality_group(case["quality"]),
                "target_words": case["target_words"],
                "text": case["text"],
                "metadata": metadata,
                "detected_issues": detected_issues,
            }
        )
        print(f"  collected: id={case['id']}, quality={case['quality']}, words={case['target_words']}")

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"\nCache saved: {CACHE_PATH}")


def _candidate_params() -> List[Dict[str, float]]:
    # Tight ranges to keep behavior interpretable and policy-constrained.
    grid = {
        "BASE_SCORE": [72.0, 75.0, 78.0],
        "FALLACY_WEIGHT": [0.95, 1.05, 1.15],
        "DISTORTION_WEIGHT": [0.55, 0.65, 0.75],
        "BIAS_WEIGHT": [0.35, 0.45, 0.55],
        "BIAS_COUNT_BOOST": [1.8, 2.2, 2.6],
        "BIAS_PRESSURE_SCALE": [60.0, 70.0, 80.0],
        "BIAS_PENALTY_GAIN": [0.30, 0.34, 0.38],
        "LOW_LOGIC_PENALTY_GAIN": [0.48, 0.52, 0.56],
        "LOW_TESTABILITY_PENALTY_GAIN": [0.20, 0.24, 0.28],
        "TESTABILITY_REWARD_GAIN": [0.08, 0.12, 0.16],
        "LOGIC_REWARD_GAIN": [0.34, 0.40, 0.46],
        "MAX_META_HINT_BONUS": [8.0, 10.0, 12.0],
    }

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    all_candidates = []
    for combo in itertools.product(*values):
        candidate = dict(zip(keys, combo))
        # Policy constraints:
        # 1) Fallacy > Distortion > Bias
        # 2) Testability boost stays moderate (below logic boost)
        if not (
            candidate["FALLACY_WEIGHT"] > candidate["DISTORTION_WEIGHT"] > candidate["BIAS_WEIGHT"]
            and candidate["TESTABILITY_REWARD_GAIN"] <= (candidate["LOGIC_REWARD_GAIN"] * 0.5)
        ):
            continue
        all_candidates.append(candidate)

    random.seed(42)
    random.shuffle(all_candidates)
    return all_candidates[:240]


def _score_candidate(analyzer: ArgumentAnalyzer, cache: List[Dict], params: Dict[str, float]) -> Dict:
    for key, value in params.items():
        os.environ[key] = str(value)

    scores_by_group = {"weak": [], "medium": [], "high": []}
    row_scores = []
    saturation_count = 0
    objective = 0.0

    for row in cache:
        artifacts = analyzer._compute_score_artifacts_linear(
            row["detected_issues"], row["text"], row["metadata"]
        )
        score = int(artifacts["final_score"])
        group = row["group"]
        scores_by_group[group].append(score)
        row_scores.append(
            {
                "id": row["id"],
                "quality": row["quality"],
                "group": group,
                "score": score,
                "status": artifacts.get("status_label", ""),
            }
        )

        if score <= 2 or score >= 98:
            saturation_count += 1
            objective += 25.0

        if group == "weak":
            objective += abs(score - 10.0) * 1.4
            if score > 20:
                objective += (score - 20.0) * 3.0
        elif group == "medium":
            if score < 30:
                objective += (30.0 - score) * 2.2
            elif score > 70:
                objective += (score - 70.0) * 2.2
            else:
                objective += abs(score - 50.0) * 0.6
        else:
            if score < 70:
                objective += (70.0 - score) * 2.6
            elif score > 90:
                objective += (score - 90.0) * 2.6
            else:
                objective += abs(score - 80.0) * 0.6

    weak_std = pstdev(scores_by_group["weak"]) if scores_by_group["weak"] else 0.0
    if weak_std > 7.0:
        objective += (weak_std - 7.0) * 3.0

    return {
        "params": params,
        "objective": round(objective, 2),
        "saturation_count": saturation_count,
        "avg_scores": {k: round(mean(v), 2) if v else 0.0 for k, v in scores_by_group.items()},
        "std_scores": {k: round(pstdev(v), 2) if v else 0.0 for k, v in scores_by_group.items()},
        "rows": row_scores,
    }


def _sweep() -> None:
    if not os.path.exists(CACHE_PATH):
        print(f"Missing cache file: {CACHE_PATH}. Run collect first.")
        sys.exit(1)

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    analyzer = ArgumentAnalyzer()
    candidates = _candidate_params()
    print(f"Evaluating {len(candidates)} constrained candidates...")

    results = []
    for idx, params in enumerate(candidates, start=1):
        outcome = _score_candidate(analyzer, cache, params)
        results.append(outcome)
        if idx % 20 == 0:
            print(f"  completed {idx}/{len(candidates)}")

    results.sort(key=lambda x: x["objective"])
    top = results[:5]

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(top, f, indent=2)

    print("\nTop candidates:")
    for i, row in enumerate(top, start=1):
        print(
            f"#{i} objective={row['objective']} saturation={row['saturation_count']} "
            f"avg={row['avg_scores']} std={row['std_scores']}"
        )

    winner = top[0]
    print("\nRecommended env values:")
    for k, v in winner["params"].items():
        print(f"{k}={v}")
    print(f"\nSaved: {RESULTS_PATH}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python calibration/calibrate_linear_sim21.py [collect|sweep|both]")
        sys.exit(1)
    cmd = sys.argv[1].lower()
    if cmd == "collect":
        _collect()
    elif cmd == "sweep":
        _sweep()
    elif cmd == "both":
        _collect()
        _sweep()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
