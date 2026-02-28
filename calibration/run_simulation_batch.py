import argparse
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyzer import ArgumentAnalyzer
from calibration.test_simulation_21 import build_cases

load_dotenv()


def run_batch(start_id: int, end_id: int) -> None:
    analyzer = ArgumentAnalyzer()
    cases = [c for c in build_cases() if start_id <= c["id"] <= end_id]

    print("ID,quality,words,score,bias,testability,logic,issues,status")
    for case in cases:
        result = analyzer.analyze_argument(case["text"])
        if not result.get("success"):
            status = (result.get("error", "unknown error") or "").replace(",", ";")
            print(f"{case['id']},{case['quality']},{case['target_words']},ERR,-,-,-,-,{status}")
            continue

        breakdown = result.get("score_breakdown", {})
        dims = breakdown.get("dimension_scores", {})
        detected = result.get("detected_issues", {})
        issue_count = (
            len(detected.get("logical_fallacies", []))
            + len(detected.get("cognitive_biases", []))
            + len(detected.get("cognitive_distortions", []))
        )

        score = int(result.get("score", 0))
        bias = float(dims.get("bias_score", 0))
        testability = float(dims.get("testability_score", breakdown.get("razor_alignment", 0)))
        logic = float(dims.get("logic_score", breakdown.get("logic_integrity_score", 0)))
        status = (breakdown.get("status_label") or "").replace(",", ";")

        print(
            f"{case['id']},{case['quality']},{case['target_words']},"
            f"{score},{bias:.1f},{testability:.1f},{logic:.1f},{issue_count},{status}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a subset of the 21-case simulation by ID range.")
    parser.add_argument("--start", type=int, required=True, help="Start ID (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End ID (inclusive)")
    args = parser.parse_args()
    run_batch(args.start, args.end)


if __name__ == "__main__":
    main()
