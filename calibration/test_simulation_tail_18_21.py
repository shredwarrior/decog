import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyzer import ArgumentAnalyzer
from calibration.test_simulation_21 import build_cases

load_dotenv()


def main():
    analyzer = ArgumentAnalyzer()
    cases = build_cases()[17:21]  # IDs 18..21

    print("ID  Quality        Words  Score   Bias   Test  Logic  Issues Status")
    print("-" * 120)

    for case in cases:
        result = analyzer.analyze_argument(case["text"])
        if not result.get("success"):
            err = (result.get("error", "unknown error") or "")[:45]
            print(f"{case['id']:<3} {case['quality']:<14} {case['target_words']:>5} {'ERR':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>7} {err}")
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
        status = (breakdown.get("status_label") or "")[:45]

        print(
            f"{case['id']:<3} {case['quality']:<14} {case['target_words']:>5} "
            f"{score:>6} {bias:>6.1f} {testability:>6.1f} {logic:>6.1f} {issue_count:>7} {status}"
        )


if __name__ == "__main__":
    main()

