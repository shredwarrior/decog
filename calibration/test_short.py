"""
Quick test: how does the substance penalty treat short-but-powerful arguments
vs short-but-vacuous ones?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from analyzer import ArgumentAnalyzer

TESTS = [
    # Vacuous / throwaway
    {"label": "Vacuous: 3 words", "text": "life is dumb"},
    {"label": "Vacuous: short opinion", "text": "all politicians are corrupt liars"},
    {"label": "Vacuous: lazy hot take", "text": "school is pointless and a waste of time for everyone"},

    # Short but powerful / precise
    {
        "label": "Flynn effect (cited, short)",
        "text": (
            "Studies have tracked IQ across countries for 50 years and seen a "
            "gradual 2% increase decade over decade. It's the Flynn effect."
        ),
    },
    {
        "label": "Formal logic (modus ponens)",
        "text": (
            "If all humans are mortal, and Socrates is a human, then Socrates "
            "is mortal. The premises are established; the conclusion follows necessarily."
        ),
    },
    {
        "label": "Precise empirical claim",
        "text": (
            "The speed of light in a vacuum is 299,792,458 meters per second. "
            "This has been measured independently by multiple labs to within "
            "1 meter per second precision."
        ),
    },
    {
        "label": "Concise causal argument",
        "text": (
            "Smoking causes lung cancer. The causal link was established through "
            "decades of cohort studies, dose-response relationships, and the "
            "mechanism of DNA damage by tobacco carcinogens."
        ),
    },
    {
        "label": "Mathematical proof statement",
        "text": (
            "There are infinitely many prime numbers. Assume finitely many; "
            "multiply them all and add one. The result is not divisible by any "
            "prime in the list, contradicting the assumption."
        ),
    },
]

def main():
    analyzer = ArgumentAnalyzer()
    print(f"{'Label':<40} {'Words':>5} {'Claims':>6} {'SubPen':>6} {'Issues':>6} {'Score':>5} {'Status'}")
    print("-" * 110)

    for t in TESTS:
        wc = len(t["text"].split())
        result = analyzer.analyze_argument(t["text"])
        if not result.get("success"):
            print(f"{t['label']:<40} {wc:>5}   ERROR: {result.get('error', '?')}")
            continue

        meta = result.get("metadata", [])
        di = result["detected_issues"]
        bd = result["score_breakdown"]
        n_issues = (len(di.get("logical_fallacies", []))
                    + len(di.get("cognitive_biases", []))
                    + len(di.get("cognitive_distortions", [])))
        sub_pen = bd.get("penalties", {}).get("_substance", 0)
        score = result["score"]
        status = bd.get("status_label", "")

        print(f"{t['label']:<40} {wc:>5} {len(meta):>6} {sub_pen:>6} {n_issues:>6} {score:>5}  {status}", flush=True)

if __name__ == "__main__":
    main()
