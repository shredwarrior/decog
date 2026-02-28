"""
Phase 3: Verify calibrated parameters with fresh arguments through the full pipeline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from analyzer import ArgumentAnalyzer

VERIFY_ARGS = [
    {
        "label": "Terrible: 5G conspiracy",
        "expected": "~10",
        "text": (
            "5G towers are spreading radiation that causes cancer and COVID was "
            "created to cover it up. The government and telecom companies are working "
            "together to poison us. Every country that rolled out 5G saw more deaths. "
            "They don't want you to know the truth because they make billions from it. "
            "Wake up people, do your own research."
        ),
    },
    {
        "label": "Weak: hot take on crypto",
        "expected": "~25",
        "text": (
            "Bitcoin is the future of money and anyone who doesn't invest is going to "
            "be left behind. Fiat currency always fails eventually and the dollar is next. "
            "Just look at how much early investors made. Banks hate crypto because it "
            "threatens their power. Countries like El Salvador have already adopted it, "
            "proving it works as real money."
        ),
    },
    {
        "label": "Mediocre: screen time argument",
        "expected": "~50",
        "text": (
            "Excessive screen time is linked to worse mental health outcomes in teenagers. "
            "A 2019 study in JAMA Pediatrics found associations between high screen use and "
            "depression. However, the evidence is correlational and effect sizes are small. "
            "Some screen activities like educational content may be beneficial. The real issue "
            "is probably displacement of sleep and physical activity rather than screens per se."
        ),
    },
    {
        "label": "Good: Reddit-quality exercise argument",
        "expected": "~75",
        "text": (
            "Resistance training has benefits beyond muscle growth that are often underappreciated. "
            "A 2018 meta-analysis in Sports Medicine found that resistance training reduces symptoms "
            "of anxiety with an effect size comparable to aerobic exercise. Bone density improvements "
            "from weight-bearing exercise are well-documented in osteoporosis research. Metabolically, "
            "muscle tissue increases resting energy expenditure. While the magnitude of these effects "
            "varies by individual, the consistency of findings across different populations suggests "
            "resistance training should be a standard public health recommendation alongside cardio."
        ),
    },
    {
        "label": "Excellent: scientific argument on vaccines",
        "expected": "~95",
        "text": (
            "The mRNA COVID-19 vaccines developed by Pfizer-BioNTech and Moderna demonstrated "
            "94-95% efficacy against symptomatic infection in Phase III randomized controlled trials "
            "involving over 70,000 participants combined. The mechanism is well-characterized: "
            "synthetic mRNA instructs cells to produce the SARS-CoV-2 spike protein, triggering "
            "an adaptive immune response without exposure to the live virus. Safety monitoring through "
            "VAERS, V-safe, and international pharmacovigilance systems covering billions of doses "
            "has confirmed that serious adverse events are rare (myocarditis occurring at approximately "
            "5 per million doses in young males). The claim that vaccines caused the observed decline "
            "in severe COVID outcomes is testable and has been confirmed through time-series analysis "
            "correlating vaccination rollout dates with hospitalization drops across 180+ countries "
            "with different healthcare systems and confound structures."
        ),
    },
]

def main():
    analyzer = ArgumentAnalyzer()
    print("=" * 70)
    print("  PHASE 3: VERIFICATION (full pipeline)")
    print("=" * 70)

    for i, arg in enumerate(VERIFY_ARGS):
        print(f"\n[{i+1}/{len(VERIFY_ARGS)}] {arg['label']} (expected {arg['expected']})", flush=True)
        result = analyzer.analyze_argument(arg["text"])
        if not result.get("success"):
            print(f"  ERROR: {result.get('error')}", flush=True)
            continue

        score = result["score"]
        bd = result["score_breakdown"]
        di = result["detected_issues"]
        n_f = len(di.get("logical_fallacies", []))
        n_b = len(di.get("cognitive_biases", []))
        n_d = len(di.get("cognitive_distortions", []))
        n_r = sum(1 for r in di.get("philosophical_razors", []) if r.get("pass"))
        n_total = n_f + n_b + n_d

        print(f"  SCORE: {score}  (structural: {bd.get('raw_score', '?'):.0f}, "
              f"testability: {bd.get('razor_alignment', '?'):.0f}%)", flush=True)
        print(f"  Issues: {n_total} total ({n_f}F {n_b}B {n_d}D), Razors passed: {n_r}/6", flush=True)
        print(f"  Status: {bd.get('status_label', '')}", flush=True)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()
