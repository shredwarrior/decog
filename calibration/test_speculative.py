"""
Test: Logically solid but speculative, hard-to-prove arguments.
These should score in the mediocre-to-good range (40-70) — reasonable
reasoning structure but lacking hard evidence.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from analyzer import ArgumentAnalyzer

TESTS = [
    {
        "label": "Corporate altruism is PR",
        "text": (
            "No corporation is genuinely altruistic. First, corporations are legally "
            "obligated to maximize shareholder value — any executive who prioritizes "
            "charity over profit can be sued by shareholders. Second, every major "
            "corporate philanthropy initiative is run through PR departments, not ethics "
            "boards, which means the primary metric is brand perception, not impact. "
            "Third, when companies like Amazon or Google donate to causes, they choose "
            "causes that align with future market expansion — Google funds digital "
            "literacy because it creates future users, not out of goodwill. Tax "
            "write-offs further ensure that donations cost less than their PR value. "
            "The structure of capitalism makes genuine corporate generosity irrational. "
            "What looks like altruism is always optimized self-interest with better optics."
        ),
    },
    {
        "label": "Social media algorithms radicalize by design",
        "text": (
            "Social media platforms are structurally designed to radicalize users, not "
            "by accident but by economic necessity. Engagement-maximizing algorithms "
            "surface content that provokes strong emotional reactions because outrage "
            "and fear generate more clicks than nuance. Internal documents from Facebook "
            "showed their own researchers found the recommendation engine pushed users "
            "toward increasingly extreme groups. The business model requires maximizing "
            "time-on-platform, and moderate content doesn't hold attention the way "
            "polarizing content does. Platforms could fix this by deprioritizing "
            "engagement metrics, but that would directly reduce ad revenue. No publicly "
            "traded company will voluntarily reduce revenue, so the radicalization is "
            "a permanent structural feature, not a bug that will get patched."
        ),
    },
    {
        "label": "Meritocracy is self-reinforcing mythology",
        "text": (
            "Meritocracy as practiced in modern economies is largely a myth that serves "
            "to justify existing inequality. Studies show that the single strongest "
            "predictor of adult income is parental income, not talent or effort. Children "
            "of wealthy families get better nutrition, better schools, tutoring, unpaid "
            "internship access, and professional networks — all before 'merit' is even "
            "measured. Standardized tests, the supposed neutral arbiter, correlate more "
            "strongly with zip code than with innate ability. Those who succeed within "
            "the system have a psychological incentive to attribute their success to "
            "merit rather than circumstance, reinforcing the myth. The few genuine "
            "rags-to-riches stories get amplified precisely because they're exceptional, "
            "creating survivorship bias that makes the system appear fairer than it is."
        ),
    },
    {
        "label": "Academic publishing is a broken incentive system",
        "text": (
            "The academic publishing system incentivizes quantity and novelty over "
            "truth, which systematically degrades scientific reliability. Tenure and "
            "funding decisions are based on publication count and journal impact factor, "
            "not on whether findings replicate. This creates pressure to produce "
            "positive results, which explains why the replication crisis found that "
            "over 60% of psychology studies and roughly 50% of cancer biology studies "
            "failed to replicate. Negative results — which are equally informative — "
            "are rarely published because journals prefer novel findings. Peer review "
            "is unpaid, rushed, and increasingly done by junior researchers who lack "
            "the standing to challenge established names. Meanwhile, publishers like "
            "Elsevier earn 35% profit margins by selling publicly funded research back "
            "to the institutions that funded it. The system rewards gaming, not rigor."
        ),
    },
    {
        "label": "Democracy selects for short-termism",
        "text": (
            "Democratic systems are structurally incapable of addressing long-term "
            "problems because election cycles create a bias toward short-term gains. "
            "Politicians who propose painful but necessary long-term reforms — like "
            "aggressive carbon taxes or pension restructuring — lose elections to those "
            "promising immediate benefits. Climate change is the clearest example: the "
            "science has been settled for decades, but democratic governments have "
            "consistently failed to act at the required scale because the costs are "
            "immediate and the benefits are generational. Singapore and China have "
            "implemented longer-horizon infrastructure and environmental policies "
            "precisely because they aren't subject to 4-year electoral pressure. This "
            "doesn't mean authoritarianism is better overall, but it reveals a specific "
            "structural weakness in democracy: the inability to impose short-term costs "
            "for long-term collective benefit when voters can simply replace leaders "
            "who try."
        ),
    },
]

def main():
    analyzer = ArgumentAnalyzer()
    print("=" * 90)
    print("  SPECULATIVE BUT LOGICALLY STRUCTURED ARGUMENTS")
    print("=" * 90)

    for i, t in enumerate(TESTS):
        wc = len(t["text"].split())
        print(f"\n[{i+1}/{len(TESTS)}] {t['label']} ({wc} words)", flush=True)
        result = analyzer.analyze_argument(t["text"])
        if not result.get("success"):
            print(f"  ERROR: {result.get('error', '?')}", flush=True)
            continue

        di = result["detected_issues"]
        bd = result["score_breakdown"]
        n_f = len(di.get("logical_fallacies", []))
        n_b = len(di.get("cognitive_biases", []))
        n_d = len(di.get("cognitive_distortions", []))
        n_r = sum(1 for r in di.get("philosophical_razors", []) if r.get("pass"))

        print(f"  SCORE: {result['score']}  (structural: {bd.get('raw_score', 0):.0f}, testability: {bd.get('razor_alignment', 0):.0f}%)", flush=True)
        print(f"  Issues: {n_f}F {n_b}B {n_d}D = {n_f+n_b+n_d} total  |  Razors: {n_r}/6 passed", flush=True)
        print(f"  Status: {bd.get('status_label', '')}", flush=True)

        issues_list = []
        for cat in ("logical_fallacies", "cognitive_biases", "cognitive_distortions"):
            for item in di.get(cat, []):
                issues_list.append(item.get("name", item.get("key", "?")))
        if issues_list:
            print(f"  Detected: {', '.join(issues_list)}", flush=True)

    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()
