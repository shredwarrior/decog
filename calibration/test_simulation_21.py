"""
Simulation matrix for 21 arguments:
- 6 very weak (2x 5-word, 2x 50-word, 2x 200-word)
- 6 medium quality (2x 5-word, 2x 50-word, 2x 200-word)
- 6 scientifically rich (2x 5-word, 2x 50-word, 2x 200-word)
- 3 fictional (low/medium/high mapped to 5/50/200 words)

Outputs a table for calibration sanity:
  overall, bias, testability, logic, status, issue counts.
"""

import os
import sys
from typing import Dict, List

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyzer import ArgumentAnalyzer

load_dotenv()


def fit_word_count(text: str, target_words: int) -> str:
    """
    Force exact word count for controlled length bins.
    Trims if long; pads with neutral tokens if short.
    """
    words = text.split()
    if len(words) > target_words:
        return " ".join(words[:target_words])
    if len(words) < target_words:
        padding = ["filler"] * (target_words - len(words))
        return " ".join(words + padding)
    return text


def build_cases() -> List[Dict]:
    weak_5_a = "This app makes no sense."
    weak_5_b = "Everything is terrible, always, everywhere."

    weak_50_a = (
        "Everyone knows this policy fails because people are selfish and leaders are useless. "
        "There is no need for evidence since daily life already proves it. "
        "If someone disagrees, they are naive and avoiding reality."
    )
    weak_50_b = (
        "This product is obviously a scam because it feels wrong and the company sounds fake. "
        "No data is needed when common sense is clear. "
        "Anybody defending it must be paid to lie."
    )

    weak_200_a = (
        "The education system is broken beyond repair and every teacher is part of the problem. "
        "Students fail because schools never care, and schools never care because teachers only want salaries. "
        "Whenever a student succeeds, it is despite teachers, not because of them. "
        "This proves reform is pointless. "
        "People keep claiming there are good programs, but those examples are propaganda from administrators. "
        "If policy makers mention data, that is manipulation because numbers can always be changed. "
        "Therefore no study can be trusted. "
        "Anyone who asks for evidence is just trying to delay action and protect the corrupt system. "
        "The only rational path is to abolish schools immediately, because partial fixes always fail and failure always means total collapse."
    )
    weak_200_b = (
        "Modern medicine is mostly fraud because hospitals earn money from sick people. "
        "If profit exists, truth cannot exist, so every treatment recommendation is automatically suspicious. "
        "Doctors claim trials prove effectiveness, but trials are run by organizations that want power. "
        "That means positive results are fake by design. "
        "When patients improve, it is placebo; when patients worsen, doctors blame biology. "
        "Either way, medicine escapes accountability, which proves conspiracy. "
        "People who ask for peer review miss the point because peers belong to the same system. "
        "Since all institutions are connected, no independent evidence is possible. "
        "Therefore rejecting medical advice is the only logical choice, and any exception simply confirms how deeply the deception operates."
    )

    medium_5_a = "Traffic is worse near schools."
    medium_5_b = "Coffee helps me focus sometimes."

    medium_50_a = (
        "Remote work can improve productivity for some teams because fewer office interruptions allow longer focus blocks. "
        "However, this benefit depends on task type and communication practices. "
        "Organizations should test hybrid schedules and compare delivery outcomes before committing."
    )
    medium_50_b = (
        "Public transit investment may reduce congestion in dense areas when routes are frequent and reliable. "
        "Still, effects differ by city layout and commuting habits. "
        "Local pilot programs with measured travel-time changes are better than blanket claims."
    )

    medium_200_a = (
        "University tuition relief likely increases access, but impact depends on targeting and institutional behavior. "
        "If aid is broad but supply is constrained, prices may rise and absorb part of the benefit. "
        "If aid prioritizes lower-income students while expanding seat capacity, access gains should be larger. "
        "Administrative design therefore matters as much as funding level. "
        "A useful policy sequence is: define eligibility, estimate capacity bottlenecks, and track enrollment and completion outcomes over time. "
        "This does not guarantee success, yet it provides a falsifiable framework. "
        "If affordability improves without completion gains, policy is incomplete. "
        "If both improve, scaling becomes defensible. "
        "The key claim is conditional: tuition relief can work, but only with capacity and accountability mechanisms."
    )
    medium_200_b = (
        "Urban tree expansion can reduce heat stress, but benefits vary with placement and maintenance quality. "
        "If trees are concentrated in already shaded neighborhoods, equity impact is weak. "
        "If planting targets high-heat blocks with low canopy, cooling and health outcomes should improve more. "
        "Species choice also matters because drought resilience affects long-term canopy survival. "
        "A practical program should map heat exposure first, then prioritize corridors with vulnerable populations, and finally evaluate temperature and hospitalization trends seasonally. "
        "This argument does not assume trees solve all climate risks. "
        "It claims targeted canopy programs are a measurable adaptation tool when linked to data-driven siting and maintenance budgets."
    )

    rich_5_a = "RCTs usually outperform anecdotes here."
    rich_5_b = "Replication strengthens claims in science."

    rich_50_a = (
        "Meta-analyses generally provide stronger evidence than single studies because they aggregate uncertainty across samples. "
        "Still, quality depends on inclusion criteria and publication bias controls. "
        "Claims should cite effect sizes, confidence intervals, and heterogeneity metrics."
    )
    rich_50_b = (
        "Randomized controlled trials reduce confounding by balancing unknown factors across groups. "
        "However, external validity can remain limited. "
        "Good inference combines trial results with observational follow-up and transparent sensitivity analysis."
    )

    rich_200_a = (
        "The link between smoking and lung cancer is supported by convergent evidence rather than one study type. "
        "Large cohort analyses repeatedly show higher incidence among smokers with dose-response patterns. "
        "Temporal ordering is clear, biological mechanisms are plausible through carcinogen exposure, and risk declines after cessation. "
        "These signals persist across populations and methods, which reduces the chance of a single-study artifact. "
        "Confounders exist, but sensitivity analyses still retain substantial effect estimates. "
        "A strong scientific claim here is not absolute certainty; it is high-confidence causal inference from consistent epidemiological, mechanistic, and longitudinal evidence."
    )
    rich_200_b = (
        "Anthropogenic warming is inferred from multiple independent measurements and physical models. "
        "Surface temperature records from separate institutions align on long-term warming trends, while ocean heat content data provide additional confirmation. "
        "Greenhouse gas concentration increases match isotopic fingerprints expected from fossil fuel combustion. "
        "Radiative forcing models predict warming patterns that broadly correspond to observed spatial and temporal distributions. "
        "Uncertainty remains in regional effects and feedback magnitudes, but core attribution is robust across model families and observational datasets. "
        "A scientifically rich argument here states confidence bounds, identifies unresolved components, and distinguishes established mechanisms from open parameters."
    )

    fiction_low_5 = "The kingdom falls because vibes."
    fiction_medium_50 = (
        "If the crystal engine drains mana faster than it recharges, then long voyages will stall. "
        "Captains should schedule recharge ports every third jump. "
        "If ships still stall, the assumption is wrong and route rules must change."
    )
    fiction_high_200 = (
        "Assume the city shield fails only when three runes desynchronize within one cycle. "
        "If that condition is true, then random sabotage attempts are unlikely to succeed without coordinated timing. "
        "If timing windows are predictable from moon phase drift, defense should concentrate on those windows rather than maintain maximal alert continuously. "
        "This creates a falsifiable plan: monitor rune phase variance nightly, compare variance spikes with breach attempts, and test whether targeted guard reinforcement lowers successful incursions. "
        "If breaches persist outside predicted windows, the core assumption is false and the model must be revised. "
        "If breaches cluster within predicted windows and decline after reinforcement, the model gains support. "
        "The argument remains conditional but structurally coherent because assumptions, predictions, and disconfirmation criteria are explicit."
    )

    rows = [
        ("weak", 5, weak_5_a), ("weak", 5, weak_5_b),
        ("weak", 50, weak_50_a), ("weak", 50, weak_50_b),
        ("weak", 200, weak_200_a), ("weak", 200, weak_200_b),
        ("medium", 5, medium_5_a), ("medium", 5, medium_5_b),
        ("medium", 50, medium_50_a), ("medium", 50, medium_50_b),
        ("medium", 200, medium_200_a), ("medium", 200, medium_200_b),
        ("scientific", 5, rich_5_a), ("scientific", 5, rich_5_b),
        ("scientific", 50, rich_50_a), ("scientific", 50, rich_50_b),
        ("scientific", 200, rich_200_a), ("scientific", 200, rich_200_b),
        ("fiction_low", 5, fiction_low_5),
        ("fiction_medium", 50, fiction_medium_50),
        ("fiction_high", 200, fiction_high_200),
    ]

    cases = []
    for idx, (quality, target_wc, text) in enumerate(rows, start=1):
        adjusted = fit_word_count(text, target_wc)
        cases.append({
            "id": idx,
            "quality": quality,
            "target_words": target_wc,
            "text": adjusted,
        })
    return cases


def run():
    analyzer = ArgumentAnalyzer()
    cases = build_cases()

    print("=" * 170)
    print("SIMULATION MATRIX: 21 ARGUMENTS")
    print("=" * 170)
    header = (
        f"{'ID':<3} {'Quality':<14} {'Words':>5} {'Score':>6} "
        f"{'Bias':>6} {'Test':>6} {'Logic':>6} {'Issues':>7} {'Status':<65}"
    )
    print(header)
    print("-" * 170)

    for case in cases:
        result = analyzer.analyze_argument(case["text"])
        if not result.get("success"):
            print(
                f"{case['id']:<3} {case['quality']:<14} {case['target_words']:>5} "
                f"{'ERR':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>7} "
                f"{(result.get('error', 'unknown error')[:64]):<65}"
            )
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
        status = (breakdown.get("status_label") or "")[:64]

        print(
            f"{case['id']:<3} {case['quality']:<14} {case['target_words']:>5} "
            f"{score:>6} {bias:>6.1f} {testability:>6.1f} {logic:>6.1f} {issue_count:>7} {status:<65}"
        )

    print("-" * 170)
    print("Done.")


if __name__ == "__main__":
    run()

