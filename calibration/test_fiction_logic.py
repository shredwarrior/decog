"""
Pilot test for fiction-focused logical discrimination.

Goal:
- Use the new logic variables as the primary discriminator.
- Check whether fictionally good logic paragraphs outrank fictionally bad ones.
- Apply the balanced gate:
  - Pairwise ranking accuracy >= 75%
  - Median score gap (good - bad) >= 10 points
"""

import os
import statistics
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyzer import ArgumentAnalyzer

load_dotenv()

GOOD_FICTION = [
    {
        "label": "Warded gate protocol",
        "text": (
            "Assume the citadel wards react only to bloodline signatures and not to intent. "
            "If that assumption holds, then a shapeshifter wearing the duke's face still cannot "
            "open the inner gate unless they also mimic the bloodline resonance. If we also accept "
            "that resonance can be copied only with a moonwell vial prepared within one hour, then "
            "the assassin's plan requires three conditions at once: access to the duke, access to "
            "a moonwell, and entry before the resonance decays. Therefore, the defense should focus "
            "on removing any one condition rather than guarding every corridor. This does not prove "
            "safety forever, but it identifies the narrowest failure path and makes the attack testable "
            "through controlled ward trials."
        ),
    },
    {
        "label": "Starship reactor triage",
        "text": (
            "Given that the antimatter regulator fails only when coolant delay exceeds nine seconds, "
            "we can reason about evacuation timing directly. If drones can restore coolant in under "
            "eight seconds, then the reactor remains stable and the crew should hold position. If delay "
            "is likely above nine seconds, then containment breach risk rises sharply and evacuation is "
            "the rational choice. Therefore the command decision should depend on one measurable variable: "
            "verified delay, not panic or rank. This argument does not claim certainty about every fault "
            "mode, but it sets a falsifiable threshold that engineering logs can confirm or reject."
        ),
    },
    {
        "label": "Treaty logic in rival kingdoms",
        "text": (
            "Suppose both kingdoms value grain access more than border prestige this winter. If so, then "
            "each side has incentive to avoid raids that destroy shared irrigation. If each side also knows "
            "the other can mobilize in three days, then surprise attacks produce only short gains and long "
            "supply losses. Therefore a narrow non-aggression treaty tied to irrigation inspection is more "
            "stable than a broad peace pledge. The conclusion is conditional, not absolute: if grain ceases "
            "to be scarce, the treaty's logic weakens and must be re-evaluated."
        ),
    },
    {
        "label": "Mage guild apprenticeship policy",
        "text": (
            "Assume firecasting accidents are primarily caused by unstable breathing rhythm in novices. "
            "If apprentices train breath control before ignition drills, then accidental flare rates should "
            "drop. If flare rates do not drop after two cycles, the assumption is wrong and curriculum must "
            "change. Therefore the guild should stage training in sequence and publish accident logs by cohort. "
            "This reasoning is modest but coherent because each step depends on an explicit assumption and an "
            "observable test."
        ),
    },
    {
        "label": "Artifact custody argument",
        "text": (
            "Given that the artifact amplifies nearby fear states, any council vote held in its chamber is biased. "
            "If fear amplification is real, then members nearest the artifact should show higher error rates in risk "
            "estimates. If that pattern appears, then policy decisions made in the chamber should be reviewed outside it. "
            "Therefore custody policy should separate storage from deliberation rooms. This is not an attack on council "
            "integrity; it is a structural argument about decision conditions and testable outcomes."
        ),
    },
    {
        "label": "Portal logistics planning",
        "text": (
            "Assume the portal remains stable for exactly fourteen minutes per cycle. If crossing times exceed that "
            "window, then rear units become stranded and supply lines collapse. If crossing times stay below ten "
            "minutes with two-minute buffers, then unit cohesion is preserved despite weather variance. Therefore "
            "deployment should prioritize lightweight scouts first and siege engines last. The argument may be wrong "
            "if portal variance is larger than measured, but the operational logic is still transparent and falsifiable."
        ),
    },
]

BAD_FICTION = [
    {
        "label": "Prophecy certainty leap",
        "text": (
            "The oracle once predicted a storm, so every future prophecy is true forever. Anyone who doubts this is "
            "blind to destiny. The chosen heir always wins and never loses, which proves caution is unnecessary. "
            "Because the moon turned red last night, the northern armies cannot possibly attack at dawn, and therefore "
            "we should disband the watch immediately."
        ),
    },
    {
        "label": "Magic absolutism rant",
        "text": (
            "All mages are liars and none of their warnings matter. Magic always corrupts and never protects anyone. "
            "If a mage says the barrier is failing, that only proves they caused it. So the city must ban every spell "
            "today and trust instinct over evidence."
        ),
    },
    {
        "label": "Conspiracy in the fleet",
        "text": (
            "The flagship lights flickered once, so the admiral must be sabotaging the fleet. Massive disasters must "
            "have massive villains, and tiny wiring faults cannot cause major failures. Therefore every engineer who "
            "disagrees is part of the cover-up."
        ),
    },
    {
        "label": "Circular throne claim",
        "text": (
            "I am the rightful ruler because I deserve the throne, and I deserve the throne because I am the rightful "
            "ruler. Since this is obvious, all objections are treason. The law is valid because I say it is, and my "
            "authority is valid because the law says so."
        ),
    },
    {
        "label": "If-then mismatch",
        "text": (
            "If the harvest is good, then taxes should rise, and if the harvest is bad, then taxes should also rise. "
            "If merchants protest, that proves taxes are fair, and if they do not protest, that also proves taxes are fair. "
            "Therefore any possible outcome confirms the same policy."
        ),
    },
    {
        "label": "Polished flat-world rhetoric",
        "text": (
            "If the world were curved, then water would visibly peel away from ship decks, but sailors still stand upright. "
            "Therefore the world is flat and every space agency must coordinate a deception. Their budgets are huge, and huge "
            "events always require huge lies; a simple misunderstanding could never explain this scale. Since authorities disagree "
            "with independent observers, the official model is certainly false."
        ),
    },
]


def _score_text(analyzer, text):
    result = analyzer.analyze_argument(text)
    if not result.get("success"):
        return {"success": False, "error": result.get("error", "unknown error")}

    bd = result.get("score_breakdown", {})
    return {
        "success": True,
        "overall_score": result.get("score", 0),
        "logic_integrity_score": bd.get("logic_integrity_score", 0),
        "evidence_dependency_score": bd.get("evidence_dependency_score", 0),
        "mode_detected": bd.get("mode_detected", "unknown"),
        "status_label": bd.get("status_label", ""),
        "logic_variables": bd.get("logic_variables", {}),
    }


def _pairwise_accuracy(good_scores, bad_scores):
    wins = 0
    total = 0
    for g in good_scores:
        for b in bad_scores:
            total += 1
            if g > b:
                wins += 1
    return (wins / total) * 100.0 if total else 0.0


def main():
    analyzer = ArgumentAnalyzer()

    print("=" * 90)
    print("FICTION LOGIC PILOT (VARIABLES-FIRST)")
    print("=" * 90)

    good_results = []
    bad_results = []

    print("\n-- GOOD FICTION LOGIC --")
    for item in GOOD_FICTION:
        scored = _score_text(analyzer, item["text"])
        if not scored["success"]:
            print(f"ERROR {item['label']}: {scored['error']}")
            continue
        good_results.append((item["label"], scored))
        print(
            f"{item['label']}: logic={scored['logic_integrity_score']:.1f}, "
            f"overall={scored['overall_score']}, mode={scored['mode_detected']}, "
            f"status={scored['status_label']}"
        )

    print("\n-- BAD FICTION LOGIC --")
    for item in BAD_FICTION:
        scored = _score_text(analyzer, item["text"])
        if not scored["success"]:
            print(f"ERROR {item['label']}: {scored['error']}")
            continue
        bad_results.append((item["label"], scored))
        print(
            f"{item['label']}: logic={scored['logic_integrity_score']:.1f}, "
            f"overall={scored['overall_score']}, mode={scored['mode_detected']}, "
            f"status={scored['status_label']}"
        )

    good_logic = [x[1]["logic_integrity_score"] for x in good_results]
    bad_logic = [x[1]["logic_integrity_score"] for x in bad_results]
    good_overall = [x[1]["overall_score"] for x in good_results]
    bad_overall = [x[1]["overall_score"] for x in bad_results]

    if not good_logic or not bad_logic:
        print("\nPilot failed: insufficient successful runs.")
        return

    pairwise_logic = _pairwise_accuracy(good_logic, bad_logic)
    pairwise_overall = _pairwise_accuracy(good_overall, bad_overall)
    median_gap_logic = statistics.median(good_logic) - statistics.median(bad_logic)
    median_gap_overall = statistics.median(good_overall) - statistics.median(bad_overall)

    print("\n" + "=" * 90)
    print("METRICS")
    print("=" * 90)
    print(f"Pairwise accuracy (logic variable score): {pairwise_logic:.2f}%")
    print(f"Pairwise accuracy (overall score):        {pairwise_overall:.2f}%")
    print(f"Median gap (logic score good-bad):       {median_gap_logic:.2f}")
    print(f"Median gap (overall score good-bad):     {median_gap_overall:.2f}")

    balanced_gate_pass = pairwise_logic >= 75.0 and median_gap_logic >= 10.0
    print("\nBALANCED GATE:")
    print("PASS" if balanced_gate_pass else "FAIL")
    print(
        "Criteria: pairwise_logic >= 75% and median_gap_logic >= 10"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
