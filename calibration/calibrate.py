"""
Scoring Calibration Pipeline
=============================
Phase 1: Run test arguments through LLM Calls 1+2, cache raw detections.
Phase 2: Sweep parameter grid against cached results, find optimal set.
Phase 3: (manual) Verify with fresh full-pipeline runs.

Usage:
    python calibration/calibrate.py collect   # Phase 1 — hits OpenAI
    python calibration/calibrate.py sweep     # Phase 2 — pure Python
    python calibration/calibrate.py both      # Phase 1 then Phase 2
"""

import itertools
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from calibration.test_arguments import ARGUMENTS

CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache.json")
DEFINITIONS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "definitions.json")


def load_definitions():
    with open(DEFINITIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


DEFINITIONS = load_definitions()


# ──────────────────────────────────────────────────────────────────
#  PHASE 1: Collect LLM detections (Calls 1+2, skip Call 3)
# ──────────────────────────────────────────────────────────────────

def _build_metadata_prompt(argument_text):
    return f"""Break this argument into its individual claims.
For EACH claim return a JSON object with ALL of these fields:

=== STRUCTURAL TAGS (boolean / enum) ===
- claim_text              : string
- cites_evidence          : bool
- evidence_type           : "scientific" | "anecdotal" | "none"
- cites_authority         : bool
- authority_named         : string (empty if none)
- emotional_tone          : "neutral" | "positive" | "negative" | "fear" | "anger"
- makes_causal_claim      : bool
- generalizes             : bool
- uses_absolute_language  : bool
- targets_person          : bool
- assumes_intent          : bool
- is_falsifiable          : bool
- is_extraordinary        : bool

=== EPISTEMIC QUALITY (scale / enum) ===
- face_validity           : "high" | "medium" | "low"
- speculation_level       : "none" | "low" | "moderate" | "high"
- claim_type              : "factual" | "historical" | "opinion" | "prediction" | "definition" | "moral"
- evidence_sufficiency    : "strong" | "moderate" | "weak" | "none"
- causal_chain_length     : integer (0-5)
- inferential_gap         : "none" | "small" | "large"
- specificity             : "high" | "medium" | "low"
- verifiability           : "easily" | "with_effort" | "not_verifiable"

Return VALID JSON ONLY — an array of objects. No markdown fences.

Argument:
\"\"\"{argument_text}\"\"\""""


def _metadata_to_context(metadata):
    """Replicate analyzer._metadata_to_context locally."""
    if not metadata:
        return ""
    total = len(metadata)
    evidence_claims = [c for c in metadata if c.get("cites_evidence")]
    causal_claims = [c for c in metadata if c.get("makes_causal_claim")]
    emotional = [c for c in metadata if c.get("emotional_tone") not in ("neutral", None)]
    absolute_lang = [c for c in metadata if c.get("uses_absolute_language")]
    person_attacks = [c for c in metadata if c.get("targets_person")]
    unfalsifiable = [c for c in metadata if not c.get("is_falsifiable")]
    extraordinary = [c for c in metadata if c.get("is_extraordinary")]
    generalizing = [c for c in metadata if c.get("generalizes")]
    intent_assumed = [c for c in metadata if c.get("assumes_intent")]
    def _count(field, value):
        return sum(1 for c in metadata if c.get(field) == value)
    high_validity = _count("face_validity", "high")
    low_validity = _count("face_validity", "low")
    high_spec = _count("speculation_level", "high") + _count("speculation_level", "moderate")
    no_spec = _count("speculation_level", "none")
    strong_ev = _count("evidence_sufficiency", "strong")
    weak_ev = _count("evidence_sufficiency", "weak") + _count("evidence_sufficiency", "none")
    large_gap = _count("inferential_gap", "large")
    no_gap = _count("inferential_gap", "none")
    high_specific = _count("specificity", "high")
    low_specific = _count("specificity", "low")
    easily_verified = _count("verifiability", "easily")
    not_verifiable = _count("verifiability", "not_verifiable")
    type_counts = {}
    for c in metadata:
        ct = c.get("claim_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    avg_chain = 0.0
    chain_vals = [c.get("causal_chain_length", 0) for c in metadata if isinstance(c.get("causal_chain_length"), (int, float))]
    if chain_vals:
        avg_chain = sum(chain_vals) / len(chain_vals)
    lines = [
        f"The argument contains {total} individual claim(s).",
        "", "STRUCTURAL PROFILE:",
        f"  - {len(evidence_claims)} cite evidence ({_count('evidence_type', 'scientific')} scientific, {_count('evidence_type', 'anecdotal')} anecdotal).",
        f"  - {len(causal_claims)} make causal claims.",
        f"  - {len(emotional)} have non-neutral emotional tone.",
        f"  - {len(absolute_lang)} use absolute language.",
        f"  - {len(person_attacks)} target a person.",
        f"  - {len(unfalsifiable)} appear unfalsifiable.",
        f"  - {len(extraordinary)} make extraordinary claims.",
        f"  - {len(generalizing)} generalize from specific cases.",
        f"  - {len(intent_assumed)} assume intent or motive.",
        "", "EPISTEMIC PROFILE:",
        f"  - Face validity:        {high_validity} high, {_count('face_validity', 'medium')} medium, {low_validity} low.",
        f"  - Speculation:           {no_spec} grounded, {high_spec} moderate-to-high.",
        f"  - Evidence sufficiency:  {strong_ev} strong, {_count('evidence_sufficiency', 'moderate')} moderate, {weak_ev} weak/none.",
        f"  - Avg causal chain:      {avg_chain:.1f} steps.",
        f"  - Inferential gaps:      {no_gap} none, {_count('inferential_gap', 'small')} small, {large_gap} large.",
        f"  - Specificity:           {high_specific} high, {low_specific} low.",
        f"  - Verifiability:         {easily_verified} easily verified, {not_verifiable} not verifiable.",
        f"  - Claim types:           {', '.join(f'{v} {k}' for k, v in sorted(type_counts.items(), key=lambda x: -x[1]))}.",
    ]
    lines.append(""); lines.append("FLAGS:")
    if person_attacks:
        lines.append("  ! Person-targeting detected.")
    if absolute_lang:
        lines.append("  ! Absolute language detected.")
    if extraordinary and not evidence_claims:
        lines.append("  ! Extraordinary claims without evidence.")
    if unfalsifiable:
        lines.append("  ! Unfalsifiable claims found.")
    if high_spec >= total * 0.5:
        lines.append("  ! Majority of claims are speculative.")
    if large_gap >= 2:
        lines.append("  ! Multiple large inferential gaps.")
    if weak_ev >= total * 0.5 and causal_claims:
        lines.append("  ! Causal claims with weak evidence.")
    if low_validity >= 2:
        lines.append("  ! Multiple low face-validity claims.")
    if not_verifiable >= total * 0.5:
        lines.append("  ! Most claims are not independently verifiable.")
    if avg_chain < 0.5 and total >= 3:
        lines.append("  ! Claims are mostly bare assertions.")
    if no_spec == total and strong_ev >= total * 0.6:
        lines.append("  + Mostly grounded claims with strong evidence.")
    if easily_verified >= total * 0.6:
        lines.append("  + Most claims are easily verifiable.")
    return "\n".join(lines)


def _build_analysis_prompt(argument_text, metadata):
    fallacies = list(DEFINITIONS.get('logical_fallacies', {}).keys())
    biases = list(DEFINITIONS.get('cognitive_biases', {}).keys())
    distortions = list(DEFINITIONS.get('cognitive_distortions', {}).keys())
    razors = list(DEFINITIONS.get('philosophical_razors', {}).keys())

    metadata_block = ""
    if metadata:
        summary = _metadata_to_context(metadata)
        metadata_block = f"""
--- PRE-ANALYSIS (structural metadata) ---
{summary}
--- END PRE-ANALYSIS ---

Use the structural metadata above to guide your analysis.
Where a flag is raised (!), pay close attention to the related category.

"""
    return f"""Analyze the following argument for:
- logical fallacies (detect ALL that are genuinely present)
- cognitive biases (detect ALL that are genuinely present)
- cognitive distortions (detect ALL that are genuinely present)
- philosophical razors (evaluate ALL razors listed, pass/fail each)

Argument: "{argument_text}"

{metadata_block}IMPORTANT RULES:
- Return VALID JSON ONLY (no markdown, no extra text).
- For fallacies, biases, and distortions: return EVERY item you detect with confidence >= 0.6.
- Only use keys from the available lists below.
- Be conservative: only label a fallacy/bias/distortion if it is clearly present.
- Include a confidence score (0.0 to 1.0) for each item.
- For philosophical_razors, you MUST return ONE object for EVERY razor listed below, each with:
  "key", "pass" (boolean), "reason" (non-empty string), "confidence" (0.0-1.0).

Return this JSON schema exactly:
{{
  "logical_fallacies": [{{"key":"...", "reason":"...", "confidence": 0.0}}],
  "cognitive_biases": [{{"key":"...", "reason":"...", "confidence": 0.0}}],
  "cognitive_distortions": [{{"key":"...", "reason":"...", "confidence": 0.0}}],
  "philosophical_razors": [
    {{"key":"...", "pass": true, "reason":"...", "confidence": 0.0}},
    {{"key":"...", "pass": false, "reason":"...", "confidence": 0.0}}
  ],
  "executive_summary_sentence": "...",
  "executive_summary_bullets": ["...", "..."],
  "summary": "..."
}}

Available logical fallacies: {', '.join(fallacies)}
Available cognitive biases: {', '.join(biases)}
Available cognitive distortions: {', '.join(distortions)}
Available philosophical razors: {', '.join(razors)}
"""


def collect_one(client, arg):
    """Run Calls 1+2 for a single argument. Return raw parsed detection dict."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Call 1: metadata
    meta_prompt = _build_metadata_prompt(arg["text"])
    try:
        r1 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a text analysis assistant. Extract structural and epistemic features from text."},
                {"role": "user", "content": meta_prompt}
            ],
            temperature=0.1
        )
        raw1 = r1.choices[0].message.content
        m = re.search(r'\[.*\]', raw1, re.DOTALL)
        metadata = json.loads(m.group()) if m else []
    except Exception as e:
        print(f"  [!] Metadata failed for {arg['id']}: {e}")
        metadata = []

    # Call 2: main analysis
    analysis_prompt = _build_analysis_prompt(arg["text"], metadata)
    try:
        r2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in logic, cognitive psychology, and philosophy. Analyze arguments objectively."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3
        )
        raw2 = r2.choices[0].message.content
        m2 = re.search(r'\{.*\}', raw2, re.DOTALL)
        if m2:
            parsed = json.loads(m2.group())
        else:
            parsed = {}
    except Exception as e:
        print(f"  [!] Analysis failed for {arg['id']}: {e}")
        parsed = {}

    return {
        "logical_fallacies": parsed.get("logical_fallacies", []),
        "cognitive_biases": parsed.get("cognitive_biases", []),
        "cognitive_distortions": parsed.get("cognitive_distortions", []),
        "philosophical_razors": parsed.get("philosophical_razors", []),
    }


def run_collection():
    """Phase 1: collect all LLM detections and save to cache."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    cache = {}

    total = len(ARGUMENTS)
    for i, arg in enumerate(ARGUMENTS):
        label = f"[{i+1}/{total}] {arg['id']} (tier {arg['tier']}) — {arg['label']}"
        print(label, flush=True)
        t0 = time.time()
        raw = collect_one(client, arg)
        elapsed = time.time() - t0
        n_issues = (
            len(raw["logical_fallacies"])
            + len(raw["cognitive_biases"])
            + len(raw["cognitive_distortions"])
        )
        print(f"  {n_issues} issues detected, {elapsed:.1f}s", flush=True)
        cache[arg["id"]] = {
            "tier": arg["tier"],
            "target_lo": arg["target_lo"],
            "target_hi": arg["target_hi"],
            "label": arg["label"],
            "raw_detections": raw,
        }

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"\nCache saved to {CACHE_PATH} ({len(cache)} arguments)")


# ──────────────────────────────────────────────────────────────────
#  PHASE 2: Parameter sweep (pure Python, no LLM calls)
# ──────────────────────────────────────────────────────────────────

def normalize_and_score(raw_detections, params):
    """
    Replicate _normalize_detected_issues + _calculate_score
    with a given parameter set. Returns (score, raw_score, n_issues, n_razors_passed).
    """
    issue_conf = params["issue_confidence"]
    razor_conf = params["razor_confidence"]
    base = params["base_score"]
    fw = params["fallacy_weight"]
    bw = params["bias_weight"]
    dw = params["distortion_weight"]
    mrb = params["max_razor_bonus"]

    def _filter_issues(items, def_group):
        allowed = set(DEFINITIONS.get(def_group, {}).keys())
        out = []
        for item in (items if isinstance(items, list) else []):
            if isinstance(item, str):
                key, confidence = item, 1.0
            else:
                key = (item or {}).get("key", "")
                confidence = float((item or {}).get("confidence", 0.0) or 0.0)
            if key == "none" or key not in allowed:
                continue
            if confidence < issue_conf:
                continue
            out.append(key)
        return out

    fallacies = _filter_issues(raw_detections.get("logical_fallacies", []), "logical_fallacies")
    biases = _filter_issues(raw_detections.get("cognitive_biases", []), "cognitive_biases")
    distortions = _filter_issues(raw_detections.get("cognitive_distortions", []), "cognitive_distortions")

    total_penalty = 0.0
    for k in fallacies:
        total_penalty += fw * abs(DEFINITIONS["logical_fallacies"][k]["penalty"])
    for k in biases:
        total_penalty += bw * abs(DEFINITIONS["cognitive_biases"][k]["penalty"])
    for k in distortions:
        total_penalty += dw * abs(DEFINITIONS["cognitive_distortions"][k]["penalty"])

    total_reward = 0
    passed_razors = 0
    razor_defs = DEFINITIONS.get("philosophical_razors", {})
    incoming = raw_detections.get("philosophical_razors", [])
    incoming_by_key = {}
    if isinstance(incoming, list):
        for r in incoming:
            if isinstance(r, dict) and r.get("key"):
                incoming_by_key[r["key"]] = r

    for key in razor_defs:
        ir = incoming_by_key.get(key, {})
        conf = float(ir.get("confidence", 0.0) or 0.0)
        passed = bool(ir.get("pass", False)) and conf >= razor_conf
        if passed:
            total_reward += razor_defs[key]["reward"]
            passed_razors += 1

    total_reward = min(total_reward, mrb)
    raw_score = max(0, min(100, base - total_penalty))
    final_score = max(0, min(100, raw_score + total_reward))
    n_issues = len(fallacies) + len(biases) + len(distortions)

    return int(round(final_score)), raw_score, n_issues, passed_razors


PARAM_GRID = {
    "base_score":          [70, 75, 80, 85, 90],
    "fallacy_weight":      [0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
    "bias_weight":         [0.4, 0.6, 0.8, 1.0],
    "distortion_weight":   [0.4, 0.6, 0.8, 1.0],
    "issue_confidence":    [0.20, 0.25, 0.30, 0.35],
    "razor_confidence":    [0.30, 0.35, 0.40, 0.45],
    "max_razor_bonus":     [15, 20, 25, 30],
}


def run_sweep():
    """Phase 2: sweep parameter grid, score every cached argument, rank results."""
    if not os.path.exists(CACHE_PATH):
        print("ERROR: cache.json not found. Run 'collect' first.")
        sys.exit(1)

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    combos = list(itertools.product(*values))
    print(f"Sweeping {len(combos)} parameter combinations across {len(cache)} arguments...\n")

    best_results = []

    for combo in combos:
        params = dict(zip(keys, combo))
        total_error = 0.0
        hard_violations = 0
        tier_scores = {1: [], 2: [], 3: [], 4: [], 5: []}
        tier_issues = {1: [], 2: [], 3: [], 4: [], 5: []}

        for arg_id, data in cache.items():
            tier = data["tier"]
            target_mid = (data["target_lo"] + data["target_hi"]) / 2
            score, raw_s, n_issues, n_razors = normalize_and_score(
                data["raw_detections"], params
            )
            tier_scores[tier].append(score)
            tier_issues[tier].append(n_issues)

            error = abs(score - target_mid)
            total_error += error

            if tier == 1 and score > 20:
                hard_violations += 1
            if tier == 5 and score < 75:
                hard_violations += 1
            if tier in (1, 2) and n_issues < 4:
                hard_violations += 1

        fitness = total_error + hard_violations * 50
        avg_by_tier = {t: (sum(s)/len(s) if s else 0) for t, s in tier_scores.items()}
        avg_issues_by_tier = {t: (sum(s)/len(s) if s else 0) for t, s in tier_issues.items()}

        best_results.append({
            "params": params,
            "fitness": fitness,
            "total_error": total_error,
            "hard_violations": hard_violations,
            "avg_by_tier": avg_by_tier,
            "avg_issues_by_tier": avg_issues_by_tier,
            "tier_scores": tier_scores,
        })

    best_results.sort(key=lambda x: x["fitness"])

    print("=" * 80)
    print("  TOP 5 PARAMETER SETS")
    print("=" * 80)
    for rank, r in enumerate(best_results[:5], 1):
        p = r["params"]
        print(f"\n--- Rank #{rank}  (fitness={r['fitness']:.1f}, error={r['total_error']:.1f}, violations={r['hard_violations']}) ---")
        print(f"  base_score:        {p['base_score']}")
        print(f"  fallacy_weight:    {p['fallacy_weight']}")
        print(f"  bias_weight:       {p['bias_weight']}")
        print(f"  distortion_weight: {p['distortion_weight']}")
        print(f"  issue_confidence:  {p['issue_confidence']}")
        print(f"  razor_confidence:  {p['razor_confidence']}")
        print(f"  max_razor_bonus:   {p['max_razor_bonus']}")
        print()
        print(f"  {'Tier':<8} {'Target':<12} {'Avg Score':<12} {'Avg Issues':<12} {'Scores'}")
        targets = {1: "5-15", 2: "20-35", 3: "40-60", 4: "65-85", 5: "85-100"}
        for t in range(1, 6):
            scores_str = ", ".join(str(s) for s in r["tier_scores"][t])
            print(f"  {t:<8} {targets[t]:<12} {r['avg_by_tier'][t]:<12.1f} {r['avg_issues_by_tier'][t]:<12.1f} [{scores_str}]")

    winner = best_results[0]
    print("\n" + "=" * 80)
    print("  RECOMMENDED PARAMETERS (Rank #1)")
    print("=" * 80)
    for k, v in winner["params"].items():
        env_key = k.upper()
        print(f"  {env_key} = {v}")

    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        top5 = []
        for r in best_results[:5]:
            top5.append({
                "params": r["params"],
                "fitness": r["fitness"],
                "avg_by_tier": {str(k): v for k, v in r["avg_by_tier"].items()},
                "avg_issues_by_tier": {str(k): v for k, v in r["avg_issues_by_tier"].items()},
                "tier_scores": {str(k): v for k, v in r["tier_scores"].items()},
            })
        json.dump(top5, f, indent=2)
    print(f"\nFull results saved to {results_path}")


# ──────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibration/calibrate.py [collect|sweep|both]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "collect":
        run_collection()
    elif cmd == "sweep":
        run_sweep()
    elif cmd == "both":
        run_collection()
        print("\n" + "=" * 80 + "\n")
        run_sweep()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
