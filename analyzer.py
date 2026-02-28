import json
import os
import re
import math
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ArgumentAnalyzer:
    def __init__(self):
        """Initialize the analyzer with definitions and OpenAI client"""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),timeout=60.0,max_retries=2,)
        self.definitions = self._load_definitions()
        # How many to show on the main cards (top-N); scoring uses ALL detected
        self.display_fallacies = 3
        self.display_biases = 3
        self.display_distortions = 2
        
    def _load_definitions(self):
        """Load definitions from JSON file"""
        try:
            with open('definitions.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Error: definitions.json not found")
            return {}
    
    def analyze_argument(self, argument_text):
        """
        Two-stage pipeline:
          Call 1 (cheap): Extract structural metadata from the argument.
          Call 2 (main):  Analyze using the metadata as context.
        """
        try:
            # CALL 1 — lightweight metadata extraction
            metadata = self._extract_metadata(argument_text)
            short_deduction_hint = self._extract_short_deduction_hint(argument_text)

            # CALL 2 — main analysis, enriched with metadata
            prompt = self._create_analysis_prompt(argument_text, metadata)

            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are an expert in logic, cognitive psychology, and philosophy. Analyze arguments objectively."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            analysis_result = response.choices[0].message.content
            
            # Parse the response to extract detected issues
            detected_issues = self._parse_openai_response(analysis_result)

            # Enforce fixed output sizes and always include all 6 razors
            detected_issues = self._normalize_detected_issues(detected_issues)
            
            # Calculate score
            score = self._calculate_score(detected_issues, argument_text, metadata, short_deduction_hint)

            # CALL 3 — improvement suggestions based on detected issues
            improvements = self._generate_improvements(argument_text, detected_issues)
            
            return {
                "success": True,
                "raw_analysis": analysis_result,
                "detected_issues": detected_issues,
                "metadata": metadata,
                "short_deduction_hint": short_deduction_hint,
                "score": score,
                "score_breakdown": self._get_score_breakdown(detected_issues, argument_text, metadata, short_deduction_hint),
                "improvements": improvements
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # ──────────────────────────────────────────────
    #  CALL 1:  Lightweight metadata extraction
    # ──────────────────────────────────────────────


    def _extract_metadata(self, argument_text):
        """
        Cheap LLM call that decomposes the argument into individual
        claims and tags each with structural + epistemic features.
        Returns a list of claim dicts (or an empty list on failure).
        """
        prompt = f"""Break this argument into its individual claims.
For EACH claim return a JSON object with ALL of these fields:

=== STRUCTURAL TAGS (boolean / enum) ===
- claim_text              : string – the verbatim claim (keep it short)
- cites_evidence          : bool – references data, studies, or observable facts?
- evidence_type           : "scientific" | "anecdotal" | "none"
- cites_authority         : bool – names an authority figure or institution?
- authority_named         : string (empty if none)
- emotional_tone          : "neutral" | "positive" | "negative" | "fear" | "anger"
- makes_causal_claim      : bool – says X causes Y?
- generalizes             : bool – generalizes from specific cases?
- uses_absolute_language  : bool – words like always, never, everyone, no one
- targets_person          : bool – attacks a person instead of their point?
- assumes_intent          : bool – attributes motive or malice?
- is_falsifiable          : bool – could this claim be tested or disproven?
- is_extraordinary        : bool – surprising / unusual claim?

=== EXEMPLAR QUALITY ===
- exemplar_type           : "population_data" | "representative_sample" | "famous_case" | "anecdote" | "none"
                            What kind of example does the claim rely on?
                            "population_data"       = cites statistics or studies covering a full population.
                            "representative_sample" = references a broadly typical case or controlled sample.
                            "famous_case"           = uses a well-known or cherry-picked example (e.g. Amazon, Einstein).
                            "anecdote"              = personal story or single isolated case.
                            "none"                  = no specific example cited.

=== EPISTEMIC QUALITY (scale / enum) ===
- face_validity           : "high" | "medium" | "low"
                            Does the claim seem plausible on its surface to a
                            reasonable, educated person — before checking sources?
- speculation_level       : "none" | "low" | "moderate" | "high"
                            How much is the claim speculating beyond available evidence?
                            "none" = directly stating a verified fact.
                            "high" = conjecture with little grounding.
- claim_type              : "factual" | "historical" | "opinion" | "prediction" | "definition" | "moral"
                            What category of statement is this?
- evidence_sufficiency    : "strong" | "moderate" | "weak" | "none"
                            How well does the cited evidence actually support THIS claim?
                            "strong" = directly relevant, verifiable evidence cited.
                            "none"   = no evidence offered or evidence is irrelevant.
- causal_chain_length     : integer (0-5)
                            How many cause/precursor steps does the claim cite
                            before reaching its conclusion?
                            0 = bare assertion, 1 = one reason given, etc.
- inferential_gap         : "none" | "small" | "large"
                            How big is the logical leap from the stated evidence
                            to the conclusion drawn?
                            "none" = conclusion follows directly.
                            "large" = major assumptions needed to bridge the gap.
- specificity             : "high" | "medium" | "low"
                            Is the claim concrete and specific, or vague?
- verifiability           : "easily" | "with_effort" | "not_verifiable"
                            Could a third party check this claim?

=== REASONING PATTERN TAGS ===
- forces_balance          : bool
                            Does the claim impose artificial symmetry — presenting
                            both sides as equally weighted (pros vs cons, good vs bad)
                            even though the evidence clearly favours one side?
                            true = forced/artificial balance despite evidence asymmetry.
                            false = no forced balance, or genuinely balanced evidence.
- proportional_causation  : bool
                            Does the claim assume the magnitude of a cause must match
                            the magnitude of its effect? E.g., "a catastrophe this big
                            must have been deliberate" or "such a small mistake couldn't
                            possibly cause this much damage."
                            true = cause-effect proportionality is assumed without evidence.
                            false = no proportionality assumption.

Return VALID JSON ONLY — an array of objects. No markdown fences.

Argument:
\"\"\"{argument_text}\"\"\"
"""
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": (
                        "You are a text analysis assistant. Extract structural and "
                        "epistemic features from text. Tag each claim honestly — "
                        "do not inflate or deflate quality. Be precise with scales."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            raw = response.choices[0].message.content
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                claims = json.loads(json_match.group())
                if isinstance(claims, list):
                    return claims
        except Exception as e:
            print(f"Metadata extraction failed (non-fatal): {e}")
        return []

    def _extract_short_deduction_hint(self, argument_text):
        """
        Extra short-input logic check for concise deductive arguments (e.g., math/philosophy).
        Runs only on short text to keep latency and cost bounded.
        """
        words = len((argument_text or "").split())
        if words == 0 or words > 80:
            return {
                "checked": False,
                "is_short_deduction": False,
                "deduction_strength": "none",
                "reason": ""
            }

        prompt = f"""Assess whether this SHORT argument is a valid concise deductive argument.
Focus on formal reasoning quality, even if evidence citations are absent.

Return VALID JSON ONLY with fields:
- is_short_deduction: boolean
- deduction_strength: "none" | "weak" | "moderate" | "strong"
- reason: brief explanation (1 sentence)

Argument:
\"\"\"{argument_text}\"\"\"
"""
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a formal reasoning evaluator. For short statements, detect "
                            "valid deductive structure and avoid penalizing lack of empirical evidence."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            raw = response.choices[0].message.content
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                hint = json.loads(json_match.group())
                strength = str(hint.get("deduction_strength", "none")).lower()
                if strength not in {"none", "weak", "moderate", "strong"}:
                    strength = "none"
                return {
                    "checked": True,
                    "is_short_deduction": bool(hint.get("is_short_deduction", False)),
                    "deduction_strength": strength,
                    "reason": str(hint.get("reason", ""))[:240],
                }
        except Exception as e:
            print(f"Short deduction hint failed (non-fatal): {e}")

        return {
            "checked": True,
            "is_short_deduction": False,
            "deduction_strength": "none",
            "reason": ""
        }

    def _metadata_to_context(self, metadata):
        """
        Aggregates raw per-claim metadata into a concise context
        block for the main analysis prompt.
        """
        if not metadata:
            return ""

        total = len(metadata)

        # --- Structural counts ---
        evidence_claims = [c for c in metadata if c.get("cites_evidence")]
        causal_claims   = [c for c in metadata if c.get("makes_causal_claim")]
        emotional       = [c for c in metadata if c.get("emotional_tone") not in ("neutral", None)]
        absolute_lang   = [c for c in metadata if c.get("uses_absolute_language")]
        person_attacks  = [c for c in metadata if c.get("targets_person")]
        unfalsifiable   = [c for c in metadata if not c.get("is_falsifiable")]
        extraordinary   = [c for c in metadata if c.get("is_extraordinary")]
        generalizing    = [c for c in metadata if c.get("generalizes")]
        intent_assumed  = [c for c in metadata if c.get("assumes_intent")]
        forced_balance  = [c for c in metadata if c.get("forces_balance")]
        prop_causation  = [c for c in metadata if c.get("proportional_causation")]

        # --- Epistemic aggregations ---
        def _count(field, value):
            return sum(1 for c in metadata if c.get(field) == value)

        high_validity   = _count("face_validity", "high")
        low_validity    = _count("face_validity", "low")
        high_spec       = _count("speculation_level", "high") + _count("speculation_level", "moderate")
        no_spec         = _count("speculation_level", "none")
        strong_ev       = _count("evidence_sufficiency", "strong")
        weak_ev         = _count("evidence_sufficiency", "weak") + _count("evidence_sufficiency", "none")
        large_gap       = _count("inferential_gap", "large")
        no_gap          = _count("inferential_gap", "none")
        high_specific   = _count("specificity", "high")
        low_specific    = _count("specificity", "low")
        easily_verified = _count("verifiability", "easily")
        not_verifiable  = _count("verifiability", "not_verifiable")

        type_counts = {}
        for c in metadata:
            ct = c.get("claim_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1

        avg_chain = 0.0
        chain_vals = [c.get("causal_chain_length", 0) for c in metadata if isinstance(c.get("causal_chain_length"), (int, float))]
        if chain_vals:
            avg_chain = sum(chain_vals) / len(chain_vals)

        # --- Build context block ---
        lines = [
            f"The argument contains {total} individual claim(s).",
            "",
            "STRUCTURAL PROFILE:",
            f"  - {len(evidence_claims)} cite evidence ({_count('evidence_type', 'scientific')} scientific, "
            f"{_count('evidence_type', 'anecdotal')} anecdotal).",
            f"  - {len(causal_claims)} make causal claims.",
            f"  - {len(emotional)} have non-neutral emotional tone.",
            f"  - {len(absolute_lang)} use absolute language (always/never/everyone).",
            f"  - {len(person_attacks)} target a person rather than their argument.",
            f"  - {len(unfalsifiable)} appear unfalsifiable.",
            f"  - {len(extraordinary)} make extraordinary claims.",
            f"  - {len(generalizing)} generalize from specific cases.",
            f"  - {len(intent_assumed)} assume intent or motive.",
            f"  - {len(forced_balance)} impose artificial symmetry/balance despite evidence asymmetry.",
            f"  - {len(prop_causation)} assume cause-effect proportionality without justification.",
            f"  - Exemplar types: {_count('exemplar_type', 'population_data')} population data, "
            f"{_count('exemplar_type', 'representative_sample')} representative sample, "
            f"{_count('exemplar_type', 'famous_case')} famous/cherry-picked case, "
            f"{_count('exemplar_type', 'anecdote')} anecdote, "
            f"{_count('exemplar_type', 'none')} none.",
            "",
            "EPISTEMIC PROFILE:",
            f"  - Face validity:        {high_validity} high, {_count('face_validity', 'medium')} medium, {low_validity} low.",
            f"  - Speculation:           {no_spec} grounded (none), {high_spec} moderate-to-high speculation.",
            f"  - Evidence sufficiency:  {strong_ev} strong, {_count('evidence_sufficiency', 'moderate')} moderate, {weak_ev} weak/none.",
            f"  - Avg causal chain:      {avg_chain:.1f} steps (0 = bare assertion, 3+ = well-built reasoning).",
            f"  - Inferential gaps:      {no_gap} none, {_count('inferential_gap', 'small')} small, {large_gap} large.",
            f"  - Specificity:           {high_specific} high, {low_specific} low.",
            f"  - Verifiability:         {easily_verified} easily verified, {not_verifiable} not verifiable.",
            f"  - Claim types:           {', '.join(f'{v} {k}' for k, v in sorted(type_counts.items(), key=lambda x: -x[1]))}.",
        ]

        # --- Flags (targeted alerts for Call 2) ---
        lines.append("")
        lines.append("FLAGS:")

        if person_attacks:
            lines.append("  ⚑ Person-targeting detected — check for Ad Hominem / Tu Quoque.")
        if absolute_lang:
            lines.append("  ⚑ Absolute language detected — check for All-or-Nothing thinking / Overgeneralization.")
        if extraordinary and not evidence_claims:
            lines.append("  ⚑ Extraordinary claims without evidence — likely fails Sagan Standard.")
        if unfalsifiable:
            lines.append("  ⚑ Unfalsifiable claims found — likely fails Popper's Falsifiability.")
        if high_spec >= total * 0.5:
            lines.append("  ⚑ Majority of claims are speculative — watch for Hasty Generalization / Jumping to Conclusions.")
        if large_gap >= 2:
            lines.append("  ⚑ Multiple large inferential gaps — watch for Non Sequitur / Slippery Slope.")
        if weak_ev >= total * 0.5 and causal_claims:
            lines.append("  ⚑ Causal claims with weak evidence — watch for Post Hoc / False Cause.")
        if low_validity >= 2:
            lines.append("  ⚑ Multiple low face-validity claims — argument may be hard to take seriously without strong evidence.")
        if not_verifiable >= total * 0.5:
            lines.append("  ⚑ Most claims are not independently verifiable — weakens overall credibility.")
        if avg_chain < 0.5 and total >= 3:
            lines.append("  ⚑ Claims are mostly bare assertions (avg chain < 0.5) — little supporting reasoning provided.")

        # ── Niche bias nudges ──

        # Exemplar-quality counts (from Call 1's new field)
        famous_cases = [c for c in metadata if c.get("exemplar_type") == "famous_case"]
        anecdotes    = [c for c in metadata if c.get("exemplar_type") == "anecdote"]
        pop_data     = [c for c in metadata if c.get("exemplar_type") in
                        ("population_data", "representative_sample")]
        cherry_picked = len(famous_cases) + len(anecdotes)

        # Survivorship bias: only when the argument genuinely cherry-picks minority
        # or famous cases while generalizing.  Population data or representative
        # samples should NOT trigger this — hypothesizing an invisible counter-sample
        # against solid data is itself unfalsifiable (Russell's Teapot).
        if cherry_picked >= 2 and generalizing and len(pop_data) < cherry_picked:
            lines.append(
                "  ⚑ Argument generalizes from famous/cherry-picked cases — check for "
                "Survivorship Bias (visible successes cited while failures are absent)."
            )

        if intent_assumed:
            lines.append(
                "  ⚑ Intent/motive assumed — check for Empathy Gap (underestimating others' "
                "emotional states) and Fundamental Attribution Error (attributing behaviour to "
                "character rather than circumstance)."
            )

        self_serving_claims = [c for c in metadata
                               if c.get("claim_type") == "opinion"
                               and c.get("speculation_level") in ("moderate", "high")]
        if self_serving_claims and len(self_serving_claims) >= total * 0.4:
            lines.append(
                "  ⚑ Many speculative opinion claims — check for Self-Serving Bias "
                "(interpreting information in a way that flatters or protects the arguer's position)."
            )

        simple_proxy = [c for c in metadata
                        if c.get("inferential_gap") == "large"
                        and c.get("specificity") == "low"]
        if simple_proxy:
            lines.append(
                "  ⚑ Vague claims with large inferential leaps — check for Attribute Substitution "
                "(swapping a hard question for an easier proxy) and Belief Bias "
                "(accepting reasoning because the conclusion feels right)."
            )

        if emotional and len(emotional) >= total * 0.5:
            lines.append(
                "  ⚑ Majority of claims carry emotional tone — check for Egocentric Bias "
                "(over-relying on one's own perspective) in addition to emotional reasoning."
            )

        # Russell's Teapot: unfalsifiable claims that shift the burden of proof
        unfalsifiable_speculative = [c for c in metadata
                                     if not c.get("is_falsifiable")
                                     and c.get("speculation_level") in ("moderate", "high")]
        if unfalsifiable_speculative and len(unfalsifiable_speculative) >= 2:
            lines.append(
                "  ⚑ Multiple unfalsifiable speculative claims — check for Russell's Teapot "
                "(shifting the burden of proof by making claims others cannot disprove)."
            )

        # Symmetry Impulse: forced balance / artificial symmetry
        if forced_balance:
            lines.append(
                "  ⚑ Claims impose artificial balance (e.g., pros must equal cons, every "
                "positive must have a negative) — check for Symmetry Impulse "
                "(using symmetry to complete judgments when evidence doesn't support equal weight)."
            )

        # Proportionality Bias (MEMC): cause-effect magnitude matching
        if prop_causation:
            lines.append(
                "  ⚑ Claims assume cause magnitude must match effect magnitude — check for "
                "Proportionality Bias / Major-Event-Major-Cause heuristic "
                "(large effects must have large causes, dismissing small or accidental causes)."
            )

        # Maslow's Hammer: applying one framework/theory to everything
        if generalizing and len(causal_claims) >= 3 and len(type_counts) <= 2:
            lines.append(
                "  ⚑ Multiple causal claims from a narrow framework — check for Maslow's Hammer "
                "(forcing one explanatory theory onto every aspect of the issue)."
            )

        if no_spec == total and strong_ev >= total * 0.6:
            lines.append("  ✓ Mostly grounded claims with strong evidence — likely a well-supported argument.")
        if easily_verified >= total * 0.6:
            lines.append("  ✓ Most claims are easily verifiable — strengthens testability for razor evaluation.")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    #  CALL 2:  Main analysis prompt (metadata-enriched)
    # ──────────────────────────────────────────────

    def _create_analysis_prompt(self, argument_text, metadata=None):
        """Create the prompt for the main OpenAI analysis call."""

        fallacies = list(self.definitions.get('logical_fallacies', {}).keys())
        biases = list(self.definitions.get('cognitive_biases', {}).keys())
        distortions = list(self.definitions.get('cognitive_distortions', {}).keys())
        razors = list(self.definitions.get('philosophical_razors', {}).keys())

        metadata_block = ""
        if metadata:
            summary = self._metadata_to_context(metadata)
            metadata_block = f"""
--- PRE-ANALYSIS (structural metadata extracted from the argument) ---
{summary}
--- END PRE-ANALYSIS ---

Use the structural metadata above to guide your analysis.
Where a flag is raised (⚑), pay close attention to the related category.
Where evidence or falsifiability is noted, factor it into your razor evaluations.

"""

        prompt = f"""Analyze the following argument for:
- logical fallacies (detect ALL that are genuinely present)
- cognitive biases (detect ALL that are genuinely present)
- cognitive distortions (detect ALL that are genuinely present)
- philosophical razors (evaluate ALL razors listed, pass/fail each, with a short reason)

Argument: "{argument_text}"

{metadata_block}IMPORTANT RULES:
- Return VALID JSON ONLY (no markdown, no extra text).
- For fallacies, biases, and distortions: return EVERY item you detect with confidence >= 0.6.
  Include as many or as few as are truly present. Do NOT pad with "none".
- Only use keys from the available lists below.
- Be accurate: only label a fallacy/bias/distortion if it is genuinely present.
- Include a confidence score (0.0 to 1.0) for each item.
- BIAS SPECIFICITY RULE: Always prefer a specific bias over a generic parent.
  If the argument interprets evidence in a self-flattering way, flag "self_serving_bias"
  rather than "confirmation_bias". Only use "confirmation_bias" when no more specific
  variant applies. Similarly, prefer "fundamental_attribution_error" or "empathy_gap"
  over vague labels when the mechanism is clear.
- SURVIVORSHIP BIAS GUARD: Only flag "survivorship_bias" when the argument *explicitly*
  draws conclusions from a few visible/successful cases while clearly ignoring known
  failures or counter-examples (e.g., "my uncle smoked and lived to 95, so smoking is
  fine"). Do NOT flag it merely because the argument generalizes or because you can
  hypothesize an unseen counter-sample. The accusation of survivorship bias must itself
  be falsifiable — there must be a concrete, identifiable missing sample, not just a
  theoretical one.
- SYMMETRY IMPULSE: Flag "symmetry_impulse" when the argument forces artificial balance
  or symmetry — e.g. assuming every pro must have a con, every positive a negative, or
  that both sides deserve equal weight despite evidence strongly favouring one side.
  Do NOT flag it when the argument presents genuinely balanced evidence.
- PROPORTIONALITY BIAS: Flag "proportionality_bias" when the argument assumes that the
  magnitude of a cause must match the magnitude of its effect (large events need large
  causes). Classic indicator: dismissing simple or accidental causes because the outcome
  is dramatic, or inflating a cause to match a dramatic consequence.
- For philosophical_razors, you MUST return ONE object for EVERY razor listed below (no omissions), each with:
  - the correct "key" from the list,
  - a boolean "pass",
  - a non-empty "reason" explaining why it passes or fails in the context of THIS argument,
  - a confidence value.

Return this JSON schema exactly:
{{
  "logical_fallacies": [{{"key":"...", "reason":"...", "confidence": 0.0}}],
  "cognitive_biases": [{{"key":"...", "reason":"...", "confidence": 0.0}}],
  "cognitive_distortions": [{{"key":"...", "reason":"...", "confidence": 0.0}}],
  "philosophical_razors": [
    {{"key":"razor_key_from_list", "pass": true, "reason":"...", "confidence": 0.0}},
    {{"key":"razor_key_from_list", "pass": false, "reason":"...", "confidence": 0.0}}
  ],
  "executive_summary_sentence": "One short sentence summarizing the argument.",
  "executive_summary_bullets": ["Bullet 1", "Bullet 2", "Bullet 3"],
  "summary": "1-3 sentences summarizing the biggest weaknesses and biggest strengths."
}}

Available logical fallacies: {', '.join(fallacies)}
Available cognitive biases: {', '.join(biases)}
Available cognitive distortions: {', '.join(distortions)}
Available philosophical razors: {', '.join(razors)}
"""

        return prompt
    
    def _parse_openai_response(self, response_text):
        """Parse OpenAI response to extract detected issues"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "logical_fallacies": parsed.get("logical_fallacies", []),
                    "cognitive_biases": parsed.get("cognitive_biases", []),
                    "cognitive_distortions": parsed.get("cognitive_distortions", []),
                    "philosophical_razors": parsed.get("philosophical_razors", []),
                    "executive_summary_sentence": parsed.get("executive_summary_sentence", ""),
                    "executive_summary_bullets": parsed.get("executive_summary_bullets", []),
                    "summary": parsed.get("summary", "")
                }
        except:
            pass
        
        # Fallback: return empty structure
        return {
            "logical_fallacies": [],
            "cognitive_biases": [],
            "cognitive_distortions": [],
            "philosophical_razors": [],
            "executive_summary_sentence": "",
            "executive_summary_bullets": [],
            "summary": response_text
        }

    # Specificity-first bias hierarchy: if a niche child is detected,
    # suppress the generic parent (the child already implies it).
    BIAS_HIERARCHY = {
        "confirmation_bias": [
            "survivorship_bias",
            "self_serving_bias",
            "belief_bias",
        ],
    }

    def _deduplicate_biases(self, biases):
        """
        Specificity-first deduplication with umbrella injection:
        1. If LLM detected both parent and child, drop the parent's penalty
           (child already carries the specific penalty).
        2. If LLM detected a child but NOT the parent, inject the parent as a
           zero-penalty umbrella label so the user sees the family relationship.
        """
        detected_keys = {b["key"] for b in biases}
        umbrella_parents = set()

        for parent, children in self.BIAS_HIERARCHY.items():
            if any(child in detected_keys for child in children):
                umbrella_parents.add(parent)

        if not umbrella_parents:
            return biases

        result = []
        for b in biases:
            if b["key"] in umbrella_parents:
                result.append({**b, "penalty": 0, "is_umbrella": True})
            else:
                result.append(b)

        for parent_key in umbrella_parents:
            if parent_key not in detected_keys:
                d = self.definitions.get("cognitive_biases", {}).get(parent_key, {})
                child_names = []
                for child_key in self.BIAS_HIERARCHY[parent_key]:
                    if child_key in detected_keys:
                        cd = self.definitions.get("cognitive_biases", {}).get(child_key, {})
                        child_names.append(cd.get("name", child_key))
                result.append({
                    "key": parent_key,
                    "name": d.get("name", parent_key),
                    "reason": f"Umbrella category — detected via: {', '.join(child_names)}.",
                    "penalty": 0,
                    "description": d.get("description", ""),
                    "confidence": 1.0,
                    "is_umbrella": True,
                })

        result.sort(key=lambda x: abs(x["penalty"]), reverse=True)
        return result

    def _normalize_detected_issues(self, detected):
        """
        - Keep ALL valid items for fallacies/biases/distortions (scoring uses them all).
        - Filter out low-confidence items.
        - Deduplicate biases: suppress generic parents when a niche child is present.
        - Sort by penalty descending so the most impactful are first.
        - Include ALL razors with pass boolean (default false if missing).
        - Attach name/penalty/reward/description from definitions.json.
        """
        issue_confidence = float(os.getenv("ISSUE_CONFIDENCE", "0.20"))
        razor_confidence = float(os.getenv("RAZOR_CONFIDENCE", "0.30"))

        def _normalize_all(items, definition_group):
            """Accept ALL valid items above the confidence threshold."""
            allowed = set(self.definitions.get(definition_group, {}).keys())
            normalized = []

            for item in items if isinstance(items, list) else []:
                if isinstance(item, str):
                    key = item
                    reason = ""
                    confidence = 1.0
                else:
                    key = (item or {}).get("key", "")
                    reason = (item or {}).get("reason", "")
                    confidence = float((item or {}).get("confidence", 0.0) or 0.0)

                if key == "none" or key not in allowed:
                    continue

                if confidence < issue_confidence:
                    continue

                d = self.definitions[definition_group][key]
                normalized.append({
                    "key": key,
                    "name": d.get("name", key),
                    "reason": reason,
                    "penalty": d.get("penalty", 0),
                    "description": d.get("description", ""),
                    "confidence": confidence
                })

            normalized.sort(key=lambda x: abs(x["penalty"]), reverse=True)
            return normalized

        biases = _normalize_all(detected.get("cognitive_biases", []), "cognitive_biases")
        biases = self._deduplicate_biases(biases)

        out = {
            "logical_fallacies": _normalize_all(detected.get("logical_fallacies", []), "logical_fallacies"),
            "cognitive_biases": biases,
            "cognitive_distortions": _normalize_all(detected.get("cognitive_distortions", []), "cognitive_distortions"),
            "philosophical_razors": [],
            "executive_summary_sentence": detected.get("executive_summary_sentence", ""),
            "executive_summary_bullets": detected.get("executive_summary_bullets", []),
            "summary": detected.get("summary", "")
        }

        # Razors: always include all defined razors
        razor_defs = self.definitions.get("philosophical_razors", {})
        incoming = detected.get("philosophical_razors", [])
        incoming_by_key = {}
        if isinstance(incoming, list):
            for r in incoming:
                if isinstance(r, dict) and r.get("key"):
                    incoming_by_key[r["key"]] = r

        for key, d in razor_defs.items():
            incoming_r = incoming_by_key.get(key, {})
            confidence = float(incoming_r.get("confidence", 0.0) or 0.0)
            passed = bool(incoming_r.get("pass", False)) and confidence >= razor_confidence
            # Fallback generic reason if the model omitted it
            raw_reason = (incoming_r.get("reason", "") or "").strip()
            if not raw_reason:
                if passed:
                    raw_reason = "This argument broadly aligns with this razor, so it is marked as passed."
                else:
                    raw_reason = "There is not enough clear support in this text to say this razor is satisfied."
            out["philosophical_razors"].append({
                "key": key,
                "name": d.get("name", key),
                "pass": passed,
                "reason": raw_reason,
                "description": d.get("description", ""),
                "confidence": confidence,
                "reward": d["reward"] if passed else 0
            })

        return out
    
    # ──────────────────────────────────────────────
    #  CALL 3:  Improvement suggestions
    # ──────────────────────────────────────────────

    def _generate_improvements(self, argument_text, detected_issues):
        """
        Build deterministic improvement context from definitions, then ask
        the LLM to synthesize exactly 5 concrete, actionable suggestions.
        """
        try:
            hints = []

            for category in ("logical_fallacies", "cognitive_biases", "cognitive_distortions"):
                for item in detected_issues.get(category, []):
                    key = item.get("key") if isinstance(item, dict) else item
                    defn = self.definitions.get(category, {}).get(key)
                    if defn and "improvements" in defn:
                        for tip in defn["improvements"]:
                            hints.append(f"[{defn['name']}] {tip}")

            for item in detected_issues.get("philosophical_razors", []):
                if isinstance(item, dict) and not item.get("pass", False):
                    key = item.get("key", "")
                    defn = self.definitions.get("philosophical_razors", {}).get(key)
                    if defn and "improvement" in defn:
                        hints.append(f"[{defn['name']} — failed] {defn['improvement']}")

            if not hints:
                return []

            hints_block = "\n".join(f"- {h}" for h in hints[:20])

            prompt = f"""Given this argument:
\"\"\"{argument_text[:1500]}\"\"\"

The analysis detected these issues with mapped improvement strategies:
{hints_block}

Synthesize exactly 5 concrete, actionable suggestions the author can implement to strengthen this argument. Each should be 1-2 sentences. Be specific to this argument's content. Do NOT repeat the issue name — focus on the fix.

Return a JSON array of 5 strings, nothing else. Example:
["suggestion 1", "suggestion 2", "suggestion 3", "suggestion 4", "suggestion 5"]"""

            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a writing coach. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            suggestions = json.loads(raw)
            if isinstance(suggestions, list):
                return [str(s) for s in suggestions[:5]]
            return []
        except Exception:
            return []

    @staticmethod
    def _substance_penalty(argument_text, metadata):
        """
        Penalize arguments that are too short or vacuous to analyze properly.
        A 3-word opinion shouldn't start at the same base as a 200-word argument.
        """
        word_count = len(argument_text.split())
        claim_count = len(metadata) if metadata else 0
        penalty = 0

        if word_count < 10:
            penalty += 40
        elif word_count <= 30:
            penalty += 15

        if claim_count <= 1:
            penalty += 10

        return penalty

    @staticmethod
    def _safe_ratio(numerator, denominator):
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    def _detect_argument_mode(self, metadata, argument_text):
        """
        Detect whether the argument is primarily empirical, deductive,
        speculative, or fictional/world-internal.
        """
        metadata = metadata or []
        text = (argument_text or "").lower()
        total = len(metadata)

        fiction_keywords = [
            "dragon", "wizard", "sorcer", "kingdom", "empire", "starship",
            "spaceship", "galaxy", "alien", "time travel", "portal",
            "magic", "prophecy", "planet", "interstellar", "cyberpunk",
            "mutant", "undead", "sword", "realm", "timeline"
        ]
        fiction_hits = sum(1 for k in fiction_keywords if k in text)

        if not metadata:
            if fiction_hits >= 2:
                return "fictional"
            if re.search(r"\b(if|therefore|hence|thus)\b", text):
                return "deductive"
            return "speculative"

        evidence_claims = sum(1 for c in metadata if c.get("cites_evidence"))
        weak_ev = sum(1 for c in metadata if c.get("evidence_sufficiency") in ("weak", "none"))
        prediction_claims = sum(1 for c in metadata if c.get("claim_type") == "prediction")
        speculative_claims = sum(1 for c in metadata if c.get("speculation_level") in ("moderate", "high"))
        not_verifiable = sum(1 for c in metadata if c.get("verifiability") == "not_verifiable")

        evidence_ratio = self._safe_ratio(evidence_claims, total)
        weak_evidence_ratio = self._safe_ratio(weak_ev, total)
        prediction_ratio = self._safe_ratio(prediction_claims, total)
        speculative_ratio = self._safe_ratio(speculative_claims, total)
        unverifiable_ratio = self._safe_ratio(not_verifiable, total)

        if fiction_hits >= 2 and (speculative_ratio >= 0.3 or unverifiable_ratio >= 0.3):
            return "fictional"
        if evidence_ratio >= 0.45 and weak_evidence_ratio < 0.55:
            return "empirical"
        if speculative_ratio >= 0.5 or prediction_ratio >= 0.4:
            return "speculative"
        return "deductive"

    def _compute_logic_variables(self, argument_text, metadata):
        """
        Deterministic structural variables for logic integrity, intentionally
        lightweight so they can run without adding more LLM calls.
        """
        metadata = metadata or []
        text = argument_text or ""
        lowered = text.lower()
        claim_count = len(metadata)

        assumption_pattern = r"\b(assume|assuming|suppose|supposing|given that|let us assume|if we accept|granted)\b"
        boundary_pattern = r"\b(under this assumption|within this framework|if this holds|therefore|thus|hence|then)\b"
        if_pattern = r"\bif\b"
        then_pattern = r"\b(then|therefore|thus|hence|so)\b"
        connector_pattern = r"\b(because|therefore|thus|hence|so|implies|suggests|indicates|evidence shows|consistent with|as a result|given that)\b"
        hedge_pattern = r"\b(might|may|could|possibly|perhaps|likely|plausibly|probably|arguably)\b"
        absolute_pattern = r"\b(always|never|everyone|no one|all|none|must|cannot|impossible|certainly|definitely)\b"

        assumption_count = len(re.findall(assumption_pattern, lowered))
        boundary_markers = len(re.findall(boundary_pattern, lowered))
        if_then_count = len(re.findall(if_pattern, lowered))
        then_count = len(re.findall(then_pattern, lowered))
        connector_count = len(re.findall(connector_pattern, lowered))
        hedges = len(re.findall(hedge_pattern, lowered))
        absolute_terms = len(re.findall(absolute_pattern, lowered))

        if assumption_count == 0:
            assumption_boundary_clarity = 0.0
        else:
            assumption_boundary_clarity = min(100.0, (boundary_markers / float(assumption_count)) * 100.0)

        if if_then_count == 0:
            if_then_completeness = 0.0
        else:
            if_then_completeness = min(100.0, (then_count / float(if_then_count)) * 100.0)

        contradiction_signals = 0
        if "always" in lowered and "never" in lowered:
            contradiction_signals += 1
        if "all " in lowered and "none " in lowered:
            contradiction_signals += 1
        if "must" in lowered and "cannot" in lowered:
            contradiction_signals += 1

        high_gap = sum(1 for c in metadata if c.get("inferential_gap") == "large")
        low_validity = sum(1 for c in metadata if c.get("face_validity") == "low")
        contradiction_signals += 1 if high_gap >= 2 and low_validity >= 1 else 0
        contradiction_index = min(100.0, contradiction_signals * 22.0 + high_gap * 12.0 + low_validity * 8.0)

        weak_ev = sum(1 for c in metadata if c.get("evidence_sufficiency") in ("weak", "none"))
        abs_claims = sum(1 for c in metadata if c.get("uses_absolute_language"))
        weak_ratio = self._safe_ratio(weak_ev, claim_count if claim_count else 1)
        abs_ratio = self._safe_ratio(abs_claims, claim_count if claim_count else 1)
        overclaim_penalty = min(100.0, (abs_ratio * 55.0) + (weak_ratio * 40.0) + (absolute_terms * 4.0))

        speculative_claims = sum(1 for c in metadata if c.get("speculation_level") in ("moderate", "high"))
        speculative_ratio = self._safe_ratio(speculative_claims, claim_count if claim_count else 1)
        uncertainty_calibration = min(100.0, max(0.0, (hedges * 9.0) + (speculative_ratio * 35.0) - (absolute_terms * 3.0)))

        chain_vals = [
            c.get("causal_chain_length", 0)
            for c in metadata
            if isinstance(c.get("causal_chain_length"), (int, float))
        ]
        avg_chain = sum(chain_vals) / len(chain_vals) if chain_vals else 0.0
        inferential_gap_penalty = min(40.0, high_gap * 10.0)
        inference_chain_stability = max(0.0, min(100.0, (avg_chain * 20.0) + 45.0 - inferential_gap_penalty))

        assumption_quality = min(100.0, assumption_count * 18.0)
        assumption_quality = max(20.0, assumption_quality)
        connector_score = min(100.0, (connector_count * 12.0) + (then_count * 8.0) + (if_then_count * 6.0))

        strong_ev = sum(1 for c in metadata if c.get("evidence_sufficiency") in ("strong", "moderate"))
        falsifiable = sum(1 for c in metadata if c.get("is_falsifiable"))
        verifiable = sum(1 for c in metadata if c.get("verifiability") in ("easily", "with_effort"))
        small_gap = sum(1 for c in metadata if c.get("inferential_gap") in ("none", "small"))
        science_reasoning_proxy = (
            0.34 * (self._safe_ratio(strong_ev, claim_count if claim_count else 1) * 100.0)
            + 0.24 * (self._safe_ratio(falsifiable, claim_count if claim_count else 1) * 100.0)
            + 0.24 * (self._safe_ratio(verifiable, claim_count if claim_count else 1) * 100.0)
            + 0.18 * (self._safe_ratio(small_gap, claim_count if claim_count else 1) * 100.0)
        )
        science_reasoning_proxy = max(0.0, min(100.0, science_reasoning_proxy))

        logic_integrity_score = (
            0.09 * assumption_quality
            + 0.17 * assumption_boundary_clarity
            + 0.11 * min(100.0, if_then_count * 20.0)
            + 0.10 * if_then_completeness
            + 0.14 * (100.0 - contradiction_index)
            + 0.14 * (100.0 - overclaim_penalty)
            + 0.06 * uncertainty_calibration
            + 0.07 * inference_chain_stability
            + 0.06 * connector_score
            + 0.06 * science_reasoning_proxy
        )

        # Dampen logic integrity for very short/vague text with no explicit structure.
        word_count = len(text.split())
        explicit_structure_signals = assumption_count + if_then_count + connector_count
        if word_count < 12 and explicit_structure_signals == 0:
            logic_integrity_score *= 0.75
        elif word_count < 20 and explicit_structure_signals <= 1:
            logic_integrity_score *= 0.90

        logic_integrity_score = max(0.0, min(100.0, logic_integrity_score))

        evidence_claims = sum(1 for c in metadata if c.get("cites_evidence"))
        evidence_dependency_score = (
            # Stronger citation/evidence hint weight so scientific evidence cues
            # more reliably influence downstream scoring.
            0.38 * min(100.0, evidence_claims * 16.0)
            + 0.32 * min(100.0, strong_ev * 16.0)
            + 0.15 * (self._safe_ratio(falsifiable, claim_count if claim_count else 1) * 100.0)
            + 0.15 * (self._safe_ratio(verifiable, claim_count if claim_count else 1) * 100.0)
        )
        evidence_dependency_score = max(0.0, min(100.0, evidence_dependency_score))

        # Binary logic-presence flag derived from existing hints only.
        # True means at least minimal explicit logic structure is present.
        logic_presence_flag = bool(
            assumption_count >= 1
            or if_then_count >= 1
            or connector_count >= 2
            or science_reasoning_proxy >= 55
            or (
                word_count >= 35
                and inference_chain_stability >= 40
                and contradiction_index < 70
                and overclaim_penalty < 85
            )
            or evidence_dependency_score >= 40
        )

        # If logic is absent, limit logic integrity by length band.
        if not logic_presence_flag:
            if word_count < 20:
                logic_integrity_score = min(logic_integrity_score, 20.0)
            elif word_count < 60:
                logic_integrity_score = min(logic_integrity_score, 38.0 if science_reasoning_proxy >= 45 else 32.0)
            else:
                logic_integrity_score = min(logic_integrity_score, 50.0 if science_reasoning_proxy >= 45 else 45.0)

        return {
            "assumption_count": assumption_count,
            "assumption_boundary_clarity": round(assumption_boundary_clarity, 2),
            "if_then_count": if_then_count,
            "if_then_completeness": round(if_then_completeness, 2),
            "connector_count": connector_count,
            "connector_score": round(connector_score, 2),
            "contradiction_signals": contradiction_signals,
            "contradiction_index": round(contradiction_index, 2),
            "overclaim_penalty": round(overclaim_penalty, 2),
            "uncertainty_calibration": round(uncertainty_calibration, 2),
            "inference_chain_stability": round(inference_chain_stability, 2),
            "science_reasoning_proxy": round(science_reasoning_proxy, 2),
            "logic_presence_flag": logic_presence_flag,
            "logic_integrity_score": round(logic_integrity_score, 2),
            "evidence_dependency_score": round(evidence_dependency_score, 2),
        }

    def _compute_score_artifacts(self, detected_issues, argument_text="", metadata=None, short_deduction_hint=None):
        """
        Single scoring source for both final score and score breakdown.
        """
        return self._compute_score_artifacts_linear(detected_issues, argument_text, metadata, short_deduction_hint)

    def _compute_score_artifacts_linear(self, detected_issues, argument_text="", metadata=None, short_deduction_hint=None):
        """
        Linear scoring model around a fixed starting point (75).
        """
        metadata = metadata or []
        base_score = float(os.getenv("BASE_SCORE", "75"))
        fallacy_weight = float(os.getenv("FALLACY_WEIGHT", "1.05"))
        bias_weight = float(os.getenv("BIAS_WEIGHT", "0.35"))
        distortion_weight = float(os.getenv("DISTORTION_WEIGHT", "0.68"))
        max_razor_bonus = float(os.getenv("MAX_RAZOR_BONUS", "25"))

        mode = self._detect_argument_mode(metadata, argument_text)
        is_fictional = mode == "fictional"
        mode_penalty_multiplier = {
            "empirical": 1.00,
            "deductive": 1.00,
            "speculative": 0.90,
            "fictional": 0.75
        }.get(mode, 1.0)
        blend_weights = {
            "legacy_weight": 0.0,
            "logic_weight": 1.0
        }

        penalties = {}
        rewards = {}
        total_penalty = 0.0
        total_reward = 0.0
        issue_penalty_total = 0.0

        sub_pen = self._substance_penalty(argument_text, metadata)
        if sub_pen > 0:
            penalties["_substance"] = sub_pen
            total_penalty += sub_pen

        for item in detected_issues.get("logical_fallacies", []):
            key = item.get("key") if isinstance(item, dict) else item
            if key in self.definitions.get("logical_fallacies", {}):
                penalty = mode_penalty_multiplier * fallacy_weight * abs(self.definitions["logical_fallacies"][key]["penalty"])
                penalties[key] = penalties.get(key, 0.0) + penalty
                total_penalty += penalty
                issue_penalty_total += penalty

        for item in detected_issues.get("cognitive_biases", []):
            key = item.get("key") if isinstance(item, dict) else item
            if key in self.definitions.get("cognitive_biases", {}):
                penalty = mode_penalty_multiplier * bias_weight * abs(self.definitions["cognitive_biases"][key]["penalty"])
                penalties[key] = penalties.get(key, 0.0) + penalty
                total_penalty += penalty
                issue_penalty_total += penalty

        for item in detected_issues.get("cognitive_distortions", []):
            key = item.get("key") if isinstance(item, dict) else item
            if key in self.definitions.get("cognitive_distortions", {}):
                penalty = mode_penalty_multiplier * distortion_weight * abs(self.definitions["cognitive_distortions"][key]["penalty"])
                penalties[key] = penalties.get(key, 0.0) + penalty
                total_penalty += penalty
                issue_penalty_total += penalty

        for item in detected_issues.get("philosophical_razors", []):
            if isinstance(item, dict) and item.get("key") in self.definitions.get("philosophical_razors", {}):
                if bool(item.get("pass", False)):
                    reward = self.definitions["philosophical_razors"][item["key"]]["reward"]
                    rewards[item["key"]] = reward
                    total_reward += reward

        total_reward_capped = min(total_reward, max_razor_bonus)
        raw_score = max(0.0, min(100.0, base_score - total_penalty))
        legacy_final = max(0.0, min(100.0, raw_score + total_reward_capped))

        logic_vars = self._compute_logic_variables(argument_text, metadata)
        logic_score = float(logic_vars["logic_integrity_score"])

        # Three-dimension scoring:
        #   1) bias_level_score  -> higher means more bias/fallacy pressure
        #   2) testability_score -> razor alignment
        #   3) logic_score       -> logic integrity variables
        issue_count = (
            len(detected_issues.get("logical_fallacies", []))
            + len(detected_issues.get("cognitive_biases", []))
            + len(detected_issues.get("cognitive_distortions", []))
        )
        bias_count_boost_unit = float(os.getenv("BIAS_COUNT_BOOST", "2.2"))
        bias_pressure_scale = float(os.getenv("BIAS_PRESSURE_SCALE", "60"))
        raw_bias_pressure = issue_penalty_total + (issue_count * bias_count_boost_unit)
        # Soft-compress bias pressure to avoid near-binary 0/100 behavior.
        bias_level_score = 100.0 * (1.0 - math.exp(-raw_bias_pressure / max(1.0, bias_pressure_scale)))
        bias_level_score = max(0.0, min(100.0, bias_level_score))
        bias_quality_score = 100.0 - bias_level_score

        # Razor alignment (for compatibility and UI)
        num_razors = len(self.definitions.get("philosophical_razors", {}))
        passed_razors = 0
        for item in detected_issues.get("philosophical_razors", []):
            if isinstance(item, dict) and bool(item.get("pass", False)):
                passed_razors += 1
        razor_alignment = (passed_razors / num_razors * 100.0) if num_razors else 0.0

        testability_score = max(0.0, min(100.0, razor_alignment))
        effective_testability = testability_score
        fiction_bonus = 0.0
        if is_fictional:
            if logic_score >= 60:
                effective_testability = max(testability_score, 55.0)
                fiction_bonus = 8.0
            elif logic_score >= 45:
                effective_testability = max(testability_score, 45.0)
                fiction_bonus = 4.0

        word_count = len((argument_text or "").split())
        explicit_structure_signals = int(logic_vars["assumption_count"]) + int(logic_vars["if_then_count"])
        logic_presence_flag = bool(logic_vars.get("logic_presence_flag", False))
        short_deduction_hint = short_deduction_hint or {}
        short_deduction_active = bool(short_deduction_hint.get("is_short_deduction", False)) and word_count <= 80
        effective_logic_presence = logic_presence_flag or short_deduction_active
        coherence_lane_active = (
            effective_logic_presence
            and (
                int(logic_vars.get("if_then_count", 0)) >= 1
                or int(logic_vars.get("assumption_count", 0)) >= 1
                or int(logic_vars.get("connector_count", 0)) >= 2
                or float(logic_vars.get("science_reasoning_proxy", 0.0)) >= 55.0
                or float(logic_vars.get("inference_chain_stability", 0.0)) >= 55.0
                or float(logic_vars.get("assumption_boundary_clarity", 0.0)) >= 55.0
                or (
                    word_count <= 30
                    and float(logic_vars.get("logic_integrity_score", 0.0)) >= 35.0
                    and float(logic_vars.get("contradiction_index", 0.0)) < 60.0
                )
            )
        )
        logic_score = max(0.0, min(100.0, logic_score))

        # Smooth short-text logic damping (no hard score cap):
        # very short + no structural cues should reduce logic contribution.
        if word_count < 12 and explicit_structure_signals == 0 and not coherence_lane_active:
            logic_score *= 0.60
        elif word_count < 20 and explicit_structure_signals <= 1 and not coherence_lane_active:
            logic_score *= 0.82

        # Linear penalties / rewards around a stable base.
        bias_penalty_gain = float(os.getenv("BIAS_PENALTY_GAIN", "0.38"))
        low_logic_penalty_gain = float(os.getenv("LOW_LOGIC_PENALTY_GAIN", "0.48"))
        low_testability_penalty_gain = float(os.getenv("LOW_TESTABILITY_PENALTY_GAIN", "0.28"))
        logic_reward_gain = float(os.getenv("LOGIC_REWARD_GAIN", "0.34"))
        testability_reward_gain = float(os.getenv("TESTABILITY_REWARD_GAIN", "0.08"))

        bias_penalty = bias_penalty_gain * bias_level_score
        low_logic_penalty = max(0.0, 55.0 - logic_score) * low_logic_penalty_gain
        low_testability_penalty = max(0.0, 45.0 - effective_testability) * low_testability_penalty_gain
        if coherence_lane_active:
            # Coherent short formal logic should not be over-penalized for low external testability.
            low_testability_penalty *= 0.55
        if is_fictional and coherence_lane_active:
            # Fictional lane emphasizes internal consistency over empirical falsifiability.
            low_testability_penalty *= 0.45

        short_vacuity_penalty = 0.0
        if word_count < 8 and logic_score < 60 and not coherence_lane_active:
            short_vacuity_penalty = 14.0
        elif word_count < 15 and logic_score < 60 and not coherence_lane_active:
            short_vacuity_penalty = 8.0
        elif word_count < 30 and logic_score < 45 and not coherence_lane_active:
            short_vacuity_penalty = 4.0

        substance_adjustment = short_vacuity_penalty

        # Binary logic-absence penalty:
        # apply -25 if logic is absent, otherwise leave untouched.
        if effective_logic_presence:
            logic_absence_penalty = 0.0
        else:
            if word_count < 20:
                logic_absence_penalty = 25.0
            elif word_count < 60:
                logic_absence_penalty = 12.0
            else:
                logic_absence_penalty = 6.0

        razor_reward_scaled = 0.40 * total_reward_capped
        testability_reward = max(0.0, effective_testability - 60.0) * testability_reward_gain
        logic_boost = max(0.0, logic_score - 45.0) * logic_reward_gain
        if coherence_lane_active and word_count < 35:
            logic_boost += max(0.0, logic_score - 35.0) * 0.14
        # Meta-hint contributors for science/logical rigor, boosted by 10%.
        max_meta_hint_bonus = float(os.getenv("MAX_META_HINT_BONUS", "13.2"))
        meta_hint_bonus = 0.0
        if logic_presence_flag:
            meta_hint_bonus += 2.2
        if int(logic_vars.get("if_then_count", 0)) >= 2:
            meta_hint_bonus += 2.75
        if float(logic_vars.get("assumption_boundary_clarity", 0.0)) >= 60:
            meta_hint_bonus += 2.75
        if float(logic_vars.get("inference_chain_stability", 0.0)) >= 60:
            meta_hint_bonus += 3.3
        if float(logic_vars.get("science_reasoning_proxy", 0.0)) >= 60:
            meta_hint_bonus += 1.8
        if coherence_lane_active and word_count < 35:
            meta_hint_bonus += 1.65
        meta_hint_bonus = min(meta_hint_bonus, max_meta_hint_bonus)

        science_logic_bonus = 0.0
        if not is_fictional:
            science_proxy = float(logic_vars.get("science_reasoning_proxy", 0.0))
            connector_score = float(logic_vars.get("connector_score", 0.0))
            contradiction_idx = float(logic_vars.get("contradiction_index", 0.0))
            if science_proxy >= 60 and contradiction_idx < 40:
                science_logic_bonus = min(
                    8.0,
                    max(0.0, (science_proxy - 55.0) * 0.198)
                    + max(0.0, (connector_score - 35.0) * 0.066),
                )

        deduction_bonus = 0.0
        deduction_logic_uplift = 0.0
        if short_deduction_active:
            strength = str(short_deduction_hint.get("deduction_strength", "none")).lower()
            deduction_bonus = {
                "none": 0.0,
                "weak": 2.0,
                "moderate": 4.0,
                "strong": 6.0,
            }.get(strength, 0.0)
            # Minimal UI alignment: concise valid deduction should raise displayed logic.
            deduction_logic_uplift = {
                "none": 0.0,
                "weak": 5.0,
                "moderate": 9.0,
                "strong": 13.0,
            }.get(strength, 0.0)

        fiction_lane_bonus = 0.0
        if is_fictional:
            if coherence_lane_active and logic_score >= 65 and float(logic_vars.get("contradiction_index", 0.0)) < 35:
                fiction_lane_bonus += 8.0
            elif coherence_lane_active and logic_score >= 55:
                fiction_lane_bonus += 4.0

        calibration_bonus = (
            razor_reward_scaled
            + testability_reward
            + logic_boost
            + meta_hint_bonus
            + science_logic_bonus
            + deduction_bonus
            + fiction_bonus
            + fiction_lane_bonus
        )

        final_score = (
            base_score
            - bias_penalty
            - low_logic_penalty
            - low_testability_penalty
            - short_vacuity_penalty
            - logic_absence_penalty
            + calibration_bonus
        )

        # Guardrails: weak logic integrity should significantly limit the score.
        guardrail_triggered = (
            logic_vars["logic_integrity_score"] < 30
            or logic_vars["contradiction_index"] >= 65
            or logic_vars["overclaim_penalty"] >= 70
        )
        if guardrail_triggered:
            final_score = min(final_score, 50.0)
        if effective_testability < 20 and logic_vars["logic_integrity_score"] < 55:
            final_score = min(final_score, 50.0)

        catastrophic_case = (
            issue_count >= 8
            and logic_vars["logic_integrity_score"] < 25
            and effective_testability < 25
        )
        if not catastrophic_case:
            final_score = max(3.0, min(97.0, final_score))

        final_score = max(0.0, min(100.0, final_score))

        # Effective logic for UI: keep raw logic for diagnostics, but show users
        # a logic reading aligned with short-deduction/science coherence hints.
        effective_logic_for_ui = min(
            100.0,
            max(
                0.0,
                logic_vars["logic_integrity_score"]
                + deduction_logic_uplift
                + (science_logic_bonus * 0.9)
                + (meta_hint_bonus * 0.35),
            ),
        )

        # Three-axis interpretation using low/mid/high with low threshold at 30.
        def _zone(score_value):
            if score_value < 30:
                return "low"
            if score_value >= 70:
                return "high"
            return "mid"

        bias_zone = _zone(bias_level_score)
        testability_zone = _zone(effective_testability)
        logic_zone = _zone(effective_logic_for_ui)

        bias_phrase = {
            "high": "Biased",
            "mid": "Some bias",
            "low": "Low bias"
        }[bias_zone]
        testability_phrase = {
            "high": "testable and explainable",
            "mid": "partly testable and explainable",
            "low": "not testable or explainable"
        }[testability_zone]
        logic_phrase = {
            "high": "well thought out",
            "mid": "partly well thought out",
            "low": "illogical/logic absent"
        }[logic_zone]

        status_label = f"{bias_phrase}, {testability_phrase}, {logic_phrase}"
        status_message = (
            f"Bias={round(bias_level_score, 1)} ({bias_zone}), "
            f"Testability={round(effective_testability, 1)} ({testability_zone}), "
            f"Logic={round(effective_logic_for_ui, 1)} ({logic_zone})."
        )
        interpretation_note = ""
        if (
            final_score >= 70
            and logic_vars["logic_integrity_score"] < 50
            and effective_testability >= 60
            and bias_level_score < 40
        ):
            interpretation_note = (
                "Interpretation note: this appears to be a high-quality claim, "
                "but the input includes limited explicit logical explanation "
                "(reasoning steps, assumptions, or connectors)."
            )
            status_message += " " + interpretation_note
        if guardrail_triggered:
            status_message += " Detected gaps in logical flow and in the steps leading to the conclusion."
        if is_fictional:
            status_message += " Input seems fictional, so internal coherence is used for analysis instead of external evidence."
        if sub_pen >= 40:
            status_message += " Input is very short, so confidence in interpretation is limited."

        return {
            "base_score": base_score,
            "penalties": penalties,
            "rewards": rewards,
            "total_penalty": round(total_penalty, 2),
            "total_reward": round(total_reward, 2),
            "total_reward_capped": round(total_reward_capped, 2),
            "raw_score": round(raw_score, 2),
            "legacy_final_score": round(legacy_final, 2),
            "final_score": int(round(final_score)),
            "mode_detected": mode,
            "mode_penalty_multiplier": mode_penalty_multiplier,
            "score_blend_weights": blend_weights,
            "dimension_weights": {
                "bias_weight": bias_penalty_gain,
                "testability_weight": low_testability_penalty_gain,
                "logic_weight": low_logic_penalty_gain
            },
            "dimension_scores": {
                "bias_score": round(bias_level_score, 2),
                "bias_quality_score": round(bias_quality_score, 2),
                "bias_issue_count": issue_count,
                "bias_count_boost": round(issue_count * bias_count_boost_unit, 2),
                "testability_score": round(effective_testability, 2),
                "logic_score": round(effective_logic_for_ui, 2),
                "logic_score_raw": round(logic_vars["logic_integrity_score"], 2),
                "bias_zone": bias_zone,
                "testability_zone": testability_zone,
                "logic_zone": logic_zone
            },
            "substance_adjustment": round(substance_adjustment, 2),
            "short_vacuity_penalty": round(short_vacuity_penalty, 2),
            "logic_absence_penalty": round(logic_absence_penalty, 2),
            "calibration_bonus": round(calibration_bonus, 2),
            "testability_reward": round(testability_reward, 2),
            "logic_boost": round(logic_boost, 2),
            "meta_hint_bonus": round(meta_hint_bonus, 2),
            "science_logic_bonus": round(science_logic_bonus, 2),
            "deduction_bonus": round(deduction_bonus, 2),
            "deduction_logic_uplift": round(deduction_logic_uplift, 2),
            "short_deduction_active": short_deduction_active,
            "coherence_lane_active": coherence_lane_active,
            "fiction_lane_bonus": round(fiction_lane_bonus, 2),
            "logic_guardrail_triggered": guardrail_triggered,
            "logic_variables": logic_vars,
            "logic_integrity_score_ui": round(effective_logic_for_ui, 2),
            "logic_integrity_score": logic_vars["logic_integrity_score"],
            "evidence_dependency_score": logic_vars["evidence_dependency_score"],
            "razor_alignment": round(razor_alignment, 2),
            "status_label": status_label,
            "status_message": status_message,
            "interpretation_note": interpretation_note
        }

    def _calculate_score(self, detected_issues, argument_text="", metadata=None, short_deduction_hint=None):
        """
        Calculate final argument strength score.
        """
        artifacts = self._compute_score_artifacts(detected_issues, argument_text, metadata, short_deduction_hint)
        return artifacts["final_score"]
    
    def _get_score_breakdown(self, detected_issues, argument_text="", metadata=None, short_deduction_hint=None):
        """Get detailed breakdown of score calculation"""
        return self._compute_score_artifacts(detected_issues, argument_text, metadata, short_deduction_hint)

