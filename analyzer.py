import json
import os
import re
import math
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ArgumentAnalyzer:
    def __init__(self):
        """Initialize the analyzer with definitions and OpenAI client"""
        timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90"))
        max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=max(5.0, timeout_seconds),
            max_retries=max(0, max_retries),
        )
        self.definitions = self._load_definitions()
        self.hint_weight_overrides, self.hint_threshold_overrides = self._load_hint_overrides()
        self._definition_profiles = None
        self._hint_keys = None
        # Extraction mode switches
        self.use_llm_metadata = os.getenv("USE_LLM_METADATA", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.use_ml_hints = os.getenv("USE_ML_HINTS", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.use_semantic_hints = os.getenv("USE_SEMANTIC_HINTS", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.use_llm_bias_patch = os.getenv("USE_LLM_BIAS_PATCH", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.llm_bias_model = os.getenv("LLM_BIAS_MODEL", "gpt-4o-mini")
        self.prompt_cache_key_base = (os.getenv("OPENAI_PROMPT_CACHE_KEY", "") or "").strip()
        retention = (os.getenv("OPENAI_PROMPT_CACHE_RETENTION", "in_memory") or "").strip().lower()
        self.prompt_cache_retention = retention if retention in {"in_memory", "24h"} else "in_memory"
        self.single_call_system_prompt = (
            "You are a strict hint extractor for argument analysis. "
            "Return only schema-valid JSON with normalized hint values from 0 to 1. "
            "Do not output direct fallacy, bias, or distortion labels."
        )
        # Display: show up to N per category in summary (all above threshold are stored)
        self.display_fallacies = int(os.getenv("DISPLAY_ISSUES_MAX", "5"))
        self.display_biases = int(os.getenv("DISPLAY_ISSUES_MAX", "5"))
        self.display_distortions = int(os.getenv("DISPLAY_ISSUES_MAX", "5"))
        self._last_usage = {}
        
    def _load_definitions(self):
        """Load definitions from JSON file"""
        try:
            with open('definitions.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Error: definitions.json not found")
            return {}

    def _load_definition_profiles(self):
        """Load hint_profile_012 for cosine prototype matching."""
        if self._definition_profiles is not None:
            return self._definition_profiles
        path = Path(__file__).resolve().parent / "phase1_artifacts" / "definitions_feature_profile.json"
        if not path.exists():
            self._definition_profiles = {}
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                self._definition_profiles = json.load(f)
        except Exception:
            self._definition_profiles = {}
        return self._definition_profiles

    def _hint_keys_for_profiles(self):
        if self._hint_keys is not None:
            return self._hint_keys
        path = Path(__file__).resolve().parent / "phase1_artifacts" / "hint_keys.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self._hint_keys = json.load(f)
        else:
            self._hint_keys = list(self._hint_schema().get("hints", {}).keys()) or []
        return self._hint_keys

    def _cosine_similarity_hints(self, hint_vec, prof_vec):
        """Cosine similarity between two 37-dim vectors. Returns 0-1."""
        if len(hint_vec) != len(prof_vec):
            return 0.0
        dot = sum(a * b for a, b in zip(hint_vec, prof_vec))
        na = math.sqrt(sum(a * a for a in hint_vec))
        nb = math.sqrt(sum(b * b for b in prof_vec))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        sim = dot / (na * nb)
        return self._clamp01((sim + 1.0) / 2.0)  # map [-1,1] to [0,1]

    def _load_hint_overrides(self):
        """
        Optional runtime overrides for hint weights/thresholds.
        JSON format:
        {
          "hint_weight_overrides": { "<category>": { "<issue_key>": { "<hint_key>": weight } } },
          "hint_threshold_overrides": { "<category>": 0.5 }
        }
        """
        path = (os.getenv("HINT_WEIGHT_OVERRIDES_FILE", "") or "").strip()
        if not path:
            return {}, {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return {}, {}

        weight_overrides = payload.get("hint_weight_overrides", {})
        threshold_overrides = payload.get("hint_threshold_overrides", {})
        if not isinstance(weight_overrides, dict):
            weight_overrides = {}
        if not isinstance(threshold_overrides, dict):
            threshold_overrides = {}
        return weight_overrides, threshold_overrides

    def _cache_key(self, stage):
        """Build stable prompt-cache key per stage when configured."""
        if not self.prompt_cache_key_base:
            return None
        return f"{self.prompt_cache_key_base}:{stage}"

    def _chat_create(self, stage, **kwargs):
        """Centralized OpenAI chat call with prompt-caching controls."""
        cache_key = self._cache_key(stage)
        if cache_key:
            kwargs["prompt_cache_key"] = cache_key
        kwargs["prompt_cache_retention"] = self.prompt_cache_retention
        response = self.client.chat.completions.create(**kwargs)
        self._record_usage(stage, response)
        return response

    def _usage_get(self, obj, key, default=0):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _record_usage(self, stage, response):
        """Store usage + cache-hit metrics per stage."""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(self._usage_get(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(self._usage_get(usage, "completion_tokens", 0) or 0)
        details = self._usage_get(usage, "prompt_tokens_details", {})
        cached_tokens = int(self._usage_get(details, "cached_tokens", 0) or 0)
        self._last_usage[stage] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "cache_hit_ratio": round((cached_tokens / prompt_tokens), 4) if prompt_tokens > 0 else 0.0,
        }

    def _get_usage_summary(self):
        stages = dict(self._last_usage)
        totals = {
            "prompt_tokens": sum(v.get("prompt_tokens", 0) for v in stages.values()),
            "completion_tokens": sum(v.get("completion_tokens", 0) for v in stages.values()),
            "cached_tokens": sum(v.get("cached_tokens", 0) for v in stages.values()),
        }
        totals["cache_hit_ratio"] = (
            round((totals["cached_tokens"] / totals["prompt_tokens"]), 4)
            if totals["prompt_tokens"] > 0
            else 0.0
        )
        return {"stages": stages, "totals": totals}

    @staticmethod
    def _clamp01(value):
        return max(0.0, min(1.0, float(value or 0.0)))

    @staticmethod
    def _signal_to_unit(value):
        """
        Normalize hint signal to 0..1.
        Accepts 0/1/2 as preferred discrete strength levels.
        """
        try:
            v = float(value)
        except Exception:
            return 0.0
        if v <= 0.0:
            return 0.0
        if v >= 2.0:
            return 1.0
        if v > 1.0:
            return v / 2.0
        return v

    def _hint_schema(self):
        schema = self.definitions.get("hint_schema", {})
        if isinstance(schema, dict) and isinstance(schema.get("hints"), dict) and schema.get("hints"):
            return schema
        # Fallback so hint-based mode remains fully operational
        # even if definitions.json does not include hint_schema.
        fallback_keys = [
            "evidence_strength",
            "evidence_relevance",
            "falsifiability",
            "causal_overreach",
            "generalization_strength",
            "personal_attack",
            "emotional_load",
            "absolute_language",
            "intent_attribution",
            "missing_counterevidence",
            "popularity_appeal",
            "authority_dependence",
            "correlation_causation",
            "binary_framing",
            "speculation_level",
            "unfalsifiable_risk",
            "symmetry_forcing",
            "proportionality_assumption",
            "inferential_gap",
            "claim_specificity",
            "counterargument_quality",
            "scope_qualification",
            "sample_representativeness",
            "cherry_picking_risk",
            "novelty_tradition_appeal",
            "distraction_risk",
            "redefinition_defense",
        ]
        return {"version": 1, "hints": {k: {"type": "signal_0_2", "label": k.replace("_", " ")} for k in fallback_keys}}

    def _hint_profiles(self):
        return self.definitions.get("hint_weight_profiles", {})

    def _hint_weights_by_category(self):
        return self.definitions.get("hint_weights", {})

    def _fallback_hint_weights(self, category, key):
        """Fallback map used when definitions.json has no explicit hint weights."""
        token = (key or "").lower()
        base = {}
        if "ad_hominem" in token or "label" in token:
            base.update({"personal_attack": 1.0, "emotional_load": 0.55})
        if "strawman" in token:
            base.update({"redefinition_defense": 0.7, "counterargument_quality": -0.6})
        if "false_dilemma" in token or "all_or_nothing" in token:
            base.update({"binary_framing": 1.0, "scope_qualification": -0.6})
        if "slippery_slope" in token:
            base.update({"causal_overreach": 1.0, "speculation_level": 0.6})
        if "appeal_to_authority" in token:
            base.update({"authority_dependence": 1.0, "evidence_strength": -0.5})
        if "appeal_to_emotion" in token or "emotional_reasoning" in token:
            base.update({"emotional_load": 1.0, "evidence_strength": -0.55})
        if "hasty_generalization" in token or "overgeneralization" in token:
            base.update({"generalization_strength": 1.0, "sample_representativeness": -0.75})
        if "post_hoc" in token or "false_cause" in token:
            base.update({"correlation_causation": 1.0, "causal_overreach": 0.55})
        if "survivorship" in token:
            base.update({"cherry_picking_risk": 1.0, "sample_representativeness": -0.8})
        if "symmetry_impulse" in token:
            base.update({"symmetry_forcing": 1.0})
        if "proportionality_bias" in token:
            base.update({"proportionality_assumption": 1.0})
        if "russells_teapot" in token:
            base.update({"unfalsifiable_risk": 1.0, "falsifiability": -0.8})
        if not base:
            if category == "logical_fallacies":
                base = {"inferential_gap": 0.55, "evidence_strength": -0.35}
            elif category == "cognitive_biases":
                base = {"emotional_load": 0.45, "counterargument_quality": -0.35}
            else:
                base = {"absolute_language": 0.45, "scope_qualification": -0.4}
        return base

    def _resolve_issue_hint_weights(self, category, key):
        cat = self._hint_weights_by_category().get(category, {})
        raw = cat.get(key, {})
        if isinstance(raw, str):
            base = self._hint_profiles().get(raw, {})
        elif isinstance(raw, dict) and raw:
            base = raw
        else:
            base = self._fallback_hint_weights(category, key)

        ovr = (((self.hint_weight_overrides or {}).get(category, {}) or {}).get(key, {}) or {})
        if isinstance(ovr, dict) and ovr:
            merged = dict(base)
            for hk, w in ovr.items():
                try:
                    merged[hk] = float(w)
                except Exception:
                    continue
            return merged
        return base

    def _category_threshold(self, category):
        threshold_env = {
            "logical_fallacies": "HINT_FALLACY_THRESHOLD",
            "cognitive_biases": "HINT_BIAS_THRESHOLD",
            "cognitive_distortions": "HINT_DISTORTION_THRESHOLD",
        }
        default_map = {
            "logical_fallacies": 0.48,
            "cognitive_biases": 0.48,
            "cognitive_distortions": 0.48,
        }
        ovr = (self.hint_threshold_overrides or {}).get(category, None)
        if ovr is not None:
            try:
                return self._clamp01(float(ovr))
            except Exception:
                pass
        return self._clamp01(float(os.getenv(threshold_env.get(category, ""), str(default_map.get(category, 0.42)))))

    def _score_issue_from_hints(self, weights, hints):
        if not isinstance(weights, dict) or not weights:
            return 0.0, []
        weighted_sum = 0.0
        abs_sum = 0.0
        contributions = []
        for hint_key, weight in weights.items():
            try:
                w = float(weight)
            except Exception:
                continue
            v = self._clamp01(hints.get(hint_key, 0.0))
            weighted_sum += w * v
            abs_sum += abs(w)
            contributions.append((hint_key, round(w * v, 4), v, w))
        if abs_sum <= 0:
            return 0.0, contributions
        normalized = weighted_sum / abs_sum
        return self._clamp01((normalized + 1.0) / 2.0), contributions

    def _reason_from_contributions(self, contributions):
        """Reason text for UI; hint profile details removed per user request."""
        return ""

    def _priority_hints(self):
        """Two highly indicative hints per definition. When both strong, modest confidence boost."""
        return {
            ("logical_fallacies", "ad_hominem"): ("personal_attack", "emotional_load"),
            ("logical_fallacies", "strawman"): ("redefinition_defense", "distraction_risk"),
            ("logical_fallacies", "false_dilemma"): ("binary_framing", "scope_qualification"),
            ("logical_fallacies", "hasty_generalization"): ("generalization_strength", "sample_representativeness"),
            ("cognitive_biases", "confirmation_bias"): ("missing_counterevidence", "cherry_picking_risk"),
            ("cognitive_distortions", "overgeneralization"): ("generalization_strength", "scope_qualification"),
        }

    def _rank_category_from_hints(self, hints, category, llm_bonus_keys=None, llm_alpha=0.0, hints_012=None):
        defs = self.definitions.get(category, {})
        threshold = self._category_threshold(category)
        profiles = self._load_definition_profiles()
        hint_keys = self._hint_keys_for_profiles()
        profile_blend = self._clamp01(float(os.getenv("PROFILE_BLEND_ALPHA", "0.4")))
        priority_floor = float(os.getenv("PRIORITY_HINT_FLOOR", "0.25"))
        priority_boost = float(os.getenv("PRIORITY_HINT_BOOST", "0.06"))

        # Build hint vector (0-1): use hints directly; for profile use 0/0.5/1 from 0/1/2
        hint_vec = [self._clamp01(hints.get(k, 0.0)) for k in hint_keys]

        ranked = []
        for key in defs.keys():
            weights = self._resolve_issue_hint_weights(category, key)
            confidence, contributions = self._score_issue_from_hints(weights, hints)

            # Blend with cosine similarity to definition profile (prototype anchor)
            if profiles and key in profiles:
                prof = profiles[key].get("hint_profile_012", {})
                prof_vec = [float(prof.get(k, 0)) / 2.0 for k in hint_keys]
                cosine = self._cosine_similarity_hints(hint_vec, prof_vec)
                confidence = (1.0 - profile_blend) * confidence + profile_blend * cosine

            # Priority hints: when both strongly present and baseline above floor, modest boost
            pk = (category, key)
            if pk in self._priority_hints() and confidence >= priority_floor:
                h1, h2 = self._priority_hints()[pk]
                v1 = self._clamp01(hints.get(h1, 0.0))
                v2 = self._clamp01(hints.get(h2, 0.0))
                if v1 >= 0.5 and v2 >= 0.5:
                    confidence = min(1.0, confidence + priority_boost)

            if confidence < threshold:
                continue
            ranked.append(
                {
                    "key": key,
                    "reason": self._reason_from_contributions(contributions),
                    "confidence": round(confidence, 4),
                }
            )
        ranked.sort(key=lambda x: x["confidence"], reverse=True)
        return ranked

    def _apply_rank_adjustments(self, detected, llm_bonus_keys=None):
        """
        Rank-based adjustments (no extra LLM reliance):
        - Strawman: demote 2 ranks (heuristic: often over-detected by hint similarity)
        - LLM-detected items: promote 2 ranks (quick LLM judgment)
        """
        llm_bonus_keys = llm_bonus_keys or set()
        RANK_DELTA = 2

        def _move_in_list(lst, key_pred, delta):
            """Move item matching key_pred by delta positions. delta>0 = up, delta<0 = down."""
            idx = next((i for i, x in enumerate(lst) if key_pred(x.get("key"))), None)
            if idx is None:
                return
            new_idx = max(0, min(len(lst) - 1, idx - delta))  # -delta: up=lower index
            if new_idx == idx:
                return
            item = lst.pop(idx)
            new_idx = min(new_idx, len(lst))  # after pop, clamp for insert
            lst.insert(new_idx, item)

        # Strawman: demote 2 ranks in fallacies
        lf = detected.get("logical_fallacies", [])
        _move_in_list(lf, lambda k: k == "strawman", -RANK_DELTA)

        # LLM-detected: promote 2 ranks in each category
        for cat in ("logical_fallacies", "cognitive_biases", "cognitive_distortions"):
            lst = detected.get(cat, [])
            for (c, key) in llm_bonus_keys:
                if c != cat:
                    continue
                _move_in_list(lst, lambda k: k == key, RANK_DELTA)

    def _razors_from_hints_formula(self, hints):
        """
        Evaluate razors via per-razor hint formulas. Each razor has violation_hints;
        pass when sum of hint values (0/1/2) <= pass_threshold. Differentiates razors.
        """
        cfg_path = Path(__file__).resolve().parent / "phase1_artifacts" / "hint_scoring_config.json"
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
        formulas = cfg.get("razor_formulas", {})

        def _hint_val(k):
            v = hints.get(k, 0)
            if v in ("?", None):
                return 0.0
            if isinstance(v, (int, float)):
                if 0 <= v <= 2 and v == int(v):
                    return float(v)  # already 0/1/2
                if 0 <= v <= 1:
                    return v * 2.0  # map 0-1 to 0-2 for formula
                return float(max(0, min(2, v)))
            return 0.0

        out = []
        for key, defn in self.definitions.get("philosophical_razors", {}).items():
            rule = formulas.get(key, {})
            violation_hints = rule.get("violation_hints", [])
            threshold = float(rule.get("pass_threshold", 3))

            if violation_hints:
                total = sum(_hint_val(k) for k in violation_hints)
                passed = total <= threshold
                conf = 1.0 - (total / (len(violation_hints) * 2))  # 0-1, higher = better
                conf = max(0.0, min(1.0, conf))
                reason = "Argument aligns with this razor." if passed else "Argument does not clearly satisfy this razor."
            else:
                passed = False
                conf = 0.0
                reason = "No formula defined for this razor."
            out.append({
                "key": key,
                "pass": passed,
                "reason": reason,
                "confidence": round(conf, 4),
            })
        return out

    def _razors_from_hints_cosine(self, hints):
        """Legacy: cosine similarity. Use _razors_from_hints_formula for differentiated evaluation."""
        return self._razors_from_hints_formula(hints)

    def _deterministic_razors_from_hints(self, hints):
        """Use formula-based razor evaluation (differentiates razors by hint rules)."""
        return self._razors_from_hints_formula(hints)

    def _build_summary_from_detected(self, detected):
        """Build exec summary from issues above confidence threshold."""
        lf = detected.get("logical_fallacies", [])
        cb = detected.get("cognitive_biases", [])
        cd = detected.get("cognitive_distortions", [])

        def names(items):
            return [item.get("name", item.get("key", "").replace("_", " ")) for item in items if item.get("key")]

        lf_names = names(lf[:5])
        cb_names = names(cb[:5])
        cd_names = names(cd[:5])

        parts = []
        if lf_names:
            parts.append(f"Fallacies: {', '.join(lf_names)}")
        if cb_names:
            parts.append(f"Biases: {', '.join(cb_names)}")
        if cd_names:
            parts.append(f"Distortions: {', '.join(cd_names)}")

        if not parts:
            sentence = "Few issues detected above confidence threshold. Scoring from structural hints."
            bullets = [
                "Issues above the confidence threshold are listed below when detected.",
                "Razors indicate testability and falsifiability.",
            ]
            return sentence, bullets, sentence

        sentence = "Detected above threshold: " + "; ".join(parts) + "."
        total_lf = len(lf)
        total_cb = len(cb)
        total_cd = len(cd)
        bullets = [f"{total_lf} fallacies, {total_cb} biases, {total_cd} distortions above confidence threshold."]
        return sentence, bullets, sentence

    def _score_to_band(self, score, thresholds):
        """Map 0-100 score to low/mid/high band."""
        if score <= thresholds.get("low", 33):
            return "low"
        if score <= thresholds.get("mid", 66):
            return "mid"
        return "high"

    def _hint_score_breakdown(self, scores, detected_issues, metadata=None, argument_text=""):
        """Build score_breakdown for UI from hint-based scores. Includes 27-term interpretation."""
        strength = scores.get("argument_strength", 0)
        bias_score = scores.get("bias_score", 0)
        test_score = scores.get("testability_score", 0)
        logic_score = scores.get("logic_score", 50)

        if strength >= 70:
            status_label = "Strong"
            status_message = "Low bias, good testability, and solid logic integrity."
        elif strength >= 45:
            status_label = "Moderate"
            status_message = "Some bias or logic issues; testability and razors may need improvement."
        else:
            status_label = "Weak"
            status_message = "High bias pressure, low testability, or weak logic integrity."

        interp_path = Path(__file__).resolve().parent / "phase1_artifacts" / "interpretation_27.json"
        dimension_bands = {}
        interpretation_27 = status_message
        if interp_path.exists():
            with open(interp_path, encoding="utf-8") as f:
                interp_cfg = json.load(f)
            th = interp_cfg.get("thresholds", {})
            dimension_bands["bias"] = self._score_to_band(bias_score, th.get("bias", {}))
            dimension_bands["testability"] = self._score_to_band(test_score, th.get("testability", {}))
            dimension_bands["logic"] = self._score_to_band(logic_score, th.get("logic", {}))
            key = f"{dimension_bands['bias']}_{dimension_bands['testability']}_{dimension_bands['logic']}"
            interpretation_27 = interp_cfg.get("interpretations", {}).get(key, status_message)
        metadata = metadata or []
        txt = (argument_text or "").lower()
        if_then_count = len(re.findall(r"\b(if|then)\b", txt))
        assumption_count = sum(
            1 for c in metadata
            if "assum" in (c.get("claim_text") or "").lower() or "suppos" in (c.get("claim_text") or "").lower()
        )

        logic_variables = {
            "assumption_count": assumption_count,
            "if_then_count": if_then_count,
            "logic_break_signals": 0,
            "coherence_score": logic_score,
            "overclaim_penalty": max(0, 50 - logic_score),
            "assumption_boundary_clarity": logic_score,
            "evidence_grounding_score": logic_score,
            "evidence_relevance_score": logic_score,
            "counterargument_balance_score": logic_score,
            "scope_calibration_score": logic_score,
            "causal_coherence_score": logic_score,
            "incoherence_index": max(0, 100 - logic_score),
        }

        out = {
            "dimension_scores": {
                "bias_score": bias_score,
                "testability_score": test_score,
                "logic_score": logic_score,
            },
            "status_label": status_label,
            "status_message": status_message,
            "logic_variables": logic_variables,
            "source": "hint_based",
            "raw_score": strength,
            "logic_integrity_score": logic_score,
            "razor_alignment": scores.get("razor_alignment", 0),
        }
        if dimension_bands:
            out["dimension_bands"] = dimension_bands
        if interpretation_27:
            out["interpretation_27"] = interpretation_27
        return out

    def _llm_bias_classifier(self, argument_text):
        """Lightweight LLM call to classify top-level bias categories and specific biases. Returns empty dict on failure."""
        if not argument_text or len(argument_text.strip()) < 10:
            return {}
        prompt = f"""Given this argument, which bias categories apply? Return JSON only.
Argument:
\"\"\"{argument_text[:2000]}\"\"\"

Return format:
{{"top_categories": ["availability"|"representative"|"anchoring"], "specific_biases": ["overgeneralization"|"hasty_generalization"|"confirmation_bias"|"anchoring_bias"|"availability_heuristic"|"survivorship_bias"]}}
Only include items that clearly apply. Keep lists short (max 2-3 each). Use snake_case keys."""
        try:
            response = self._chat_create(
                "llm_bias_classifier",
                model=self.llm_bias_model,
                messages=[
                    {"role": "system", "content": "You classify arguments into cognitive bias categories. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=120,
            )
            raw = response.choices[0].message.content
            data = self._safe_json_loads(raw) if isinstance(raw, str) else {}
            if not isinstance(data, dict):
                return {}
            return {
                "top_categories": data.get("top_categories") or [],
                "specific_biases": data.get("specific_biases") or [],
            }
        except Exception:
            return {}

    def _llm_bias_to_definition_keys(self, llm_response):
        """Map LLM response to (category, key) pairs that receive confidence bonus."""
        if not llm_response:
            return set()
        out = set()
        cats = [c.lower().replace(" ", "_").strip() for c in (llm_response.get("top_categories") or [])]
        biases = [b.lower().replace(" ", "_").strip() for b in (llm_response.get("specific_biases") or [])]
        mapping = {
            "availability": ("cognitive_biases", "availability_heuristic"),
            "anchoring": ("cognitive_biases", "anchoring_bias"),
            "representative": ("cognitive_biases", "survivorship_bias"),
        }
        for c in cats:
            if c in mapping:
                out.add(mapping[c])
        bias_map = {
            "overgeneralization": ("cognitive_distortions", "overgeneralization"),
            "hasty_generalization": ("logical_fallacies", "hasty_generalization"),
            "confirmation_bias": ("cognitive_biases", "confirmation_bias"),
            "anchoring_bias": ("cognitive_biases", "anchoring_bias"),
            "availability_heuristic": ("cognitive_biases", "availability_heuristic"),
            "survivorship_bias": ("cognitive_biases", "survivorship_bias"),
        }
        for b in biases:
            if b in bias_map:
                out.add(bias_map[b])
        return out

    def _detected_from_hint_vector(self, hint_vector, llm_response=None, hints_for_razors=None):
        llm_bonus_keys = self._llm_bias_to_definition_keys(llm_response) if llm_response else set()
        razor_hints = hints_for_razors if hints_for_razors is not None else hint_vector
        detected = {
            "logical_fallacies": self._rank_category_from_hints(hint_vector, "logical_fallacies"),
            "cognitive_biases": self._rank_category_from_hints(hint_vector, "cognitive_biases"),
            "cognitive_distortions": self._rank_category_from_hints(hint_vector, "cognitive_distortions"),
            "philosophical_razors": self._razors_from_hints_formula(razor_hints),
        }
        self._apply_rank_adjustments(detected, llm_bonus_keys)
        s, b, m = self._build_summary_from_detected(detected)
        detected["executive_summary_sentence"] = s
        detected["executive_summary_bullets"] = b
        detected["summary"] = m
        return detected

    def _hint_vector_fast(self, argument_text, metadata=None):
        """Deterministic, cheap hint vector from metadata + text."""
        metadata = metadata or self._extract_metadata_fast(argument_text)
        total = max(1, len(metadata))

        def ratio(pred):
            return sum(1 for c in metadata if pred(c)) / float(total)

        txt = (argument_text or "").lower()
        causal_markers = len(re.findall(r"\b(because|therefore|thus|hence|causes|leads to|due to|since)\b", txt))
        absolute_markers = len(re.findall(r"\b(always|never|everyone|no one|all|none|must|cannot)\b", txt))

        hints = {
            "evidence_strength": 1.0 - ratio(lambda c: c.get("evidence_sufficiency") in ("weak", "none")),
            "evidence_relevance": 1.0 - ratio(lambda c: c.get("evidence_relevance") in ("low", "none")),
            "falsifiability": ratio(lambda c: c.get("is_falsifiable")),
            "testability": ratio(lambda c: c.get("verifiability") in ("easily", "with_effort")),
            "causal_overreach": min(1.0, ratio(lambda c: c.get("inferential_gap") == "large") + (0.05 * causal_markers)),
            "generalization_strength": ratio(lambda c: c.get("generalizes")),
            "personal_attack": ratio(lambda c: c.get("targets_person")),
            "emotional_load": ratio(lambda c: c.get("emotional_tone") not in ("neutral", None)),
            "absolute_language": min(1.0, ratio(lambda c: c.get("uses_absolute_language")) + (0.02 * absolute_markers)),
            "intent_attribution": ratio(lambda c: c.get("assumes_intent")),
            "missing_counterevidence": 1.0 - ratio(lambda c: c.get("acknowledges_counterargument")),
            "popularity_appeal": self._clamp01(0.8 if re.search(r"\b(everyone|most people|popular|widely)\b", txt) else 0.0),
            "authority_dependence": ratio(lambda c: c.get("cites_authority")),
            "correlation_causation": self._clamp01(ratio(lambda c: c.get("makes_causal_claim")) * 0.7),
            "binary_framing": self._clamp01(0.7 if re.search(r"\b(either|only two|no other option)\b", txt) else 0.0),
            "speculation_level": ratio(lambda c: c.get("speculation_level") in ("moderate", "high")),
            "unfalsifiable_risk": ratio(lambda c: not c.get("is_falsifiable")),
            "symmetry_forcing": ratio(lambda c: c.get("forces_balance")),
            "proportionality_assumption": ratio(lambda c: c.get("proportional_causation")),
            "inferential_gap": ratio(lambda c: c.get("inferential_gap") == "large"),
            "claim_specificity": ratio(lambda c: c.get("specificity") == "high"),
            "counterargument_quality": ratio(lambda c: c.get("acknowledges_counterargument")),
            "scope_qualification": ratio(lambda c: c.get("scope_qualified")),
            "sample_representativeness": ratio(
                lambda c: c.get("exemplar_type") in ("population_data", "representative_sample")
            ),
            "cherry_picking_risk": ratio(lambda c: c.get("exemplar_type") in ("famous_case", "anecdote")),
            "novelty_tradition_appeal": self._clamp01(0.7 if re.search(r"\b(tradition|always done|new and better)\b", txt) else 0.0),
            "distraction_risk": self._clamp01(0.6 if re.search(r"\b(by the way|anyway|unrelated)\b", txt) else 0.0),
            "redefinition_defense": self._clamp01(0.7 if re.search(r"\b(no true|real .* would)\b", txt) else 0.0),
        }
        return {k: round(self._clamp01(v), 4) for k, v in hints.items()}
    
    def analyze_argument(self, argument_text, include_improvements=False):
        """
        SLA single-call path:
        - one LLM call for hint extraction only
        - deterministic issue matching + scoring in Python
        """
        try:
            self._last_usage = {}
            word_count = len((argument_text or "").strip().split())
            if word_count <= 2:
                return {
                    "success": True,
                    "raw_analysis": "{}",
                    "detected_issues": {
                        "logical_fallacies": [],
                        "cognitive_biases": [],
                        "cognitive_distortions": [],
                        "philosophical_razors": [],
                        "executive_summary_sentence": "Insufficient input to analyze.",
                        "executive_summary_bullets": ["Insufficient input to analyze, argument is likely just a statement."],
                        "summary": "Insufficient input to analyze, argument is likely just a statement.",
                    },
                    "metadata": [],
                    "short_deduction_hint": {},
                    "score": 0,
                    "score_breakdown": {
                        "status_label": "Insufficient",
                        "status_message": "Insufficient input to analyze, argument is likely just a statement.",
                        "interpretation_27": "Insufficient input to analyze, argument is likely just a statement.",
                        "dimension_scores": {"bias_score": 0, "testability_score": 0, "logic_score": 0},
                        "raw_score": 0,
                        "logic_integrity_score": 0,
                        "razor_alignment": 0,
                    },
                    "improvements": [],
                    "improvements_pending": False,
                    "pipeline_mode": "insufficient_input",
                    "hint_values": {},
                    "hint_vector_012": {},
                    "hint_labels": {},
                    "logic_hint_keys": [],
                    "llm_usage": {},
                    "llm_bias_signal": None,
                }
            metadata = self._extract_metadata_fast(argument_text)
            short_deduction_hint = self._extract_short_deduction_hint_fast(argument_text)
            fast_hints = self._hint_vector_fast(argument_text, metadata)

            payload = {}
            if self.use_ml_hints:
                from semantic_hint_predictor import get_hint_vector, get_hint_vector_012
                hints = get_hint_vector(argument_text)
                hints_012 = get_hint_vector_012(argument_text)
                pipeline_mode = "ml_semantic"
                payload["hint_values"] = hints
            elif self.use_llm_metadata:
                include_bias = self.use_llm_bias_patch
                prompt = self._build_hints_only_prompt(argument_text, include_bias=include_bias)
                response = self._chat_create(
                    "single_call_hints",
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.single_call_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_schema", "json_schema": self._hints_only_schema(include_bias=include_bias)},
                    max_tokens=int(os.getenv("SLA_MAX_COMPLETION_TOKENS", "280")),
                )
                raw = response.choices[0].message.content
                payload = self._safe_json_loads(raw) if isinstance(raw, str) else {}
                llm_hints = self._sanitize_llm_hints(payload)
                blend = self._clamp01(float(os.getenv("HINT_BLEND_LLM", "0.7")))
                hints = self._blend_hints(llm_hints, fast_hints, blend)
                payload["hint_values"] = hints
                pipeline_mode = "single_call_hints_only"
                if include_bias and isinstance(payload, dict):
                    payload.setdefault("top_categories", [])
                    payload.setdefault("specific_biases", [])
            else:
                hints = fast_hints
                payload["hint_values"] = hints
                pipeline_mode = "fast_hints_only"

            if pipeline_mode != "ml_semantic":
                hints_012 = self._hints_to_012(hints)

            llm_bias_signal = {}
            if self.use_llm_bias_patch and argument_text:
                if pipeline_mode == "single_call_hints_only" and isinstance(payload, dict):
                    tc = payload.get("top_categories") or []
                    sb = payload.get("specific_biases") or []
                    if tc or sb:
                        llm_bias_signal = {"top_categories": tc, "specific_biases": sb}
                if not llm_bias_signal:
                    llm_bias_signal = self._llm_bias_classifier(argument_text)
            detected = self._detected_from_hint_vector(hints, llm_bias_signal if llm_bias_signal else None, hints_for_razors=hints_012)
            detected_issues = self._normalize_detected_issues(detected)
            # Rebuild exec summary from normalized list so it matches displayed order
            s, b, m = self._build_summary_from_detected(detected_issues)
            detected_issues["executive_summary_sentence"] = s
            detected_issues["executive_summary_bullets"] = b
            detected_issues["summary"] = m
            if pipeline_mode == "ml_semantic":
                from hint_based_scoring import compute_scores
                scores = compute_scores(hints_012, detected_issues, argument_text)
                score = scores["argument_strength"]
                score_breakdown = self._hint_score_breakdown(scores, detected_issues, metadata, argument_text)
            else:
                score = self._calculate_score(detected_issues, argument_text, metadata, short_deduction_hint)
                score_breakdown = self._get_score_breakdown(detected_issues, argument_text, metadata, short_deduction_hint)
            improvements = self._deterministic_improvements(detected_issues) if include_improvements else []

            payload["hint_values"] = hints
            hint_labels = self._resolve_hint_labels(hints_012)
            logic_hint_keys = self._get_logic_hint_keys()
            return {
                "success": True,
                "raw_analysis": json.dumps(payload, ensure_ascii=True),
                "detected_issues": detected_issues,
                "metadata": metadata,
                "short_deduction_hint": short_deduction_hint,
                "score": score,
                "score_breakdown": score_breakdown,
                "improvements": improvements,
                "improvements_pending": False,
                "pipeline_mode": pipeline_mode,
                "hint_values": hints,
                "hint_vector_012": hints_012,
                "hint_labels": hint_labels,
                "logic_hint_keys": logic_hint_keys,
                "llm_usage": self._get_usage_summary(),
                "llm_bias_signal": llm_bias_signal if llm_bias_signal else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_logic_hint_keys(self):
        """Logic-related hints only (for free-tool display)."""
        try:
            path = Path(__file__).resolve().parent / "phase1_artifacts" / "hint_scoring_config.json"
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            pos = cfg.get("logic_hints_positive", [])
            neg = cfg.get("logic_hints_negative", [])
            return list(dict.fromkeys(pos + neg))
        except Exception:
            return ["claim_specificity", "counterargument_quality", "scope_qualification",
                    "evidence_strength", "evidence_relevance", "falsifiability", "inferential_gap"]

    def _resolve_hint_labels(self, hints_012):
        """Resolve hint values to meaning labels from hint_docs.json."""
        try:
            path = Path(__file__).resolve().parent / "hint_docs.json"
            with open(path, encoding="utf-8") as f:
                docs = json.load(f)
        except Exception:
            return {}
        hints_cfg = docs.get("hints", {})
        out = {}
        for k, v in (hints_012 or {}).items():
            val = int(v) if v not in ("?", None) else 0
            val = max(0, min(2, val))
            h = hints_cfg.get(k, {})
            meaning = h.get(f"meaning_{val}", h.get("meaning_0", ""))
            out[k] = meaning or str(val)
        return out

    def _hints_to_012(self, hints):
        """Convert 0-1 float hints to 0/1/2 for non-ML paths."""
        out = {}
        for k, v in (hints or {}).items():
            v = self._clamp01(float(v) if v not in ("?", None) else 0)
            if v < 0.25:
                out[k] = 0
            elif v < 0.75:
                out[k] = 1
            else:
                out[k] = 2
        return out

    def _hints_only_schema(self, include_bias=False):
        hint_keys = sorted(self._hint_schema().get("hints", {}).keys())
        props = {
            "hint_values": {
                "type": "object",
                "properties": {k: {"type": "integer", "minimum": 0, "maximum": 2} for k in hint_keys},
                "required": hint_keys,
                "additionalProperties": False,
            },
            "summary_sentence": {"type": "string"},
        }
        if include_bias:
            props["top_categories"] = {
                "type": "array",
                "items": {"type": "string", "enum": ["availability", "representative", "anchoring"]},
                "description": "Top-level bias categories that apply",
            }
            props["specific_biases"] = {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific biases e.g. overgeneralization, hasty_generalization, confirmation_bias",
            }
        return {
            "name": "hints_only_argument_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": props,
                "required": ["hint_values", "summary_sentence"],
                "additionalProperties": False,
            },
        }

    def _build_hints_only_prompt(self, argument_text, include_bias=False):
        keys = ", ".join(sorted(self._hint_schema().get("hints", {}).keys()))
        bias_extra = ""
        if include_bias:
            bias_extra = """
- Also return top_categories (max 2-3 from: availability, representative, anchoring) and specific_biases (max 2-3 from: overgeneralization, hasty_generalization, confirmation_bias, anchoring_bias, availability_heuristic, survivorship_bias). Use snake_case."""
        return f"""Extract normalized hint values for this argument.

Argument:
\"\"\"{argument_text}\"\"\"

Requirements:
- Return one value for each hint key using discrete signal levels:
  0 = absent, 1 = medium presence, 2 = strong presence.
- Be conservative and avoid inflated values.
- Do not return any fallacy, bias, or distortion labels.{bias_extra}

Hint keys:
{keys}
"""

    def _sanitize_llm_hints(self, payload):
        if not isinstance(payload, dict):
            return {}
        hints = payload.get("hint_values", {})
        if not isinstance(hints, dict):
            return {}
        out = {}
        for k in self._hint_schema().get("hints", {}).keys():
            out[k] = self._clamp01(self._signal_to_unit(hints.get(k, 0)))
        return out

    def _blend_hints(self, llm_hints, fast_hints, blend):
        out = {}
        all_keys = set((llm_hints or {}).keys()) | set((fast_hints or {}).keys())
        for key in all_keys:
            lv = self._clamp01((llm_hints or {}).get(key, 0.0))
            fv = self._clamp01((fast_hints or {}).get(key, 0.0))
            out[key] = round((blend * lv) + ((1.0 - blend) * fv), 4)
        return out

    def _deterministic_improvements(self, detected_issues):
        tips = []
        for category in ("logical_fallacies", "cognitive_biases", "cognitive_distortions"):
            for item in (detected_issues.get(category) or [])[:3]:
                key = item.get("key")
                defn = self.definitions.get(category, {}).get(key, {})
                arr = defn.get("improvements", [])
                if arr:
                    tips.append(str(arr[0]))
        for item in (detected_issues.get("philosophical_razors") or []):
            if item.get("pass", False):
                continue
            key = item.get("key")
            tip = self.definitions.get("philosophical_razors", {}).get(key, {}).get("improvement")
            if tip:
                tips.append(str(tip))
            if len(tips) >= 5:
                break
        return tips[:5]

    @staticmethod
    def _safe_json_loads(raw_text):
        """Parse strict JSON with a light markdown-fence fallback."""
        if isinstance(raw_text, dict):
            return raw_text
        text = (raw_text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                try:
                    return json.loads(text.strip())
                except Exception:
                    return {}
        return {}


    def _extract_metadata_fast(self, argument_text):
        """
        Deterministic extractor for obvious signals only.
        Non-obvious claim quality variables are enriched by a single LLM call
        (with graceful fallback to deterministic defaults on timeout/failure).
        """
        text = (argument_text or "").strip()
        if not text:
            return []

        claim_candidates = [c.strip() for c in re.split(r'(?<=[\.\!\?])\s+|\n+', text) if c.strip()]
        claims = claim_candidates[:14] if claim_candidates else [text[:240]]

        evidence_pattern = r"\b(study|studies|research|report|data|evidence|survey|statistics|trial|meta-analysis|paper|journal|dataset)\b"
        anecdote_pattern = r"\b(i|my|me|we|our|friend|family|personally|in my experience|someone i know)\b"
        authority_pattern = r"\b(expert|scientist|doctor|professor|institution|university|organization|according to)\b"
        causal_pattern = r"\b(because|therefore|thus|hence|as a result|resulting in|causes|leads to|due to|since)\b"
        absolute_pattern = r"\b(always|never|everyone|no one|all|none|must|cannot|impossible|definitely|certainly)\b"
        hedge_pattern = r"\b(might|may|could|possibly|perhaps|likely|probably|arguably|typically|often|sometimes)\b"
        measurable_pattern = r"\b(test|measure|verify|disprove|falsif|experiment|trial)\b"
        counter_pattern = r"\b(however|although|though|on the other hand|nevertheless|but)\b"
        qualifier_pattern = r"\b(in some cases|under certain conditions|typically|often|sometimes|unless|except)\b"

        metadata = []
        for claim in claims:
            c = claim.lower()
            cites_evidence = bool(re.search(evidence_pattern, c))
            has_anecdote = bool(re.search(anecdote_pattern, c))
            cites_authority = bool(re.search(authority_pattern, c))
            makes_causal_claim = bool(re.search(causal_pattern, c))
            uses_absolute = bool(re.search(absolute_pattern, c))
            has_hedges = bool(re.search(hedge_pattern, c))
            has_counter = bool(re.search(counter_pattern, c))
            has_qualifier = bool(re.search(qualifier_pattern, c))

            if cites_evidence and not has_anecdote:
                evidence_type = "scientific"
            elif has_anecdote:
                evidence_type = "anecdotal"
            else:
                evidence_type = "none"

            metadata.append({
                "claim_text": claim[:220],
                "cites_evidence": cites_evidence,
                "evidence_type": evidence_type,
                "cites_authority": cites_authority,
                "authority_named": "",
                "emotional_tone": "neutral",  # placeholder; refined by LLM enrich
                "makes_causal_claim": makes_causal_claim,
                "generalizes": bool(re.search(r"\b(all|everyone|people|society|humans)\b", c)),
                "uses_absolute_language": uses_absolute,
                "targets_person": bool(re.search(r"\b(idiot|stupid|ignorant|fool|liar)\b", c)),
                "assumes_intent": False,  # placeholder; refined by LLM enrich
                "is_falsifiable": bool(re.search(measurable_pattern, c)),
                "is_extraordinary": False,  # placeholder; refined by LLM enrich
                "exemplar_type": "none",  # placeholder; refined by LLM enrich
                "face_validity": "medium",  # placeholder; refined by LLM enrich
                "speculation_level": "low" if has_hedges else "moderate",
                "claim_type": "prediction" if re.search(r"\b(will|going to|future)\b", c) else "factual",
                "evidence_sufficiency": "none",  # placeholder; refined by LLM enrich
                "evidence_relevance": "none",  # placeholder; refined by LLM enrich
                "causal_chain_length": min(5, len(re.findall(causal_pattern, c))),
                "inferential_gap": "small",  # placeholder; refined by LLM enrich
                "specificity": "high" if re.search(r"\b\d+|percent|rate|sample|group|year\b", c) else "medium",
                "verifiability": "with_effort" if cites_evidence else "not_verifiable",
                "forces_balance": False,  # placeholder; refined by LLM enrich
                "proportional_causation": False,  # placeholder; refined by LLM enrich
                "acknowledges_counterargument": has_counter,
                "scope_qualified": has_qualifier or has_hedges,
            })

        return metadata

    def _extract_short_deduction_hint_fast(self, argument_text):
        """
        Deterministic short-input deduction signal to avoid an extra LLM call.
        """
        words = len((argument_text or "").split())
        if words == 0 or words > 80:
            return {
                "checked": False,
                "is_short_deduction": False,
                "deduction_strength": "none",
                "reason": ""
            }

        text = (argument_text or "").lower()
        has_if = bool(re.search(r"\bif\b", text))
        has_then = bool(re.search(r"\b(then|therefore|thus|hence)\b", text))
        has_conclusion = bool(re.search(r"\b(therefore|thus|hence)\b", text))
        has_contradiction = bool(("always" in text and "never" in text) or ("must" in text and "cannot" in text))

        is_short_deduction = has_if and (has_then or has_conclusion)
        if has_contradiction:
            strength = "weak"
        elif is_short_deduction and has_then and has_conclusion:
            strength = "strong"
        elif is_short_deduction:
            strength = "moderate"
        else:
            strength = "none"

        reason = "Deterministic short-form deduction signal based on condition/conclusion markers."
        return {
            "checked": True,
            "is_short_deduction": is_short_deduction,
            "deduction_strength": strength,
            "reason": reason[:240],
        }

    def _extract_short_deduction_hint(self, argument_text):
        """Deprecated LLM path removed; deterministic hint only."""
        return self._extract_short_deduction_hint_fast(argument_text)

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
        issue_confidence = float(os.getenv("ISSUE_CONFIDENCE", "0.64"))
        razor_confidence = float(os.getenv("RAZOR_CONFIDENCE", "0.30"))

        def _normalize_all(items, definition_group):
            """Accept valid items above confidence threshold only. No top-k; threshold gates detection."""
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

            # Rank by confidence × severity (not penalty alone): high-confidence serious issues first
            rank_by = os.getenv("RANK_BY", "confidence_x_severity").strip().lower()
            if rank_by == "penalty":
                normalized.sort(key=lambda x: abs(x["penalty"]), reverse=True)
            else:
                normalized.sort(key=lambda x: x["confidence"] * max(1, abs(x["penalty"])), reverse=True)
            return normalized

        biases = self._deduplicate_biases(_normalize_all(detected.get("cognitive_biases", []), "cognitive_biases"))

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
    
    def generate_improvements(self, argument_text, detected_issues):
        """Deterministic suggestions used by deferred improvement jobs."""
        return self._deterministic_improvements(detected_issues)

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
        llm_claim_count = metadata[0].get("__llm_claim_count") if metadata else None
        if isinstance(llm_claim_count, int) and llm_claim_count > 0:
            claim_count = llm_claim_count

        assumption_pattern = r"\b(assume|assuming|suppose|supposing|given that|let us assume|if we accept|granted)\b"
        boundary_pattern = r"\b(under this assumption|within this framework|if this holds|therefore|thus|hence|then)\b"
        if_pattern = r"\bif\b"
        then_pattern = r"\b(then|therefore|thus|hence|so)\b"
        connector_pattern = r"\b(because|therefore|thus|hence|so|implies|suggests|indicates|evidence shows|consistent with|as a result|given that|this means|which implies|follows that)\b"
        causal_marker_pattern = r"\b(because|therefore|thus|hence|as a result|resulting in|leads to|causes|due to|since|drives|produces|triggers)\b"
        evidence_marker_pattern = r"\b(study|studies|research|data|evidence|report|survey|statistics|meta-analysis|trial|paper|journal|dataset|sample|cohort|rct|longitudinal)\b"
        citation_pattern = r"(\(\s*(19|20)\d{2}\s*\)|\[[0-9]{1,3}\]|doi|et al\.)"
        counterargument_pattern = r"\b(however|although|though|on the other hand|nevertheless|nonetheless|yet|but)\b"
        qualifier_pattern = r"\b(in some cases|under certain conditions|typically|often|sometimes|in general|for example|for instance|except|unless)\b"
        hedge_pattern = r"\b(might|may|could|possibly|perhaps|likely|plausibly|probably|arguably)\b"
        absolute_pattern = r"\b(always|never|everyone|no one|all|none|must|cannot|impossible|certainly|definitely)\b"

        assumption_count = len(re.findall(assumption_pattern, lowered))
        boundary_markers = len(re.findall(boundary_pattern, lowered))
        if_then_count = len(re.findall(if_pattern, lowered))
        then_count = len(re.findall(then_pattern, lowered))
        connector_count = len(re.findall(connector_pattern, lowered))
        causal_marker_count = len(re.findall(causal_marker_pattern, lowered))
        llm_marker_hits = sum(1 for c in metadata if c.get("__llm_evidence_marker") is True)
        if llm_marker_hits > 0:
            evidence_marker_count = llm_marker_hits
        else:
            evidence_marker_count = len(re.findall(evidence_marker_pattern, lowered))
        citation_count = len(re.findall(citation_pattern, lowered))
        counterargument_markers = len(re.findall(counterargument_pattern, lowered))
        qualifier_markers = len(re.findall(qualifier_pattern, lowered))
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

        high_gap = sum(1 for c in metadata if c.get("inferential_gap") == "large")
        low_validity = sum(1 for c in metadata if c.get("face_validity") == "low")
        weak_evidence_claims = sum(1 for c in metadata if c.get("evidence_sufficiency") in ("weak", "none"))
        unsupported_link_ratio = self._safe_ratio(max(0, high_gap + low_validity), claim_count if claim_count else 1)

        logic_break_signals = 0
        if high_gap >= 2:
            logic_break_signals += 1
        if low_validity >= 2:
            logic_break_signals += 1
        if weak_evidence_claims >= 2 and high_gap >= 1:
            logic_break_signals += 1
        if if_then_count >= 1 and if_then_completeness < 45:
            logic_break_signals += 1

        incoherence_index = min(
            100.0,
            (logic_break_signals * 18.0)
            + (high_gap * 10.0)
            + (low_validity * 8.0)
            + (unsupported_link_ratio * 24.0),
        )
        coherence_score = max(0.0, min(100.0, 100.0 - incoherence_index))

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
        relevance_high = sum(1 for c in metadata if c.get("evidence_relevance") == "high")
        relevance_mid = sum(1 for c in metadata if c.get("evidence_relevance") == "medium")
        relevance_low = sum(1 for c in metadata if c.get("evidence_relevance") in ("low", "none"))
        falsifiable = sum(1 for c in metadata if c.get("is_falsifiable"))
        verifiable = sum(1 for c in metadata if c.get("verifiability") in ("easily", "with_effort"))
        small_gap = sum(1 for c in metadata if c.get("inferential_gap") in ("none", "small"))
        scope_qualified_claims = sum(1 for c in metadata if c.get("scope_qualified"))
        counterargument_claims = sum(1 for c in metadata if c.get("acknowledges_counterargument"))
        science_reasoning_proxy = (
            0.34 * (self._safe_ratio(strong_ev, claim_count if claim_count else 1) * 100.0)
            + 0.24 * (self._safe_ratio(falsifiable, claim_count if claim_count else 1) * 100.0)
            + 0.24 * (self._safe_ratio(verifiable, claim_count if claim_count else 1) * 100.0)
            + 0.18 * (self._safe_ratio(small_gap, claim_count if claim_count else 1) * 100.0)
        )
        science_reasoning_proxy = max(0.0, min(100.0, science_reasoning_proxy))

        evidence_grounding_score = (
            0.42 * min(100.0, evidence_marker_count * 8.0)
            + 0.18 * min(100.0, citation_count * 18.0)
            + 0.20 * (self._safe_ratio(strong_ev, claim_count if claim_count else 1) * 100.0)
            + 0.20 * (self._safe_ratio(verifiable, claim_count if claim_count else 1) * 100.0)
        )
        evidence_grounding_score = max(0.0, min(100.0, evidence_grounding_score))

        evidence_relevance_score = (
            0.55 * (self._safe_ratio(relevance_high, claim_count if claim_count else 1) * 100.0)
            + 0.25 * (self._safe_ratio(relevance_mid, claim_count if claim_count else 1) * 100.0)
            + 0.20 * max(0.0, 100.0 - (self._safe_ratio(relevance_low, claim_count if claim_count else 1) * 100.0))
        )
        evidence_relevance_score = max(0.0, min(100.0, evidence_relevance_score))

        counterargument_balance_score = (
            0.45 * min(100.0, counterargument_markers * 22.0)
            + 0.30 * (self._safe_ratio(counterargument_claims, claim_count if claim_count else 1) * 100.0)
            + 0.25 * min(100.0, qualifier_markers * 9.0)
        )
        counterargument_balance_score = max(0.0, min(100.0, counterargument_balance_score))

        scope_calibration_score = (
            0.35 * min(100.0, qualifier_markers * 9.0)
            + 0.30 * (self._safe_ratio(scope_qualified_claims, claim_count if claim_count else 1) * 100.0)
            + 0.20 * min(100.0, hedges * 10.0)
            + 0.15 * max(0.0, 100.0 - min(100.0, absolute_terms * 10.0))
        )
        scope_calibration_score = max(0.0, min(100.0, scope_calibration_score))

        causal_coherence_score = (
            0.38 * min(100.0, causal_marker_count * 8.0)
            + 0.36 * inference_chain_stability
            + 0.26 * max(0.0, 100.0 - min(100.0, high_gap * 22.0))
        )
        causal_coherence_score = max(0.0, min(100.0, causal_coherence_score))

        logic_integrity_score = (
            0.08 * assumption_quality
            + 0.14 * assumption_boundary_clarity
            + 0.11 * min(100.0, if_then_count * 20.0)
            + 0.09 * if_then_completeness
            + 0.12 * coherence_score
            + 0.11 * (100.0 - overclaim_penalty)
            + 0.06 * uncertainty_calibration
            + 0.07 * inference_chain_stability
            + 0.06 * connector_score
            + 0.06 * science_reasoning_proxy
            + 0.04 * evidence_grounding_score
            + 0.03 * evidence_relevance_score
            + 0.03 * counterargument_balance_score
            + 0.02 * scope_calibration_score
            + 0.01 * causal_coherence_score
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
            + 0.15 * evidence_relevance_score
            + 0.15 * (self._safe_ratio(falsifiable, claim_count if claim_count else 1) * 100.0)
            + 0.10 * (self._safe_ratio(verifiable, claim_count if claim_count else 1) * 100.0)
        )
        evidence_dependency_score = max(0.0, min(100.0, evidence_dependency_score))

        # Binary logic-presence flag derived from existing hints only.
        # True means at least minimal explicit logic structure is present.
        logic_presence_flag = bool(
            assumption_count >= 1
            or if_then_count >= 1
            or connector_count >= 2
            or science_reasoning_proxy >= 55
            or evidence_grounding_score >= 45
            or (
                word_count >= 35
                and inference_chain_stability >= 40
                and incoherence_index < 70
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
            "causal_marker_count": causal_marker_count,
            "evidence_marker_count": evidence_marker_count,
            "citation_count": citation_count,
            "counterargument_markers": counterargument_markers,
            "qualifier_markers": qualifier_markers,
            "logic_break_signals": logic_break_signals,
            "incoherence_index": round(incoherence_index, 2),
            "coherence_score": round(coherence_score, 2),
            "overclaim_penalty": round(overclaim_penalty, 2),
            "uncertainty_calibration": round(uncertainty_calibration, 2),
            "inference_chain_stability": round(inference_chain_stability, 2),
            "science_reasoning_proxy": round(science_reasoning_proxy, 2),
            "evidence_grounding_score": round(evidence_grounding_score, 2),
            "evidence_relevance_score": round(evidence_relevance_score, 2),
            "counterargument_balance_score": round(counterargument_balance_score, 2),
            "scope_calibration_score": round(scope_calibration_score, 2),
            "causal_coherence_score": round(causal_coherence_score, 2),
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
        claim_count = len(metadata)
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
                    and float(logic_vars.get("incoherence_index", 100.0)) < 60.0
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

        # Non-linear short-length penalty:
        # short text is not automatically penalized hard; penalty scales by quality signal.
        quality_signal = max(
            0.0,
            min(
                100.0,
                0.40 * (100.0 - bias_level_score)
                + 0.35 * effective_testability
                + 0.25 * logic_score,
            ),
        )
        short_vacuity_penalty = (
            20.0
            * math.exp(-word_count / 32.0)
            * max(0.0, (55.0 - quality_signal) / 55.0)
        )

        # Description/statement mode:
        # captures short neutral factual statements that provide information
        # but little explicit reasoning/context.
        factual_like = sum(
            1 for c in metadata
            if c.get("claim_type") in {"factual", "historical", "definition"}
        )
        opinion_like = sum(
            1 for c in metadata
            if c.get("claim_type") in {"opinion", "moral", "prediction"}
        )
        neutral_tone = sum(1 for c in metadata if c.get("emotional_tone") in {"neutral", None})
        factual_ratio = self._safe_ratio(factual_like, claim_count if claim_count else 1)
        opinion_ratio = self._safe_ratio(opinion_like, claim_count if claim_count else 1)
        neutral_ratio = self._safe_ratio(neutral_tone, claim_count if claim_count else 1)
        factual_observation_mode = bool(
            word_count <= 24
            and factual_ratio >= 0.6
            and neutral_ratio >= 0.6
            and opinion_ratio <= 0.35
            and bias_level_score <= 45.0
            and issue_count <= 1
            and float(logic_vars.get("overclaim_penalty", 0.0)) < 55.0
            and float(logic_vars.get("incoherence_index", 100.0)) < 55.0
            and not effective_logic_presence
        )

        # Keep penalization for logic-less statements, but avoid over-amplifying it.
        description_statement_penalty = 0.0
        if factual_observation_mode:
            description_statement_penalty = 8.0
            short_vacuity_penalty *= 1.05

        substance_adjustment = short_vacuity_penalty

        # Non-linear logic-absence penalty:
        # scales with length and quality, avoids abrupt cliff behavior.
        absence_base = 22.0 * math.exp(-word_count / 45.0)
        if effective_logic_presence:
            logic_absence_penalty = absence_base * (0.35 if logic_score < 35.0 else 0.0)
        else:
            logic_absence_penalty = absence_base * max(0.25, (60.0 - quality_signal) / 60.0)

        razor_reward_scaled = 0.40 * total_reward_capped
        testability_reward = max(0.0, effective_testability - 60.0) * testability_reward_gain
        logic_boost = max(0.0, logic_score - 45.0) * logic_reward_gain
        if coherence_lane_active and word_count < 35:
            logic_boost += max(0.0, logic_score - 35.0) * 0.14

        # Density-aware scaling:
        # reward sustained structure/evidence density, especially as length grows.
        density_den = max(1.0, word_count / 100.0)
        density_raw = (
            1.4 * float(logic_vars.get("connector_count", 0.0))
            + 1.3 * float(logic_vars.get("if_then_count", 0.0))
            + 1.2 * float(logic_vars.get("evidence_marker_count", 0.0))
            + 1.5 * float(logic_vars.get("citation_count", 0.0))
            + 1.0 * float(logic_vars.get("counterargument_markers", 0.0))
        ) / density_den
        density_score = max(0.0, min(100.0, density_raw * 12.0))

        # Non-linear bonus for maintaining quality at high word counts.
        length_complexity_factor = 1.0 - math.exp(-max(0.0, word_count - 120.0) / 140.0)
        high_length_quality_bonus = (
            length_complexity_factor
            * max(0.0, quality_signal - 58.0)
            * 0.22
        )
        high_length_density_bonus = (
            max(0.0, 1.0 - math.exp(-max(0.0, word_count - 80.0) / 130.0))
            * max(0.0, density_score - 45.0)
            * 0.09
        )
        # Meta-hint contributors for science/logical rigor, with bounded lift.
        max_meta_hint_bonus = float(os.getenv("MAX_META_HINT_BONUS", "16.0"))
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
        if float(logic_vars.get("evidence_grounding_score", 0.0)) >= 55:
            meta_hint_bonus += 1.6
        if float(logic_vars.get("evidence_relevance_score", 0.0)) >= 55:
            meta_hint_bonus += 1.3
        if float(logic_vars.get("counterargument_balance_score", 0.0)) >= 45:
            meta_hint_bonus += 1.1
        if float(logic_vars.get("scope_calibration_score", 0.0)) >= 50:
            meta_hint_bonus += 1.1
        if float(logic_vars.get("causal_coherence_score", 0.0)) >= 55:
            meta_hint_bonus += 1.2
        if coherence_lane_active and word_count < 35:
            meta_hint_bonus += 1.65
        meta_hint_bonus = min(meta_hint_bonus, max_meta_hint_bonus)

        science_logic_bonus = 0.0
        if not is_fictional:
            science_proxy = float(logic_vars.get("science_reasoning_proxy", 0.0))
            connector_score = float(logic_vars.get("connector_score", 0.0))
            incoherence_idx = float(logic_vars.get("incoherence_index", 100.0))
            if science_proxy >= 60 and incoherence_idx < 40:
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
            if coherence_lane_active and logic_score >= 65 and float(logic_vars.get("incoherence_index", 100.0)) < 35:
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
            + high_length_quality_bonus
            + high_length_density_bonus
        )

        final_score = (
            base_score
            - bias_penalty
            - low_logic_penalty
            - low_testability_penalty
            - short_vacuity_penalty
            - logic_absence_penalty
            - description_statement_penalty
            + calibration_bonus
        )

        # Guardrails: weak logic integrity should significantly limit the score.
        short_quality_exception = (
            word_count <= 30
            and effective_testability >= 60
            and bias_level_score <= 40
            and float(logic_vars.get("evidence_relevance_score", 0.0)) >= 55
        )
        guardrail_triggered = (
            (logic_vars["logic_integrity_score"] < 30 and not short_quality_exception)
            or logic_vars["incoherence_index"] >= 65
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
        if factual_observation_mode:
            status_message += " Input was treated as a description/statement with limited logic context."

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
            "description_statement_penalty": round(description_statement_penalty, 2),
            "quality_signal": round(quality_signal, 2),
            "factual_observation_mode": factual_observation_mode,
            "density_score": round(density_score, 2),
            "high_length_quality_bonus": round(high_length_quality_bonus, 2),
            "high_length_density_bonus": round(high_length_density_bonus, 2),
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

