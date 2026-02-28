import json
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List

_LOG_LOCK = Lock()


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(payload, ensure_ascii=True)
    with _LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _length_bucket(word_count: int) -> str:
    if word_count < 20:
        return "short"
    if word_count < 80:
        return "medium"
    return "long"


def _keys(items: List[Any]) -> List[str]:
    keys: List[str] = []
    for item in items or []:
        if isinstance(item, dict):
            key = item.get("key")
            if key:
                keys.append(str(key))
        elif isinstance(item, str) and item:
            keys.append(item)
    return keys


def log_analysis_artifact(argument_text: str, result: Dict[str, Any], analysis_id: str = "") -> None:
    score_breakdown = result.get("score_breakdown", {})
    dims = score_breakdown.get("dimension_scores", {})
    detected = result.get("detected_issues", {})
    metadata = result.get("metadata", [])
    razors = detected.get("philosophical_razors", [])

    passed_razors = [
        r.get("key")
        for r in razors
        if isinstance(r, dict) and bool(r.get("pass", False)) and r.get("key")
    ]
    failed_razors = [
        r.get("key")
        for r in razors
        if isinstance(r, dict) and (not bool(r.get("pass", False))) and r.get("key")
    ]
    word_count = len((argument_text or "").split())

    payload = {
        "schema_version": "analysis_artifact.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_id": analysis_id,
        "word_count": word_count,
        "length_bucket": _length_bucket(word_count),
        "mode_detected": score_breakdown.get("mode_detected", "unknown"),
        "score": result.get("score", 0),
        "status_label": score_breakdown.get("status_label", ""),
        "logic_guardrail_triggered": bool(score_breakdown.get("logic_guardrail_triggered", False)),
        "dimension_scores": {
            "bias_score": dims.get("bias_score", 0),
            "testability_score": dims.get("testability_score", 0),
            "logic_score": dims.get("logic_score", 0),
        },
        "issue_counts": {
            "logical_fallacies": len(detected.get("logical_fallacies", [])),
            "cognitive_biases": len(detected.get("cognitive_biases", [])),
            "cognitive_distortions": len(detected.get("cognitive_distortions", [])),
        },
        "issue_keys": {
            "logical_fallacies": _keys(detected.get("logical_fallacies", [])),
            "cognitive_biases": _keys(detected.get("cognitive_biases", [])),
            "cognitive_distortions": _keys(detected.get("cognitive_distortions", [])),
        },
        "razors": {
            "passed": passed_razors,
            "failed": failed_razors,
            "alignment_score": score_breakdown.get("razor_alignment", 0),
        },
        "metadata_summary": {
            "claim_count": len(metadata if isinstance(metadata, list) else []),
        },
    }
    path = os.getenv("ANALYSIS_ARTIFACT_LOG", os.path.join("logs", "analysis_artifacts.jsonl"))
    _append_jsonl(path, payload)


def log_feedback_event(payload: Dict[str, Any]) -> None:
    event = {
        "schema_version": "analysis_feedback.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_id": payload.get("analysis_id", ""),
        "score_feedback": payload.get("score_feedback", ""),
        "suggestion_feedback": payload.get("suggestion_feedback", ""),
        "report_persona_intent": payload.get("report_persona_intent", ""),
        "revised_argument": payload.get("revised_argument", ""),
    }
    path = os.getenv("ANALYSIS_FEEDBACK_LOG", os.path.join("logs", "analysis_feedback.jsonl"))
    _append_jsonl(path, event)
