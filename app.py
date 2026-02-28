from datetime import timedelta
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from analyzer import ArgumentAnalyzer
from engagement import (
    engagement,
    init_db,
    save_analysis,
    get_analysis_record,
    update_analysis_improvements,
)
from prototype_logging import log_analysis_artifact, log_feedback_event
import json
import os
import threading

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("SESSION_SECRET_KEY", "change-me-in-production")
app.permanent_session_lifetime = timedelta(days=30)

app.register_blueprint(engagement)

init_db()

analyzer = ArgumentAnalyzer()
app.config["ARG_ANALYZER"] = analyzer

FREE_MAX_WORDS = 300
SUBSCRIBED_MAX_WORDS = 2000
DEFER_IMPROVEMENTS = os.getenv("DEFER_IMPROVEMENTS", "true").strip().lower() in {"1", "true", "yes", "on"}
_improvement_jobs = {}


def _run_improvement_job(analysis_id, argument_text, detected_issues):
    """Background task to compute improvements without blocking /api/analyze."""
    _improvement_jobs[analysis_id] = {"status": "running"}
    try:
        improvements = analyzer.generate_improvements(argument_text, detected_issues)
        update_analysis_improvements(analysis_id, improvements)
        _improvement_jobs[analysis_id] = {"status": "done", "improvements": improvements}
    except Exception as exc:
        _improvement_jobs[analysis_id] = {"status": "failed", "error": str(exc)}


def _start_improvement_job(analysis_id, argument_text, detected_issues):
    state = _improvement_jobs.get(analysis_id, {})
    if state.get("status") in {"pending", "running"}:
        return
    _improvement_jobs[analysis_id] = {"status": "pending"}
    worker = threading.Thread(
        target=_run_improvement_job,
        args=(analysis_id, argument_text, detected_issues),
        daemon=True,
    )
    worker.start()


@app.route('/')
def index():
    """Serve the main HTML page"""
    is_subscribed = session.get("subscribed", False)
    max_words = SUBSCRIBED_MAX_WORDS if is_subscribed else FREE_MAX_WORDS
    return render_template('index.html', max_words=max_words, is_subscribed=is_subscribed)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze an argument"""
    try:
        data = request.get_json()
        argument_text = data.get('argument', '')

        if not argument_text:
            return jsonify({
                "success": False,
                "error": "No argument provided"
            }), 400

        is_subscribed = session.get("subscribed", False)
        max_words = SUBSCRIBED_MAX_WORDS if is_subscribed else FREE_MAX_WORDS
        word_count = len(argument_text.split())
        if word_count > max_words:
            return jsonify({
                "success": False,
                "error": f"Argument is too long ({word_count} words). Please keep it under {max_words} words."
            }), 400

        result = analyzer.analyze_argument(argument_text, include_improvements=not DEFER_IMPROVEMENTS)

        if result.get("success"):
            analysis_id = save_analysis(argument_text, result)
            result["analysis_id"] = analysis_id
            if DEFER_IMPROVEMENTS and not result.get("improvements"):
                _start_improvement_job(analysis_id, argument_text, result.get("detected_issues", {}))
                result["improvements_pending"] = True
            try:
                log_analysis_artifact(argument_text, result, analysis_id)
            except Exception:
                # Telemetry must not break user-facing analysis.
                pass

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/analysis/<analysis_id>/improvements', methods=['GET'])
def analysis_improvements(analysis_id):
    """Return deferred improvements status and data."""
    row = get_analysis_record(analysis_id)
    if not row:
        return jsonify({"success": False, "error": "Analysis not found"}), 404

    stored_improvements = json.loads(row.get("improvements") or "[]")
    if stored_improvements:
        _improvement_jobs[analysis_id] = {"status": "done", "improvements": stored_improvements}
        return jsonify({
            "success": True,
            "status": "done",
            "improvements": stored_improvements,
        })

    state = _improvement_jobs.get(analysis_id, {})
    status = state.get("status", "pending")
    if status not in {"running", "pending"}:
        detected_issues = json.loads(row.get("detected_issues") or "{}")
        _start_improvement_job(analysis_id, row.get("argument_text", ""), detected_issues)
        status = "pending"

    return jsonify({
        "success": True,
        "status": status,
        "improvements": state.get("improvements", []),
        "error": state.get("error"),
    })


@app.route('/api/analysis-feedback', methods=['POST'])
def analysis_feedback():
    """Capture lightweight prototype feedback for future calibration/persona models."""
    data = request.get_json(silent=True) or {}
    analysis_id = (data.get("analysis_id") or "").strip()
    score_feedback = (data.get("score_feedback") or "").strip()
    suggestion_feedback = (data.get("suggestion_feedback") or "").strip()
    report_persona_intent = (data.get("report_persona_intent") or "").strip()
    revised_argument = (data.get("revised_argument") or "").strip()

    if not analysis_id:
        return jsonify({
            "success": False,
            "error": "analysis_id is required"
        }), 400

    if score_feedback and score_feedback not in {"too_low", "about_right", "too_high"}:
        return jsonify({
            "success": False,
            "error": "score_feedback must be one of: too_low, about_right, too_high"
        }), 400

    if suggestion_feedback and suggestion_feedback not in {"helpful", "mixed", "unhelpful"}:
        return jsonify({
            "success": False,
            "error": "suggestion_feedback must be one of: helpful, mixed, unhelpful"
        }), 400

    try:
        log_feedback_event({
            "analysis_id": analysis_id,
            "score_feedback": score_feedback,
            "suggestion_feedback": suggestion_feedback,
            "report_persona_intent": report_persona_intent,
            "revised_argument": revised_argument,
        })
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/definitions', methods=['GET'])
def get_definitions():
    """API endpoint to get all definitions"""
    try:
        return jsonify({
            "success": True,
            "definitions": analyzer.definitions
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
