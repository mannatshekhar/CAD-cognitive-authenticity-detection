"""
app.py — Flask web application for CAD.

Run:
    python ui/app.py
    # or from project root:
    python -m ui.app

Then open: http://localhost:5000
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from src.infer import CADInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Load model once at startup ─────────────────────────────────────────────────
MODEL_TYPE = os.environ.get("CAD_MODEL", "tfidf")   # set env var to switch model

engine = None

def get_engine():
    global engine
    if engine is None:
        try:
            engine = CADInferenceEngine(model_type=MODEL_TYPE)
            logger.info(f"CAD engine ready ({MODEL_TYPE})")
        except FileNotFoundError as e:
            logger.warning(f"Trained model not found — falling back to rule-based. ({e})")
            engine = CADInferenceEngine(model_type="rule")
    return engine


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", model_type=MODEL_TYPE)


@app.route("/analyse", methods=["POST"])
def analyse():
    data = request.get_json()
    answer = (data.get("answer") or "").strip()
    question = (data.get("question") or "").strip()
    subject = data.get("subject", "General")
    grade = data.get("grade", "")
    time_spent = data.get("time_spent")
    edits = data.get("edits")

    if not answer:
        return jsonify({"error": "Answer text is required."}), 400

    try:
        eng = get_engine()
        result = eng.analyse(answer, question)

        # Incorporate behavioral features if provided
        if time_spent is not None:
            try:
                ts = float(time_spent)
                # Very fast answers (<5s for a short question) suggest guessing
                if ts < 5 and result["gus"] > 60:
                    result["signals"].append({
                        "type": "neutral",
                        "text": f"Answer submitted very quickly ({ts:.0f}s) — may warrant further verification."
                    })
                    result["gus"] = max(result["gus"] - 5, 0)
            except (ValueError, TypeError):
                pass

        if edits is not None:
            try:
                e = int(edits)
                if e >= 3:
                    result["signals"].append({
                        "type": "positive",
                        "text": f"Student revised their answer {e} time(s) — indicates reflective thinking."
                    })
            except (ValueError, TypeError):
                pass

        result["subject"] = subject
        result["grade"] = grade
        return jsonify(result)

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_TYPE})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
