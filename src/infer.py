"""
infer.py — Load trained models and run inference for CAD.

Usage:
    python src/infer.py --answer "Photosynthesis converts light to glucose via the Calvin cycle."
    python src/infer.py --answer "Plants make food." --question "What is photosynthesis?"
    python src/infer.py --model bert --answer "Your answer here"
    python src/infer.py --csv data/answers.csv --output results/batch_predictions.csv
"""

import os
import sys
import json
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    extract_linguistic_features, compute_gus, verdict_from_gus, preprocess_text
)
from src.model import RuleBasedCAD, TFIDFModel, BERTModel, BayesianConfidenceWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CADInferenceEngine:
    """
    Main inference engine. Loads the best available model and runs the full
    CAD pipeline: preprocessing → features → model → Bayesian calibration → GUS.
    """

    def __init__(self, model_type: str = "tfidf"):
        self.model_type = model_type
        self.model = None
        self.bayesian = BayesianConfidenceWrapper(temperature=1.5)
        self._load_model()

    def _load_model(self):
        if self.model_type == "bert":
            bert_dir = "results/bert_model"
            if not os.path.exists(bert_dir):
                raise FileNotFoundError(
                    f"BERT model not found at {bert_dir}. Run: python src/train.py --model bert"
                )
            self.model = BERTModel()
            self.model.load(bert_dir)
            logger.info("Loaded BERT model.")

        elif self.model_type == "tfidf":
            model_path = "results/tfidf_model.pkl"
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"TF-IDF model not found at {model_path}. Run: python src/train.py --model tfidf"
                )
            self.model = TFIDFModel()
            self.model.load(model_path)
            logger.info("Loaded TF-IDF model.")

        elif self.model_type == "rule":
            self.model = RuleBasedCAD()
            logger.info("Using rule-based model (no training required).")

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def analyse(self, answer: str, question: str = "") -> dict:
        """
        Full CAD pipeline for a single answer.

        Returns a dict with:
            gus, verdict, prob_deep, calibrated_prob, uncertainty,
            linguistic_features, signals, recommendation
        """
        answer = preprocess_text(answer)
        if not answer:
            return {
                "gus": 0,
                "verdict": "Shallow Understanding",
                "prob_deep": 0.0,
                "calibrated_prob": 0.0,
                "uncertainty": 1.0,
                "linguistic_features": {},
                "signals": [{"type": "negative", "text": "Empty answer submitted."}],
                "recommendation": "No answer was provided. Please ask the student to attempt the question.",
            }

        # 1. Linguistic features
        ling = extract_linguistic_features(answer)

        # 2. Model probability
        if self.model_type == "bert":
            raw_prob = self.model.predict_proba_single(answer)
        elif self.model_type == "tfidf":
            raw_prob = self.model.predict_proba_single(answer, ling)
        else:
            raw_prob = self.model.predict_proba(answer)

        # 3. Bayesian calibration
        calibrated_prob, uncertainty, anomaly_warnings = self.bayesian.score(raw_prob, ling)

        # 4. GUS score
        gus = compute_gus(calibrated_prob, ling)
        verdict = verdict_from_gus(gus)

        # 5. Build signals
        signals = _build_signals(ling, calibrated_prob, anomaly_warnings)

        # 6. Recommendation
        recommendation = _build_recommendation(verdict, ling, anomaly_warnings, question)

        return {
            "gus": gus,
            "verdict": verdict,
            "prob_deep": round(raw_prob, 4),
            "calibrated_prob": round(calibrated_prob, 4),
            "uncertainty": round(uncertainty, 4),
            "semantic_depth": _semantic_depth_score(ling, calibrated_prob),
            "conceptual_links": _conceptual_links_score(ling),
            "lexical_richness": int(ling.get("type_token_ratio", 0) * 100),
            "confidence_score": int(calibrated_prob * 100),
            "linguistic_features": ling,
            "signals": signals,
            "recommendation": recommendation,
            "model_used": self.model_type,
        }

    def analyse_batch(self, records: list) -> list:
        """Analyse a list of dicts with 'Answer' (and optionally 'Question') keys."""
        results = []
        for r in records:
            result = self.analyse(r.get("Answer", ""), r.get("Question", ""))
            result["question"] = r.get("Question", "")
            result["answer"] = r.get("Answer", "")
            result["true_label"] = r.get("Label", "")
            results.append(result)
        return results


def _semantic_depth_score(ling: dict, prob: float) -> int:
    import numpy as np
    base = prob * 60
    causal = min(ling.get("causal_word_count", 0) * 5, 20)
    sentence = min(ling.get("sentence_count", 0) * 3, 15)
    formula = 5 if ling.get("has_formula") else 0
    return int(np.clip(base + causal + sentence + formula, 0, 100))


def _conceptual_links_score(ling: dict) -> int:
    import numpy as np
    connector = min(ling.get("connector_count", 0) * 8, 40)
    causal = min(ling.get("causal_word_count", 0) * 8, 40)
    hedge = min(ling.get("hedge_word_count", 0) * 4, 20)
    return int(np.clip(connector + causal + hedge, 0, 100))


def _build_signals(ling: dict, prob: float, anomalies: list) -> list:
    signals = []
    if ling.get("causal_word_count", 0) >= 2:
        signals.append({"type": "positive", "text": f"Uses causal reasoning language ({ling['causal_word_count']} instances)."})
    elif ling.get("causal_word_count", 0) == 0:
        signals.append({"type": "negative", "text": "No causal or explanatory language detected."})

    if ling.get("word_count", 0) > 50:
        signals.append({"type": "positive", "text": f"Detailed answer ({ling['word_count']} words)."})
    elif ling.get("word_count", 0) < 15:
        signals.append({"type": "negative", "text": f"Very short answer ({ling['word_count']} words)."})

    ttr = ling.get("type_token_ratio", 0)
    if ttr > 0.65:
        signals.append({"type": "positive", "text": f"High lexical diversity (TTR: {ttr:.2f}) — original phrasing."})
    elif ttr < 0.45:
        signals.append({"type": "negative", "text": f"Low lexical diversity (TTR: {ttr:.2f}) — possible rote phrasing."})

    if ling.get("connector_count", 0) >= 2:
        signals.append({"type": "positive", "text": "Uses discourse connectors — structured argument present."})

    if ling.get("has_formula"):
        signals.append({"type": "positive", "text": "Contains formulae or numerical specificity."})

    for a in anomalies:
        signals.append({"type": "neutral", "text": f"⚠ Confidence flag: {a}"})

    return signals[:6]   # cap at 6


def _build_recommendation(verdict: str, ling: dict, anomalies: list, question: str) -> str:
    if verdict == "Deep Understanding":
        base = "The student demonstrates strong conceptual understanding with clear explanatory depth."
        if ling.get("causal_word_count", 0) >= 2:
            base += " Their use of causal reasoning suggests genuine comprehension rather than memorisation."
        base += " No immediate intervention required — consider extending with a more challenging follow-up question."
    elif verdict == "Moderate Understanding":
        base = "The student shows partial understanding but lacks depth in certain areas."
        if ling.get("causal_word_count", 0) == 0:
            base += " Encourage them to explain the 'why' behind their statements, not just the 'what'."
        base += " Targeted follow-up questions or concept mapping exercises are recommended."
    else:
        base = "The student's answer appears superficial — likely based on recall rather than understanding."
        if ling.get("word_count", 0) < 15:
            base += " The answer is too brief to assess reasoning."
        base += " Consider one-on-one discussion, guided questioning, or worked examples to uncover and address gaps."
    if anomalies:
        base += " Note: the system flagged potential confidence issues — human review is advised."
    return base


def print_result(result: dict):
    print("\n" + "="*55)
    print(f"  CAD Analysis Result")
    print("="*55)
    print(f"  GUS Score : {result['gus']}/100")
    print(f"  Verdict   : {result['verdict']}")
    print(f"  Prob Deep : {result['prob_deep']:.3f}  (calibrated: {result['calibrated_prob']:.3f})")
    print(f"  Uncertainty: {result['uncertainty']:.3f}")
    print(f"\n  Sub-scores:")
    print(f"    Semantic Depth    : {result.get('semantic_depth', '—')}/100")
    print(f"    Conceptual Links  : {result.get('conceptual_links', '—')}/100")
    print(f"    Lexical Richness  : {result.get('lexical_richness', '—')}/100")
    print(f"\n  Signals:")
    for s in result.get("signals", []):
        icon = "✓" if s["type"] == "positive" else ("✗" if s["type"] == "negative" else "•")
        print(f"    {icon} {s['text']}")
    print(f"\n  Recommendation:")
    print(f"    {result['recommendation']}")
    print("="*55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run CAD inference")
    parser.add_argument("--model", choices=["rule", "tfidf", "bert"], default="tfidf")
    parser.add_argument("--answer", type=str, help="Single answer text to analyse")
    parser.add_argument("--question", type=str, default="", help="Question context (optional)")
    parser.add_argument("--csv", type=str, help="Batch CSV file path")
    parser.add_argument("--output", type=str, default="results/batch_predictions.csv")
    args = parser.parse_args()

    engine = CADInferenceEngine(model_type=args.model)

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        records = df.to_dict("records")
        results = engine.analyse_batch(records)
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output, index=False)
        logger.info(f"Batch results saved to {args.output}")
        # Print summary
        gusList = [r["gus"] for r in results]
        print(f"\nBatch Summary: {len(results)} answers | Avg GUS: {sum(gusList)/len(gusList):.1f}")
        verdicts = [r["verdict"] for r in results]
        for v in ["Deep Understanding", "Moderate Understanding", "Shallow Understanding"]:
            count = verdicts.count(v)
            print(f"  {v}: {count} ({100*count//len(verdicts)}%)")

    elif args.answer:
        result = engine.analyse(args.answer, args.question)
        print_result(result)
        with open("results/last_result.json", "w") as f:
            json.dump(result, f, indent=2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
