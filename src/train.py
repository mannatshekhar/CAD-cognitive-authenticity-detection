"""
train.py — Train all CAD models: Rule-based, TF-IDF + LR, BERT fine-tune.

Usage:
    python src/train.py --model tfidf          # Train TF-IDF model (recommended for quick start)
    python src/train.py --model bert           # Fine-tune DistilBERT (requires GPU recommended)
    python src/train.py --model all            # Train both
    python src/train.py --model tfidf --data data/answers.csv
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_data, split_data, label_encode, label_decode,
    extract_linguistic_features, features_to_array
)
from src.model import RuleBasedCAD, TFIDFModel, BERTModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate(y_true, y_pred, y_prob=None, model_name="Model"):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    report = classification_report(y_true, y_pred, target_names=["Shallow", "Deep"])
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if auc:
        print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n{report}")

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    if auc:
        metrics["roc_auc"] = auc
    return metrics


def train_rule_based(train_df, test_df):
    logger.info("Training Rule-Based baseline...")
    model = RuleBasedCAD()
    y_true = label_encode(test_df["Label"])
    preds, probs = [], []
    for _, row in test_df.iterrows():
        label, prob = model.predict(row["Answer"])
        preds.append(1 if label == "Deep" else 0)
        probs.append(prob)
    metrics = evaluate(y_true, np.array(preds), np.array(probs), "Rule-Based Baseline")
    with open(os.path.join(RESULTS_DIR, "rule_based_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Rule-based evaluation saved.")
    return model


def train_tfidf(train_df, test_df):
    logger.info("Training TF-IDF + Logistic Regression model...")

    train_texts = train_df["Answer"].tolist()
    test_texts = test_df["Answer"].tolist()
    train_labels = label_encode(train_df["Label"])
    test_labels = label_encode(test_df["Label"])

    train_ling = [extract_linguistic_features(t) for t in train_texts]
    test_ling = [extract_linguistic_features(t) for t in test_texts]

    model = TFIDFModel(max_features=5000, ngram_range=(1, 2))
    model.fit(train_texts, train_labels, train_ling)

    # Cross-validation on training set
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    cv_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))
    ])
    cv_labels = label_encode(train_df["Label"])
    cv_scores = cross_val_score(cv_model, train_texts, cv_labels, cv=5, scoring="f1")
    logger.info(f"5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Test evaluation
    probs = [model.predict_proba_single(t, l) for t, l in zip(test_texts, test_ling)]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    metrics = evaluate(test_labels, np.array(preds), np.array(probs), "TF-IDF + Logistic Regression")
    metrics["cv_f1_mean"] = float(cv_scores.mean())
    metrics["cv_f1_std"] = float(cv_scores.std())

    model_path = "results/tfidf_model.pkl"
    model.save(model_path)
    with open(os.path.join(RESULTS_DIR, "tfidf_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"TF-IDF model saved to {model_path}")

    # Save example predictions
    examples = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        examples.append({
            "question": row["Question"],
            "answer": row["Answer"],
            "true_label": row["Label"],
            "predicted": "Deep" if preds[i] == 1 else "Shallow",
            "prob_deep": round(probs[i], 3),
        })
    with open(os.path.join(RESULTS_DIR, "tfidf_predictions.json"), "w") as f:
        json.dump(examples, f, indent=2)

    return model


def train_bert(train_df, test_df, epochs=3):
    logger.info("Fine-tuning DistilBERT...")
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch transformers")
        return None

    train_texts = train_df["Answer"].tolist()
    test_texts = test_df["Answer"].tolist()
    train_labels = label_encode(train_df["Label"])
    test_labels = label_encode(test_df["Label"])

    model = BERTModel(max_length=128)
    model.fit(train_texts, train_labels, epochs=epochs, batch_size=8, lr=2e-5)

    probs = [model.predict_proba_single(t) for t in test_texts]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    metrics = evaluate(test_labels, np.array(preds), np.array(probs), "DistilBERT Fine-tuned")

    bert_dir = "results/bert_model"
    model.save(bert_dir)
    with open(os.path.join(RESULTS_DIR, "bert_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"BERT model saved to {bert_dir}")
    return model


def plot_results():
    """Generate comparison bar chart if matplotlib available."""
    try:
        import matplotlib.pyplot as plt

        results = {}
        for name, fname in [("Rule-Based", "rule_based_metrics.json"),
                             ("TF-IDF+LR", "tfidf_metrics.json"),
                             ("DistilBERT", "bert_metrics.json")]:
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    results[name] = json.load(f)

        if not results:
            return

        models = list(results.keys())
        metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
        x = np.arange(len(models))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, metric in enumerate(metrics_to_plot):
            vals = [results[m].get(metric, 0) for m in models]
            ax.bar(x + i * width, vals, width, label=metric.capitalize())

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("CAD Model Comparison")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150)
        logger.info("Saved model_comparison.png to results/")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train CAD models")
    parser.add_argument("--model", choices=["rule", "tfidf", "bert", "all"], default="tfidf")
    parser.add_argument("--data", default="data/answers.csv")
    parser.add_argument("--epochs", type=int, default=3, help="BERT training epochs")
    args = parser.parse_args()

    df = load_data(args.data)
    train_df, test_df = split_data(df, test_size=0.2)
    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

    if args.model in ("rule", "all"):
        train_rule_based(train_df, test_df)

    if args.model in ("tfidf", "all"):
        train_tfidf(train_df, test_df)

    if args.model in ("bert", "all"):
        train_bert(train_df, test_df, epochs=args.epochs)

    plot_results()
    logger.info("Training complete. Results saved to results/")


if __name__ == "__main__":
    main()
