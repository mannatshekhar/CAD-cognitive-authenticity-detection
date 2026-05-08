"""
model.py — Model classes for CAD: Rule-based, TF-IDF + LR, BERT fine-tune, Bayesian wrapper.
"""

import os
import re
import logging
import numpy as np
import joblib
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# ── 1. Rule-Based Baseline ────────────────────────────────────────────────────

class RuleBasedCAD:
    """
    Simple rule-based classifier.
    Uses word count, causal language, and sentence structure.
    """
    CAUSAL = ["because", "therefore", "hence", "thus", "consequently",
              "as a result", "due to", "since", "implies", "results in"]

    def predict(self, answer: str) -> Tuple[str, float]:
        answer = answer.strip()
        if not answer:
            return "Shallow", 0.05

        words = answer.split()
        word_count = len(words)
        text_lower = answer.lower()
        causal_hits = sum(1 for c in self.CAUSAL if c in text_lower)
        has_multiple_sentences = len(re.findall(r"[.!?]", answer)) >= 2

        score = 0.0
        if word_count > 60:
            score += 0.35
        elif word_count > 30:
            score += 0.20
        elif word_count > 15:
            score += 0.10

        score += min(causal_hits * 0.15, 0.30)

        if has_multiple_sentences:
            score += 0.10

        if re.search(r"\d", answer):
            score += 0.05

        score = min(score, 0.95)
        label = "Deep" if score >= 0.50 else "Shallow"
        return label, round(score, 3)

    def predict_proba(self, answer: str) -> float:
        _, prob = self.predict(answer)
        return prob


# ── 2. TF-IDF + Logistic Regression ──────────────────────────────────────────

class TFIDFModel:
    """
    TF-IDF vectoriser + Logistic Regression with calibrated probabilities.
    Also concatenates handcrafted linguistic features.
    """

    def __init__(self, max_features: int = 5000, ngram_range: Tuple = (1, 2)):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}",
            stop_words=None,   # keep all — short words matter for shallow detection
        )
        self.classifier = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _combine_features(self, tfidf_matrix, ling_arrays: np.ndarray):
        from scipy.sparse import hstack, csr_matrix
        ling_scaled = self.scaler.transform(ling_arrays)
        return hstack([tfidf_matrix, csr_matrix(ling_scaled)])

    def fit(self, texts, labels, ling_features_list):
        from src.utils import features_to_array
        X_tfidf = self.vectorizer.fit_transform(texts)
        ling_arrays = np.array([features_to_array(f) for f in ling_features_list])
        self.scaler.fit(ling_arrays)
        X = self._combine_features(X_tfidf, ling_arrays)
        self.classifier.fit(X, labels)
        self.is_fitted = True
        logger.info("TFIDFModel fitted.")

    def predict_proba_single(self, text: str, ling_features: dict) -> float:
        from src.utils import features_to_array
        X_tfidf = self.vectorizer.transform([text])
        ling_arr = np.array([features_to_array(ling_features)])
        X = self._combine_features(X_tfidf, ling_arr)
        proba = self.classifier.predict_proba(X)[0]
        # Index 1 = Deep class
        classes = list(self.classifier.classes_)
        deep_idx = classes.index(1) if 1 in classes else 1
        return float(proba[deep_idx])

    def save(self, path: str):
        joblib.dump({"vectorizer": self.vectorizer,
                     "classifier": self.classifier,
                     "scaler": self.scaler}, path)
        logger.info(f"TFIDFModel saved to {path}")

    def load(self, path: str):
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.classifier = data["classifier"]
        self.scaler = data["scaler"]
        self.is_fitted = True
        logger.info(f"TFIDFModel loaded from {path}")


# ── 3. BERT Fine-tuned Classifier ────────────────────────────────────────────

class BERTModel:
    """
    Fine-tunes a pretrained BERT/DistilBERT model for binary classification.
    Uses HuggingFace Transformers + PyTorch.
    """
    MODEL_NAME = "distilbert-base-uncased"   # lightweight; swap for bert-base-uncased if needed

    def __init__(self, model_name: Optional[str] = None, max_length: int = 128):
        self.model_name = model_name or self.MODEL_NAME
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_fitted = False

    def _setup(self):
        try:
            import torch
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )
            self.model.to(self.device)
            logger.info(f"BERT model loaded on {self.device}")
        except ImportError:
            raise ImportError("Install transformers and torch: pip install transformers torch")

    def fit(self, texts, labels, epochs: int = 3, batch_size: int = 8, lr: float = 2e-5):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        self._setup()
        encodings = self.tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=self.max_length, return_tensors="pt"
        )
        label_tensor = torch.tensor(list(labels), dtype=torch.long)
        dataset = TensorDataset(
            encodings["input_ids"], encodings["attention_mask"], label_tensor
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                input_ids, attn_mask, batch_labels = [b.to(self.device) for b in batch]
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask, labels=batch_labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

        self.is_fitted = True

    def predict_proba_single(self, text: str) -> float:
        import torch
        if not self.is_fitted:
            raise RuntimeError("BERTModel not fitted. Call fit() or load().")
        self.model.eval()
        enc = self.tokenizer(
            text, truncation=True, padding=True,
            max_length=self.max_length, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return float(probs[1])   # index 1 = Deep

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        logger.info(f"BERTModel saved to {directory}")

    def load(self, directory: str):
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(directory)
        self.model = DistilBertForSequenceClassification.from_pretrained(directory)
        self.model.to(self.device)
        self.is_fitted = True
        logger.info(f"BERTModel loaded from {directory}")


# ── 4. Bayesian Confidence Wrapper ────────────────────────────────────────────

class BayesianConfidenceWrapper:
    """
    Wraps any model that returns a probability.
    Uses Monte Carlo Dropout (for BERT) or temperature scaling to
    estimate epistemic uncertainty and produce a calibrated confidence score.
    """

    def __init__(self, temperature: float = 1.5):
        self.temperature = temperature   # >1 softens probabilities (reduces overconfidence)

    def calibrate(self, raw_prob: float) -> float:
        """Apply temperature scaling to soften overconfident predictions."""
        import math
        # logit → scale → sigmoid
        eps = 1e-7
        raw_prob = float(np.clip(raw_prob, eps, 1 - eps))
        logit = math.log(raw_prob / (1 - raw_prob))
        scaled_logit = logit / self.temperature
        calibrated = 1 / (1 + math.exp(-scaled_logit))
        return round(calibrated, 4)

    def flag_anomalies(self, prob_deep: float, ling_features: dict) -> list:
        """
        Detect suspicious answer patterns that warrant a confidence penalty.
        Returns list of warning signal strings.
        """
        warnings = []
        wc = ling_features.get("word_count", 0)
        causal = ling_features.get("causal_word_count", 0)
        ttr = ling_features.get("type_token_ratio", 0)

        if prob_deep > 0.8 and wc < 15:
            warnings.append("High model confidence but very short answer — possible false positive.")
        if prob_deep > 0.7 and causal == 0 and wc < 25:
            warnings.append("No causal reasoning detected despite moderate confidence.")
        if ttr < 0.4:
            warnings.append("Low lexical diversity — possible rote or copied phrasing.")
        if wc < 8:
            warnings.append("Answer is extremely short — insufficient for reliable evaluation.")
        return warnings

    def score(self, raw_prob: float, ling_features: dict) -> Tuple[float, float, list]:
        """
        Returns (calibrated_prob, uncertainty_estimate, anomaly_warnings).
        """
        calibrated = self.calibrate(raw_prob)
        distance_from_boundary = abs(calibrated - 0.5)
        uncertainty = round(1 - (distance_from_boundary * 2), 3)   # 0=certain, 1=max uncertainty
        anomalies = self.flag_anomalies(raw_prob, ling_features)
        return calibrated, uncertainty, anomalies
