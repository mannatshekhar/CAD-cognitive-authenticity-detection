"""
utils.py — Data loading, preprocessing, and shared helpers for CAD.
"""

import re
import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CAUSAL_WORDS = [
    "because", "therefore", "hence", "thus", "consequently", "as a result",
    "due to", "which causes", "this leads to", "since", "so that",
    "in order to", "results in", "implies", "it follows that",
]
HEDGING_WORDS = [
    "perhaps", "might", "could", "may", "possibly", "probably",
    "approximately", "generally", "typically", "often",
]
TECHNICAL_CONNECTORS = [
    "however", "moreover", "furthermore", "in contrast", "on the other hand",
    "in addition", "specifically", "for example", "for instance",
    "such as", "that is", "in other words", "namely",
]


def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    required = {"Question", "Answer", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["Label"] = df["Label"].str.strip()
    df = df[df["Label"].isin(["Deep", "Shallow"])].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} samples | Deep: {(df['Label']=='Deep').sum()} | Shallow: {(df['Label']=='Shallow').sum()}")
    return df


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_linguistic_features(text: str) -> dict:
    text = preprocess_text(text)
    if not text:
        return _zero_features()
    words = text.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    avg_sentence_len = word_count / sentence_count
    unique_words = set(w.lower() for w in words)
    type_token_ratio = len(unique_words) / max(word_count, 1)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    text_lower = text.lower()
    causal_count = sum(1 for c in CAUSAL_WORDS if c in text_lower)
    hedge_count = sum(1 for h in HEDGING_WORDS if h in text_lower)
    connector_count = sum(1 for t in TECHNICAL_CONNECTORS if t in text_lower)
    has_numbers = int(bool(re.search(r"\d", text)))
    has_formula = int(bool(re.search(r"[a-zA-Z]\s*[=²³]|[+\-×÷/^]", text)))
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": round(avg_sentence_len, 2),
        "type_token_ratio": round(type_token_ratio, 4),
        "avg_word_len": round(avg_word_len, 2),
        "causal_word_count": causal_count,
        "hedge_word_count": hedge_count,
        "connector_count": connector_count,
        "has_numbers": has_numbers,
        "has_formula": has_formula,
    }


def _zero_features() -> dict:
    return {k: 0 for k in [
        "word_count", "sentence_count", "avg_sentence_len", "type_token_ratio",
        "avg_word_len", "causal_word_count", "hedge_word_count",
        "connector_count", "has_numbers", "has_formula",
    ]}


def features_to_array(feat_dict: dict) -> np.ndarray:
    keys = [
        "word_count", "sentence_count", "avg_sentence_len", "type_token_ratio",
        "avg_word_len", "causal_word_count", "hedge_word_count",
        "connector_count", "has_numbers", "has_formula",
    ]
    return np.array([feat_dict.get(k, 0) for k in keys], dtype=float)


def label_encode(labels: pd.Series) -> np.ndarray:
    return (labels == "Deep").astype(int).values


def label_decode(preds: np.ndarray) -> List[str]:
    return ["Deep" if p == 1 else "Shallow" for p in preds]


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["Label"], random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_gus(prob_deep: float, ling_features: dict) -> int:
    """
    Genuine Understanding Score (0-100).
    Combines model probability with linguistic signal adjustments.
    """
    base = prob_deep * 100
    causal_boost = min(ling_features.get("causal_word_count", 0) * 3, 10)
    connector_boost = min(ling_features.get("connector_count", 0) * 2, 6)
    formula_boost = 4 if ling_features.get("has_formula") else 0
    length_penalty = -15 if ling_features.get("word_count", 0) < 10 else 0
    ttr = ling_features.get("type_token_ratio", 0)
    ttr_boost = min((ttr - 0.5) * 20, 8) if ttr > 0.5 else 0
    gus = base + causal_boost + connector_boost + formula_boost + length_penalty + ttr_boost
    return int(np.clip(round(gus), 0, 100))


def verdict_from_gus(gus: int) -> str:
    if gus >= 70:
        return "Deep Understanding"
    elif gus >= 40:
        return "Moderate Understanding"
    else:
        return "Shallow Understanding"
