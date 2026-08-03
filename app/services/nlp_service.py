from __future__ import annotations

import re
import string
from pathlib import Path

import joblib
import numpy as np

from app.services.storage_service import StorageService


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "hers", "him", "his", "i", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "ours", "she", "so", "that", "the",
    "their", "them", "they", "this", "to", "was", "we", "were", "will", "with", "you",
    "your", "very"
}


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token not in STOPWORDS and len(token) > 1]
    return " ".join(tokens)


class NLPService:
    def __init__(self, models_dir: Path, storage: StorageService):
        self.models_dir = Path(models_dir)
        self.storage = storage
        self.vectorizer = joblib.load(self.models_dir / "vectorizer.pkl")
        self.model = joblib.load(self.models_dir / "sentiment_model.pkl")

    def analyze_sentiment(self, text: str) -> dict:
        cleaned = clean_text(text)
        if not cleaned:
            cleaned = text.lower().strip()
        features = self.vectorizer.transform([cleaned])
        probabilities = self.model.predict_proba(features)[0]
        best_index = int(np.argmax(probabilities))
        label = str(self.model.classes_[best_index])
        confidence = float(probabilities[best_index])
        self.storage.log_sentiment(text, label, confidence)
        return {
            "sentiment": label,
            "confidence": round(confidence, 4),
            "cleaned_text": cleaned,
        }
