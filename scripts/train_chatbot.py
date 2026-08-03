from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.nlp_service import clean_text




def train(intents_path: Path, output_path: Path) -> None:
    data = json.loads(intents_path.read_text(encoding="utf-8"))
    texts, labels = [], []
    for intent in data["intents"]:
        if intent["tag"] == "fallback":
            continue
        for pattern in intent["patterns"]:
            texts.append(clean_text(pattern) or pattern.lower())
            labels.append(intent["tag"])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    classifier.fit(matrix, labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "version": 1}, output_path)
    print(f"Saved chatbot model with {len(classifier.classes_)} intents to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intents", type=Path, default=ROOT / "data" / "intents.json")
    parser.add_argument("--output", type=Path, default=ROOT / "app" / "models" / "chatbot_model.pkl")
    args = parser.parse_args()
    train(args.intents, args.output)
