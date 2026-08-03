from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.nlp_service import clean_text




def train(data_path: Path, models_dir: Path) -> None:
    frame = pd.read_csv(data_path).dropna(subset=["review_text", "sentiment"])
    frame["cleaned_text"] = frame["review_text"].astype(str).map(clean_text)
    x_train, x_test, y_train, y_test = train_test_split(
        frame["cleaned_text"],
        frame["sentiment"],
        test_size=0.20,
        stratify=frame["sentiment"],
        random_state=42,
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=10000)
    train_matrix = vectorizer.fit_transform(x_train)
    test_matrix = vectorizer.transform(x_test)
    model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    model.fit(train_matrix, y_train)
    print(classification_report(y_test, model.predict(test_matrix), digits=4))
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, models_dir / "vectorizer.pkl")
    joblib.dump(model, models_dir / "sentiment_model.pkl")
    print(f"Saved sentiment artifacts to {models_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "reviews.csv")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "app" / "models")
    args = parser.parse_args()
    train(args.data, args.models_dir)
