# Model artifacts

This folder contains the artifacts expected by the project document:

- `product_classifier.h5` — a **functional synthetic demo classifier** so the API starts immediately. Train MobileNetV2 with `scripts/train_product_classifier.py` before reporting real accuracy.
- `face_db.pkl` — an empty face-embedding database. Add only consenting demo customers.
- `sentiment_model.pkl` — TF-IDF logistic-regression classifier.
- `vectorizer.pkl` — sentiment TF-IDF vectorizer.
- `chatbot_model.pkl` — TF-IDF intent classifier bundle.
- `model_metadata.json` — model classes, categories, generation date, and limitations.

Run `python scripts/bootstrap_models.py` to rebuild all lightweight artifacts.
