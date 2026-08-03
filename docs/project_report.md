# Major Project Report — Smart Retail & Customer Intelligence Platform

## 1. Overview

The system integrates computer vision, natural-language processing, a hybrid chatbot, model serialization, FastAPI serving, persistent analytics, testing, and Docker deployment in one capstone repository.

## 2. Modules

### Computer vision

`app/utils/cv_utils.py` contains grayscale conversion, resizing, Gaussian blur, Canny edge extraction, Haar-cascade face detection, face cropping, demo embeddings, and product feature extraction. `CVService` handles product classification, consent-based registration, returning-customer matching, and visit logging.

### NLP

`NLPService` cleans review text and uses TF-IDF with logistic regression to return Positive, Negative, or Neutral sentiment and a confidence score.

### Chatbot

`ChatbotService` uses rules for important FAQs and a TF-IDF intent classifier as the ML fallback. Responses are retrieved from `data/intents.json`.

### Unified API

`app/pipeline.py` loads models once. `app/main.py` exposes the documented endpoints, Swagger at `/docs`, health checks, CORS, Pydantic validation, and API-key security.

## 3. Evaluation plan

Report train/validation/test splits, class counts, macro F1, per-class precision and recall, confusion matrices, face false-match/false-non-match rates, API latency, Docker start-up, and automated test results. Do not report the synthetic product model as a real accuracy result.

## 4. Deployment

The repository provides Docker, Docker Compose, a Render blueprint, and a GitHub Actions workflow. Secrets must be configured in the deployment platform rather than committed.

## 5. Ethics and trade-offs

See `docs/ethics_privacy.md`. The main trade-off is a lightweight, explainable classroom baseline versus the accuracy and operational complexity of larger neural models.
