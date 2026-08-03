from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np

from app.services.nlp_service import clean_text
from app.services.storage_service import StorageService


RULES: list[tuple[str, tuple[str, ...]]] = [
    ("privacy_face_recognition", ("face data", "face recognition", "facial data", "privacy", "consent")),
    ("human_agent", ("human agent", "real person", "customer care", "representative")),
    ("return_policy", ("return policy", "return an item", "send it back", "refund policy")),
    ("order_status", ("order status", "track my order", "where is my order", "delivery status")),
    ("store_hours", ("store hours", "opening time", "closing time", "when are you open")),
    ("greeting", ("hello", "hi", "hey", "good morning", "good evening")),
    ("goodbye", ("bye", "goodbye", "see you", "thanks bye")),
]


class ChatbotService:
    def __init__(
        self,
        models_dir: Path,
        intents_path: Path,
        storage: StorageService,
        confidence_threshold: float = 0.34,
    ):
        self.models_dir = Path(models_dir)
        self.storage = storage
        self.confidence_threshold = confidence_threshold
        self.bundle = joblib.load(self.models_dir / "chatbot_model.pkl")
        self.vectorizer = self.bundle["vectorizer"]
        self.classifier = self.bundle["classifier"]
        with Path(intents_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.intents = {intent["tag"]: intent for intent in payload["intents"]}

    def _response_for(self, intent: str, message: str) -> str:
        responses = self.intents.get(intent, self.intents["fallback"])["responses"]
        digest = hashlib.sha256(f"{intent}:{message}".encode("utf-8")).hexdigest()
        return responses[int(digest[:8], 16) % len(responses)]

    @staticmethod
    def _rule_intent(message: str) -> str | None:
        normalized = re.sub(r"\s+", " ", message.lower()).strip()
        for intent, phrases in RULES:
            if any(phrase in normalized for phrase in phrases):
                return intent
        return None

    def reply(self, message: str) -> dict:
        rule_intent = self._rule_intent(message)
        if rule_intent:
            intent, confidence, strategy = rule_intent, 1.0, "rule"
        else:
            cleaned = clean_text(message) or message.lower().strip()
            features = self.vectorizer.transform([cleaned])
            probabilities = self.classifier.predict_proba(features)[0]
            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])
            predicted = str(self.classifier.classes_[best_index])
            if confidence < self.confidence_threshold:
                intent, strategy = "fallback", "fallback"
            else:
                intent, strategy = predicted, "ml"
        response = self._response_for(intent, message)
        self.storage.log_chat(message, intent, confidence)
        return {
            "reply": response,
            "intent": intent,
            "confidence": round(confidence, 4),
            "strategy": strategy,
        }
