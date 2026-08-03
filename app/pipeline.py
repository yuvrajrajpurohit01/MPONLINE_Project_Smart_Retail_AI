from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.services.chatbot_service import ChatbotService
from app.services.cv_service import CVService
from app.services.nlp_service import NLPService
from app.services.storage_service import StorageService


class SmartRetailPipeline:
    """Load every model once and expose one integrated application pipeline."""

    def __init__(self):
        self.storage = StorageService(settings.database_path)
        self.cv = CVService(
            models_dir=settings.models_dir,
            storage=self.storage,
            face_match_threshold=settings.face_match_threshold,
        )
        self.nlp = NLPService(models_dir=settings.models_dir, storage=self.storage)
        self.chatbot = ChatbotService(
            models_dir=settings.models_dir,
            intents_path=settings.data_dir / "intents.json",
            storage=self.storage,
            confidence_threshold=settings.chatbot_confidence_threshold,
        )


@lru_cache(maxsize=1)
def get_pipeline() -> SmartRetailPipeline:
    return SmartRetailPipeline()
