from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI-Powered Smart Retail & Customer Intelligence Platform"
    app_version: str = "1.0.0"
    api_key: str = os.getenv("SMART_RETAIL_API_KEY", "dev-secret-key")
    api_key_header: str = "X-API-Key"
    models_dir: Path = Path(os.getenv("SMART_RETAIL_MODELS_DIR", BASE_DIR / "app" / "models"))
    data_dir: Path = Path(os.getenv("SMART_RETAIL_DATA_DIR", BASE_DIR / "data"))
    database_path: Path = Path(
        os.getenv("SMART_RETAIL_DB_PATH", BASE_DIR / "data" / "smart_retail.db")
    )
    max_upload_bytes: int = int(os.getenv("SMART_RETAIL_MAX_UPLOAD_BYTES", 8 * 1024 * 1024))
    face_match_threshold: float = float(os.getenv("SMART_RETAIL_FACE_THRESHOLD", "0.88"))
    chatbot_confidence_threshold: float = float(
        os.getenv("SMART_RETAIL_CHATBOT_THRESHOLD", "0.34")
    )


settings = Settings()
settings.models_dir.mkdir(parents=True, exist_ok=True)
settings.data_dir.mkdir(parents=True, exist_ok=True)
