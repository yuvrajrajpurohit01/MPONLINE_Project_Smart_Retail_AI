from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=["The delivery was fast and the product is excellent."])


class SentimentResponse(BaseModel):
    sentiment: Literal["Positive", "Negative", "Neutral"]
    confidence: float = Field(..., ge=0, le=1)
    cleaned_text: str


class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, examples=["What is your return policy?"])


class ChatbotResponse(BaseModel):
    reply: str
    intent: str
    confidence: float = Field(..., ge=0, le=1)
    strategy: Literal["rule", "ml", "fallback"]


class ProductClassificationResponse(BaseModel):
    category: str
    confidence: float = Field(..., ge=0, le=1)
    model_type: str


class FaceRecognitionResponse(BaseModel):
    status: Literal["recognized", "unknown", "no_face_detected"]
    customer_id: str | None = None
    customer_name: str | None = None
    confidence: float = Field(..., ge=0, le=1)
    face_detected: bool
    visit_logged: bool


class FaceRegistrationResponse(BaseModel):
    customer_id: str
    customer_name: str
    status: str
    face_database_size: int


class ActivityItem(BaseModel):
    activity_type: str
    label: str
    confidence: float
    created_at: datetime


class DashboardStatsResponse(BaseModel):
    total_face_visits: int
    recognized_visits: int
    unknown_visits: int
    unique_returning_customers: int
    sentiment_distribution: dict[str, int]
    product_distribution: dict[str, int]
    top_chatbot_intents: dict[str, int]
    latest_activity: list[ActivityItem]
