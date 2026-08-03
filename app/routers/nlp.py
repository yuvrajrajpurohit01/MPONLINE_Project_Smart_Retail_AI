from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import pipeline_from_request
from app.pipeline import SmartRetailPipeline
from app.schemas import SentimentRequest, SentimentResponse
from app.security import verify_api_key


router = APIRouter(tags=["Natural Language Processing"], dependencies=[Depends(verify_api_key)])


@router.post("/analyze-sentiment", response_model=SentimentResponse)
def analyze_sentiment(
    request: SentimentRequest,
    pipeline: SmartRetailPipeline = Depends(pipeline_from_request),
):
    return pipeline.nlp.analyze_sentiment(request.text)
