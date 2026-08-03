from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import pipeline_from_request
from app.pipeline import SmartRetailPipeline
from app.schemas import ChatbotRequest, ChatbotResponse
from app.security import verify_api_key


router = APIRouter(tags=["Chatbot"], dependencies=[Depends(verify_api_key)])


@router.post("/chatbot", response_model=ChatbotResponse)
def chatbot(
    request: ChatbotRequest,
    pipeline: SmartRetailPipeline = Depends(pipeline_from_request),
):
    return pipeline.chatbot.reply(request.message)
