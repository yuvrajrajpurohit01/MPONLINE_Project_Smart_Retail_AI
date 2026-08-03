from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import pipeline_from_request
from app.pipeline import SmartRetailPipeline
from app.schemas import DashboardStatsResponse
from app.security import verify_api_key


router = APIRouter(tags=["Dashboard"], dependencies=[Depends(verify_api_key)])


@router.get("/dashboard/stats", response_model=DashboardStatsResponse)
def dashboard_stats(pipeline: SmartRetailPipeline = Depends(pipeline_from_request)):
    return pipeline.storage.get_dashboard_stats()
