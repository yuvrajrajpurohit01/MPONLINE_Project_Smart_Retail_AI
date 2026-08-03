from __future__ import annotations

from fastapi import Request

from app.pipeline import SmartRetailPipeline, get_pipeline


def pipeline_from_request(request: Request) -> SmartRetailPipeline:
    return getattr(request.app.state, "pipeline", get_pipeline())
