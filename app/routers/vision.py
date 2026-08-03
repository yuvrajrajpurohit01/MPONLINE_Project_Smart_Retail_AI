from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.dependencies import pipeline_from_request
from app.pipeline import SmartRetailPipeline
from app.schemas import (
    FaceRecognitionResponse,
    FaceRegistrationResponse,
    ProductClassificationResponse,
)
from app.security import verify_api_key
from app.utils.cv_utils import InvalidImageError


router = APIRouter(tags=["Computer Vision"], dependencies=[Depends(verify_api_key)])


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit.")
    return payload


@router.post("/classify-product", response_model=ProductClassificationResponse)
async def classify_product(
    image: UploadFile = File(...),
    pipeline: SmartRetailPipeline = Depends(pipeline_from_request),
):
    try:
        return pipeline.cv.classify_product(await _read_image(image))
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/recognize-face", response_model=FaceRecognitionResponse)
async def recognize_face(
    image: UploadFile = File(...),
    pipeline: SmartRetailPipeline = Depends(pipeline_from_request),
):
    try:
        return pipeline.cv.recognize_face(await _read_image(image))
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/register-face",
    response_model=FaceRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a consenting demo customer face",
)
async def register_face(
    customer_id: str = Form(..., min_length=1, max_length=80),
    customer_name: str = Form(..., min_length=1, max_length=120),
    consent: bool = Form(...),
    image: UploadFile = File(...),
    pipeline: SmartRetailPipeline = Depends(pipeline_from_request),
):
    try:
        return pipeline.cv.register_face(
            await _read_image(image), customer_id.strip(), customer_name.strip(), consent
        )
    except (InvalidImageError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
