from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.pipeline import get_pipeline
from app.routers import chatbot, dashboard, nlp, vision


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = get_pipeline()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Integrated educational retail AI platform with product classification, "
        "consent-based face recognition, sentiment analysis, a hybrid FAQ chatbot, "
        "and aggregate analytics."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vision.router)
app.include_router(nlp.router)
app.include_router(chatbot.router)
app.include_router(dashboard.router)


@app.get("/", tags=["System"])
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "models_loaded": True}
