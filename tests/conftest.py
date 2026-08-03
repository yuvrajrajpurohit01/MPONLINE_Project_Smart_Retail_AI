from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("SMART_RETAIL_API_KEY", "test-key")
os.environ.setdefault("SMART_RETAIL_DB_PATH", "/tmp/smart_retail_test.db")

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth_headers():
    return {"X-API-Key": os.environ["SMART_RETAIL_API_KEY"]}
