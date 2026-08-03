from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("SMART_RETAIL_API_KEY", "dev-secret-key")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


headers = {"X-API-Key": os.environ["SMART_RETAIL_API_KEY"]}

with TestClient(app) as client:
    assert client.get("/health").json()["status"] == "ok"
    print(client.post("/analyze-sentiment", headers=headers, json={"text": "Excellent quality and fast delivery"}).json())
    print(client.post("/chatbot", headers=headers, json={"message": "What is your return policy?"}).json())
    sample = ROOT / "data" / "product_images" / "demo" / "electronics" / "electronics_1.png"
    with sample.open("rb") as handle:
        print(client.post("/classify-product", headers=headers, files={"image": (sample.name, handle, "image/png")}).json())
    print(client.get("/dashboard/stats", headers=headers).json())
