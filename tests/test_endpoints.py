from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np


def png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_health_is_public(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_protected_endpoint_rejects_missing_key(client):
    response = client.post("/analyze-sentiment", json={"text": "good"})
    assert response.status_code == 401


def test_sentiment_endpoint(client, auth_headers):
    response = client.post(
        "/analyze-sentiment",
        headers=auth_headers,
        json={"text": "Excellent quality and very fast delivery"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["sentiment"] in {"Positive", "Negative", "Neutral"}
    assert 0 <= payload["confidence"] <= 1


def test_chatbot_rule_route(client, auth_headers):
    response = client.post(
        "/chatbot",
        headers=auth_headers,
        json={"message": "What is your return policy?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "return_policy"
    assert payload["strategy"] == "rule"


def test_product_classification(client, auth_headers):
    image = np.full((224, 224, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (40, 55), (184, 165), (75, 115, 150), -1)
    response = client.post(
        "/classify-product",
        headers=auth_headers,
        files={"image": ("product.png", png_bytes(image), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] in {"shoes", "bags", "electronics", "clothing", "groceries"}


def test_face_endpoint_handles_no_face(client, auth_headers):
    blank = np.full((160, 160, 3), 255, dtype=np.uint8)
    response = client.post(
        "/recognize-face",
        headers=auth_headers,
        files={"image": ("blank.png", png_bytes(blank), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "no_face_detected"


def test_dashboard_stats(client, auth_headers):
    response = client.get("/dashboard/stats", headers=auth_headers)
    assert response.status_code == 200
    assert "sentiment_distribution" in response.json()
