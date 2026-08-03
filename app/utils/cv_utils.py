from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


def decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty.")
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("The uploaded file is not a supported image.")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def canny_edges(image: np.ndarray, low_threshold: int = 80, high_threshold: int = 160) -> np.ndarray:
    gray = to_grayscale(image) if image.ndim == 3 else image
    return cv2.Canny(gray, low_threshold, high_threshold)


def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return []
    gray = to_grayscale(image)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    return [tuple(map(int, face)) for face in faces]


def largest_face_crop(image: np.ndarray) -> np.ndarray | None:
    faces = detect_faces(image)
    if not faces:
        return None
    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    margin_x = int(width * 0.10)
    margin_y = int(height * 0.10)
    x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
    x2 = min(image.shape[1], x + width + margin_x)
    y2 = min(image.shape[0], y + height + margin_y)
    return image[y1:y2, x1:x2]


def face_embedding(face_image: np.ndarray) -> np.ndarray:
    """Create a lightweight demo embedding without storing the raw face image.

    This is intentionally a teaching baseline. Replace it with a validated face
    embedding model for any real deployment.
    """
    gray = to_grayscale(face_image)
    normalized = cv2.equalizeHist(resize_image(gray, 64, 64)).astype(np.float32) / 255.0
    # Combine coarse appearance and local gradient features.
    pixels = cv2.resize(normalized, (16, 16), interpolation=cv2.INTER_AREA).flatten()
    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    gradients = cv2.resize(magnitude, (16, 16), interpolation=cv2.INTER_AREA).flatten()
    embedding = np.concatenate([pixels, gradients]).astype(np.float32)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def product_feature_vector(image: np.ndarray) -> np.ndarray:
    """Extract a compact 27-value colour/appearance vector for demo inference."""
    resized = resize_image(image, 224, 224)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    features: list[float] = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [8], [0, 256]).flatten()
        hist = hist / max(float(hist.sum()), 1.0)
        features.extend(hist.tolist())
    means = hsv.reshape(-1, 3).mean(axis=0) / 255.0
    features.extend(means.tolist())
    return np.asarray(features, dtype=np.float32)


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise InvalidImageError("Unable to encode processed image.")
    return encoded.tobytes()
