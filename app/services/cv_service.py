from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from threading import Lock

import cv2
import h5py
import numpy as np

from app.services.storage_service import StorageService
from app.utils.cv_utils import (
    cosine_similarity,
    decode_image,
    face_embedding,
    largest_face_crop,
    product_feature_vector,
    resize_image,
)


class CVService:
    def __init__(
        self,
        models_dir: Path,
        storage: StorageService,
        face_match_threshold: float = 0.88,
    ):
        self.models_dir = Path(models_dir)
        self.storage = storage
        self.face_match_threshold = face_match_threshold
        self.face_db_path = self.models_dir / "face_db.pkl"
        self.product_model_path = self.models_dir / "product_classifier.h5"
        self.metadata_path = self.models_dir / "model_metadata.json"
        self._face_lock = Lock()
        self._keras_model = None
        self._load_face_database()
        self._load_product_model()

    def _load_face_database(self) -> None:
        if not self.face_db_path.exists():
            self.face_database = {"version": 1, "customers": []}
            self._save_face_database()
            return
        with self.face_db_path.open("rb") as handle:
            self.face_database = pickle.load(handle)
        self.face_database.setdefault("customers", [])

    def _save_face_database(self) -> None:
        self.face_db_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=self.face_db_path.parent) as handle:
            pickle.dump(self.face_database, handle)
            temp_name = handle.name
        os.replace(temp_name, self.face_db_path)

    def _load_product_model(self) -> None:
        try:
            with h5py.File(self.product_model_path, "r") as model_file:
                model_type = model_file.attrs.get("model_type", "")
                if isinstance(model_type, bytes):
                    model_type = model_type.decode("utf-8")
                self.product_model_type = str(model_type)
                if self.product_model_type == "color_histogram_centroid_v1":
                    self.product_categories = [
                        value.decode("utf-8") if isinstance(value, bytes) else str(value)
                        for value in model_file["categories"][:]
                    ]
                    self.product_centroids = model_file["centroids"][:].astype(np.float32)
                    return
        except OSError:
            pass

        # A real MobileNetV2/Keras model can replace the included demo H5 file.
        try:
            from tensorflow.keras.models import load_model

            self._keras_model = load_model(self.product_model_path)
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.product_categories = metadata["product_categories"]
            self.product_model_type = "keras_mobilenetv2"
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Unable to load product_classifier.h5. Run `python scripts/bootstrap_models.py` "
                "or install TensorFlow for a MobileNetV2 model."
            ) from exc

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        return exp / exp.sum()

    def classify_product(self, image_bytes: bytes) -> dict:
        image = decode_image(image_bytes)
        if self.product_model_type == "color_histogram_centroid_v1":
            feature = product_feature_vector(image)
            distances = np.linalg.norm(self.product_centroids - feature[None, :], axis=1)
            probabilities = self._softmax(-distances * 8.0)
        else:  # pragma: no cover - exercised after user trains the optional TensorFlow model
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

            rgb = cv2.cvtColor(resize_image(image, 224, 224), cv2.COLOR_BGR2RGB)
            batch = preprocess_input(rgb.astype(np.float32))[None, ...]
            probabilities = np.asarray(self._keras_model.predict(batch, verbose=0)[0], dtype=float)
        best_index = int(np.argmax(probabilities))
        category = self.product_categories[best_index]
        confidence = float(probabilities[best_index])
        self.storage.log_product(category, confidence)
        return {
            "category": category,
            "confidence": round(confidence, 4),
            "model_type": self.product_model_type,
        }

    def recognize_face(self, image_bytes: bytes) -> dict:
        image = decode_image(image_bytes)
        face = largest_face_crop(image)
        if face is None:
            return {
                "status": "no_face_detected",
                "customer_id": None,
                "customer_name": None,
                "confidence": 0.0,
                "face_detected": False,
                "visit_logged": False,
            }
        embedding = face_embedding(face)
        customers = self.face_database.get("customers", [])
        if not customers:
            self.storage.log_face_visit(None, "unknown", 0.0)
            return {
                "status": "unknown",
                "customer_id": None,
                "customer_name": None,
                "confidence": 0.0,
                "face_detected": True,
                "visit_logged": True,
            }

        similarities = [
            cosine_similarity(embedding, np.asarray(customer["embedding"], dtype=np.float32))
            for customer in customers
        ]
        best_index = int(np.argmax(similarities))
        confidence = max(0.0, min(1.0, float(similarities[best_index])))
        best_customer = customers[best_index]
        if confidence >= self.face_match_threshold:
            status = "recognized"
            customer_id = best_customer["customer_id"]
            customer_name = best_customer["name"]
        else:
            status = "unknown"
            customer_id = None
            customer_name = None
        self.storage.log_face_visit(customer_id, status, confidence)
        return {
            "status": status,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "confidence": round(confidence, 4),
            "face_detected": True,
            "visit_logged": True,
        }

    def register_face(self, image_bytes: bytes, customer_id: str, name: str, consent: bool) -> dict:
        if not consent:
            raise ValueError("Explicit consent is required before registering a face.")
        image = decode_image(image_bytes)
        face = largest_face_crop(image)
        if face is None:
            raise ValueError("No clear frontal face was detected in the image.")
        embedding = face_embedding(face).tolist()
        with self._face_lock:
            customers = self.face_database.setdefault("customers", [])
            existing = next((item for item in customers if item["customer_id"] == customer_id), None)
            record = {"customer_id": customer_id, "name": name, "embedding": embedding}
            if existing:
                existing.update(record)
                status = "updated"
            else:
                customers.append(record)
                status = "registered"
            self._save_face_database()
        return {
            "customer_id": customer_id,
            "customer_name": name,
            "status": status,
            "face_database_size": len(self.face_database["customers"]),
        }
