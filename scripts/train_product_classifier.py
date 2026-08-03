from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["shoes", "bags", "electronics", "clothing", "groceries"]


def product_feature_vector(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [8], [0, 256]).flatten()
        hist = hist / max(float(hist.sum()), 1.0)
        features.extend(hist.tolist())
    features.extend((hsv.reshape(-1, 3).mean(axis=0) / 255.0).tolist())
    return np.asarray(features, dtype=np.float32)


def build_demo(data_dir: Path, output: Path) -> None:
    categories, centroids = [], []
    for folder in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        vectors = []
        for image_path in folder.glob("*"):
            image = cv2.imread(str(image_path))
            if image is not None:
                vectors.append(product_feature_vector(image))
        if vectors:
            categories.append(folder.name)
            centroids.append(np.mean(vectors, axis=0))
    if len(categories) < 2:
        raise ValueError("Demo classifier requires at least two populated category folders.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as model_file:
        model_file.attrs["model_type"] = "color_histogram_centroid_v1"
        model_file.create_dataset("categories", data=np.asarray(categories, dtype=h5py.string_dtype("utf-8")))
        model_file.create_dataset("centroids", data=np.stack(centroids).astype(np.float32))
    print(f"Saved demo H5 classifier to {output}")


def train_mobilenet(data_dir: Path, output: Path, epochs: int) -> None:
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
        from tensorflow.keras.applications import MobileNetV2
    except ImportError as exc:
        raise SystemExit("Install optional ML packages: pip install -r requirements-ml.txt") from exc

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224, 224),
        batch_size=32,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=32,
    )
    class_names = train_ds.class_names
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ])
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(output, save_best_only=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    metadata_path = output.parent / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update({"product_categories": class_names, "product_model_type": "keras_mobilenetv2"})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved MobileNetV2 model to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "product_images" / "demo")
    parser.add_argument("--output", type=Path, default=ROOT / "app" / "models" / "product_classifier.h5")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--demo", action="store_true", help="Build the lightweight H5 centroid model instead of MobileNetV2")
    args = parser.parse_args()
    if args.demo:
        build_demo(args.data_dir, args.output)
    else:
        train_mobilenet(args.data_dir, args.output, args.epochs)
