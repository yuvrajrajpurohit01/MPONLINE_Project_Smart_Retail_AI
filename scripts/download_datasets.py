from __future__ import annotations

import argparse
from pathlib import Path


DATASETS = {
    "fashion_mnist": "https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist",
    "rpc": "https://rpc-dataset.github.io/",
    "lfw": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html",
    "womens_reviews": "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print approved dataset references; large downloads are intentionally opt-in.")
    parser.add_argument("--fetch-lfw", action="store_true")
    args = parser.parse_args()
    for name, url in DATASETS.items():
        print(f"{name}: {url}")
    if args.fetch_lfw:
        from sklearn.datasets import fetch_lfw_people
        dataset = fetch_lfw_people(min_faces_per_person=20, resize=0.5)
        print(f"Downloaded LFW cache with {dataset.images.shape[0]} images. Use for research practice only; do not treat it as consent for retail identification.")
