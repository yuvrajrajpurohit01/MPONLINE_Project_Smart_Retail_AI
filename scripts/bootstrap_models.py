from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*parts: str) -> None:
    command = [sys.executable, *parts]
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    run("scripts/train_sentiment.py")
    run("scripts/train_chatbot.py")
    run("scripts/train_product_classifier.py", "--demo")
    face_db = ROOT / "app" / "models" / "face_db.pkl"
    if not face_db.exists():
        run("scripts/init_face_db.py")
    print("All lightweight model artifacts are ready.")
