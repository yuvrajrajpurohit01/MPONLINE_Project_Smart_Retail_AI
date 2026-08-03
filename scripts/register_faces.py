from __future__ import annotations

import argparse
import sys
import pickle
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.cv_utils import face_embedding, largest_face_crop




def register_folder(input_dir: Path, database_path: Path) -> None:
    """Read folders named CUSTOMER_ID__Customer_Name and save average embeddings."""
    if database_path.exists():
        with database_path.open("rb") as handle:
            database = pickle.load(handle)
    else:
        database = {"version": 1, "customers": []}
    by_id = {item["customer_id"]: item for item in database.get("customers", [])}
    for person_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        if "__" not in person_dir.name:
            print(f"Skipping {person_dir.name}: expected CUSTOMER_ID__Customer_Name")
            continue
        customer_id, name = person_dir.name.split("__", 1)
        embeddings = []
        for image_path in person_dir.glob("*"):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            face = largest_face_crop(image)
            if face is not None:
                embeddings.append(face_embedding(face))
        if not embeddings:
            print(f"Skipping {person_dir.name}: no clear face detected")
            continue
        average = sum(embeddings) / len(embeddings)
        average = average / max(float((average ** 2).sum() ** 0.5), 1e-12)
        by_id[customer_id] = {"customer_id": customer_id, "name": name.replace("_", " "), "embedding": average.tolist()}
        print(f"Registered {customer_id} from {len(embeddings)} image(s)")
    database["customers"] = list(by_id.values())
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with database_path.open("wb") as handle:
        pickle.dump(database, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="Folder containing CUSTOMER_ID__Customer_Name subfolders")
    parser.add_argument("--database", type=Path, default=ROOT / "app" / "models" / "face_db.pkl")
    parser.add_argument("--consent-confirmed", action="store_true", help="Required acknowledgement that every subject consented")
    args = parser.parse_args()
    if not args.consent_confirmed:
        raise SystemExit("Refusing to register faces without --consent-confirmed")
    register_folder(args.input_dir, args.database)
