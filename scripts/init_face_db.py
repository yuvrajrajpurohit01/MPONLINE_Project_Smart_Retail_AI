from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an empty, consent-first face database.")
    parser.add_argument("--output", type=Path, default=ROOT / "app" / "models" / "face_db.pkl")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists. Add --force to reset it.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump({"version": 1, "customers": [], "created_at": datetime.now(timezone.utc).isoformat()}, handle)
    print(f"Created empty face database at {args.output}")
