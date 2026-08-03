from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class StorageService:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS customer_visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            status TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sentiment_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_hash TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_hash TEXT NOT NULL,
            intent TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS product_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        """
        with self._lock, self._connect() as connection:
            connection.executescript(schema)
            connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def log_face_visit(self, customer_id: str | None, status: str, confidence: float) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO customer_visits(customer_id, status, confidence, created_at) VALUES (?, ?, ?, ?)",
                (customer_id, status, float(confidence), self._now()),
            )
            connection.commit()

    def log_sentiment(self, text: str, sentiment: str, confidence: float) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO sentiment_logs(text_hash, sentiment, confidence, created_at) VALUES (?, ?, ?, ?)",
                (self._hash_text(text), sentiment, float(confidence), self._now()),
            )
            connection.commit()

    def log_chat(self, message: str, intent: str, confidence: float) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO chat_logs(message_hash, intent, confidence, created_at) VALUES (?, ?, ?, ?)",
                (self._hash_text(message), intent, float(confidence), self._now()),
            )
            connection.commit()

    def log_product(self, category: str, confidence: float) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO product_predictions(category, confidence, created_at) VALUES (?, ?, ?)",
                (category, float(confidence), self._now()),
            )
            connection.commit()

    @staticmethod
    def _distribution(connection: sqlite3.Connection, table: str, column: str) -> dict[str, int]:
        rows = connection.execute(
            f"SELECT {column} AS label, COUNT(*) AS total FROM {table} GROUP BY {column} ORDER BY total DESC"
        ).fetchall()
        return {str(row["label"]): int(row["total"]) for row in rows}

    def get_dashboard_stats(self) -> dict:
        with self._connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM customer_visits").fetchone()[0])
            recognized = int(
                connection.execute("SELECT COUNT(*) FROM customer_visits WHERE status='recognized'").fetchone()[0]
            )
            unknown = int(
                connection.execute("SELECT COUNT(*) FROM customer_visits WHERE status='unknown'").fetchone()[0]
            )
            unique_customers = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT customer_id) FROM customer_visits WHERE status='recognized' AND customer_id IS NOT NULL"
                ).fetchone()[0]
            )
            latest = connection.execute(
                """
                SELECT 'face' AS activity_type, COALESCE(customer_id, status) AS label, confidence, created_at
                FROM customer_visits
                UNION ALL
                SELECT 'sentiment', sentiment, confidence, created_at FROM sentiment_logs
                UNION ALL
                SELECT 'chatbot', intent, confidence, created_at FROM chat_logs
                UNION ALL
                SELECT 'product', category, confidence, created_at FROM product_predictions
                ORDER BY created_at DESC
                LIMIT 15
                """
            ).fetchall()
            return {
                "total_face_visits": total,
                "recognized_visits": recognized,
                "unknown_visits": unknown,
                "unique_returning_customers": unique_customers,
                "sentiment_distribution": self._distribution(connection, "sentiment_logs", "sentiment"),
                "product_distribution": self._distribution(connection, "product_predictions", "category"),
                "top_chatbot_intents": self._distribution(connection, "chat_logs", "intent"),
                "latest_activity": [dict(row) for row in latest],
            }
