"""A durable local inbox and materialized event log with atomic progress.

Exactly-once database effects within this database, not distributed exactly-once delivery.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from .contracts import SensorEvent


class Conflict(ValueError):
    pass


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS inbox(
                    position INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT UNIQUE NOT NULL,
                    payload TEXT NOT NULL, received_at INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    reason TEXT);
                CREATE INDEX IF NOT EXISTS pending_messages ON inbox(status,position);
                CREATE TABLE IF NOT EXISTS events(
                    event_id TEXT PRIMARY KEY, asset_id TEXT NOT NULL, event_at INTEGER NOT NULL,
                    received_at INTEGER NOT NULL, temperature REAL NOT NULL,
                    vibration REAL NOT NULL, load REAL NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS asset_clock ON events(asset_id,event_at,received_at);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, messages):
        """Trusted internal envelope; HTTP callers cannot choose received_at."""
        inserted = 0
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for message in messages:
                raw = canonical(message["payload"])
                old = db.execute(
                    "SELECT payload FROM inbox WHERE message_id=?", (message["message_id"],)
                ).fetchone()
                if old:
                    if old[0] != raw:
                        raise Conflict("message_id already has a different payload")
                    continue
                db.execute(
                    "INSERT INTO inbox(message_id,payload,received_at) VALUES (?,?,?)",
                    (message["message_id"], raw, message["received_at"]),
                )
                inserted += 1
        return inserted

    def process(self, limit=1000, before_commit=None):
        stats = {"applied": 0, "duplicate": 0, "quarantined": 0}
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT * FROM inbox WHERE status='pending' ORDER BY position LIMIT ?", (limit,)
            ).fetchall()
            for row in rows:
                try:
                    event = SensorEvent.model_validate(json.loads(row["payload"]))
                    if event.event_at > row["received_at"] + 60:
                        raise ValueError("event timestamp is more than 60 seconds ahead of receipt")
                    raw = canonical(event.model_dump())
                    old = db.execute(
                        "SELECT payload FROM events WHERE event_id=?", (event.event_id,)
                    ).fetchone()
                    if old and old[0] != raw:
                        raise ValueError("event_id conflicts with an earlier accepted event")
                    status, reason = ("duplicate", None) if old else ("applied", None)
                    if not old:
                        db.execute(
                            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
                            (
                                event.event_id,
                                event.asset_id,
                                event.event_at,
                                row["received_at"],
                                event.temperature,
                                event.vibration,
                                event.load,
                                raw,
                            ),
                        )
                except (ValidationError, ValueError) as error:
                    status = "quarantined"
                    reason = (
                        "; ".join(e["msg"] for e in error.errors())
                        if isinstance(error, ValidationError)
                        else str(error)
                    )[:500]
                db.execute(
                    "UPDATE inbox SET status=?,reason=? WHERE position=?",
                    (status, reason, row["position"]),
                )
                stats[status] += 1
            if before_commit:
                before_commit()
        return stats

    def drain(self):
        totals = {"applied": 0, "duplicate": 0, "quarantined": 0}
        while True:
            batch = self.process()
            for key in totals:
                totals[key] += batch[key]
            if sum(batch.values()) == 0:
                return totals

    def events(self):
        with self.connection() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT event_id,asset_id,event_at,received_at,temperature,vibration,load FROM events ORDER BY event_at,event_id"
                )
            ]

    def summary(self):
        with self.connection() as db:
            statuses = {
                r[0]: r[1] for r in db.execute("SELECT status,COUNT(*) FROM inbox GROUP BY status")
            }
            row = db.execute(
                "SELECT COUNT(*) AS events, COUNT(DISTINCT asset_id) AS assets,"
                " SUM(received_at-event_at>900) AS late FROM events"
            ).fetchone()
            quarantine = [
                dict(r)
                for r in db.execute(
                    "SELECT message_id,reason,received_at FROM inbox "
                    "WHERE status='quarantined' ORDER BY position DESC LIMIT 10"
                )
            ]
        return {
            "inbox": {
                k: statuses.get(k, 0) for k in ("pending", "applied", "duplicate", "quarantined")
            },
            **dict(row),
            "quarantine": quarantine,
        }

    def get_metadata(self, key):
        with self.connection() as db:
            row = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def set_metadata(self, key, value):
        with self.connection() as db:
            db.execute(
                "INSERT INTO metadata VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, canonical(value)),
            )
