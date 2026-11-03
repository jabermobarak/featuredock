"""Seeded synthetic sensor fixture with two clocks and future-defined labels."""

from __future__ import annotations

import hashlib
import json

import numpy as np

START = 1_792_353_600  # 2026-10-18 20:00:00 UTC; historical virtual clock.


def fixture(seed=42, assets=24, hours=96):
    if assets < 4 or hours < 24 or hours % 4:
        raise ValueError("Use at least four assets and a multiple of four hours, at least 24")
    rng = np.random.default_rng(seed)
    messages, anchors = [], []
    for machine in range(assets):
        asset = f"asset-{machine:03d}"
        z, temperatures, vibrations = 0.0, [], []
        for step in range(hours * 12 + 1):
            z = 0.975 * z + rng.normal(0, 0.18)
            if rng.random() < 0.008:
                z += rng.uniform(1.5, 2.8)
            temp = float(np.clip(62 + 8 * z + 3 * np.sin(step / 45) + rng.normal(0, 1), -20, 150))
            vib = float(np.clip(1.1 + 0.34 * z + rng.normal(0, 0.07), 0.01, 10))
            load = float(np.clip(55 + 12 * np.sin(step / 70) + 3 * z + rng.normal(0, 4), 5, 100))
            temperatures.append(temp)
            vibrations.append(vib)
            event_at = START + step * 300
            delay = (
                int(rng.integers(5, 100)) if rng.random() > 0.12 else int(rng.integers(1200, 7200))
            )
            payload = {
                "event_id": f"s-{machine:03d}-{step:05d}",
                "asset_id": asset,
                "event_at": event_at,
                "temperature": round(temp, 6),
                "vibration": round(vib, 6),
                "load": round(load, 6),
            }
            envelope = {
                "message_id": f"raw-{machine}-{step}",
                "payload": payload,
                "received_at": event_at + delay,
            }
            messages.append(envelope)
            if step % 97 == 0:
                messages.append(
                    {
                        **envelope,
                        "message_id": f"retry-{machine}-{step}",
                        "received_at": event_at + delay + 30,
                    }
                )
            if step % 223 == 0:
                messages.append(
                    {
                        **envelope,
                        "message_id": f"bad-{machine}-{step}",
                        "payload": {
                            **payload,
                            "event_id": f"bad-{machine}-{step}",
                            "temperature": 999.0,
                        },
                    }
                )
        for hour in range(2, hours):
            start = hour * 12
            # An illustrative future alert, not an observed mechanical failure.
            label = int(
                max(temperatures[start + 1 : start + 13]) > 82
                or max(vibrations[start + 1 : start + 13]) > 2.15
            )
            anchors.append(
                {
                    "asset_id": asset,
                    "prediction_at": START + hour * 3600,
                    "label_available_at": START + (hour + 1) * 3600,
                    "label": label,
                }
            )
    messages.sort(key=lambda m: (m["received_at"], m["message_id"]))
    anchors.sort(key=lambda a: (a["prediction_at"], a["asset_id"]))
    identity = {
        "seed": seed,
        "assets": assets,
        "hours": hours,
        "messages": messages,
        "anchors": anchors,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {
        "messages": messages,
        "anchors": anchors,
        "seed": seed,
        "assets": assets,
        "hours": hours,
        "start": START,
        "train_end": START + hours // 2 * 3600,
        "selection_end": START + hours * 3 // 4 * 3600,
        "end": START + hours * 3600,
        "dataset_sha256": fingerprint,
        "data_source": "synthetic_sensor_demo",
    }


def load_fixture(store, data):
    old = store.get_metadata("fixture")
    config = {
        k: data[k]
        for k in (
            "seed",
            "assets",
            "hours",
            "start",
            "train_end",
            "selection_end",
            "end",
            "dataset_sha256",
            "data_source",
        )
    }
    if old and old != config:
        raise ValueError("This database belongs to another fixture. Choose a new --db path.")
    if not old and store.summary()["events"]:
        raise ValueError("Refusing to seed a database containing other events")
    store.enqueue(data["messages"])
    stats = store.drain()
    store.set_metadata("fixture", config)
    return stats
