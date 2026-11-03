"""Standardized logistic model stored as data-only JSON, without pickle."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .contracts import FEATURE_NAMES, FEATURE_VERSION
from .features import lookup


def digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


class Model:
    def __init__(self, artifact):
        if (
            artifact["feature_version"] != FEATURE_VERSION
            or artifact["feature_names"] != FEATURE_NAMES
        ):
            raise ValueError("Feature contract does not match the model")
        expected = digest({k: v for k, v in artifact.items() if k != "model_version"})
        if expected != artifact["model_version"]:
            raise ValueError("Model artifact checksum mismatch")
        self.artifact = artifact
        self.mean, self.scale = np.array(artifact["mean"]), np.array(artifact["scale"])
        self.coef = np.array(artifact["coef"])

    @classmethod
    def load(cls, directory: Path):
        return cls(json.loads((directory / "model.json").read_text()))

    def probability(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        z = ((matrix - self.mean) / self.scale) @ self.coef + self.artifact["intercept"]
        return 1 / (1 + np.exp(-np.clip(z, -40, 40)))

    def predict(self, row):
        if not row["usable"]:
            return {"status": "insufficient_data", "issues": row["issues"], "probability": None}
        if row["prediction_at"] < self.artifact["model_available_at"]:
            return {"status": "model_not_available_at_this_time", "probability": None}
        x = [row["features"][k] for k in FEATURE_NAMES]
        probability = float(self.probability([x])[0])
        contributions = ((np.array(x) - self.mean) / self.scale) * self.coef
        return {
            "status": "scored",
            "probability": probability,
            "threshold": self.artifact["threshold"],
            "alert": probability >= self.artifact["threshold"],
            "model_version": self.artifact["model_version"],
            "contributions": [
                {"feature": k, "log_odds": float(v)}
                for k, v in zip(FEATURE_NAMES, contributions, strict=True)
            ],
            "intercept": self.artifact["intercept"],
        }


def scores(y, probability, threshold):
    predicted = probability >= threshold
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(set(y)) > 1 else None,
        "brier": float(brier_score_loss(y, probability)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "positive_rate": float(np.mean(y)),
        "rows": len(y),
    }


def train(store, data, output: Path):
    started = time.perf_counter()
    rows = [
        {**a, "snapshot": lookup(store, a["asset_id"], a["prediction_at"])} for a in data["anchors"]
    ]
    usable = [r for r in rows if r["snapshot"]["usable"]]
    # Label availability, not just prediction time, determines split membership.
    training = [r for r in usable if r["label_available_at"] <= data["train_end"]]
    validation = [
        r
        for r in usable
        if r["prediction_at"] >= data["train_end"]
        and r["label_available_at"] <= data["selection_end"]
    ]
    test = [
        r
        for r in usable
        if r["prediction_at"] >= data["selection_end"] and r["label_available_at"] <= data["end"]
    ]

    def arrays(part):
        return np.array(
            [[r["snapshot"]["features"][k] for k in FEATURE_NAMES] for r in part]
        ), np.array([r["label"] for r in part])

    tx, ty = arrays(training)
    vx, vy = arrays(validation)
    ex, ey = arrays(test)
    if not len(vy) or not len(ey) or len(set(ty)) < 2 or len(set(vy)) < 2:
        raise ValueError("Training and validation need both classes and test must be nonempty")
    scaler = StandardScaler().fit(tx)
    trials, candidates = [], []
    for c in (0.1, 1.0, 10.0):
        fitted = LogisticRegression(C=c, max_iter=1500, random_state=data["seed"]).fit(
            scaler.transform(tx), ty
        )
        vp = fitted.predict_proba(scaler.transform(vx))[:, 1]
        trials.append({"C": c, "validation_ap": float(average_precision_score(vy, vp))})
        candidates.append(fitted)
    best = int(np.argmax([t["validation_ap"] for t in trials]))
    fitted = candidates[best]
    vp = fitted.predict_proba(scaler.transform(vx))[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    threshold = float(max(thresholds, key=lambda t: f1_score(vy, vp >= t, zero_division=0)))
    artifact = {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": fitted.coef_[0].tolist(),
        "intercept": float(fitted.intercept_[0]),
        "threshold": threshold,
        "C": trials[best]["C"],
        "train_end": data["train_end"],
        "model_available_at": data["selection_end"],
    }
    artifact["model_version"] = digest(artifact)
    model = Model(artifact)
    probability = model.probability(ex)
    np.testing.assert_allclose(
        probability, fitted.predict_proba(scaler.transform(ex))[:, 1], atol=1e-12
    )
    diagnostic = []
    for row in test:
        naive = lookup(store, row["asset_id"], row["prediction_at"], include_late=True)
        diagnostic.append(
            any(
                abs(naive["features"][k] - row["snapshot"]["features"][k]) > 1e-9
                for k in FEATURE_NAMES
            )
        )
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
    report = {
        **config,
        "model_version": artifact["model_version"],
        "feature_version": FEATURE_VERSION,
        "training_rows": len(training),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "quality_gated_rows": len(rows) - len(usable),
        "selection_trials": trials,
        "threshold": threshold,
        "test": scores(ey, probability, threshold),
        "baseline": scores(ey, np.full(len(ey), ty.mean()), threshold),
        "late_data_diagnostic": {
            "test_rows_changed_if_arrival_clock_is_ignored": sum(diagnostic),
            "test_rows": len(test),
        },
        "ingestion": store.summary(),
        "training_seconds": time.perf_counter() - started,
        "spark": {"status": "not_run"},
        "target": "Any simulated temperature >82 or vibration >2.15 in the next hour; not mechanical failure",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.json").write_text(json.dumps(artifact, indent=2))
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report
