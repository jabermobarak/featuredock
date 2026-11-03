"""Versioned event and feature contracts. Epoch timestamps are UTC seconds."""

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "sensor-v1"
FEATURE_VERSION = "hourly-v1"
FEATURE_NAMES = [
    "sample_count",
    "temperature_mean",
    "temperature_max",
    "vibration_mean",
    "vibration_max",
    "load_mean",
    "temperature_delta",
    "freshness_seconds",
]


class SensorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    event_id: str = Field(min_length=1, max_length=80)
    asset_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,60}$")
    event_at: int = Field(ge=0)
    temperature: float = Field(ge=-40, le=180)
    vibration: float = Field(ge=0, le=100)
    load: float = Field(ge=0, le=100)


def feature_row(asset_id, at, values):
    current, previous = int(values["sample_count"] or 0), int(values["previous_count"] or 0)
    fresh = at - values["latest_event_at"] if values["latest_event_at"] is not None else None
    issues = []
    if current < 6:
        issues.append("fewer than 6 readings in the last hour")
    if previous < 6:
        issues.append("fewer than 6 readings in the previous hour")
    if fresh is None or fresh > 900:
        issues.append("latest reading is missing or older than 15 minutes")
    delta = (
        values["temperature_mean"] - values["previous_temperature_mean"]
        if values["temperature_mean"] is not None
        and values["previous_temperature_mean"] is not None
        else None
    )
    features = {
        k: values[k] for k in FEATURE_NAMES if k not in ("temperature_delta", "freshness_seconds")
    }
    features.update(sample_count=current, temperature_delta=delta, freshness_seconds=fresh)
    return {
        "asset_id": asset_id,
        "prediction_at": int(at),
        "feature_version": FEATURE_VERSION,
        "usable": not issues,
        "issues": issues,
        "features": features,
        "previous_count": previous,
    }
