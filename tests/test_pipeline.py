import copy
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from featuredock.api import create_app
from featuredock.contracts import FEATURE_NAMES, SensorEvent
from featuredock.demo import fixture, load_fixture
from featuredock.features import lookup
from featuredock.model import Model, train
from featuredock.store import Conflict, Store


def event(identifier="event-1", at=7200, **overrides):
    return {
        "event_id": identifier,
        "asset_id": "asset-000",
        "event_at": at,
        "temperature": 60.0,
        "vibration": 1.0,
        "load": 50.0,
        **overrides,
    }


def envelope(identifier, payload, received_at=7300):
    return {"message_id": identifier, "payload": payload, "received_at": received_at}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "events.db")


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixture")
    data = fixture(assets=8, hours=48)
    store = Store(root / "events.db")
    load_fixture(store, data)
    train(store, data, root / "artifacts")
    return root, data


@pytest.fixture
def client(trained, tmp_path):
    root, _ = trained
    # Isolate each test's mutable database, using SQLite's online backup API.
    target = Store(tmp_path / "copy.db")
    with Store(root / "events.db").connection() as source, target.connection() as destination:
        source.backup(destination)
    with TestClient(create_app(target.path, root / "artifacts", api_token="")) as client:
        yield client


def test_duplicate_delivery_has_one_effect(store):
    store.enqueue([envelope("m1", event()), envelope("m2", event(), 7400)])
    assert store.drain() == {"applied": 1, "duplicate": 1, "quarantined": 0}
    assert len(store.events()) == 1
    assert store.events()[0]["received_at"] == 7300
    assert store.enqueue([envelope("m1", event(), 9000)]) == 0
    assert store.drain() == {"applied": 0, "duplicate": 0, "quarantined": 0}


def test_conflicting_message_rolls_back_entire_enqueue(store):
    store.enqueue([envelope("m1", event())])
    with pytest.raises(Conflict):
        store.enqueue([envelope("m2", event("two")), envelope("m1", event(temperature=70.0))])
    assert store.summary()["inbox"]["pending"] == 1


def test_conflicting_event_is_quarantined(store):
    store.enqueue([envelope("m1", event()), envelope("m2", event(temperature=70.0))])
    store.drain()
    assert store.summary()["inbox"]["quarantined"] == 1
    assert store.events()[0]["temperature"] == 60


@pytest.mark.parametrize(
    "bad",
    [
        {"temperature": 999.0},
        {"vibration": -1.0},
        {"load": 101.0},
        {"event_at": True},
        {"asset_id": "invalid space"},
        {"unknown": "field"},
        {"temperature": "60"},
    ],
)
def test_schema_rejects_bad_readings(bad):
    with pytest.raises(ValueError):
        SensorEvent.model_validate(event(**bad))


def test_bad_event_does_not_block_good_event(store):
    store.enqueue([envelope("bad", event("bad", temperature=999.0)), envelope("good", event())])
    assert store.drain() == {"applied": 1, "duplicate": 0, "quarantined": 1}


def test_future_timestamp_quarantined(store):
    store.enqueue([envelope("m", event(at=8000), 7300)])
    store.drain()
    assert store.summary()["inbox"]["quarantined"] == 1


def test_crash_before_commit_leaves_checkpoint_and_events_unchanged(store):
    store.enqueue([envelope("m", event())])

    def fail():
        raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError):
        store.process(before_commit=fail)
    restarted = Store(store.path)
    assert restarted.events() == []
    assert restarted.summary()["inbox"]["pending"] == 1
    assert restarted.drain()["applied"] == 1


def test_concurrent_workers_cannot_apply_twice(store):
    store.enqueue([envelope(f"m-{i}", event()) for i in range(20)])
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: Store(store.path).drain(), range(4)))
    assert store.summary()["events"] == 1
    assert store.summary()["inbox"]["duplicate"] == 19


def test_two_clock_boundaries_against_hand_calculation(store):
    at = 15000
    store.enqueue(
        [
            envelope("old-boundary", event("old", at=at - 7200, temperature=150.0), at - 7200),
            envelope("previous", event("prev", at=at - 3600, temperature=40.0), at - 3600),
            envelope("current", event("now", at=at, temperature=60.0), at),
            envelope("late", event("late", at=at - 100, temperature=100.0), at + 1),
            envelope("future", event("future", at=at + 1, temperature=120.0), at + 1),
        ]
    )
    store.drain()
    snapshot = lookup(store, "asset-000", at)
    assert snapshot["features"]["sample_count"] == 1
    assert snapshot["previous_count"] == 1
    assert snapshot["features"]["temperature_mean"] == 60
    assert snapshot["features"]["temperature_delta"] == 20
    assert snapshot["features"]["freshness_seconds"] == 0
    assert lookup(store, "asset-000", at, include_late=True)["features"]["temperature_mean"] == 80


def test_late_event_cannot_rewrite_historical_snapshot(store):
    store.enqueue([envelope("m1", event())])
    store.drain()
    before = lookup(store, "asset-000", 8000)
    store.enqueue([envelope("m2", event("late", at=7500, temperature=100.0), 9000)])
    store.drain()
    assert lookup(store, "asset-000", 8000) == before
    assert lookup(store, "asset-000", 9000)["features"]["sample_count"] == 2


def test_missing_history_abstains(store):
    result = lookup(store, "missing", 10000)
    assert result["usable"] is False
    assert result["features"]["freshness_seconds"] is None


def test_fixture_reproducible_and_labels_have_a_horizon():
    a = fixture(assets=4, hours=24)
    assert a["dataset_sha256"] == fixture(assets=4, hours=24)["dataset_sha256"]
    assert all(x["label_available_at"] == x["prediction_at"] + 3600 for x in a["anchors"])


def test_model_selection_independent_of_test_labels(trained, tmp_path):
    root, data = trained
    changed = copy.deepcopy(data)
    for a in changed["anchors"]:
        if a["prediction_at"] >= changed["selection_end"]:
            a["label"] = 1 - a["label"]
    changed["dataset_sha256"] = "different-test-labels"
    train(Store(root / "events.db"), changed, tmp_path)
    assert json.loads((tmp_path / "model.json").read_text()) == json.loads(
        (root / "artifacts/model.json").read_text()
    )


def test_model_rejects_tampered_weights(trained):
    root, _ = trained
    artifact = json.loads((root / "artifacts/model.json").read_text())
    artifact["coef"][0] += 0.01
    with pytest.raises(ValueError, match="checksum"):
        Model(artifact)


def test_prediction_contributions_sum_to_log_odds(trained):
    root, data = trained
    model = Model.load(root / "artifacts")
    snapshot = lookup(Store(root / "events.db"), "asset-000", data["selection_end"] + 3600)
    result = model.predict(snapshot)
    logit = result["intercept"] + sum(x["log_odds"] for x in result["contributions"])
    assert result["probability"] == pytest.approx(1 / (1 + np.exp(-logit)))
    snapshot["prediction_at"] = data["train_end"]
    assert model.predict(snapshot)["status"] == "model_not_available_at_this_time"
    assert set(snapshot["features"]) == set(FEATURE_NAMES)


def test_health_ready_and_snapshot(client, trained):
    _, data = trained
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/").status_code == 200
    result = client.get(f"/api/snapshot?asset_id=asset-000&at={data['selection_end'] + 3600}")
    assert result.status_code == 200
    assert result.json()["prediction"]["status"] == "scored"
    assert client.get("/api/snapshot?asset_id=x&at=9999999999").status_code == 422


def test_api_assigns_receipt_time_and_queues_processing(client):
    payload = {"messages": [{"message_id": "live-1", "payload": event()}]}
    response = client.post("/api/ingest", json=payload)
    assert response.status_code == 200 and response.json()["enqueued"] == 1
    assert client.get("/api/summary").json()["inbox"]["pending"] == 1
    assert client.post("/api/process").json()["applied"] == 1
    payload["messages"][0]["received_at"] = 0
    assert client.post("/api/ingest", json=payload).status_code == 422


def test_api_message_conflict(client):
    body = {"messages": [{"message_id": "m-api", "payload": event()}]}
    assert client.post("/api/ingest", json=body).status_code == 200
    body["messages"][0]["payload"]["temperature"] = 70.0
    assert client.post("/api/ingest", json=body).status_code == 409


def test_probe_paths_have_real_effects(client, trained):
    _, data = trained
    at = data["selection_end"] + 3600
    query = f"/api/snapshot?asset_id=asset-000&at={at}"
    before = client.get(query).json()["point_in_time"]
    original = client.get("/api/summary").json()["events"]
    for scenario, key in (
        ("duplicate", "duplicate"),
        ("invalid", "quarantined"),
        ("late", "applied"),
    ):
        response = client.post(
            "/api/demo/probe",
            json={"scenario": scenario, "asset_id": "asset-000", "prediction_at": at},
        )
        assert response.status_code == 200
        assert response.json()["result"][key] == 1
    after = client.get(query).json()
    assert after["point_in_time"] == before
    assert after["all_arrivals_diagnostic"]["features"]["temperature_max"] == 110
    assert client.get("/api/summary").json()["events"] == original + 1
    assert "featuredock_events" in client.get("/metrics").text


def test_missing_artifacts_not_ready(tmp_path):
    with TestClient(create_app(tmp_path / "events.db", tmp_path)) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503


def test_token_guards_data_and_writes(trained):
    root, _ = trained
    with TestClient(create_app(root / "events.db", root / "artifacts", api_token="secret")) as c:
        for path in ("/api/config", "/api/summary", "/api/report", "/metrics"):
            assert c.get(path).status_code == 401
            assert c.get(path, headers={"Authorization": "Bearer secret"}).status_code == 200
        assert c.post("/api/process").status_code == 401
