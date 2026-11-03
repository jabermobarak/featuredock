"""Local historical replay service; ingestion receipt time is assigned by the server."""

import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .features import evidence, lookup
from .model import Model
from .store import Conflict, Store

STATIC = Path(__file__).parent / "static"


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: str = Field(min_length=1, max_length=100)
    payload: dict


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[Envelope] = Field(min_length=1, max_length=100)


class Probe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: Literal["duplicate", "invalid", "late"]
    asset_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,60}$")
    prediction_at: int = Field(ge=0)


def create_app(db_path=None, directory=None, api_token=None):
    db_path = Path(db_path or os.getenv("FD_DB", "runtime/events.db"))
    directory = Path(directory or os.getenv("FD_ARTIFACTS", "artifacts"))
    token = os.getenv("FD_API_TOKEN", "") if api_token is None else api_token

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(db_path)
        app.state.model = None
        app.state.config = app.state.store.get_metadata("fixture")
        try:
            model = Model.load(directory)
            report = json.loads((directory / "report.json").read_text())
            if (
                not app.state.config
                or app.state.config["dataset_sha256"] != report["dataset_sha256"]
            ):
                raise ValueError("Database fixture and model report do not match")
            if model.artifact["model_version"] != report["model_version"]:
                raise ValueError("Model and report do not match")
            app.state.report, app.state.model = report, model
        except (OSError, ValueError, KeyError):
            pass
        yield

    app = FastAPI(title="FeatureDock", version="1.0.0", lifespan=lifespan)

    def auth(authorization: Annotated[str | None, Header()] = None):
        if token and not secrets.compare_digest(authorization or "", "Bearer " + token):
            raise HTTPException(401, "Valid bearer token required")

    def ready_model():
        if not app.state.model:
            raise HTTPException(503, "Run python -m featuredock.cli demo before serving")
        return app.state.model

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def readiness(model: Annotated[Model, Depends(ready_model)]):
        return {"status": "ready", "model_version": model.artifact["model_version"]}

    @app.get("/api/config", dependencies=[Depends(auth)])
    def config(model: Annotated[Model, Depends(ready_model)]):
        with app.state.store.connection() as db:
            assets = [
                r[0] for r in db.execute("SELECT DISTINCT asset_id FROM events ORDER BY asset_id")
            ]
        return {
            **app.state.config,
            "assets": assets,
            "model_version": model.artifact["model_version"],
        }

    @app.get("/api/report", dependencies=[Depends(auth)])
    def report(model: Annotated[Model, Depends(ready_model)]):
        return app.state.report

    @app.get("/api/summary", dependencies=[Depends(auth)])
    def summary():
        return app.state.store.summary()

    @app.get("/api/snapshot", dependencies=[Depends(auth)])
    def snapshot(
        model: Annotated[Model, Depends(ready_model)],
        asset_id: Annotated[str, Query(pattern=r"^[a-zA-Z0-9_-]{1,60}$")],
        at: Annotated[int, Query(ge=0)],
    ):
        if at > int(time.time()):
            raise HTTPException(422, "Prediction timestamp must not be in the future")
        correct = lookup(app.state.store, asset_id, at)
        naive = lookup(app.state.store, asset_id, at, include_late=True)
        return {
            "point_in_time": correct,
            "all_arrivals_diagnostic": naive,
            "prediction": model.predict(correct),
            "evidence": evidence(app.state.store, asset_id, at),
            "note": "The all-arrivals column is a deliberately leaky diagnostic, never used for scoring",
        }

    @app.post("/api/ingest", dependencies=[Depends(auth)])
    def ingest(body: Batch):
        now = int(time.time())
        try:
            inserted = app.state.store.enqueue(
                [{**m.model_dump(), "received_at": now} for m in body.messages]
            )
        except Conflict as error:
            raise HTTPException(409, str(error)) from error
        return {
            "enqueued": inserted,
            "received_at": now,
            "processing": "pending; call POST /api/process",
        }

    @app.post("/api/process", dependencies=[Depends(auth)])
    def process():
        return app.state.store.process()

    @app.post("/api/demo/probe", dependencies=[Depends(auth)])
    def probe(body: Probe, model: Annotated[Model, Depends(ready_model)]):
        cfg = app.state.config
        if not cfg["selection_end"] <= body.prediction_at < cfg["end"]:
            raise HTTPException(422, "Choose a timestamp in the bundled test window")
        with app.state.store.connection() as db:
            original = db.execute(
                "SELECT payload FROM events WHERE asset_id=? ORDER BY event_at LIMIT 1",
                (body.asset_id,),
            ).fetchone()
        if not original:
            raise HTTPException(404, "Unknown demo asset")
        payload = json.loads(original[0])
        identifier = str(uuid.uuid4())
        if body.scenario == "invalid":
            payload.update(event_id="bad-" + identifier, temperature=999.0)
        elif body.scenario == "late":
            payload.update(
                event_id="late-" + identifier,
                event_at=body.prediction_at - 600,
                temperature=110.0,
                vibration=6.0,
                load=80.0,
            )
        app.state.store.enqueue(
            [
                {
                    "message_id": "probe-" + identifier,
                    "payload": payload,
                    "received_at": int(time.time()),
                }
            ]
        )
        return {"scenario": body.scenario, "result": app.state.store.process(), "payload": payload}

    @app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(auth)])
    def metrics():
        values = app.state.store.summary()
        return (
            "".join(
                f"# TYPE featuredock_{k} gauge\nfeaturedock_{k} {v}\n"
                for k, v in values["inbox"].items()
            )
            + f"# TYPE featuredock_events gauge\nfeaturedock_events {values['events']}\n"
        )

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
