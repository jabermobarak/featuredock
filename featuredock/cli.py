from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import fixture, load_fixture
from .model import train
from .store import Store


def main():
    parser = argparse.ArgumentParser(description="FeatureDock data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--db", type=Path, default=Path("runtime/events.db"))
    demo.add_argument("--output", type=Path, default=Path("artifacts"))
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--assets", type=int, default=24)
    demo.add_argument("--hours", type=int, default=96)
    demo.add_argument("--spark", action="store_true")
    demo.add_argument("--retrain", action="store_true")
    check = sub.add_parser("spark-check")
    check.add_argument("--db", type=Path, default=Path("runtime/events.db"))
    check.add_argument("--output", type=Path, default=Path("artifacts"))
    worker = sub.add_parser("worker")
    worker.add_argument("--db", type=Path, default=Path("runtime/events.db"))
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("featuredock.api:create_app", factory=True, host=args.host, port=args.port)
    elif args.command == "worker":
        print(json.dumps(Store(args.db).drain()))
    else:
        store = Store(args.db)
        if args.command == "demo":
            data = fixture(args.seed, args.assets, args.hours)
            load_fixture(store, data)
            existing = args.output / "report.json"
            if not args.retrain and existing.exists() and (args.output / "model.json").exists():
                from .model import Model

                Model.load(args.output)
                report = json.loads(existing.read_text())
                if report["dataset_sha256"] != data["dataset_sha256"]:
                    report = train(store, data, args.output)
            else:
                report = train(store, data, args.output)
        else:
            config = store.get_metadata("fixture")
            if not config:
                raise ValueError("Run demo before spark-check")
            data = fixture(config["seed"], config["assets"], config["hours"])
            report = json.loads((args.output / "report.json").read_text())
        if args.command == "spark-check" or args.spark:
            from .spark_job import backfill, update_report

            result = backfill(store, data["anchors"], args.db.parent / "spark-features")
            update_report(args.output, result)
            report["spark"] = result
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "model_version",
                        "training_rows",
                        "validation_rows",
                        "test_rows",
                        "test",
                        "spark",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
