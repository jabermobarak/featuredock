"""Actual PySpark point-in-time backfill, checked against independent online SQL."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .contracts import FEATURE_NAMES, feature_row
from .features import lookup

QUERY = """
SELECT a.asset_id, a.prediction_at,
 SUM(CASE WHEN e.event_at>a.prediction_at-3600 THEN 1 ELSE 0 END) AS sample_count,
 SUM(CASE WHEN e.event_at<=a.prediction_at-3600 THEN 1 ELSE 0 END) AS previous_count,
 AVG(CASE WHEN e.event_at>a.prediction_at-3600 THEN e.temperature END) AS temperature_mean,
 MAX(CASE WHEN e.event_at>a.prediction_at-3600 THEN e.temperature END) AS temperature_max,
 AVG(CASE WHEN e.event_at>a.prediction_at-3600 THEN e.vibration END) AS vibration_mean,
 MAX(CASE WHEN e.event_at>a.prediction_at-3600 THEN e.vibration END) AS vibration_max,
 AVG(CASE WHEN e.event_at>a.prediction_at-3600 THEN e.load END) AS load_mean,
 AVG(CASE WHEN e.event_at<=a.prediction_at-3600 THEN e.temperature END) AS previous_temperature_mean,
 MAX(CASE WHEN e.event_at>a.prediction_at-3600 THEN e.event_at END) AS latest_event_at
FROM anchors a LEFT JOIN sensor_events e ON a.asset_id=e.asset_id
 AND e.event_at>a.prediction_at-7200 AND e.event_at<=a.prediction_at
 AND e.received_at<=a.prediction_at
GROUP BY a.asset_id,a.prediction_at
"""


def backfill(store, anchors, parquet: Path):
    from pyspark.sql import SparkSession

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    started = time.perf_counter()
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("FeatureDockBackfill")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        events = store.events()
        event_rows = [
            (
                e["event_id"],
                e["asset_id"],
                e["event_at"],
                e["received_at"],
                e["temperature"],
                e["vibration"],
                e["load"],
            )
            for e in events
        ]
        spark.createDataFrame(
            event_rows,
            "event_id string,asset_id string,event_at long,received_at long,temperature double,vibration double,load double",
        ).createOrReplaceTempView("sensor_events")
        spark.createDataFrame(
            [(a["asset_id"], a["prediction_at"]) for a in anchors],
            "asset_id string,prediction_at long",
        ).createOrReplaceTempView("anchors")
        result = spark.sql(QUERY).cache()
        collected = result.collect()
        differences, mismatches = [], 0
        for value in collected:
            row = value.asDict()
            batch = feature_row(row["asset_id"], row["prediction_at"], row)
            online = lookup(store, row["asset_id"], row["prediction_at"])
            if batch["usable"] != online["usable"]:
                mismatches += 1
            for name in FEATURE_NAMES:
                bv, ov = batch["features"][name], online["features"][name]
                if bv is None or ov is None:
                    if bv != ov:
                        mismatches += 1
                else:
                    differences.append(abs(bv - ov))
        max_difference = max(differences, default=0)
        if mismatches or max_difference > 1e-9:
            raise AssertionError(
                f"Training/serving mismatch: {mismatches}, max difference {max_difference}"
            )
        result.write.mode("overwrite").parquet(str(parquet.resolve()))
        reread = spark.read.parquet(str(parquet.resolve())).count()
        if reread != len(anchors):
            raise AssertionError("Parquet row count mismatch")
        return {
            "status": "passed",
            "engine": spark.version,
            "master": "local[2]",
            "events": len(events),
            "rows": reread,
            "feature_values_compared": len(differences),
            "max_abs_difference": max_difference,
            "null_or_quality_mismatches": mismatches,
            "seconds": time.perf_counter() - started,
        }
    finally:
        spark.stop()


def update_report(directory, result):
    path = Path(directory) / "report.json"
    report = json.loads(path.read_text())
    report["spark"] = result
    path.write_text(json.dumps(report, indent=2))
