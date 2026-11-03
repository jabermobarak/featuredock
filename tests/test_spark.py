import os

import pytest

from featuredock.demo import fixture, load_fixture
from featuredock.spark_job import backfill
from featuredock.store import Store


@pytest.mark.spark
@pytest.mark.skipif(
    os.getenv("FD_RUN_SPARK") != "1", reason="Set FD_RUN_SPARK=1 with PySpark and Java 17"
)
def test_real_spark_backfill_matches_sql_and_parquet(tmp_path):
    data = fixture(assets=4, hours=24)
    store = Store(tmp_path / "events.db")
    load_fixture(store, data)
    result = backfill(store, data["anchors"], tmp_path / "parquet")
    assert result["status"] == "passed"
    assert result["rows"] == len(data["anchors"])
    assert result["max_abs_difference"] < 1e-9
