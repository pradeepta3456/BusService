"""Tests for the cleaning rules.

Each rule here exists because of something found in the real data, so each test
pins that specific finding rather than a hypothetical.
"""

from __future__ import annotations

import pytest
from pyspark.sql import Row, functions as F

from src.process.clean_counts import COUNT_COLUMNS, read_named
from src.spark_session import get_spark


@pytest.fixture(scope="module")
def spark():
    session = get_spark("tests_clean")
    yield session
    session.stop()


def test_try_cast_nulls_na_rather_than_raising(spark, tmp_path):
    """The feed encodes missing numerics as 'NA'.

    Under Spark's ANSI mode a plain cast raises CAST_INVALID_INPUT and kills the
    job on the first one - which is exactly what happened on the first run.
    try_cast must yield null so missingness becomes measurable.
    """
    csv = tmp_path / "counts.csv"
    csv.write_text(
        "count_point_id,all_motor_vehicles,link_length_km\n"
        "A,100,1.5\n"
        "B,NA,NA\n"
    )
    frame = spark.read.csv(str(csv), header=True, inferSchema=False).select(
        F.col("count_point_id"),
        F.expr("try_cast(`all_motor_vehicles` AS int)").alias("all_motor_vehicles"),
        F.expr("try_cast(`link_length_km` AS double)").alias("link_length_km"),
    )
    rows = {r["count_point_id"]: r for r in frame.collect()}
    assert rows["A"]["all_motor_vehicles"] == 100
    assert rows["B"]["all_motor_vehicles"] is None
    assert rows["B"]["link_length_km"] is None


def test_read_named_raises_on_missing_column(spark, tmp_path):
    """Columns are bound by name; a missing one must fail loudly, not silently.

    Passing an explicit schema with header=True binds POSITIONALLY and ignores
    the header, so an upstream column insertion would be read as wrong data.
    """
    csv = tmp_path / "bad.csv"
    csv.write_text("count_point_id,hour\nA,8\n")
    with pytest.raises(ValueError, match="missing expected column"):
        read_named(spark, csv, COUNT_COLUMNS)


def test_hour_window_keeps_only_the_real_count_hours(spark):
    """DfT counts run 07:00-18:59. Hours 0/1/3/4/5 were 8 bad rows in 5.27M."""
    rows = [Row(hour=h) for h in [0, 1, 3, 5, 6, 7, 12, 18, 19, 23]]
    frame = spark.createDataFrame(rows).filter(F.col("hour").between(7, 18))
    kept = sorted(r["hour"] for r in frame.collect())
    assert kept == [7, 12, 18]


def test_direction_case_folds_to_one_category(spark):
    """'n' and 'N' are the same direction.

    Left alone, StringIndexer would learn two encodings for one direction and
    split the data on a typo.
    """
    rows = [Row(direction_of_travel=d) for d in ["N", "n", " S ", "s", "E", "W"]]
    frame = spark.createDataFrame(rows).withColumn(
        "direction_of_travel", F.upper(F.trim(F.col("direction_of_travel")))
    )
    values = sorted({r["direction_of_travel"] for r in frame.collect()})
    assert values == ["E", "N", "S", "W"]


def test_is_major_road_matches_dft_road_classes(spark):
    """link_length_km is null for exactly MCU+MB; is_major_road states it directly."""
    rows = [Row(road_category=c) for c in ["TM", "PM", "TA", "PA", "MB", "MCU"]]
    frame = spark.createDataFrame(rows).withColumn(
        "is_major_road",
        F.when(F.col("road_category").isin("PA", "TA", "TM", "PM"), 1).otherwise(0),
    )
    result = {r["road_category"]: r["is_major_road"] for r in frame.collect()}
    assert result == {"TM": 1, "PM": 1, "TA": 1, "PA": 1, "MB": 0, "MCU": 0}
