"""Tests for the leakage contract and the split that protects it.

The leakage rule is this project's central methodological claim, so it gets
assertions rather than a paragraph. The rule is not a precaution here - it is
arithmetic: the mode columns SUM to the target on 100.0000% of the 5.27M rows,
so a model given any of them would re-add the parts and report R^2 ~ 1.0.
"""

from __future__ import annotations

import pytest
from pyspark.sql import Row

from src.ml.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_WITH_REASON,
    LABEL_COLUMN,
    LEAKY_COLUMNS,
    NUMERIC_FEATURES,
    split_by_year,
)
from src.ml.train_models import assert_no_leakage
from src.spark_session import get_spark


@pytest.fixture(scope="module")
def spark():
    session = get_spark("tests_features")
    yield session
    session.stop()


def test_no_target_component_is_a_feature():
    """The mode counts sum to the target; none may be a feature."""
    leaked = set(LEAKY_COLUMNS) & set(ALL_FEATURES)
    assert not leaked, f"Target components used as features: {leaked}"


def test_label_is_not_a_feature():
    assert LABEL_COLUMN not in ALL_FEATURES


def test_every_mode_column_is_barred():
    """Pin the specific columns, so a future edit re-adding one is caught here."""
    for column in ["cars_and_taxis", "buses_and_coaches", "LGVs", "all_HGVs",
                   "two_wheeled_motor_vehicles", "pedal_cycles"]:
        assert column in LEAKY_COLUMNS, f"{column} must be barred as a target component"
        assert column not in ALL_FEATURES


def test_year_is_excluded_because_it_indexes_the_split():
    """`year` is the split variable: every test value is outside the train range."""
    assert "year" in EXCLUDED_WITH_REASON
    assert "year" not in ALL_FEATURES


def test_link_length_is_excluded():
    """Null for 100% of minor roads and 0% of major - road class as missingness."""
    assert "link_length_km" in EXCLUDED_WITH_REASON
    assert "link_length_km" not in ALL_FEATURES


def test_assert_no_leakage_actually_fires(monkeypatch):
    """The guard must raise when violated, not merely exist.

    A guard that has never been seen to fail is not evidence of anything.
    """
    monkeypatch.setattr(
        "src.ml.train_models.ALL_FEATURES", ALL_FEATURES + ["cars_and_taxis"]
    )
    with pytest.raises(RuntimeError, match="LEAKAGE"):
        assert_no_leakage()


def test_assert_no_leakage_passes_on_the_real_feature_list():
    assert_no_leakage()  # must not raise


def test_split_by_year_is_chronological(spark):
    rows = [Row(year=y, count_point_id="A", all_motor_vehicles=100 + y) for y in range(2000, 2026)]
    frame = spark.createDataFrame(rows)

    train, test = split_by_year(frame, train_max_year=2019, test_min_year=2023)
    train_years = {r["year"] for r in train.collect()}
    test_years = {r["year"] for r in test.collect()}

    assert max(train_years) == 2019
    assert min(test_years) == 2023
    assert not (train_years & test_years), "A year appears in both blocks"
    assert max(train_years) < min(test_years), "Test block is not strictly in the future"


def test_split_honours_test_max_year(spark):
    """The control split must not leak the COVID years into its test block."""
    rows = [Row(year=y, count_point_id="A", all_motor_vehicles=1) for y in range(2000, 2026)]
    frame = spark.createDataFrame(rows)

    train, test = split_by_year(frame, train_max_year=2016, test_min_year=2017,
                               test_max_year=2019)
    test_years = {r["year"] for r in test.collect()}
    assert test_years == {2017, 2018, 2019}
    assert 2020 not in test_years, "Control split must exclude the pandemic"


def test_feature_lists_are_disjoint():
    assert not set(CATEGORICAL_FEATURES) & set(NUMERIC_FEATURES)
