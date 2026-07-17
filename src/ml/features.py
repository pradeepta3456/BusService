"""Build the ML feature table under a strict leakage rule.

TASK
----
Regression. One row per observed (count point, direction, date, hour). Target:
`all_motor_vehicles` - the motorised vehicles counted in that hour. The question
is the brief's example #4, for the brief's named stakeholder: an Urban Planner
forecasting hourly demand to size infrastructure.

THE LEAKAGE RULE
----------------
Every feature must be knowable *before the hour is counted*. Three groups are
excluded, and the first is not a judgement call - it is arithmetic:

1. MODE COUNTS ARE THE TARGET'S OWN COMPONENTS. Verified on the real data:

       two_wheeled + cars_and_taxis + buses_and_coaches + LGVs + all_HGVs
           == all_motor_vehicles      on 5,269,608 / 5,269,609 rows (100.0000%)

   `cars_and_taxis` alone correlates r = 0.9908 with the target. A model given
   any of these does not predict traffic; it re-adds the parts and reports
   R^2 ~ 1.0. `pedal_cycles` is excluded on the same principle even though bikes
   are not motor vehicles: they are counted in the same hour by the same
   enumerator, so a forecaster would not hold them either.

2. `year` IS THE SPLIT VARIABLE. The train/test boundary is a year, so every
   test value falls outside the training range: trees clamp at their last split
   and a linear model extrapolates off the end of the data. The column can only
   index the split, never carry a transferable pattern.

3. `link_length_km` IS ROAD CLASS IN DISGUISE. It is null for 100% of minor
   roads (MCU, MB) and 0% of major roads (PA, TA, TM, PM) - a structural
   property of how DfT defines the network, not gappy data. Keeping it would
   either discard 54.8% of rows or smuggle road class in through a null pattern.
   `road_category` and `is_major_road` say the same thing honestly.

WHAT IS LEFT, AND WHY IT IS LEGITIMATE
--------------------------------------
  * hour, day_of_week, is_weekend, month - calendar, known indefinitely ahead.
  * road_category, road_type, is_major_road, direction - published network
    attributes of the link.
  * latitude, longitude, region - where the link is.
  * count-point history (mean/sd of past flow at this link) - computed from the
    TRAINING YEARS ONLY and joined onto both blocks. A planner genuinely holds a
    link's past counts before forecasting its future. Computing it across all
    years would leak the test period's outcomes into training.

SPLIT
-----
By year, never random. Random would place a link's 2024 count in training and
its 2023 count in test, on the same road, in the same traffic regime - the model
would interpolate a link it has already seen, which is not forecasting. Two
splits are run (see config): a primary that spans COVID and a control that does
not, because otherwise "the model cannot forecast" and "the pandemic changed the
roads" are indistinguishable.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, functions as F

from src.config import load_config, resolve

LOGGER = logging.getLogger("features")

LABEL_COLUMN = "all_motor_vehicles"

CATEGORICAL_FEATURES = [
    "day_of_week",
    "road_category",
    "road_type",
    "direction_of_travel",
    "region_name",
]

NUMERIC_FEATURES = [
    # The core signal: the commuter double-peak lives here.
    "hour",
    "is_weekend",
    "month",
    "is_major_road",
    "latitude",
    "longitude",
    # Link history - TRAINING YEARS ONLY. See build_history().
    "cp_hist_mean_flow",
    "cp_hist_sd_flow",
    "cp_hist_years",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Columns that must never become features, with the reason each is barred.
LEAKY_COLUMNS = {
    "all_motor_vehicles": "the target itself",
    "cars_and_taxis": "exact component of the target (r=0.991)",
    "buses_and_coaches": "exact component of the target",
    "LGVs": "exact component of the target",
    "all_HGVs": "exact component of the target",
    "two_wheeled_motor_vehicles": "exact component of the target",
    "pedal_cycles": "counted in the same hour; not knowable in advance",
    "non_car_share": "derived from the target",
    "aadf_all_motor_vehicles": "annual mean of the same flow, same year",
}

EXCLUDED_WITH_REASON = {
    "year": "the split variable; every test value lies outside the training range",
    "link_length_km": "null for 100% of minor roads / 0% of major - road class as missingness",
    "count_point_id": "identifier, not a property; 43k levels would memorise links",
    "road_name": "free text, ~unbounded cardinality",
    "local_authority_name": "374 levels; region_name carries the same geography at usable cardinality",
    "easting": "duplicate of latitude/longitude in another projection",
    "northing": "duplicate of latitude/longitude in another projection",
}


def load_counts(spark) -> DataFrame:
    return spark.read.parquet(str(resolve("processed") / "counts.parquet"))


def build_history(train: DataFrame) -> DataFrame:
    """Per-count-point flow history, from the TRAINING BLOCK ONLY.

    Returned as a lookup so the caller joins identical values onto both blocks.
    Recomputing over the test years would hand the model the very flows it is
    being asked to forecast.
    """
    return train.groupBy("count_point_id").agg(
        F.round(F.mean(LABEL_COLUMN), 2).alias("cp_hist_mean_flow"),
        F.round(F.coalesce(F.stddev(LABEL_COLUMN), F.lit(0.0)), 2).alias("cp_hist_sd_flow"),
        F.countDistinct("year").alias("cp_hist_years"),
    )


def split_by_year(
    counts: DataFrame, train_max_year: int, test_min_year: int, test_max_year: int | None = None
) -> tuple[DataFrame, DataFrame]:
    """Time-aware split on `year`, with an explicit gap where configured."""
    train = counts.filter(F.col("year") <= train_max_year)
    test = counts.filter(F.col("year") >= test_min_year)
    if test_max_year is not None:
        test = test.filter(F.col("year") <= test_max_year)
    return train, test


def attach_history(block: DataFrame, history: DataFrame, defaults: dict) -> DataFrame:
    """Join training-block history onto a block, filling unseen links.

    A count point that appears only in the test years has no training history.
    Dropping those rows would quietly restrict the evaluation to links the model
    has already met, which flatters it; filling with the training-block mean is
    what a planner would assume for a link with no record.
    """
    return block.join(F.broadcast(history), "count_point_id", "left").fillna(defaults)


def prepare(spark, train_max_year: int, test_min_year: int, test_max_year: int | None = None):
    """Return (train, test, info) with leakage-safe features attached."""
    counts = load_counts(spark)
    train_raw, test_raw = split_by_year(counts, train_max_year, test_min_year, test_max_year)

    history = build_history(train_raw)
    stats = history.select(
        F.avg("cp_hist_mean_flow").alias("m"), F.avg("cp_hist_sd_flow").alias("s")
    ).collect()[0]
    defaults = {
        "cp_hist_mean_flow": float(stats["m"] or 0.0),
        "cp_hist_sd_flow": float(stats["s"] or 0.0),
        "cp_hist_years": 0,
    }

    train = attach_history(train_raw, history, defaults)
    test = attach_history(test_raw, history, defaults)

    info = {
        "train_max_year": train_max_year,
        "test_min_year": test_min_year,
        "test_max_year": test_max_year,
        "history_defaults": {k: round(v, 2) if isinstance(v, float) else v
                             for k, v in defaults.items()},
    }
    return train, test, info
