"""Clean the raw hourly counts and write partitioned Parquet.

Tool choice, stated for the report: the raw counts CSV is ~1GB and 5.3M rows.
On an 8GB machine already hosting a JVM it does not fit comfortably in pandas
alongside the AADF table, and every downstream stage joins or aggregates it, so
Spark reads it once with name-bound columns and writes columnar Parquet
partitioned by year. Later stages then read it back with predicate pushdown and
partition pruning, which is what makes a per-year time split cheap.

What cleaning actually does here - each rule is a response to something found in
the data, not a precaution:

  * Hour window. DfT manual counts run 07:00-18:59. Hours 7..18 each hold
    exactly 1/12 of the file across all 26 years (439,136 rows each). Hours
    0,1,3,4,5 hold 8 rows in total, in single years, with impossible flows (5
    vehicles in an hour). Those are source data-entry errors and are dropped.
  * Direction case. `direction_of_travel` arrives as both 'N' and 'n' (and so
    on). Left alone, Spark would treat them as two categories and the
    StringIndexer would learn two separate encodings for the same direction.
  * Date parsing. `count_date` gives day-of-week and month, which is where the
    commuter signal lives.

Run:
    python -m src.process.clean_counts
"""

from __future__ import annotations

import logging
import sys

from pyspark.sql import DataFrame, functions as F

from src.config import load_config, resolve
from src.spark_session import get_spark

LOGGER = logging.getLogger("clean_counts")

# Columns retained from the raw counts, selected BY NAME and cast explicitly.
#
# Passing an explicit StructType alongside header=True binds columns positionally
# and silently ignores the header, so any upstream column re-ordering or
# insertion is read as wrong data rather than as an error. Reading header=True
# with inferSchema=False gives all-string columns in a single pass - no inference
# scan over 1GB - and casting then happens by name.
COUNT_COLUMNS: dict[str, str] = {
    "count_point_id": "string",
    "direction_of_travel": "string",
    "year": "int",
    "count_date": "string",
    "hour": "int",
    "region_id": "string",
    "region_name": "string",
    "local_authority_id": "string",
    "local_authority_name": "string",
    "road_name": "string",
    "road_category": "string",
    "road_type": "string",
    "easting": "double",
    "northing": "double",
    "latitude": "double",
    "longitude": "double",
    "link_length_km": "double",
    "pedal_cycles": "int",
    "two_wheeled_motor_vehicles": "int",
    "cars_and_taxis": "int",
    "buses_and_coaches": "int",
    "LGVs": "int",
    "all_HGVs": "int",
    "all_motor_vehicles": "int",
}

AADF_COLUMNS: dict[str, str] = {
    "count_point_id": "string",
    "year": "int",
    "road_category": "string",
    "link_length_km": "double",
    "estimation_method": "string",
    "all_motor_vehicles": "int",
}


def read_named(spark, path, wanted: dict[str, str]) -> DataFrame:
    """Read a CSV selecting the needed columns by name, failing loudly if absent.

    `try_cast` rather than `cast`: this feed encodes missing numerics as the
    literal string 'NA', and under Spark's ANSI mode a plain cast raises
    CAST_INVALID_INPUT and kills the job on the first one. try_cast yields null
    instead, so missingness becomes data to be measured and reported rather than
    a crash. The nulls are then counted explicitly in clean().
    """
    frame = spark.read.csv(str(path), header=True, inferSchema=False)
    missing = [c for c in wanted if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} missing expected column(s) {missing}. Present: {frame.columns}")
    return frame.select(
        *[
            F.col(c).alias(c) if t == "string"
            else F.expr(f"try_cast(`{c}` AS {t})").alias(c)
            for c, t in wanted.items()
        ]
    )


def clean(spark) -> DataFrame:
    cfg = load_config()
    raw_dir = resolve("raw")
    low_hour, high_hour = cfg["clean"]["valid_hours"]
    target = cfg["target"]["column"]

    counts = read_named(spark, raw_dir / "dft_traffic_counts_raw_counts.csv", COUNT_COLUMNS).cache()
    before = counts.count()

    # --- data quality profile, measured rather than assumed -----------------
    # link_length_km is null for 54.8% of rows, but that is NOT gappy data: it
    # is null for 100% of minor roads (MCU, MB) and 0% of major roads (PA, TA,
    # TM, PM), because DfT only defines junction-to-junction links on the major
    # network. The column is therefore a perfect proxy for road class encoded as
    # missingness. It is kept in the stored table for completeness but excluded
    # from the feature set - see ml/features.py.
    profile = counts.select(
        F.count("*").alias("rows"),
        F.sum(F.col(target).isNull().cast("int")).alias("target_null"),
        F.sum(F.col("link_length_km").isNull().cast("int")).alias("link_length_null"),
        F.sum(F.col("latitude").isNull().cast("int")).alias("lat_null"),
        F.sum(F.col("count_date").isNull().cast("int")).alias("date_null"),
    ).collect()[0]
    LOGGER.info(
        "Quality: rows=%s | target null=%s | link_length null=%s (%.1f%%) | lat null=%s",
        f"{profile['rows']:,}", profile["target_null"],
        f"{profile['link_length_null']:,}",
        100 * profile["link_length_null"] / profile["rows"], profile["lat_null"],
    )

    counts = (
        counts
        .withColumn("direction_of_travel", F.upper(F.trim(F.col("direction_of_travel"))))
        .filter(F.col("hour").between(low_hour, high_hour))
        # The target is 'NA' on 15 rows out of 5.27M. A row with no measured flow
        # cannot supervise a flow model, so it is dropped rather than imputed.
        .filter(F.col(target).isNotNull() & (F.col(target) >= F.lit(cfg["clean"]["min_flow"])))
        .filter(F.col("latitude").isNotNull() & F.col("longitude").isNotNull())
        # Major/minor is what link_length_km was really encoding; state it
        # directly instead of carrying it through a null pattern.
        .withColumn(
            "is_major_road",
            F.when(F.col("road_category").isin("PA", "TA", "TM", "PM"), 1).otherwise(0),
        )
    )

    # Calendar features. count_date is ISO yyyy-MM-dd throughout - verified
    # across all 5,269,632 rows, zero exceptions.
    #
    # `try_to_date` rather than `to_date`: under Spark's default
    # timeParserPolicy=EXCEPTION, to_date THROWS on a value that does not match
    # the pattern, so a single malformed date in a future vintage would abort the
    # job instead of being counted. try_to_date returns null, which the check
    # below turns into a visible warning and a documented row count.
    counts = counts.withColumn(
        "count_date_parsed", F.expr("try_to_date(count_date, 'yyyy-MM-dd')")
    )
    unparsed = counts.filter(F.col("count_date_parsed").isNull()).count()
    if unparsed:
        LOGGER.warning("%s rows have an unparseable count_date; dropping them.", unparsed)
    counts = counts.filter(F.col("count_date_parsed").isNotNull())

    counts = (
        counts
        .withColumn("day_of_week", F.date_format("count_date_parsed", "EEEE"))
        .withColumn("month", F.month("count_date_parsed"))
        .withColumn(
            "is_weekend",
            F.when(F.dayofweek("count_date_parsed").isin(1, 7), 1).otherwise(0),
        )
        # Share of the hour's motorised flow that is not a car. A link's traffic
        # mix is a property of the place, and it is known independently of the
        # hour being predicted only when taken from history - see features.py.
        .withColumn(
            "non_car_share",
            F.when(
                F.col(target) > 0,
                F.round(
                    (F.col(target) - F.coalesce(F.col("cars_and_taxis"), F.lit(0)))
                    / F.col(target), 4,
                ),
            ).otherwise(F.lit(None).cast("double")),
        )
        .drop("count_date")
        .withColumnRenamed("count_date_parsed", "count_date")
    )

    after = counts.count()
    LOGGER.info("Rows: %s -> %s (dropped %s)", f"{before:,}", f"{after:,}", f"{before - after:,}")
    return counts


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stdout
    )
    processed = resolve("processed")
    spark = get_spark("clean_counts")
    try:
        counts = clean(spark)

        # Partition by year: every downstream split filters on it, so this turns
        # the time-aware split into partition pruning rather than a full scan.
        (
            counts
            .repartition(8, "year")
            .write.mode("overwrite")
            .partitionBy("year")
            .parquet(str(processed / "counts.parquet"))
        )
        LOGGER.info("Wrote %s", processed / "counts.parquet")

        aadf = read_named(spark, resolve("raw") / "dft_traffic_counts_aadf.csv", AADF_COLUMNS)
        aadf = aadf.withColumnRenamed("all_motor_vehicles", "aadf_all_motor_vehicles")
        aadf.write.mode("overwrite").parquet(str(processed / "aadf.parquet"))
        LOGGER.info("AADF rows: %s", f"{aadf.count():,}")

        written = spark.read.parquet(str(processed / "counts.parquet"))
        LOGGER.info("Clean counts: %s rows across %s years",
                    f"{written.count():,}", written.select("year").distinct().count())
        written.groupBy("hour").agg(
            F.count("*").alias("rows"),
            F.round(F.mean("all_motor_vehicles"), 1).alias("mean_flow"),
        ).orderBy("hour").show(24, truncate=False)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
