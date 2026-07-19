"""Aggregate the hourly counts into the per-link daily profile.

This is the bridge between the 5.27M-row Parquet fact and the SQLite warehouse.
Spark does the aggregation (millions of rows, groupBy + window); only the ~520k
resulting profile rows land in the database.

`peak_index` is the useful quantity: a link's flow at an hour divided by that
link's mean across all its hours. Raw flow cannot be compared between a motorway
and a country lane - one carries fifty times the other, and that gap swamps the
time-of-day signal entirely. Normalising to the link's own mean strips the level
out and leaves the shape, which is what "peak hour" actually means.

Run:
    python -m src.process.build_profiles
"""

from __future__ import annotations

import logging
import sys

from pyspark.sql import Window, functions as F

from src.config import resolve
from src.spark_session import get_spark

LOGGER = logging.getLogger("build_profiles")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stdout
    )
    processed = resolve("processed")
    spark = get_spark("build_profiles")
    try:
        counts = spark.read.parquet(str(processed / "counts.parquet"))

        # --- per (count point, hour) profile ---
        profile = counts.groupBy("count_point_id", "hour").agg(
            F.count("*").alias("observations"),
            F.round(F.mean("all_motor_vehicles"), 2).alias("mean_flow"),
            F.round(F.expr("percentile_approx(all_motor_vehicles, 0.5)"), 2).alias("median_flow"),
            F.round(F.coalesce(F.stddev("all_motor_vehicles"), F.lit(0.0)), 2).alias("sd_flow"),
            F.round(F.mean("cars_and_taxis"), 2).alias("mean_cars"),
            F.round(F.mean("buses_and_coaches"), 2).alias("mean_buses"),
            F.round(F.mean("pedal_cycles"), 2).alias("mean_cycles"),
        )

        # Link mean across its observed hours -> peak_index. A window rather than a
        # second groupBy + join: one shuffle instead of two.
        link_window = Window.partitionBy("count_point_id")
        profile = (
            profile
            .withColumn("link_mean", F.mean("mean_flow").over(link_window))
            .withColumn(
                "peak_index",
                F.when(F.col("link_mean") > 0,
                       F.round(F.col("mean_flow") / F.col("link_mean"), 4)),
            )
            .drop("link_mean")
        )
        profile.write.mode("overwrite").parquet(str(processed / "hourly_profile.parquet"))
        LOGGER.info("hourly_profile rows: %s", f"{profile.count():,}")

        # --- count point dimension ---
        dim = counts.groupBy("count_point_id").agg(
            F.first("region_id", ignorenulls=True).alias("region_id"),
            F.first("local_authority_name", ignorenulls=True).alias("local_authority_name"),
            F.first("road_name", ignorenulls=True).alias("road_name"),
            F.first("road_category", ignorenulls=True).alias("road_category"),
            F.first("road_type", ignorenulls=True).alias("road_type"),
            F.max("is_major_road").alias("is_major_road"),
            F.round(F.first("latitude", ignorenulls=True), 6).alias("latitude"),
            F.round(F.first("longitude", ignorenulls=True), 6).alias("longitude"),
            F.min("year").alias("first_year"),
            F.max("year").alias("last_year"),
        )
        dim.write.mode("overwrite").parquet(str(processed / "dim_count_point.parquet"))
        LOGGER.info("dim_count_point rows: %s", f"{dim.count():,}")

        # --- region dimension ---
        region = counts.groupBy("region_id", "region_name").agg(
            F.countDistinct("count_point_id").alias("count_points"),
            F.countDistinct("local_authority_name").alias("local_authorities"),
        )
        region.write.mode("overwrite").parquet(str(processed / "dim_region.parquet"))
        LOGGER.info("dim_region rows: %s", region.count())

        LOGGER.info("Peak index sanity - mean by hour (should be bimodal around 1.0):")
        (
            profile.groupBy("hour")
            .agg(F.round(F.mean("peak_index"), 3).alias("mean_peak_index"))
            .orderBy("hour").show(24, truncate=False)
        )
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
