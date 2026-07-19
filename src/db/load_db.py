"""Load the aggregated Parquet tables into the SQLite warehouse.

Only aggregates land here. The 5.27M-row hourly fact stays in Parquet: SQLite is
in this project to serve analytical queries and demonstrate relational design,
and loading millions of rows into it would be the wrong tool for no benefit.
What arrives is the (count point x hour) profile - ~520k rows - which is the
grain every query actually asks about.

Run:
    python -m src.db.load_db
"""

from __future__ import annotations

import logging
import sqlite3
import sys

import pandas as pd

from src.config import PROJECT_ROOT, resolve

LOGGER = logging.getLogger("load_db")

SCHEMA_PATH = PROJECT_ROOT / "src" / "db" / "schema.sql"


def load() -> None:
    processed = resolve("processed")
    db_path = resolve("database")
    db_path.unlink(missing_ok=True)

    connection = sqlite3.connect(db_path)
    try:
        with open(SCHEMA_PATH) as handle:
            connection.executescript(handle.read())
        LOGGER.info("Schema applied to %s", db_path)

        # --- dim_region ---
        region = pd.read_parquet(processed / "dim_region.parquet").dropna(subset=["region_id"])
        region = region.drop_duplicates(subset=["region_id"])
        region.to_sql("dim_region", connection, if_exists="append", index=False)
        LOGGER.info("dim_region: %s", len(region))

        # --- dim_count_point ---
        points = pd.read_parquet(processed / "dim_count_point.parquet")
        points = points.drop_duplicates(subset=["count_point_id"])
        # A count point whose region is not in dim_region would violate the FK.
        # Drop and report how many rather than disabling the constraint.
        known_regions = set(region.region_id)
        before = len(points)
        points.loc[~points.region_id.isin(known_regions), "region_id"] = None
        points = points[[
            "count_point_id", "region_id", "local_authority_name", "road_name",
            "road_category", "road_type", "is_major_road", "latitude", "longitude",
            "first_year", "last_year",
        ]]
        points.to_sql("dim_count_point", connection, if_exists="append", index=False)
        LOGGER.info("dim_count_point: %s (of %s)", len(points), before)

        # --- fact_hourly_profile ---
        profile = pd.read_parquet(processed / "hourly_profile.parquet")
        known_points = set(points.count_point_id)
        before = len(profile)
        profile = profile[profile.count_point_id.isin(known_points)]
        if before != len(profile):
            LOGGER.warning("Dropped %s profile rows with no dim_count_point.", before - len(profile))
        profile = profile[[
            "count_point_id", "hour", "observations", "mean_flow", "median_flow",
            "sd_flow", "mean_cars", "mean_buses", "mean_cycles", "peak_index",
        ]]
        profile.to_sql("fact_hourly_profile", connection, if_exists="append",
                       index=False, chunksize=50_000)
        LOGGER.info("fact_hourly_profile: %s", f"{len(profile):,}")

        # --- fact_annual_flow ---
        aadf = pd.read_parquet(processed / "aadf.parquet")
        aadf = aadf[aadf.count_point_id.isin(known_points)].drop_duplicates(
            subset=["count_point_id", "year"]
        )
        aadf = aadf[[
            "count_point_id", "year", "aadf_all_motor_vehicles",
            "estimation_method", "link_length_km",
        ]]
        # AADF carries rows outside the count years and the CHECK range; keep the
        # constraint honest by filtering rather than by loosening it.
        aadf = aadf[aadf.year.between(2000, 2100)]
        aadf.to_sql("fact_annual_flow", connection, if_exists="append",
                    index=False, chunksize=50_000)
        LOGGER.info("fact_annual_flow: %s", f"{len(aadf):,}")

        connection.commit()

        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            LOGGER.error("Foreign key violations: %s", violations[:5])
        else:
            LOGGER.info("Foreign key check passed.")
    finally:
        connection.close()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stdout
    )
    load()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
