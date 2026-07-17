"""Parameterised analytical queries against the SQLite warehouse.

Every query binds user-supplied values with `?` placeholders passed to `execute`
as a separate tuple. None builds SQL by string formatting or f-strings.

Why that matters here, concretely: `link_profile(conn, count_point_id)` takes an
id that in a planning dashboard would arrive from a URL. Interpolated into the
SQL, a caller could pass

    ' UNION SELECT name, sql, 1, 1, 1 FROM sqlite_master --

and read the schema, or close the statement and append their own. Bound as a
parameter, the driver sends the value out-of-band and it is only ever a value -
never parseable SQL. `tests/test_queries.py` asserts exactly that.

Run for a demo of each:
    python -m src.db.queries
"""

from __future__ import annotations

import sqlite3
import sys

import pandas as pd

from src.config import resolve


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(resolve("database"))
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def link_profile(connection: sqlite3.Connection, count_point_id: str,
                 min_observations: int = 1) -> pd.DataFrame:
    """The daily shape of one link: flow and peak index by hour."""
    sql = """
        SELECT
            p.hour,
            p.observations,
            ROUND(p.mean_flow, 1)  AS mean_flow,
            ROUND(p.peak_index, 3) AS peak_index,
            c.road_name,
            c.road_category
        FROM fact_hourly_profile AS p
        JOIN dim_count_point AS c ON c.count_point_id = p.count_point_id
        WHERE p.count_point_id = ?
          AND p.observations >= ?
        ORDER BY p.hour
    """
    return pd.read_sql_query(sql, connection, params=(count_point_id, min_observations))


def peak_hour_by_road_class(connection: sqlite3.Connection,
                            min_observations: int = 5) -> pd.DataFrame:
    """When does each class of road peak, and how sharply?

    THE headline query. peak_index is normalised to each link's own mean, so a
    motorway and a country lane are comparable - which raw flow never is.
    """
    sql = """
        SELECT
            c.road_category,
            p.hour,
            COUNT(*)                       AS links,
            ROUND(AVG(p.peak_index), 4)    AS mean_peak_index,
            ROUND(AVG(p.mean_flow), 1)     AS mean_flow
        FROM fact_hourly_profile AS p
        JOIN dim_count_point AS c ON c.count_point_id = p.count_point_id
        WHERE p.observations >= ?
        GROUP BY c.road_category, p.hour
        ORDER BY c.road_category, p.hour
    """
    return pd.read_sql_query(sql, connection, params=(min_observations,))


def peakiest_links(connection: sqlite3.Connection, limit: int = 10,
                   min_observations: int = 5) -> pd.DataFrame:
    """Links with the sharpest single-hour peak - where capacity is strained.

    Exactly the list an urban planner would ask for: roads whose busiest hour is
    furthest above their own average, i.e. built for a peak they meet once a day.
    """
    sql = """
        SELECT
            c.road_name,
            c.road_category,
            c.local_authority_name,
            p.hour                         AS peak_hour,
            ROUND(p.peak_index, 3)         AS peak_index,
            ROUND(p.mean_flow, 1)          AS peak_hour_flow,
            p.observations
        FROM fact_hourly_profile AS p
        JOIN dim_count_point AS c ON c.count_point_id = p.count_point_id
        WHERE p.observations >= ?
        ORDER BY p.peak_index DESC
        LIMIT ?
    """
    return pd.read_sql_query(sql, connection, params=(min_observations, limit))


def annual_trend(connection: sqlite3.Connection, first_year: int = 2000,
                 last_year: int = 2025) -> pd.DataFrame:
    """Long-run AADF trend - the 2020 collapse and what followed."""
    sql = """
        SELECT
            f.year,
            COUNT(*)                                AS links,
            ROUND(AVG(f.aadf_all_motor_vehicles), 1) AS mean_aadf
        FROM fact_annual_flow AS f
        WHERE f.year BETWEEN ? AND ?
          AND f.aadf_all_motor_vehicles IS NOT NULL
        GROUP BY f.year
        ORDER BY f.year
    """
    return pd.read_sql_query(sql, connection, params=(first_year, last_year))


def region_summary(connection: sqlite3.Connection, min_links: int = 1) -> pd.DataFrame:
    """Regional comparison of flow level and peak sharpness."""
    sql = """
        SELECT
            r.region_name,
            COUNT(DISTINCT c.count_point_id) AS count_points,
            ROUND(AVG(p.mean_flow), 1)       AS mean_flow,
            ROUND(MAX(p.peak_index), 3)      AS sharpest_peak_index
        FROM fact_hourly_profile AS p
        JOIN dim_count_point AS c ON c.count_point_id = p.count_point_id
        JOIN dim_region AS r      ON r.region_id = c.region_id
        GROUP BY r.region_name
        HAVING COUNT(DISTINCT c.count_point_id) >= ?
        ORDER BY mean_flow DESC
    """
    return pd.read_sql_query(sql, connection, params=(min_links,))


def main() -> int:
    connection = connect()
    try:
        print("\n=== Peak hour by road class (mean peak index) ===")
        data = peak_hour_by_road_class(connection)
        pivot = data.pivot(index="hour", columns="road_category", values="mean_peak_index")
        print(pivot.to_string())

        print("\n=== Sharpest-peaking links ===")
        print(peakiest_links(connection, limit=10).to_string(index=False))

        print("\n=== Regional summary ===")
        print(region_summary(connection).to_string(index=False))

        print("\n=== Annual AADF trend ===")
        print(annual_trend(connection).to_string(index=False))

        sample = connection.execute(
            "SELECT count_point_id FROM fact_hourly_profile LIMIT 1"
        ).fetchone()
        if sample:
            print(f"\n=== Daily profile for count point {sample[0]} ===")
            print(link_profile(connection, sample[0]).to_string(index=False))
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
