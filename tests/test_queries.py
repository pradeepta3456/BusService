"""Tests that the SQL layer resists injection and that the schema holds.

The injection tests are the point of this file: they prove the `?` placeholders
in queries.py are doing real work, rather than the claim resting on inspection of
the source.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.db import queries


@pytest.fixture()
def connection():
    """An in-memory database with the real schema and one link's profile."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(open("src/db/schema.sql").read())
    conn.execute(
        "INSERT INTO dim_region (region_id, region_name) VALUES (?, ?)",
        ("1", "Test Region"),
    )
    conn.execute(
        "INSERT INTO dim_count_point (count_point_id, region_id, local_authority_name,"
        " road_name, road_category, road_type, is_major_road, latitude, longitude)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("CP1", "1", "Test LA", "A1", "PA", "Major", 1, 53.8, -1.55),
    )
    for hour, flow, index in [(8, 900.0, 1.4), (12, 500.0, 0.8)]:
        conn.execute(
            "INSERT INTO fact_hourly_profile (count_point_id, hour, observations,"
            " mean_flow, peak_index) VALUES (?, ?, ?, ?, ?)",
            ("CP1", hour, 10, flow, index),
        )
    conn.commit()
    yield conn
    conn.close()


def test_link_profile_returns_the_link(connection):
    result = queries.link_profile(connection, "CP1")
    assert len(result) == 2
    assert set(result["hour"]) == {8, 12}


def test_link_profile_unknown_id_is_empty(connection):
    assert len(queries.link_profile(connection, "NOPE")) == 0


def test_injection_tautology_is_treated_as_a_value(connection):
    """The classic payload must match nothing, not everything.

    With string interpolation this returns every row. Bound as a parameter it is
    just a count_point_id containing punctuation.
    """
    assert len(queries.link_profile(connection, "' OR '1'='1")) == 0


def test_injection_cannot_terminate_the_statement(connection):
    payload = "CP1'; DROP TABLE fact_hourly_profile; --"
    assert len(queries.link_profile(connection, payload)) == 0

    remaining = connection.execute("SELECT COUNT(*) FROM fact_hourly_profile").fetchone()[0]
    assert remaining == 2, "The table must survive the payload"


def test_injection_via_union_cannot_read_the_schema(connection):
    payload = "' UNION SELECT name, sql, 1, 1, 1, 1 FROM sqlite_master --"
    assert len(queries.link_profile(connection, payload)) == 0


def test_injection_through_a_numeric_parameter(connection):
    """Integer params are bound too - not only the obvious string ones.

    Assert the PROPERTY (the table survives), not the mechanism. SQLite rejects
    this particular payload with 'datatype mismatch' because LIMIT wants a
    number, and pandas re-wraps that as DatabaseError - but pinning the exception
    class would be testing which library raises what, not whether the injection
    landed. The only thing that matters is that DROP TABLE never executed.
    """
    payload = "10; DROP TABLE dim_region"
    try:
        result = queries.peakiest_links(connection, limit=payload)
        # If it did not raise, it must at least not have run the payload.
        assert len(result) >= 0
    except Exception:
        pass  # rejected outright - equally acceptable

    survived = connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='dim_region'"
    ).fetchone()[0]
    assert survived == 1, "dim_region was dropped - the parameter was not bound"
    assert connection.execute("SELECT COUNT(*) FROM dim_region").fetchone()[0] == 1


def test_check_constraint_rejects_impossible_hour(connection):
    """DfT counts run 07:00-18:59; hour 3 must be refused by the schema."""
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO fact_hourly_profile (count_point_id, hour, observations, mean_flow)"
            " VALUES (?, ?, ?, ?)",
            ("CP1", 3, 5, 100.0),
        )


def test_check_constraint_rejects_negative_flow(connection):
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO fact_hourly_profile (count_point_id, hour, observations, mean_flow)"
            " VALUES (?, ?, ?, ?)",
            ("CP1", 9, 5, -1.0),
        )


def test_foreign_key_rejects_unknown_count_point(connection):
    connection.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO fact_hourly_profile (count_point_id, hour, observations, mean_flow)"
            " VALUES (?, ?, ?, ?)",
            ("GHOST", 9, 5, 100.0),
        )


def test_peakiest_links_orders_by_peak_index(connection):
    result = queries.peakiest_links(connection, limit=5, min_observations=1)
    assert list(result["peak_index"]) == sorted(result["peak_index"], reverse=True)
