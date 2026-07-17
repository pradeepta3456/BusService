-- Star schema for the GB road traffic count warehouse.
--
-- Two dimensions (count point, region) and two facts. The grain of each fact is
-- stated in its comment, because a fact table with an ambiguous grain is one
-- that will be double-counted.
--
-- What is deliberately NOT here: the 5.27M-row hourly fact at full grain. SQLite
-- exists in this project to serve analytical queries and to demonstrate
-- relational design, not to be the pipeline's bulk store - that is Parquet's
-- job, and it does it in 71 MB with partition pruning. Loading 5.27M rows in
-- would be the wrong tool for no benefit. The hourly fact is therefore loaded at
-- the (count point x hour) profile grain, which is what every query below
-- actually needs.
--
-- SQLite is keyless, so there are no credentials here or anywhere in the repo.
-- Every query in queries.py binds parameters with ? placeholders.

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS fact_hourly_profile;
DROP TABLE IF EXISTS fact_annual_flow;
DROP TABLE IF EXISTS dim_count_point;
DROP TABLE IF EXISTS dim_region;

-- ---------------------------------------------------------------- dimensions

CREATE TABLE dim_region (
    region_id           TEXT PRIMARY KEY,
    region_name         TEXT NOT NULL,
    count_points        INTEGER,
    local_authorities   INTEGER
);

CREATE TABLE dim_count_point (
    count_point_id      TEXT PRIMARY KEY,
    region_id           TEXT,
    local_authority_name TEXT,
    road_name           TEXT,
    -- TM/PM = motorway, TA/PA = A road, MB = minor B, MCU = minor C/unclassified.
    road_category       TEXT NOT NULL,
    road_type           TEXT,
    -- Derived from road_category. This is what link_length_km was really
    -- encoding: DfT publishes a link length for major roads only, so that
    -- column is null for 100% of MCU/MB and 0% of PA/TA/TM/PM. Stating the
    -- distinction directly beats carrying it in a null pattern.
    is_major_road       INTEGER NOT NULL CHECK (is_major_road IN (0, 1)),
    latitude            REAL,
    longitude           REAL,
    first_year          INTEGER,
    last_year           INTEGER,
    FOREIGN KEY (region_id) REFERENCES dim_region (region_id)
);

-- --------------------------------------------------------------------- facts

-- Grain: one row per (count_point_id, hour) - a link's average daily profile
-- across every date it was ever counted. This is the table the peak-hour
-- questions are asked of.
CREATE TABLE fact_hourly_profile (
    count_point_id      TEXT NOT NULL,
    hour                INTEGER NOT NULL CHECK (hour BETWEEN 7 AND 18),
    observations        INTEGER NOT NULL CHECK (observations > 0),
    mean_flow           REAL NOT NULL CHECK (mean_flow >= 0),
    median_flow         REAL,
    sd_flow             REAL,
    mean_cars           REAL,
    mean_buses          REAL,
    mean_cycles         REAL,
    -- mean_flow at this hour / the link's mean across all its hours. >1 means
    -- this hour is busier than the link's typical hour; the peak hour is the
    -- argmax. This is the normalised shape, comparable across a motorway and a
    -- country lane, which raw flow is not.
    peak_index          REAL,
    PRIMARY KEY (count_point_id, hour),
    FOREIGN KEY (count_point_id) REFERENCES dim_count_point (count_point_id)
);

-- Grain: one row per (count_point_id, year) - AADF, the published annual
-- average daily flow. Used for the long-run trend, not as a model target.
CREATE TABLE fact_annual_flow (
    count_point_id      TEXT NOT NULL,
    year                INTEGER NOT NULL CHECK (year BETWEEN 2000 AND 2100),
    aadf_all_motor_vehicles INTEGER CHECK (aadf_all_motor_vehicles >= 0),
    estimation_method   TEXT,
    link_length_km      REAL,
    PRIMARY KEY (count_point_id, year),
    FOREIGN KEY (count_point_id) REFERENCES dim_count_point (count_point_id)
);

-- Indexes chosen for the access patterns in queries.py: ranking hours within a
-- link, filtering by road class, and trending a link across years.
CREATE INDEX idx_profile_hour      ON fact_hourly_profile (hour);
CREATE INDEX idx_profile_peak      ON fact_hourly_profile (peak_index);
CREATE INDEX idx_cp_category       ON dim_count_point (road_category);
CREATE INDEX idx_cp_region         ON dim_count_point (region_id);
CREATE INDEX idx_annual_year       ON fact_annual_flow (year);
