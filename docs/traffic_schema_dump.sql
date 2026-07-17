BEGIN TRANSACTION;
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
CREATE TABLE dim_region (
    region_id           TEXT PRIMARY KEY,
    region_name         TEXT NOT NULL,
    count_points        INTEGER,
    local_authorities   INTEGER
);
CREATE TABLE fact_annual_flow (
    count_point_id      TEXT NOT NULL,
    year                INTEGER NOT NULL CHECK (year BETWEEN 2000 AND 2100),
    aadf_all_motor_vehicles INTEGER CHECK (aadf_all_motor_vehicles >= 0),
    estimation_method   TEXT,
    link_length_km      REAL,
    PRIMARY KEY (count_point_id, year),
    FOREIGN KEY (count_point_id) REFERENCES dim_count_point (count_point_id)
);
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
CREATE INDEX idx_profile_hour      ON fact_hourly_profile (hour);
CREATE INDEX idx_profile_peak      ON fact_hourly_profile (peak_index);
CREATE INDEX idx_cp_category       ON dim_count_point (road_category);
CREATE INDEX idx_cp_region         ON dim_count_point (region_id);
CREATE INDEX idx_annual_year       ON fact_annual_flow (year);
COMMIT;