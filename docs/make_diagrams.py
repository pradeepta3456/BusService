"""Generate architecture.png and schema.png for the report (required deliverables).

Run:
    python docs/make_diagrams.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
BLUE, EDGE = "#eaf2fb", "#2b5f8a"


def box(ax, x, y, w, h, title, lines, fc=BLUE):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                        lw=1.3, ec=EDGE, fc=fc))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top",
            fontsize=9, fontweight="bold")
    for i, line in enumerate(lines):
        ax.text(x + 0.1, y + h - 0.46 - i * 0.24, line, ha="left", va="top", fontsize=7)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=EDGE))


# ------------------------------------------------------------- architecture
fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")
ax.text(6.5, 6.95, "Pipeline Architecture - Peak-Hour Traffic Demand Forecasting",
        ha="center", fontsize=12, fontweight="bold")

box(ax, 0.3, 4.9, 2.9, 1.7, "INGESTION (Python)",
    ["download_counts.py", "  DfT raw hourly (1.0 GB)", "  DfT AADF (150 MB)", "",
     "I/O-bound; two HTTP GETs.", "Spark adds nothing here."])
box(ax, 3.6, 4.9, 3.1, 1.7, "PROCESSING (PySpark)",
    ["clean_counts.py", "  try_cast, hour window, calendar", "build_profiles.py",
     "  per-link daily profile + peak_index", "", "1 GB CSV -> 71 MB Parquet, by year"])
box(ax, 7.1, 5.5, 2.6, 1.1, "STORAGE (SQLite)",
    ["dim_count_point / dim_region", "fact_hourly_profile", "fact_annual_flow"])
box(ax, 7.1, 4.15, 2.6, 1.15, "ML (MLlib)",
    ["leakage-safe features", "time-aware split by year", "LR / RF / GBT + CV"])
box(ax, 10.1, 4.9, 2.6, 1.7, "VISUALISATION",
    ["hourly profile", "COVID break, modal shift", "predicted vs actual",
     "skill decomposition", "feature importance"])
arrow(ax, 3.2, 5.75, 3.6, 5.75); arrow(ax, 6.7, 5.95, 7.1, 6.0)
arrow(ax, 6.7, 5.4, 7.1, 4.8); arrow(ax, 9.7, 6.0, 10.1, 5.95)
arrow(ax, 9.7, 4.7, 10.1, 5.4)

box(ax, 0.3, 2.35, 12.4, 1.5, "DATA SOURCE - DfT GB Road Traffic Counts (real, open, OGL v3.0, key-free)",
    ["raw hourly counts: 5,269,632 rows x 35 cols - 43,401 count points, 26 years (2000-2025), 12-hour day (07:00-18:59)",
     "AADF: 600,551 rows - annual average daily flow per count point per year",
     "Cleaning: 8 impossible-hour rows + 15 'NA' targets dropped -> 5,269,609. No synthetic augmentation anywhere.",
     "100k requirement met 53x over by one real table."], fc="#f5f5f5")

box(ax, 0.3, 0.35, 12.4, 1.7, "LEAKAGE DISCIPLINE (measured, not assumed)",
    ["two_wheeled + cars + buses + LGVs + all_HGVs == all_motor_vehicles  on 5,269,608 / 5,269,609 rows (100.0000%)",
     "=> every mode column is a COMPONENT of the target, not a correlate. All barred; assert_no_leakage() aborts the run if one returns.",
     "year excluded: it IS the split variable.   link_length_km excluded: null for 100% of minor roads, 0% of major - road class as missingness.",
     "Split by year, never random: train <=2019 -> forecast 2023-25 (primary, spans COVID); train <=2016 -> 2017-19 (control, no break)."],
    fc="#eef7ee")
fig.tight_layout(); fig.savefig(OUT / "architecture.png", dpi=160); plt.close(fig)

# -------------------------------------------------------------------- schema
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
ax.text(6, 6.75, "Database Schema - SQLite Warehouse", ha="center",
        fontsize=12, fontweight="bold")

box(ax, 0.4, 4.5, 2.8, 1.9, "dim_region",
    ["PK region_id", "region_name", "count_points", "local_authorities"])
box(ax, 4.3, 4.3, 3.4, 2.1, "dim_count_point",
    ["PK count_point_id", "FK region_id -> dim_region", "road_name / category / type",
     "is_major_road  CHECK 0|1", "latitude / longitude", "first_year / last_year"])
box(ax, 8.8, 4.5, 2.8, 1.9, "fact_annual_flow",
    ["PK (count_point_id, year)", "FK -> dim_count_point", "aadf_all_motor_vehicles",
     "estimation_method", "", "grain: link x year"], fc="#eef7ee")
box(ax, 3.6, 1.5, 4.8, 2.3, "fact_hourly_profile",
    ["PK (count_point_id, hour)", "FK -> dim_count_point",
     "observations · mean/median/sd_flow", "mean_cars / mean_buses / mean_cycles",
     "hour        CHECK BETWEEN 7 AND 18", "mean_flow   CHECK >= 0",
     "peak_index  = hour flow / link's mean hour", "", "grain: link x hour"], fc="#eef7ee")
arrow(ax, 4.3, 5.4, 3.2, 5.4); arrow(ax, 8.8, 5.4, 7.7, 5.4)
arrow(ax, 5.6, 4.3, 5.6, 3.8)

ax.text(6, 1.05, "The 5.27M-row hourly fact stays in Parquet: SQLite is here to serve "
        "analytical queries, not to be the bulk store.", ha="center", fontsize=8, style="italic")
ax.text(6, 0.72, "All access via parameterised ? placeholders (src/db/queries.py); "
        "injection resistance asserted in tests/test_queries.py.",
        ha="center", fontsize=8, style="italic")
ax.text(6, 0.39, "No credentials anywhere: SQLite is keyless and the DfT source is open and key-free.",
        ha="center", fontsize=8, style="italic")
fig.tight_layout(); fig.savefig(OUT / "schema.png", dpi=160); plt.close(fig)

print("wrote docs/architecture.png and docs/schema.png")
