"""Exploratory data analysis figures.

Every aggregation happens in Spark; only the resulting few-hundred-row summaries
are collected to pandas for drawing. That division is the point - the count table
is 5.27M rows and does not need to cross the driver boundary to produce a bar
chart.

Run:
    python -m src.viz.eda_figures
"""

from __future__ import annotations

import logging
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import functions as F

from src.config import resolve
from src.spark_session import get_spark

LOGGER = logging.getLogger("eda_figures")

FIGURE_DPI = 150
BLUE, LIGHT, RED, GREEN = "#2b5f8a", "#7aa6c2", "#c1553b", "#4a8c5f"

TARGET = "all_motor_vehicles"


def _save(fig, name: str) -> None:
    target = resolve("outputs") / name
    fig.tight_layout()
    fig.savefig(target, dpi=FIGURE_DPI)
    plt.close(fig)
    LOGGER.info("Wrote %s", target)


def plot_data_scale(spark) -> None:
    """Record counts across the ingested tables - the 100k+ scale evidence."""
    processed = resolve("processed")
    counts = {
        "hourly counts\n(the fact)": spark.read.parquet(str(processed / "counts.parquet")).count(),
        "AADF\n(annual)": spark.read.parquet(str(processed / "aadf.parquet")).count(),
    }
    for name, label in [("hourly_profile", "profile\n(cp x hour)"),
                        ("dim_count_point", "count points")]:
        path = processed / f"{name}.parquet"
        if path.exists():
            counts[label] = spark.read.parquet(str(path)).count()

    ordered = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(list(ordered), list(ordered.values()), color=BLUE)
    ax.axhline(100_000, linestyle="--", color=RED, linewidth=1.2,
               label="Brief's 100,000-record requirement")
    # Log scale: the fact table is ~9x the next table and ~120x the smallest, so
    # a linear axis would flatten the remaining tables onto the baseline.
    ax.set_yscale("log")
    ax.set_ylabel("Records (log scale)")
    ax.set_title("Ingested Data Scale - DfT Road Traffic Counts (all real, no augmentation)")
    ax.legend()
    for bar, value in zip(bars, ordered.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}",
                ha="center", va="bottom", fontsize=8)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    _save(fig, "eda_data_scale.png")


def plot_hourly_profile(spark) -> None:
    """The headline chart: the bimodal commuter day."""
    counts = spark.read.parquet(str(resolve("processed") / "counts.parquet"))
    data = (
        counts.groupBy("hour")
        .agg(
            F.count("*").alias("n"),
            F.mean(TARGET).alias("mean_flow"),
            F.expr(f"percentile_approx({TARGET}, 0.5)").alias("median_flow"),
            F.expr(f"percentile_approx({TARGET}, 0.9)").alias("p90_flow"),
        )
        .orderBy("hour").toPandas()
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(data["hour"], data["mean_flow"], marker="o", color=BLUE, label="Mean")
    ax.plot(data["hour"], data["median_flow"], marker="s", color=LIGHT,
            linestyle="--", label="Median")
    peak = data.loc[data.mean_flow.idxmax()]
    trough = data.loc[data.mean_flow.idxmin()]
    ax.annotate(f"PM peak {int(peak.hour)}:00\n{peak.mean_flow:.0f} veh/h",
                (peak.hour, peak.mean_flow), textcoords="offset points",
                xytext=(0, 14), ha="center", fontsize=8, color=RED)
    ax.annotate(f"trough {int(trough.hour)}:00\n{trough.mean_flow:.0f} veh/h",
                (trough.hour, trough.mean_flow), textcoords="offset points",
                xytext=(0, -30), ha="center", fontsize=8, color=RED)
    ax.set_xlabel("Hour of day (DfT counts run 07:00-18:59 only)")
    ax.set_ylabel("Motor vehicles per hour")
    ax.set_title(f"The Shape of the Day - GB Road Traffic, 2000-2025\n"
                 f"peak/trough ratio = {peak.mean_flow / trough.mean_flow:.3f}")
    ax.set_xticks(data["hour"])
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, "eda_hourly_profile.png")


def plot_covid_break(spark) -> None:
    """Flow by year - the structural break the time-aware split has to cross."""
    counts = spark.read.parquet(str(resolve("processed") / "counts.parquet"))
    data = (
        counts.groupBy("year")
        .agg(F.count("*").alias("observations"), F.mean(TARGET).alias("mean_flow"))
        .orderBy("year").toPandas()
    )

    fig, (ax_flow, ax_obs) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax_flow.plot(data["year"], data["mean_flow"], marker="o", color=BLUE)
    ax_flow.axvspan(2020, 2022, color=RED, alpha=0.12, label="COVID period")
    ax_flow.axvline(2019.5, linestyle="--", color=GREEN, linewidth=1,
                    label="Primary split boundary")
    ax_flow.set_xlabel("Year")
    ax_flow.set_ylabel("Mean flow (veh/h)")
    ax_flow.set_title("Mean Hourly Flow by Year")
    ax_flow.legend(fontsize=8)
    ax_flow.grid(alpha=0.25)

    ax_obs.bar(data["year"], data["observations"], color=LIGHT)
    ax_obs.axvspan(2020, 2022, color=RED, alpha=0.12)
    ax_obs.set_xlabel("Year")
    ax_obs.set_ylabel("Counts taken")
    ax_obs.set_title("Counting Effort by Year\n(survey volume halves from 2020)")
    ax_obs.grid(alpha=0.25, axis="y")
    _save(fig, "eda_covid_break.png")


def plot_modal_shift(spark) -> None:
    """26 years of modal change, indexed so modes of different size compare."""
    counts = spark.read.parquet(str(resolve("processed") / "counts.parquet"))
    modes = ["pedal_cycles", "cars_and_taxis", "buses_and_coaches", "LGVs", "all_HGVs"]
    data = (
        counts.groupBy("year")
        .agg(*[F.mean(m).alias(m) for m in modes])
        .orderBy("year").toPandas()
    )

    fig, ax = plt.subplots(figsize=(9.5, 5))
    base = data[data.year == data.year.min()].iloc[0]
    colours = [GREEN, BLUE, RED, LIGHT, "#8a6fb0"]
    for mode, colour in zip(modes, colours):
        # Indexed to 2000 = 100. Cars average ~400/h and cycles ~5/h, so raw
        # counts on one axis would render every non-car mode as a flat line.
        ax.plot(data["year"], 100 * data[mode] / base[mode], marker="o",
                markersize=3, color=colour, label=mode.replace("_", " "))
    ax.axhline(100, color="#444", linewidth=0.8)
    ax.axvspan(2020, 2022, color=RED, alpha=0.10, label="COVID")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Index ({int(data.year.min())} = 100)")
    ax.set_title("Modal Shift on Britain's Roads, Indexed to 2000")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, "eda_modal_shift.png")


def plot_profile_by_road_class(spark) -> None:
    """Does a motorway peak like a country lane? Normalised so they compare."""
    counts = spark.read.parquet(str(resolve("processed") / "counts.parquet"))
    hourly = counts.groupBy("road_category", "hour").agg(F.mean(TARGET).alias("mean_flow"))
    # Normalise each road class to its own mean across hours: raw flow differs
    # 50-fold between classes, which would hide the shape entirely.
    from pyspark.sql import Window
    window = Window.partitionBy("road_category")
    data = (
        hourly.withColumn("class_mean", F.mean("mean_flow").over(window))
        .withColumn("peak_index", F.col("mean_flow") / F.col("class_mean"))
        .orderBy("road_category", "hour").toPandas()
    )

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for category, group in data.groupby("road_category"):
        ax_raw.plot(group["hour"], group["mean_flow"], marker="o", markersize=3, label=category)
        ax_norm.plot(group["hour"], group["peak_index"], marker="o", markersize=3, label=category)
    ax_raw.set_xlabel("Hour"); ax_raw.set_ylabel("Mean flow (veh/h)")
    ax_raw.set_title("Raw Flow by Road Class\n(level dominates; shape invisible)")
    ax_raw.legend(fontsize=7); ax_raw.grid(alpha=0.25)

    ax_norm.axhline(1.0, color="#444", linewidth=0.8)
    ax_norm.set_xlabel("Hour"); ax_norm.set_ylabel("Flow / that class's mean hour")
    ax_norm.set_title("Normalised Shape\n(the actual peak-hour question)")
    ax_norm.legend(fontsize=7); ax_norm.grid(alpha=0.25)
    _save(fig, "eda_profile_by_road_class.png")


def plot_flow_distribution(spark) -> None:
    """The right-skewed target, with the moments the brief names."""
    counts = spark.read.parquet(str(resolve("processed") / "counts.parquet"))
    stats = counts.select(
        F.round(F.mean(TARGET), 1).alias("mean"),
        F.round(F.expr(f"percentile_approx({TARGET}, 0.5)"), 1).alias("median"),
        F.round(F.stddev(TARGET), 1).alias("sd"),
        F.round(F.skewness(TARGET), 3).alias("skewness"),
        F.round(F.kurtosis(TARGET), 3).alias("kurtosis"),
    ).collect()[0]
    LOGGER.info("Target moments: %s", stats.asDict())

    binned = (
        counts.withColumn("bin", (F.floor(F.col(TARGET) / 100) * 100).cast("int"))
        .filter(F.col("bin") <= 5000)
        .groupBy("bin").count().orderBy("bin").toPandas()
    )

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(binned["bin"], binned["count"], width=95, color=BLUE)
    ax.axvline(stats["mean"], color=RED, linestyle="--", label=f"mean {stats['mean']:.0f}")
    ax.axvline(stats["median"], color=GREEN, linestyle=":", label=f"median {stats['median']:.0f}")
    ax.set_yscale("log")
    ax.set_xlabel("Motor vehicles per hour (binned; tail beyond 5,000 clipped for display)")
    ax.set_ylabel("Observations (log scale)")
    ax.set_title(f"Distribution of Hourly Flow - skewness {stats['skewness']}, "
                 f"kurtosis {stats['kurtosis']}\nmean > median: the long right tail of "
                 f"busy links")
    ax.legend()
    _save(fig, "eda_flow_distribution.png")


def plot_correlation_heatmap(spark) -> None:
    """Correlations among the numeric features and the target."""
    counts = spark.read.parquet(str(resolve("processed") / "counts.parquet"))
    columns = ["hour", "is_weekend", "month", "is_major_road",
               "latitude", "longitude", TARGET]
    frame = counts.select(*[F.col(c).cast("double") for c in columns]).dropna()

    matrix = np.eye(len(columns))
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = frame.stat.corr(columns[i], columns[j])
            matrix[i, j] = matrix[j, i] = value if value is not None else np.nan

    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns))); ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(columns, fontsize=8)
    for i in range(len(columns)):
        for j in range(len(columns)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(matrix[i, j]) > 0.55 else "#222")
    ax.set_title("Feature Correlations (Spark-computed, Pearson)\n"
                 "no single feature carries the target", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "eda_correlation_heatmap.png")


def plot_data_quality(spark) -> None:
    """The link_length_km finding, drawn.

    54.8% of rows have no link length - which reads as gappy data until you
    split it by road class and find a perfect 100%/0% break. DfT defines
    junction-to-junction links only on the major network, so the column is road
    class encoded as missingness. This figure is the argument for excluding it.
    """
    counts = spark.read.parquet(str(resolve("processed") / "counts.parquet"))
    data = (
        counts.groupBy("road_category")
        .agg(
            F.count("*").alias("rows"),
            F.round(100 * F.mean(F.col("link_length_km").isNull().cast("double")), 1).alias("pct_null"),
        )
        .orderBy(F.desc("rows")).toPandas()
    )

    fig, (ax_null, ax_rows) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colours = [RED if v > 50 else GREEN for v in data["pct_null"]]
    bars = ax_null.bar(data["road_category"], data["pct_null"], color=colours)
    ax_null.set_ylim(0, 108)
    ax_null.set_ylabel("% of rows with link_length_km missing")
    ax_null.set_xlabel("Road category")
    ax_null.set_title("link_length_km Is Not Missing Data - It Is Road Class\n"
                      "100% missing on minor roads (MCU, MB), 0% on major (PA/TA/TM/PM)",
                      fontsize=10)
    for bar, value in zip(bars, data["pct_null"]):
        ax_null.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%",
                     ha="center", fontsize=9)

    # Compute the identity rather than hard-coding it. The raw file gives
    # MCU+MB = 2,888,208, but this figure is drawn from the CLEANED table, where
    # 23 rows have been dropped - so a hard-coded number would be quietly wrong
    # by 13 and nobody would notice.
    minor_rows = int(data.loc[data.road_category.isin(["MCU", "MB"]), "rows"].sum())
    null_rows = int(
        (data["rows"] * data["pct_null"] / 100).round().sum()
    )
    ax_rows.bar(data["road_category"], data["rows"], color=BLUE)
    ax_rows.set_ylabel("Rows")
    ax_rows.set_xlabel("Road category")
    ax_rows.set_title(f"Rows by Road Category\n"
                      f"MCU + MB = {minor_rows:,} = exactly the {null_rows:,} nulls",
                      fontsize=10)
    for i, value in enumerate(data["rows"]):
        ax_rows.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=7)
    _save(fig, "eda_data_quality.png")


def plot_weekday_weekend(spark) -> None:
    """Does the weekend have a commuter peak? (It should not.)"""
    counts = spark.read.parquet(str(resolve("processed") / "counts.parquet"))
    hourly = counts.groupBy("is_weekend", "hour").agg(
        F.mean(TARGET).alias("mean_flow"), F.count("*").alias("n")
    )
    from pyspark.sql import Window
    window = Window.partitionBy("is_weekend")
    data = (
        hourly.withColumn("day_mean", F.mean("mean_flow").over(window))
        .withColumn("peak_index", F.col("mean_flow") / F.col("day_mean"))
        .orderBy("is_weekend", "hour").toPandas()
    )

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    labels = {0: "Weekday", 1: "Weekend"}
    for flag, group in data.groupby("is_weekend"):
        colour = BLUE if flag == 0 else RED
        ax_raw.plot(group["hour"], group["mean_flow"], marker="o", color=colour,
                    label=labels[flag])
        ax_norm.plot(group["hour"], group["peak_index"], marker="o", color=colour,
                     label=labels[flag])
    ax_raw.set_xlabel("Hour"); ax_raw.set_ylabel("Mean flow (veh/h)")
    ax_raw.set_title("Weekday vs Weekend - Raw Flow")
    ax_raw.legend(); ax_raw.grid(alpha=0.25)

    ax_norm.axhline(1.0, color="#444", linewidth=0.8)
    ax_norm.set_xlabel("Hour"); ax_norm.set_ylabel("Flow / that day-type's mean hour")
    ax_norm.set_title("Normalised Shape\nthe commuter peak is a weekday phenomenon")
    ax_norm.legend(); ax_norm.grid(alpha=0.25)
    _save(fig, "eda_weekday_weekend.png")


def plot_geography(spark) -> None:
    """Where the count points are, and where the sharp peaks are.

    A scatter, not a choropleth: the data is points (count sites), not areas,
    and drawing it as areas would imply a spatial coverage that ~43k discrete
    sites do not have.
    """
    processed = resolve("processed")
    points = spark.read.parquet(str(processed / "dim_count_point.parquet"))
    profile = spark.read.parquet(str(processed / "hourly_profile.parquet"))

    peak = (
        profile.groupBy("count_point_id")
        .agg(F.max("peak_index").alias("max_peak_index"),
             F.sum("observations").alias("obs"))
        .filter(F.col("obs") >= 24)
    )
    data = (
        points.join(peak, "count_point_id")
        .select("latitude", "longitude", "max_peak_index", "is_major_road")
        .dropna().toPandas()
    )

    fig, (ax_all, ax_peak) = plt.subplots(1, 2, figsize=(12, 7))
    ax_all.scatter(data["longitude"], data["latitude"], s=0.4, alpha=0.25, c=BLUE)
    ax_all.set_title(f"DfT Count Point Coverage\n{len(data):,} sites with >=24 observations",
                     fontsize=10)

    # Clip the colour scale at the 99th percentile: a handful of links reach 4.2
    # and would otherwise compress every other point into one colour.
    vmax = float(data["max_peak_index"].quantile(0.99))
    sc = ax_peak.scatter(data["longitude"], data["latitude"], s=0.8,
                         c=data["max_peak_index"].clip(upper=vmax),
                         cmap="YlOrRd", alpha=0.7)
    ax_peak.set_title(f"Sharpness of the Daily Peak\n"
                      f"(colour clipped at the 99th pct = {vmax:.2f})", fontsize=10)
    fig.colorbar(sc, ax=ax_peak, label="Max peak index", fraction=0.04)

    for axis in (ax_all, ax_peak):
        axis.set_xlabel("Longitude"); axis.set_ylabel("Latitude")
        axis.set_aspect(1.6)  # rough mercator correction at GB latitudes
        axis.grid(alpha=0.2)
    _save(fig, "eda_geography.png")


def plot_peakiest_links(spark) -> None:
    """The planner's actionable list: roads carrying the most above their own average."""
    processed = resolve("processed")
    points = spark.read.parquet(str(processed / "dim_count_point.parquet"))
    profile = spark.read.parquet(str(processed / "hourly_profile.parquet"))

    data = (
        profile.filter(F.col("observations") >= 10)
        .join(F.broadcast(points.select("count_point_id", "road_category",
                                        "local_authority_name")), "count_point_id")
        .orderBy(F.desc("peak_index")).limit(15)
        .select("local_authority_name", "road_category", "hour", "peak_index", "mean_flow")
        .toPandas().iloc[::-1]
    )
    labels = [f"{r.local_authority_name[:22]} ({r.road_category}) {int(r.hour)}:00"
              for r in data.itertuples()]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    bars = ax.barh(labels, data["peak_index"], color=BLUE)
    ax.axvline(1.0, color="#444", linewidth=1, label="Link's own average hour")
    ax.set_xlabel("Peak index (peak-hour flow / that link's mean hour)")
    ax.set_title("Britain's Sharpest-Peaking Road Links\n"
                 "every one is a minor road - raw flow would never surface them",
                 fontsize=10)
    for bar, flow in zip(bars, data["mean_flow"]):
        ax.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height() / 2,
                f"{flow:.0f} veh/h", va="center", fontsize=7)
    ax.legend(fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    _save(fig, "eda_peakiest_links.png")


def plot_region_comparison(spark) -> None:
    """Regional flow level vs peak sharpness."""
    processed = resolve("processed")
    counts = spark.read.parquet(str(processed / "counts.parquet"))
    hourly = counts.groupBy("region_name", "hour").agg(F.mean(TARGET).alias("mean_flow"))
    from pyspark.sql import Window
    window = Window.partitionBy("region_name")
    data = (
        hourly.withColumn("region_mean", F.mean("mean_flow").over(window))
        .withColumn("peak_index", F.col("mean_flow") / F.col("region_mean"))
        .groupBy("region_name")
        .agg(F.round(F.first("region_mean"), 1).alias("mean_flow"),
             F.round(F.max("peak_index"), 3).alias("peak_sharpness"))
        .orderBy(F.desc("mean_flow")).toPandas().dropna()
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    scatter = ax.scatter(data["mean_flow"], data["peak_sharpness"],
                         s=90, c=BLUE, alpha=0.8)
    for row in data.itertuples():
        ax.annotate(row.region_name[:20], (row.mean_flow, row.peak_sharpness),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("Mean hourly flow (veh/h)")
    ax.set_ylabel("Peak sharpness (max hourly index)")
    ax.set_title("Region: Busy Is Not the Same as Peaky\n"
                 "flow level and peak sharpness are different problems for a planner",
                 fontsize=10)
    ax.grid(alpha=0.25)
    _save(fig, "eda_region_comparison.png")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stdout
    )
    spark = get_spark("eda_figures")
    try:
        plot_data_scale(spark)
        plot_hourly_profile(spark)
        plot_covid_break(spark)
        plot_modal_shift(spark)
        plot_profile_by_road_class(spark)
        plot_flow_distribution(spark)
        plot_correlation_heatmap(spark)
        plot_data_quality(spark)
        plot_weekday_weekend(spark)
        plot_geography(spark)
        plot_peakiest_links(spark)
        plot_region_comparison(spark)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
