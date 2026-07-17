"""Interactive Streamlit dashboard for the completed traffic forecasting pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.db.queries import (
    annual_trend,
    connect,
    link_profile,
    peak_hour_by_road_class,
    peakiest_links,
    region_summary,
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    connection = connect()
    try:
        road_classes = peak_hour_by_road_class(connection)
        peaky = peakiest_links(connection, limit=20)
        regions = region_summary(connection)
        trend = annual_trend(connection)
        ids = pd.read_sql_query(
            "SELECT DISTINCT count_point_id FROM fact_hourly_profile ORDER BY count_point_id",
            connection,
        )["count_point_id"].astype(str).tolist()
    finally:
        connection.close()
    return road_classes, peaky, regions, trend, ids


@st.cache_data(show_spinner=False)
def load_link_profile(count_point_id: str) -> pd.DataFrame:
    connection = connect()
    try:
        return link_profile(connection, count_point_id)
    finally:
        connection.close()


@st.cache_data(show_spinner=False)
def load_results() -> dict:
    with (OUTPUTS / "model_results.json").open(encoding="utf-8") as result_file:
        return json.load(result_file)


st.set_page_config(page_title="When Does the Road Fill?", page_icon="🚌", layout="wide")
st.title("When Does the Road Fill?")
st.caption("Interactive results dashboard — GB road traffic peak-hour demand forecasting")

try:
    road_classes, peaky, regions, trend, count_point_ids = load_dashboard_data()
    model_results = load_results()
except FileNotFoundError as error:
    st.error(f"Required pipeline output is missing: {error}. Run `python run_pipeline.py` first.")
    st.stop()

st.sidebar.header("Navigation")
page = st.sidebar.radio("View", ["Overview", "Road profiles", "Model results", "Figures"])
min_observations = st.sidebar.slider("Minimum observations", 1, 20, 5)

if page == "Overview":
    st.subheader("Network overview")
    latest = trend.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Count points", f"{len(count_point_ids):,}")
    col2.metric("Latest mean annual flow", f"{latest['mean_aadf']:,.0f}")
    col3.metric("Latest year", int(latest["year"]))

    st.subheader("Annual traffic trend")
    st.line_chart(trend.set_index("year")["mean_aadf"], use_container_width=True)

    st.subheader("Peak-hour shape by road class")
    filtered = road_classes[road_classes["links"] >= min_observations]
    pivot = filtered.pivot(index="hour", columns="road_category", values="mean_peak_index")
    st.line_chart(pivot, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Sharpest peak-hour links")
        st.dataframe(peaky, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Regional summary")
        st.dataframe(regions, use_container_width=True, hide_index=True)

elif page == "Road profiles":
    st.subheader("Individual road profile")
    selected = st.selectbox("Count point ID", count_point_ids)
    profile = load_link_profile(selected)
    if profile.empty:
        st.info("No profile is available for this count point.")
    else:
        road_name = profile.iloc[0]["road_name"] or "Unnamed road"
        st.caption(f"{road_name} · {profile.iloc[0]['road_category']}")
        st.line_chart(profile.set_index("hour")[["mean_flow", "peak_index"]], use_container_width=True)
        st.dataframe(profile, use_container_width=True, hide_index=True)

elif page == "Model results":
    st.subheader("Time-aware model evaluation")
    rows: list[dict] = []
    for split in model_results["splits"]:
        for model in split["results"]:
            rows.append({key: model.get(key) for key in ("model", "rmse", "mae", "r2", "train_seconds")}
                        | {"split": split["split"]})
    metrics = pd.DataFrame(rows)
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.bar_chart(metrics.pivot(index="model", columns="split", values="r2"), use_container_width=True)
    st.info("The link-history baseline remains very strong; the dashboard reports model results alongside it to avoid overstating forecasting skill.")

else:
    st.subheader("Generated figures")
    images = sorted(OUTPUTS.glob("*.png"))
    if not images:
        st.info("No figures found. Run the pipeline to generate them.")
    for start in range(0, len(images), 2):
        columns = st.columns(2)
        for column, image in zip(columns, images[start:start + 2]):
            with column:
                st.image(str(image), caption=image.stem.replace("_", " ").title(), use_container_width=True)
