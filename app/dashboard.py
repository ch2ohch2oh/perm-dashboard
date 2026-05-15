#!/usr/bin/env python3
from __future__ import annotations

from calendar import monthrange
from datetime import date
from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

DATA_PATH = Path("data/processed/perm_filings.parquet")
ACCENT_COLOR = "#60a5fa"

BASE_CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="Segoe UI, sans-serif", color="#31333F", size=13),
    margin=dict(l=18, r=18, t=48, b=18),
    xaxis=dict(showgrid=False, zeroline=False, linecolor="#dfe3eb"),
    yaxis=dict(showgrid=False, zeroline=False, linecolor="#dfe3eb"),
)


def add_time_bucket(frame: pl.DataFrame, grain: str, source_col: str = "event_date") -> pl.DataFrame:
    truncate_map = {"Month": "1mo", "Quarter": "1q", "Year": "1y"}
    return frame.with_columns(pl.col(source_col).dt.truncate(truncate_map[grain]).alias("time_bucket"))


def format_pretty_date(value) -> str:
    return value.strftime("%b %d, %Y")


def month_start(value):
    return date(value.year, value.month, 1)


def month_end(value):
    return date(value.year, value.month, monthrange(value.year, value.month)[1])


def month_label(value: date) -> str:
    return value.strftime("%Y-%m")


def style_figure(fig, *, height: int) -> None:
    fig.update_layout(height=height, **BASE_CHART_LAYOUT)


def top_n_options(frame: pl.DataFrame, column: str, n: int = 100) -> list[str]:
    return (
        frame.group_by(column)
        .agg(pl.len().alias("count"))
        .sort(["count", column], descending=[True, False])
        .head(n)
        .get_column(column)
        .to_list()
    )


def complete_time_series(frame: pl.DataFrame, grain: str, start_date, end_date) -> pl.DataFrame:
    interval_map = {"Month": "1mo", "Quarter": "1q", "Year": "1y"}
    bucket_start = start_date if grain == "Month" else add_time_bucket(
        pl.DataFrame({"event_date": [start_date]}),
        grain,
    ).get_column("time_bucket")[0]
    bucket_end = add_time_bucket(
        pl.DataFrame({"event_date": [end_date]}),
        grain,
    ).get_column("time_bucket")[0]

    full_range = pl.DataFrame(
        {
            "time_bucket": pl.date_range(
                start=bucket_start,
                end=bucket_end,
                interval=interval_map[grain],
                eager=True,
            )
        }
    )

    counts = (
        add_time_bucket(frame, grain)
        .group_by("time_bucket")
        .agg(pl.len().alias("filings"))
    )

    return (
        full_range.join(counts, on="time_bucket", how="left")
        .with_columns(pl.col("filings").fill_null(0))
        .sort("time_bucket")
    )


def top_n_table(frame: pl.DataFrame, column: str, top_n: int = 10) -> pl.DataFrame:
    counts = (
        frame.group_by(column)
        .agg(pl.len().alias("filings"))
        .sort(["filings", column], descending=[True, False])
        .head(top_n)
    )

    total_filings = frame.height
    return counts.with_columns((pl.col("filings") * 100 / total_filings).alias("share"))


def infer_completeness_cutoff(frame: pl.DataFrame):
    monthly = (
        frame.filter(pl.col("event_date").is_not_null())
        .with_columns(pl.col("event_date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.len().alias("filings"))
        .sort("month")
    )

    rows = monthly.to_dicts()
    for i in range(12, len(rows) - 2):
        trailing = sorted(r["filings"] for r in rows[i - 12 : i])
        median = trailing[len(trailing) // 2]
        if median < 1000:
            continue
        ratios = [rows[i + j]["filings"] / median for j in range(3)]
        if all(r < 0.6 for r in ratios):
            return rows[i]["month"]
    return None


@st.cache_data(show_spinner=False)
def load_dataset(path: str, data_version: float) -> pl.DataFrame:
    df = pl.read_parquet(path)

    return (
        df.with_columns(
            [
                pl.col("job_title").fill_null("Unknown").alias("job_title"),
                pl.col("employer_name").fill_null("Unknown").alias("employer_name"),
                pl.col("worksite_city").fill_null("Unknown").alias("worksite_city"),
                pl.col("worksite_state").fill_null("Unknown").alias("worksite_state"),
                pl.col("case_status").fill_null("UNKNOWN").alias("case_status"),
                pl.col("received_date").alias("event_date"),
            ]
        )
        .filter(pl.col("event_date").is_not_null())
        .with_columns(
            [
                pl.col("event_date").dt.year().alias("event_year"),
                pl.concat_str(["worksite_city", pl.lit(", "), "worksite_state"]).alias("location"),
            ]
        )
    )


st.set_page_config(page_title="PERM Dashboard", page_icon="📈", layout="wide")

if not DATA_PATH.exists():
    st.error("Dataset not found. Run `scripts/build_perm_dataset.py` first.")
    st.stop()

df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
if df.height == 0:
    st.warning("The dataset is empty after normalization.")
    st.stop()

completeness_cutoff = infer_completeness_cutoff(df)

min_event, default_start, max_event = (
    df.select(
        [
            pl.min("event_date").alias("min_event"),
            pl.col("event_date").quantile(0.05).alias("default_start"),
            pl.max("event_date").alias("max_event"),
        ]
    ).row(0)
)
default_start = month_start(default_start)
job_options = top_n_options(df, "job_title", n=100)
employer_options = top_n_options(df, "employer_name", n=100)
row0_spacer1, row0_1, row0_spacer2 = st.columns((0.1, 3.2, 0.1))
row0_1.title("PERM Filing Dashboard")


row2_spacer1, row2_1, row2_spacer2 = st.columns((0.1, 3.2, 0.1))
with row2_1:
    top_controls = st.columns([1.35, 1.0], gap="medium")
    with top_controls[0]:
        date_window = st.date_input(
            "Choose a date range (All available is the default)",
            value=(default_start, max_event),
            min_value=min_event,
            max_value=max_event,
        )
    with top_controls[1]:
        time_grain = st.radio(
            "Granularity",
            options=["Month", "Quarter", "Year"],
            index=0,
            horizontal=True,
        )

    filter_controls = st.columns(2, gap="medium")
    with filter_controls[0]:
        selected_jobs = st.multiselect(
            "Job titles",
            options=job_options,
            placeholder="Top 100 jobs",
            help="Limited to the 100 most frequent job titles for faster interaction.",
        )
    with filter_controls[1]:
        selected_employers = st.multiselect(
            "Employers",
            options=employer_options,
            placeholder="Top 100 employers",
            help="Limited to the 100 most frequent employers for faster interaction.",
        )

if isinstance(date_window, tuple) and len(date_window) == 2:
    start_date, end_date = date_window
else:
    start_date, end_date = min_event, max_event

filtered = df.filter(pl.col("event_date").is_between(start_date, end_date))
filtered = filtered.filter(pl.col("case_status") != "WITHDRAWN")
if selected_jobs:
    filtered = filtered.filter(pl.col("job_title").is_in(selected_jobs))
if selected_employers:
    filtered = filtered.filter(pl.col("employer_name").is_in(selected_employers))

if filtered.height == 0:
    st.warning("No rows match the current filters.")
    st.stop()

trend = complete_time_series(filtered, time_grain, start_date, end_date).to_pandas()

scope_parts = [f"from {format_pretty_date(start_date)} to {format_pretty_date(end_date)}"]
if selected_jobs:
    scope_parts.append(f"{len(selected_jobs)} selected job titles")
else:
    scope_parts.append("all job titles")
if selected_employers:
    scope_parts.append(f"{len(selected_employers)} selected employers")
else:
    scope_parts.append("all employers")
note_parts = ["Withdrawn cases are excluded from all charts on this page"]
if completeness_cutoff is not None:
    note_parts.append(
        f"coverage considered complete through {format_pretty_date(completeness_cutoff)} based on the full dataset trend"
    )

row3_spacer1, row3_1, row3_spacer2 = st.columns((0.1, 3.2, 0.1))

with row3_1:
    st.subheader("PERM filings over time")
    fig = px.line(trend, x="time_bucket", y="filings", markers=True, color_discrete_sequence=[ACCENT_COLOR])
    fig.update_traces(line_color=ACCENT_COLOR, marker_color=ACCENT_COLOR)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Received date")
    fig.update_yaxes(title="Filings")
    if completeness_cutoff is not None:
        fig.add_shape(
            type="line",
            x0=completeness_cutoff,
            x1=completeness_cutoff,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(width=2, dash="dash", color="#6b7280"),
        )
        fig.add_annotation(
            x=completeness_cutoff,
            y=1,
            xref="x",
            yref="paper",
            text="Coverage cutoff",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="#6b7280"),
            bgcolor="rgba(255,255,255,0.85)",
        )
    style_figure(fig, height=420)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    st.markdown(
        f"Showing filings {' across '.join(scope_parts)}. {' | '.join(note_parts)}."
    )

st.write("")
row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))

with row4_1:
    st.subheader("Top job titles")
    top_jobs = top_n_table(filtered, column="job_title", top_n=10)
    fig_jobs = px.bar(
        top_jobs.to_pandas(),
        x="filings",
        y="job_title",
        orientation="h",
        custom_data=["share"],
        color_discrete_sequence=[ACCENT_COLOR],
    )
    fig_jobs.update_traces(
        texttemplate="%{x}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Filings: %{x}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
    )
    fig_jobs.update_layout(showlegend=False, margin=dict(l=18, r=72, t=48, b=18))
    job_max = top_jobs.get_column("filings").max()
    fig_jobs.update_xaxes(title="Filings", range=[0, job_max * 1.3 if job_max else 1])
    fig_jobs.update_yaxes(title=None, categoryorder="total ascending")
    style_figure(fig_jobs, height=420)
    st.plotly_chart(fig_jobs, theme="streamlit", use_container_width=True)
    st.markdown(
        "This chart shows which roles dominate the current filing window and how quickly the distribution narrows to a handful of titles."
    )

with row4_2:
    st.subheader("Top locations")
    top_locations = top_n_table(filtered, column="location", top_n=10)
    fig_locations = px.bar(
        top_locations.to_pandas(),
        x="filings",
        y="location",
        orientation="h",
        custom_data=["share"],
        color_discrete_sequence=[ACCENT_COLOR],
    )
    fig_locations.update_traces(
        texttemplate="%{x}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Filings: %{x}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
    )
    fig_locations.update_layout(showlegend=False, margin=dict(l=18, r=72, t=48, b=18))
    location_max = top_locations.get_column("filings").max()
    fig_locations.update_xaxes(title="Filings", range=[0, location_max * 1.3 if location_max else 1])
    fig_locations.update_yaxes(title=None, categoryorder="total ascending")
    style_figure(fig_locations, height=420)
    st.plotly_chart(fig_locations, theme="streamlit", use_container_width=True)
    st.markdown(
        "Location concentration gives a quick read on where PERM demand is clustered in the selected slice without requiring a map."
    )

st.write("")
row5_space1, row5_1, row5_space2 = st.columns((0.1, 3.2, 0.1))

with row5_1:
    st.subheader("Top employers")
    top_employers = top_n_table(filtered, "employer_name", top_n=15).to_pandas()
    fig_employers = px.bar(
        top_employers,
        x="filings",
        y="employer_name",
        orientation="h",
        custom_data=["share"],
        color_discrete_sequence=[ACCENT_COLOR],
    )
    fig_employers.update_traces(
        texttemplate="%{x}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Filings: %{x}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
    )
    fig_employers.update_layout(showlegend=False, margin=dict(l=18, r=72, t=48, b=18))
    employer_max = top_employers["filings"].max() if not top_employers.empty else 1
    fig_employers.update_xaxes(title="Filings", range=[0, employer_max * 1.3 if employer_max else 1])
    fig_employers.update_yaxes(title=None, categoryorder="total ascending")
    employer_height = max(420, 42 * len(top_employers) + 40)
    style_figure(fig_employers, height=employer_height)
    st.plotly_chart(fig_employers, theme="streamlit", use_container_width=True)
    st.markdown(
        "Employer rankings make it easier to see which companies dominate the current view and whether filing activity is concentrated in a small set of firms."
    )

st.markdown("---")
footer_spacer1, footer_1, footer_spacer2 = st.columns((0.1, 3.2, 0.1))
with footer_1:
    st.caption("Data sourced from the [Office of Foreign Labor Certification](https://www.dol.gov/agencies/eta/foreign-labor/performance).")
