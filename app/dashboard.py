#!/usr/bin/env python3
from __future__ import annotations

from calendar import monthrange
from datetime import date
from pathlib import Path
from typing import Any

import plotly.express as px
import polars as pl
import streamlit as st

# --- Configuration & Constants ---
ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "perm_filings.parquet"
ACCENT_COLOR = "#60a5fa"

BASE_CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="Segoe UI, sans-serif", color="#31333F", size=13),
    margin=dict(l=18, r=18, t=48, b=18),
    xaxis=dict(showgrid=False, zeroline=False, linecolor="#dfe3eb"),
    yaxis=dict(showgrid=False, zeroline=False, linecolor="#dfe3eb"),
    showlegend=False,
)

# --- Data Helpers ---

@st.cache_data(show_spinner=False)
def load_dataset(path: Path, mtime: float) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    
    # Use lazy loading to reduce memory pressure
    lf = pl.scan_parquet(path)
    
    return (
        lf.with_columns([
            pl.col("job_title").fill_null("Unknown"),
            pl.col("employer_name").fill_null("Unknown"),
            pl.col("worksite_city").fill_null("Unknown"),
            pl.col("worksite_state").fill_null("Unknown").cast(pl.Categorical),
            pl.col("case_status").fill_null("UNKNOWN").cast(pl.Categorical),
            pl.col("received_date").alias("event_date"),
        ])
        .filter(pl.col("event_date").is_not_null())
        .with_columns([
            pl.col("event_date").dt.year().alias("event_year"),
            pl.concat_str(["worksite_city", pl.lit(", "), "worksite_state"]).alias("location"),
        ])
        .collect()
    )

def add_time_bucket(frame: pl.DataFrame, grain: str, source_col: str = "event_date") -> pl.DataFrame:
    truncate_map = {"Month": "1mo", "Quarter": "1q", "Year": "1y"}
    return frame.with_columns(pl.col(source_col).dt.truncate(truncate_map[grain]).alias("time_bucket"))

def complete_time_series(frame: pl.DataFrame, grain: str, start_date: date, end_date: date) -> pl.DataFrame:
    interval_map = {"Month": "1mo", "Quarter": "1q", "Year": "1y"}
    
    # Ensure start/end align with buckets
    dummy = pl.DataFrame({"event_date": [start_date, end_date]})
    buckets = add_time_bucket(dummy, grain).get_column("time_bucket")
    bucket_start, bucket_end = buckets[0], buckets[1]

    full_range = pl.DataFrame({
        "time_bucket": pl.date_range(start=bucket_start, end=bucket_end, interval=interval_map[grain], eager=True)
    })

    counts = add_time_bucket(frame, grain).group_by("time_bucket").agg(pl.len().alias("filings"))
    return full_range.join(counts, on="time_bucket", how="left").with_columns(pl.col("filings").fill_null(0)).sort("time_bucket")

def top_n_table(frame: pl.DataFrame, column: str, top_n: int = 10) -> pl.DataFrame:
    counts = (
        frame.group_by(column)
        .agg(pl.len().alias("filings"))
        .sort(["filings", column], descending=[True, False])
        .head(top_n)
    )
    total = frame.height
    return counts.with_columns((pl.col("filings") * 100 / total).alias("share")) if total > 0 else counts

def infer_completeness_cutoff(frame: pl.DataFrame) -> date | None:
    monthly = (
        frame.with_columns(pl.col("event_date").dt.truncate("1mo").alias("month"))
        .group_by("month").agg(pl.len().alias("filings")).sort("month")
    ).to_dicts()

    for i in range(12, len(monthly) - 2):
        trailing = sorted(r["filings"] for r in monthly[i-12:i])
        median = trailing[len(trailing)//2]
        if median > 1000 and all(monthly[i+j]["filings"] / median < 0.6 for j in range(3)):
            return monthly[i]["month"]
    return None


@st.cache_data(show_spinner=False)
def get_top_options(frame: pl.DataFrame, column: str, n: int = 100) -> list[str]:
    return (
        frame.group_by(column)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(n)
        .get_column(column)
        .to_list()
    )

# --- UI Components ---

def centered_row():
    """Returns the middle column of a 3-column layout for centering content."""
    _, col, _ = st.columns((0.1, 3.2, 0.1))
    return col

def style_figure(fig, height: int):
    fig.update_layout(height=height, **BASE_CHART_LAYOUT)

def render_bar_chart(df: pl.DataFrame, x: str, y: str, title: str, height: int = 420):
    st.subheader(title)
    fig = px.bar(
        df.to_pandas(), x=x, y=y, orientation="h",
        custom_data=["share"], color_discrete_sequence=[ACCENT_COLOR]
    )
    fig.update_traces(
        texttemplate="%{x}", textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Filings: %{x}<br>Share: %{customdata[0]:.1f}%<extra></extra>"
    )
    val_max = df.get_column(x).max() if df.height > 0 else 1
    fig.update_xaxes(title="Filings", range=[0, val_max * 1.3])
    fig.update_yaxes(title=None, categoryorder="total ascending")
    style_figure(fig, height=height)
    st.plotly_chart(fig, theme="streamlit", width="stretch")

# --- Main Application ---

def main():
    st.set_page_config(page_title="PERM Dashboard", layout="wide")

    mtime = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0
    df = load_dataset(DATA_PATH, mtime)
    if df.height == 0:
        st.error("Dataset not found or empty. Run `scripts/build_perm_dataset.py` first.")
        st.stop()

    # Initial Stats & Options
    completeness_cutoff = infer_completeness_cutoff(df)
    stats = df.select([
        pl.min("event_date").alias("min"),
        pl.col("event_date").quantile(0.05).alias("start"),
        pl.max("event_date").alias("max")
    ]).row(0)
    
    default_start = date(stats[1].year, stats[1].month, 1)
    
    max_selectable = stats[2]
    if completeness_cutoff:
        limit_date = pl.Series([completeness_cutoff]).dt.offset_by("3mo")[0]
        max_selectable = min(stats[2], limit_date)
    
    job_options = get_top_options(df, "job_title")
    employer_options = get_top_options(df, "employer_name")

    # --- Header ---
    with centered_row():
        st.title("PERM Filing Dashboard")
        st.markdown(
            "Insights into **Program Electronic Review Management (PERM)** trends. "
            "PERM is the process used by employers to obtain a permanent labor certification "
            "for foreign nationals seeking to work in the United States."
        )

    # --- Filters ---
    with centered_row():
        c1, c2 = st.columns([1.35, 1.0], gap="medium")
        date_range = c1.date_input("Choose a date range", value=(default_start, max_selectable), min_value=stats[0], max_value=max_selectable)
        grain = c2.radio("Granularity", options=["Month", "Quarter", "Year"], horizontal=True)

        f1, f2 = st.columns(2, gap="medium")
        sel_jobs = f1.multiselect(
            "Job titles", options=job_options, placeholder="Top 100 jobs",
            help="Limited to the 100 most frequent job titles for faster interaction."
        )
        sel_employers = f2.multiselect(
            "Employers", options=employer_options, placeholder="Top 100 employers",
            help="Limited to the 100 most frequent employers for faster interaction."
        )

    # Apply Filters
    start_date, end_date = date_range if isinstance(date_range, (list, tuple)) and len(date_range) == 2 else (stats[0], stats[2])
    filtered = df.filter(
        (pl.col("event_date").is_between(start_date, end_date)) & 
        (pl.col("case_status") != "WITHDRAWN")
    )
    if sel_jobs: filtered = filtered.filter(pl.col("job_title").is_in(sel_jobs))
    if sel_employers: filtered = filtered.filter(pl.col("employer_name").is_in(sel_employers))

    if filtered.height == 0:
        st.warning("No rows match the current filters.")
        st.stop()

    # --- Trend Chart ---
    with centered_row():
        st.subheader("PERM filings over time")
        trend = complete_time_series(filtered, grain, start_date, end_date).to_pandas()
        fig = px.line(trend, x="time_bucket", y="filings", markers=True, color_discrete_sequence=[ACCENT_COLOR])
        fig.update_xaxes(title="Received date")
        fig.update_yaxes(title="Filings")
        
        if completeness_cutoff:
            fig.add_vline(x=completeness_cutoff, line_dash="dash", line_color="#6b7280")
            fig.add_annotation(x=completeness_cutoff, y=1, yref="paper", text="Coverage cutoff", showarrow=False, xanchor="left", bgcolor="rgba(255,255,255,0.8)")
        
        style_figure(fig, height=420)
        st.plotly_chart(fig, theme="streamlit", width="stretch")
        
        # Summary Note
        scope_parts = [f"from {start_date:%b %d, %Y} to {end_date:%b %d, %Y}"]
        scope_parts.append(f"{len(sel_jobs)} selected job titles" if sel_jobs else "all job titles")
        scope_parts.append(f"{len(sel_employers)} selected employers" if sel_employers else "all employers")
        
        notes = ["Withdrawn cases are excluded from all charts on this page"]
        if completeness_cutoff:
            notes.append(f"coverage considered complete through {completeness_cutoff:%b %d, %Y} based on the full dataset trend")
        
        st.markdown(f"Showing filings {' across '.join(scope_parts)}. {' | '.join(notes)}.")

    st.write("")

    # --- Distributions ---
    _, d1, _, d2, _ = st.columns((0.1, 1, 0.1, 1, 0.1))
    with d1:
        render_bar_chart(top_n_table(filtered, "job_title"), "filings", "job_title", "Top job titles")
        st.markdown("This chart shows which roles dominate the current filing window and how quickly the distribution narrows to a handful of titles.")
    with d2:
        render_bar_chart(top_n_table(filtered, "location"), "filings", "location", "Top locations")
        st.markdown("Location concentration gives a quick read on where PERM demand is clustered in the selected slice without requiring a map.")

    st.write("")

    # --- Employers ---
    with centered_row():
        top_emp = top_n_table(filtered, "employer_name", top_n=15)
        render_bar_chart(top_emp, "filings", "employer_name", "Top employers", height=max(420, 42 * len(top_emp) + 40))
        st.markdown("Employer rankings make it easier to see which companies dominate the current view and whether filing activity is concentrated in a small set of firms.")

    # --- Footer ---
    st.markdown("---")
    with centered_row():
        st.caption("Data: [Office of Foreign Labor Certification](https://www.dol.gov/agencies/eta/foreign-labor/performance).")

if __name__ == "__main__":
    main()
