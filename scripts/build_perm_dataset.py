#!/usr/bin/env python3
"""Build a unified Parquet dataset from raw PERM disclosure files."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import polars as pl

STATE_ABBREV_MAP = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
    "PUERTO RICO": "PR",
    "GUAM": "GU",
    "U.S. VIRGIN ISLANDS": "VI",
    "NORTHERN MARIANA ISLANDS": "MP",
    "AMERICAN SAMOA": "AS",
}

TARGET_COLUMNS = [
    "source_file",
    "employer_name",
    "case_status",
    "decision_date",
    "received_date",
    "job_title",
    "worksite_city",
    "worksite_state",
    "wage_offer_from",
    "wage_offer_to",
    "wage_unit",
    "pw_wage",
    "pw_unit",
    "naics_code",
    "soc_code",
]

DEFAULT_COLUMN_ALIASES = {
    "employer_name": ["employer_name", "employer name", "employername", "employer", "emp_name"],
    "case_status": ["case_status", "status", "final_case_status"],
    "decision_date": ["decision_date", "final_decision_date", "decisiondate"],
    "received_date": ["received_date", "case_received_date", "receiveddate"],
    "job_title": ["job_title", "pw_job_title", "jobtitle"],
    "worksite_city": ["worksite_city", "job_info_work_city", "work_city", "city"],
    "worksite_state": ["worksite_state", "job_info_work_state", "work_state", "state"],
    "wage_offer_from": ["wage_offer_from", "wage_from", "wage_rate_of_pay_from", "offered_wage_from"],
    "wage_offer_to": ["wage_offer_to", "wage_to", "wage_rate_of_pay_to", "offered_wage_to"],
    "wage_unit": ["wage_unit", "wage_offer_unit", "wage_rate_unit", "pw_unit_of_pay", "wage_offer_unit_of_pay"],
    "pw_wage": ["pw_wage", "pw_wage_level", "pw_amount", "prevailing_wage"],
    "pw_unit": ["pw_unit", "pw_unit_of_pay", "prevailing_wage_unit"],
    "naics_code": ["naics_code", "naics", "employer_naics_code"],
    "soc_code": ["soc_code", "pw_soc_code", "soc", "soc_occupation_code"],
}


def normalize_colname(col: str) -> str:
    return "_".join(col.strip().lower().replace("/", " ").replace("-", " ").split())


def normalize_text_expr(column: str, *, uppercase: bool = True) -> pl.Expr:
    expr = (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    if uppercase:
        expr = expr.str.to_uppercase()

    return (
        pl.when(
            expr.is_null()
            | expr.is_in(["", "N/A", "NA", "NULL", "NONE", "NOT AVAILABLE", "UNKNOWN"])
        )
        .then(pl.lit(None))
        .otherwise(expr)
    )


def normalize_state_expr(column: str) -> pl.Expr:
    cleaned = normalize_text_expr(column)
    return cleaned.replace_strict(STATE_ABBREV_MAP, default=cleaned)


def normalize_city_expr() -> pl.Expr:
    city = normalize_text_expr("worksite_city")
    state = normalize_state_expr("worksite_state")
    return (
        pl.when((city == "NY") & (state == "NY"))
        .then(pl.lit("NEW YORK"))
        .otherwise(city)
    )


def load_column_aliases(mapping_path: Path | None) -> dict[str, list[str]]:
    """Load aliases from a simple YAML mapping file.

    Expected shape:
      columns:
        target_field:
          - alias_one
          - alias_two
    """
    if mapping_path is None or not mapping_path.exists():
        return DEFAULT_COLUMN_ALIASES

    lines = mapping_path.read_text(encoding="utf-8").splitlines()
    in_columns = False
    current_key: str | None = None
    loaded: dict[str, list[str]] = {}

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped == "columns:":
            in_columns = True
            continue
        if not in_columns:
            continue

        if line.startswith("  ") and stripped.endswith(":") and not stripped.startswith("-"):
            current_key = stripped[:-1]
            loaded[current_key] = []
            continue

        if line.startswith("    - ") and current_key is not None:
            alias = line.replace("    - ", "", 1).strip()
            if alias:
                loaded[current_key].append(normalize_colname(alias))

    if not loaded:
        return DEFAULT_COLUMN_ALIASES

    normalized_loaded: dict[str, list[str]] = {}
    for target in TARGET_COLUMNS:
        if target == "source_file":
            continue
        aliases = loaded.get(target) or DEFAULT_COLUMN_ALIASES.get(target, [])
        normalized_loaded[target] = [normalize_colname(a) for a in aliases]
    return normalized_loaded


def read_one_table_from_bytes(content: bytes, suffix: str) -> list[pl.DataFrame]:
    suffix = suffix.lower()
    dfs: list[pl.DataFrame] = []

    if suffix == ".csv":
        dfs.append(pl.read_csv(io.BytesIO(content), infer_schema_length=2000, ignore_errors=True))
    elif suffix == ".xlsx":
        dfs.append(pl.read_excel(io.BytesIO(content)))
    elif suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                lower = name.lower()
                if lower.endswith(".csv"):
                    dfs.append(
                        pl.read_csv(io.BytesIO(zf.read(name)), infer_schema_length=2000, ignore_errors=True)
                    )
                elif lower.endswith(".xlsx"):
                    dfs.append(pl.read_excel(io.BytesIO(zf.read(name))))
    return dfs


def map_columns(df: pl.DataFrame, column_aliases: dict[str, list[str]]) -> pl.DataFrame:
    original = df.columns
    rename_map = {c: normalize_colname(c) for c in original}
    df = df.rename(rename_map)

    selected = {}
    for target, aliases in column_aliases.items():
        found = next((a for a in aliases if a in df.columns), None)
        if found:
            selected[target] = pl.col(found)
        else:
            selected[target] = pl.lit(None)

    return df.select([selected[c].alias(c) for c in column_aliases.keys()])


def cast_and_clean(df: pl.DataFrame) -> pl.DataFrame:
    decision_date_expr = pl.coalesce(
        pl.col("decision_date").cast(pl.Utf8, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("decision_date").cast(pl.Utf8, strict=False).str.strptime(pl.Date, "%m/%d/%Y", strict=False),
        pl.col("decision_date").cast(pl.Utf8, strict=False).str.strptime(pl.Date, "%m-%d-%Y", strict=False),
    )
    received_date_expr = pl.coalesce(
        pl.col("received_date").cast(pl.Utf8, strict=False).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("received_date").cast(pl.Utf8, strict=False).str.strptime(pl.Date, "%m/%d/%Y", strict=False),
        pl.col("received_date").cast(pl.Utf8, strict=False).str.strptime(pl.Date, "%m-%d-%Y", strict=False),
    )

    return df.with_columns(
        [
            normalize_text_expr("employer_name").alias("employer_name"),
            normalize_text_expr("case_status").alias("case_status"),
            normalize_text_expr("job_title").alias("job_title"),
            normalize_city_expr().alias("worksite_city"),
            normalize_state_expr("worksite_state").alias("worksite_state"),
            normalize_text_expr("wage_unit").alias("wage_unit"),
            normalize_text_expr("pw_unit").alias("pw_unit"),
            normalize_text_expr("naics_code", uppercase=False).alias("naics_code"),
            normalize_text_expr("soc_code", uppercase=False).alias("soc_code"),
            pl.col("wage_offer_from").cast(pl.Float64, strict=False),
            pl.col("wage_offer_to").cast(pl.Float64, strict=False),
            pl.col("pw_wage").cast(pl.Float64, strict=False),
            decision_date_expr.alias("decision_date"),
            received_date_expr.alias("received_date"),
        ]
    )


def build_dataset(raw_dir: Path, column_aliases: dict[str, list[str]]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []

    for path in sorted(raw_dir.glob("*")):
        if path.suffix.lower() not in {".csv", ".xlsx", ".zip"}:
            continue

        content = path.read_bytes()
        tables = read_one_table_from_bytes(content, path.suffix)

        for tbl in tables:
            if tbl.height == 0:
                continue
            mapped = map_columns(tbl, column_aliases)
            cleaned = cast_and_clean(mapped).with_columns(pl.lit(path.name).alias("source_file"))
            frames.append(cleaned.select(TARGET_COLUMNS))

    if not frames:
        raise RuntimeError("No readable PERM tables were found in raw directory.")

    return pl.concat(frames, how="vertical_relaxed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize PERM files into a single Parquet dataset")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/perm_filings.parquet")
    parser.add_argument("--mapping", default="config/column_mapping.yaml", help="YAML file with column alias mapping")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    aliases = load_column_aliases(Path(args.mapping))
    df = build_dataset(raw_dir, aliases)
    df.write_parquet(out_path)

    print(f"Rows: {df.height}")
    print(f"Columns: {df.width}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
