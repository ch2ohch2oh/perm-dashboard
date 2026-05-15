# US PERM Filing Analysis App

Modern data analysis web app for US PERM filing trends, with a reproducible pipeline:

1. Download PERM disclosure data from DOL/OFLC
2. Normalize schema across years/releases
3. Convert to Parquet for fast analytics
4. Launch an interactive dashboard for trend and employer insights

## Tech choices

- Python 3.11+
- `polars` for fast dataframe transforms
- `pyarrow` for Parquet interoperability
- `requests` + `beautifulsoup4` for link discovery/download
- `streamlit` + `plotly` for a modern, interactive dashboard

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/download_perm_data.py --max-files 8
python scripts/build_perm_dataset.py
streamlit run app/dashboard.py
```

## Data outputs

- Raw downloads: `data/raw/`
- Unified Parquet: `data/processed/perm_filings.parquet`

## Notes

- The downloader discovers likely PERM disclosure files from the DOL site and filters to CSV/XLSX/ZIP resources.
- The normalizer performs robust column mapping to handle header drift across releases.
