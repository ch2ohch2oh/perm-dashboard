#!/usr/bin/env python3
"""Generic disclosure-data downloader with optional PERM FY probing."""

from __future__ import annotations

import argparse
import re
from collections import deque
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_START_PAGES = [
    "https://www.dol.gov/agencies/eta/foreign-labor/performance",
    "https://www.dol.gov/agencies/eta/foreign-labor/disclosure-data",
]
DEFAULT_EXTENSIONS = {".csv", ".xlsx", ".zip", ".xls"}
DEFAULT_KEYWORDS = ["perm", "9089", "disclosure", "oflc"]
DEFAULT_FILE_BASE = "https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/"


def clean_filename(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name or "downloaded_file"
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def parse_extensions(text: str) -> set[str]:
    out = set()
    for e in text.split(","):
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = f".{e}"
        out.add(e)
    return out or set(DEFAULT_EXTENSIONS)


def parse_keywords(text: str) -> list[str]:
    kws = [k.strip().lower() for k in text.split(",") if k.strip()]
    return kws or list(DEFAULT_KEYWORDS)


def parse_years(text: str) -> list[int]:
    years: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        years.append(int(token))
    return sorted(set(years))


def looks_like_data_file(url: str, allowed_extensions: set[str]) -> bool:
    path = urlparse(url).path.lower()
    ext = Path(path).suffix.lower()
    return ext in allowed_extensions


def keyword_match(url: str, anchor_text: str, keywords: list[str]) -> bool:
    hay = f"{url.lower()} {anchor_text.lower()}"
    return any(k in hay for k in keywords)


def discover_links_crawl(
    session: requests.Session,
    start_pages: list[str],
    allowed_extensions: set[str],
    keywords: list[str],
    max_pages: int,
) -> list[str]:
    """Breadth-first crawl for candidate file links."""
    links: set[str] = set()
    visited: set[str] = set()
    q: deque[str] = deque(start_pages)

    while q and len(visited) < max_pages:
        page = q.popleft()
        if page in visited:
            continue
        visited.add(page)

        try:
            res = session.get(page, timeout=45)
            res.raise_for_status()
        except Exception:
            continue

        soup = BeautifulSoup(res.text, "lxml")

        # Parse href-based links.
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            text = (a.get_text() or "").strip()
            full = urljoin(page, href)
            host = urlparse(full).netloc

            if looks_like_data_file(full, allowed_extensions) and keyword_match(full, text, keywords):
                links.add(full)
                continue

            # Crawl likely index pages.
            if host.endswith("dol.gov") and any(k in full.lower() for k in ["disclosure", "performance", "oflc", "foreign-labor"]):
                if full not in visited:
                    q.append(full)

        # Parse raw URLs embedded in scripts/JSON blobs.
        for raw in re.findall(r"https?://[^\"'\s>]+", res.text):
            cleaned = raw.rstrip(".,;)")
            if looks_like_data_file(cleaned, allowed_extensions) and keyword_match(cleaned, "", keywords):
                links.add(cleaned)

    return sorted(links)


def discover_fy_links(
    session: requests.Session,
    file_base: str,
    since_fy: int,
    through_fy: int,
    allowed_extensions: set[str],
) -> list[str]:
    """Probe common fiscal-year naming patterns with GET for reliability."""
    links: set[str] = set()

    stems: list[str] = []
    for fy in range(since_fy, through_fy + 1):
        stems.append(f"PERM_Disclosure_Data_FY{fy}")
        for q in range(1, 5):
            stems.extend(
                [
                    f"PERM_Disclosure_Data_FY{fy}_Q{q}",
                    f"PERM_Disclosure_Data_FY{fy}_QTR{q}",
                    f"PERM_Disclosure_Data_FY{fy}_Quarter{q}",
                ]
            )

    for stem in stems:
        for ext in sorted(allowed_extensions):
            url = f"{file_base.rstrip('/')}/{stem}{ext}"
            try:
                r = session.get(url, timeout=30, stream=True)
                ok = r.status_code == 200 and "text/html" not in (r.headers.get("content-type") or "").lower()
                r.close()
                if ok:
                    links.add(url)
            except Exception:
                continue

    return sorted(links)


def download_file(session: requests.Session, url: str, out_dir: Path) -> Path | None:
    filename = clean_filename(url)
    out = out_dir / filename

    try:
        r = session.get(url, timeout=180)
        r.raise_for_status()
    except Exception:
        return None

    out.write_bytes(r.content)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic disclosure-data downloader")
    parser.add_argument("--out", default="data/raw", help="Output directory")
    parser.add_argument("--max-files", type=int, default=0, help="Max files to download (0 = all)")
    parser.add_argument("--start-page", action="append", default=[], help="Start page(s) for crawl (repeatable)")
    parser.add_argument("--extensions", default=",".join(sorted(DEFAULT_EXTENSIONS)), help="Allowed extensions, comma-separated")
    parser.add_argument("--keywords", default=",".join(DEFAULT_KEYWORDS), help="Keyword filters, comma-separated")
    parser.add_argument("--max-pages", type=int, default=25, help="Max HTML pages to crawl")
    parser.add_argument("--since-fy", type=int, default=None, help="Probe fiscal-year file patterns from this FY")
    parser.add_argument("--through-fy", type=int, default=None, help="End FY for pattern probing")
    parser.add_argument("--years", default=None, help="Exact fiscal years to include, comma-separated (e.g. 2020,2021,2024)")
    parser.add_argument("--year-range", default=None, help="Inclusive fiscal year range like 2020-2026")
    parser.add_argument("--file-base", default=DEFAULT_FILE_BASE, help="Base URL for FY pattern probing")
    parser.add_argument("--dry-run", action="store_true", help="Only list discovered links")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_extensions = parse_extensions(args.extensions)
    keywords = parse_keywords(args.keywords)
    start_pages = args.start_page or list(DEFAULT_START_PAGES)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "perm-analysis/1.0 (research dashboard)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    links = set(discover_links_crawl(session, start_pages, allowed_extensions, keywords, args.max_pages))

    selected_years: list[int] = []
    if args.years:
        selected_years.extend(parse_years(args.years))
    if args.year_range:
        left, right = [int(x.strip()) for x in args.year_range.split("-", 1)]
        lo, hi = min(left, right), max(left, right)
        selected_years.extend(list(range(lo, hi + 1)))
    selected_years = sorted(set(selected_years))

    if selected_years:
        for fy in selected_years:
            links.update(discover_fy_links(session, args.file_base, fy, fy, allowed_extensions))
    elif args.since_fy is not None:
        through = args.through_fy or (date.today().year + 1)
        links.update(discover_fy_links(session, args.file_base, args.since_fy, through, allowed_extensions))

    links = sorted(links)
    if not links:
        print("No candidate links discovered.")
        return

    print(f"Discovered {len(links)} candidate links")
    for url in links:
        print(f"CANDIDATE {url}")

    if args.dry_run:
        return

    downloaded = 0
    for url in links:
        if args.max_files > 0 and downloaded >= args.max_files:
            break

        path = download_file(session, url, out_dir)
        if path is None:
            print(f"FAILED {url}")
            continue

        downloaded += 1
        print(f"DOWNLOADED {path.name}")

    print(f"Downloaded {downloaded} files into {out_dir}")


if __name__ == "__main__":
    main()
