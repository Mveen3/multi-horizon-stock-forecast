"""
Preprocessing Pipeline
================================

Pipeline:
  Phase 1 — Raw CSV Ingest & Merge
      • Scan datasets/ for *_news.csv source files.
      • Normalize columns to (date, title, news, url).
      • Per-file dedup + cross-file global dedup on URL.
      • Output: datasets/combined_market_news.csv

  Phase 2 — News Cleaning
      • HTML entity decode, ALSO READ removal, whitespace normalization.
      • Short article filtering (< 100 chars).
      • Source domain extraction.
      • Date parsing & date-range enforcement.
      • Output: datasets/processed/cleaned_news.csv

Date Window: January 1, 2023  →  February 28, 2026


Usage:
    python unified_preprocessor.py              # full pipeline
    python unified_preprocessor.py --skip-merge  # skip Phase 1 (already merged)
    python unified_preprocessor.py --force       # re-run all steps even if outputs exist
"""

import os
import re
import html
import glob
import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent   
DATASET_DIR = BASE_DIR / "datasets"
OUT_DIR     = DATASET_DIR / "processed"

NEWS_COMBINED = DATASET_DIR / "combined_market_news.csv"

# ── Date window  ──────────────────────────────────────────────────────────────
START_DATE = "2023-01-01"
END_DATE   = "2026-02-28"

# ── Target CSV columns ────────────────────────────────────────────────────────
TARGET_COLUMNS = ["date", "title", "news", "url"]

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: RAW CSV INGEST & MERGE
# ══════════════════════════════════════════════════════════════════════════════

def phase1_merge_raw_sources(force: bool = False):
    """
    Scan datasets/ for per-source news CSVs, normalize and deduplicate,
    write combined_market_news.csv.
    """
    print("\n" + "=" * 70)
    print("PHASE 1 — Raw CSV Ingest & Merge")
    print("=" * 70)

    if NEWS_COMBINED.exists() and not force:
        existing = _count_lines(NEWS_COMBINED) - 1  # minus header
        print(f"  combined_market_news.csv already exists ({existing:,} rows).")
        print(f"  Use --force to regenerate. Skipping Phase 1.")
        return

    # Find all per-source CSVs (pattern: *_news.csv, excluding the combined file)
    all_csvs = sorted(DATASET_DIR.glob("*_news.csv"))
    all_csvs = [f for f in all_csvs if f.name != "combined_market_news.csv"]

    if not all_csvs:
        print("  No *_news.csv source files found in datasets/. Nothing to merge.")
        return

    frames = []
    for filepath in all_csvs:
        print(f"\n  Processing: {filepath.name}")
        try:
            df = pd.read_csv(filepath, on_bad_lines="skip", engine="python")
            initial = len(df)
            print(f"    Read {initial:,} rows")

            if initial == 0:
                continue

            # ── Normalize column names ──
            col_lower = [str(c).strip().lower() for c in df.columns]
            has_header = any(k in c for c in col_lower
                            for k in ["date", "title", "news", "url"])

            if not has_header:
                # Headers are actually data — push them down
                df.loc[-1] = df.columns.tolist()
                df.index = df.index + 1
                df = df.sort_index()

            if len(df.columns) < 4:
                print(f"    ✗ Fewer than 4 columns — skipping.")
                continue

            # Keep first 4 columns → rename to standard
            df = df.iloc[:, :4]
            df.columns = TARGET_COLUMNS

            # ── Type coercion & cleaning ──
            df = df.dropna(subset=["title", "news", "date"])
            df["title"] = df["title"].astype(str).str.replace(r"[\n\r]+", ". ", regex=True).str.strip()
            df["news"]  = df["news"].astype(str).str.replace(r"[\n\r]+", ". ", regex=True).str.strip()

            # Drop stubs
            df = df[(df["title"].str.len() > 3) & (df["news"].str.len() > 5)]

            # ── Date parsing (coerce unparseable → NaT → drop) ──
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

            # ── Per-file URL dedup ──
            df = df.drop_duplicates(subset=["url"], keep="first")

            print(f"    Cleaned: {len(df):,} rows (dropped {initial - len(df):,})")
            frames.append(df)

        except Exception as e:
            print(f"    ✗ Failed: {e}")

    if not frames:
        print("\n  No valid data to merge.")
        return

    # ── Global merge & dedup ──
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["url"], keep="first")

    # Sort newest-first
    combined["_dt"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values("_dt", ascending=False).drop(columns=["_dt"])
    combined.reset_index(drop=True, inplace=True)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(NEWS_COMBINED, index=False)

    print(f"\n  Merged: {before:,} → {len(combined):,} (global URL dedup)")
    print(f"  ✓ Saved: {NEWS_COMBINED.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2:
# ══════════════════════════════════════════════════════════════════════════════

# Pre-compile heavy regex once
_ALSO_READ_RE = re.compile(
    r"ALSO\s*READ\s*[:\-–]?\s*.{5,300}?(?=\.\s+[A-Z]|\.\s*$)",
    re.DOTALL,
)
_MULTI_SPACE_RE = re.compile(r"\s+")


def _clean_text_column(series: pd.Series) -> pd.Series:
    """Vectorized text cleaning: HTML decode → ALSO READ removal → whitespace."""
    s = series.apply(html.unescape)
    s = s.apply(lambda x: _ALSO_READ_RE.sub("", x))
    s = s.str.replace(_MULTI_SPACE_RE, " ", regex=True).str.strip()
    return s


def phase2_clean_news(force: bool = False):
    """Clean combined_market_news.csv → cleaned_news.csv."""
    print("\n" + "=" * 70)
    print("PHASE 2")
    print("=" * 70)

    out_path = OUT_DIR / "cleaned_news.csv"
    if out_path.exists() and not force:
        n = _count_lines(out_path) - 1
        print(f"  cleaned_news.csv exists ({n:,} rows). Use --force to redo. Skipping.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not NEWS_COMBINED.exists():
        print("  ✗ combined_market_news.csv not found. Run Phase 1 first.")
        return

    print("  Loading combined news...")
    # For very large files, use chunked reading to keep RAM in check
    file_size_mb = NEWS_COMBINED.stat().st_size / (1024 * 1024)
    if file_size_mb > 500:
        # Chunked processing for 500 MB+ files
        print(f"  Large file ({file_size_mb:.0f} MB) — processing in chunks...")
        _clean_news_chunked(out_path)
    else:
        df = pd.read_csv(NEWS_COMBINED)
        print(f"  Loaded {len(df):,} articles ({file_size_mb:.0f} MB)")
        df = _clean_news_dataframe(df)
        df.to_csv(out_path, index=False)
        print(f"  ✓ Saved {len(df):,} articles → {out_path.name}")


def _clean_news_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Full in-memory news cleaning."""
    print("  [1/5] Decoding HTML + removing ALSO READ...")
    df["title"] = _clean_text_column(df["title"])
    df["news"]  = _clean_text_column(df["news"])

    print("  [2/5] Removing short articles (< 100 chars)...")
    before = len(df)
    df = df[df["news"].str.len() >= 100].copy()
    print(f"         {before:,} → {len(df):,} (dropped {before - len(df):,})")

    print("  [3/5] Extracting source domain...")
    df["source"] = df["url"].str.extract(
        r"https?://(?:www\.)?([\w\-]+\.[\w]+)", expand=False
    )

    print("  [4/5] Parsing dates...")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["date"])

    # Enforce date window
    mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
    df = df[mask].copy()
    print(f"         Kept {len(df):,} in date range [{START_DATE} → {END_DATE}]")

    print("  [5/5] Sorting by date ascending...")
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    return df


def _clean_news_chunked(out_path: Path, chunk_size: int = 50_000):
    """Chunk-based cleaning for files too large to fit in RAM."""
    first_chunk = True
    total_rows = 0
    reader = pd.read_csv(NEWS_COMBINED, chunksize=chunk_size)

    for i, chunk in enumerate(reader, 1):
        print(f"  Chunk {i}: {len(chunk):,} rows...")
        cleaned = _clean_news_dataframe(chunk)
        cleaned.to_csv(
            out_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        total_rows += len(cleaned)
        first_chunk = False

    # Final sort (read → sort → overwrite)
    print(f"  Final sort of {total_rows:,} rows...")
    df = pd.read_csv(out_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(df):,} cleaned articles → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _count_lines(filepath: Path) -> int:
    """Count lines in a file without loading it into memory."""
    count = 0
    with open(filepath, "rb") as f:
        for _ in f:
            count += 1
    return count


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified preprocessing pipeline for stock forecasting project.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all steps even if output files already exist.",
    )
    global START_DATE, END_DATE

    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip Phase 1 (raw CSV merge). Useful when combined file is ready.",
    )
    parser.add_argument(
        "--start-date", default=None,
        help=f"Override start date (YYYY-MM-DD). Default: {START_DATE}",
    )
    parser.add_argument(
        "--end-date", default=None,
        help=f"Override end date (YYYY-MM-DD). Default: {END_DATE}",
    )
    args = parser.parse_args()
    if args.start_date:
        START_DATE = args.start_date
    if args.end_date:
        END_DATE = args.end_date

    print("=" * 70)
    print("  UNIFIED PREPROCESSING PIPELINE")
    print(f"  Date window: {START_DATE} → {END_DATE}")
    print("=" * 70)

    # Phase 1: Merge raw CSVs
    if not args.skip_merge:
        phase1_merge_raw_sources(force=args.force)
    else:
        print("\n  Phase 1 skipped (--skip-merge).")

    # Phase 2: Clean news
    phase2_clean_news(force=args.force)

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("  PREPROCESSING COMPLETE — Summary")
    print("=" * 70)

    for f in sorted(OUT_DIR.glob("*.csv")):
        rows = _count_lines(f) - 1
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:35s}  {rows:>8,} rows  ({size_kb:>8,.0f} KB)")

    print(f"\n  Output directory: {OUT_DIR.relative_to(BASE_DIR)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
