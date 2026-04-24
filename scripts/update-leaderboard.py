#!/usr/bin/env python3
"""Merge new detector results into leaderboard/results.csv and regenerate index.html."""

import glob
import os
import sys
import polars as pl
from jinja2 import Environment, FileSystemLoader

REPO_URL = "https://github.com/wenig/touchstone-rs"
LEADERBOARD_DIR = "leaderboard"
DIST_DIR = os.path.join(LEADERBOARD_DIR, "dist")
RESULTS_CSV = os.path.join(DIST_DIR, "results.csv")
TEMPLATE_FILE = "template.html"
OUTPUT_HTML = os.path.join(DIST_DIR, "index.html")

METRICS = [
    "ROC-AUC", "PR-AUC", "AvgPrec",
    "Precision", "Recall", "F1",
    "RangePrec", "RangeRec", "RangeF1",
    "RangePR-AUC", "RangePR-VUS", "RangeROC-VUS",
    "time_sec",
]


def load_new_results() -> pl.DataFrame:
    csvs = glob.glob("touchstone-*.csv")
    if not csvs:
        print("No touchstone-*.csv files found", file=sys.stderr)
        sys.exit(1)
    return pl.concat([pl.read_csv(f) for f in csvs])


def merge_results(new: pl.DataFrame) -> pl.DataFrame:
    if os.path.exists(RESULTS_CSV):
        existing = pl.read_csv(RESULTS_CSV)
        updated_detectors = new["detector"].unique().implode()
        existing = existing.filter(~pl.col("detector").is_in(updated_detectors))
        return pl.concat([existing, new])
    return new


def build_leaderboard(df: pl.DataFrame) -> list[dict]:
    """Average each metric across datasets per detector, sorted by ROC-AUC desc."""
    agg = (
        df.group_by("detector")
        .agg([pl.col(m).mean() for m in METRICS])
        .sort("ROC-AUC", descending=True)
    )
    return agg.to_dicts()


def render_html(rows: list[dict]) -> str:
    env = Environment(loader=FileSystemLoader(LEADERBOARD_DIR), autoescape=True)
    template = env.get_template(TEMPLATE_FILE)
    return template.render(rows=rows, metrics=METRICS, repo_url=REPO_URL)


def main():
    os.makedirs(DIST_DIR, exist_ok=True)

    new = load_new_results()
    merged = merge_results(new)
    merged.write_csv(RESULTS_CSV)
    print(f"Saved {len(merged)} rows to {RESULTS_CSV}")

    rows = build_leaderboard(merged)
    html = render_html(rows)
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    print(f"Rendered leaderboard to {OUTPUT_HTML} ({len(rows)} detectors)")


if __name__ == "__main__":
    main()
