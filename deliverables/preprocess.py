"""
preprocess.py
-------------
Load and preprocess raw warehouse Activity data into a clean parquet file
ready for model training, evaluation, and incremental updates.

Usage:
    python preprocess.py OE
    python preprocess.py OE --data_dir training_data

Args:
    warehouse:   Warehouse code (OE, OF, RT)
    --data_dir:  Root training_data directory (default: training_data)

Input (in training_data/WH/):
    WH_Activity.csv
    WH_Locations.csv
    WH_Products.csv
    WH_Distance_Matrix.csv  (optional)

Output:
    training_data/WH/WH_Processed.parquet
    logs/WH/preprocess_YYYY-MM-DD.log

Processing steps:
    1. Load Activity, Locations, Products (and Distance if available)
    2. Sort each user's picks chronologically, compute Time_Delta_sec
    3. Filter: remove top 2% of Time_Delta_sec globally, then cap at 600s
    4. Join Locations and Products attributes onto each pick row
    5. If Distance matrix available, join Travel_Distance per pick
    6. Save to parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils.io import (
    load_activity_csv,
    load_distance_matrix,
    load_reference_tables,
    save_parquet,
    setup_logging,
)

MAX_TIME   = 600
PERCENTILE = 98


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw warehouse Activity data to parquet"
    )
    parser.add_argument("warehouse", help="Warehouse code e.g. OE, OF, RT")
    parser.add_argument(
        "--data_dir", default="training_data",
        help="Root training_data directory (default: training_data)"
    )
    return parser.parse_args()


def compute_time_delta(df):
    """
    Sort each user's picks by Timestamp and compute Time_Delta_sec as the
    elapsed seconds since their previous pick. First pick per user gets NaN.
    """
    df = df.sort_values(["UserID", "Timestamp"]).reset_index(drop=True)
    df["Prev_Timestamp"] = df.groupby("UserID")["Timestamp"].shift(1)
    df["Prev_LocationID"] = df.groupby("UserID")["LocationID"].shift(1)
    df["Time_Delta_sec"] = (
        df["Timestamp"] - df["Prev_Timestamp"]
    ).dt.total_seconds()
    return df


def filter_time(df, logger):
    """
    Two-stage time filter:
      Stage 1 — remove top (100 - PERCENTILE)% of Time_Delta_sec globally.
      Stage 2 — hard cap at MAX_TIME seconds.
    """
    n_start = len(df)

    threshold = df["Time_Delta_sec"].quantile(PERCENTILE / 100)
    df = df[df["Time_Delta_sec"] <= threshold].copy()
    n_after_pct = len(df)

    df = df[df["Time_Delta_sec"] <= MAX_TIME].copy()
    n_after_cap = len(df)

    dropped_pct = n_start - n_after_pct
    dropped_cap = n_after_pct - n_after_cap
    total       = n_start - n_after_cap

    logger.info(
        f"  Time filter:  {n_start:,} rows in"
    )
    logger.info(
        f"    p{PERCENTILE} cut (≤ {threshold:.1f}s): "
        f"removed {dropped_pct:,} rows"
    )
    logger.info(
        f"    {MAX_TIME}s hard cap:          "
        f"removed {dropped_cap:,} rows"
    )
    logger.info(
        f"    Total dropped: {total:,} "
        f"({100 * total / n_start:.1f}%)"
    )
    logger.info(f"    Rows retained: {n_after_cap:,}")
    return df


def main():
    args      = parse_args()
    warehouse = args.warehouse.upper()
    data_dir  = args.data_dir

    setup_logging("preprocess", warehouse)
    logger = logging.getLogger(__name__)

    logger.info(f"{'='*60}")
    logger.info(f"PREPROCESS — {warehouse}")
    logger.info(f"{'='*60}")

    wh_dir = Path(data_dir) / warehouse

    # ── Load Activity ─────────────────────────────────────────────────────────
    activity_path = wh_dir / f"{warehouse}_Activity.csv"
    logger.info(f"Loading Activity: {activity_path}")
    df = load_activity_csv(activity_path, warehouse)
    n_raw = len(df)
    logger.info(f"  Rows loaded: {n_raw:,}")

    df = df.dropna(subset=["Timestamp"]).copy()
    n_after_ts = len(df)
    if n_after_ts < n_raw:
        logger.info(
            f"  Dropped {n_raw - n_after_ts:,} rows with invalid Timestamp"
        )

    # ── Compute time deltas ───────────────────────────────────────────────────
    logger.info("Computing time deltas ...")
    df = compute_time_delta(df)

    n_before_filter = df["Time_Delta_sec"].notna().sum()
    df = df.dropna(subset=["Time_Delta_sec"]).copy()
    df = df[df["Time_Delta_sec"] > 0].copy()
    logger.info(
        f"  {len(df):,} rows with valid positive Time_Delta_sec "
        f"(first pick per user dropped)"
    )

    # ── Filter outliers ───────────────────────────────────────────────────────
    logger.info("Filtering time outliers ...")
    df = filter_time(df, logger)

    # ── Load reference tables ─────────────────────────────────────────────────
    logger.info("Loading Locations and Products ...")
    locations_df, products_df = load_reference_tables(data_dir, warehouse)
    logger.info(
        f"  Locations: {len(locations_df):,} rows  |  "
        f"Products: {len(products_df):,} rows"
    )

    # ── Join Locations and Products ───────────────────────────────────────────
    logger.info("Joining Locations and Products ...")
    df = df.merge(products_df,  on="ProductID",  how="left")
    df = df.merge(locations_df, on="LocationID", how="left")

    # Join previous location attributes (for sequenced mode later)
    prev_loc_cols = locations_df[["LocationID", "Aisle", "Level"]].rename(columns={
        "LocationID": "Prev_LocationID",
        "Aisle":      "Prev_Aisle",
        "Level":      "Prev_Level",
    })
    df = df.merge(prev_loc_cols, on="Prev_LocationID", how="left")

    # ── Load Distance matrix (optional) ──────────────────────────────────────
    logger.info("Checking for Distance matrix ...")
    distance_df = load_distance_matrix(data_dir, warehouse)
    if distance_df is not None:
        logger.info(
            f"  Distance matrix found ({len(distance_df):,} location pairs) — joining"
        )
        # Build LocKey for current and previous location to join distance
        df["LocKey"]     = df["LocationID"].astype(str)
        df["PrevLocKey"] = df["Prev_LocationID"].astype(str)

        distance_df = distance_df.rename(columns={
            "FromLoc":  "PrevLocKey",
            "ToLoc":    "LocKey",
        })
        distance_df["PrevLocKey"] = distance_df["PrevLocKey"].astype(str)
        distance_df["LocKey"]     = distance_df["LocKey"].astype(str)

        df = df.merge(distance_df, on=["PrevLocKey", "LocKey"], how="left")
        df["Travel_Distance"] = df["distance"].fillna(0.0)
        df = df.drop(columns=["distance"], errors="ignore")
    else:
        logger.info(
            "  No distance matrix found — Travel_Distance column will be absent. "
            "Sequenced prediction will not be available unless distance matrix "
            "is added and preprocess.py is rerun."
        )

    # ── Summary stats ─────────────────────────────────────────────────────────
    date_min = df["Timestamp"].dt.date.min()
    date_max = df["Timestamp"].dt.date.max()
    n_days   = (date_max - date_min).days + 1
    wc_counts = df["WorkCode"].value_counts().to_dict()

    logger.info(f"\nSummary:")
    logger.info(f"  Date range:    {date_min} → {date_max}  ({n_days} days)")
    logger.info(f"  Final rows:    {len(df):,}")
    logger.info(f"  Workers:       {df['UserID'].nunique():,}")
    logger.info(f"  WorkCodes:     {wc_counts}")
    logger.info(
        f"  Total dropped: {n_raw - len(df):,} "
        f"({100*(n_raw - len(df))/n_raw:.1f}%)"
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = save_parquet(df, data_dir, warehouse)
    logger.info(f"\nSaved: {out_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
