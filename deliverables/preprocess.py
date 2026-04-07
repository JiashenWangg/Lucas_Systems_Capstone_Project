"""
preprocess.py
-------------
Load, clean and join raw warehouse CSVs into a single parquet for training.

First time (full preprocess):
    python preprocess.py OE

Daily update (append new picks to existing parquet):
    python preprocess.py OE --new_data daily_activity.csv
    python preprocess.py OE --new_data daily_activity.csv --max_days 180

Args:
    warehouse:   Warehouse code (OE, OF, RT)
    --new_data:  Optional CSV of new completed picks to append
    --data_path: Root training_data directory (default: training_data)
    --threshold: Quantile threshold for time delta filtering (default: 0.98)
    --max_days:  Trim parquet to last N days after update (default: no limit)

Output:
    training_data/WH/WH_startdate_enddate.parquet
    Log: rows loaded, percentage dropped, day range, number of days
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLUMN_NAMES = {
    "Activity":  ["ActivityCode", "UserID", "WorkCode", "AssignmentID",
                  "ProductID", "Quantity", "Timestamp", "LocationID"],
    "Locations": ["LocationID", "Aisle", "Bay", "Level", "Slot"],
    "Products":  ["ProductID", "ProductCode", "UnitOfMeasure", "Weight", "Cube",
                  "Length", "Width", "Height"],
}


# CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess warehouse training data")
    parser.add_argument("warehouse",   help="Warehouse code e.g. OE, OF, RT")
    parser.add_argument("--new_data",  default=None,
                        help="CSV of new completed picks to append to existing parquet")
    parser.add_argument("--data_path", default="training_data",
                        help="Root training_data directory (default: training_data)")
    parser.add_argument("--threshold", type=float, default=0.98,
                        help="Quantile threshold for time delta filtering (default: 0.98)")
    parser.add_argument("--max_days",  type=int, default=None,
                        help="Trim parquet to last N days after update (default: no limit)")
    return parser.parse_args()


# Helpers ────────────────────────────────────────────────────────────────────
def to_int(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")


def find_existing_parquet(wh_dir, warehouse):
    """Find the most recent processed parquet if one exists."""
    parquets = sorted(wh_dir.glob(f"{warehouse.upper()}_*.parquet"))
    return parquets[-1] if parquets else None


def make_parquet_name(warehouse, df):
    """Generate parquet filename from date range in data."""
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    start_date = ts.min().strftime("%Y%m%d")
    end_date = ts.max().strftime("%Y%m%d")
    return f"{warehouse.upper()}_{start_date}_{end_date}.parquet"


# Load ───────────────────────────────────────────────────────────────────────
def load_tables(wh_dir, warehouse):
    wh        = warehouse.upper()
    activity  = pd.read_csv(wh_dir / f"{wh}_Activity.csv",
                            header=0, names=COLUMN_NAMES["Activity"])
    locations = pd.read_csv(wh_dir / f"{wh}_Locations.csv",
                            header=0, names=COLUMN_NAMES["Locations"])
    products  = pd.read_csv(wh_dir / f"{wh}_Products.csv",
                            header=0, names=COLUMN_NAMES["Products"])

    logger.info(f"Loaded Activity:  {len(activity):,} rows")
    logger.info(f"Loaded Locations: {len(locations):,} rows")
    logger.info(f"Loaded Products:  {len(products):,} rows")

    return activity, locations, products


# Clean ──────────────────────────────────────────────────────────────────────
def clean_activity(activity):
    to_int(activity, ["ProductID", "Quantity", "LocationID"])
    activity["Timestamp"] = pd.to_datetime(activity["Timestamp"],
                                           errors="coerce")
    activity["UserID"] = activity["UserID"].astype(str)
    activity["WorkCode"] = activity["WorkCode"].astype(str).apply(
        lambda x: x.split(".")[0] if isinstance(x, str) else x
    )
    activity["AssignmentID"] = activity["AssignmentID"].astype(str)
    activity = activity.dropna(subset=["Timestamp"]).copy()

    if activity.empty:
        raise ValueError("Activity data is empty after cleaning — check input file")

    return activity


def clean_locations(locations):
    to_int(locations, ["LocationID", "Bay", "Level", "Slot"])
    return locations


def clean_products(products):
    to_int(products, ["ProductID"])
    return products[["ProductID", "ProductCode", "UnitOfMeasure",
                     "Weight", "Cube"]]


# Process ────────────────────────────────────────────────────────────────────
def compute_time_deltas(activity, threshold_q):
    df = activity.sort_values(["UserID", "Timestamp"]).reset_index(drop=True)

    g = df.groupby("UserID", sort=False)
    df["Prev_Timestamp"] = g["Timestamp"].shift(1)
    df["Prev_LocationID"] = g["LocationID"].shift(1)
    df["Time_Delta_sec"] = (df["Timestamp"] - df["Prev_Timestamp"]).dt.total_seconds()

    n_before = df["Time_Delta_sec"].notna().sum()
    threshold = df["Time_Delta_sec"].quantile(threshold_q)
    df.loc[df["Time_Delta_sec"] > threshold, "Time_Delta_sec"] = np.nan
    n_after = df["Time_Delta_sec"].notna().sum()
    pct_drop = 100 * (1 - n_after / n_before) if n_before > 0 else 0

    logger.info(f"Time delta threshold ({threshold_q:.0%}): {threshold:.1f}s")
    logger.info(f"Valid time deltas: {n_after:,} ({pct_drop:.1f}% dropped)")

    return df


def join_data(activity, products, locations):
    df = activity.merge(products, on="ProductID", how="left")
    df = df.merge(locations, on="LocationID", how="left")

    missing_products = df["ProductCode"].isna().mean()
    missing_locations = df["Aisle"].isna().mean()
    if missing_products > 0.1:
        logger.warning(f"{missing_products:.1%} of rows missing product info")
    if missing_locations > 0.1:
        logger.warning(f"{missing_locations:.1%} of rows missing location info")

    return df


def drop_assignment_open(df):
    """Drop AssignmentOpen rows and the first pick after them."""
    if "ActivityCode" not in df.columns:
        return df
    open_idx = df[df["ActivityCode"] == "AssignmentOpen"].index
    first_idx = open_idx + 1
    to_drop = open_idx.union(first_idx).intersection(df.index)
    return df.drop(to_drop).reset_index(drop=True)


# Full preprocess ────────────────────────────────────────────────────────────
def run_full_preprocess(wh_dir, warehouse, threshold_q):
    """
    Process all historical CSVs from scratch.
    Called on first run or when doing a full retrain.
    """
    logger.info("Running full preprocess on all historical data...")

    activity, locations, products = load_tables(wh_dir, warehouse)

    n_raw = len(activity)
    activity = clean_activity(activity)
    locations = clean_locations(locations)
    products = clean_products(products)

    activity_prepped = compute_time_deltas(activity, threshold_q)
    df = join_data(activity_prepped, products, locations)
    df = drop_assignment_open(df)

    pct_drop = 100 * (1 - len(df) / n_raw)
    logger.info(f"Raw rows: {n_raw:,} → Final rows: {len(df):,} ({pct_drop:.1f}% dropped)")

    return df, locations, products


# Incremental append ─────────────────────────────────────────────────────────
def run_incremental_preprocess(new_data_csv, locations, products, threshold_q):
    """
    Process only new picks from a daily CSV and return processed rows
    ready to append to the existing parquet.
    """
    logger.info(f"Running incremental preprocess on: {new_data_csv}")

    new_activity = pd.read_csv(new_data_csv,
                               header=0, names=COLUMN_NAMES["Activity"])
    logger.info(f"New rows loaded: {len(new_activity):,}")

    n_raw = len(new_activity)
    new_activity = clean_activity(new_activity)
    new_prepped = compute_time_deltas(new_activity, threshold_q)
    df_new = join_data(new_prepped, products, locations)
    df_new = drop_assignment_open(df_new)

    pct_drop = 100 * (1 - len(df_new) / n_raw)
    logger.info(f"New rows: {n_raw:,} → Processed: {len(df_new):,} ({pct_drop:.1f}% dropped)")

    return df_new


# Export ─────────────────────────────────────────────────────────────────────
def export_parquet(df, wh_dir, warehouse, existing_parquet=None):
    """
    Save parquet with date-range filename: WH_startdate_enddate.parquet
    Removes old parquet if filename changes due to updated date range.
    """
    filename = make_parquet_name(warehouse, df)
    out_path = wh_dir / filename
    df.to_parquet(out_path, index=False)

    if existing_parquet and existing_parquet != out_path and existing_parquet.exists():
        existing_parquet.unlink()
        logger.info(f"Removed old parquet: {existing_parquet.name}")

    return out_path


def log_summary(df, warehouse):
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    start_date = ts.min().strftime("%Y-%m-%d")
    end_date = ts.max().strftime("%Y-%m-%d")
    n_days = ts.dt.date.nunique()
    workcodes = sorted(df["WorkCode"].dropna().unique().tolist())

    logger.info(f"\n{'='*50}")
    logger.info(f"PREPROCESS COMPLETE — {warehouse}")
    logger.info(f"{'='*50}")
    logger.info(f"  Rows:       {len(df):,}")
    logger.info(f"  Date range: {start_date} — {end_date} ({n_days} days)")
    logger.info(f"  WorkCodes:  {workcodes}")
    logger.info(f"{'='*50}")


# Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    warehouse = args.warehouse.upper()
    wh_dir = Path(args.data_path) / warehouse

    if not wh_dir.exists():
        raise FileNotFoundError(f"Warehouse directory not found: {wh_dir}")

    existing_parquet = find_existing_parquet(wh_dir, warehouse)

    if args.new_data is None:
        # Full preprocess ────────────────────────────────────────────────────
        df, locations, products = run_full_preprocess(
            wh_dir, warehouse, args.threshold
        )

    else:
        # Incremental append ─────────────────────────────────────────────────
        if existing_parquet is None:
            logger.warning(
                "No existing parquet found — running full preprocess instead of append"
            )
            df, locations, products = run_full_preprocess(
                wh_dir, warehouse, args.threshold
            )
        else:
            logger.info(f"Existing parquet: {existing_parquet.name}")

            _, locations, products = load_tables(wh_dir, warehouse)
            locations = clean_locations(locations)
            products = clean_products(products)

            df_new = run_incremental_preprocess(
                args.new_data, locations, products, args.threshold
            )

            df_existing = pd.read_parquet(existing_parquet)
            df = pd.concat([df_existing, df_new], ignore_index=True)
            df = df.drop_duplicates(
                subset=["UserID", "Timestamp", "LocationID"], keep="last"
            )
            df = df.sort_values(["UserID", "Timestamp"]).reset_index(drop=True)
            logger.info(
                f"Appended {len(df_new):,} new rows — total: {len(df):,} rows"
            )

    # Trim to max_days if specified
    if args.max_days is not None:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        all_days = sorted(ts.dt.date.dropna().unique())
        if len(all_days) > args.max_days:
            cutoff = all_days[-args.max_days]
            n_before = len(df)
            df = df[ts.dt.date >= cutoff].copy()
            logger.info(
                f"Trimmed to last {args.max_days} days: "
                f"{n_before:,} → {len(df):,} rows (cutoff: {cutoff})"
            )

    out_path = export_parquet(df, wh_dir, warehouse, existing_parquet)
    logger.info(f"Saved: {out_path}")

    log_summary(df, warehouse)


if __name__ == "__main__":
    main()
