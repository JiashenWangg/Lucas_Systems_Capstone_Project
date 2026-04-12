"""
utils/data_pipeline.py
-----------------------
Convenience wrappers that chain data loading, preprocessing, and feature
engineering into single calls. Used by model_training.py, eval.py, and
update_model_incremental.py so they don't have to orchestrate multiple
steps manually.

Public API
----------
load_and_engineer(data_dir, warehouse, wc, sequenced=False,
                  encodings=None, models_dir=None)
    Load parquet, filter to one WorkCode, apply feature engineering.
    If encodings=None, computes them from the data (training mode).
    If encodings provided, applies them without recomputing (update/predict mode).
    Returns (df_engineered, X, y, encodings).

prepare_new_data(activity_csv, data_dir, warehouse, wc,
                 encodings, sequenced=False)
    Load a raw daily Activity CSV, run preprocessing steps, apply
    saved feature encodings. Used by update_model_incremental.py.
    Returns (df_engineered, X, y).

prepare_predict_data(predict_csv, data_dir, warehouse,
                     encodings, sequenced=False)
    Load a work queue CSV (no timestamps), apply saved feature encodings.
    Used by predict_primary.py.
    Returns (df_engineered, X, detected_wc).
"""

import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from utils.feature_engineer import (
    apply_features,
    compute_encodings,
    compute_prev_location,
    make_X,
)
from utils.io import (
    load_activity_csv,
    load_distance_matrix,
    load_parquet,
    load_predict_csv,
    load_reference_tables,
)

logger = logging.getLogger(__name__)

MAX_TIME     = 600   # hard cap after percentile filter (seconds)
PERCENTILE   = 98    # keep bottom PERCENTILE % of Time_Delta_sec


# ── Internal helpers ──────────────────────────────────────────────────────────

def _filter_time(df, label=""):
    """
    Apply two-stage time filter:
      1. Remove top (100 - PERCENTILE)% of Time_Delta_sec globally.
      2. Hard cap at MAX_TIME seconds.
    Logs how many rows were dropped at each stage.
    """
    n_start = len(df)

    # Stage 1: percentile cut
    threshold = df["Time_Delta_sec"].quantile(PERCENTILE / 100)
    df = df[df["Time_Delta_sec"] <= threshold].copy()
    n_after_pct = len(df)

    # Stage 2: hard cap
    df = df[df["Time_Delta_sec"] <= MAX_TIME].copy()
    n_after_cap = len(df)

    dropped_pct = n_start - n_after_pct
    dropped_cap = n_after_pct - n_after_cap
    total_dropped = n_start - n_after_cap

    tag = f" [{label}]" if label else ""
    logger.info(
        f"{tag} Time filter: {n_start:,} → {n_after_cap:,} rows kept "
        f"(p{PERCENTILE} cut: -{dropped_pct:,}, "
        f"{MAX_TIME}s cap: -{dropped_cap:,}, "
        f"total dropped: {total_dropped:,} = "
        f"{100*total_dropped/n_start:.1f}%)"
    )
    return df


def _clean_workcode(df):
    df = df.copy()
    df["WorkCode"] = (
        df["WorkCode"].astype(str)
        .apply(lambda x: x.split(".")[0] if isinstance(x, str) else x)
    )
    return df


def _compute_time_delta(df):
    """
    Compute Time_Delta_sec from consecutive Timestamps per user.
    Rows where the delta cannot be computed (first pick per user,
    or NaT timestamps) get NaN and are dropped downstream.
    """
    df = df.sort_values(["UserID", "Timestamp"]).copy()
    df["Prev_Timestamp"] = df.groupby("UserID")["Timestamp"].shift(1)
    df["Time_Delta_sec"] = (
        df["Timestamp"] - df["Prev_Timestamp"]
    ).dt.total_seconds()
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_engineer(data_dir, warehouse, wc,
                      sequenced=False, encodings=None,
                      data_dir_ref=None):
    """
    Load processed parquet, filter to one WorkCode, apply feature engineering.

    In training mode (encodings=None):
        - Computes top_aisles, top_uoms, product_tiers from this WorkCode's data
        - Returns computed encodings so caller can save them to meta.pkl

    In update/eval mode (encodings provided):
        - Applies saved encodings without recomputing

    Args:
        data_dir:    Root training_data directory containing WH_Processed.parquet
        warehouse:   Warehouse code e.g. "OE"
        wc:          WorkCode string e.g. "30"
        sequenced:   Whether to compute sequence-dependent features
        encodings:   Dict with top_aisles, top_uoms, product_tiers, locations_df,
                     distance_df (None = training mode, compute from data)
        data_dir_ref: If sequenced=True and encodings=None, path to training_data
                     dir to load Locations and Distance tables from. Defaults to
                     data_dir if not provided.

    Returns:
        df:        Feature-engineered DataFrame (includes target Time_Delta_sec)
        X:         Feature matrix (pd.DataFrame of floats)
        y:         Target series (Time_Delta_sec)
        encodings: Dict of encodings (same object if passed in, newly computed if not)
    """
    df = load_parquet(data_dir, warehouse)
    df = _clean_workcode(df)

    df = df[df["WorkCode"] == str(wc)].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for WorkCode={wc} in {warehouse} parquet."
        )

    df = df.dropna(subset=["Time_Delta_sec"]).copy()
    df = df[df["Time_Delta_sec"] > 0].copy()

    # Load reference tables for sequenced mode
    ref_dir = data_dir_ref or data_dir
    locations_df = None
    distance_df  = None

    if sequenced:
        locations_df, _ = load_reference_tables(ref_dir, warehouse)
        distance_df = load_distance_matrix(ref_dir, warehouse)
        if distance_df is None:
            logger.info(
                "No distance matrix found — Travel_Distance will be 0.0"
            )
        df = compute_prev_location(df)

    # Compute or apply encodings
    if encodings is None:
        encodings = compute_encodings(df)
        encodings["locations_df"] = locations_df
        encodings["distance_df"]  = distance_df
        logger.info(
            f"  Encodings computed from {len(df):,} rows "
            f"(WC {wc}, sequenced={sequenced})"
        )

    df = apply_features(
        df,
        top_aisles    = encodings["top_aisles"],
        top_uoms      = encodings["top_uoms"],
        product_tiers = encodings["product_tiers"],
        sequenced     = sequenced,
        locations_df  = encodings.get("locations_df"),
        distance_df   = encodings.get("distance_df"),
    )

    X = make_X(df, sequenced=sequenced)
    y = df["Time_Delta_sec"].astype(float)

    return df, X, y, encodings


def prepare_new_data(activity_csv, data_dir, warehouse, wc,
                     encodings, sequenced=False):
    """
    Load a raw daily Activity CSV for an update run.

    Runs full preprocessing (time delta, join products/locations, filter)
    then applies saved feature encodings. Used by update_model_incremental.py.

    Args:
        activity_csv: Path to raw Activity CSV (has Timestamps — completed picks)
        data_dir:     Root training_data directory (for Products and Locations)
        warehouse:    Warehouse code
        wc:           WorkCode to filter to
        encodings:    Saved encodings dict from meta.pkl
        sequenced:    Whether to compute sequence features

    Returns:
        df:  Feature-engineered DataFrame
        X:   Feature matrix aligned to train_columns
        y:   Target series (Time_Delta_sec)
    """
    locations_df, products_df = load_reference_tables(data_dir, warehouse)
    distance_df = load_distance_matrix(data_dir, warehouse) if sequenced else None

    df = load_activity_csv(activity_csv, warehouse)
    df = _clean_workcode(df)
    df = df.dropna(subset=["Timestamp"]).copy()

    # Compute time deltas from timestamps
    df = _compute_time_delta(df)
    df = df.dropna(subset=["Time_Delta_sec"]).copy()
    df = df[df["Time_Delta_sec"] > 0].copy()
    df = _filter_time(df, label=f"{warehouse} WC{wc} new data")

    # Filter to requested WorkCode
    df = df[df["WorkCode"] == str(wc)].copy()
    if df.empty:
        raise ValueError(
            f"No rows for WorkCode={wc} in {activity_csv} after filtering."
        )

    # Join product and location attributes
    df = df.merge(products_df,  on="ProductID",  how="left")
    df = df.merge(locations_df, on="LocationID", how="left")

    if sequenced:
        df = compute_prev_location(df)

    df = apply_features(
        df,
        top_aisles    = encodings["top_aisles"],
        top_uoms      = encodings["top_uoms"],
        product_tiers = encodings["product_tiers"],
        sequenced     = sequenced,
        locations_df  = locations_df,
        distance_df   = distance_df,
    )

    train_columns = encodings.get("train_columns")
    X = make_X(df, sequenced=sequenced, train_columns=train_columns)
    y = df["Time_Delta_sec"].astype(float)

    return df, X, y


def prepare_predict_data(predict_csv, data_dir, warehouse,
                         encodings, sequenced=False):
    """
    Load a work queue CSV (no timestamps) for prediction.

    Applies saved feature encodings and returns the feature matrix.
    WorkCode is auto-detected from the CSV. Raises ValueError if multiple
    WorkCodes are found.

    Args:
        predict_csv: Path to work queue CSV (no Time_Delta_sec column)
        data_dir:    Root training_data directory (for Products and Locations)
        warehouse:   Warehouse code
        encodings:   Saved encodings dict from meta.pkl
        sequenced:   Whether to compute sequence features

    Returns:
        df:  Feature-engineered DataFrame
        X:   Feature matrix aligned to train_columns
        wc:  Detected WorkCode string
    """
    locations_df, products_df = load_reference_tables(data_dir, warehouse)
    distance_df = load_distance_matrix(data_dir, warehouse) if sequenced else None

    df, wc = load_predict_csv(predict_csv)

    # Join product and location attributes
    df = df.merge(products_df,  on="ProductID",  how="left")
    df = df.merge(locations_df, on="LocationID", how="left")

    if sequenced:
        # For prediction, sequence order is the CSV row order
        df = df.reset_index(drop=True)
        df["Prev_LocationID"] = df.groupby("UserID")["LocationID"].shift(1)

    df = apply_features(
        df,
        top_aisles    = encodings["top_aisles"],
        top_uoms      = encodings["top_uoms"],
        product_tiers = encodings["product_tiers"],
        sequenced     = sequenced,
        locations_df  = locations_df,
        distance_df   = distance_df,
    )

    train_columns = encodings.get("train_columns")
    X = make_X(df, sequenced=sequenced, train_columns=train_columns)

    return df, X, wc
