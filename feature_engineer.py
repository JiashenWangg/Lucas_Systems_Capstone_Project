"""
This module contains the function get_engineered_df which loads the data,
applies preprocessing and feature engineering steps, and returns the processed
DataFrame along with lists of feature columns and categorical columns for modeling.

Instructions:
- Place this file in the same directory as your main modeling script (e.g., xgboost.ipynb)
- Import the function using: from feature_engineer import get_engineered_df
- Example call:
    df, features, cat_cols = get_engineered_df("data/OE_30.parquet", warehouse="OE", max_time=300, work_code="30")

Note: For OE and OF only, RT might have different encodings
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_engineered_df(file_path, warehouse="OE", max_time=300, work_code="30"):
    """
    Loads data and applies preprocessing/feature engineering
    Args:
    - warehouse: "OE" or "OF" to determine aisle grouping logic (default "OE")
    - file_path: path to the parquet file containing the data
    - max_time: maximum Time_Delta_sec to consider for filtering (default 300s)
    - work_code: WorkCode to filter on (default '30')
    Returns:
    - df: the processed DataFrame ready for modeling
    - feature_cols: list of columns to be used as features
    - cat_cols: list of categorical feature columns
    """
    # Load data
    df = pd.read_parquet(file_path)
    # Ensure Timestamp and basic numerics
    df["Timestamp"] = pd.to_datetime(df.get("Timestamp"), errors="coerce")
    for col in ["Time_Delta_sec", "Weight", "Cube", "Quantity", "Travel_Distance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["WorkCode"] = df["WorkCode"].astype(str)
    df = df.dropna(subset=["Timestamp"]).copy()
    df = df[
        (df["Time_Delta_sec"] < max_time)
        & (df["Travel_Distance"] >= 0)
        & (df["WorkCode"] == work_code)
    ].copy()

    # Feature: Aisle Grouping, top-5 encoding
    top_aisles = df["Aisle"].value_counts().head(5).index
    df["Aisle"] = pd.to_numeric(df["Aisle"], errors="coerce").fillna(-1).astype(int)
    df["Aisle_group"] = df["Aisle"].apply(
        lambda a: str(a) if a in top_aisles else "other"
    )

    df["Aisle"] = pd.to_numeric(df["Aisle"], errors="coerce").fillna(-1).astype(int)

    # Feature: Level Grouping
    def level_group(l):
        try:
            val = int(l)
            return "5+" if val >= 5 else str(val)
        except:
            return str(l)

    df["Level_group"] = df["Level"].apply(level_group)

    # Feature: Time of Day Buckets
    df["hour"] = df["Timestamp"].dt.hour.astype(int)

    def tod_bucket(h):
        if 6 <= h < 12:
            return "6-12"
        elif 12 <= h < 16:
            return "12-4"
        elif 16 <= h < 20:
            return "4-8"
        elif 20 <= h <= 23:
            return "8-12"
        else:
            return "after_midnight"

    df["time_of_day"] = df["hour"].apply(tod_bucket)

    # Feature: UOM Grouping
    valid_uoms = ["EA", "BX", "PK", "CA", "CS"]
    df["UOM_group"] = df["UnitOfMeasure"].apply(
        lambda u: u if u in valid_uoms else "other"
    )

    # Feature: Day of Week
    df["day_of_week"] = df["Timestamp"].dt.day_name()

    # Feature: Relationship with Previous Row
    df["Prev_Aisle"] = (
        pd.to_numeric(df["Prev_Aisle"], errors="coerce").fillna(-1).astype(int)
    )
    df["same_aisle"] = (df["Aisle"] == df["Prev_Aisle"]).astype(int)
    df["same_lockey"] = (df["LocKey"] == df["PrevLocKey"]).astype(int)
    df["diff_level"] = (
        (df["LocKey"] == df["PrevLocKey"]) & (df["Level"] != df.get("Prev_Level"))
    ).astype(int)

    # Feature: Top 100 Products
    top_100_products = df["ProductID"].value_counts().head(100).index
    df["top_100_product"] = df["ProductID"].isin(top_100_products).astype(int)

    # Feature: Define efficient user as those with average pick time in top 50% and total picks in top 50%
    worker_stats = df.groupby("UserID")["Time_Delta_sec"].agg(["mean", "count"])
    worker_stats["mean"] = worker_stats["mean"].rank(pct=True)
    worker_stats["count"] = worker_stats["count"].rank(pct=True)
    df = df.merge(worker_stats, on="UserID", how="left")
#    df["efficient_user"] = ((df["mean"] <= 0.5) & (df["count"] <= 0.5)).astype(int)

    # Final feature lists
    feature_cols = [
        "Travel_Distance",
        "Weight",
        "Cube",
        "Quantity",
        "Aisle_group",
        "Level_group",
        "time_of_day",
        "same_aisle",
        "same_lockey",
        "diff_level",
        "UOM_group",
        "day_of_week",
        "top_100_product",
        "efficient_user",
    ]

    cat_cols = [
        "Aisle_group",
        "Level_group",
        "time_of_day",
        "same_aisle",
        "same_lockey",
        "diff_level",
        "UOM_group",
        "day_of_week",
        "top_100_product",
        "efficient_user",
    ]

    return df, feature_cols, cat_cols
