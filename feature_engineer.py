"""
This module contains the function get_engineered_df which loads the data,
applies preprocessing and feature engineering steps, and returns the processed
DataFrame along with lists of feature columns and categorical columns for modeling.

Instructions:
- Place this file in the same directory as your main modeling script (e.g., xgboost.ipynb)
- Import the function using:
    from feature_engineer import get_engineered_df
    from feature_engineer import get_engineered_df_allWC
- Example call:
    df, features, cat_cols = get_engineered_df("data/OE_30.parquet", warehouse="OE", max_time=300, work_code="30")
    df, features_allWC, cat_cols_allWC = get_engineered_df_allWC("data/OE_allWC.parquet", warehouse="OE", max_time=300)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_engineered_df(file_path, warehouse="OE", max_time=300, work_code="30"):
    """
    Loads data and applies preprocessing/feature engineering
    Args:
    - warehouse: string indicating the warehouse
    - file_path: path to the parquet file containing the data
    - max_time: maximum Time_Delta_sec to consider for filtering (default 300s)
    - work_code: WorkCode to filter on
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
    df["WorkCode"] = df["WorkCode"].apply(
        lambda x: x.split(".")[0] if isinstance(x, str) else x
    )

    df = df.dropna(subset=["Timestamp"]).copy()
    df = df[
        (df["Time_Delta_sec"] < max_time)
        & (df["Travel_Distance"] >= 0)
        & (df["WorkCode"] == work_code)
    ].copy()

    # Feature: Aisle Grouping, top-5 encoding
    top_aisles = df["Aisle"].value_counts().head(5).index
    df["Aisle"] = pd.to_numeric(df["Aisle"], errors="coerce").fillna(-1).astype(int)
    df["aisle"] = df["Aisle"].apply(lambda a: str(a) if a in top_aisles else "other")

    # Feature: Level Grouping
    def level_group(l):
        try:
            val = int(l)
            return "5+" if val >= 5 else str(val)
        except:
            return str(l)

    df["level"] = df["Level"].apply(level_group)

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
    top_uoms = df["UnitOfMeasure"].value_counts().head(5).index
    df["UoM"] = df["UnitOfMeasure"].apply(lambda u: u if u in top_uoms else "other")

    # Feature: Day of Week
    df["day_of_week"] = df["Timestamp"].dt.day_name()

    # Feature: Relationship with Previous Row
    df["Prev_Aisle"] = (
        pd.to_numeric(df["Prev_Aisle"], errors="coerce").fillna(-1).astype(int)
    )
    df["same_aisle"] = (df["Aisle"] == df["Prev_Aisle"]).astype(int)
    df["same_lockey"] = (df["LocKey"] == df["PrevLocKey"]).astype(int)
    df["same_level"] = (
        (df["LocKey"] == df["PrevLocKey"]) & (df["Level"] == df.get("Prev_Level"))
    ).astype(int)

    # Feature: Top 100 Products
    top_100_products = df["ProductID"].value_counts().head(100).index
    df["top_100_product"] = df["ProductID"].isin(top_100_products).astype(int)

    # Final feature lists
    feature_cols = [
        "Travel_Distance",
        "Weight",
        "Cube",
        "Quantity",
        "aisle",
        "level",
        "time_of_day",
        "same_aisle",
        "same_lockey",
        "same_level",
        "UoM",
        "day_of_week",
        "top_100_product",
    ]

    cat_cols = [
        "aisle",
        "level",
        "time_of_day",
        "same_aisle",
        "same_lockey",
        "same_level",
        "UoM",
        "day_of_week",
        "top_100_product",
    ]

    return df, feature_cols, cat_cols


def get_engineered_df_allWC(file_path, warehouse="OE", max_time=300):
    """
    Loads data and applies preprocessing/feature engineering
    Args:
    - warehouse: string indicating the warehouse
    - file_path: path to the parquet file containing the data
    - max_time: maximum Time_Delta_sec to consider for filtering (default 300s)
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
    df["WorkCode"] = df["WorkCode"].apply(
        lambda x: x.split(".")[0] if isinstance(x, str) else x
    )

    df = df.dropna(subset=["Timestamp"]).copy()
    df = df[(df["Time_Delta_sec"] < max_time) & (df["Travel_Distance"] >= 0)].copy()

    # Feature: Aisle Grouping, top-5 encoding
    top_aisles = df["Aisle"].value_counts().head(5).index
    df["Aisle"] = pd.to_numeric(df["Aisle"], errors="coerce").fillna(-1).astype(int)
    df["aisle"] = df["Aisle"].apply(lambda a: str(a) if a in top_aisles else "other")

    # Feature: Level Grouping
    def level_group(l):
        try:
            val = int(l)
            return "5+" if val >= 5 else str(val)
        except:
            return str(l)

    df["level"] = df["Level"].apply(level_group)

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

    # Feature: UOM Grouping (top-5 Encoding)
    top_uoms = df["UnitOfMeasure"].value_counts().head(5).index
    df["UoM"] = df["UnitOfMeasure"].apply(lambda u: u if u in top_uoms else "other")

    # Feature: Day of Week
    df["day_of_week"] = df["Timestamp"].dt.day_name()

    # Feature: Relationship with Previous Row
    df["Prev_Aisle"] = (
        pd.to_numeric(df["Prev_Aisle"], errors="coerce").fillna(-1).astype(int)
    )
    df["same_aisle"] = (df["Aisle"] == df["Prev_Aisle"]).astype(int)
    df["same_lockey"] = (df["LocKey"] == df["PrevLocKey"]).astype(int)
    df["same_level"] = (
        (df["LocKey"] == df["PrevLocKey"]) & (df["Level"] == df.get("Prev_Level"))
    ).astype(int)

    # Feature: Top 100 Products
    top_100_products = df["ProductID"].value_counts().head(100).index
    df["top_100_product"] = df["ProductID"].isin(top_100_products).astype(int)

    # Final feature lists
    feature_cols = [
        "WorkCode",
        "Travel_Distance",
        "Weight",
        "Cube",
        "Quantity",
        "aisle",
        "level",
        "time_of_day",
        "same_aisle",
        "same_lockey",
        "same_level",
        "UoM",
        "day_of_week",
        "top_100_product",
    ]

    cat_cols = [
        "WorkCode",
        "aisle",
        "level",
        "time_of_day",
        "same_aisle",
        "same_lockey",
        "same_level",
        "UoM",
        "day_of_week",
        "top_100_product",
    ]

    return df, feature_cols, cat_cols
