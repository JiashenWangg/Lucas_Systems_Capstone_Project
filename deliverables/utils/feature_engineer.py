"""
utils/feature_engineer.py
--------------------------
Core feature engineering for all warehouse pick-time models.

All functions take a cleaned DataFrame (post-preprocess) and return
feature matrices or enriched DataFrames. Encodings (top aisles, UoMs,
product tiers) are always computed on training data and passed in as
arguments for update and predict runs — never recomputed from new data.

Public API
----------
compute_encodings(df)
    Compute top_aisles, top_uoms, product_tiers from training data.
    Call once at training time; save results to meta.pkl.

apply_features(df, top_aisles, top_uoms, product_tiers, sequenced=False)
    Apply feature engineering using saved encodings.
    Use for both training (after compute_encodings) and inference.

make_X(df, train_columns=None)
    Build one-hot feature matrix from an engineered DataFrame.
    Pass train_columns to align update/predict matrices to training shape.

compute_prev_location(df)
    When sequenced=True: compute Prev_LocationID by shifting LocationID
    within each user's sequence, sorted by Timestamp.
"""

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_COLS_BASE = [
    "Weight",
    "Cube",
    "Quantity",
    "aisle",
    "level",
    "UoM",
    "t25_products",
    "t25_50_products",
    "t50_75_products",
    "other_products",
]

FEATURE_COLS_SEQ = FEATURE_COLS_BASE + [
    "same_aisle",
    "same_level",
    "Travel_Distance",
]

CAT_COLS_BASE = [
    "aisle",
    "level",
    "UoM",
    "t25_products",
    "t25_50_products",
    "t50_75_products",
    "other_products",
]

CAT_COLS_SEQ = CAT_COLS_BASE + [
    "same_aisle",
    "same_level",
]


# ── Encoding computation (training time only) ─────────────────────────────────

def compute_encodings(df):
    """
    Compute categorical encodings from training data.
    Returns a dict that should be saved to meta.pkl and passed to
    apply_features() on all future update and predict runs.

    Args:
        df: cleaned training DataFrame with Aisle, UnitOfMeasure, ProductID

    Returns:
        dict with keys: top_aisles, top_uoms, product_tiers
            product_tiers: tuple of (t25, t25_50, t50_75, other) index sets
    """
    top_aisles = df["Aisle"].value_counts().head(5).index

    top_uoms = df["UnitOfMeasure"].value_counts().head(5).index

    product_counts = df["ProductID"].value_counts()
    cumsum = product_counts.cumsum()
    total  = product_counts.sum()

    t25    = product_counts[cumsum <= 0.25 * total].index
    t25_50 = product_counts[
        (cumsum > 0.25 * total) & (cumsum <= 0.50 * total)
    ].index
    t50_75 = product_counts[
        (cumsum > 0.50 * total) & (cumsum <= 0.75 * total)
    ].index
    other  = product_counts[cumsum > 0.75 * total].index

    return {
        "top_aisles":    top_aisles,
        "top_uoms":      top_uoms,
        "product_tiers": (t25, t25_50, t50_75, other),
    }


# ── Previous location (sequenced mode) ───────────────────────────────────────

def compute_prev_location(df):
    """
    Compute Prev_LocationID by shifting LocationID within each user's
    sequence sorted by Timestamp. Called when sequenced=True.

    Args:
        df: DataFrame with UserID, LocationID, Timestamp columns

    Returns:
        df with Prev_LocationID column added (NaN for first pick per user)
    """
    df = df.copy()
    df = df.sort_values(["UserID", "Timestamp"]).reset_index(drop=True)
    df["Prev_LocationID"] = (
        df.groupby("UserID")["LocationID"].shift(1)
    )
    return df


# ── Feature application (training, update, and predict) ──────────────────────

def _level_group(val):
    """Bucket Level into 0/1/2/3/4/5+ groups."""
    try:
        v = int(val)
        return "5+" if v >= 5 else str(v)
    except Exception:
        return str(val)


def apply_features(df, top_aisles, top_uoms, product_tiers,
                   sequenced=False, locations_df=None, distance_df=None):
    """
    Apply feature engineering using encodings computed at training time.

    Args:
        df:             Cleaned DataFrame (post-preprocess or post-load_new_data)
        top_aisles:     Index of top-5 aisles from training encodings
        top_uoms:       Index of top-5 UoMs from training encodings
        product_tiers:  Tuple (t25, t25_50, t50_75, other) from training encodings
        sequenced:      If True, compute sequence-dependent features.
                        Requires Prev_LocationID to already be in df
                        (call compute_prev_location first).
        locations_df:   Locations table (needed when sequenced=True to get
                        Prev_Aisle and Prev_Level from Prev_LocationID)
        distance_df:    Long-format distance DataFrame with columns
                        [FromLoc, ToLoc, distance] (optional, sequenced only)

    Returns:
        df with all feature columns added in place
    """
    df = df.copy()

    t25, t25_50, t50_75, other = product_tiers

    # ── Aisle ─────────────────────────────────────────────────────────────────
    df["Aisle"] = pd.to_numeric(df["Aisle"], errors="coerce").fillna(-1).astype(int)
    df["aisle"] = df["Aisle"].apply(
        lambda a: str(a) if a in top_aisles else "other"
    )

    # ── Level ─────────────────────────────────────────────────────────────────
    df["level"] = df["Level"].apply(_level_group)

    # ── UoM ───────────────────────────────────────────────────────────────────
    df["UoM"] = df["UnitOfMeasure"].apply(
        lambda u: u if u in top_uoms else "other"
    )

    # ── Product tiers ─────────────────────────────────────────────────────────
    df["t25_products"]    = df["ProductID"].apply(lambda p: "1" if p in t25    else "0")
    df["t25_50_products"] = df["ProductID"].apply(lambda p: "1" if p in t25_50 else "0")
    df["t50_75_products"] = df["ProductID"].apply(lambda p: "1" if p in t50_75 else "0")
    df["other_products"]  = df["ProductID"].apply(lambda p: "1" if p in other  else "0")

    # ── Sequence features (only when sequenced=True) ──────────────────────────
    if sequenced:
        
        if "Prev_LocationID" not in df.columns:
            raise ValueError(
                "sequenced=True but Prev_LocationID not in df. "
                "Call compute_prev_location(df) before apply_features()."
            )

        if locations_df is None:
            raise ValueError(
                "sequenced=True requires locations_df to look up "
                "Prev_Aisle and Prev_Level from Prev_LocationID."
            )

        # Join previous location attributes
        prev_cols = locations_df[["LocationID", "Aisle", "Level"]].rename(columns={
            "LocationID": "Prev_LocationID",
            "Aisle":      "Prev_Aisle",
            "Level":      "Prev_Level",
        })
        df = df.merge(prev_cols, on="Prev_LocationID", how="left")

        df["Prev_Aisle"] = pd.to_numeric(
            df["Prev_Aisle"], errors="coerce"
        ).fillna(-1).astype(int)

        df["same_aisle"] = (df["Aisle"] == df["Prev_Aisle"]).astype(int).fillna(0)
        df["same_level"] = (df["Level"] == df["Prev_Level"]).astype(int).fillna(0)

        # Travel distance (optional — only when distance matrix is available)
        if distance_df is not None:
            df = df.merge(
                distance_df.rename(columns={
                    "FromLoc": "Prev_LocationID",
                    "ToLoc":   "LocationID",
                    "distance": "Travel_Distance",
                }),
                on=["Prev_LocationID", "LocationID"],
                how="left",
            )
            df["Travel_Distance"] = df["Travel_Distance"].fillna(0.0)


        
        else:
            df["Travel_Distance"] = 0.0


    
    return df

# ── Feature matrix builder ───────────────────────────────────────────────────

def make_X(df, sequenced=False, train_columns=None):
    """
    Build one-hot encoded feature matrix from an engineered DataFrame.

    Args:
        df:             DataFrame after apply_features()
        sequenced:      Whether to include sequence-dependent feature columns
        train_columns:  If provided, reindex output to match training column
                        order exactly. Required for update and predict runs.

    Returns:
        DataFrame of float features ready to pass to xgb.DMatrix
    """
    feature_cols = FEATURE_COLS_SEQ if sequenced else FEATURE_COLS_BASE
    cat_cols     = CAT_COLS_SEQ     if sequenced else CAT_COLS_BASE

    # Keep only columns that exist (distance may be absent even in sequenced mode)
    feature_cols = [c for c in feature_cols if c in df.columns]
    cat_cols     = [c for c in cat_cols     if c in df.columns]

    X = pd.get_dummies(df[feature_cols], columns=cat_cols, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

    if train_columns is not None:
        X = X.reindex(columns=train_columns, fill_value=0)

    return X
