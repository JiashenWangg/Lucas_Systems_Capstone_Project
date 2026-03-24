import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path


# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------
# CLI ARGUMENTS
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Process warehouse data")
    parser.add_argument("data_path", help="Path to root data folder")
    parser.add_argument("wh_name", help="Warehouse name")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help="Quantile threshold for time delta filtering (default=0.98)"
    )
    return parser.parse_args()


# -----------------------------
# VALIDATION
# -----------------------------
def validate_inputs(base_path: Path, wh_name: str):
    csv_path = base_path / "database_backups_csv" / wh_name

    required_files = [
        csv_path / f"{wh_name}_Activity.csv",
        csv_path / f"{wh_name}_Locations.csv",
        csv_path / f"{wh_name}.csv",
    ]

    missing = [str(p) for p in required_files if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )


def validate_outputs(df: pd.DataFrame):
    """Basic sanity checks on final dataset."""
    if df.empty:
        raise ValueError("Final dataset is empty")


# -----------------------------
# HELPERS
# -----------------------------
def to_int(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")


def to_str(df, cols):
    for c in cols:
        df[c] = df[c].astype(str)


def zfill_str(series, width=2):
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(width)
    )


# -----------------------------
# LOAD DATA
# -----------------------------
def load_tables(base_path, wh_name, column_names):
    csv_path = base_path / "database_backups_csv" / wh_name

    tables = {
        f"{wh_name}_Activity": csv_path / f"{wh_name}_Activity.csv",
        f"{wh_name}_Locations": csv_path / f"{wh_name}_Locations.csv",
        f"{wh_name}_Products": csv_path / f"{wh_name}.csv",
    }

    dfs = {}
    for name, fp in tables.items():
        logger.info(f"Loading {fp}")
        dfs[name] = pd.read_csv(fp, header=0, names=column_names[name])

    return dfs


# -----------------------------
# CLEAN DATA
# -----------------------------
def clean_data(dfs, wh_name):
    activity = dfs[f"{wh_name}_Activity"].copy()
    locations = dfs[f"{wh_name}_Locations"].copy()
    products = dfs[f"{wh_name}_Products"].copy()

    # Validate required columns exist
    required_cols = ["ProductID", "LocationID", "Timestamp", "WorkCode"]
    missing = [c for c in required_cols if c not in activity.columns]
    if missing:
        raise KeyError(f"Missing required columns in Activity: {missing}")

    # Activity
    to_int(activity, ["ProductID", "Quantity", "LocationID"])
    activity["Timestamp"] = pd.to_datetime(activity["Timestamp"],
                                           errors="coerce")
    to_str(activity, ["UserID", "WorkCode", "AssignmentID"])
    activity = activity.dropna(subset=["Timestamp"]).copy()

    # Locations
    to_int(locations, ["LocationID", "Bay", "Level", "Slot"])

    # Products
    to_int(products, ["ProductID"])
    products = products[["ProductID", "ProductCode", "UnitOfMeasure",
                         "Weight", "Cube"]]

    if activity.empty:
        raise ValueError("Activity data is empty after cleaning (check input)")

    return activity, locations, products


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def compute_time_deltas(activity: pd.DataFrame, threshold_q: float) -> pd.DataFrame:
    df = activity.sort_values(["UserID", "Timestamp"]).reset_index(drop=True)

    g = df.groupby("UserID", sort=False)
    df["Prev_Timestamp"] = g["Timestamp"].shift(1)
    df["Prev_LocationID"] = g["LocationID"].shift(1)

    df["Time_Delta_sec"] = (df["Timestamp"] -
                            df["Prev_Timestamp"]).dt.total_seconds()

    threshold = df["Time_Delta_sec"].quantile(threshold_q)
    logger.info(f"{threshold_q:.2%} percentile threshold: {threshold:.2f}s")

    df.loc[df["Time_Delta_sec"] > threshold, "Time_Delta_sec"] = np.nan

    return df


def join_data(activity, products, locations):
    df = activity.merge(products, on="ProductID", how="left")
    df = df.merge(locations, on="LocationID", how="left")

    prev_loc = locations.rename(columns={
        "LocationID": "Prev_LocationID",
        "Aisle": "Prev_Aisle",
        "Bay": "Prev_Bay",
        "Level": "Prev_Level",
        "Slot": "Prev_Slot",
    })

    df = df.merge(prev_loc, on="Prev_LocationID", how="left")

    return df


# -----------------------------
# EXPORT
# -----------------------------
def export_outputs(base_path, wh_name, df_detailed, df_activity, df_joined):
    output_dir = base_path / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_detailed.to_parquet(output_dir / f"{wh_name}_detailed.parquet",
                           index=False)
    df_activity.to_parquet(output_dir / f"{wh_name}_activity_prepped.parquet",
                           index=False)
    df_joined.to_parquet(output_dir / f"{wh_name}_joined.parquet",
                         index=False)

    logger.info(f"Exported files to {output_dir}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()

    base_path = Path(args.data_path)
    wh_name = args.wh_name.lower()
    threshold_q = args.threshold

    validate_inputs(base_path, wh_name)

    column_names = {
        f"{wh_name}_Activity": ["ActivityCode", "UserID", "WorkCode",
                                       "AssignmentID", "ProductID", "Quantity",
                                       "Timestamp", "LocationID"],
        f"{wh_name}_Locations": ["LocationID", "Aisle", "Bay", "Level",
                                        "Slot"],
        f"{wh_name}_Products": ["ProductID", "ProductCode",
                                       "UnitOfMeasure", "Weight", "Cube",
                                       "Length", "Width", "Height"],
    }

    dfs = load_tables(base_path, wh_name, column_names)

    activity, locations, products = clean_data(dfs, wh_name)

    activity_prepped = compute_time_deltas(activity, threshold_q)
    df_joined = join_data(activity_prepped, products, locations)

    # Check join quality
    missing_products = df_joined["ProductCode"].isna().mean()
    missing_locations = df_joined["Aisle"].isna().mean()

    if missing_products > 0.1:
        logger.warning(f"{missing_products:.1%} of rows missing product info")

    if missing_locations > 0.1:
        logger.warning(f"{missing_locations:.1%} of rows missing location info")

    df_detailed = df_joined.copy()

    validate_outputs(df_detailed)

    export_outputs(base_path, wh_name, df_detailed,
                   activity_prepped, df_joined)


if __name__ == "__main__":
    main()


###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

# Preprocessing script including distances

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path


# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------
# CLI ARGUMENTS
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Process warehouse data")
    parser.add_argument("data_path", help="Path to root data folder")
    parser.add_argument("wh_name", help="Warehouse name")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help="Quantile threshold for time delta filtering (default=0.98)"
    )
    return parser.parse_args()


# -----------------------------
# VALIDATION
# -----------------------------
def validate_inputs(base_path: Path, wh_name: str):
    csv_path = base_path / "database_backups_csv" / wh_name

    required_files = [
        csv_path / f"{wh_name}_Activity.csv",
        csv_path / f"{wh_name}_Locations.csv",
        csv_path / f"{wh_name}.csv",
        base_path / "distance_matrices" / f"distance_matrix_{wh_name}.csv"
    ]

    missing = [str(p) for p in required_files if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )


def validate_outputs(df: pd.DataFrame):
    """Basic sanity checks on final dataset."""
    missing_ratio = df["Travel_Distance"].isna().mean()

    if missing_ratio > 0.2:
        raise ValueError(
            f"High missing Travel_Distance: {missing_ratio:.2%}. "
            "Check distance matrix or location keys."
        )


# -----------------------------
# HELPERS
# -----------------------------
def to_int(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")


def to_str(df, cols):
    for c in cols:
        df[c] = df[c].astype(str)


def zfill_str(series, width=2):
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(width)
    )


# -----------------------------
# LOAD DATA
# -----------------------------
def load_tables(base_path, wh_name, column_names):
    csv_path = base_path / "database_backups_csv" / wh_name

    tables = {
        f"{wh_name}_Activity": csv_path / f"{wh_name}_Activity.csv",
        f"{wh_name}_Locations": csv_path / f"{wh_name}_Locations.csv",
        f"{wh_name}_Products": csv_path / f"{wh_name}.csv",
    }

    dfs = {}
    for name, fp in tables.items():
        logger.info(f"Loading {fp}")
        dfs[name] = pd.read_csv(fp, header=0, names=column_names[name])

    return dfs


def load_distance_matrix(base_path, wh_name):
    path = base_path / "distance_matrices" / f"distance_matrix_{wh_name}.csv"
    logger.info(f"Loading distance matrix: {path}")

    dist = pd.read_csv(path, index_col=0)

    for c in dist.columns:
        dist[c] = pd.to_numeric(dist[c], errors="coerce")

    return (
        dist.stack()
        .rename("distance")
        .reset_index()
        .rename(columns={"level_0": "FromLoc", "level_1": "ToLoc"})
    )


# -----------------------------
# CLEAN DATA
# -----------------------------
def clean_data(dfs, wh_name):
    activity = dfs[f"{wh_name}_Activity"].copy()
    locations = dfs[f"{wh_name}_Locations"].copy()
    products = dfs[f"{wh_name}_Products"].copy()

    # Validate required columns exist
    required_cols = ["ProductID", "LocationID", "Timestamp", "WorkCode"]
    missing = [c for c in required_cols if c not in activity.columns]
    if missing:
        raise KeyError(f"Missing required columns in Activity: {missing}")

    # Activity
    to_int(activity, ["ProductID", "Quantity", "LocationID"])
    activity["Timestamp"] = pd.to_datetime(activity["Timestamp"], errors="coerce")
    to_str(activity, ["UserID", "WorkCode", "AssignmentID"])
    activity = activity.dropna(subset=["Timestamp"]).copy()

    # Locations
    to_int(locations, ["LocationID", "Bay", "Level", "Slot"])

    # Products
    to_int(products, ["ProductID"])
    products = products[["ProductID", "ProductCode", "UnitOfMeasure", "Weight", "Cube"]]


    if activity.empty:
        raise ValueError("Activity dataset is empty after cleaning (check input data)")

    return activity, locations, products



# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def compute_time_deltas(activity: pd.DataFrame, threshold_q: float) -> pd.DataFrame:
    df = activity.sort_values(["UserID", "Timestamp"]).reset_index(drop=True)

    g = df.groupby("UserID", sort=False)
    df["Prev_Timestamp"] = g["Timestamp"].shift(1)
    df["Prev_LocationID"] = g["LocationID"].shift(1)

    df["Time_Delta_sec"] = (df["Timestamp"] - df["Prev_Timestamp"]).dt.total_seconds()

    threshold = df["Time_Delta_sec"].quantile(threshold_q)
    logger.info(f"{threshold_q:.2%} percentile threshold: {threshold:.2f}s")

    df.loc[df["Time_Delta_sec"] > threshold, "Time_Delta_sec"] = np.nan

    return df


def join_data(activity, products, locations):
    df = activity.merge(products, on="ProductID", how="left")
    df = df.merge(locations, on="LocationID", how="left")

    prev_loc = locations.rename(columns={
        "LocationID": "Prev_LocationID",
        "Aisle": "Prev_Aisle",
        "Bay": "Prev_Bay",
        "Level": "Prev_Level",
        "Slot": "Prev_Slot",
    })

    df = df.merge(prev_loc, on="Prev_LocationID", how="left")

    return df


def compute_distances(df, dist_long):
    """
    Adds travel distance between consecutive locations
    using precomputed distance matrix.
    """
    df = df.copy()

    df["Aisle2"] = zfill_str(df["Aisle"])
    df["Bay2"] = zfill_str(df["Bay"])
    df["Prev_Aisle2"] = zfill_str(df["Prev_Aisle"])
    df["Prev_Bay2"] = zfill_str(df["Prev_Bay"])

    df["LocKey"] = df["Aisle2"] + "|" + df["Bay2"] + "|||"
    df["PrevLocKey"] = df["Prev_Aisle2"] + "|" + df["Prev_Bay2"] + "|||"

    df = df.merge(
        dist_long,
        left_on=["LocKey", "PrevLocKey"],
        right_on=["FromLoc", "ToLoc"],
        how="left"
    )

    return df.rename(columns={"distance": "Travel_Distance"}).drop(columns=["FromLoc", "ToLoc"])


# -----------------------------
# EXPORT
# -----------------------------
def export_outputs(base_path, wh_name, df_detailed, df_activity, df_joined):
    output_dir = base_path / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_detailed.to_parquet(output_dir / f"{wh_name}_detailed.parquet", index=False)
    df_activity.to_parquet(output_dir / f"{wh_name}_activity_prepped.parquet", index=False)
    df_joined.to_parquet(output_dir / f"{wh_name}_joined.parquet", index=False)

    logger.info(f"Exported files to {output_dir}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()

    base_path = Path(args.data_path)
    wh_name = args.wh_name.lower()
    threshold_q = args.threshold

    validate_inputs(base_path, wh_name)

    column_names = {
        f"{wh_name}_Activity": ["ActivityCode", "UserID", "WorkCode",
                                       "AssignmentID", "ProductID", "Quantity",
                                       "Timestamp", "LocationID"],
        f"{wh_name}_Locations": ["LocationID", "Aisle", "Bay", "Level",
                                        "Slot"],
        f"{wh_name}_Products": ["ProductID", "ProductCode",
                                       "UnitOfMeasure", "Weight", "Cube",
                                       "Length", "Width", "Height"],
    }

    dfs = load_tables(base_path, wh_name, column_names)
    dist_long = load_distance_matrix(base_path, wh_name)

    activity, locations, products = clean_data(dfs, wh_name)

    activity_prepped = compute_time_deltas(activity, threshold_q)
    df_joined = join_data(activity_prepped, products, locations)

    # Check join quality
    missing_products = df_joined["ProductCode"].isna().mean()
    missing_locations = df_joined["Aisle"].isna().mean()

    if missing_products > 0.1:
        logger.warning(f"{missing_products:.1%} of rows missing product info")

    if missing_locations > 0.1:
        logger.warning(f"{missing_locations:.1%} of rows missing location info")

    df_detailed = compute_distances(df_joined, dist_long)

    validate_outputs(df_detailed)

    export_outputs(base_path, wh_name, df_detailed, activity_prepped, df_joined)


if __name__ == "__main__":
    main()

