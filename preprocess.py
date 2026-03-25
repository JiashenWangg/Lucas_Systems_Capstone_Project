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
    f"{warehouse_name}_Activity": ["ActivityCode", "UserID", "WorkCode",
                                   "AssignmentID", "ProductID", "Quantity",
                                   "Timestamp", "LocationID"],
    f"{warehouse_name}_Locations": ["LocationID", "Aisle", "Bay", "Level",
                                    "Slot"],
    f"{warehouse_name}_Products": ["ProductID", "ProductCode","UnitOfMeasure",
                                   "Weight", "Cube", "Length", "Width",
                                   "Height"],
}

# Load tables into dictionary
dfs = {}
for name, fp in tables.items():
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


###############################################################################


activity_key = f"{warehouse_name}_Activity"
locations_key = f"{warehouse_name}_Locations"
products_key = f"{warehouse_name}_Products"

# Clean activity data for merging
dfs[activity_key]["ProductID"] = pd.to_numeric(dfs[activity_key]["ProductID"],
                                               errors="coerce").astype("Int64")
dfs[activity_key]["Quantity"]  = pd.to_numeric(dfs[activity_key]["Quantity"],
                                               errors="coerce").astype("Int64")
dfs[activity_key]["LocationID"] = pd.to_numeric(dfs[activity_key]["LocationID"],
                                                errors="coerce").astype("Int64")
dfs[activity_key]["Timestamp"] = pd.to_datetime(dfs[activity_key]["Timestamp"],
                                                errors="coerce")
dfs[activity_key]["UserID"] = dfs[activity_key]["UserID"].astype(str)
dfs[activity_key]["WorkCode"] = dfs[activity_key]["WorkCode"].astype(str)
dfs[activity_key]["AssignmentID"] = dfs[activity_key]["AssignmentID"].astype(str)
dfs[activity_key] = dfs[activity_key].dropna(subset=["Timestamp"]).copy()

# Clean location data for merging
dfs[locations_key]["LocationID"] = pd.to_numeric(dfs[locations_key]["LocationID"], errors="coerce").astype("Int64")
for col in ["Bay", "Level", "Slot"]:
    dfs[locations_key][col] = pd.to_numeric(dfs[locations_key][col],
                                            errors="coerce").astype("Int64")

# Clean product data for merging
dfs[products_key]["ProductID"] = pd.to_numeric(dfs[products_key]["ProductID"],
                                               errors="coerce").astype("Int64")
dfs[products_key] = dfs[products_key][["ProductID", "ProductCode", 
                                       "UnitOfMeasure", "Weight", "Cube"]]


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

# Filter out gaps exceeding the 98th percentile
# Interpreting these as breaks, chats, etc.
threshold = df_work["Time_Delta_sec"].quantile(0.98)
print(f"98th Percentile Threshold: {threshold:.2f} seconds ({(threshold/60):.2f} minutes)")
df_work.loc[df_work["Time_Delta_sec"] > threshold, "Time_Delta_sec"] = np.nan
print(f"Number of rows with Time Deltas: {len(df_work[df_work['Time_Delta_sec'].notnull()])}")

Activity_prepped = df_work


###############################################################################


# Join Activity data with Product and Location features
df_joined = Activity_prepped.merge(Products, on="ProductID", how="left")
df_joined = df_joined.merge(Locations, on="LocationID", how="left")

df_joined = df_joined.merge(
    Locations[["LocationID", "Aisle", "Bay", "Level", "Slot"]].rename(columns={
        "LocationID": "Prev_LocationID",
        "Aisle": "Prev_Aisle",
        "Bay": "Prev_Bay",
        "Level": "Prev_Level",
        "Slot": "Prev_Slot",
    }),
    on="Prev_LocationID",
    how="left"
)


###############################################################################


df_detailed = df_joined.copy()

df_detailed["Aisle2"] = pd.to_numeric(df_detailed["Aisle"],
                                      errors="coerce").astype("Int64").astype(str).str.zfill(2)
df_detailed["Bay2"] = pd.to_numeric(df_detailed["Bay"],
                                    errors="coerce").astype("Int64").astype(str).str.zfill(2)
df_detailed["Prev_Aisle2"] = pd.to_numeric(df_detailed["Prev_Aisle"],
                                           errors="coerce").astype("Int64").astype(str).str.zfill(2)
df_detailed["Prev_Bay2"] = pd.to_numeric(df_detailed["Prev_Bay"],
                                         errors="coerce").astype("Int64").astype(str).str.zfill(2)

# Create location keys and map travel distances
df_detailed["LocKey"] = df_detailed["Aisle2"] + "|" + df_detailed["Bay2"] + "|||"
df_detailed["PrevLocKey"] = df_detailed["Prev_Aisle2"] + "|" + df_detailed["Prev_Bay2"] + "|||"

# Calculate travel distances using the distance matrix (long format)
df_detailed = df_detailed.merge(
    dist_long,
    left_on=["LocKey", "PrevLocKey"],
    right_on=["FromLoc", "ToLoc"],
    how="left"
).rename(columns={"distance": "Travel_Distance"}).drop(columns=["FromLoc", "ToLoc"])


# -----------------------------
# EXPORT
# -----------------------------
def export_outputs(base_path, wh_name, df_detailed, df_activity, df_joined):
    output_dir = base_path / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

df_detailed.to_parquet(output_dir / f"{warehouse_name}_detailed.parquet",
                       index=False)
Activity_prepped.to_parquet(output_dir / f"{warehouse_name}_activity_prepped.parquet",
                            index=False)
df_joined.to_parquet(output_dir / f"{warehouse_name}_joined.parquet",
                     index=False)

print(f"Successfully exported all {warehouse_name} files to {output_dir}")
