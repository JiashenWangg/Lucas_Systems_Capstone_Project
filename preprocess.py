import sys
import pandas as pd
import numpy as np
from pathlib import Path

data_path = sys.argv[1]  # path to data folder, don't include final /
warehouse_name = sys.argv[2]
warehouse_name = warehouse_name.lower()

tables = {
    f"{warehouse_name}_Activity": f"{data_path}/database_backups_csv/{warehouse_name}/{warehouse_name}_Activity.csv",
    f"{warehouse_name}_Locations": f"{data_path}/database_backups_csv/{warehouse_name}/{warehouse_name}_Locations.csv",
    f"{warehouse_name}_Products": f"{data_path}/database_backups_csv/{warehouse_name}/{warehouse_name}.csv",
}

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

"""
Run this if there are no column names

for name, fp in tables.items():
    dfs[name] = pd.read_csv(fp, header=None, names=column_names[name])
"""

# Load distance matrix
distance_path = f"{data_path}/distance_matrices/distance_matrix_{warehouse_name}.csv"
Distance = pd.read_csv(distance_path, index_col=0)
for c in Distance.columns:
    Distance[c] = pd.to_numeric(Distance[c], errors="coerce")

# Convert Distance matrix to a long-format DataFrame for easier lookups
dist_long = (
    Distance.stack()
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

Activity = dfs[activity_key]
Locations = dfs[locations_key]
Products = dfs[products_key]


###############################################################################


df_work = Activity.copy()
df_work = df_work.sort_values(["UserID", "Timestamp"]).reset_index(drop=True)

# Calculate time deltas between consecutive user tasks
# Goal: to identify task completion time

g = df_work.groupby("UserID", sort=False)
df_work["Prev_Timestamp"] = g["Timestamp"].shift(1)
df_work["Prev_LocationID"] = g["LocationID"].shift(1)

df_work["Time_Delta_sec"] = (
    df_work["Timestamp"] - df_work["Prev_Timestamp"]
).dt.total_seconds()

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


###############################################################################


# Export processed files
output_dir = Path(f"{data_path}/processed")
output_dir.mkdir(parents=True, exist_ok=True)

df_detailed.to_parquet(output_dir / f"{warehouse_name}_detailed.parquet",
                       index=False)
Activity_prepped.to_parquet(output_dir / f"{warehouse_name}_activity_prepped.parquet",
                            index=False)
df_joined.to_parquet(output_dir / f"{warehouse_name}_joined.parquet",
                     index=False)

print(f"Successfully exported all {warehouse_name} files to {output_dir}")
