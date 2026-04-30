"""
Split OE demo data for the knowledge transfer workflow.

This creates three demo inputs:
  1. Historical training data: first N activity dates
  2. Incremental update data: last activity date
  3. Prediction queue: a continuous 50-row slice from the last day for one
     user and one WorkCode

Default source:
  deliverables/training_data/OE/OE_Activity.csv

Default outputs:
  deliverables/KT_demo/demo_data/training_data/OE/OE_Activity.csv
  deliverables/KT_demo/demo_data/incremental/OE/OE_activity_lastday.csv
  deliverables/KT_demo/demo_data/predict/OE/OE_predict_50.csv
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


ACTIVITY_COLS = [
    "ActivityCode",
    "UserID",
    "WorkCode",
    "AssignmentID",
    "ProductID",
    "Quantity",
    "Timestamp",
    "LocationID",
]

REFERENCE_FILES = [
    "{wh}_Locations.csv",
    "{wh}_Products.csv",
    "{wh}_Distance_Matrix.csv",
]

PREDICT_COLS = [
    "ActivityCode",
    "UserID",
    "WorkCode",
    "AssignmentID",
    "ProductID",
    "Quantity",
    "LocationID",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create historical, incremental, and prediction demo splits."
    )
    parser.add_argument("--warehouse", default="OE", help="Warehouse code.")
    parser.add_argument(
        "--source_dir",
        default="deliverables/training_data",
        help="Root directory containing WH/WH_Activity.csv and reference files.",
    )
    parser.add_argument(
        "--out_root",
        default="deliverables/KT_demo/demo_data",
        help="Root output directory for demo data.",
    )
    parser.add_argument(
        "--history_days",
        type=int,
        default=7,
        help="Number of earliest distinct activity dates for historical training.",
    )
    parser.add_argument(
        "--predict_rows",
        type=int,
        default=50,
        help="Number of continuous activities in the prediction queue.",
    )
    return parser.parse_args()


def clean_workcode(series):
    return series.astype(str).str.split(".", n=1).str[0]


def load_activity(activity_path):
    df = pd.read_csv(activity_path, header=0, names=ACTIVITY_COLS)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).copy()
    df["WorkCode"] = clean_workcode(df["WorkCode"])
    return df.sort_values(["Timestamp", "UserID"]).reset_index(drop=True)


def choose_prediction_slice(last_day_df, predict_rows):
    """
    Choose a deterministic continuous queue from the last day.

    The prediction scripts require exactly one WorkCode, so this uses the first
    user/WorkCode pair with at least predict_rows activities after chronological
    sorting. If none exists, it falls back to the largest available pair.
    """
    ordered = last_day_df.sort_values(["UserID", "WorkCode", "Timestamp"]).copy()
    groups = []

    for (user_id, workcode), group in ordered.groupby(["UserID", "WorkCode"], sort=True):
        groups.append((len(group), str(user_id), str(workcode), group))

    if not groups:
        raise ValueError("No rows available on the incremental day.")

    groups.sort(key=lambda item: (-item[0], item[1], item[2]))
    for n_rows, _user_id, _workcode, group in groups:
        if n_rows >= predict_rows:
            return group.head(predict_rows).copy()

    n_rows, user_id, workcode, group = groups[0]
    print(
        f"WARNING: no single user/WorkCode has {predict_rows} rows on the last day. "
        f"Using {n_rows} rows for UserID={user_id}, WorkCode={workcode}."
    )
    return group.copy()


def copy_reference_files(source_wh_dir, target_wh_dir, warehouse):
    target_wh_dir.mkdir(parents=True, exist_ok=True)
    for template in REFERENCE_FILES:
        name = template.format(wh=warehouse)
        src = source_wh_dir / name
        if src.exists():
            shutil.copy2(src, target_wh_dir / name)
        elif "Distance_Matrix" not in name:
            raise FileNotFoundError(f"Required reference file missing: {src}")


def main():
    args = parse_args()
    warehouse = args.warehouse.upper()
    source_wh_dir = Path(args.source_dir) / warehouse
    activity_path = source_wh_dir / f"{warehouse}_Activity.csv"

    out_root = Path(args.out_root)
    history_wh_dir = out_root / "training_data" / warehouse
    incremental_wh_dir = out_root / "incremental" / warehouse
    predict_wh_dir = out_root / "predict" / warehouse

    df = load_activity(activity_path)
    df["ActivityDate"] = df["Timestamp"].dt.date
    dates = sorted(df["ActivityDate"].unique())

    if len(dates) < args.history_days + 1:
        raise ValueError(
            f"Need at least {args.history_days + 1} distinct dates, found {len(dates)}."
        )

    history_dates = dates[: args.history_days]
    incremental_date = dates[-1]

    history_df = df[df["ActivityDate"].isin(history_dates)].drop(columns=["ActivityDate"])
    incremental_df = df[df["ActivityDate"] == incremental_date].drop(columns=["ActivityDate"])
    predict_df = choose_prediction_slice(incremental_df, args.predict_rows)

    history_wh_dir.mkdir(parents=True, exist_ok=True)
    incremental_wh_dir.mkdir(parents=True, exist_ok=True)
    predict_wh_dir.mkdir(parents=True, exist_ok=True)

    copy_reference_files(source_wh_dir, history_wh_dir, warehouse)

    history_path = history_wh_dir / f"{warehouse}_Activity.csv"
    incremental_path = incremental_wh_dir / f"{warehouse}_activity_lastday.csv"
    predict_path = predict_wh_dir / f"{warehouse}_predict_{len(predict_df)}.csv"

    history_df.to_csv(history_path, index=False)
    incremental_df.to_csv(incremental_path, index=False)
    predict_df[PREDICT_COLS].to_csv(predict_path, index=False)

    print("Demo split complete")
    print(f"  Historical dates: {history_dates[0]} to {history_dates[-1]}")
    print(f"  Historical rows:  {len(history_df):,} -> {history_path}")
    print(f"  Incremental date: {incremental_date}")
    print(f"  Incremental rows: {len(incremental_df):,} -> {incremental_path}")
    print(
        "  Predict queue:    "
        f"{len(predict_df):,} rows, UserID={predict_df['UserID'].iloc[0]}, "
        f"WorkCode={predict_df['WorkCode'].iloc[0]} -> {predict_path}"
    )


if __name__ == "__main__":
    main()
