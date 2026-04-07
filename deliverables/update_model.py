"""
update_model.py
---------------
Incrementally update saved XGBoost models with new completed pick data.
Adds UPDATE_TREES new trees on top of existing models — no full retrain.
Also updates worker effects for any new workers that appear in the new data.

Run this daily after new picks are completed and preprocessed.

Usage:
    python update_model.py OE --new_data daily_activity.csv
    python update_model.py OE --new_data daily_activity.csv --data_path training_data --trees 50

Args:
    warehouse:   Warehouse code (OE, OF, RT)
    --new_data:  CSV of new completed picks — same format as training data,
                 must already have Time_Delta_sec computed (run preprocess.py first)
    --data_path: Root training_data directory (for reference tables)
    --trees:     Number of new trees to add per WorkCode (default: 100)
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from feature_engineer import engineer_features, make_X, compute_worker_levels


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_TIME = 300
DEFAULT_TREES = 100

COLUMN_NAMES = {
    "Locations": ["LocationID", "Aisle", "Bay", "Level", "Slot"],
    "Products":  ["ProductID", "ProductCode", "UnitOfMeasure",
                  "Weight", "Cube", "Length", "Width", "Height"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Incrementally update XGBoost models with new completed picks"
    )
    parser.add_argument("warehouse",   help="Warehouse code e.g. OE, OF, RT")
    parser.add_argument("--new_data",  required=True,
                        help="CSV of new completed picks with Time_Delta_sec")
    parser.add_argument("--data_path", default="training_data",
                        help="Root training_data directory (default: training_data)")
    parser.add_argument("--trees",     type=int, default=DEFAULT_TREES,
                        help=f"New trees to add per WorkCode (default: {DEFAULT_TREES})")
    return parser.parse_args()


def load_reference_tables(data_path, warehouse):
    wh_dir = Path(data_path) / warehouse.upper()
    locations = pd.read_csv(
        wh_dir / f"{warehouse.upper()}_Locations.csv",
        header=0, names=COLUMN_NAMES["Locations"]
    )
    products = pd.read_csv(
        wh_dir / f"{warehouse.upper()}_Products.csv",
        header=0, names=["ProductID", "ProductCode", "UnitOfMeasure",
                         "Weight", "Cube", "Length", "Width", "Height"]
    )
    for col in ["LocationID", "Bay", "Level", "Slot"]:
        locations[col] = pd.to_numeric(locations[col],
                                       errors="coerce").astype("Int64")
    products["ProductID"] = pd.to_numeric(products["ProductID"],
                                          errors="coerce").astype("Int64")
    products = products[["ProductID", "ProductCode", "UnitOfMeasure",
                         "Weight", "Cube"]]
    return locations, products


def load_new_data(new_data_csv, locations, products):
    """
    Load new completed picks CSV, join location and product attributes.
    Expects Time_Delta_sec to already be present — run preprocess.py first.
    """
    df = pd.read_csv(new_data_csv)

    # Clean types — same as preprocess.py
    for col in ["ProductID", "Quantity", "LocationID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["Timestamp"] = pd.to_datetime(df.get("Timestamp"), errors="coerce")
    df["UserID"]     = df["UserID"].astype(str)
    df["WorkCode"]   = df["WorkCode"].astype(str).apply(
        lambda x: x.split(".")[0] if isinstance(x, str) else x
    )
    df["Time_Delta_sec"] = pd.to_numeric(df.get("Time_Delta_sec"), errors="coerce")

    # Drop AssignmentOpen rows and outliers
    if "ActivityCode" in df.columns:
        df = df[df["ActivityCode"] != "AssignmentOpen"].copy()
    df = df[df["Time_Delta_sec"] < MAX_TIME].dropna(subset=["Time_Delta_sec"]).copy()

    if df.empty:
        raise ValueError("New data is empty after cleaning. Check Time_Delta_sec column.")

    # Join product and location attributes
    df = df.merge(products, on="ProductID", how="left")
    df = df.merge(locations, on="LocationID", how="left")

    return df


def estimate_worker_effects_new(df_new):
    """
    Estimate worker effects from new data only.
    Used to get effects for workers not seen in original training.
    """
    df_re = df_new[["UserID", "Time_Delta_sec"]].dropna().copy()
    if df_re["UserID"].nunique() < 2:
        return pd.DataFrame({"UserID": df_re["UserID"].unique(),
                             "worker_effect": 0.0})

    result = smf.mixedlm(
        "Time_Delta_sec ~ 1", data=df_re, groups=df_re["UserID"]
    ).fit(reml=True, disp=False)

    return pd.DataFrame({
        "UserID":        list(result.random_effects.keys()),
        "worker_effect": [float(v.iloc[0]) for v in result.random_effects.values()]
    })


def main():
    args = parse_args()
    warehouse = args.warehouse.upper()
    new_data = args.new_data
    data_path = args.data_path
    n_trees = args.trees

    # ── Load saved metadata ──────────────────────────────────────────────────
    models_dir = Path("models") / warehouse
    meta_file = models_dir / "meta.pkl"

    if not meta_file.exists():
        raise FileNotFoundError(
            f"No saved model found at {meta_file}. Run model_training.py first."
        )

    with open(meta_file, "rb") as f:
        meta = pickle.load(f)

    logger.info(f"Loaded metadata for {warehouse}")
    logger.info(f"Existing WorkCodes: {meta['workcodes']}")

    # ── Load and clean new data ──────────────────────────────────────────────
    locations, products = load_reference_tables(data_path, warehouse)
    df_new = load_new_data(new_data, locations, products)

    workcodes_in_new = sorted(df_new["WorkCode"].dropna().unique().tolist())
    logger.info(f"WorkCodes in new data: {workcodes_in_new}")
    logger.info(f"New rows total: {len(df_new):,}")

    updated_wcs = []
    skipped_wcs = []
    new_workers_added = 0

    for wc in workcodes_in_new:

        # Skip WorkCodes not in original training
        if wc not in meta["workcodes"]:
            logger.warning(f"WC {wc} not in original training — skipping. "
                           f"Run model_training.py to add new WorkCodes.")
            skipped_wcs.append(wc)
            continue

        model_file = models_dir / f"{warehouse}_WC{wc}_mod.json"
        if not model_file.exists():
            logger.warning(f"Model file missing for WC {wc}: {model_file} — skipping")
            skipped_wcs.append(wc)
            continue

        df_wc = df_new[df_new["WorkCode"] == wc].copy()
        if df_wc.empty:
            continue

        logger.info(f"\n{'='*50}\nUpdating WorkCode {wc} — {len(df_wc):,} new rows\n{'='*50}")

        # ── Engineer features using saved training encodings ─────────────────
        # Critical: use saved top_aisles and top_uoms from training so encoding
        # is consistent between training and update — not recomputed from new data
        df_wc, _, _ = engineer_features(
            df_wc,
            meta["bucket_map"],
            top_aisles=meta["top_aisles"].get(wc),
            top_uoms=meta["top_uoms"].get(wc),
        )

        # ── Update worker effects ────────────────────────────────────────────
        # Existing workers: keep their original estimates (based on more data)
        # New workers: estimate from new data and add to the table
        existing_effects = meta["worker_effects"].get(
            wc, pd.DataFrame(columns=["UserID", "worker_effect"])
        )
        existing_user_ids = set(existing_effects["UserID"].astype(str))
        new_user_ids = set(df_wc["UserID"].astype(str))
        truly_new_users = new_user_ids - existing_user_ids

        if truly_new_users:
            df_new_workers = df_wc[df_wc["UserID"].isin(truly_new_users)].copy()
            try:
                new_effects = estimate_worker_effects_new(df_new_workers)
                new_effects["UserID"] = new_effects["UserID"].astype(str)
                updated_effects = pd.concat(
                    [existing_effects, new_effects], ignore_index=True
                )
                logger.info(
                    f"  Added {len(truly_new_users)} new worker(s): {sorted(truly_new_users)}"
                )
                new_workers_added += len(truly_new_users)
            except Exception as e:
                logger.warning(f"  Could not estimate effects for new workers: {e}")
                updated_effects = existing_effects.copy()
        else:
            updated_effects = existing_effects.copy()
            logger.info(f"  No new workers — all {len(new_user_ids)} workers seen in training")

        # Recompute user levels with updated worker pool
        updated_effects_levelled, thresholds, level_medians = compute_worker_levels(
            updated_effects
        )
        meta["worker_effects"][wc]   = updated_effects_levelled
        meta["level_thresholds"][wc] = thresholds
        meta["level_medians"][wc]    = level_medians

        # ── Build feature matrix for new data ────────────────────────────────
        # Use each worker's effect — new workers get their new estimate,
        # existing workers get their original estimate
        df_wc = df_wc.merge(
            updated_effects_levelled[["UserID", "worker_effect"]],
            on="UserID", how="left"
        )
        df_wc["worker_effect"] = df_wc["worker_effect"].fillna(0.0)

        y_new = df_wc["Time_Delta_sec"].astype(float)
        X_new = make_X(df_wc, train_columns=None)
        X_new["worker_effect"] = df_wc["worker_effect"].values

        # Align to original training column order — critical for XGBoost
        train_cols = meta["train_columns"][wc]
        X_new = X_new.reindex(columns=train_cols, fill_value=0)

        d_new = xgb.DMatrix(X_new, label=y_new)

        # ── Load existing model and add new trees ────────────────────────────
        existing_model = xgb.Booster()
        existing_model.load_model(str(model_file))

        # Get current number of trees before update
        trees_before = existing_model.num_boosted_rounds()

        # XGB_PARAMS from model_training.py — same params for consistency
        xgb_params = dict(
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:tweedie",
            tweedie_variance_power=1.3,
            tree_method="hist",
            seed=2026,
        )

        updated_model = xgb.train(
            xgb_params,
            d_new,
            num_boost_round=n_trees,
            xgb_model=existing_model,   # incremental — adds on top of existing
            verbose_eval=False,
        )

        trees_after = updated_model.num_boosted_rounds()

        # ── Save updated model (overwrites previous) ─────────────────────────
        updated_model.save_model(str(model_file))
        logger.info(
            f"  Model updated: {trees_before} → {trees_after} trees "
            f"(+{trees_after - trees_before} added)"
        )
        logger.info(f"  Saved: {model_file}")

        updated_wcs.append(wc)

    # ── Save updated metadata ────────────────────────────────────────────────
    with open(meta_file, "wb") as f:
        pickle.dump(meta, f)
    logger.info(f"\nMetadata saved: {meta_file}")

    # ── Summary log ─────────────────────────────────────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info(f"UPDATE COMPLETE — {warehouse}")
    logger.info(f"{'='*50}")
    logger.info(f"  New rows processed: {len(df_new):,}")
    logger.info(f"  WorkCodes updated:  {updated_wcs}")
    if skipped_wcs:
        logger.info(f"  WorkCodes skipped:  {skipped_wcs}")
    logger.info(f"  New workers added:  {new_workers_added}")
    logger.info(f"  Trees added per WC: {n_trees}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
