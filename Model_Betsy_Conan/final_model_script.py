"""
final_model_script.py
----------------
Chunk-level XGBoost with worker random intercept effects.
Trains on deployment-safe features only — no distance, sequential, or time features.
Discovers WorkCodes automatically from the parquet file.

Usage:
    python final_model_script.py --warehouse OE
    python final_model_script.py --warehouse OF
    python final_model_script.py --warehouse RT
    python final_model_script.py --warehouse OE --data_dir /path/to/data/processed
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from feature_engineer import get_engineered_df

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_TIME = 300
BLOCK_SIZE = 50
RANDOM_STATE = 2026

# Features not available at prediction time — excluded from XGBoost
NOT_AVAILABLE = [
    "Travel_Distance",
    "same_aisle", "same_lockey", "same_location", "same_level", "diff_level",
    "time_of_day", "day_of_week", "hour",
]

XGB_PARAMS = dict(
    n_estimators=1200,
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
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ── Helper Functions ─────────────────────────────────────────────────────────
def resolve_data_path(data_dir, warehouse):
    return Path(data_dir) / f"{warehouse.lower()}_detailed.parquet"


def discover_workcodes(data_dir, warehouse):
    """Discover all WorkCodes from the parquet file automatically."""
    df_raw = pd.read_parquet(resolve_data_path(data_dir, warehouse))
    df_raw["WorkCode"] = df_raw["WorkCode"].astype(str).apply(
        lambda x: x.split(".")[0] if isinstance(x, str) else x
    )
    workcodes = sorted(df_raw["WorkCode"].dropna().unique().tolist())
    print(f"WorkCodes found: {workcodes}")
    del df_raw
    return workcodes


def load_engineered_data(data_dir, warehouse, workcode, max_time=MAX_TIME):
    d, features_all, cat_cols_all = get_engineered_df(
        file_path=resolve_data_path(data_dir, warehouse),
        warehouse=warehouse,
        max_time=max_time,
        work_code=workcode,
    )
    d = d.copy()
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")
    d = d.dropna(subset=["Timestamp"]).copy()
    d["date"]     = d["Timestamp"].dt.date
    d["WorkCode"] = d["WorkCode"].astype(str).str.replace(".0", 
                                                          "", regex=False)

    features = [f for f in features_all if f not in NOT_AVAILABLE]
    cat_cols = [c for c in cat_cols_all if c not in NOT_AVAILABLE]
    return d, features, cat_cols


def split_by_days(df, test_ratio=0.15):
    all_days = sorted(df["date"].dropna().unique())
    n_test_days = max(1, int(round(len(all_days) * test_ratio)))
    test_days = all_days[-n_test_days:]
    train_df = df[df["date"] < test_days[0]].copy()
    test_df = df[df["date"].isin(test_days)].copy()
    return train_df, test_df, test_days


def make_X(train_df, test_df, features, cat_cols):
    X_train = pd.get_dummies(train_df[features],
                             columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(test_df[features],
                            columns=cat_cols, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train = X_train.replace([np.inf, -np.inf],
                              np.nan).fillna(0).astype(float)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    return X_train, X_test


def eval_predictions(y_true, pred):
    return {
        "r2":   r2_score(y_true, pred),
        "mae":  mean_absolute_error(y_true, pred),
        "rmse": np.sqrt(mean_squared_error(y_true, pred)),
    }


def make_test_blocks(test_df, block_size=BLOCK_SIZE):
    d = test_df.sort_values(["UserID", "Timestamp"]).copy()
    blocks, block_rows = [], []
    for (uid, day), g in d.groupby(["UserID", "date"], sort=False):
        g = g.sort_values("Timestamp").reset_index().rename(columns={"index": "orig_index"}).copy()
        for start in range(0, len(g), block_size):
            chunk = g.iloc[start:start + block_size].copy()
            if len(chunk) < block_size:
                continue
            if chunk["WorkCode"].nunique() != 1:
                continue
            if (chunk["Time_Delta_sec"] > MAX_TIME).any():
                continue
            block_id = f"{uid}_{day}_{start // block_size}"
            chunk["BlockID"] = block_id
            block_rows.append(chunk)
            blocks.append({
                "BlockID":     block_id,
                "UserID":      uid,
                "date":        day,
                "WorkCode":    chunk["WorkCode"].iloc[0],
                "n_tasks":     len(chunk),
                "actual_time": chunk["Time_Delta_sec"].sum(),
                "start_ts":    chunk["Timestamp"].min(),
                "end_ts":      chunk["Timestamp"].max(),
            })
    block_df = pd.DataFrame(blocks)
    block_rows_df = pd.concat(block_rows, ignore_index=True) if block_rows else pd.DataFrame()
    return block_df, block_rows_df


def estimate_worker_effects(train_df):
    """
    Fits a random intercept model on training data.
    Returns a DataFrame with columns [UserID, worker_effect].
    worker_effect is the b_j estimate: positive = slower than average.
    Unseen workers at test time should get effect = 0 (grand mean fallback).
    """
    df_re = train_df[["UserID", "Time_Delta_sec"]].dropna().copy()
    if df_re["UserID"].nunique() < 2:
        print("  [Warning] Not enough workers — skipping worker effects")
        return pd.DataFrame({"UserID": df_re["UserID"].unique(),
                             "worker_effect": 0.0})

    result = smf.mixedlm(
        "Time_Delta_sec ~ 1", data=df_re, groups=df_re["UserID"]
    ).fit(reml=True, disp=False)

    icc = result.cov_re.values[0][0] / (result.cov_re.values[0][0] + result.scale)
    print(f"  Grand mean: {result.fe_params['Intercept']:.1f}s | "
          f"Worker SD: {np.sqrt(result.cov_re.values[0][0]):.1f}s | ICC: {icc:.3f}")

    return pd.DataFrame({
        "UserID":        list(result.random_effects.keys()),
        "worker_effect": [float(v.iloc[0]) for v in result.random_effects.values()]
    })


def run_baseline(warehouse, workcodes, data_dir):
    """Run XGBoost baseline (no worker effects) for all WorkCodes."""
    all_block_results = []
    all_block_detail = []

    for wc in workcodes:
        print(f"\n{'='*50}")
        print(f"WorkCode {wc}")
        print(f"{'='*50}")

        df_wc, features, cat_cols = load_engineered_data(data_dir, warehouse, wc)
        train_df, test_df, test_days = split_by_days(df_wc)
        print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
        print(f"Features: {features}")

        y_train = train_df["Time_Delta_sec"].astype(float)
        y_test = test_df["Time_Delta_sec"].astype(float)
        X_train, X_test = make_X(train_df, test_df, features, cat_cols)

        t0 = time.perf_counter()
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        runtime = time.perf_counter() - t0

        block_df, block_rows_df = make_test_blocks(test_df,
                                                   block_size=BLOCK_SIZE)
        if len(block_df) == 0:
            print(f"No valid blocks for WC {wc}")
            continue

        temp = test_df.copy().reset_index().rename(columns={"index": "orig_index"})
        temp["pred"] = preds
        block_rows_df = block_rows_df.merge(temp[["orig_index", "pred"]], on="orig_index", how="left")

        block_pred = (
            block_rows_df.groupby("BlockID")
            .agg(
                actual_time=("Time_Delta_sec", "sum"),
                pred=("pred", "sum"),
                WorkCode=("WorkCode", "first"),
                UserID=("UserID", "first"),
                date=("date", "first"),
                n_tasks=("Time_Delta_sec", "size"),
            )
            .reset_index()
        )

        metrics = eval_predictions(block_pred["actual_time"], block_pred["pred"])
        print(f"  Blocks: {len(block_pred)} | MAE: {metrics['mae']:.1f}s | "
              f"MAE/task: {metrics['mae']/BLOCK_SIZE:.3f}s | R²: {metrics['r2']:.4f}")

        all_block_results.append({
            "Warehouse": warehouse,
            "WorkCode":  wc,
            "n_blocks":  len(block_pred),
            **metrics
        })
        block_pred["Warehouse"] = warehouse
        all_block_detail.append(block_pred)

    block_results_df = pd.DataFrame(all_block_results)
    block_detail_df = pd.concat(all_block_detail, ignore_index=True) if all_block_detail else pd.DataFrame()
    return block_results_df, block_detail_df


def run_worker_effects(warehouse, workcodes, data_dir):
    """Run XGBoost + worker random intercept for all WorkCodes."""
    all_block_results_w = []
    all_block_detail_w = []

    for wc in workcodes:
        print(f"\n{'='*50}")
        print(f"WorkCode {wc} — + Worker Effect")
        print(f"{'='*50}")

        df_wc, features, cat_cols = load_engineered_data(data_dir, warehouse, wc)
        train_df, test_df, test_days = split_by_days(df_wc)

        print("  Fitting mixed model...")
        worker_effects = estimate_worker_effects(train_df)

        train_df = train_df.merge(worker_effects, on="UserID", how="left")
        test_df = test_df.merge(worker_effects,  on="UserID", how="left")
        train_df["worker_effect"] = train_df["worker_effect"].fillna(0.0)
        test_df["worker_effect"]  = test_df["worker_effect"].fillna(0.0)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        y_train = train_df["Time_Delta_sec"].astype(float)
        y_test = test_df["Time_Delta_sec"].astype(float)

        feats_w = features + ["worker_effect"]
        X_train, X_test = make_X(train_df, test_df, feats_w, cat_cols)

        t0 = time.perf_counter()
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        runtime = time.perf_counter() - t0

        block_df, block_rows_df = make_test_blocks(test_df, block_size=BLOCK_SIZE)
        if len(block_df) == 0:
            print(f"  No valid blocks for WC {wc}")
            continue

        temp = test_df.copy().reset_index().rename(columns={"index": "orig_index"})
        temp["pred"] = preds
        block_rows_df = block_rows_df.merge(temp[["orig_index", "pred"]],
                                            on="orig_index", how="left")

        block_pred = (
            block_rows_df.groupby("BlockID")
            .agg(
                actual_time=("Time_Delta_sec", "sum"),
                pred=("pred", "sum"),
                WorkCode=("WorkCode", "first"),
                UserID=("UserID", "first"),
                date=("date", "first"),
                n_tasks=("Time_Delta_sec", "size"),
            )
            .reset_index()
        )

        metrics = eval_predictions(block_pred["actual_time"], block_pred["pred"])
        print(f"  Blocks: {len(block_pred)} | MAE: {metrics['mae']:.1f}s | "
              f"MAE/task: {metrics['mae']/BLOCK_SIZE:.3f}s | R²: {metrics['r2']:.4f}")

        all_block_results_w.append({
            "Warehouse": warehouse,
            "WorkCode":  wc,
            "n_blocks":  len(block_pred),
            **metrics
        })
        block_pred["Warehouse"] = warehouse
        all_block_detail_w.append(block_pred)

    block_results_w_df = pd.DataFrame(all_block_results_w)
    block_detail_w_df = pd.concat(all_block_detail_w,
                                  ignore_index=True) if all_block_detail_w else pd.DataFrame()
    return block_results_w_df, block_detail_w_df


def print_comparison(warehouse, workcodes, block_results_df,
                     block_results_w_df):
    """Print baseline vs + worker comparison table."""
    rows = []
    for wc in workcodes:
        base = block_results_df[block_results_df["WorkCode"] == wc]
        enh = block_results_w_df[block_results_w_df["WorkCode"] == wc]
        if base.empty or enh.empty:
            continue
        mae_base = base["mae"].values[0]
        mae_enh = enh["mae"].values[0]
        rows.append({
            "WorkCode":              wc,
            "MAE/task baseline (s)": round(mae_base / BLOCK_SIZE, 3),
            "MAE/task + worker (s)": round(mae_enh / BLOCK_SIZE, 3),
            "Improvement (s)":       round((mae_base - mae_enh) / BLOCK_SIZE, 3),
            "Improvement (%)":       round((mae_base - mae_enh) / mae_base * 100, 2),
            "R² baseline":           round(base["r2"].values[0], 4),
            "R² + worker":           round(enh["r2"].values[0],  4),
        })
    print(f"\nWarehouse: {warehouse} | Block size: {BLOCK_SIZE} tasks")
    print(pd.DataFrame(rows).to_string(index=False))

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Chunk-level XGBoost with worker random intercept effects."
    )
    parser.add_argument(
        "--warehouse", required=True,
        help="Warehouse name e.g. OE, OF, RT"
    )
    parser.add_argument(
        "--data_dir", default="../data/processed",
        help="Path to directory containing warehouse_detailed.parquet files"
    )
    args = parser.parse_args()

    warehouse = args.warehouse.upper()
    data_dir  = args.data_dir

    print(f"\nWarehouse: {warehouse}")
    print(f"Data dir:  {data_dir}")
    print(f"Block size: {BLOCK_SIZE} tasks\n")

    # Discover WorkCodes automatically
    workcodes = discover_workcodes(data_dir, warehouse)

    # Run baseline
    print("\n" + "="*60)
    print("BASELINE — XGBoost (no worker effects)")
    print("="*60)
    block_results_df, block_detail_df = run_baseline(warehouse, workcodes, data_dir)

    print(f"\nBaseline results — {warehouse} | Block size: {BLOCK_SIZE}")
    clean = block_results_df.copy()
    clean["mae_per_task"] = (clean["mae"] / BLOCK_SIZE).round(3)
    clean["r2"]  = clean["r2"].round(3)
    clean["mae"] = clean["mae"].round(1)
    print(clean.drop(columns=["rmse", "Warehouse"], errors="ignore").to_string(index=False))


    # Run worker effects
    print("\n" + "="*60)
    print("+ WORKER EFFECTS — XGBoost + random intercept (UserID)")
    print("="*60)
    block_results_w_df, block_detail_w_df = run_worker_effects(warehouse, workcodes, data_dir)

    # Comparison table
    print("\n" + "="*60)
    print("COMPARISON — Baseline vs + Worker")
    print("="*60)
    print_comparison(warehouse, workcodes, block_results_df, block_results_w_df)

    # Save results to CSV
    # out_base = f"results_baseline_{warehouse.lower()}.csv"
    # out_work = f"results_worker_{warehouse.lower()}.csv"
    # block_results_df.to_csv(out_base, index=False)
    # block_results_w_df.to_csv(out_work, index=False)
    # print(f"\nResults saved to {out_base} and {out_work}")


if __name__ == "__main__":
    main()
