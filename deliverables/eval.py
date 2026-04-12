"""
eval.py
-------
Evaluate pick-time model performance on held-out test data.

Trains a fresh model internally with a proper chronological train/test split —
does NOT load from the models/ folder to avoid data leakage.

Primary goal (arg1=1): predict chunk-level completion time.
    Chunks are consecutive picks by the same worker on the same day.

Secondary goal (arg1=2): deferred — not yet implemented.

Usage:
    python eval.py OE 1 50
    python eval.py OE 1 50 --data_dir training_data --sequenced
    python eval.py OE 1 50 --out eval_results

Args:
    warehouse:    Warehouse code (OE, OF, RT)
    goal:         1 = primary (predict time), 2 = secondary (deferred)
    chunk_size:   Number of tasks per chunk (goal=1)
    --data_dir:   Root training_data directory (default: training_data)
    --models_dir: Not used — eval trains its own models to avoid leakage
    --sequenced:  Evaluate with sequence features
    --test_pct:   Fraction of data to hold out as test set (default: 0.2)
    --out:        Root output directory (default: eval_results)
    --trees:      Boosting rounds for eval model (default: 1200)

Output:
    eval_results/WH/primary/WH_eval_primary_{chunk_size}.csv
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_pipeline import load_and_engineer
from utils.io import (
    load_parquet,
    setup_logging,
)
from utils.worker_effects import compute_worker_levels, estimate_worker_effects

XGB_PARAMS = dict(
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

DEFAULT_TREES    = 1200
DEFAULT_TEST_PCT = 0.20
DEFAULT_MIN_ROWS = 500
MAX_TIME         = 600


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate pick-time model with train/test split"
    )
    parser.add_argument("warehouse",   help="Warehouse code e.g. OE, OF, RT")
    parser.add_argument("goal",        type=int, choices=[1, 2],
                        help="1=primary (predict time), 2=secondary (deferred)")
    parser.add_argument("chunk_size",  type=int,
                        help="Tasks per chunk (goal=1)")
    parser.add_argument("--data_dir",  default="training_data")
    parser.add_argument("--sequenced", action="store_true")
    parser.add_argument("--test_pct",  type=float, default=DEFAULT_TEST_PCT)
    parser.add_argument("--out",       default="eval_results")
    parser.add_argument("--trees",     type=int, default=DEFAULT_TREES)
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def discover_workcodes(data_dir, warehouse):
    df = load_parquet(data_dir, warehouse)
    df["WorkCode"] = (
        df["WorkCode"].astype(str)
        .apply(lambda x: x.split(".")[0] if isinstance(x, str) else x)
    )
    wcs = df["WorkCode"].dropna().unique().tolist()
    return sorted(w for w in wcs if w.lower() != "nan")


def chronological_split(df, test_pct):
    """
    Split by row count (chronological order preserved by Timestamp sort).
    Returns (train_df, test_df).
    """
    df = df.sort_values(["Timestamp"]).reset_index(drop=True)
    n_train = int(len(df) * (1 - test_pct))
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def make_chunks(df, chunk_size, max_time=MAX_TIME):
    """
    Group consecutive picks per user per day into fixed-size chunks.
    A chunk is valid if:
      - Exactly chunk_size picks
      - All picks have Time_Delta_sec <= max_time
    Returns DataFrame with BlockID, actual_time (sum), and pred (filled later).
    """
    df = df.sort_values(["UserID", "Timestamp"]).copy()
    df["date"] = pd.to_datetime(df["Timestamp"]).dt.date

    chunks = []
    for (uid, day), g in df.groupby(["UserID", "date"], sort=False):
        g = g.reset_index(drop=True)
        for start in range(0, len(g), chunk_size):
            chunk = g.iloc[start:start + chunk_size]
            if len(chunk) < chunk_size:
                continue
            if (chunk["Time_Delta_sec"] > max_time).any():
                continue
            chunks.append({
                "BlockID":     f"{uid}_{day}_{start // chunk_size}",
                "UserID":      uid,
                "date":        day,
                "n_tasks":     len(chunk),
                "actual_time": chunk["Time_Delta_sec"].sum(),
                "indices":     chunk.index.tolist(),
            })
    return pd.DataFrame(chunks)


# ── Primary goal evaluation ───────────────────────────────────────────────────

def eval_primary(warehouse, data_dir, chunk_size, sequenced,
                 test_pct, trees, out_dir, logger):
    """Train and evaluate all WorkCodes for goal 1 (predict completion time)."""

    workcodes = discover_workcodes(data_dir, warehouse)
    logger.info(f"WorkCodes: {workcodes}")

    results = []

    for wc in workcodes:
        logger.info(f"\n{'-'*50}")
        logger.info(f"WorkCode {wc}")

        try:
            df, X_full, y_full, encodings = load_and_engineer(
                data_dir  = data_dir,
                warehouse = warehouse,
                wc        = wc,
                sequenced = sequenced,
                encodings = None,
            )
        except Exception as e:
            logger.warning(f"  Could not load: {e} — skipping")
            continue

        if len(df) < DEFAULT_MIN_ROWS:
            logger.info(f"  Only {len(df):,} rows — skipping")
            continue

        # Chronological split
        train_df, test_df = chronological_split(df, test_pct)
        logger.info(
            f"  Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows"
        )

        # Re-engineer train only (so encodings are computed from train only)
        try:
            train_df_eng, X_train, y_train, train_encodings = load_and_engineer(
                data_dir  = data_dir,
                warehouse = warehouse,
                wc        = wc,
                sequenced = sequenced,
                encodings = None,
            )
            # Use only training rows
            train_mask = df.index.isin(train_df.index)
            X_train = X_full.loc[train_mask].copy()
            y_train = y_full.loc[train_mask].copy()

            # Apply train encodings to test set
            from utils.feature_engineer import apply_features, make_X
            test_df_eng = apply_features(
                test_df,
                top_aisles    = encodings["top_aisles"],
                top_uoms      = encodings["top_uoms"],
                product_tiers = encodings["product_tiers"],
                sequenced     = sequenced,
            )
            X_test = make_X(
                test_df_eng,
                sequenced     = sequenced,
                train_columns = X_train.columns.tolist(),
            )
            y_test = test_df["Time_Delta_sec"].astype(float)

        except Exception as e:
            logger.warning(f"  Feature engineering failed: {e} — skipping")
            continue

        # Worker effects (fit on train only, applied to both)
        effects = estimate_worker_effects(train_df)
        effects_lev, _, _ = compute_worker_levels(effects)

        train_df = train_df.copy()
        train_df = train_df.merge(
            effects_lev[["UserID", "worker_effect"]], on="UserID", how="left"
        )
        train_df["worker_effect"] = train_df["worker_effect"].fillna(0.0)
        X_train["worker_effect"]  = train_df["worker_effect"].values

        test_df = test_df.copy()
        test_df = test_df.merge(
            effects_lev[["UserID", "worker_effect"]], on="UserID", how="left"
        )
        test_df["worker_effect"] = test_df["worker_effect"].fillna(0.0)
        X_test["worker_effect"]  = test_df["worker_effect"].values

        train_cols = X_train.columns.tolist()
        X_test = X_test.reindex(columns=train_cols, fill_value=0)

        # Train
        logger.info(f"  Training ({trees} trees) ...")
        t0     = time.time()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model  = xgb.train(
            XGB_PARAMS, dtrain,
            num_boost_round=trees,
            verbose_eval=False,
        )
        train_time = time.time() - t0
        logger.info(f"  Done in {train_time:.1f}s")

        # Predict on test set
        dtest       = xgb.DMatrix(X_test)
        preds       = model.predict(dtest)
        test_df     = test_df.reset_index(drop=True)
        test_df["pred"] = preds

        # Chunk-level evaluation
        chunk_df = make_chunks(test_df, chunk_size)
        if chunk_df.empty:
            logger.warning(
                f"  No valid chunks found (chunk_size={chunk_size}) — skipping"
            )
            continue

        # Aggregate predictions to chunk level
        chunk_preds = []
        for _, row in chunk_df.iterrows():
            idx_preds = test_df.loc[row["indices"], "pred"].sum()
            chunk_preds.append(idx_preds)
        chunk_df["pred_time"] = chunk_preds

        chunk_mae  = mean_absolute_error(
            chunk_df["actual_time"], chunk_df["pred_time"]
        )
        mae_per_task = chunk_mae / chunk_size
        r2           = r2_score(chunk_df["actual_time"], chunk_df["pred_time"])

        actual_times = chunk_df["actual_time"]
        logger.info(
            f"  Chunks: {len(chunk_df)}  |  "
            f"MAE/task: {mae_per_task:.3f}s  |  R²: {r2:.4f}"
        )

        results.append({
            "warehouse":       warehouse,
            "workcode":        wc,
            "train_rows":      len(train_df),
            "train_time_s":    round(train_time, 1),
            "test_rows":       len(test_df),
            "test_chunks":     len(chunk_df),
            "mean_time_s":     round(float(actual_times.mean()), 2),
            "median_time_s":   round(float(actual_times.median()), 2),
            "r2":              round(r2, 4),
            "mae_per_task_s":  round(mae_per_task, 4),
        })

    if not results:
        logger.warning("No WorkCodes produced results.")
        return

    results_df = pd.DataFrame(results)

    out_path = (
        Path(out_dir) / warehouse / "primary"
        / f"{warehouse}_eval_primary_{chunk_size}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    logger.info(f"\nResults saved: {out_path}")
    logger.info(f"\n{results_df.to_string(index=False)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    warehouse = args.warehouse.upper()

    setup_logging("eval", warehouse)
    logger = logging.getLogger(__name__)

    logger.info(f"{'='*60}")
    logger.info(f"EVAL — {warehouse}  goal={args.goal}  chunk={args.chunk_size}")
    logger.info(f"  sequenced: {args.sequenced}  |  test_pct: {args.test_pct}")
    logger.info(f"{'='*60}")

    if args.goal == 1:
        eval_primary(
            warehouse  = warehouse,
            data_dir   = args.data_dir,
            chunk_size = args.chunk_size,
            sequenced  = args.sequenced,
            test_pct   = args.test_pct,
            trees      = args.trees,
            out_dir    = args.out,
            logger     = logger,
        )
    elif args.goal == 2:
        logger.info("Secondary goal evaluation is not yet implemented.")
        sys.exit(0)

    logger.info(f"\n{'='*60}")
    logger.info("EVAL COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
