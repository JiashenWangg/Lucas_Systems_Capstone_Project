"""
predict_primary.py
------------------
Predict total completion time for a warehouse assignment (work queue).

Given a CSV of tasks with no timestamps (upcoming picks), predicts the
total time in minutes for a worker to complete the entire assignment.

WorkCode is auto-detected from the CSV. An assignment must contain exactly
one WorkCode — if multiple are found the script exits with a clear error.

Usage:
    python predict_primary.py OE assignment.csv
    python predict_primary.py OE assignment.csv --sequenced
    python predict_primary.py OE assignment.csv --user_level 3
    python predict_primary.py OE assignment.csv --out results/prediction.csv

Args:
    warehouse:      Warehouse code (OE, OF, RT)
    predict_csv:    CSV of tasks to predict (no Timestamp column)
    --data_dir:     Root training_data directory (default: training_data)
    --models_dir:   Root models directory (default: models)
    --sequenced:    Use sequenced model — rows must be in pick order
    --user_level:   Worker performance level 1-5 (1=slowest, 5=fastest).
                    Applied uniformly to all rows in the assignment.
                    If omitted, uses grand mean (level 3 equivalent).
    --out:          Output CSV path (default: predict_data/WH/prediction.csv)

Output CSV columns:
    warehouse, workcode, n_tasks, predicted_time_sec, predicted_time_min,
    user_level_applied
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_pipeline import prepare_predict_data
from utils.io import (
    load_meta,
    load_model,
    setup_logging,
)
from utils.worker_effects import level_to_effect


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict total assignment completion time"
    )
    parser.add_argument("warehouse",     help="Warehouse code e.g. OE, OF, RT")
    parser.add_argument("predict_csv",   help="CSV of tasks to predict")
    parser.add_argument("--data_dir",    default="training_data")
    parser.add_argument("--models_dir",  default="models")
    parser.add_argument("--sequenced",   action="store_true",
                        help="Use sequenced model (rows must be in pick order)")
    parser.add_argument("--user_level",  type=int, default=None,
                        choices=[1, 2, 3, 4, 5],
                        help="Worker level 1-5 (1=slowest, 5=fastest). "
                             "Default: grand mean (level 3 equivalent).")
    parser.add_argument("--out",         default=None,
                        help="Output CSV path")
    return parser.parse_args()


def main():
    args      = parse_args()
    warehouse = args.warehouse.upper()

    setup_logging("predict_primary", warehouse)
    logger = logging.getLogger(__name__)

    logger.info(f"{'='*60}")
    logger.info(f"PREDICT PRIMARY — {warehouse}")
    logger.info(f"  input:      {args.predict_csv}")
    logger.info(f"  sequenced:  {args.sequenced}")
    logger.info(f"  user_level: {args.user_level if args.user_level else 'grand mean'}")
    logger.info(f"{'='*60}")

    # ── Load meta ─────────────────────────────────────────────────────────────
    meta = load_meta(args.models_dir, warehouse)

    # ── Load and engineer prediction data ─────────────────────────────────────
    # WorkCode is auto-detected; raises ValueError if multiple WCs found
    try:
        df, X, wc = prepare_predict_data(
            predict_csv = args.predict_csv,
            data_dir    = args.data_dir,
            warehouse   = warehouse,
            encodings   = {
                **meta["encodings"].get(wc_key := meta["workcodes"][0], {}),
                "train_columns": None,  # resolved per-WC below
            },
            sequenced   = args.sequenced,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"  WorkCode detected: {wc}")
    logger.info(f"  Tasks: {len(df):,}")

    # Validate WorkCode against trained models
    if wc not in meta["workcodes"]:
        logger.error(
            f"WorkCode {wc} not in trained models for {warehouse}. "
            f"Available: {meta['workcodes']}. "
            f"Run model_training.py to train a model for this WorkCode."
        )
        sys.exit(1)

    # Re-prepare with correct per-WC encodings
    wc_encodings = {
        **meta["encodings"].get(wc, {}),
        "train_columns": meta["train_columns"].get(wc),
    }
    try:
        df, X, wc = prepare_predict_data(
            predict_csv = args.predict_csv,
            data_dir    = args.data_dir,
            warehouse   = warehouse,
            encodings   = wc_encodings,
            sequenced   = args.sequenced,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Apply worker effect ───────────────────────────────────────────────────
    if args.user_level is not None:
        worker_effect = level_to_effect(
            args.user_level,
            meta["level_medians"].get(wc, {}),
        )
        logger.info(
            f"  User level {args.user_level} → "
            f"worker_effect = {worker_effect:+.3f}s"
        )
    else:
        worker_effect = 0.0
        logger.info("  No user level supplied — using grand mean (effect = 0.0)")

    X["worker_effect"] = worker_effect
    X = X.reindex(columns=meta["train_columns"][wc], fill_value=0)

    # ── Load model and predict ────────────────────────────────────────────────
    try:
        model = load_model(
            args.models_dir, warehouse, wc, sequenced=args.sequenced
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    preds = model.predict(xgb.DMatrix(X))

    # Total predicted time = sum of per-task predictions
    total_sec = float(np.sum(preds))
    total_min = total_sec / 60.0

    logger.info(f"\n  Tasks:              {len(df):,}")
    logger.info(f"  Predicted total:    {total_sec:.1f}s  ({total_min:.2f} min)")

    # ── Save output ───────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (
            Path("predict_data") / warehouse
            / f"{warehouse}_{wc}_prediction.csv"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame([{
        "warehouse":           warehouse,
        "workcode":            wc,
        "n_tasks":             len(df),
        "predicted_time_sec":  round(total_sec, 2),
        "predicted_time_min":  round(total_min, 4),
        "user_level_applied":  args.user_level if args.user_level else "grand_mean",
    }])
    result.to_csv(out_path, index=False)
    logger.info(f"\n  Output saved: {out_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
