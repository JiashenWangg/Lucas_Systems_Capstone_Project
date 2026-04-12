"""
update_model_incremental.py
----------------------------
Stateless incremental update for XGBoost pick-time models.

Each run:
  1. Loads today's raw Activity CSV (completed picks — has Timestamps)
  2. Runs preprocessing steps: computes time delta, joins Products/Locations,
     filters outliers
  3. Applies saved feature encodings from meta.pkl (never recomputed)
  4. Adds UPDATE_TREES new trees using a low learning rate (0.005)
  5. Updates worker effects for any new workers
  6. Saves updated model files and meta.pkl
  7. Tracks batch MAE before and after update; alerts if degradation detected

Why low learning rate (0.005 vs 0.03 at training time):
  Each day's batch is small relative to the full training history. At the
  training learning rate, new trees overfit to that day's noise and pull
  predictions in random directions. At 0.005, each tree contributes ~6x
  less — real patterns that persist across many days accumulate, noise
  averages out.

Usage:
    python update_model_incremental.py OE --new_data daily_activity.csv
    python update_model_incremental.py OE --new_data daily_activity.csv --trees 50
    python update_model_incremental.py OE --new_data daily_activity.csv --sequenced
    python update_model_incremental.py OE --new_data daily_activity.csv --alert_pct 20

Args:
    warehouse:    Warehouse code (OE, OF, RT)
    --new_data:   Path to today's raw Activity CSV (completed picks with Timestamps)
    --data_dir:   Root training_data directory (default: training_data)
    --models_dir: Root models directory (default: models)
    --sequenced:  Use sequenced models (must match how models were trained)
    --trees:      Trees to add per WorkCode per run (default: 50)
    --alert_pct:  Alert if post-update batch MAE exceeds baseline by this %
                  (default: 20)
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_pipeline import prepare_new_data
from utils.io import (
    load_meta,
    load_model,
    save_meta,
    save_model,
    setup_logging,
)
from utils.worker_effects import (
    compute_worker_levels,
    estimate_worker_effects,
)

UPDATE_LR        = 0.005
DEFAULT_TREES    = 150
DEFAULT_ALERT_PCT = 20

XGB_PARAMS_BASE = dict(
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stateless incremental update with low-LR naive update"
    )
    parser.add_argument("warehouse",     help="Warehouse code e.g. OE, OF, RT")
    parser.add_argument("--new_data",    required=True,
                        help="Path to today's raw Activity CSV (with Timestamps)")
    parser.add_argument("--data_dir",    default="training_data")
    parser.add_argument("--models_dir",  default="models")
    parser.add_argument("--sequenced",   action="store_true")
    parser.add_argument("--trees",       type=int, default=DEFAULT_TREES)
    parser.add_argument("--alert_pct",   type=float, default=DEFAULT_ALERT_PCT)
    return parser.parse_args()


def compute_batch_mae(model, X, y):
    preds = model.predict(xgb.DMatrix(X))
    return float(np.mean(np.abs(preds - y.values)))


def main():
    args      = parse_args()
    warehouse = args.warehouse.upper()
    today     = date.today().isoformat()

    setup_logging("update_model", warehouse)
    logger = logging.getLogger(__name__)

    logger.info(f"{'='*60}")
    logger.info(f"UPDATE MODEL — {warehouse}  ({today})")
    logger.info(f"  new_data   = {args.new_data}")
    logger.info(f"  trees/run  = {args.trees}  |  lr = {UPDATE_LR}")
    logger.info(f"  alert_pct  = {args.alert_pct}%")
    logger.info(f"  sequenced  = {args.sequenced}")
    logger.info(f"{'='*60}")

    meta = load_meta(args.models_dir, warehouse)

    if meta.get("sequenced", False) != args.sequenced:
        logger.warning(
            f"  --sequenced={args.sequenced} but models were trained with "
            f"sequenced={meta.get('sequenced')}. Make sure they match."
        )

    updated_wcs      = []
    skipped_wcs      = []
    alerted_wcs      = []
    new_workers_total = 0

    for wc in meta["workcodes"]:
        logger.info(f"\n{'-'*50}")
        logger.info(f"WorkCode {wc}")

        # Load saved encodings for this WC
        wc_encodings = meta["encodings"].get(wc, {})
        wc_encodings["train_columns"] = meta["train_columns"].get(wc)

        try:
            df, X, y = prepare_new_data(
                activity_csv = args.new_data,
                data_dir     = args.data_dir,
                warehouse    = warehouse,
                wc           = wc,
                encodings    = wc_encodings,
                sequenced    = args.sequenced,
            )
        except ValueError as e:
            logger.info(f"  {e} — skipping")
            skipped_wcs.append(wc)
            continue
        except Exception as e:
            logger.warning(f"  Error loading data for WC {wc}: {e} — skipping")
            skipped_wcs.append(wc)
            continue

        logger.info(f"  New rows: {len(df):,}")

        # ── Worker effects ────────────────────────────────────────────────────
        existing_effects  = meta["worker_effects"].get(wc)
        existing_user_ids = set(existing_effects["UserID"].astype(str))
        new_user_ids      = set(df["UserID"].astype(str))
        truly_new         = new_user_ids - existing_user_ids

        if truly_new:
            logger.info(
                f"  New workers: {sorted(truly_new)} — estimating effects"
            )
            try:
                new_eff = estimate_worker_effects(
                    df[df["UserID"].isin(truly_new)]
                )
                import pandas as pd
                updated_effects = pd.concat(
                    [existing_effects, new_eff], ignore_index=True
                )
                new_workers_total += len(truly_new)
            except Exception as e:
                logger.warning(
                    f"  Could not estimate new worker effects: {e} "
                    "— using grand mean (0.0)"
                )
                updated_effects = existing_effects.copy()
        else:
            updated_effects = existing_effects.copy()
            logger.info(
                f"  All {len(new_user_ids)} workers seen in training"
            )

        updated_levelled, thresholds, level_medians = compute_worker_levels(
            updated_effects
        )
        meta["worker_effects"][wc]   = updated_levelled
        meta["level_thresholds"][wc] = thresholds
        meta["level_medians"][wc]    = level_medians

        # Add worker effect to feature matrix
        import pandas as pd
        df = df.merge(
            updated_levelled[["UserID", "worker_effect"]],
            on="UserID", how="left"
        )
        df["worker_effect"] = df["worker_effect"].fillna(0.0)
        X["worker_effect"]  = df["worker_effect"].values
        X = X.reindex(columns=meta["train_columns"][wc], fill_value=0)

        d_new = xgb.DMatrix(X, label=y)

        # ── Load model and compute MAE before update ──────────────────────────
        try:
            existing_model = load_model(
                args.models_dir, warehouse, wc, sequenced=args.sequenced
            )
        except FileNotFoundError as e:
            logger.warning(f"  {e} — skipping")
            skipped_wcs.append(wc)
            continue

        trees_before = existing_model.num_boosted_rounds()
        mae_before   = compute_batch_mae(existing_model, X, y)
        logger.info(
            f"  MAE before: {mae_before:.3f}s  |  Trees: {trees_before}"
        )

        # ── Update ────────────────────────────────────────────────────────────
        xgb_params = {**XGB_PARAMS_BASE, "learning_rate": UPDATE_LR}
        updated_model = xgb.train(
            xgb_params, d_new,
            num_boost_round=args.trees,
            xgb_model=existing_model,
            verbose_eval=False,
        )
        trees_after = updated_model.num_boosted_rounds()
        mae_after   = compute_batch_mae(updated_model, X, y)

        direction = "↓" if mae_after < mae_before else "↑"
        logger.info(
            f"  MAE after:  {mae_after:.3f}s {direction}  |  "
            f"Trees: {trees_before} → {trees_after} "
            f"(+{trees_after - trees_before})"
        )

        # ── Alert check ───────────────────────────────────────────────────────
        alerted = False
        baseline = meta["baseline_mae"].get(wc)

        if baseline is None:
            meta["baseline_mae"][wc] = mae_before
            baseline = mae_before
            logger.info(
                f"  Baseline MAE set: {baseline:.3f}s (first update run)"
            )

        if baseline > 0:
            pct_above = 100.0 * (mae_after - baseline) / baseline
            if pct_above > args.alert_pct:
                logger.warning(
                    f"  ⚠ ALERT WC {wc}: MAE {mae_after:.3f}s is "
                    f"{pct_above:.1f}% above baseline {baseline:.3f}s "
                    f"(threshold: {args.alert_pct:.0f}%). "
                    "Model may be degrading."
                )
                alerted = True
                alerted_wcs.append(wc)

        # ── Record MAE history ────────────────────────────────────────────────
        if wc not in meta["mae_history"]:
            meta["mae_history"][wc] = []
        meta["mae_history"][wc].append({
            "date":       today,
            "mae_before": round(mae_before, 4),
            "mae_after":  round(mae_after, 4),
            "n_rows":     len(df),
            "trees":      trees_after,
            "alerted":    alerted,
        })

        # Print last 7-run trend
        history = meta["mae_history"][wc][-7:]
        trend   = "  ".join(
            f"{e['date']} {e['mae_after']:.3f}s" for e in history
        )
        logger.info(f"  Trend: {trend}")

        # ── Save model ────────────────────────────────────────────────────────
        save_model(updated_model, args.models_dir, warehouse, wc,
                   sequenced=args.sequenced)
        updated_wcs.append(wc)

    # ── Save meta ─────────────────────────────────────────────────────────────
    save_meta(meta, args.models_dir, warehouse)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"UPDATE COMPLETE — {warehouse}  ({today})")
    logger.info(f"{'='*60}")
    logger.info(f"  WorkCodes updated:  {updated_wcs}")
    if skipped_wcs:
        logger.info(f"  WorkCodes skipped:  {skipped_wcs}")
    logger.info(f"  New workers added:  {new_workers_total}")
    logger.info(f"  Trees added/WC:     {args.trees}  (lr={UPDATE_LR})")
    if alerted_wcs:
        logger.warning(
            f"  ⚠ ALERTS fired for: {alerted_wcs} — "
            "consider retraining if alerts persist."
        )
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
