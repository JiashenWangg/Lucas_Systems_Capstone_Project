"""
model_training.py
-----------------
Train XGBoost pick-time models for all WorkCodes in a warehouse.

Trains on ALL available data — no train/test split. For evaluation
with a proper holdout, use eval.py instead.

Optional: per WorkCode, run 5-fold CV to tune XGBoost hyperparameters,
then train final model with CV-selected params.

Usage:
    python model_training.py OE
    python model_training.py OE --data_dir training_data --models_dir models
    python model_training.py OE --sequenced

Args:
    warehouse:     Warehouse code (OE, OF, RT)
    --data_dir:    Root training_data directory (default: training_data)
    --models_dir:  Root models directory (default: models)
    --sequenced:   Train with sequence-dependent features (Travel_Distance,
                   same_aisle, same_level). Requires distance matrix in
                   training_data/WH/. Saves as WH_WCxx_seq.json.
    --trees:       Number of boosting rounds (default: 1200)
    --min_rows:    Skip WorkCodes with fewer rows than this (default: 500)
    --min_days:    Advisory threshold for unique dates. If below this, training
                   still runs but logs a warning (default: 3).
    --cv:          Enable 5-fold CV parameter tuning (default: False)

Output (in models/WH/):
    WH_WCxx.json or WH_WCxx_seq.json   — one per WorkCode
    meta.pkl                            — encodings, worker effects, column order,
                                          selected xgb params and rounds
    logs/WH/model_training_YYYY-MM-DD.log
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_pipeline import load_and_engineer
from utils.io import (
    load_parquet,
    save_meta,
    save_model,
    setup_logging,
)
from utils.worker_effects import compute_worker_levels, estimate_worker_effects


XGB_FIXED_PARAMS = dict(
    objective="reg:tweedie",
    tweedie_variance_power=1.3,
    tree_method="hist",
    seed=2026,
)

XGB_BASELINE_TRAIN_PARAMS = dict(
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
)

DEFAULT_TREES = 1200
DEFAULT_MIN_ROWS = 500
DEFAULT_MIN_DAYS = 3
CV_FOLDS = 5
CV_EARLY_STOPPING = 50
CV_EVAL_METRIC = "mae"

# Candidate parameter sets for 5-fold CV tuning.
CV_PARAM_CANDIDATES = [
    dict(
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
    ),
    dict(
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
    ),
    dict(
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
    ),
    dict(
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
    ),
    dict(
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
    ),
    dict(
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
    ),
    dict(
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
    ),
    dict(
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XGBoost pick-time models for all WorkCodes"
    )
    parser.add_argument("warehouse", help="Warehouse code e.g. OE, OF, RT")
    parser.add_argument("--data_dir", default="training_data")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument(
        "--sequenced",
        action="store_true",
        help="Train with sequence features (requires distance matrix)",
    )
    parser.add_argument("--trees", type=int, default=DEFAULT_TREES)
    parser.add_argument("--min_rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--min_days", type=int, default=DEFAULT_MIN_DAYS)
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Enable 5-fold CV parameter tuning before final training.",
    )
    return parser.parse_args()


def discover_workcodes(data_dir, warehouse):
    """Read the parquet and return sorted list of unique WorkCodes."""
    df = load_parquet(data_dir, warehouse)
    df["WorkCode"] = (
        df["WorkCode"]
        .astype(str)
        .apply(lambda x: x.split(".")[0] if isinstance(x, str) else x)
    )
    wcs = df["WorkCode"].dropna().unique().tolist()
    wcs = [w for w in wcs if w.lower() != "nan"]
    return sorted(wcs)


def tune_xgb_params_cv(dtrain, num_boost_round, logger):
    """
    Run 5-fold CV over a small candidate set and return the best params.
    """
    best_params = None
    best_rounds = None
    best_score = float("inf")

    logger.info(
        f"  CV tuning: {CV_FOLDS}-fold, metric={CV_EVAL_METRIC}, "
        f"candidates={len(CV_PARAM_CANDIDATES)}"
    )

    for i, candidate in enumerate(CV_PARAM_CANDIDATES, start=1):
        params = {**XGB_FIXED_PARAMS, **candidate}
        cv = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            nfold=CV_FOLDS,
            stratified=False,
            metrics=(CV_EVAL_METRIC,),
            early_stopping_rounds=CV_EARLY_STOPPING,
            seed=2026,
            shuffle=True,
            verbose_eval=False,
        )

        metric_col = f"test-{CV_EVAL_METRIC}-mean"
        score = float(cv[metric_col].min())
        rounds = int(cv[metric_col].idxmin()) + 1
        logger.info(
            f"    CV candidate {i:02d}: best_{CV_EVAL_METRIC}={score:.4f}, "
            f"best_rounds={rounds}, params={candidate}"
        )

        if score < best_score:
            best_score = score
            best_rounds = rounds
            best_params = params

    return best_params, int(best_rounds), float(best_score)


def main():
    args = parse_args()
    warehouse = args.warehouse.upper()

    setup_logging("model_training", warehouse)
    logger = logging.getLogger(__name__)

    logger.info(f"{'='*60}")
    logger.info(f"MODEL TRAINING — {warehouse}")
    logger.info(f"  sequenced  = {args.sequenced}")
    logger.info(f"  trees      = {args.trees}")
    logger.info(f"  min_rows   = {args.min_rows}")
    logger.info(f"  min_days   = {args.min_days}")
    logger.info(f"  cv_tuning  = {args.cv}")
    logger.info(f"{'='*60}")

    workcodes = discover_workcodes(args.data_dir, warehouse)
    logger.info(f"WorkCodes found: {workcodes}")

    # meta.pkl — populated per WorkCode and saved once at the end
    meta = {
        "workcodes": [],
        "sequenced": args.sequenced,
        "train_columns": {},
        "encodings": {},  # top_aisles, top_uoms, product_tiers per WC
        "worker_effects": {},
        "level_thresholds": {},
        "level_medians": {},
        "mae_history": {},
        "baseline_mae": {},
        "seen_dates": {},  # unique YYYY-MM-DD dates observed per WC
        "days_trained": {},  # len(seen_dates[wc]); kept for convenience
        "xgb_params": {},  # tuned params per WC from 5-fold CV
        "xgb_best_rounds": {},  # best boosting rounds from CV per WC
        "xgb_cv_mae": {},  # best CV MAE per WC
    }

    trained_wcs = []
    skipped_wcs = []

    for wc in workcodes:
        logger.info(f"\n{'-'*50}")
        logger.info(f"WorkCode {wc}")

        try:
            df, X, y, encodings = load_and_engineer(
                data_dir=args.data_dir,
                warehouse=warehouse,
                wc=wc,
                sequenced=args.sequenced,
                encodings=None,  # compute from data (training mode)
                data_dir_ref=args.data_dir,
            )
        except Exception as e:
            logger.warning(f"  Could not load WC {wc}: {e} — skipping")
            skipped_wcs.append(wc)
            continue

        if len(df) < args.min_rows:
            logger.info(
                f"  Only {len(df):,} rows — below min_rows={args.min_rows}. "
                "Skipping."
            )
            skipped_wcs.append(wc)
            continue

        unique_days = int(df["Timestamp"].dropna().dt.date.nunique())
        if unique_days < args.min_days:
            logger.warning(
                f"  Only {unique_days} unique day(s) — below "
                f"min_days={args.min_days}. Continuing training, but "
                "performance may be unstable. Recommended: 3+ days."
            )

        logger.info(f"  Rows: {len(df):,}")

        # ── Worker effects ────────────────────────────────────────────────────
        logger.info("  Estimating worker effects ...")
        effects = estimate_worker_effects(df)
        effects_levelled, thresholds, level_medians = compute_worker_levels(effects)

        df = df.merge(
            effects_levelled[["UserID", "worker_effect"]], on="UserID", how="left"
        )
        df["worker_effect"] = df["worker_effect"].fillna(0.0)

        # Add worker_effect to feature matrix
        X["worker_effect"] = df["worker_effect"].values
        train_columns = X.columns.tolist()

        dtrain = xgb.DMatrix(X, label=y)

        # ── Tune (optional) and train ────────────────────────────────────────
        t0 = time.time()
        if args.cv:
            try:
                best_params, best_rounds, best_cv_mae = tune_xgb_params_cv(
                    dtrain=dtrain,
                    num_boost_round=args.trees,
                    logger=logger,
                )
            except Exception as e:
                logger.warning(
                    f"  CV tuning failed ({e}) — falling back to baseline params."
                )
                best_params = {
                    **XGB_FIXED_PARAMS,
                    **XGB_BASELINE_TRAIN_PARAMS,
                }
                best_rounds = args.trees
                best_cv_mae = float("nan")

            logger.info(
                f"  Training final model with CV params "
                f"(rounds={best_rounds}, cv_{CV_EVAL_METRIC}={best_cv_mae:.4f}) ..."
            )
        else:
            best_params = {
                **XGB_FIXED_PARAMS,
                **XGB_BASELINE_TRAIN_PARAMS,
            }
            best_rounds = args.trees
            best_cv_mae = float("nan")
            logger.info(
                "  CV disabled — training with baseline params "
                f"(rounds={best_rounds})."
            )
        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=best_rounds,
            verbose_eval=False,
        )
        elapsed = time.time() - t0
        logger.info(f"  Done in {elapsed:.1f}s")

        # ── Save model ────────────────────────────────────────────────────────
        model_file = save_model(
            model, args.models_dir, warehouse, wc, sequenced=args.sequenced
        )
        logger.info(f"  Model saved: {model_file}")

        # ── Populate meta ─────────────────────────────────────────────────────
        meta["workcodes"].append(wc)
        meta["train_columns"][wc] = train_columns
        meta["encodings"][wc] = {
            "top_aisles": encodings["top_aisles"],
            "top_uoms": encodings["top_uoms"],
            "product_tiers": encodings["product_tiers"],
        }
        meta["worker_effects"][wc] = effects_levelled
        meta["level_thresholds"][wc] = thresholds
        meta["level_medians"][wc] = level_medians
        meta["mae_history"][wc] = []
        meta["baseline_mae"][wc] = None  # set on first update run
        meta["xgb_params"][wc] = best_params
        meta["xgb_best_rounds"][wc] = best_rounds
        meta["xgb_cv_mae"][wc] = best_cv_mae
        seen_dates = sorted({ts.date().isoformat() for ts in df["Timestamp"].dropna()})
        meta["seen_dates"][wc] = seen_dates
        meta["days_trained"][wc] = len(seen_dates)

        trained_wcs.append(wc)
        logger.info(
            f"  Workers: {len(effects_levelled):,}  |  "
            f"Levels — "
            + "  ".join(
                f"L{lv}: {(effects_levelled['level']==lv).sum()}" for lv in range(1, 6)
            )
            + f"  |  Days seen: {meta['days_trained'][wc]}"
        )

    # ── Save meta ─────────────────────────────────────────────────────────────
    meta_file = save_meta(meta, args.models_dir, warehouse)
    logger.info(f"\nmeta.pkl saved: {meta_file}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE — {warehouse}")
    logger.info(f"{'='*60}")
    logger.info(f"  WorkCodes trained:  {trained_wcs}")
    if skipped_wcs:
        logger.info(f"  WorkCodes skipped:  {skipped_wcs}")
    logger.info(f"  sequenced:          {args.sequenced}")
    logger.info(f"  Trees per model:    {args.trees}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
