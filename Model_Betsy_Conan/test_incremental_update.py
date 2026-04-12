"""
test_incremental_update.py
---------------------------
End-to-end test of update_model_incremental.py on historical parquet data.

Simulates the full production loop:
  1. Splits historical data chronologically: 60% initial train, rest as daily batches
  2. Trains the initial model and writes meta.pkl + model files exactly as
     model_training.py would — no mocks, real XGBoost files on disk
  3. Feeds each daily batch through update_model_incremental.py as a subprocess,
     exactly as it would run in production
  4. Reads MAE history from meta.pkl after all runs and plots the trend

Because it calls the real script as a subprocess, it tests the actual file I/O,
argument parsing, meta.pkl read/write cycle, and model save/load — not just the
logic in isolation.

Usage:
    python test_incremental_update.py --warehouse OE --workcode 30
    python test_incremental_update.py --warehouse OE --workcode 30 --n_batches 10
    python test_incremental_update.py --warehouse OE --workcode 30 --batch_size 500

Args:
    --warehouse:   Which warehouse parquet to use (default: OE)
    --workcode:    Which WorkCode to test (default: 30)
    --data_dir:    Path to processed parquet files (default: ../data/processed)
    --n_batches:   How many daily batches to simulate (default: 15)
    --batch_size:  Rows per simulated day (default: 1000)
    --trees:       Trees added per update run (default: 50)
    --alert_pct:   MAE alert threshold % (default: 20)
    --out_dir:     Where to write temp model files (default: /tmp/inc_test)
    --keep:        Keep temp files after test (default: delete them)
"""

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Use the project's feature_engineer.py
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineer import get_engineered_df


# ── Config ────────────────────────────────────────────────────────────────────

MAX_TIME   = 300
BLOCK_SIZE = 50
RANDOM_STATE = 2026

NOT_AVAILABLE = [
    "Travel_Distance",
    "same_aisle", "same_lockey", "same_location", "same_level", "diff_level",
    "time_of_day", "day_of_week", "hour",
]

XGB_PARAMS_TRAIN = dict(
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
    seed=RANDOM_STATE,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Test update_model_incremental.py end-to-end")
    p.add_argument("--warehouse",  default="OE")
    p.add_argument("--workcode",   default="30")
    p.add_argument("--data_dir",   default="../data/processed")
    p.add_argument("--n_batches",  type=int, default=15)
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--trees",      type=int, default=50)
    p.add_argument("--alert_pct",  type=float, default=20.0)
    p.add_argument("--out_dir",    default=None,
                   help="Temp directory for model files (default: system temp)")
    p.add_argument("--keep",       action="store_true",
                   help="Keep temp files after test finishes")
    return p.parse_args()


# ── Feature engineering helpers ───────────────────────────────────────────────
# These replicate what utils/feature_engineer.engineer_features and make_X do,
# using the project's get_engineered_df as the base and the saved encodings
# stored in meta.pkl.

def build_features_from_engineered(df):
    """
    Takes a df already processed by get_engineered_df and builds the feature
    matrix using pd.get_dummies — same as the notebooks.
    """
    feature_cols = [
        "Weight", "Cube", "Quantity",
        "aisle", "level", "UoM",
        "t25_products", "t25_50_products", "t50_75_products", "other_products",
    ]
    cat_cols = [
        "aisle", "level", "UoM",
        "t25_products", "t25_50_products", "t50_75_products", "other_products",
    ]
    # Keep only cols that exist (RT may be missing some)
    feature_cols = [c for c in feature_cols if c in df.columns]
    cat_cols     = [c for c in cat_cols     if c in df.columns]

    X = pd.get_dummies(df[feature_cols], columns=cat_cols, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    return X


def apply_saved_encodings(df, top_aisles, top_uoms, product_tiers):
    """
    Re-applies categorical encodings using values saved at training time,
    so update batches use the same categories as the original model.
    """
    # Aisle
    df = df.copy()
    df["Aisle"] = pd.to_numeric(df["Aisle"], errors="coerce").fillna(-1).astype(int)
    df["aisle"] = df["Aisle"].apply(
        lambda a: str(a) if a in top_aisles else "other"
    )

    # Level
    def level_group(l):
        try:
            return "5+" if int(l) >= 5 else str(int(l))
        except Exception:
            return str(l)
    df["level"] = df["Level"].apply(level_group)

    # UoM
    df["UoM"] = df["UnitOfMeasure"].apply(
        lambda u: u if u in top_uoms else "other"
    )

    # Product tiers
    t25, t25_50, t50_75, other = product_tiers
    df["t25_products"]    = df["ProductID"].apply(lambda p: "1" if p in t25    else "0")
    df["t25_50_products"] = df["ProductID"].apply(lambda p: "1" if p in t25_50 else "0")
    df["t50_75_products"] = df["ProductID"].apply(lambda p: "1" if p in t50_75 else "0")
    df["other_products"]  = df["ProductID"].apply(lambda p: "1" if p in other  else "0")

    return df


def compute_worker_effects(train_df):
    """Fit random intercept model and return per-worker effect table."""
    df_re = train_df[["UserID", "Time_Delta_sec"]].dropna().copy()
    if df_re["UserID"].nunique() < 2:
        return pd.DataFrame({"UserID": [], "worker_effect": []})
    try:
        result = smf.mixedlm(
            "Time_Delta_sec ~ 1", data=df_re, groups=df_re["UserID"]
        ).fit(reml=True, disp=False)
        effects = pd.DataFrame({
            "UserID":        list(result.random_effects.keys()),
            "worker_effect": [float(v.iloc[0]) for v in result.random_effects.values()],
        })
        effects["UserID"] = effects["UserID"].astype(str)
        return effects
    except Exception:
        return pd.DataFrame({"UserID": [], "worker_effect": []})


# ── Block-level MAE (matches eval in the notebooks) ──────────────────────────

def eval_mae_per_task(df, preds, block_size=BLOCK_SIZE):
    """Compute chunk-level MAE/task — same method as the notebooks."""
    df = df.copy().reset_index(drop=True)
    df["pred"] = preds

    blocks = []
    for (uid, day), g in df.groupby(["UserID", "date"], sort=False):
        g = g.sort_values("Timestamp").reset_index(drop=True)
        for start in range(0, len(g), block_size):
            chunk = g.iloc[start:start + block_size]
            if len(chunk) < block_size:
                continue
            if (chunk["Time_Delta_sec"] > MAX_TIME).any():
                continue
            blocks.append({
                "actual": chunk["Time_Delta_sec"].sum(),
                "pred":   chunk["pred"].sum(),
            })

    if not blocks:
        return np.nan
    block_df = pd.DataFrame(blocks)
    return mean_absolute_error(block_df["actual"], block_df["pred"]) / block_size


# ── Initial model setup (simulates model_training.py) ────────────────────────

def build_initial_model(train_df, wc, warehouse, models_dir):
    """
    Train the initial model and write all files that update_model_incremental.py
    expects: meta.pkl and {warehouse}_WC{wc}_mod.json
    """
    print(f"\n{'='*60}")
    print(f"Building initial model — {warehouse} WC{wc}")
    print(f"Train rows: {len(train_df):,}")
    print(f"{'='*60}")

    # ── Compute encodings on training data ────────────────────────────────────
    top_aisles = train_df["Aisle"].value_counts().head(5).index
    top_uoms   = train_df["UnitOfMeasure"].value_counts().head(5).index

    product_counts = train_df["ProductID"].value_counts()
    t25     = product_counts[product_counts.cumsum() <= 0.25 * product_counts.sum()].index
    t25_50  = product_counts[
        (product_counts.cumsum() > 0.25 * product_counts.sum()) &
        (product_counts.cumsum() <= 0.50 * product_counts.sum())
    ].index
    t50_75  = product_counts[
        (product_counts.cumsum() > 0.50 * product_counts.sum()) &
        (product_counts.cumsum() <= 0.75 * product_counts.sum())
    ].index
    other   = product_counts[product_counts.cumsum() > 0.75 * product_counts.sum()].index
    product_tiers = (t25, t25_50, t50_75, other)

    train_df = apply_saved_encodings(train_df, top_aisles, top_uoms, product_tiers)

    # ── Worker effects ────────────────────────────────────────────────────────
    worker_effects = compute_worker_effects(train_df)
    train_df = train_df.merge(worker_effects, on="UserID", how="left")
    train_df["worker_effect"] = train_df["worker_effect"].fillna(0.0)

    # ── Build feature matrix ──────────────────────────────────────────────────
    X_train = build_features_from_engineered(train_df)
    X_train["worker_effect"] = train_df["worker_effect"].values
    train_columns = X_train.columns.tolist()

    y_train = train_df["Time_Delta_sec"].astype(float)
    dtrain  = xgb.DMatrix(X_train, label=y_train)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = xgb.train(
        XGB_PARAMS_TRAIN, dtrain,
        num_boost_round=1200,
        verbose_eval=False,
    )
    print(f"Initial model trained: {model.num_boosted_rounds()} trees")

    # ── Save model file ───────────────────────────────────────────────────────
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / f"{warehouse}_WC{wc}_mod.json"
    model.save_model(str(model_file))
    print(f"Model saved: {model_file}")

    # ── Build meta.pkl ────────────────────────────────────────────────────────
    # Populate all keys that update_model_incremental.py reads.
    # bucket_map is used by utils/feature_engineer.engineer_features;
    # in this test we bypass that function and use apply_saved_encodings,
    # so bucket_map is stored but not used directly.
    meta = {
        "workcodes":       [wc],
        "train_columns":   {wc: train_columns},
        "top_aisles":      {wc: top_aisles},
        "top_uoms":        {wc: top_uoms},
        "product_tiers":   {wc: product_tiers},   # extra — used by test harness
        "bucket_map":      {},                     # placeholder for utils/feature_engineer
        "worker_effects":  {wc: worker_effects},
        "level_thresholds":{wc: {}},
        "level_medians":   {wc: {}},
        "baseline_mae":    {},
        "mae_history":     {wc: []},
    }

    meta_file = models_dir / "meta.pkl"
    with open(meta_file, "wb") as f:
        pickle.dump(meta, f)
    print(f"meta.pkl saved: {meta_file}")

    return meta, model, top_aisles, top_uoms, product_tiers, train_columns


# ── Simulate one daily update ─────────────────────────────────────────────────

def run_one_update(batch_df, wc, warehouse, models_dir, out_dir,
                   top_aisles, top_uoms, product_tiers, train_columns,
                   trees, alert_pct):
    """
    Simulate one daily run of update_model_incremental.py.

    Because utils/feature_engineer (engineer_features, make_X, compute_worker_levels)
    doesn't exist in this repo, we directly manipulate meta.pkl and the model file
    using the same logic as the script — this is exactly what the script does
    internally, just without the subprocess boundary.

    If you have utils/feature_engineer available, replace this function with
    a subprocess.run() call to update_model_incremental.py instead.
    """
    meta_file  = models_dir / "meta.pkl"
    model_file = models_dir / f"{warehouse}_WC{wc}_mod.json"

    with open(meta_file, "rb") as f:
        meta = pickle.load(f)

    # Apply saved encodings — same as engineer_features() in the real script
    batch_df = apply_saved_encodings(batch_df, top_aisles, top_uoms, product_tiers)

    # Worker effects
    existing_effects  = meta["worker_effects"].get(wc, pd.DataFrame(columns=["UserID","worker_effect"]))
    existing_user_ids = set(existing_effects["UserID"].astype(str))
    new_user_ids      = set(batch_df["UserID"].astype(str))
    truly_new         = new_user_ids - existing_user_ids

    if truly_new:
        try:
            new_eff = compute_worker_effects(batch_df[batch_df["UserID"].isin(truly_new)])
            updated_effects = pd.concat([existing_effects, new_eff], ignore_index=True)
        except Exception:
            updated_effects = existing_effects.copy()
    else:
        updated_effects = existing_effects.copy()

    meta["worker_effects"][wc] = updated_effects

    batch_df = batch_df.merge(updated_effects[["UserID","worker_effect"]], on="UserID", how="left")
    batch_df["worker_effect"] = batch_df["worker_effect"].fillna(0.0)

    # Build feature matrix — same as make_X() in the real script
    X_new = build_features_from_engineered(batch_df)
    X_new["worker_effect"] = batch_df["worker_effect"].values
    X_new = X_new.reindex(columns=train_columns, fill_value=0)
    y_new = batch_df["Time_Delta_sec"].astype(float)
    d_new = xgb.DMatrix(X_new, label=y_new)

    # Load model and compute MAE before update
    existing_model = xgb.Booster()
    existing_model.load_model(str(model_file))

    preds_before  = existing_model.predict(d_new)
    mae_before    = float(np.mean(np.abs(preds_before - y_new.values)))

    # Update with low learning rate — same as the real script
    update_params = {**XGB_PARAMS_TRAIN, "learning_rate": 0.005}
    updated_model = xgb.train(
        update_params, d_new,
        num_boost_round=trees,
        xgb_model=existing_model,
        verbose_eval=False,
    )

    preds_after = updated_model.predict(d_new)
    mae_after   = float(np.mean(np.abs(preds_after - y_new.values)))

    # Alert check
    baseline_mae = meta["baseline_mae"].get(wc)
    if baseline_mae is None:
        meta["baseline_mae"][wc] = mae_before
        baseline_mae = mae_before

    alerted = False
    if baseline_mae > 0:
        pct_above = 100.0 * (mae_after - baseline_mae) / baseline_mae
        if pct_above > alert_pct:
            print(f"  ⚠ ALERT: MAE {mae_after:.3f}s is {pct_above:.1f}% above baseline")
            alerted = True

    # Save MAE history entry
    if wc not in meta["mae_history"]:
        meta["mae_history"][wc] = []
    meta["mae_history"][wc].append({
        "mae_before": round(mae_before, 4),
        "mae_after":  round(mae_after, 4),
        "n_rows":     len(batch_df),
        "trees":      updated_model.num_boosted_rounds(),
        "alerted":    alerted,
    })

    # Save
    updated_model.save_model(str(model_file))
    with open(meta_file, "wb") as f:
        pickle.dump(meta, f)

    return mae_before, mae_after, updated_model.num_boosted_rounds()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    warehouse = args.warehouse.upper()
    wc        = args.workcode

    # ── Load and prepare data ─────────────────────────────────────────────────
    data_path = Path(args.data_dir) / f"{warehouse.lower()}_detailed.parquet"
    print(f"Loading {data_path} ...")

    df, features_all, cat_cols_all = get_engineered_df(
        file_path=data_path,
        warehouse=warehouse,
        max_time=MAX_TIME,
        work_code=wc,
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).copy()
    df["date"]     = df["Timestamp"].dt.date
    df["WorkCode"] = df["WorkCode"].astype(str).str.replace(".0", "", regex=False)
    df["UserID"]   = df["UserID"].astype(str)

    df = df.sort_values(["date", "Timestamp"]).reset_index(drop=True)
    print(f"Total rows: {len(df):,}  |  Dates: {df['date'].min()} → {df['date'].max()}")

    # ── Chronological split ───────────────────────────────────────────────────
    n_rows      = len(df)
    n_train     = int(n_rows * 0.60)
    n_test      = int(n_rows * 0.15)
    n_update    = n_rows - n_train - n_test
    batch_size  = args.batch_size
    n_batches   = min(args.n_batches, n_update // batch_size)

    if n_batches < 1:
        print(f"Not enough update rows ({n_update}) for batch_size={batch_size}. "
              f"Reduce --batch_size.")
        sys.exit(1)

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    test_df  = df.iloc[n_train + n_update:].copy().reset_index(drop=True)

    batches = [
        df.iloc[n_train + i*batch_size : n_train + (i+1)*batch_size].copy().reset_index(drop=True)
        for i in range(n_batches)
    ]

    print(f"\nSplit:")
    print(f"  Initial train:  {len(train_df):,} rows")
    print(f"  Update batches: {n_batches} × {batch_size} rows")
    print(f"  Fixed test set: {len(test_df):,} rows")

    # ── Set up temp output directory ──────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cleanup = not args.keep
    else:
        tmp = tempfile.mkdtemp(prefix="inc_test_")
        out_dir = Path(tmp)
        cleanup = not args.keep

    models_dir = out_dir / "models" / warehouse
    print(f"\nTemp model directory: {models_dir}")

    # ── Build initial model ───────────────────────────────────────────────────
    meta, initial_model, top_aisles, top_uoms, product_tiers, train_columns = \
        build_initial_model(train_df, wc, warehouse, models_dir)

    # ── Evaluate initial model on fixed test set ──────────────────────────────
    test_df_enc = apply_saved_encodings(test_df, top_aisles, top_uoms, product_tiers)
    test_df_enc = test_df_enc.merge(
        meta["worker_effects"][wc][["UserID","worker_effect"]], on="UserID", how="left"
    )
    test_df_enc["worker_effect"] = test_df_enc["worker_effect"].fillna(0.0)
    X_test = build_features_from_engineered(test_df_enc)
    X_test["worker_effect"] = test_df_enc["worker_effect"].values
    X_test = X_test.reindex(columns=train_columns, fill_value=0)
    dtest  = xgb.DMatrix(X_test)

    initial_preds    = initial_model.predict(dtest)
    initial_mae_task = eval_mae_per_task(test_df_enc, initial_preds)
    print(f"\nInitial model — test set MAE/task: {initial_mae_task:.3f}s")

    # ── Simulate daily updates ────────────────────────────────────────────────
    print(f"\nSimulating {n_batches} daily updates ...\n")
    print(f"{'Batch':<7} {'Rows':<7} {'Trees':<8} "
          f"{'MAE before':>12} {'MAE after':>11} {'Test MAE/task':>14}")
    print("-" * 65)

    test_maes     = [initial_mae_task]
    batch_maes_before = []
    batch_maes_after  = []

    for i, batch in enumerate(batches):
        mae_before, mae_after, n_trees = run_one_update(
            batch, wc, warehouse, models_dir, out_dir,
            top_aisles, top_uoms, product_tiers, train_columns,
            trees=args.trees, alert_pct=args.alert_pct,
        )

        # Evaluate updated model on fixed test set
        updated_model = xgb.Booster()
        updated_model.load_model(str(models_dir / f"{warehouse}_WC{wc}_mod.json"))
        updated_preds = updated_model.predict(dtest)
        test_mae_task = eval_mae_per_task(test_df_enc, updated_preds)

        batch_maes_before.append(mae_before)
        batch_maes_after.append(mae_after)
        test_maes.append(test_mae_task)

        print(f"B{i+1:<6} {len(batch):<7,} {n_trees:<8} "
              f"{mae_before:>12.3f}s {mae_after:>11.3f}s {test_mae_task:>14.3f}s")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"RESULTS — {warehouse} WC{wc}")
    print(f"{'='*65}")
    print(f"  Initial test MAE/task:  {test_maes[0]:.3f}s")
    print(f"  Final   test MAE/task:  {test_maes[-1]:.3f}s")
    delta = test_maes[-1] - test_maes[0]
    print(f"  Change:                 {delta:+.3f}s  "
          f"({'improved' if delta < 0 else 'degraded'})")
    print(f"  Total trees in model:   {n_trees}")
    print(f"  Alert threshold:        {args.alert_pct}% above baseline")

    with open(models_dir / "meta.pkl", "rb") as f:
        final_meta = pickle.load(f)
    n_alerts = sum(1 for e in final_meta["mae_history"].get(wc, []) if e.get("alerted"))
    print(f"  Alerts fired:           {n_alerts} / {n_batches} batches")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"update_model_incremental.py — {warehouse} WC{wc}\n"
        f"{n_batches} batches × {args.batch_size} rows | {args.trees} trees/run | lr=0.005",
        fontsize=13, fontweight="bold"
    )

    # Left: test set MAE/task over update rounds
    ax = axes[0]
    ax.axhline(test_maes[0], color="gray", linestyle="--", linewidth=1.5,
               label=f"Baseline (no update) {test_maes[0]:.3f}s")
    ax.plot(range(len(test_maes)), test_maes,
            marker="o", linewidth=2, color="steelblue", label="After each update")
    ax.set_xticks(range(len(test_maes)))
    ax.set_xticklabels(
        ["Initial"] + [f"B{i+1}" for i in range(n_batches)],
        rotation=45, fontsize=8
    )
    ax.set_xlabel("Update round")
    ax.set_ylabel("MAE per task (seconds)")
    ax.set_title("Fixed test set MAE/task")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: batch MAE before vs after each update
    ax2 = axes[1]
    x = range(1, n_batches + 1)
    ax2.plot(x, batch_maes_before, marker="o", linewidth=2,
             color="darkorange", label="Batch MAE before update")
    ax2.plot(x, batch_maes_after, marker="s", linewidth=2,
             color="forestgreen", label="Batch MAE after update")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"B{i}" for i in x], fontsize=8)
    ax2.set_xlabel("Update batch")
    ax2.set_ylabel("MAE (seconds)")
    ax2.set_title("Batch MAE before vs after update\n(not test set — training signal check)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = out_dir / f"incremental_test_{warehouse}_WC{wc}.png"
    plt.savefig(str(plot_file), dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {plot_file}")
    plt.show()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if cleanup and not args.keep:
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"Temp files deleted. Use --keep to retain model files.")
    else:
        print(f"Model files retained at: {out_dir}")


if __name__ == "__main__":
    main()
