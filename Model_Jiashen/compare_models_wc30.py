from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

# Reduce runtime stderr noise from parallelism / low-level logging.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("GLOG_minloglevel", "3")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# Suppress estimator convergence chatter so output is just the final table.
warnings.filterwarnings("ignore")


def load_and_prepare_data(repo_root: Path, workcode: str = "30", max_time_delta: float = 300.0):
    deliverables_dir = repo_root / "deliverables"
    if str(deliverables_dir) not in sys.path:
        sys.path.insert(0, str(deliverables_dir))

    from utils.feature_engineer import apply_features, compute_encodings, make_X

    parquet_path = deliverables_dir / "training_data" / "OE" / "OE_Processed.parquet"
    df = pd.read_parquet(parquet_path)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["WorkCode"] = (
        df["WorkCode"].astype(str).apply(lambda x: x.split(".")[0] if isinstance(x, str) else x)
    )

    df = df[df["WorkCode"] == str(workcode)].copy()
    df = df.dropna(subset=["Timestamp", "Time_Delta_sec"]).copy()
    df = df[(df["Time_Delta_sec"] > 0) & (df["Time_Delta_sec"] <= max_time_delta)].copy()

    if df.empty:
        raise ValueError("No rows left after filtering.")

    df["Date"] = df["Timestamp"].dt.date
    last_day = df["Date"].max()

    train_df = df[df["Date"] < last_day].copy()
    test_df = df[df["Date"] == last_day].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty. Need data before and on the last day.")

    encodings = compute_encodings(train_df)

    train_fe = apply_features(
        train_df,
        top_aisles=encodings["top_aisles"],
        top_uoms=encodings["top_uoms"],
        product_tiers=encodings["product_tiers"],
        sequenced=False,
    )
    test_fe = apply_features(
        test_df,
        top_aisles=encodings["top_aisles"],
        top_uoms=encodings["top_uoms"],
        product_tiers=encodings["product_tiers"],
        sequenced=False,
    )

    X_train = make_X(train_fe, sequenced=False)
    X_test = make_X(test_fe, sequenced=False, train_columns=X_train.columns.tolist())

    y_train = train_fe["Time_Delta_sec"].astype(float).values
    y_test = test_fe["Time_Delta_sec"].astype(float).values

    return X_train, X_test, y_train, y_test


def build_models(random_state: int = 2026):
    return {
        "RF": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=1,
        ),
        "xgb": XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=1.3,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
        ),
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "lasso_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.001, max_iter=5000, random_state=random_state)),
        ]),
        "neural_net": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                learning_rate_init=0.001,
                max_iter=400,
                random_state=random_state,
            )),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=15)),
        ]),
    }


def main():
    t0 = time.time()
    print("[1/2] Loading and preparing data...", flush=True)
    repo_root = Path(__file__).resolve().parent.parent
    X_train, X_test, y_train, y_test = load_and_prepare_data(repo_root=repo_root)
    print(
        f"      Train rows: {len(y_train):,} | Test rows: {len(y_test):,}",
        flush=True,
    )

    rows = []
    models = build_models()
    print(f"[2/2] Training {len(models)} models...", flush=True)
    for idx, (name, model) in enumerate(models.items(), start=1):
        m0 = time.time()
        print(f"      ({idx}/{len(models)}) Fitting {name} ...", flush=True)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rows.append({"Model": name, "MAE_per_task": float(mae)})
        print(
            f"      ({idx}/{len(models)}) Done {name}: "
            f"MAE={mae:.4f}, elapsed={time.time()-m0:.1f}s",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values("MAE_per_task", ascending=True)
    result["MAE_per_task"] = result["MAE_per_task"].round(4)

    print(f"Completed in {time.time()-t0:.1f}s", flush=True)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
