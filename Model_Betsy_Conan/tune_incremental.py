"""
tune_incremental.py
--------------------
Sweeps learning rate and trees-per-update to find the best combination
for the naive incremental update strategy used in update_model_incremental.py.

Uses the same data pipeline, split logic, and evaluation method as
incremental_learning_test.ipynb so results are directly comparable.

Usage:
    python tune_incremental.py
    python tune_incremental.py --warehouse OE --workcode 30
    python tune_incremental.py --warehouse OE --workcode 30 --n_batches 10

Args:
    --warehouse:   Warehouse code (default: OE)
    --workcode:    WorkCode to tune on (default: 30)
    --data_dir:    Path to processed parquets (default: ../data/processed)
    --n_batches:   Number of update batches to simulate (default: 10)
    --batch_size:  Rows per batch (default: auto — 25% of data / n_batches)
    --train_frac:  Fraction of rows for initial training (default: 0.60)
    --test_frac:   Fraction of rows for fixed test set (default: 0.15)
    --trees:       Comma-separated list of trees-per-update to try
                   (default: 25,50,100,150,200)
    --lrs:         Comma-separated list of learning rates to try
                   (default: 0.001,0.003,0.005,0.01,0.02)
    --init_trees:  Trees for initial model (default: 1200)
    --out:         Output directory for results and plots (default: tune_results)
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineer import get_engineered_df

MAX_TIME   = 300
BLOCK_SIZE = 50
RANDOM_STATE = 2026

NOT_AVAILABLE = [
    'Travel_Distance',
    'same_aisle', 'same_lockey', 'same_location', 'same_level', 'diff_level',
    'time_of_day', 'day_of_week', 'hour',
]

XGB_PARAMS_BASE = dict(
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective='reg:tweedie',
    tweedie_variance_power=1.3,
    tree_method='hist',
    seed=RANDOM_STATE,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep learning rate and tree count for incremental updates"
    )
    p.add_argument("--warehouse",   default="OE")
    p.add_argument("--workcode",    default="30")
    p.add_argument("--data_dir",    default="../data/processed")
    p.add_argument("--n_batches",   type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--train_frac",  type=float, default=0.60)
    p.add_argument("--test_frac",   type=float, default=0.15)
    p.add_argument("--trees",       default="25,50,100,150,200")
    p.add_argument("--lrs",         default="0.001,0.003,0.005,0.01,0.02")
    p.add_argument("--init_trees",  type=int,   default=1200)
    p.add_argument("--out",         default="tune_results")
    return p.parse_args()


# ── Data helpers (same as incremental_learning_test.ipynb) ────────────────────

def load_data(data_dir, warehouse, workcode):
    path = Path(data_dir) / f"{warehouse.lower()}_detailed.parquet"
    df, features_all, cat_cols_all = get_engineered_df(
        file_path=path,
        warehouse=warehouse,
        max_time=MAX_TIME,
        work_code=workcode,
    )
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).copy()
    df['date']     = df['Timestamp'].dt.date
    df['WorkCode'] = df['WorkCode'].astype(str).str.replace('.0', '', regex=False)
    features = [f for f in features_all if f not in NOT_AVAILABLE]
    cat_cols = [c for c in cat_cols_all if c not in NOT_AVAILABLE]
    return df, features, cat_cols


def make_X(df, features, cat_cols, train_columns=None):
    X = pd.get_dummies(df[features], columns=cat_cols, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    if train_columns is not None:
        X = X.reindex(columns=train_columns, fill_value=0)
    return X


def eval_blocks(test_df, preds, train_columns):
    test_df = test_df.copy().reset_index(drop=True)
    d = test_df.sort_values(['UserID', 'Timestamp']).copy()
    blocks, block_rows = [], []
    for (uid, day), g in d.groupby(['UserID', 'date'], sort=False):
        g = g.sort_values('Timestamp').reset_index().rename(
            columns={'index': 'orig_index'}
        ).copy()
        for start in range(0, len(g), BLOCK_SIZE):
            chunk = g.iloc[start:start + BLOCK_SIZE].copy()
            if len(chunk) < BLOCK_SIZE:
                continue
            if chunk['WorkCode'].nunique() != 1:
                continue
            if (chunk['Time_Delta_sec'] > MAX_TIME).any():
                continue
            block_id = f'{uid}_{day}_{start // BLOCK_SIZE}'
            chunk['BlockID'] = block_id
            block_rows.append(chunk)
            blocks.append({
                'BlockID':     block_id,
                'actual_time': chunk['Time_Delta_sec'].sum(),
            })
    if not blocks:
        return np.nan, np.nan, 0
    block_df      = pd.DataFrame(blocks)
    block_rows_df = pd.concat(block_rows, ignore_index=True)

    temp = test_df.copy().reset_index().rename(columns={'index': 'orig_index'})
    temp['pred'] = preds
    block_rows_df = block_rows_df.merge(
        temp[['orig_index', 'pred']], on='orig_index', how='left'
    )
    block_pred = (
        block_rows_df.groupby('BlockID')
        .agg(actual_time=('Time_Delta_sec', 'sum'), pred=('pred', 'sum'))
        .reset_index()
    )
    mae = mean_absolute_error(block_pred['actual_time'], block_pred['pred'])
    r2  = r2_score(block_pred['actual_time'], block_pred['pred'])
    return mae / BLOCK_SIZE, r2, len(block_pred)


# ── Run one combination ───────────────────────────────────────────────────────

def run_combo(lr, n_trees, init_trees,
              train_df, batch_dfs, test_df,
              features, cat_cols, train_columns,
              dtrain, dtest):
    """
    Train initial model then simulate n_batches incremental updates.
    Returns list of MAE/task: [initial, after_B1, ..., after_BN].
    """
    # Initial model — always uses training lr=0.03
    init_params = {**XGB_PARAMS_BASE, 'learning_rate': 0.03}
    model = xgb.train(
        init_params, dtrain,
        num_boost_round=init_trees,
        verbose_eval=False,
    )

    maes = []
    mae0, _, _ = eval_blocks(test_df, model.predict(dtest), train_columns)
    maes.append(mae0)

    # Update params use the tuned lr
    update_params = {**XGB_PARAMS_BASE, 'learning_rate': lr}

    for batch_df in batch_dfs:
        X_b = make_X(batch_df, features, cat_cols, train_columns=train_columns)
        y_b = batch_df['Time_Delta_sec'].astype(float)
        d_b = xgb.DMatrix(X_b, label=y_b)

        model = xgb.train(
            update_params, d_b,
            num_boost_round=n_trees,
            xgb_model=model,
            verbose_eval=False,
        )
        mae, _, _ = eval_blocks(test_df, model.predict(dtest), train_columns)
        maes.append(mae)

    return maes


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    lr_grid    = [float(x) for x in args.lrs.split(',')]
    trees_grid = [int(x)   for x in args.trees.split(',')]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.warehouse} WC{args.workcode} ...")
    df, features, cat_cols = load_data(
        args.data_dir, args.warehouse, args.workcode
    )
    df = df.sort_values(['date', 'Timestamp']).reset_index(drop=True)
    n_rows = len(df)
    print(f"  {n_rows:,} rows")

    # ── Split ─────────────────────────────────────────────────────────────────
    n_train  = int(n_rows * args.train_frac)
    n_test   = int(n_rows * args.test_frac)
    n_update = n_rows - n_train - n_test
    n_batches = args.n_batches

    if args.batch_size:
        n_batch = args.batch_size
        n_batches = min(n_batches, n_update // n_batch)
    else:
        n_batch = max(1, n_update // n_batches)

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    batch_dfs = [
        df.iloc[n_train + i*n_batch : n_train + (i+1)*n_batch].copy().reset_index(drop=True)
        for i in range(n_batches)
    ]
    batch_dfs = [b for b in batch_dfs if len(b) > 0]
    n_batches = len(batch_dfs)
    test_df  = df.iloc[n_train + n_update:].copy().reset_index(drop=True)

    print(f"  Train: {len(train_df):,} | Batches: {n_batches} x ~{n_batch:,} | Test: {len(test_df):,}")

    X_train_init  = make_X(train_df, features, cat_cols)
    train_columns = X_train_init.columns.tolist()
    X_test        = make_X(test_df, features, cat_cols, train_columns=train_columns)
    y_test        = test_df['Time_Delta_sec'].astype(float)

    dtrain = xgb.DMatrix(X_train_init, label=train_df['Time_Delta_sec'].astype(float))
    dtest  = xgb.DMatrix(X_test)

    # ── Baseline (no updates) ─────────────────────────────────────────────────
    print("\nTraining baseline (no updates) ...")
    baseline_params = {**XGB_PARAMS_BASE, 'learning_rate': 0.03}
    baseline_model  = xgb.train(
        baseline_params, dtrain,
        num_boost_round=args.init_trees,
        verbose_eval=False,
    )
    baseline_mae, _, _ = eval_blocks(
        test_df, baseline_model.predict(dtest), train_columns
    )
    print(f"  Baseline MAE/task: {baseline_mae:.4f}s")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print(f"\nSweeping {len(lr_grid)} learning rates x {len(trees_grid)} tree counts "
          f"= {len(lr_grid)*len(trees_grid)} combinations ...")
    print(f"  Learning rates: {lr_grid}")
    print(f"  Trees/update:   {trees_grid}")
    print()

    all_results = {}   # (lr, trees) -> list of MAE/task
    summary_rows = []

    total = len(lr_grid) * len(trees_grid)
    done  = 0

    for lr, n_trees in product(lr_grid, trees_grid):
        label = f"lr={lr}  trees={n_trees}"
        maes  = run_combo(
            lr, n_trees, args.init_trees,
            train_df, batch_dfs, test_df,
            features, cat_cols, train_columns,
            dtrain, dtest,
        )
        all_results[(lr, n_trees)] = maes

        final_mae   = maes[-1]
        improvement = baseline_mae - final_mae
        stable      = max(maes[1:]) - min(maes[1:]) if len(maes) > 1 else 0

        summary_rows.append({
            'learning_rate': lr,
            'trees_per_update': n_trees,
            'initial_mae':  round(maes[0], 4),
            'final_mae':    round(final_mae, 4),
            'vs_baseline':  round(final_mae - baseline_mae, 4),
            'improvement':  round(improvement, 4),
            'volatility':   round(stable, 4),
        })

        done += 1
        direction = 'better' if improvement > 0 else 'worse'
        print(f"  [{done}/{total}]  {label:<30}  "
              f"final={final_mae:.3f}s  "
              f"({improvement:+.3f}s vs baseline, {direction})")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows).sort_values('final_mae')

    print(f"\n{'='*70}")
    print(f"RESULTS — {args.warehouse} WC{args.workcode}  |  Baseline: {baseline_mae:.4f}s")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))

    best = summary_df.iloc[0]
    print(f"\nBest combination:")
    print(f"  learning_rate    = {best['learning_rate']}")
    print(f"  trees_per_update = {int(best['trees_per_update'])}")
    print(f"  final MAE/task   = {best['final_mae']:.4f}s")
    print(f"  vs baseline      = {best['vs_baseline']:+.4f}s")

    csv_path = out_dir / f"{args.warehouse}_WC{args.workcode}_tune_results.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # One plot per trees value — lines are different learning rates
    x_labels = ['Initial'] + [f'B{i+1}' for i in range(n_batches)]

    for n_trees in trees_grid:
        fig, ax = plt.subplots(figsize=(12, 5))

        # Baseline
        ax.axhline(baseline_mae, color='gray', linestyle='--',
                   linewidth=1.5, label=f'Baseline {baseline_mae:.3f}s')

        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lr_grid)))
        for lr, color in zip(lr_grid, colors):
            maes = all_results[(lr, n_trees)]
            ax.plot(range(len(maes)), maes,
                    marker='o', linewidth=2, markersize=5,
                    color=color, label=f'lr={lr}')

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
        ax.set_xlabel('Update batch')
        ax.set_ylabel('MAE per task (seconds)')
        ax.set_title(
            f'{args.warehouse} WC{args.workcode} — Trees per update = {n_trees}\n'
            f'Lower is better | Fixed test set | {n_batches} batches'
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = out_dir / f"{args.warehouse}_WC{args.workcode}_trees{n_trees}.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plot saved: {plot_path}")

    # One heatmap — final MAE for every (lr, trees) combination
    matrix = np.array([
        [all_results[(lr, t)][-1] for t in trees_grid]
        for lr in lr_grid
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(trees_grid)))
    ax.set_xticklabels(trees_grid)
    ax.set_yticks(range(len(lr_grid)))
    ax.set_yticklabels(lr_grid)
    ax.set_xlabel('Trees per update')
    ax.set_ylabel('Learning rate')
    ax.set_title(
        f'{args.warehouse} WC{args.workcode} — Final MAE/task heatmap\n'
        f'Green = lower (better)  |  Baseline = {baseline_mae:.3f}s'
    )
    for i in range(len(lr_grid)):
        for j in range(len(trees_grid)):
            ax.text(j, i, f'{matrix[i,j]:.3f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if matrix[i,j] > matrix.mean() else 'black')
    plt.colorbar(im, ax=ax, label='MAE per task (s)')
    plt.tight_layout()

    heatmap_path = out_dir / f"{args.warehouse}_WC{args.workcode}_heatmap.png"
    plt.savefig(str(heatmap_path), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Heatmap saved: {heatmap_path}")
    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
