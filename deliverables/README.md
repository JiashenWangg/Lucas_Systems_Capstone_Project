Functional Documentation

# Warehouse Labor Prediction Pipeline

XGBoost-based pipeline for predicting warehouse pick task durations. Supports warehouses **OE**, **OF**, and **RT**.

> **Audience note:** Scripts marked `[warehouse]` are designed for non-technical client use. Scripts marked `[internal]` are for the data science team only.

---

## Table of Contents

- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Prediction CSV Format](#prediction-csv-format)
- [Scripts](#scripts)
  - [1. preprocess.py](#1-preprocesspy)
  - [2. model_training.py](#2-model_trainingpy)
  - [3. update_model_incremental.py](#3-update_model_incrementalpy)
  - [4. predict_primary.py](#4-predict_primarypy)
  - [5. predict_secondary.py](#5-predict_secondarypy)
  - [6. eval.py](#6-evalpy)
  - [7. dashboard.py](#7-dashboardpy)
- [Utils Modules](#utils-modules)
- [Quick Reference](#quick-reference)

---

## Overview

This pipeline trains XGBoost models on historical warehouse pick data, keeps those models current with daily incremental updates, and serves predictions through command-line scripts and an interactive Streamlit dashboard.

All scripts are run from the **project root directory**. The `utils/` folder contains shared modules imported by every script.

| Script | Audience | Purpose |
|---|---|---|
| `preprocess.py` | internal | Clean raw CSVs and export a processed parquet |
| `model_training.py` | internal | Train one XGBoost model per WorkCode on all data |
| `update_model_incremental.py` | internal | Daily incremental model update from new completed picks |
| `predict_primary.py` | warehouse | Predict total time to complete an assignment |
| `predict_secondary.py` | warehouse | Predict number of tasks completable within a time budget |
| `eval.py` | internal | Evaluate model performance with a proper train/test split |

---

## Folder Structure

```
project_root/
├── training_data/
│   └── WH/                          # e.g. OE/, OF/, RT/
│       ├── WH_Activity.csv
│       ├── WH_Products.csv
│       ├── WH_Locations.csv
│       ├── WH_Distance_Matrix.csv   # optional
│       └── WH_Processed.parquet     # created by preprocess.py
├── models/
│   └── WH/
│       ├── WH_WCxx.json             # non-sequenced model
│       ├── WH_WCxx_seq.json         # sequenced model (if trained)
│       └── meta.pkl                 # encodings, worker effects, MAE history
├── predict_data/
│   └── WH/
│       └── WH_pred1.csv             # work queue files for prediction
├── logs/
│   └── WH/
│       ├── preprocess_YYYY-MM-DD.log
│       ├── model_training_YYYY-MM-DD.log
│       ├── update_model_YYYY-MM-DD.log
│       └── predict_primary_YYYY-MM-DD.log
├── eval_results/
│   └── WH/
│       ├── primary/
│       │   └── WH_eval_primary_50.csv
│       └── secondary/
│           └── WH_eval_secondary_30.csv
├── preprocess.py
├── model_training.py
├── update_model_incremental.py
├── predict_primary.py
├── predict_secondary.py
├── eval.py
├── dashboard.py
└── utils/
    ├── feature_engineer.py
    ├── worker_effects.py
    ├── data_pipeline.py
    └── io.py
```

---

## Prediction CSV Format

Work queue files submitted for prediction must follow this column schema. There is no `Timestamp` column — these are future tasks with no recorded completion time.

| Column | Type | Notes |
|---|---|---|
| `ActivityCode` | string | e.g. `PickPut`. `AssignmentOpen` rows are dropped automatically. |
| `WorkCode` | integer or string | Must be a single WorkCode across all rows. Script errors if multiple are found. |
| `AssignmentID` | integer | Used for reference only, not a model feature. |
| `ProductID` | integer | Joined to Products table to get Weight, Cube, UoM. |
| `Quantity` | integer | Number of units to pick. |
| `LocationID` | integer | Joined to Locations table to get Aisle, Bay, Level, Slot. |

> `UserID` is optional. If absent, all rows are assigned a single dummy user ID for sequencing purposes.

---

## Scripts

### 1. `preprocess.py`

`[internal]` Converts raw warehouse CSVs into a single cleaned parquet file ready for model training, evaluation, and incremental updates. Run once per warehouse before any other script.

**Command**

```bash
python preprocess.py <warehouse> [--data_dir training_data]
```

**Arguments**

| Argument | Type | Description |
|---|---|---|
| `warehouse` | string (required) | Warehouse code: `OE`, `OF`, or `RT` |
| `--data_dir` | path (optional) | Root training_data directory. Default: `training_data` |

**Input files** — all must be present in `training_data/WH/`:

- `WH_Activity.csv` — timestamped pick events per worker
- `WH_Locations.csv` — location ID to Aisle, Bay, Level, Slot mapping
- `WH_Products.csv` — product ID to UoM, Weight, Cube mapping
- `WH_Distance_Matrix.csv` — travel distance between locations *(optional)*

**Processing steps**

1. Load Activity CSV and drop `AssignmentOpen` rows
2. Sort each worker's picks chronologically by `Timestamp`
3. Compute `Time_Delta_sec` as elapsed seconds since that worker's previous pick
4. Drop the first pick per worker (no previous pick to compute delta from)
5. Apply two-stage time filter:
   - Remove the top 2% of `Time_Delta_sec` values globally (98th percentile cut)
   - Hard cap any remaining values above 600 seconds
6. Join Locations and Products attributes onto each row
7. If Distance Matrix is present: join `Travel_Distance` for each pick pair; join previous location attributes (`Prev_Aisle`, `Prev_Level`) for sequenced mode

**Output**

| Output | Details |
|---|---|
| `training_data/WH/WH_Processed.parquet` | Single parquet file with all cleaned rows for all WorkCodes. Overwrites any previous file. |
| `logs/WH/preprocess_YYYY-MM-DD.log` | Rows loaded, rows dropped at each stage, % dropped, date range, unique day count, WorkCode counts, distance matrix status. |

> If the Distance Matrix is absent, the script logs this and continues. Sequenced prediction will not be available until the matrix is added and `preprocess.py` is rerun.

---

### 2. `model_training.py`

`[internal]` Trains one XGBoost model per WorkCode using all available data. No train/test split is performed — the full dataset is used to maximise model quality. For evaluation with a proper holdout, use `eval.py` instead.

**Command**

```bash
python model_training.py <warehouse> [--data_dir ...] [--models_dir ...] [--sequenced] [--trees 1200] [--min_rows 500]
```

**Arguments**

| Argument | Type | Description |
|---|---|---|
| `warehouse` | string (required) | Warehouse code: `OE`, `OF`, or `RT` |
| `--data_dir` | path (optional) | Root training_data directory. Default: `training_data` |
| `--models_dir` | path (optional) | Root models directory. Default: `models` |
| `--sequenced` | flag (optional) | Train with sequence-dependent features (`same_aisle`, `same_level`, `Travel_Distance`). Requires `WH_Distance_Matrix.csv`. Model saved as `WH_WCxx_seq.json`. |
| `--trees` | integer (optional) | Number of XGBoost boosting rounds. Default: `1200` |
| `--min_rows` | integer (optional) | Skip WorkCodes with fewer rows than this threshold. Default: `500` |

**Processing steps**

For each WorkCode with sufficient rows:

1. Compute categorical encodings from that WorkCode's training data:
   - Top 5 aisles by frequency
   - Top 5 UoMs by frequency
   - Product tier boundaries (25th / 50th / 75th percentile of cumulative volume)
2. Fit a random intercept mixed effects model to estimate per-worker speed effects
3. Assign each worker a level 1–5 based on their effect percentile (5 = fastest)
4. Apply feature engineering and build the training feature matrix
5. Train XGBoost with Tweedie regression objective
6. Save model file and all encodings to `meta.pkl`

**Output**

| Output | Details |
|---|---|
| `models/WH/WH_WCxx.json` | Non-sequenced model (one per WorkCode). Saved as `WH_WCxx_seq.json` when `--sequenced` is used. |
| `models/WH/meta.pkl` | Contains: `workcodes` list, `train_columns` per WC, categorical encodings per WC (`top_aisles`, `top_uoms`, `product_tiers`), `worker_effects` table per WC, `level_thresholds` and `level_medians` per WC, empty `mae_history` and `baseline_mae` dicts for update tracking. |
| `logs/WH/model_training_YYYY-MM-DD.log` | WorkCodes trained and skipped, rows per WC, training time per WC, worker counts and level distribution per WC. |

> Run `model_training.py` separately with and without `--sequenced` if you want both model variants. Each produces different model files and does not overwrite the other.

---

### 3. `update_model_incremental.py`

`[internal]` Incrementally updates saved XGBoost models with new completed pick data. Run daily after new picks have been recorded. No historical data is stored between runs — the model file itself accumulates the learned patterns.

**Command**

```bash
python update_model_incremental.py <warehouse> --new_data <csv> [--data_dir ...] [--models_dir ...] [--sequenced] [--trees 150] [--alert_pct 20]
```

**Arguments**

| Argument | Type | Description |
|---|---|---|
| `warehouse` | string (required) | Warehouse code: `OE`, `OF`, or `RT` |
| `--new_data` | path (required) | Path to today's raw Activity CSV. Must have a `Timestamp` column — these are completed picks with recorded times. |
| `--data_dir` | path (optional) | Root training_data directory (for Locations and Products reference tables). Default: `training_data` |
| `--models_dir` | path (optional) | Root models directory. Default: `models` |
| `--sequenced` | flag (optional) | Update sequenced models. Must match how models were originally trained. |
| `--trees` | integer (optional) | Trees to add per WorkCode per run. Default: `150` |
| `--alert_pct` | float (optional) | Warn if batch MAE after update exceeds baseline by this percentage. Default: `20` |

**Processing steps**

1. Load `meta.pkl` to retrieve saved encodings for each WorkCode
2. Load today's Activity CSV and run preprocessing (time deltas, outlier filter, table joins)
3. Apply saved categorical encodings — never recomputed from new data
4. For each WorkCode in today's data:
   - Detect any new workers not seen at training; estimate their effects from today's data
   - Compute batch MAE on today's data using the current model (MAE before)
   - Add `--trees` new trees using learning rate `0.005`
   - Compute batch MAE again with the updated model (MAE after)
   - Compare MAE after to the stored baseline; log an alert if degradation exceeds `--alert_pct`
5. Save updated model file and updated `meta.pkl` with new MAE history entry

**Why learning rate 0.005?**

Each daily batch is small relative to the full training history. At the training learning rate (0.03), new trees overfit to that day's noise. At 0.005, each new tree contributes ~6× less to the final prediction — real patterns accumulate gradually while noise averages out.

**MAE alert system**

Each run computes batch MAE before and after the update and stores both in `meta.pkl`. On the first update run, the pre-update MAE is set as the baseline. On subsequent runs, if the post-update MAE exceeds the baseline by more than `--alert_pct`, a warning is logged. This is the signal that a WorkCode's model may be degrading and should be investigated.

**Output**

| Output | Details |
|---|---|
| `models/WH/WH_WCxx.json` | Updated model file, overwriting the previous one. Total tree count increases by `--trees` per run. |
| `models/WH/meta.pkl` | Updated with new worker effects for any new workers, and a new MAE history entry for each WorkCode processed. |
| `logs/WH/update_model_YYYY-MM-DD.log` | WorkCodes updated and skipped, new workers added, trees before/after, MAE before/after, rolling 7-day MAE trend, any alert warnings. |

---

### 4. `predict_primary.py`

`[warehouse]` Given a work queue CSV of upcoming tasks, predicts the total time in minutes for a worker to complete the entire assignment. WorkCode is auto-detected from the CSV — the script errors clearly if multiple WorkCodes are found.

**Command**

```bash
python predict_primary.py <warehouse> <predict_csv> [--sequenced] [--user_level 1-5] [--out path/to/output.csv]
```

**Arguments**

| Argument | Type | Description |
|---|---|---|
| `warehouse` | string (required) | Warehouse code: `OE`, `OF`, or `RT` |
| `predict_csv` | path (required) | CSV of upcoming tasks. No `Timestamp` column. See [Prediction CSV Format](#prediction-csv-format). |
| `--data_dir` | path (optional) | Root training_data directory. Default: `training_data` |
| `--models_dir` | path (optional) | Root models directory. Default: `models` |
| `--sequenced` | flag (optional) | Use sequenced model. Rows must be in intended pick order. Distance features computed from row sequence. |
| `--user_level` | integer 1–5 (optional) | Worker performance level. 1 = slowest, 5 = fastest. Applied uniformly to all tasks. If omitted, uses the grand mean (average worker). |
| `--out` | path (optional) | Output CSV path. Default: `predict_data/WH/WH_WC_prediction.csv` |

**Processing steps**

1. Detect WorkCode from the CSV — error if multiple WorkCodes found
2. Validate WorkCode against trained models in `meta.pkl`
3. Apply saved feature encodings from `meta.pkl`
4. Apply worker level effect (or grand mean if omitted)
5. Run XGBoost predictions on each task individually
6. Sum all per-task predictions to get total assignment time

**Output**

| Output | Details |
|---|---|
| `predict_data/WH/WH_WC_prediction.csv` | One-row CSV with columns: `warehouse`, `workcode`, `n_tasks`, `predicted_time_sec`, `predicted_time_min`, `user_level_applied`. |
| `logs/WH/predict_primary_YYYY-MM-DD.log` | WorkCode detected, task count, user level applied, predicted total time. |

---

### 5. `predict_secondary.py`

`[warehouse]` Given a work queue and a time budget in minutes, predicts how many tasks a worker can complete before the budget is exhausted. Tasks are processed in CSV row order — predictions are accumulated sequentially until the budget is used.

**Command**

```bash
python predict_secondary.py <warehouse> <predict_csv> <budget_min> [--sequenced] [--user_level 1-5] [--out path/to/output.csv]
```

**Arguments**

| Argument | Type | Description |
|---|---|---|
| `warehouse` | string (required) | Warehouse code: `OE`, `OF`, or `RT` |
| `predict_csv` | path (required) | CSV of upcoming tasks. Same format as `predict_primary.py`. |
| `budget_min` | integer (required) | Time budget in minutes. Converted to seconds internally. |
| `--data_dir` | path (optional) | Root training_data directory. Default: `training_data` |
| `--models_dir` | path (optional) | Root models directory. Default: `models` |
| `--sequenced` | flag (optional) | Use sequenced model. |
| `--user_level` | integer 1–5 (optional) | Worker level. Default: `3` (average worker). |
| `--out` | path (optional) | Output CSV path. Default: `predict_data/WH/WH_WC_secondary.csv` |

**Capacity simulation logic**

Tasks are processed in the order they appear in the CSV. For each task:

- Predicted time is capped at 600 seconds to prevent a single outlier from consuming the entire budget
- If the running total plus this task's predicted time is within budget: add the task and continue
- If adding the task would exceed the budget: stop and return the count so far

**Output**

| Output | Details |
|---|---|
| `predict_data/WH/WH_WC_secondary.csv` | One-row CSV with columns: `warehouse`, `workcode`, `budget_min`, `predicted_tasks`, `time_used_sec`, `user_level`. |
| `logs/WH/predict_secondary_YYYY-MM-DD.log` | WorkCode detected, budget, user level, predicted task count. |

---

### 6. `eval.py`

`[internal]` Evaluates model performance on a held-out test set. Trains a fresh model internally with a chronological train/test split — never loads from the `models/` folder, avoiding the data leakage that would occur if the production model (trained on all data) were used for evaluation.

**Command**

```bash
python eval.py <warehouse> <goal> <chunk_size> [--data_dir ...] [--sequenced] [--test_pct 0.2] [--out eval_results] [--trees 1200]
```

**Arguments**

| Argument | Type | Description |
|---|---|---|
| `warehouse` | string (required) | Warehouse code: `OE`, `OF`, or `RT` |
| `goal` | `1` or `2` (required) | `1` = predict completion time. `2` = not yet implemented. |
| `chunk_size` | integer (required) | Number of tasks per evaluation chunk. Typical values: 25, 50, 100. |
| `--data_dir` | path (optional) | Root training_data directory. Default: `training_data` |
| `--sequenced` | flag (optional) | Evaluate with sequence features. |
| `--test_pct` | float (optional) | Fraction of rows to hold out as test set. Default: `0.20` |
| `--out` | path (optional) | Root output directory. Default: `eval_results` |
| `--trees` | integer (optional) | Boosting rounds for the eval model. Default: `1200` |

**Processing steps**

For each WorkCode:

1. Sort rows chronologically and split: first `(1 - test_pct)` rows for training, remainder for test
2. Compute encodings and worker effects from training rows only — test rows never influence encodings
3. Apply training encodings to both train and test sets
4. Train a fresh XGBoost model on the training set
5. Predict on the test set at pick level
6. Group consecutive picks per worker per day into chunks of `chunk_size`
7. Discard any chunks containing a pick exceeding 600 seconds
8. Aggregate pick-level predictions to chunk-level totals
9. Compute chunk-level MAE, MAE per task, and R²

**Output**

| Output | Details |
|---|---|
| `eval_results/WH/primary/WH_eval_primary_50.csv` | One row per WorkCode with columns: `warehouse`, `workcode`, `train_rows`, `train_time_s`, `test_rows`, `test_chunks`, `mean_time_s`, `median_time_s`, `r2`, `mae_per_task_s`. |
| `logs/WH/eval_YYYY-MM-DD.log` | WorkCodes evaluated, chunk counts, MAE per task, R² per WorkCode. |

> Goal 2 (secondary evaluation) is not yet implemented. Passing `goal=2` exits immediately with a message.

---

### 7. `dashboard.py`

`[warehouse]` An interactive Streamlit web application that wraps both prediction goals into a visual interface. Designed for non-technical warehouse staff who need predictions without running command-line scripts.

**Command**

```bash
streamlit run dashboard.py
```

**Features**

- Upload a work queue CSV directly in the browser
- Select warehouse and configure options (sequenced, time budget) from the sidebar
- Auto-detects WorkCode from the uploaded file
- Runs predictions for all five worker levels (1–5) simultaneously
- Goal 1: bar chart showing predicted total completion time per worker level
- Goal 2: bar chart showing tasks completable within the configured time budget per level
- Summary table comparing all levels side by side

**Requirements**

- `models/` folder must exist with trained models and `meta.pkl` for the selected warehouse
- `training_data/` folder must exist with Locations and Products CSVs for the selected warehouse
- Python packages: `streamlit`, `plotly`, `xgboost`, `pandas`, `numpy`

> The dashboard loads models and reference data from the local filesystem. It must be run from the project root directory and does not connect to any external server.

---

## Utils Modules

The `utils/` folder contains shared modules imported by all pipeline scripts. These are not run directly.

### `utils/io.py`

Handles all file I/O and logging configuration.

| Function | Purpose |
|---|---|
| `setup_logging(script, wh)` | Configure stdout + dated log file in `logs/WH/` |
| `load_reference_tables(dir, wh)` | Load Locations and Products CSVs with correct column schemas |
| `load_distance_matrix(dir, wh)` | Load Distance Matrix if present; return `None` if absent |
| `load_activity_csv(path)` | Load and type-cast a raw Activity CSV |
| `load_predict_csv(path)` | Load work queue CSV; validate single WorkCode; assign dummy UserID if absent |
| `load_parquet / save_parquet` | Load or save `WH_Processed.parquet` |
| `load_model / save_model` | Load or save XGBoost model. Handles seq/non-seq filename convention. |
| `load_meta / save_meta` | Load or save `meta.pkl` |

### `utils/feature_engineer.py`

Core feature engineering logic. All encodings are computed once at training time and passed in as arguments for all subsequent runs — never recomputed from new or prediction data.

| Function | Purpose |
|---|---|
| `compute_encodings(df)` | Compute `top_aisles`, `top_uoms`, `product_tiers` from training data. Call once; save to `meta.pkl`. |
| `apply_features(df, ...)` | Apply saved encodings to build aisle, level, UoM, product tier, and (optionally) sequence features. |
| `make_X(df, sequenced, train_columns)` | Build one-hot feature matrix. Pass `train_columns` to align to training shape. |
| `compute_prev_location(df)` | Shift `LocationID` within each user's sequence to create `Prev_LocationID`. Used for sequenced mode. |

**Features built (non-sequenced):**

- `aisle` — top 5 aisles by frequency; all others bucketed as `other`
- `level` — shelf level bucketed as 0, 1, 2, 3, 4, 5+
- `UoM` — top 5 units of measure; all others bucketed as `other`
- `t25_products`, `t25_50_products`, `t50_75_products`, `other_products` — product volume tier
- `Weight`, `Cube`, `Quantity` — numeric, used as-is

**Additional features (sequenced only):**

- `same_aisle` — 1 if consecutive picks share an aisle
- `same_level` — 1 if consecutive picks share a shelf level
- `Travel_Distance` — distance between consecutive pick locations (0.0 if no distance matrix)

### `utils/worker_effects.py`

Estimates and manages per-worker speed effects using a random intercept mixed effects model.

| Function | Purpose |
|---|---|
| `estimate_worker_effects(df)` | Fit mixed effects model. Workers with < 10 picks get effect = 0.0. |
| `compute_worker_levels(effects_df)` | Assign levels 1–5 by effect percentile. Return enriched df, thresholds, and level medians. |
| `level_to_effect(level, medians)` | Convert client-supplied level (1–5) to a numeric effect value using training medians. |
| `get_worker_effect(user_id, df)` | Look up a specific worker's effect. Returns 0.0 if not found. |

Worker level convention: Level 5 = fastest 20% of workers (most negative effect). Level 1 = slowest 20% (most positive effect). Level 3 = middle 20% (average).

### `utils/data_pipeline.py`

Convenience wrappers that chain loading, preprocessing, and feature engineering into single calls.

| Function | Used by / Purpose |
|---|---|
| `load_and_engineer(dir, wh, wc, ...)` | `model_training.py`, `eval.py` — load parquet, filter to one WC, compute or apply encodings, build feature matrix. |
| `prepare_new_data(csv, dir, wh, wc, ...)` | `update_model_incremental.py` — load raw Activity CSV, compute time deltas, filter, join tables, apply saved encodings. |
| `prepare_predict_data(csv, dir, wh, ...)` | `predict_primary.py`, `predict_secondary.py`, `dashboard.py` — load work queue CSV, join tables, apply saved encodings, return feature matrix and detected WorkCode. |

---

## Quick Reference

Typical sequence for a new warehouse (example: `OE`):

| Step | Command | When to run |
|---|---|---|
| 1 | `python preprocess.py OE` | Once, before any other script. Rerun if raw CSVs change. |
| 2 | `python model_training.py OE` | Once after preprocessing. Rerun to incorporate major data changes. |
| 3 | `python update_model_incremental.py OE --new_data today.csv` | Daily, as new completed picks become available. |
| 4a | `python predict_primary.py OE queue.csv` | When a total completion time estimate is needed. |
| 4b | `python predict_secondary.py OE queue.csv 30` | When a capacity estimate for a given time budget is needed. |
| 5 | `python eval.py OE 1 50` | Internal use only: measure model performance before deployment. |

All scripts are run from the **project root directory**. Logs are always written to `logs/WH/` regardless of which directory the script is called from.

