# Lucas Systems Warehouse Planning Capstone Project

This project builds a production-style modeling pipeline for forecasting warehouse pick labor from historical execution data. The final deliverable is an XGBoost-based system that predicts how long a future work queue will take, estimates how many tasks can be completed within a time budget, and supports incremental model updates as new completed picks arrive.

The current handoff package lives in [`deliverables/`](deliverables/). Earlier exploratory notebooks and prototypes remain in the repository for project history, but the maintained pipeline is the script-based workflow under `deliverables`.

## Project Goals

- Predict total completion time for a future warehouse assignment before work begins.
- Predict task capacity within a fixed time budget.
- Preserve warehouse-specific model behavior by training one model per warehouse WorkCode.
- Account for product, location, quantity, and worker-speed effects.
- Support incremental learning from new completed pick data without retraining from scratch every day.
- Provide both command-line and dashboard workflows for client handoff.

## Current Modeling Approach

The final pipeline uses one XGBoost regression model per WorkCode. The target is `Time_Delta_sec`, computed as the elapsed time between consecutive completed pick events for the same worker.

Main modeling choices:

- **XGBoost Tweedie regression** for the main pick-time model.
- **Quantile XGBoost models** for lower and upper prediction interval bounds.
- **Per-WorkCode models** so different work types can learn separate patterns.
- **Random-intercept worker effects** from `statsmodels` mixed effects modeling.
- **Worker levels 1-5** for prediction-time adjustment; in code, level 1 is fastest and level 5 is slowest.
- **Saved categorical encodings** in `meta.pkl` so training, prediction, and incremental updates use the same feature schema.
- **Incremental XGBoost updates** that add new trees to existing models using an adaptive learning rate based on the ratio of new data days to historical training days.

## Data Sources

Each warehouse uses four source tables:

| File | Purpose |
|---|---|
| `WH_Activity.csv` | Completed pick events with `Timestamp`, `UserID`, `WorkCode`, `ProductID`, `Quantity`, and `LocationID`. |
| `WH_Products.csv` | Product attributes, including unit of measure, weight, and cube. |
| `WH_Locations.csv` | Location attributes, including aisle, bay, level, and slot. |
| `WH_Distance_Matrix.csv` | Optional travel distance lookup between warehouse locations. |

The delivered sample artifacts currently focus on warehouse `OE`.

## Feature Engineering

The preprocessing and feature pipeline:

1. Drops `AssignmentOpen` rows.
2. Sorts each worker's picks chronologically.
3. Computes `Time_Delta_sec` from consecutive timestamps.
4. Drops first picks per worker and non-positive time deltas.
5. Removes the top 2 percent of time deltas and caps remaining rows at 600 seconds.
6. Joins product and location attributes.
7. Builds stable training encodings:
   - top 5 aisles, with all others bucketed as `other`
   - top 5 units of measure, with all others bucketed as `other`
   - product frequency tiers
   - shelf level buckets
   - numeric weight, cube, and quantity
8. Adds `worker_effect` as a model feature.

The code also contains a sequenced feature path for same-aisle, same-level, and travel-distance features. The current handoff models are non-sequenced.

## Repository Structure

```text
.
|-- README.md                         # Project-level overview
|-- deliverables/                     # Maintained handoff package
|   |-- README.md                     # Detailed functional documentation
|   |-- preprocess.py                 # Raw data -> processed parquet
|   |-- model_training.py             # Train WorkCode-level XGBoost models
|   |-- update_model_incremental.py   # Add new trees from new completed picks
|   |-- predict_primary.py            # Goal 1: total completion time
|   |-- predict_secondary.py          # Goal 2: tasks within time budget
|   |-- dashboard.py                  # Streamlit dashboard
|   |-- utils/                        # Shared loading, features, worker effects
|   |-- training_data/                # Active sample/reference data
|   |-- models/                       # Active trained model artifacts
|   `-- KT_demo/                      # Knowledge transfer demo package
|-- Model_Jiashen/                    # Modeling notebooks/prototypes
|-- EDA_* / Initial_EDA/              # Exploratory analysis notebooks
`-- data/                             # Original/raw data area where applicable
```

## Main Pipeline

Run commands from the repository root with explicit `deliverables/...` paths.

### 1. Preprocess Historical Data

```bash
python deliverables/preprocess.py OE \
  --data_dir deliverables/training_data
```

Creates:

```text
deliverables/training_data/OE/OE_Processed.parquet
```

### 2. Train Models

```bash
python deliverables/model_training.py OE \
  --data_dir deliverables/training_data \
  --models_dir deliverables/models
```

Creates one main model and two quantile interval models per WorkCode:

```text
deliverables/models/OE/OE_WC10.json
deliverables/models/OE/OE_WC10LB.json
deliverables/models/OE/OE_WC10UB.json
deliverables/models/OE/meta.pkl
```

Optional training arguments:

- `--trees`: boosting rounds, default `1200`
- `--min_rows`: skip sparse WorkCodes, default `500`
- `--min_days`: warning threshold for unique dates, default `3`
- `--cv`: enable 5-fold XGBoost parameter search
- `--coverage`: prediction interval coverage percentage, default `95`

### 3. Predict Goal 1: Total Assignment Time

```bash
python deliverables/predict_primary.py OE path/to/queue.csv \
  --data_dir deliverables/training_data \
  --models_dir deliverables/models \
  --out path/to/primary_prediction.csv
```

The prediction CSV must contain a single WorkCode. `UserID` is optional.

### 4. Predict Goal 2: Capacity Within a Budget

```bash
python deliverables/predict_secondary.py OE path/to/queue.csv 30 \
  --data_dir deliverables/training_data \
  --models_dir deliverables/models \
  --out path/to/secondary_prediction.csv
```

This processes rows in CSV order and returns how many tasks fit in the time budget.

### 5. Incrementally Update Models

```bash
python deliverables/update_model_incremental.py OE \
  --new_data path/to/new_completed_activity.csv \
  --data_dir deliverables/training_data \
  --models_dir deliverables/models
```

The update data must be completed Activity data with timestamps. The updater:

- reuses saved feature encodings
- detects new workers
- computes MAE before and after update
- adds new trees using adaptive learning rate
- stores MAE history and baseline alerts in `meta.pkl`

## Dashboard

The Streamlit dashboard wraps both prediction goals:

```bash
cd deliverables
streamlit run dashboard.py
```

It loads local `models/` and `training_data/` folders, accepts an uploaded queue CSV, and compares predicted completion time and capacity across worker levels.

## Knowledge Transfer Demo

The KT demo is isolated under [`deliverables/KT_demo/`](deliverables/KT_demo/). It demonstrates the full lifecycle:

1. Split OE activity data into historical training data, last-day incremental data, and a 50-row prediction queue.
2. Preprocess historical data.
3. Train models.
4. Run primary and secondary predictions.
5. Run incremental learning.
6. Run predictions again after update.

Create demo data:

```bash
python deliverables/KT_demo/data_split_for_demo.py
```

Print the exact Step 2 commands:

```bash
python deliverables/KT_demo/knowledge_transfer_demo.py --print_commands
```

Run the demo:

```bash
python deliverables/KT_demo/knowledge_transfer_demo.py --run
```

Fast smoke-test version:

```bash
python deliverables/KT_demo/knowledge_transfer_demo.py --run --trees 1 --update_trees 1
```

## Output Artifacts

Important artifacts generated by the maintained pipeline:

| Artifact | Purpose |
|---|---|
| `WH_Processed.parquet` | Cleaned and joined training table. |
| `WH_WCxx.json` | Main XGBoost model for WorkCode `xx`. |
| `WH_WCxxLB.json` / `WH_WCxxUB.json` | Lower/upper quantile models for prediction intervals. |
| `meta.pkl` | WorkCode list, feature columns, encodings, worker effects, worker levels, update history, and XGBoost parameters. |
| `logs/WH/*.log` | Script-level execution logs. |
| prediction CSVs | One-row outputs for primary and secondary goals. |

## Technologies

- Python
- pandas / numpy
- XGBoost
- statsmodels
- scikit-learn
- Streamlit
- Plotly

## Notes For Handoff

- The final client-facing workflow is in `deliverables/`; notebooks are historical and exploratory.
- Use explicit `--data_dir` and `--models_dir` paths when running from the repository root.
- Current shipped models are non-sequenced. Avoid `--sequenced` unless sequenced models are retrained and validated.
- `eval.py` exists for internal evaluation work; secondary-goal evaluation is not implemented.
- The KT demo files and generated demo data/models/outputs are intentionally isolated under `deliverables/KT_demo/`.
