"""
predict_secondary.py
--------------------
Predict the number of tasks a worker can complete within a given time budget.

Usage:
    python predict_secondary.py OE assignment.csv 30
    python predict_secondary.py OE assignment.csv 30 --user_level 4 --sequenced
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_pipeline import prepare_predict_data
from utils.io import load_meta, load_model, setup_logging
from utils.worker_effects import level_to_effect

# Hard cap from your notebook to prevent outliers from eating the budget
MAX_TASK_TIME = 600 

def parse_args():
    parser = argparse.ArgumentParser(description="Predict task capacity within time budget")
    parser.add_argument("warehouse", help="Warehouse code (OE, OF, RT)")
    parser.add_argument("predict_csv", help="CSV of tasks to predict")
    parser.add_argument("budget_min", type=int, help="Time budget in minutes")
    parser.add_argument("--data_dir", default="training_data")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--sequenced", action="store_true", help="Use sequence-aware logic")
    parser.add_argument("--user_level", type=int, choices=[1, 2, 3, 4, 5], 
                        help="Worker level 1-5. Default: average (3)")
    parser.add_argument("--out", default=None, help="Output CSV path")
    return parser.parse_args()

def main():
    args = parse_args()
    warehouse = args.warehouse.upper()
    budget_sec = args.budget_min * 60

    setup_logging("predict_secondary", warehouse)
    logger = logging.getLogger(__name__)

    # 1. Load Meta and Model
    meta = load_meta(args.models_dir, warehouse)
    
    # 2. Initial data load to detect WorkCode
    try:
        # Dummy encodings for detection
        df, _, wc = prepare_predict_data(args.predict_csv, args.data_dir, warehouse, 
                                         {"top_aisles": [], "top_uoms": [], "product_tiers": ([],[],[],[])}, 
                                         args.sequenced)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if wc not in meta["workcodes"]:
        logger.error(f"WorkCode {wc} not trained for {warehouse}")
        sys.exit(1)

    # 3. Proper Engineering with WC-specific encodings
    wc_encodings = meta["encodings"][wc]
    wc_encodings["train_columns"] = meta["train_columns"][wc]
    
    df, X, _ = prepare_predict_data(args.predict_csv, args.data_dir, warehouse, 
                                    wc_encodings, args.sequenced)

    # 4. Apply Worker Effect
    effect = level_to_effect(args.user_level or 3, meta["level_medians"].get(wc, {}))
    X["worker_effect"] = effect
    X = X.reindex(columns=meta["train_columns"][wc], fill_value=0)

    # 5. Predict Task Times
    model = load_model(args.models_dir, warehouse, wc, args.sequenced)
    preds = model.predict(xgb.DMatrix(X))
    
    # 6. Cumulative Simulation (Secondary Goal Logic)
    
    tasks_completed = 0
    time_accumulated = 0.0
    
    for p_time in preds:
        # Cap prediction to prevent budget exhaustion by outliers
        effective_time = min(p_time, MAX_TASK_TIME)
        
        if time_accumulated + effective_time <= budget_sec:
            time_accumulated += effective_time
            tasks_completed += 1
        else:
            break

    logger.info(f"Result: {tasks_completed} tasks predicted in {args.budget_min} min budget.")

    # 7. Save Output
    out_path = Path(args.out) if args.out else Path(f"predict_data/{warehouse}/{warehouse}_{wc}_secondary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame([{
        "warehouse": warehouse, "workcode": wc, "budget_min": args.budget_min,
        "predicted_tasks": tasks_completed, "time_used_sec": round(time_accumulated, 2),
        "user_level": args.user_level or "3 (mean)"
    }]).to_csv(out_path, index=False)

if __name__ == "__main__":
    main()