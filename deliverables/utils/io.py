"""
utils/io.py
-----------
Data loading, saving, and logging setup for all pipeline scripts.

Public API
----------
setup_logging(script_name, warehouse, log_dir="logs")
    Configure logging to stdout and a dated log file.

load_reference_tables(data_dir, warehouse)
    Load Locations and Products CSVs. Returns (locations_df, products_df).

load_distance_matrix(data_dir, warehouse)
    Load Distance Matrix CSV if available. Returns long-format df or None.

load_activity_csv(path, warehouse)
    Load and type-cast a raw Activity CSV.

load_parquet(data_dir, warehouse)
    Load the processed parquet for a warehouse.

save_parquet(df, data_dir, warehouse)
    Save processed DataFrame to WH_Processed.parquet.

load_meta(models_dir, warehouse)
    Load meta.pkl for a warehouse. Raises FileNotFoundError if missing.

save_meta(meta, models_dir, warehouse)
    Save meta.pkl for a warehouse.

load_model(models_dir, warehouse, wc, sequenced=False)
    Load a saved XGBoost Booster from disk.

save_model(model, models_dir, warehouse, wc, sequenced=False)
    Save an XGBoost Booster to disk.

model_path(models_dir, warehouse, wc, sequenced=False)
    Return the Path for a model file without loading it.
"""

import logging
import pickle
from datetime import date
from pathlib import Path

import pandas as pd
import xgboost as xgb


# ── Column schemas ───────────────────────────────────────────────────────────

ACTIVITY_COLS = [
    "ActivityCode", "UserID", "WorkCode", "AssignmentID",
    "ProductID", "Quantity", "Timestamp", "LocationID",
]
LOCATIONS_COLS = ["LocationID", "Aisle", "Bay", "Level", "Slot"]
PRODUCTS_COLS = [
    "ProductID", "ProductCode", "UnitOfMeasure",
    "Weight", "Cube", "Length", "Width", "Height",
]
DISTANCE_COLS = ["FromLoc", "ToLoc", "distance"]


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(script_name, warehouse, log_dir="logs"):
    """
    Configure root logger to write to both stdout and a dated log file.

    Log file: logs/WH/script_name_YYYY-MM-DD.log

    Args:
        script_name: e.g. "preprocess", "model_training", "update_model"
        warehouse:   Warehouse code e.g. "OE"
        log_dir:     Root log directory (default: "logs")
    """
    log_path = Path(log_dir) / warehouse.upper()
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"{script_name}_{date.today().isoformat()}.log"

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )
    logging.getLogger(__name__).info(
        f"Logging to {log_file}"
    )


# ── Reference table loading ──────────────────────────────────────────────────

def load_reference_tables(data_dir, warehouse):
    """
    Load Locations and Products CSVs for a warehouse.

    Args:
        data_dir:  Root training_data directory.
        warehouse: Warehouse code e.g. "OE".

    Returns:
        (locations_df, products_df) — type-cast and trimmed.
    """
    wh_dir = Path(data_dir) / warehouse.upper()

    locations = pd.read_csv(
        wh_dir / f"{warehouse.upper()}_Locations.csv",
        header=0, names=LOCATIONS_COLS,
    )
    
    for col in ["LocationID", "Bay", "Level", "Slot"]:
        locations[col] = pd.to_numeric(
            locations[col], errors="coerce"
        ).astype("Int64")
    locations["Aisle"] = pd.to_numeric(
        locations["Aisle"], errors="coerce"
    )

    products = pd.read_csv(
        wh_dir / f"{warehouse.upper()}_Products.csv",
        header=0, names=PRODUCTS_COLS,
    )
    products["ProductID"] = pd.to_numeric(
        products["ProductID"], errors="coerce"
    ).astype("Int64")
    for col in ["Weight", "Cube"]:
        products[col] = pd.to_numeric(products[col], errors="coerce")
    products = products[["ProductID", "UnitOfMeasure", "Weight", "Cube"]]

    return locations, products


def load_distance_matrix(data_dir, warehouse):
    """
    Load the Distance Matrix CSV if it exists.

    Returns long-format DataFrame with columns [FromLoc, ToLoc, distance],
    or None if the file is not present. Caller should log accordingly.
    """
    dist_path = (
        Path(data_dir) / warehouse.upper()
        / f"{warehouse.upper()}_Distance_Matrix.csv"
    )
    if not dist_path.exists():
        return None

    dist_wide = pd.read_csv(dist_path, index_col=0)
    dist_long = (
        dist_wide.stack()
        .rename("distance")
        .reset_index()
        .rename(columns={"level_0": "FromLoc", "level_1": "ToLoc"})
    )
    dist_long["distance"] = pd.to_numeric(
        dist_long["distance"], errors="coerce"
    ).fillna(0.0)
    return dist_long


def load_activity_csv(path, warehouse=None):
    """
    Load a raw Activity CSV and apply type casting.

    Args:
        path:      Path to the CSV file.
        warehouse: Optional — used only for logging.

    Returns:
        DataFrame with correct dtypes. Does NOT compute time deltas —
        that is preprocess.py's job.
    """
    df = pd.read_csv(path, header=0, names=ACTIVITY_COLS)

    for col in ["ProductID", "Quantity", "LocationID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["UserID"] = df["UserID"].astype(str)
    df["WorkCode"] = (
        df["WorkCode"].astype(str)
        .apply(lambda x: x.split(".")[0] if isinstance(x, str) else x)
    )

    if "ActivityCode" in df.columns:
        df = df[df["ActivityCode"] != "AssignmentOpen"].copy()

    return df


# ── Parquet I/O ──────────────────────────────────────────────────────────────

def parquet_path(data_dir, warehouse):
    return Path(data_dir) / warehouse.upper() / f"{warehouse.upper()}_Processed.parquet"


def load_parquet(data_dir, warehouse):
    """Load WH_Processed.parquet. Raises FileNotFoundError if missing."""
    path = parquet_path(data_dir, warehouse)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed parquet not found: {path}. "
            f"Run preprocess.py {warehouse} first."
        )
    df = pd.read_parquet(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["UserID"] = df["UserID"].astype(str)
    df["WorkCode"] = (
        df["WorkCode"].astype(str)
        .apply(lambda x: x.split(".")[0] if isinstance(x, str) else x)
    )
    return df


def save_parquet(df, data_dir, warehouse):
    """Save DataFrame to WH_Processed.parquet, overwriting if exists."""
    path = parquet_path(data_dir, warehouse)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


# ── Model I/O ────────────────────────────────────────────────────────────────

def model_path(models_dir, warehouse, wc, sequenced=False):
    """
    Return the Path object for a model file.
    Naming convention:
        Non-sequenced: models/WH/WH_WCxx.json
        Sequenced:     models/WH/WH_WCxx_seq.json
    """
    suffix = "_seq" if sequenced else ""
    return (
        Path(models_dir) / warehouse.upper()
        / f"{warehouse.upper()}_WC{wc}{suffix}.json"
    )


def load_model(models_dir, warehouse, wc, sequenced=False):
    """Load a saved XGBoost Booster. Raises FileNotFoundError if missing."""
    path = model_path(models_dir, warehouse, wc, sequenced)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            f"Run model_training.py {warehouse} first."
        )
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def save_model(model, models_dir, warehouse, wc, sequenced=False):
    """Save an XGBoost Booster. Creates directories if needed."""
    path = model_path(models_dir, warehouse, wc, sequenced)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return path


# ── Meta I/O ─────────────────────────────────────────────────────────────────

def _meta_path(models_dir, warehouse):
    return Path(models_dir) / warehouse.upper() / "meta.pkl"


def load_meta(models_dir, warehouse):
    """Load meta.pkl. Raises FileNotFoundError if missing."""
    path = _meta_path(models_dir, warehouse)
    if not path.exists():
        raise FileNotFoundError(
            f"meta.pkl not found: {path}. "
            f"Run model_training.py {warehouse} first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def save_meta(meta, models_dir, warehouse):
    """Save meta.pkl, overwriting if exists."""
    path = _meta_path(models_dir, warehouse)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(meta, f)
    return path


# ── Prediction CSV loading ───────────────────────────────────────────────────

def load_predict_csv(path):
    """
    Load a work queue CSV for prediction.
    Same column schema as Activity CSV but without Timestamp
    (these are upcoming tasks with no recorded completion time).

    Returns DataFrame and the detected WorkCode.
    Raises ValueError if multiple WorkCodes are found — an assignment
    must contain exactly one WorkCode.
    """
    df = pd.read_csv(path)

    for col in ["ProductID", "Quantity", "LocationID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Update: Make UserID optional
    if "UserID" in df.columns:
        df["UserID"] = df["UserID"].astype(str)
    else:
        # Assign a dummy ID, downstream grouping logic (sequencing) still works
        df["UserID"] = "PREDICT_USER"

    df["WorkCode"] = (
        df["WorkCode"].astype(str)
        .apply(lambda x: x.split(".")[0] if isinstance(x, str) else x)
    )

    if "ActivityCode" in df.columns:
        df = df[df["ActivityCode"] != "AssignmentOpen"].copy()

    workcodes = df["WorkCode"].dropna().unique().tolist()
    if len(workcodes) == 0:
        raise ValueError("No valid WorkCode found in prediction CSV.")
    if len(workcodes) > 1:
        raise ValueError(
            f"Prediction CSV contains multiple WorkCodes: {workcodes}. "
            "An assignment must contain exactly one WorkCode. "
            "Check that you have submitted the correct file."
        )

    return df, str(workcodes[0])
