from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


KEEP_COLUMNS = [
    "ActivityCode",
    "UserID",
    "WorkCode",
    "AssignmentID",
    "ProductID",
    "Quantity",
    "Timestamp",
    "LocationID",
]

WAREHOUSE_FILES = {
    "oe": "oe_detailed.parquet",
    "of": "of_detailed.parquet",
    "rt": "rt_detailed.parquet",
}


def slugify(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def build_valid_windows(df: pd.DataFrame, window_size: int) -> list[tuple[int, int, str, object]]:
    # Mark contiguous runs where both WorkCode and UserID stay constant.
    run_break = (
        df["WorkCode"].ne(df["WorkCode"].shift())
        | df["UserID"].ne(df["UserID"].shift())
    )
    run_id = run_break.cumsum()

    windows: list[tuple[int, int, str, object]] = []
    for _, idx in df.groupby(run_id).indices.items():
        run_len = len(idx)
        if run_len < window_size:
            continue

        run_start = idx[0]
        run_end = idx[-1]
        workcode = df.at[run_start, "WorkCode"]
        userid = df.at[run_start, "UserID"]

        for start in range(run_start, run_end - window_size + 2):
            end = start + window_size
            windows.append((start, end, workcode, userid))

    return windows


def sample_and_save(
    warehouse: str,
    parquet_path: Path,
    output_dir: Path,
    repeats: int = 3,
    window_size: int = 50,
    seed: int = 42,
) -> None:
    df = pd.read_parquet(parquet_path)

    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{parquet_path.name} missing required columns: {missing}")

    windows = build_valid_windows(df, window_size)
    if len(windows) < repeats:
        raise ValueError(
            f"{parquet_path.name} only has {len(windows)} valid {window_size}-row windows; "
            f"need at least {repeats}."
        )

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(windows), size=repeats, replace=False)

    for sample_num, win_idx in enumerate(chosen_idx, start=1):
        start, end, workcode, _userid = windows[int(win_idx)]
        sample_df = df.iloc[start:end][KEEP_COLUMNS].copy()

        fname = f"{warehouse}_{slugify(workcode)}_sample{sample_num}.csv"
        sample_df.to_csv(output_dir / fname, index=False)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    processed_dir = repo_root / "data" / "processed"
    output_dir = repo_root / "data" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    for warehouse, filename in WAREHOUSE_FILES.items():
        sample_and_save(
            warehouse=warehouse,
            parquet_path=processed_dir / filename,
            output_dir=output_dir,
            repeats=3,
            window_size=50,
            seed=42,
        )


if __name__ == "__main__":
    main()
