"""
utils/worker_effects.py
------------------------
Worker random effects estimation and level assignment.

Worker effects capture how much faster or slower each worker is relative
to the warehouse average, fit as random intercepts in a mixed effects model.

Level convention (consistent across all scripts):
    Level 5 = slowest workers  (most positive effect)
    Level 1 = fastest workers  (most negative effect)

Public API
----------
estimate_worker_effects(df)
    Fit random intercept model. Returns DataFrame [UserID, worker_effect].

compute_worker_levels(effects_df)
    Assign levels 1-5 by percentile.
    Returns (enriched_df, thresholds_dict, level_medians_dict).

level_to_effect(level, level_medians)
    Convert a user-supplied level (1-5) to a numeric effect value.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


# ── Effect estimation ─────────────────────────────────────────────────────────

def estimate_worker_effects(df, min_picks=10):
    """
    Fit a random intercept mixed effects model to estimate per-worker effects.

        Time_Delta_sec ~ 1 + (1 | UserID)

    Partial pooling from the mixed model naturally handles workers with few
    picks by shrinking their estimate toward the grand mean.
    Workers below min_picks get effect = 0.0 (grand mean) directly.

    Args:
        df:         DataFrame with UserID and Time_Delta_sec columns.
        min_picks:  Workers with fewer picks than this get effect = 0.0.

    Returns:
        DataFrame with columns [UserID, worker_effect].
        Positive = slower than average. Negative = faster than average.
    """
    df_re = df[["UserID", "Time_Delta_sec"]].dropna().copy()
    df_re["UserID"] = df_re["UserID"].astype(str)

    if df_re["UserID"].nunique() < 2:
        logger.warning("Fewer than 2 unique workers — returning zero effects.")
        return pd.DataFrame({
            "UserID":        df_re["UserID"].unique(),
            "worker_effect": 0.0,
        })

    pick_counts  = df_re["UserID"].value_counts()
    sparse_users = set(pick_counts[pick_counts < min_picks].index)
    eligible     = df_re[~df_re["UserID"].isin(sparse_users)]

    if eligible["UserID"].nunique() < 2:
        logger.warning(
            f"Fewer than 2 workers with >= {min_picks} picks — "
            "returning zero effects for all."
        )
        return pd.DataFrame({
            "UserID":        df_re["UserID"].unique(),
            "worker_effect": 0.0,
        })

    try:
        result = smf.mixedlm(
            "Time_Delta_sec ~ 1",
            data=eligible,
            groups=eligible["UserID"],
        ).fit(reml=True, disp=False)

        effects = pd.DataFrame({
            "UserID": list(result.random_effects.keys()),
            "worker_effect": [
                float(v.iloc[0]) for v in result.random_effects.values()
            ],
        })
        effects["UserID"] = effects["UserID"].astype(str)

    except Exception as e:
        logger.warning(f"Mixed effects model failed ({e}) — using zero effects.")
        effects = pd.DataFrame({
            "UserID":        eligible["UserID"].unique(),
            "worker_effect": 0.0,
        })

    # Sparse workers get grand mean (0.0)
    if sparse_users:
        sparse_df = pd.DataFrame({
            "UserID":        list(sparse_users),
            "worker_effect": 0.0,
        })
        effects = pd.concat([effects, sparse_df], ignore_index=True)
        logger.info(
            f"  {len(sparse_users)} worker(s) with < {min_picks} picks "
            "assigned effect = 0.0"
        )

    return effects


# ── Level assignment ──────────────────────────────────────────────────────────

# Percentile boundaries for the 5 levels.
# Effect is positive = slow, negative = fast.
# Level 1 = fastest = lowest (most negative) effects = bottom percentile.
LEVEL_PERCENTILES = {
    1: (0,   20),   # fastest 20%
    2: (20,  40),
    3: (40,  60),   # middle (average)
    4: (60,  80),
    5: (80, 100),   # slowest 20%
}


def compute_worker_levels(effects_df):
    """
    Assign a level 1-5 to each worker based on their effect percentile.

    Args:
        effects_df: DataFrame with [UserID, worker_effect].

    Returns:
        Tuple of:
            enriched_df:   effects_df with added 'level' column (int 1-5)
            thresholds:    dict {level: (low_bound, high_bound)} in seconds
            level_medians: dict {level: median_effect} — used at predict time
                           to convert a supplied level number to an effect value
    """
    df = effects_df.copy()

    if df.empty or df["worker_effect"].isna().all():
        df["level"] = 3
        return df, {}, {lv: 0.0 for lv in range(1, 6)}

    effects = df["worker_effect"].values
    thresholds    = {}
    level_medians = {}

    df["level"] = 3  # default

    for level, (lo, hi) in LEVEL_PERCENTILES.items():
        low_val  = np.percentile(effects, lo)
        high_val = np.percentile(effects, hi)
        thresholds[level] = (low_val, high_val)

        mask = (
            (effects >= low_val) & (effects < high_val)
            if hi < 100
            else (effects >= low_val)
        )
        df.loc[mask, "level"] = level
        level_medians[level] = float(np.median(effects[mask])) if mask.any() else 0.0

    return df, thresholds, level_medians


# ── Prediction helpers ────────────────────────────────────────────────────────

def get_worker_effect(user_id, effects_df, fallback=0.0):
    """
    Look up a single worker's effect by UserID.
    Returns fallback (default 0.0 = grand mean) if worker not found.
    """
    row = effects_df[effects_df["UserID"].astype(str) == str(user_id)]
    if row.empty:
        return fallback
    return float(row["worker_effect"].iloc[0])


def level_to_effect(level, level_medians):
    """
    Convert a user-supplied level (1-5) to a numeric worker_effect value
    using the median effect of workers at that level from training data.

    Level 5 = slowest = most positive effect.
    Level 1 = fastest = most negative effect.

    Args:
        level:         Int 1-5 supplied by the client at predict time.
        level_medians: Dict {level: median_effect} from compute_worker_levels().

    Returns:
        Float worker_effect to apply uniformly to all rows in the assignment.
    """
    level = int(level)
    if level not in level_medians:
        logger.warning(
            f"Level {level} not in level_medians — using 0.0 (grand mean)."
        )
        return 0.0
    return float(level_medians[level])
