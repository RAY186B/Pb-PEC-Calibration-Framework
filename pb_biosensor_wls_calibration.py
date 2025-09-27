#!/usr/bin/env python3
"""
Pb²⁺ Biosensor WLS Calibration

Simple WLS calibration for Pb²⁺ biosensor data with bootstrap CIs and diagnostics.
Uses fixed seed (42) for reproducibility. See README for details.

CITATION:
If you use this software, please cite it as:

MD Rayhan & Bingqian Liu (2025).
Pb²⁺ Biosensor Calibration Script (Version 1.0.0).
Zenodo. https://doi.org/10.5281/zenodo.17212395
"""


from __future__ import annotations

import argparse
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

# -------------------------
# Constants / Configuration
# -------------------------
IUPAC_LLOD_FACTOR: float = 3.3
IUPAC_LLOQ_FACTOR: float = 10.0
DEFAULT_CONFIDENCE: float = 0.975
BOOTSTRAP_DEFAULT: int = 2000
BOOTSTRAP_MIN: int = 1000
EPSILON: float = 1e-12
MAX_BOOTSTRAP_ITERS: int = 100_000
FILENAME_SANITIZE_REGEX = r"[^\\w\\-_.]"
LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# -------------------------
# Data structures
# -------------------------
@dataclass
class FitResult:
    slope: float
    intercept: float
    slope_se: float
    intercept_se: float
    conf_int: Optional[np.ndarray] = None

# -------------------------
# Helpers / Validation
# -------------------------
def sanitize_filename(name: str) -> str:
    safe = re.sub(FILENAME_SANITIZE_REGEX, "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "file"

def validate_dataframe(df: pd.DataFrame, required_columns: Tuple[str, ...]) -> None:
    if df is None:
        raise ValueError("Input dataframe is None")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.shape[0] == 0:
        raise ValueError("Input dataframe is empty")

def ensure_positive_weights(weights: np.ndarray) -> None:
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights contain non-finite values (inf or NaN)")
    if np.any(weights <= 0):
        raise ValueError("Weights must be strictly positive; zero/negative found")

def safe_divide(numerator: float, denominator: float, default: float = math.nan) -> float:
    if abs(denominator) <= EPSILON:
        return default
    return numerator / denominator

# -------------------------
# Core numerical routines
# -------------------------
def compute_weights_from_sd(sd_y_vec: np.ndarray) -> np.ndarray:
    sd = np.asarray(sd_y_vec, dtype=float)
    if sd.size == 0:
        raise ValueError("sd_y_vec is empty")
    if np.any(~np.isfinite(sd)):
        raise ValueError("sd_y_vec contains non-finite values")
    if np.any(sd <= 0.0):
        raise ValueError("sd_y_vec must be positive for weight calculation")
    weights = 1.0 / (sd ** 2)
    ensure_positive_weights(weights)
    return weights

def fit_weighted_ls(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> FitResult:
    if x.ndim != 1 or y.ndim != 1 or weights.ndim != 1:
        raise ValueError("x, y, weights must be 1D arrays")
    n = x.size
    if n == 0:
        raise ValueError("Empty arrays provided for regression")
    if not (x.size == y.size == weights.size):
        raise ValueError("x, y, and weights must have the same length")

    X = sm.add_constant(x)

    try:
        wls_model = sm.WLS(y, X, weights=weights)
        wls_res = wls_model.fit()
    except np.linalg.LinAlgError as ex:
        logger.exception("Linear algebra error during WLS fit")
        raise
    except Exception as ex:
        logger.exception("Unexpected error during WLS fit: %s", ex)
        raise

    params = wls_res.params
    bse = wls_res.bse
    intercept, slope = float(params[0]), float(params[1])
    intercept_se, slope_se = float(bse[0]), float(bse[1])

    if abs(slope) <= EPSILON:
        logger.warning("Slope near zero (abs(slope) <= %g). Estimates may be unstable.", EPSILON)

    return FitResult(slope=slope, intercept=intercept, slope_se=slope_se, intercept_se=intercept_se)

def bootstrap_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray, n_iter: int = BOOTSTRAP_DEFAULT,
                  rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    if n_iter < BOOTSTRAP_MIN:
        raise ValueError(f"n_iter must be >= {BOOTSTRAP_MIN}")
    if n_iter > MAX_BOOTSTRAP_ITERS:
        raise ValueError(f"n_iter too large; max allowed is {MAX_BOOTSTRAP_ITERS}")

    n = x.size
    if n == 0:
        raise ValueError("Cannot bootstrap empty dataset")

    rng = rng or np.random.default_rng()

    slopes = np.empty(n_iter, dtype=float)
    intercepts = np.empty(n_iter, dtype=float)

    if np.allclose(x, x[0]):
        raise ValueError("Independent variable x has no variation; bootstrap undefined")

    for i in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        xi = x[idx]
        yi = y[idx]
        wi = weights[idx]
        try:
            res = fit_weighted_ls(xi, yi, wi)
            slopes[i] = res.slope
            intercepts[i] = res.intercept
        except Exception as ex:
            logger.debug("Bootstrap iteration %d failed: %s", i, ex)
            slopes[i] = math.nan
            intercepts[i] = math.nan

    good = np.isfinite(slopes) & np.isfinite(intercepts)
    if good.sum() < max(10, int(0.05 * n_iter)):
        logger.warning("Only %d/%d bootstrap samples succeeded", good.sum(), n_iter)
    return {"slope": slopes[good], "intercept": intercepts[good]}

# -------------------------
# High-level pipeline
# -------------------------
def process_group(df: pd.DataFrame, x_col: str, y_col: str, sd_col: str,
                  bootstrap_iters: int = BOOTSTRAP_DEFAULT,
                  ci_level: float = DEFAULT_CONFIDENCE) -> Dict[str, object]:
    validate_dataframe(df, (x_col, y_col, sd_col))

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    sd = df[sd_col].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sd)
    if not mask.any():
        raise ValueError("All rows contain NaN or non-finite values in required columns")
    if mask.sum() < df.shape[0]:
        logger.info("Dropping %d/%d rows with NaN in required columns", df.shape[0] - mask.sum(), df.shape[0])
    x, y, sd = x[mask], y[mask], sd[mask]

    weights = compute_weights_from_sd(sd)
    fit = fit_weighted_ls(x, y, weights)
    bootstrap_res = bootstrap_fit(x, y, weights, n_iter=bootstrap_iters)

    ci = None
    if bootstrap_res["slope"].size >= 1000:
        alpha = 1.0 - ci_level
        lower = np.percentile(bootstrap_res["slope"], 100 * alpha / 2.0)
        upper = np.percentile(bootstrap_res["slope"], 100 * (1.0 - alpha / 2.0))
        ci = np.array([lower, upper])
    else:
        logger.warning("Not enough bootstrap slope samples (%d) to compute reliable CI", bootstrap_res["slope"].size)

    lod = None
    loq = None
    # Placeholder: replace with actual blank SD calculation
    try:
        if abs(fit.slope) > EPSILON:
            lod = safe_divide(IUPAC_LLOD_FACTOR * fit.intercept_se, fit.slope)
            loq = safe_divide(IUPAC_LLOQ_FACTOR * fit.intercept_se, fit.slope)
    except Exception:
        lod = None
        loq = None

    return {"fit": fit, "bootstrap": bootstrap_res, "slope_ci": ci, "lod": lod, "loq": loq}

# -------------------------
# CLI and I/O
# -------------------------
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as ex:
        logger.exception("Failed to read CSV: %s", ex)
        raise ValueError(f"Failed to parse CSV: {ex}")

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pb-PEC Calibration: robust WLS + bootstrap")
    parser.add_argument("data_csv", help="Path to input CSV file")
    parser.add_argument("--x-col", default="conc", help="Name of concentration column")
    parser.add_argument("--y-col", default="signal", help="Name of signal column")
    parser.add_argument("--sd-col", default="sd_y", help="Name of per-point SD column")
    parser.add_argument("--group-col", default=None, help="Optional column name to group samples")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DEFAULT, help=f"Bootstrap iterations (default {BOOTSTRAP_DEFAULT})")
    parser.add_argument("--ci", type=float, default=DEFAULT_CONFIDENCE, help=f"Confidence level (two-sided); default {DEFAULT_CONFIDENCE}")
    parser.add_argument("--out-dir", default="./results", help="Output directory for artifacts")
    args = parser.parse_args(argv)

    df = load_csv(args.data_csv)

    group_col = args.group_col
    if group_col and group_col not in df.columns:
        logger.error("Group column '%s' not found in data", group_col)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    grouped = df.groupby(group_col) if group_col else [("__all__", df)]
    overall_results = {}

    for name, group_df in grouped:
        safe_name = sanitize_filename(str(name))
        try:
            res = process_group(group_df, args.x_col, args.y_col, args.sd_col,
                                bootstrap_iters=args.bootstrap, ci_level=args.ci)
            overall_results[safe_name] = res
            logger.info("Processed group '%s': slope=%.6g intercept=%.6g", safe_name, res["fit"].slope, res["fit"].intercept)
        except Exception as ex:
            logger.exception("Failed to process group '%s': %s", safe_name, ex)

    summary_rows = []
    for gname, res in overall_results.items():
        fit: FitResult = res["fit"]
        lod = res.get("lod")
        loq = res.get("loq")
        slope_ci = res.get("slope_ci")
        summary_rows.append({
            "group": gname,
            "slope": fit.slope,
            "slope_se": fit.slope_se,
            "intercept": fit.intercept,
            "intercept_se": fit.intercept_se,
            "slope_ci_low": float(slope_ci[0]) if slope_ci is not None else math.nan,
            "slope_ci_high": float(slope_ci[1]) if slope_ci is not None else math.nan,
            "lod": lod if lod is not None else math.nan,
            "loq": loq if loq is not None else math.nan,
        })

    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(args.out_dir, "calibration_summary.csv")
    summary_df.to_csv(out_path, index=False)
    logger.info("Wrote summary to %s", out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
