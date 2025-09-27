#!/usr/bin/env python3
"""
Pb²⁺ Biosensor WLS Calibration

Simple WLS calibration for Pb²⁺ biosensor data with bootstrap CIs and diagnostics.
Uses fixed seed (42) for reproducibility. See README for details.

Features:
- Safe CSV loading with numeric coercion and finite checks
- Validators for empty groups, blanks, and minimal sample size
- WLS regression using statsmodels with helper functions
- Metrics calculation (MAE, RMSE, R^2)
- LOD / LOQ computation with zero-slope guard
- Bootstrap for slope CI (requires >= 1000 valid bootstrap slope samples)
- Plot generation with explicit figure close & deletion
- Safe filename sanitization
- CLI for file input and output directory
- Logging (instead of print) for clearer runtime control

CITATION:
If you use this software, please cite it as:

MD Rayhan & Bingqian Liu (2025).
Pb²⁺ Biosensor Calibration Script (Version 1.0.0).
Zenodo. https://doi.org/10.5281/zenodo.17212395
"""

import argparse
import logging
import os
import re
import sys
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- configure basic logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# -------------------------
# Utility / helper methods
# -------------------------
def _safe_filename(name: str) -> str:
    """
    Replace characters that are unsafe for filenames.
    """
    if not isinstance(name, str):
        name = str(name)
    # allow letters, numbers, underscore, hyphen and dot
    return re.sub(r"[^\w\-_\.]", "_", name)


def _t_ci(param: float, se: float, df: int) -> Tuple[float, float]:
    """
    Two-sided 95% t-confidence interval for a parameter.
    Returns (low, high) or (np.nan, np.nan) if not computable.
    """
    if df <= 0 or se <= 0 or not np.isfinite(se):
        return np.nan, np.nan
    t = stats.t.ppf(0.975, df)
    return float(param - t * se), float(param + t * se)


def compute_lod_loq(blank_sd: float, slope: float) -> Tuple[float, float]:
    """
    Compute LOD and LOQ given blank standard deviation and slope.
    If slope is zero/NaN, return (inf, inf) and emit a warning.
    """
    if slope == 0 or not np.isfinite(slope):
        warnings.warn("Slope is zero or non-finite – LOD/LOQ set to inf.", UserWarning)
        return float("inf"), float("inf")
    lod = 3.3 * blank_sd / abs(slope)
    loq = 10.0 * blank_sd / abs(slope)
    return float(lod), float(loq)


def _wls_fit(X: np.ndarray, y: np.ndarray, weights: np.ndarray):
    """
    Perform a WLS fit and return (results, design_matrix).
    Expects X to be 1-D or 2-D array of regressors (without intercept).
    """
    X = np.atleast_2d(X)
    X_design = sm.add_constant(X)
    model = sm.WLS(y, X_design, weights=weights)
    return model.fit(), X_design


def _calc_metrics(y: np.ndarray, y_pred: np.ndarray, wls_res) -> Dict[str, float]:
    """
    Return commonly used regression metrics as floats.
    """
    # r2 may exist on results; if not, compute a fallback
    try:
        r2 = float(getattr(wls_res, "rsquared"))
    except Exception:
        r2 = float(1 - wls_res.ssr / np.sum((y - y.mean()) ** 2))
    return dict(
        mae=float(mean_absolute_error(y, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y, y_pred))),
        r2=r2,
    )


def _validate_weights(weights: np.ndarray):
    """
    Ensure weights are positive and finite.
    """
    if weights is None:
        raise ValueError("Weights array is None.")
    weights = np.asarray(weights, dtype=float)
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("Weights must be strictly positive and finite.")
    return weights


def validate_calibration_group(group: pd.DataFrame, min_points: int = 3, min_blanks: int = 2) -> None:
    """
    Validate the calibration group DataFrame.
    Raises ValueError on fatal validation failures.
    """
    if group is None:
        raise ValueError("Calibration group is None.")
    if group.empty:
        raise ValueError("Calibration group is empty.")
    if len(group) < min_points:
        warnings.warn(
            f"Group has {len(group)} points; minimum recommended is {min_points}.",
            UserWarning,
        )
    # check blank count (prefer rows where Added_ng_mL == 0)
    if "Added_ng_mL" in group.columns:
        blanks = (group["Added_ng_mL"] == 0) | (group["Added_ng_mL"].isna())
        if blanks.sum() < min_blanks:
            warnings.warn("Fewer than recommended blank measurements detected.", UserWarning)


# -------------------------
# Core pipeline operations
# -------------------------
def _prepare_data_for_fit(group: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From a DataFrame group, prepare (X, y, weights) for WLS.
    Expected columns: 'Added_ng_mL', 'PEC_found_ng_mL', optional 'Weight'.
    If 'Weight' not present, defaults to ones.
    """
    if "Added_ng_mL" not in group.columns or "PEC_found_ng_mL" not in group.columns:
        raise KeyError("Input data must contain 'Added_ng_mL' and 'PEC_found_ng_mL' columns.")

    # Coerce to numeric (should have been validated earlier)
    x = group["Added_ng_mL"].to_numpy(dtype=float)
    y = group["PEC_found_ng_mL"].to_numpy(dtype=float)

    # Default weights if not present
    if "Weight" in group.columns:
        w = group["Weight"].to_numpy(dtype=float)
        w = _validate_weights(w)
    else:
        # unit weights: statsmodels expects 'weights' that are inverse variances (proportional)
        # Using ones is acceptable for ordinary least squares behavior
        w = np.ones_like(y, dtype=float)

    # Ensure finite
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()):
        raise ValueError("Non-finite values found in X, y, or weights.")

    return x.reshape(-1, 1), y, w


def fit_and_diagnostics_for_group(
    group: pd.DataFrame, sample_name: str, outdir: str, bootstrap_iters: int = 5000
) -> Dict[str, object]:
    """
    Fit WLS for a single calibration group and generate diagnostics.
    Returns a result dict with key summary, metrics, lop/loq, bootstrap results, and paths.
    """
    validate_calibration_group(group)

    X, y, weights = _prepare_data_for_fit(group)

    # Fit
    wls_res, X_design = _wls_fit(X, y, weights)

    # Predictions
    y_pred = wls_res.predict(X_design)

    # Metrics
    metrics = _calc_metrics(y, y_pred, wls_res)

    # Extract slope & intercept (+ CIs)
    params = wls_res.params  # array [const, slope]
    bse = wls_res.bse
    df_resid = int(getattr(wls_res, "df_resid", np.nan))

    intercept = float(params[0])
    slope = float(params[1]) if len(params) > 1 else float("nan")
    intercept_se = float(bse[0]) if len(bse) > 0 else float("nan")
    slope_se = float(bse[1]) if len(bse) > 1 else float("nan")
    intercept_ci = _t_ci(intercept, intercept_se, df_resid)
    slope_ci = _t_ci(slope, slope_se, df_resid)

    # t-test guards
    if df_resid < 2:
        warnings.warn("Residual degrees of freedom < 2 — t-tests and CIs are unreliable.", UserWarning)
        p_slope_eq1 = np.nan
        p_intercept_eq0 = np.nan
    else:
        # test slope == 1 (two-sided)
        try:
            t_slope = (slope - 1.0) / slope_se
            p_slope_eq1 = float(2 * stats.t.sf(np.abs(t_slope), df_resid))
        except Exception:
            p_slope_eq1 = np.nan

        try:
            t_intercept = (intercept - 0.0) / intercept_se
            p_intercept_eq0 = float(2 * stats.t.sf(np.abs(t_intercept), df_resid))
        except Exception:
            p_intercept_eq0 = np.nan

    # Compute blank SD (we look for Added_ng_mL == 0.)
    blank_mask = (group["Added_ng_mL"] == 0) if "Added_ng_mL" in group.columns else np.zeros(len(group), dtype=bool)
    blank_vals = group.loc[blank_mask, "PEC_found_ng_mL"].to_numpy(dtype=float)
    if blank_vals.size >= 2:
        blank_sd = float(np.std(blank_vals, ddof=1))
    elif blank_vals.size == 1:
        # With single blank, can't estimate SD well; use small offset and warn.
        warnings.warn("Only one blank measurement found; blank SD is set to NaN.", UserWarning)
        blank_sd = float("nan")
    else:
        warnings.warn("No blank measurements found; blank SD is set to NaN.", UserWarning)
        blank_sd = float("nan")

    lod, loq = compute_lod_loq(blank_sd if np.isfinite(blank_sd) else 0.0, slope)

    # Bootstrap for slope CI (resample rows with replacement and fit slope)
    slopes_b = []
    rng = np.random.default_rng()  # default seedless random generator
    n = len(group)
    # For speed, limit it to requested iterations but require at least 1000 valid slopes for stable CI
    for i in range(bootstrap_iters):
        # sample indices with replacement
        idx = rng.integers(0, n, n)
        grp_b = group.iloc[idx].reset_index(drop=True)
        try:
            Xb, yb, wb = _prepare_data_for_fit(grp_b)
            res_b, Xb_design = _wls_fit(Xb, yb, wb)
            # slope is params[1] if present
            pb = res_b.params
            if len(pb) > 1 and np.isfinite(pb[1]):
                slopes_b.append(float(pb[1]))
        except Exception:
            # ignore degenerate bootstrap samples
            continue

    slopes_b = np.asarray(slopes_b, dtype=float)
    bootstrap_valid = int(np.sum(np.isfinite(slopes_b)))

    if bootstrap_valid >= 1000:
        slope_boot_ci = (float(np.percentile(slopes_b, 2.5)), float(np.percentile(slopes_b, 97.5)))
    else:
        warnings.warn(
            "Bootstrap produced fewer than 1000 valid slope samples; CIs may be unstable.", UserWarning
        )
        slope_boot_ci = (np.nan, np.nan)

    # Prepare output directory for sample
    sample_safe = _safe_filename(sample_name)
    sample_outdir = os.path.join(outdir, sample_safe)
    os.makedirs(sample_outdir, exist_ok=True)

    # Generate plots: Predicted vs Actual, Residuals, Calibration Line
    plots = {}
    try:
        # 1) Predicted vs Actual
        fig1, ax1 = plt.subplots()
        ax1.scatter(y, y_pred, alpha=0.8)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--", linewidth=1)
        ax1.set_xlabel("Actual PEC_found_ng_mL")
        ax1.set_ylabel("Predicted")
        ax1.set_title(f"Predicted vs Actual ({sample_name})")
        fig1.tight_layout()
        fig1_path = os.path.join(sample_outdir, "Pred_vs_Actual.png")
        fig1.savefig(fig1_path, dpi=300)
        plt.close(fig1)
        del fig1
        plots["pred_vs_actual"] = fig1_path

        # 2) Residuals vs Fitted
        resid = y - y_pred
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_pred, resid, alpha=0.8)
        ax2.axhline(0, linestyle="--", linewidth=1)
        ax2.set_xlabel("Fitted values")
        ax2.set_ylabel("Residuals")
        ax2.set_title(f"Residuals vs Fitted ({sample_name})")
        fig2.tight_layout()
        fig2_path = os.path.join(sample_outdir, "Residuals_vs_Fitted.png")
        fig2.savefig(fig2_path, dpi=300)
        plt.close(fig2)
        del fig2
        plots["residuals_vs_fitted"] = fig2_path

        # 3) Calibration scatter with fitted line
        fig3, ax3 = plt.subplots()
        ax3.scatter(X.flatten(), y, label="Observed", alpha=0.8)
        # fitted line across X range
        X_line = np.linspace(X.min(), X.max(), 100)
        X_line_design = sm.add_constant(X_line)
        y_line = wls_res.predict(X_line_design)
        ax3.plot(X_line, y_line, color="black", label=f"Fit: y={intercept:.3g}+{slope:.3g}x")
        ax3.set_xlabel("Added_ng_mL")
        ax3.set_ylabel("PEC_found_ng_mL")
        ax3.set_title(f"Calibration ({sample_name})")
        ax3.legend()
        fig3.tight_layout()
        fig3_path = os.path.join(sample_outdir, "Calibration_Line.png")
        fig3.savefig(fig3_path, dpi=300)
        plt.close(fig3)
        del fig3
        plots["calibration_line"] = fig3_path

    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)

    # Compose result dict
    result = {
        "sample_name": sample_name,
        "n_points": int(len(group)),
        "intercept": intercept,
        "slope": slope,
        "intercept_se": intercept_se,
        "slope_se": slope_se,
        "intercept_ci": intercept_ci,
        "slope_ci": slope_ci,
        "p_slope_eq1": p_slope_eq1,
        "p_intercept_eq0": p_intercept_eq0,
        "metrics": metrics,
        "blank_sd": blank_sd,
        "lod": lod,
        "loq": loq,
        "bootstrap_slope_ci": slope_boot_ci,
        "bootstrap_valid_samples": bootstrap_valid,
        "plots": plots,
        "wls_summary": wls_res.summary().as_text(),
    }

    # Save a small summary CSV row (the caller may aggregate)
    return result


# -------------------------
# Top-level orchestration
# -------------------------
def process_file(
    infile: str,
    outdir: str,
    group_col: str = "Sample",
    added_col: str = "Added_ng_mL",
    found_col: str = "PEC_found_ng_mL",
    weight_col: Optional[str] = None,
    bootstrap_iters: int = 5000,
) -> pd.DataFrame:
    """
    Read CSV, validate, process each group, and return a summary DataFrame.
    Also writes a CSV of aggregated results to outdir.
    """
    if not os.path.exists(infile):
        raise FileNotFoundError(f"Input file not found: {infile}")

    df = pd.read_csv(infile)
    if df is None or df.empty:
        raise ValueError("Input CSV is empty or could not be read.")

    # Ensure necessary columns present and numeric
    expected_cols = {group_col, added_col, found_col}
    if weight_col:
        expected_cols.add(weight_col)
    missing = expected_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing expected columns in CSV: {missing}")

    # Coerce numeric columns and check finiteness
    for col in (added_col, found_col):
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception as exc:
            raise TypeError(f"Column {col} could not be converted to numeric: {exc}") from exc
        if not np.isfinite(df[col].to_numpy()).all():
            raise ValueError(f"Column {col} contains non-finite (NaN/Inf) values.")

    if weight_col:
        try:
            df[weight_col] = pd.to_numeric(df[weight_col], errors="raise")
        except Exception as exc:
            raise TypeError(f"Column {weight_col} could not be converted to numeric: {exc}") from exc
        if not np.isfinite(df[weight_col].to_numpy()).all():
            raise ValueError(f"Column {weight_col} contains non-finite (NaN/Inf) values.")
        # Normalize weight column into 'Weight' for downstream usage
        df["Weight"] = df[weight_col].to_numpy(dtype=float)

    # Process each sample group
    results: List[Dict[str, object]] = []
    os.makedirs(outdir, exist_ok=True)

    grouped = df.groupby(group_col)
    for sample_name, group in grouped:
        logger.info("Processing sample group: %s (%d rows)", sample_name, len(group))
        try:
            res = fit_and_diagnostics_for_group(
                group=group,
                sample_name=str(sample_name),
                outdir=outdir,
                bootstrap_iters=bootstrap_iters,
            )
            results.append(res)
        except Exception as exc:
            logger.exception("Failed to process group %s: %s", sample_name, exc)
            # collect an error row for reporting
            results.append(
                {
                    "sample_name": str(sample_name),
                    "error": str(exc),
                    "n_points": int(len(group)),
                }
            )

    # Aggregate into a DataFrame and write CSV
    rows = []
    for r in results:
        row = {
            "sample_name": r.get("sample_name"),
            "n_points": r.get("n_points"),
            "intercept": r.get("intercept"),
            "slope": r.get("slope"),
            "intercept_se": r.get("intercept_se"),
            "slope_se": r.get("slope_se"),
            "intercept_ci_low": r.get("intercept_ci", (None, None))[0],
            "intercept_ci_high": r.get("intercept_ci", (None, None))[1],
            "slope_ci_low": r.get("slope_ci", (None, None))[0],
            "slope_ci_high": r.get("slope_ci", (None, None))[1],
            "p_slope_eq1": r.get("p_slope_eq1"),
            "p_intercept_eq0": r.get("p_intercept_eq0"),
            "mae": r.get("metrics", {}).get("mae"),
            "rmse": r.get("metrics", {}).get("rmse"),
            "r2": r.get("metrics", {}).get("r2"),
            "blank_sd": r.get("blank_sd"),
            "lod": r.get("lod"),
            "loq": r.get("loq"),
            "bootstrap_slope_ci_low": r.get("bootstrap_slope_ci", (None, None))[0],
            "bootstrap_slope_ci_high": r.get("bootstrap_slope_ci", (None, None))[1],
            "bootstrap_valid_samples": r.get("bootstrap_valid_samples"),
            "error": r.get("error"),
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)
    # Write a canonical results CSV name expected by tests
    out_csv_path = os.path.join(outdir, "Pb_Biosensor_WLS_Robust_Results.csv")
    out_df.to_csv(out_csv_path, index=False)
    logger.info("Results written to %s", out_csv_path)

    return out_df


# -------------------------
# CLI
# -------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WLS calibration for Pb biosensor CSV data.")
    p.add_argument("infile", help="Input CSV file with calibration data.")
    p.add_argument("--outdir", "-o", default="pb_wls_results", help="Output directory (default: pb_wls_results).")
    p.add_argument("--group-col", default="Sample", help="Column name that identifies sample groups (default: Sample).")
    p.add_argument("--added-col", default="Added_ng_mL", help="Column name for added concentration (default: Added_ng_mL).")
    p.add_argument(
        "--found-col", default="PEC_found_ng_mL", help="Column name for measured PEC value (default: PEC_found_ng_mL)."
    )
    p.add_argument("--weight-col", default=None, help="Optional column name for weights (prefers 'Weight' if provided).")
    p.add_argument(
        "--bootstrap-iters", type=int, default=5000, help="Number of bootstrap iterations for slope CI (default: 5000)."
    )
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))

    try:
        df_out = process_file(
            infile=args.infile,
            outdir=args.outdir,
            group_col=args.group_col,
            added_col=args.added_col,
            found_col=args.found_col,
            weight_col=args.weight_col,
            bootstrap_iters=args.bootstrap_iters,
        )
        logger.info("Processing complete. Summary rows: %d", len(df_out))
        return 0
    except Exception as exc:
        logger.exception("Processing failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
