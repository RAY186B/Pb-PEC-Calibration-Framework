#!/usr/bin/env python3
"""
Pb²⁺ Biosensor WLS Calibration

Weighted Least Squares calibration for Pb²⁺ biosensor data with LOD/LOQ calculation.

CITATION:
MD Rayhan & Bingqian Liu (2025).
Pb²⁺ Biosensor Calibration Script.
Zenodo. https://doi.org/10.5281/zenodo.17218702

Dependencies:
- Python ≥ 3.8
- numpy
- pandas
- matplotlib
- statsmodels
- scikit-learn
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # head-less safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- constants ----------
IUPAC_LOD_FACTOR: float = 3.3
IUPAC_LOQ_FACTOR: float = 10.0
DEFAULT_SEED: int = 42
VALID_UNITS: List[str] = ["ng/mL", "ug/L", "μg/L", "mg/L", "ppm", "ppb"]
UNIT_SCALE_TO_NG_PER_ML: Dict[str, float] = {
    "ng/mL": 1.0,
    "ug/L": 1.0,   # 1 µg/L = 1 ng/mL
    "μg/L": 1.0,
    "mg/L": 1e6,
    "ppm": 1e6,    # aqueous approx
    "ppb": 1e3,
}

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------- helpers ----------
def safe_sample_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe or str(uuid.uuid4())


def ascii_unit(unit: str) -> str:
    return unit.replace("μ", "u")


# ---------- io ----------
def load_data(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No CSV at {path}")

    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode CSV file")

    if df.empty:
        raise ValueError("CSV file is empty")

    column_map = detect_columns(df)
    validate_data(df, column_map)
    logger.info(f"Loaded {len(df)} rows")
    return df, column_map


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    required = {
        "Sample": ["Sample", "sample", "Sample_Name", "SampleName", "sample_name"],
        "Added_ng_mL": ["Added_ng_mL", "added_ng_ml", "Added", "added", "Spike_ng_mL"],
        "PEC_found_ng_mL": ["PEC_found_ng_mL", "found_ng_ml", "Found", "found", "Measured_ng_mL"],
    }
    column_map: Dict[str, str] = {}
    for std, variants in required.items():
        for v in variants:
            if v in df.columns:
                column_map[std] = v
                break
        else:
            raise ValueError(f"Required column '{std}' not found. Columns: {list(df.columns)}")

    optional = {"RSD_percent": ["RSD_percent", "RSD", "rsd_percent", "RSD_percent_external"]}
    for std, variants in optional.items():
        for v in variants:
            if v in df.columns:
                column_map[std] = v
                break
    return column_map


def validate_data(df: pd.DataFrame, column_map: Dict[str, str]) -> None:
    conc = column_map["Added_ng_mL"]
    found = column_map["PEC_found_ng_mL"]
    for col in (conc, found):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Non-numeric values in {col}")
        if np.isinf(df[col]).any():
            raise ValueError(f"Infinite values in {col}")
        if (df[col] < 0).any():
            raise ValueError(f"Negative concentrations in {col}")

    if "RSD_percent" in column_map:
        rsd_col = column_map["RSD_percent"]
        df[rsd_col] = pd.to_numeric(df[rsd_col], errors="coerce")
        bad = df[rsd_col].dropna()
        if len(bad) and ((bad < 0) | (bad > 100)).any():
            logger.warning("Clipping RSD values to 0–100 %")
            df[rsd_col] = df[rsd_col].clip(0, 100)


def validate_group(group: pd.DataFrame, column_map: Dict[str, str]) -> None:
    if group.empty:
        raise ValueError("Group is empty")
    conc = column_map["Added_ng_mL"]
    name = str(group[column_map["Sample"]].iat[0])
    if len(group) < 3:
        raise ValueError(f"{name}: need ≥ 3 points")
    if group[conc].nunique() < 2:
        raise ValueError(f"{name}: need ≥ 2 distinct concentrations")
    blanks = (group[conc] == 0.0).sum()
    if blanks < 2:
        raise ValueError(f"{name}: need ≥ 2 blank replicates")


# ---------- calibration ----------
def compute_lod_loq(blank_sd: float, slope: float, min_slope: float) -> Tuple[float, float]:
    if not (np.isfinite(blank_sd) and blank_sd >= 0):
        raise ValueError("Invalid blank SD")
    if not np.isfinite(slope):
        raise ValueError("Invalid slope")
    if abs(slope) < min_slope:
        raise ValueError(f"Slope too small for LOD/LOQ (|slope| < {min_slope})")
    if blank_sd == 0:
        logger.warning("Zero blank SD → LOD/LOQ = 0")
        return 0.0, 0.0
    lod = IUPAC_LOD_FACTOR * blank_sd / abs(slope)
    loq = IUPAC_LOQ_FACTOR * blank_sd / abs(slope)
    return lod, loq


def estimate_noise_sd(
    group: pd.DataFrame,
    default_sd: Optional[float],
    column_map: Dict[str, str],
    fallback_frac: float,
) -> float:
    if default_sd is not None and default_sd > 0 and np.isfinite(default_sd):
        return default_sd

    found = column_map["PEC_found_ng_mL"]
    conc = column_map["Added_ng_mL"]
    stds = group.groupby(conc)[found].std().dropna()
    if len(stds):
        return float(stds.mean())

    mean_resp = group[found].mean()
    if mean_resp == 0:
        logger.warning("All responses zero; using fallback SD = 0.01 ng/mL")
        return 0.01
    return float(mean_resp * fallback_frac)


def wls_calibration(
    group: pd.DataFrame,
    sd_y: float,
    outdir: str,
    unit: str,
    column_map: Dict[str, str],
    rsd_clip: Tuple[float, float],
    small_number: float,
    min_slope: float,
) -> Dict[str, Any]:
    sample = str(group[column_map["Sample"]].iat[0])
    conc = column_map["Added_ng_mL"]
    found = column_map["PEC_found_ng_mL"]

    # blanks
    blanks = group[group[conc] == 0.0][found].dropna()
    if len(blanks) < 2:
        raise ValueError("Need ≥ 2 blank replicates")
    blank_mean = float(blanks.mean())
    blank_sd = float(blanks.std(ddof=1))

    # prepare data
    g = group.copy()
    g["Found_corr"] = g[found] - blank_mean
    negs = (g["Found_corr"] < 0).sum()
    if negs > len(g) * 0.1:
        logger.warning(f"{sample}: {negs} negative corrected values")

    X = g[conc].to_numpy(dtype=float)
    y = g["Found_corr"].to_numpy(dtype=float)

    # weights
    if "RSD_percent" in column_map and column_map["RSD_percent"] in g.columns:
        rsd_col = column_map["RSD_percent"]
        rsd = g[rsd_col].fillna(1.0).clip(*rsd_clip).to_numpy() / 100.0
        resp = g[found].replace(0, blank_mean).to_numpy()
        resp = np.maximum(resp, small_number)
        sd_vals = np.maximum(rsd * resp, small_number)
        weights = 1.0 / (sd_vals ** 2)
        logger.info(f"{sample}: RSD-based weighting")
    else:
        if sd_y <= 0 or not np.isfinite(sd_y):
            logger.warning(f"{sample}: invalid SD, using equal weights")
            weights = np.ones_like(y)
        else:
            weights = np.full_like(y, 1.0 / (sd_y ** 2))

    # fit
    X_design = sm.add_constant(X)
    model = sm.WLS(y, X_design, weights=weights).fit()
    if np.any(~np.isfinite(model.bse)):
        raise RuntimeError(f"{sample}: perfect multicollinearity – need more distinct concentrations")

    intercept, slope = model.params
    intercept_se, slope_se = model.bse

    # metrics
    y_pred = model.predict(X_design)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)

    # LOD/LOQ
    lod, loq = compute_lod_loq(blank_sd, slope, min_slope)

    # scale back to user unit
    user_scale = UNIT_SCALE_TO_NG_PER_ML.get(unit.replace("μ", "u"), 1.0)
    lod_scaled = lod / user_scale
    loq_scaled = loq / user_scale

    # plot
    out_path = Path(outdir) / safe_sample_name(sample)
    out_path.mkdir(parents=True, exist_ok=True)
    plot_path = out_path / "calibration_plot.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.scatter(g[found] / user_scale, (y_pred + blank_mean) / user_scale,
                   color="#0072B2", s=60, label="Data")
        min_v = min(g[found].min(), float((y_pred + blank_mean).min())) / user_scale
        max_v = max(g[found].max(), float((y_pred + blank_mean).max())) / user_scale
        ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="1:1 line")
        ax.set_xlabel(f"Measured Pb²⁺ ({unit})")
        ax.set_ylabel(f"Fitted Pb²⁺ ({unit})")
        ax.set_title(f"Calibration – {sample}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)

    return {
        "sample": sample,
        "n_points": len(g),
        "blank_mean": blank_mean / user_scale,
        "blank_sd": blank_sd / user_scale,
        "slope": float(slope),          # dimensionless – do NOT scale
        "slope_se": float(slope_se),
        "intercept": float(intercept) / user_scale,
        "intercept_se": float(intercept_se) / user_scale,
        "r_squared": float(r2),
        "mae": float(mae) / user_scale,
        "rmse": float(rmse) / user_scale,
        "lod": lod_scaled,
        "loq": loq_scaled,
        "plot_path": str(plot_path),
    }


# ---------- results ----------
def save_results(results: List[Dict[str, Any]], output_dir: str, unit: str) -> None:
    out_path = Path(output_dir)
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Cannot create output directory: {e}") from e

    ascii_u = ascii_unit(unit)
    rows = [
        {
            "Sample": r["sample"],
            "n_points": r["n_points"],
            f"blank_mean ({ascii_u})": r["blank_mean"],
            f"blank_sd ({ascii_u})": r["blank_sd"],
            "slope": r["slope"],
            "slope_se": r["slope_se"],
            f"intercept ({ascii_u})": r["intercept"],
            "intercept_se": r["intercept_se"],
            "r_squared": r["r_squared"],
            f"mae ({ascii_u})": r["mae"],
            f"rmse ({ascii_u})": r["rmse"],
            f"lod ({ascii_u})": r["lod"],
            f"loq ({ascii_u})": r["loq"],
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_path / "calibration_results.csv", index=False)
    df[["Sample", "slope", f"lod ({ascii_u})", f"loq ({ascii_u})", "r_squared"]].to_csv(
        out_path / "simple_results.csv", index=False
    )
    logger.info(f"Results saved to {out_path.resolve()}")


def create_demo_data() -> pd.DataFrame:
    # Synthetic data for quick sanity check only – not used in any real calibration
    return pd.DataFrame(
        {
            "Sample": ["Tap water"] * 5 + ["Experiment water"] * 5,
            "Added_ng_mL": [0.0, 0.0, 0.01, 1.0, 100.0] * 2,
            "PEC_found_ng_mL": [
                0.001,
                0.0012,
                0.011,
                1.177,
                117.287,
                0.001,
                0.0011,
                0.009,
                1.027,
                107.297,
            ],
            "Recovery_percent": [np.nan, np.nan, 97.33, 117.54, 117.29,
                                 np.nan, np.nan, 80.06, 102.58, 107.3],
            "RSD_percent": [4.56, 4.60, 2.41, 3.54, 3.4,
                            4.49, 4.50, 4.71, 4.39, 3.71],
        }
    )


# ---------- CLI ----------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pb²⁺ Biosensor WLS Calibration\n\n"
        "Examples:\n"
        "  python calibrate.py data.csv --unit ng/mL --outdir results\n"
        "  python calibrate.py data.csv --unit μg/L --outdir out_ugL\n"
        "  python calibrate.py --demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", nargs="?", help="Input CSV file")
    parser.add_argument("--outdir", default="calibration_results", help="Output directory")
    parser.add_argument("--default-sd", type=float, help="Default measurement SD (ng/mL)")
    parser.add_argument(
        "--unit",
        choices=VALID_UNITS,
        default="ng/mL",
        help="Concentration unit (outputs scaled to this unit)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--demo", action="store_true", help="Run with demo dataset")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--min-slope",
        type=float,
        default=1e-12,
        help="Minimum |slope| for meaningful LOD/LOQ (empirical default, Anal Chem 1983)",
    )
    parser.add_argument(
        "--small-number",
        type=float,
        default=1e-9,
        help="Epsilon against div-by-zero in weighting (empirical default)",
    )
    parser.add_argument(
        "--fallback-fraction",
        type=float,
        default=0.05,
        help="Fallback noise SD = fraction of mean response (empirical default)",
    )
    parser.add_argument(
        "--rsd-clip",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(0.1, 99.9),
        help="Clip RSD (%%) before converting to weight",
    )

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # validate numeric args
    for name, val in (
        ("--min-slope", args.min_slope),
        ("--small-number", args.small_number),
        ("--fallback-fraction", args.fallback_fraction),
    ):
        if val <= 0 or not np.isfinite(val):
            parser.error(f"{name} must be positive and finite")

    tmp_path: Optional[Path] = None
    try:
        if args.demo:
            logger.info("Creating and using demo dataset")
            tmp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, prefix="pb_demo_", dir=tempfile.gettempdir()
            )
            tmp_path = Path(tmp_file.name)
            create_demo_data().to_csv(tmp_file, index=False)
            tmp_file.close()
            csv = str(tmp_path)
        else:
            if not args.csv:
                parser.error("Must provide CSV file or use --demo")
            csv = args.csv

        df, colmap = load_data(csv)
        unit_scale = UNIT_SCALE_TO_NG_PER_ML.get(args.unit.replace("μ", "u"), 1.0)
        if unit_scale != 1.0:
            logger.info(f"Scaling concentrations to ng/mL (factor {unit_scale})")
            df[colmap["Added_ng_mL"]] *= unit_scale
            df[colmap["PEC_found_ng_mL"]] *= unit_scale
            validate_data(df, colmap)

        results: List[Dict[str, Any]] = []
        for sample, group in df.groupby(colmap["Sample"]):
            try:
                validate_group(group, colmap)
                sd_est = estimate_noise_sd(
                    group, args.default_sd, colmap, args.fallback_fraction
                )
                res = wls_calibration(
                    group=group,
                    sd_y=sd_est,
                    outdir=args.outdir,
                    unit=args.unit,
                    column_map=colmap,
                    rsd_clip=args.rsd_clip,
                    small_number=args.small_number,
                    min_slope=args.min_slope,
                )
                results.append(res)
            except Exception as e:
                logger.exception(f"Skipping sample '{sample}': {e}")
                continue

        if not results:
            logger.error("No successful calibrations")
            sys.exit(1)

        save_results(results, args.outdir, args.unit)
        logger.info(f"Done – results in {Path(args.outdir).resolve()}")

    except Exception as e:
        logger.exception("Fatal error")
        sys.exit(1)

    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                logger.debug("Temp demo file left for inspection")


if __name__ == "__main__":
    main()
