#!/usr/bin/env python3
"""
Pb²⁺ Biosensor WLS Calibration

Simple WLS calibration for Pb²⁺ biosensor data with bootstrap CIs and diagnostics.
Uses fixed seed (42) for reproducibility. See README for details.

- Validates data (≥3 points, ≥2 blanks, no negatives).
- Applies blank correction and WLS regression.
- Computes LOD/LOQ (IUPAC: 3.3σ/|slope|, 10σ/|slope|).
- Optional bootstrap and outlier removal via args.

CITATION:
If you use this software, please cite it as:

MD Rayhan & Bingqian Liu (2025).
Pb²⁺ Biosensor Calibration Script (Version 1.0.0).
Zenodo. https://doi.org/10.5281/zenodo.17212395
"""

import sys
import os
import argparse
import warnings
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Utility functions ---

def validate_calibration_group(group: pd.DataFrame, min_points: int = 3, min_blanks: int = 2) -> None:
    """Validate minimum requirements for a calibration group (sample/matrix)."""
    sample_name = group['Sample'].iloc[0]
    if len(group) < min_points:
        raise ValueError(f"Sample '{sample_name}' has {len(group)} points. Minimum {min_points} required.")
    if len(group) < 6:
        warnings.warn(f"Sample '{sample_name}' has {len(group)} points. At least 6 recommended for statistical power.", UserWarning)
    if group['Added_ng_mL'].nunique() < 2:
        raise ValueError(f"Sample '{sample_name}' requires at least 2 distinct Added concentration levels.")
    if (group['Added_ng_mL'] < 0).any() or (group['PEC_found_ng_mL'] < 0).any():
        raise ValueError(f"Sample '{sample_name}' has negative concentrations; check input data.")
    n_blanks = (group['Added_ng_mL'] == 0.0).sum()
    if n_blanks < min_blanks:
        raise ValueError(f"Sample '{sample_name}' has {n_blanks} blank replicate(s). At least {min_blanks} required.")
    if n_blanks < 10:
        warnings.warn(f"Sample '{sample_name}' has only {n_blanks} blank replicate(s). "
                      "IUPAC recommends ≥10 for reliable LOD/LOQ.", UserWarning)

def compute_lod_loq(blank_sd: float, slope: float) -> Tuple[float, float]:
    """Compute LOD and LOQ using IUPAC formulas (sd_blank in same units as response)."""
    if slope == 0 or np.isnan(slope):
        return np.nan, np.nan
    lod = 3.3 * blank_sd / abs(slope)
    loq = 10.0 * blank_sd / abs(slope)
    return lod, loq

def estimate_noise_sd(group: pd.DataFrame, default_sd: float = None) -> float:
    """
    Estimate measurement SD (not circular):
    - Prefer standard deviation of blank replicates (Added == 0)
    - Else use pooled SD of lowest non-zero spike replicates if multiple exist
    - Else use user-specified default_sd or a conservative floor (1e-3)
    """
    blanks = group[group['Added_ng_mL'] == 0.0]['PEC_found_ng_mL'].dropna()
    if len(blanks) >= 2:
        return float(blanks.std(ddof=1))
    nonzero = group[group['Added_ng_mL'] > 0]
    if not nonzero.empty:
        min_added = nonzero['Added_ng_mL'].min()
        low_reps = nonzero[nonzero['Added_ng_mL'] == min_added]['PEC_found_ng_mL'].dropna()
        if len(low_reps) >= 2:
            return float(low_reps.std(ddof=1))
    return default_sd if default_sd is not None else 1e-3  # Default to 0.001

def wls_fit_and_diagnostics(group: pd.DataFrame, sd_y: float, bootstrap_iter: int = 2000,
                            use_external_rsd_col: str = None, outdir: str = ".",
                            do_bootstrap: bool = True, remove_outliers: bool = False,
                            unit: str = "ng/mL") -> List[Dict[str, Any]]:
    """
    Fit WLS to blank-corrected response and return metrics and plots.
    If remove_outliers=True and outliers are detected, also fit a model excluding outliers.
    """
    sample_name = group['Sample'].iloc[0]
    # Blank correction
    blanks = group[group['Added_ng_mL'] == 0.0]['PEC_found_ng_mL'].dropna()
    if len(blanks) == 0:
        raise ValueError(f"No blank replicates found for sample '{sample_name}'. Blank correction required.")
    blank_mean = float(blanks.mean())
    blank_sd = float(blanks.std(ddof=1)) if len(blanks) >= 2 else sd_y

    # Prepare data
    g = group.copy().reset_index(drop=True)
    g['Found_corr'] = g['PEC_found_ng_mL'] - blank_mean
    X = g['Added_ng_mL'].astype(float).values
    y = g['Found_corr'].astype(float).values

    # Weights (avoid circular per-point RSD)
    if use_external_rsd_col and use_external_rsd_col in g.columns:
        ext = g[use_external_rsd_col].astype(float).values
        ext = np.nan_to_num(ext, nan=1.0)
        ext = np.clip(ext, 0.1, 99.9)
        sd_y_vec = np.maximum(ext / 100.0 * g['PEC_found_ng_mL'].replace(0, np.nan).fillna(blank_mean).values, 1e-9)
        weights = 1.0 / (sd_y_vec ** 2)
    else:
        weights = np.full_like(y, fill_value=(1.0 / (sd_y ** 2)))

    # Design matrix with intercept
    X_design = sm.add_constant(X)
    wls_model = sm.WLS(y, X_design, weights=weights)
    wls_res = wls_model.fit()

    # Predictions and CI (parametric)
    try:
        pred_frame = wls_res.get_prediction(X_design).summary_frame(alpha=0.05)
        y_pred = pred_frame['mean'].values
        ci_lower = pred_frame['mean_ci_lower'].values
        ci_upper = pred_frame['mean_ci_upper'].values
    except Exception:
        y_pred = wls_res.predict(X_design)
        ci_lower = np.full_like(y_pred, np.nan)
        ci_upper = np.full_like(y_pred, np.nan)

    # Metrics on blank-corrected scale
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = float(wls_res.rsquared) if hasattr(wls_res, 'rsquared') else 1 - wls_res.ssr / np.sum((y - y.mean())**2)

    # Params and standard errors
    params = np.asarray(wls_res.params)  # [intercept, slope]
    bse = np.asarray(wls_res.bse)
    intercept, slope = float(params[0]), float(params[1])
    intercept_se, slope_se = float(bse[0]), float(bse[1])

    # t-based CIs
    df_resid = int(wls_res.df_resid)
    tval = stats.t.ppf(0.975, df_resid) if df_resid > 0 else np.nan
    slope_ci = (slope - tval * slope_se, slope + tval * slope_se) if df_resid > 0 else (np.nan, np.nan)
    intercept_ci = (intercept - tval * intercept_se, intercept + tval * intercept_se) if df_resid > 0 else (np.nan, np.nan)

    # Hypothesis tests (t-based)
    p_slope_eq1 = np.nan
    p_intercept_eq0 = np.nan
    if slope_se > 0 and df_resid > 0:
        t_s = (slope - 1.0) / slope_se
        p_slope_eq1 = 2 * (1 - stats.t.cdf(abs(t_s), df_resid))
        t_i = (intercept - 0.0) / intercept_se
        p_intercept_eq0 = 2 * (1 - stats.t.cdf(abs(t_i), df_resid))
        if df_resid < 5:
            warnings.warn(f"Sample '{sample_name}' has low statistical power (df={df_resid}); t-tests may be unreliable.", UserWarning)
    else:
        warnings.warn(f"Sample '{sample_name}' has insufficient degrees of freedom; t-tests skipped.", UserWarning)

    # LOD/LOQ
    lod, loq = compute_lod_loq(blank_sd, slope)

    # Bootstrap CIs
    bootstrap_res = {}
    if do_bootstrap:
        n = len(g)
        if n < 6:
            warnings.warn(f"Sample '{sample_name}' has {n} points; bootstrap disabled due to limited resampling space.", UserWarning)
            do_bootstrap = False
    if do_bootstrap:
        slopes_b, intercepts_b, lods_b, loqs_b = [], [], [], []
        blanks_values = blanks.values
        for _ in range(bootstrap_iter):
            idx = np.random.randint(0, n, n)
            gb = g.iloc[idx].reset_index(drop=True)
            weights_b = weights[idx]  # Resample weights
            if len(blanks_values) >= 2:
                b_sample = np.random.choice(blanks_values, size=len(blanks_values), replace=True)
                bmean_bs = float(np.mean(b_sample))
                bsd_bs = float(np.std(b_sample, ddof=1)) if len(b_sample) > 1 else blank_sd
            else:
                bmean_bs, bsd_bs = blank_mean, blank_sd
            gb['Found_corr'] = gb['PEC_found_ng_mL'] - bmean_bs
            Xb = sm.add_constant(gb['Added_ng_mL'].astype(float).values)
            yb = gb['Found_corr'].astype(float).values
            try:
                wls_b = sm.WLS(yb, Xb, weights=weights_b).fit()
                params_b = np.asarray(wls_b.params)
                intercepts_b.append(params_b[0])
                slopes_b.append(params_b[1])
                lod_b, loq_b = compute_lod_loq(bsd_bs, float(params_b[1]))
                lods_b.append(lod_b)
                loqs_b.append(loq_b)
            except Exception:
                continue
        if len(slopes_b) >= 100:
            bootstrap_res['slope_ci_boot'] = (np.percentile(slopes_b, 2.5), np.percentile(slopes_b, 97.5))
            bootstrap_res['intercept_ci_boot'] = (np.percentile(intercepts_b, 2.5), np.percentile(intercepts_b, 97.5))
            bootstrap_res['lod_ci_boot'] = (np.percentile(lods_b, 2.5), np.percentile(lods_b, 97.5))
            bootstrap_res['loq_ci_boot'] = (np.percentile(loqs_b, 2.5), np.percentile(loqs_b, 97.5))
        else:
            warnings.warn(f"Sample '{sample_name}' has too few valid bootstrap resamples (<100); CIs set to NaN.", UserWarning)

    # Diagnostics: Standardized residuals, Cook's distance (using OLS for influence)
    ols_res = sm.OLS(y, X_design).fit()
    influence = ols_res.get_influence()
    standardized_resid = influence.resid_studentized_internal
    cooks_d = influence.cooks_distance[0]
    cooks_threshold = 4.0 / len(g)
    outlier_idx = np.where(cooks_d > cooks_threshold)[0].tolist()

    # Plots
    sample_outdir = os.path.join(outdir, sample_name.replace(' ', '_'))
    os.makedirs(sample_outdir, exist_ok=True)

    # Predicted vs Actual
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(g['PEC_found_ng_mL'], (y_pred + blank_mean), color='#0072B2', label='Predicted', zorder=3)
    mn = min(g['PEC_found_ng_mL'].min(), float((y_pred + blank_mean).min()))
    mx = max(g['PEC_found_ng_mL'].max(), float((y_pred + blank_mean).max()))
    ax1.plot([mn, mx], [mn, mx], color='#D55E00', linestyle='--', label='Ideal (y=x)')
    if not np.all(np.isnan(ci_lower)):
        ax1.fill_between(g['PEC_found_ng_mL'].values, (ci_lower + blank_mean), (ci_upper + blank_mean),
                         color='gray', alpha=0.25, label='95% mean CI')
    ax1.set_xlabel(f"Actual Found Pb²⁺ ({unit})")
    ax1.set_ylabel(f"Predicted Pb²⁺ ({unit})")
    ax1.set_title(f"Predicted vs Actual — {sample_name}")
    ax1.legend()
    ax1.grid(True, linestyle=':', linewidth=0.4)
    fig1.tight_layout()
    fig1_path = os.path.join(sample_outdir, "Pred_vs_Actual.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    # Standardized residuals
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(y_pred, standardized_resid, color='#0072B2', zorder=3)
    ax2.axhline(0, color='#D55E00', linestyle='--')
    ax2.set_xlabel(f"Predicted Pb²⁺ ({unit}, blank-corrected)")
    ax2.set_ylabel("Standardized Residuals")
    ax2.set_title(f"Std Residuals vs Predicted — {sample_name}")
    ax2.grid(True, linestyle=':', linewidth=0.4)
    fig2.tight_layout()
    fig2_path = os.path.join(sample_outdir, "StdResid_vs_Pred.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    # Q-Q plot
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    sm.qqplot(standardized_resid, line='45', ax=ax3, linecolor='#D55E00', markerfacecolor='#0072B2')
    ax3.set_xlabel(f"Theoretical Quantiles")
    ax3.set_ylabel(f"Standardized Residuals")
    ax3.set_title(f"Q-Q Plot Residuals — {sample_name}")
    fig3.tight_layout()
    fig3_path = os.path.join(sample_outdir, "QQ.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close(fig3)

    # Cook's distance
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    ax4.stem(np.arange(len(cooks_d)), cooks_d, markerfmt="#0072B2", basefmt=" ", use_line_collection=True)
    ax4.axhline(cooks_threshold, color='#D55E00', linestyle='--', label=f'4/n={cooks_threshold:.3g}')
    ax4.set_xlabel("Index")
    ax4.set_ylabel("Cook's Distance (OLS)")
    ax4.set_title(f"Cook's Distance — {sample_name}")
    ax4.legend()
    fig4.tight_layout()
    fig4_path = os.path.join(sample_outdir, "Cooks.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close(fig4)

    # Results dictionary
    results = [{
        'sample': sample_name,
        'n_points': len(g),
        'blank_mean': blank_mean,
        'blank_sd': blank_sd,
        'slope': slope,
        'slope_se': slope_se,
        'slope_ci_t': slope_ci,
        'intercept': intercept,
        'intercept_se': intercept_se,
        'intercept_ci_t': intercept_ci,
        'r_squared': r2,
        'mae': mae,
        'rmse': rmse,
        'lod': lod,
        'loq': loq,
        'p_slope_eq_1': p_slope_eq1,
        'p_intercept_eq_0': p_intercept_eq0,
        'outliers_idx': outlier_idx,
        'plots': {'pred_vs_actual': fig1_path, 'stdres_vs_pred': fig2_path, 'qq': fig3_path, 'cooks': fig4_path}
    }]
    results[0].update(bootstrap_res)

    # Refit without outliers if requested
    if remove_outliers and outlier_idx:
        if len(outlier_idx) == len(g):
            warnings.warn(f"All points in '{sample_name}' flagged as outliers; skipping refit.", UserWarning)
        else:
            clean_g = g.drop(outlier_idx).reset_index(drop=True)
            try:
                validate_calibration_group(clean_g, min_points=3, min_blanks=2)
                clean_res = wls_fit_and_diagnostics(clean_g, sd_y=sd_y, bootstrap_iter=0, outdir=outdir,
                                                    do_bootstrap=False, remove_outliers=False, unit=unit)
                clean_res[0]['sample'] = f"{sample_name}_outliers_removed"
                results.append(clean_res[0])
            except ValueError as e:
                warnings.warn(f"Cannot refit '{sample_name}' without outliers: {e}", UserWarning)

    return results

# --- Main pipeline ---
def main(csv_path: str, outdir: str = None, bootstrap_iter: int = 2000, demo: bool = False,
         default_sd: float = None, remove_outliers: bool = False, no_bootstrap: bool = False,
         unit: str = "ng/mL", seed: int = 42):
    if outdir is None:
        outdir = os.path.join(os.path.dirname(csv_path), "WLS_outputs")
    os.makedirs(outdir, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Load data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_path}' not found.")
    required = {'Sample', 'Added_ng_mL', 'PEC_found_ng_mL'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

    overall_results = []
    for sample, group in df.groupby('Sample'):
        try:
            validate_calibration_group(group, min_points=3, min_blanks=2)
        except ValueError as e:
            print(f"Validation error for sample '{sample}': {e}", file=sys.stderr)
            continue

        sd_est = estimate_noise_sd(group, default_sd)
        res_list = wls_fit_and_diagnostics(group, sd_y=sd_est, bootstrap_iter=bootstrap_iter,
                                           use_external_rsd_col='RSD_percent_external' if 'RSD_percent_external' in group.columns else None,
                                           outdir=outdir, do_bootstrap=not no_bootstrap, remove_outliers=remove_outliers,
                                           unit=unit)
        overall_results.extend(res_list)

        # Print summary
        for res in res_list:
            print(f"\nSample: {res['sample']}")
            print(f" n={res['n_points']}, blank_mean={res['blank_mean']:.6g}, blank_sd={res['blank_sd']:.3g}")
            print(f" slope={res['slope']:.6g} (t-CI {res['slope_ci_t'][0]:.3g} — {res['slope_ci_t'][1]:.3g})")
            if 'slope_ci_boot' in res:
                print(f" bootstrap slope CI: {res['slope_ci_boot'][0]:.3g} — {res['slope_ci_boot'][1]:.3g}")
            print(f" intercept={res['intercept']:.6g}, R²={res['r_squared']:.6g}")
            print(f" LOD={res['lod']:.3g} {unit}, LOQ={res['loq']:.3g} {unit}")
            if res['outliers_idx']:
                print(f" WARNING: potential outliers at indices {res['outliers_idx']} (check Cook's distance)")

    # Save detailed results
    rows = []
    for r in overall_results:
        row = {
            f'Sample': r['sample'],
            f'n_points': r['n_points'],
            f'blank_mean ({unit})': r['blank_mean'],
            f'blank_sd ({unit})': r['blank_sd'],
            f'slope': r['slope'],
            f'slope_se': r['slope_se'],
            f'slope_ci_t_low': r['slope_ci_t'][0] if r['slope_ci_t'] else np.nan,
            f'slope_ci_t_high': r['slope_ci_t'][1] if r['slope_ci_t'] else np.nan,
            f'intercept ({unit})': r['intercept'],
            f'intercept_se': r['intercept_se'],
            f'intercept_ci_t_low': r['intercept_ci_t'][0] if r['intercept_ci_t'] else np.nan,
            f'intercept_ci_t_high': r['intercept_ci_t'][1] if r['intercept_ci_t'] else np.nan,
            f'r_squared': r['r_squared'],
            f'mae ({unit})': r['mae'],
            f'rmse ({unit})': r['rmse'],
            f'lod ({unit})': r['lod'],
            f'loq ({unit})': r['loq'],
            f'p_slope_eq_1': r['p_slope_eq_1'],
            f'p_intercept_eq_0': r['p_intercept_eq_0']
        }
        if 'slope_ci_boot' in r:
            row['slope_ci_boot_low'] = r['slope_ci_boot'][0]
            row['slope_ci_boot_high'] = r['slope_ci_boot'][1]
            row[f'lod_ci_boot_low ({unit})'] = r.get('lod_ci_boot', (np.nan, np.nan))[0]
            row[f'lod_ci_boot_high ({unit})'] = r.get('lod_ci_boot', (np.nan, np.nan))[1]
        rows.append(row)

    results_df = pd.DataFrame(rows)
    out_csv = os.path.join(outdir, "Pb_Biosensor_WLS_Robust_Results.csv")
    results_df.to_csv(out_csv, index=False)

    # Save simplified results
    simple_df = results_df[[f'Sample', f'slope', f'lod ({unit})', f'loq ({unit})', f'r_squared']]
    simple_df.to_csv(os.path.join(outdir, "Pb_Biosensor_Simple_Results.csv"), index=False)

    print(f"\nDetailed results saved to: {out_csv}")
    print(f"Simplified results saved to: {os.path.join(outdir, 'Pb_Biosensor_Simple_Results.csv')}")
    print(f"Plots saved in sample-specific subdirectories under: {outdir}")

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pb²⁺ Biosensor WLS Calibration (Robust)")
    parser.add_argument("csv", help="Input CSV (must contain Sample, Added_ng_mL, PEC_found_ng_mL)")
    parser.add_argument("--outdir", help="Output directory (default: same folder + /WLS_outputs)", default=None)
    parser.add_argument("--bootstrap", help="Bootstrap iterations (default 2000)", type=int, default=2000)
    parser.add_argument("--default_sd", type=float, default=None, help="Fallback SD if blanks/replicates insufficient")
    parser.add_argument("--remove_outliers", action="store_true", help="Refit model excluding outliers (Cook's distance > 4/n)")
    parser.add_argument("--no_bootstrap", action="store_true", help="Disable bootstrap CIs explicitly")
    parser.add_argument("--unit", default="ng/mL", help="Concentration unit (default: ng/mL)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap (default: 42)")
    parser.add_argument("--demo", help="Run demo on embedded dataset", action="store_true")
    args = parser.parse_args()

    if args.demo:
        demo_df = pd.DataFrame({
            "Sample": ["Tap water"]*6 + ["Experiment water"]*6,
            "Added_ng_mL": [0.0, 0.0, 0.01, 0.01, 1.0, 100.0]*2,
            "PEC_found_ng_mL": [0.001, 0.0011, 0.011, 0.0108, 1.177, 117.287, 0.001, 0.0009, 0.009, 0.0092, 1.027, 107.297]
        })
        demo_path = os.path.join(os.getcwd(), "demo_pb_data.csv")
        demo_df.to_csv(demo_path, index=False)
        try:
            main(demo_path, outdir=args.outdir, bootstrap_iter=args.bootstrap, demo=True,
                 default_sd=args.default_sd, remove_outliers=args.remove_outliers, no_bootstrap=args.no_bootstrap,
                 unit=args.unit, seed=args.seed)
        finally:
            os.remove(demo_path)
    else:
        main(args.csv, outdir=args.outdir, bootstrap_iter=args.bootstrap, demo=False,
             default_sd=args.default_sd, remove_outliers=args.remove_outliers, no_bootstrap=args.no_bootstrap,
             unit=args.unit, seed=args.seed)
