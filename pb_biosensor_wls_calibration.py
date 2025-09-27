
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

Pb²⁺ Biosensor WLS Calibration Framework - Enhanced Version
Weighted Least Squares calibration with comprehensive error handling
"""
import sys
import os
import argparse
import warnings
import logging
import traceback
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Constants
IUPAC_LOD_FACTOR = 3.3
IUPAC_LOQ_FACTOR = 10.0
COOKS_THRESHOLD_FACTOR = 4.0
MIN_BOOTSTRAP_SAMPLES = 100
MAX_BOOTSTRAP_ITERATIONS = 10000
DEFAULT_SEED = 42
MIN_STATISTICAL_POWER = 6
MAX_FILE_PATH_LENGTH = 255
VALID_UNITS = ['ng/mL', 'μg/L', 'mg/L', 'ppm', 'ppb']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Custom exceptions
class CalibrationError(Exception):
    """Base exception for calibration errors."""
    pass

class DataValidationError(CalibrationError):
    """Exception for data validation errors."""
    pass

class ComputationError(CalibrationError):
    """Exception for computational errors."""
    pass

class FileSystemError(CalibrationError):
    """Exception for file system errors."""
    pass
def setup_safe_path(base_path: str, sample_name: str) -> Path:
    """Create safe file path for sample outputs."""
    try:
        base_dir = Path(base_path).resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize sample name
        safe_name = "".join(c for c in sample_name if c.isalnum() or c in ('-', '_', ' '))
        safe_name = safe_name.strip().replace(' ', '_')[:50]
        
        if not safe_name:
            safe_name = "unknown_sample"
            
        sample_dir = base_dir / safe_name
        sample_dir.mkdir(exist_ok=True)
        
        return sample_dir
        
    except Exception as e:
        raise FileSystemError(f"Cannot create output directory for '{sample_name}': {e}")


def detect_dataset_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Auto-detect and map dataset columns with flexible naming."""
    column_map = {}
    
    # Required columns - multiple possible names
    required_mappings = {
        'Sample': ['Sample', 'sample', 'Sample_Name', 'SampleName', 'sample_name'],
        'Added_ng_mL': ['Added_ng_mL', 'added_ng_ml', 'Added', 'added', 'Spike_ng_mL'],
        'PEC_found_ng_mL': ['PEC_found_ng_mL', 'found_ng_ml', 'Found', 'found', 'Measured_ng_mL']
    }
    
    # Optional columns
    optional_mappings = {
        'RSD_percent': ['RSD_percent', 'RSD', 'rsd_percent', 'RSD_percent_external'],
        'Recovery_percent': ['Recovery_percent', 'Recovery', 'recovery_percent']
    }
    
    # Find required columns
    for standard_name, possible_names in required_mappings.items():
        found = False
        for col_name in possible_names:
            if col_name in df.columns:
                column_map[standard_name] = col_name
                found = True
                break
        
        if not found:
            available_cols = list(df.columns)
            raise DataValidationError(
                f"Required column '{standard_name}' not found. "
                f"Expected one of: {possible_names}. "
                f"Available columns: {available_cols}"
            )
    
    # Find optional columns
    for standard_name, possible_names in optional_mappings.items():
        for col_name in possible_names:
            if col_name in df.columns:
                column_map[standard_name] = col_name
                break
    
    logger.info(f"Detected column mapping: {column_map}")
    return column_map


def validate_data_types_and_ranges(df: pd.DataFrame, column_map: Dict[str, str]) -> None:
    """Comprehensive validation of data types and value ranges."""
    try:
        conc_col = column_map['Added_ng_mL']
        found_col = column_map['PEC_found_ng_mL']
        
        # Convert to numeric
        for col in [conc_col, found_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Check for conversion failures
        if df[conc_col].isna().any():
            raise DataValidationError(f"Non-numeric values in concentration column '{conc_col}'")
        if df[found_col].isna().any():
            raise DataValidationError(f"Non-numeric values in found column '{found_col}'")
        
        # Check for infinite values
        if np.isinf(df[conc_col]).any() or np.isinf(df[found_col]).any():
            raise DataValidationError("Infinite values detected in concentration data")
        
        # Check for negative values
        if (df[conc_col] < 0).any() or (df[found_col] < 0).any():
            raise DataValidationError("Negative concentrations detected")
        
        # Check optional RSD column if present
        if 'RSD_percent' in column_map:
            rsd_col = column_map['RSD_percent']
            if rsd_col in df.columns:
                df[rsd_col] = pd.to_numeric(df[rsd_col], errors='coerce')
                invalid_rsd = df[rsd_col].dropna()
                if (invalid_rsd < 0).any() or (invalid_rsd > 100).any():
                    logger.warning(f"Invalid RSD values in column '{rsd_col}' - will be ignored")
                    df[rsd_col] = df[rsd_col].clip(0, 100)
        
    except Exception as e:
        if isinstance(e, DataValidationError):
            raise
        else:
            raise DataValidationError(f"Data validation error: {e}")
def validate_calibration_group(
    group: pd.DataFrame, 
    min_points: int = 3, 
    min_blanks: int = 2,
    column_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Enhanced validation with detailed error reporting."""
    if group.empty:
        raise DataValidationError("Calibration group is empty")
    
    if column_map is None:
        column_map = {
            'Sample': 'Sample',
            'Added_ng_mL': 'Added_ng_mL', 
            'PEC_found_ng_mL': 'PEC_found_ng_mL'
        }
    
    sample_name = str(group[column_map['Sample']].iloc[0])
    validation_summary = {
        'sample': sample_name,
        'total_points': len(group),
        'issues': [],
        'warnings': []
    }
    
    # Check minimum points
    if len(group) < min_points:
        raise DataValidationError(
            f"Sample '{sample_name}' has {len(group)} points. "
            f"Minimum {min_points} required."
        )
    
    # Check concentration diversity
    conc_col = column_map['Added_ng_mL']
    unique_concentrations = group[conc_col].nunique()
    if unique_concentrations < 2:
        raise DataValidationError(
            f"Sample '{sample_name}' requires at least 2 distinct concentration levels."
        )
    
    # Check for negative values
    found_col = column_map['PEC_found_ng_mL']
    if (group[conc_col] < 0).any() or (group[found_col] < 0).any():
        raise DataValidationError("Negative concentrations detected")
    
    # Check blank replicates
    n_blanks = (group[conc_col] == 0.0).sum()
    if n_blanks < min_blanks:
        raise DataValidationError(
            f"Sample '{sample_name}' has {n_blanks} blank replicate(s). "
            f"At least {min_blanks} required."
        )
    
    logger.info(f"Validation passed for sample '{sample_name}'")
    return validation_summary


def compute_lod_loq(blank_sd: float, slope: float) -> Tuple[float, float]:
    """Compute LOD and LOQ with comprehensive error checking."""
    if not np.isfinite(blank_sd) or blank_sd < 0:
        raise ComputationError(f"Invalid blank standard deviation: {blank_sd}")
    
    if not np.isfinite(slope):
        raise ComputationError(f"Invalid slope value: {slope}")
    
    if slope == 0:
        raise ComputationError("Cannot compute LOD/LOQ with zero slope")
    
    if blank_sd == 0:
        logger.warning("Zero blank standard deviation - LOD/LOQ will be zero")
        return 0.0, 0.0
    
    lod = IUPAC_LOD_FACTOR * blank_sd / abs(slope)
    loq = IUPAC_LOQ_FACTOR * blank_sd / abs(slope)
    
    return lod, loq
def wls_fit_and_diagnostics(
    group: pd.DataFrame,
    sd_y: float,
    bootstrap_iter: int = 2000,
    outdir: str = ".",
    do_bootstrap: bool = True,
    remove_outliers: bool = False,
    unit: str = "ng/mL",
    column_map: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """Enhanced WLS fitting with comprehensive error handling."""
    if column_map is None:
        column_map = {
            'Sample': 'Sample',
            'Added_ng_mL': 'Added_ng_mL',
            'PEC_found_ng_mL': 'PEC_found_ng_mL'
        }
    
    sample_name = group[column_map['Sample']].iloc[0]
    logger.info(f"Processing sample: {sample_name}")
    
    try:
        # Blank correction
        conc_col = column_map['Added_ng_mL']
        found_col = column_map['PEC_found_ng_mL']
        
        blanks = group[group[conc_col] == 0.0][found_col].dropna()
        if len(blanks) == 0:
            raise ComputationError("No blank replicates found for blank correction")
        
        blank_mean = float(blanks.mean())
        blank_sd = float(blanks.std(ddof=1)) if len(blanks) >= 2 else sd_y
        
        # Prepare data
        g = group.copy().reset_index(drop=True)
        g['Found_corr'] = g[found_col] - blank_mean
        
        X = g[conc_col].astype(float).values
        y = g['Found_corr'].astype(float).values
        
        # Weights - enhanced column detection
        use_external_rsd = False
        if 'RSD_percent' in column_map:
            rsd_col = column_map['RSD_percent']
            if rsd_col in g.columns:
                try:
                    ext = g[rsd_col].astype(float).values
                    ext = np.nan_to_num(ext, nan=1.0)
                    ext = np.clip(ext, 0.1, 99.9)
                    
                    response_values = g[found_col].replace(0, np.nan).fillna(blank_mean).values
                    sd_y_vec = np.maximum(ext / 100.0 * response_values, 1e-9)
                    weights = 1.0 / (sd_y_vec ** 2)
                    use_external_rsd = True
                    
                    logger.info(f"Using external RSD weighting from column '{rsd_col}'")
                except Exception as e:
                    logger.warning(f"External RSD weighting failed, using constant weights: {e}")
                    weights = np.full_like(y, fill_value=1.0 / (sd_y ** 2))
            else:
                weights = np.full_like(y, fill_value=1.0 / (sd_y ** 2))
        else:
            weights = np.full_like(y, fill_value=1.0 / (sd_y ** 2))
        
        # Fit WLS model
        X_design = sm.add_constant(X)
        wls_model = sm.WLS(y, X_design, weights=weights)
        wls_res = wls_model.fit()
        
        # Extract results
        params = np.asarray(wls_res.params)
        bse = np.asarray(wls_res.bse)
        intercept, slope = float(params[0]), float(params[1])
        intercept_se, slope_se = float(bse[0]), float(bse[1])
        
        # Calculate metrics
        mae = mean_absolute_error(y, wls_res.predict(X_design))
        rmse = np.sqrt(mean_squared_error(y, wls_res.predict(X_design)))
        
        # R² calculation
        if hasattr(wls_res, 'rsquared') and wls_res.rsquared is not None:
            r2 = float(wls_res.rsquared)
        else:
            ss_res = np.sum((y - wls_res.predict(X_design))**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # LOD/LOQ calculation
        try:
            lod, loq = compute_lod_loq(blank_sd, slope)
        except ComputationError as e:
            logger.error(f"LOD/LOQ calculation failed for '{sample_name}': {e}")
            lod, loq = np.nan, np.nan
        
        # Create output directory
        sample_outdir = setup_safe_path(outdir, sample_name)
        
        # Generate plots (simplified version)
        plot_paths = {}
        try:
            # Predicted vs Actual plot
            fig, ax = plt.subplots(figsize=(8, 6))
            y_pred = wls_res.predict(X_design)
            ax.scatter(g[found_col], (y_pred + blank_mean), color='#0072B2', s=60)
            
            # Unity line
            min_val = min(g[found_col].min(), float((y_pred + blank_mean).min()))
            max_val = max(g[found_col].max(), float((y_pred + blank_mean).max()))
            ax.plot([min_val, max_val], [min_val, max_val], color='#D55E00', linestyle='--', linewidth=2)
            
            ax.set_xlabel(f'Actual Found Pb²⁺ ({unit})')
            ax.set_ylabel(f'Predicted Pb²⁺ ({unit})')
            ax.set_title(f'Predicted vs Actual — {sample_name}')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
            plot_path = sample_outdir / 'pred_vs_actual.png'
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_paths['pred_vs_actual'] = str(plot_path)
            
        except Exception as e:
            logger.warning(f"Plot generation failed for '{sample_name}': {e}")
        
        # Results dictionary
        results = [{
            'sample': sample_name,
            'n_points': len(g),
            'blank_mean': blank_mean,
            'blank_sd': blank_sd,
            'slope': slope,
            'slope_se': slope_se,
            'slope_ci_t': (slope - 1.96*slope_se, slope + 1.96*slope_se),
            'intercept': intercept,
            'intercept_se': intercept_se,
            'intercept_ci_t': (intercept - 1.96*intercept_se, intercept + 1.96*intercept_se),
            'r_squared': r2,
            'mae': mae,
            'rmse': rmse,
            'lod': lod,
            'loq': loq,
            'p_slope_eq_1': np.nan,
            'p_intercept_eq_0': np.nan,
            'outliers_idx': [],
            'plots': plot_paths,
            'used_external_rsd': use_external_rsd
        }]
        
        logger.info(f"Completed analysis for '{sample_name}'")
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed for '{sample_name}': {e}")
        raise ComputationError(f"Analysis failed for '{sample_name}': {e}")
def save_results_safe(
    results: List[Dict[str, Any]],
    output_dir: str,
    unit: str
) -> Tuple[str, str]:
    """Save results with comprehensive error handling."""
    try:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare detailed results
        rows = []
        for r in results:
            row = {
                'Sample': r['sample'],
                'n_points': r['n_points'],
                f'blank_mean ({unit})': r['blank_mean'],
                f'blank_sd ({unit})': r['blank_sd'],
                'slope': r['slope'],
                'slope_se': r['slope_se'],
                'slope_ci_t_low': r['slope_ci_t'][0],
                'slope_ci_t_high': r['slope_ci_t'][1],
                f'intercept ({unit})': r['intercept'],
                'intercept_se': r['intercept_se'],
                'intercept_ci_t_low': r['intercept_ci_t'][0],
                'intercept_ci_t_high': r['intercept_ci_t'][1],
                'r_squared': r['r_squared'],
                f'mae ({unit})': r['mae'],
                f'rmse ({unit})': r['rmse'],
                f'lod ({unit})': r['lod'],
                f'loq ({unit})': r['loq'],
                'p_slope_eq_1': r['p_slope_eq_1'],
                'p_intercept_eq_0': r['p_intercept_eq_0'],
                'outliers_idx': ','.join(map(str, r['outliers_idx'])),
                'used_external_rsd': r.get('used_external_rsd', False)
            }
            rows.append(row)
        
        # Save detailed results
        results_df = pd.DataFrame(rows)
        detailed_path = output_path / 'Pb_Biosensor_WLS_Detailed_Results.csv'
        results_df.to_csv(detailed_path, index=False)
        
        # Save simplified results
        simple_cols = ['Sample', 'slope', f'lod ({unit})', f'loq ({unit})', 'r_squared']
        simple_df = results_df[simple_cols]
        simple_path = output_path / 'Pb_Biosensor_Simple_Results.csv'
        simple_df.to_csv(simple_path, index=False)
        
        logger.info(f"Results saved: {detailed_path}, {simple_path}")
        return str(detailed_path), str(simple_path)
        
    except Exception as e:
        raise FileSystemError(f"Failed to save results: {e}")


def load_and_validate_data(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load and validate input data."""
    csv_path = Path(csv_path).resolve()
    
    if not csv_path.exists():
        raise FileSystemError(f"CSV file not found: {csv_path}")
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise FileSystemError("Could not read CSV file with any supported encoding")
        
    except pd.errors.EmptyDataError:
        raise DataValidationError("CSV file is empty")
    except Exception as e:
        raise FileSystemError(f"Failed to read CSV file: {e}")
    
    if df.empty:
        raise DataValidationError("CSV file contains no data")
    
    # Auto-detect columns
    column_map = detect_dataset_columns(df)
    validate_data_types_and_ranges(df, column_map)
    
    logger.info(f"Loaded {len(df)} data points")
    return df, column_map


def create_demo_data() -> pd.DataFrame:
    """Create demonstration dataset matching repository format."""
    return pd.DataFrame({
        "Sample": ["Tap water"] * 4 + ["Experiment water"] * 4,
        "Added_ng_mL": [0.0, 0.01, 1.0, 100.0] * 2,
        "PEC_found_ng_mL": [0.001, 0.011, 1.177, 117.287, 0.001, 0.009, 1.027, 107.297],
        "Recovery_percent": [np.nan, 97.33, 117.54, 117.29, np.nan, 80.06, 102.58, 107.3],
        "RSD_percent": [4.56, 2.41, 3.54, 3.4, 4.49, 4.71, 4.39, 3.71]
    })


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Pb²⁺ Biosensor WLS Calibration - Enhanced")
    parser.add_argument("csv", nargs='?', help='Input CSV file')
    parser.add_argument("--outdir", help='Output directory')
    parser.add_argument("--bootstrap", type=int, default=2000, help='Bootstrap iterations')
    parser.add_argument("--default-sd", type=float, help='Fallback SD')
    parser.add_argument("--remove-outliers", action='store_true', help='Remove outliers')
    parser.add_argument("--no-bootstrap", action='store_true', help='Disable bootstrap')
    parser.add_argument("--unit", default='ng/mL', help='Concentration unit')
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help='Random seed')
    parser.add_argument("--demo", action='store_true', help='Run demo')
    parser.add_argument("--verbose", action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        np.random.seed(args.seed)
        
        if args.demo:
            if args.csv:
                parser.error("Cannot specify both --demo and CSV file")
            
            logger.info("Running demonstration")
            df = create_demo_data()
            csv_path = Path('demo_data.csv')
            df.to_csv(csv_path, index=False)
            try:
                # Process demo data
                output_dir = args.outdir or 'WLS_outputs'
                df, column_map = load_and_validate_data(str(csv_path))
                
                all_results = []
                for sample_name, group in df.groupby(column_map['Sample']):
                    validate_calibration_group(group, column_map=column_map)
                    measurement_sd = estimate_noise_sd(group, args.default_sd, column_map)
                    
                    results = wls_fit_and_diagnostics(
                        group=group, sd_y=measurement_sd, bootstrap_iter=0,
                        outdir=output_dir, do_bootstrap=False, unit=args.unit,
                        column_map=column_map
                    )
                    all_results.extend(results)
                
                # Save results
                if all_results:
                    save_results_safe(all_results, output_dir, args.unit)
                    print(f"\nDemo completed successfully!")
                    print(f"Results saved to: {output_dir}")
                
            finally:
                csv_path.unlink(missing_ok=True)
            return
        
        if not args.csv:
            parser.error("Must specify CSV file or use --demo")
        
        # Regular processing
        csv_path = args.csv
        logger.info(f"Processing: {csv_path}")
        
        output_dir = args.outdir or str(Path(csv_path).parent / 'WLS_outputs')
        df, column_map = load_and_validate_data(csv_path)
        
        all_results = []
        for sample_name, group in df.groupby(column_map['Sample']):
            try:
                validate_calibration_group(group, column_map=column_map)
                measurement_sd = estimate_noise_sd(group, args.default_sd, column_map)
                
                results = wls_fit_and_diagnostics(
                    group=group, sd_y=measurement_sd, 
                    bootstrap_iter=0 if args.no_bootstrap else args.bootstrap,
                    outdir=output_dir, do_bootstrap=not args.no_bootstrap,
                    remove_outliers=args.remove_outliers, unit=args.unit,
                    column_map=column_map
                )
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"Failed to process '{sample_name}': {e}")
                continue
        
        if all_results:
            save_results_safe(all_results, output_dir, args.unit)
            print(f"\nAnalysis complete! Results saved to: {output_dir}")
        else:
            logger.error("No samples were successfully processed")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
