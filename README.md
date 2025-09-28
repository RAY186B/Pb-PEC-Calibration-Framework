```markdown
# Pb²⁺ Biosensor Calibration Script

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17212395.svg)](https://doi.org/10.5281/zenodo.17212395)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
![Python](https://img.shields.io/badge/python-≥3.8-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Purpose

**Pb-PEC-Calibration-Framework** is a scientific software package for calibrating Pb²⁺ (Lead ion) biosensor data using Weighted Least Squares (WLS) regression with bootstrap uncertainty estimation. The framework implements IUPAC-standard methods for calculating Limits of Detection (LOD) and Quantification (LOQ).

### Scientific Methodology

This framework employs:
- **Weighted Least Squares (WLS) regression** to account for heteroscedastic measurement uncertainty
- **Bootstrap resampling** (default 2000 iterations) for robust confidence interval estimation
- **IUPAC standard calculations**: LOD = 3.3σ/slope, LOQ = 10σ/slope
- **Comprehensive diagnostic plots** for calibration validation

## Installation

### System Requirements
- Python ≥ 3.8
- pip package manager

### Quick Install
```bash
# Clone the repository
git clone https://github.com/RAY186B/Pb-PEC-Calibration-Framework.git
cd Pb-PEC-Calibration-Framework

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies for testing
pip install -r requirements-dev.txt
```

Virtual Environment (Recommended)

```bash
python -m venv pb_calib_env
source pb_calib_env/bin/activate  # On Windows: pb_calib_env\Scripts\activate
pip install -r requirements.txt
```

Usage

Quick Start - Demo Mode

```bash
# Generate synthetic data and run calibration
python pb_biosensor_wls_calibration.py --demo
```

Real Calibration

```bash
# Basic usage
python pb_biosensor_wls_calibration.py your_data.csv

# With custom options
python pb_biosensor_wls_calibration.py your_data.csv \
    --unit ng/mL \
    --outdir results \
    --bootstrap 5000 \
    --ci 0.95
```

Input Data Format

Your CSV file should contain these columns:

Sample	Added_ng_mL	PEC_found_ng_mL	RSD_percent	
Tap water	0.0	0.001	4.5	
Tap water	0.01	0.011	2.4	
Tap water	1.0	1.177	3.5	

Required Columns:
- `Sample`: Sample identifier
- `Added_ng_mL`: Known concentration (spike level)
- `PEC_found_ng_mL`: Measured concentration

Optional Columns:
- `RSD_percent`: Relative standard deviation for weighting
- Alternative column names are auto-detected (e.g., "conc", "signal", "sd_y")

Command Line Options

```bash
python pb_biosensor_wls_calibration.py -h

positional arguments:
  data_csv              Path to input CSV file

optional arguments:
  --x-col TEXT         Concentration column name (default: conc)
  --y-col TEXT         Signal column name (default: signal)
  --sd-col TEXT        SD column name (default: sd_y)
  --group-col TEXT     Optional grouping column
  --bootstrap INT      Bootstrap iterations (default: 2000)
  --ci FLOAT           Confidence level (default: 0.975)
  --out-dir TEXT       Output directory (default: ./results)
```

Outputs

The framework generates:

1. Calibration Summary (`calibration_summary.csv`)

group	slope	intercept	lod	loq	r_squared	
Tap water	1.173	0.0012	0.0003	0.0010	0.9998	

2. Diagnostic Plots (per sample group)
- pred_vs_actual.png: Predicted vs. actual concentrations
- stdres_vs_pred.png: Standardized residuals vs. predicted values
- qq_plot.png: Q-Q plot for normality assessment
- cooks_distance.png: Cook's distance for outlier detection

3. Bootstrap Results (if enabled)
- Confidence intervals for slope and intercept
- Uncertainty estimates for LOD/LOQ values

Example Analysis

Using Provided Test Data

```bash
# Run calibration on included dataset
python pb_biosensor_wls_calibration.py Pb_Biosensor_Data.csv --unit ng/mL

# Results will be in ./results/calibration_summary.csv
```

Expected Results for Test Data
For the included `Pb_Biosensor_Data.csv`:
- Tap water: slope ≈ 1.17, LOD ≈ 0.0003 ng/mL
- Experiment water: slope ≈ 1.07, LOD ≈ 0.0004 ng/mL

Development & Testing

Running Tests

```bash
# Run full test suite
pytest test_pb_biosensor.py -v

# Run with coverage
pytest test_pb_biosensor.py --cov=pb_biosensor_wls_calibration --cov-report=html
```

Test Coverage
Current coverage: 95% with comprehensive tests for:
- Data validation and error handling
- Mathematical accuracy of calculations
- Edge cases and robustness
- Memory efficiency and performance

Development Transparency

This project was developed using traditional scientific programming methods with AI-assisted code review and documentation improvements.

Citation

If you use this software in your research, please cite:

```bibtex
@software{pb_pec_calibration_2025,
  title={Pb²⁺ Biosensor Calibration Script},
  author={Rayhan, MD and Liu, Bingqian},
  year={2025},
  url={https://github.com/RAY186B/Pb-PEC-Calibration-Framework},
  doi={10.5281/zenodo.17212395},
  version={1.0.0}
}
```

Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Development Setup

```bash
# Install in development mode
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests before committing
pytest test_pb_biosensor.py -v
```

Troubleshooting

Common Issues

"Missing required columns" error
- Verify your CSV has the required column names
- Check for typos in column headers
- Use `--x-col`, `--y-col`, `--sd-col` to specify custom names

"Weights must be strictly positive" error
- Check for zero or negative values in RSD/SD columns
- Remove or correct problematic data points

Poor calibration results
- Ensure you have sufficient blank measurements (≥2)
- Check for outliers in your data
- Verify linearity assumption holds

Performance Issues
- Reduce bootstrap iterations with `--bootstrap 1000` for faster results
- Use `--no-plot` option (if implemented) to skip plot generation

License

MIT License - see [LICENSE](LICENSE) file for details.

Contact

For questions or issues:
- Open an issue on GitHub
- Email: [mdrayhan.186b@qq.com]

---

Last Updated: September 28, 2025
Version: 1.0.0

```
