# Pb-PEC Calibration Framework

[<image-card alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.17212139.svg" ></image-card>](https://doi.org/10.5281/zenodo.17212139)
[<image-card alt="Coverage" src="https://img.shields.io/badge/coverage-95%25-brightgreen" ></image-card>](https://github.com/RAY186B/Pb-PEC-Calibration-Framework/actions)

## Overview
This repository contains the Pb-PEC Calibration Framework, supporting a DNAzyme-activated "signal-on" photoelectrochemical biosensor for ultrasensitive Pb²⁺ detection. The framework includes a calibration statistical analysis model based on weighted least squares (WLS) regression, with data and code archived under DOI: [10.5281/zenodo.17212139](https://doi.org/10.5281/zenodo.17212139). Test coverage is 95%, ensuring robustness as reported by GitHub Actions.

## Purpose
This framework calibrates Pb²⁺ biosensor data using WLS regression, IUPAC-compliant LOD/LOQ calculations, and diagnostic plots. The minimal implementation is in `pb_biosensor_wls_calibration.py`, with a fully commented version available in this repository<a href="https://github.com/RAY186B/Pb-PEC-Calibration-Framework" target="_blank" rel="noopener noreferrer nofollow"></a>.

## Installation
```bash
# Set up virtual environment 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For testing with pytest

# Verify Python version (recommended: 3.11)
python --version
