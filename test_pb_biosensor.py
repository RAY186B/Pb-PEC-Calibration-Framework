import pytest
import os
import sys
import pandas as pd
import numpy as np
from pb_biosensor_wls_calibration import main, validate_calibration_group

def test_pb_biosensor_calibration(tmp_path):
    """Test the Pb²⁺ biosensor calibration script on demo data."""
    demo_data = pd.DataFrame({
        "Sample": ["Tap water"]*6 + ["Experiment water"]*6,
        "Added_ng_mL": [0.0, 0.0, 0.01, 0.01, 1.0, 100.0]*2,
        "PEC_found_ng_mL": [0.001, 0.0011, 0.011, 0.0108, 1.177, 117.287, 0.001, 0.0009, 0.009, 0.0092, 1.027, 107.297]
    })
    csv_path = tmp_path / "test_data.csv"
    demo_data.to_csv(csv_path, index=False)

    outdir = tmp_path / "WLS_outputs"
    main(csv_path=str(csv_path), outdir=str(outdir), bootstrap_iter=100, default_sd=0.001,
         remove_outliers=False, no_bootstrap=False, unit="ng/mL", seed=42)

    assert (outdir / "Pb_Biosensor_WLS_Results.csv").exists()
    assert (outdir / "Pb_Biosensor_Simple_Results.csv").exists()
    assert (outdir / "Tap_water" / "Pred_vs_Actual.png").exists()
    assert (outdir / "Experiment_water" / "QQ.png").exists()

    results_df = pd.read_csv(outdir / "Pb_Biosensor_WLS_Results.csv")
    assert len(results_df) >= 2
    tap_results = results_df[results_df['Sample'] == 'Tap water']
    assert abs(tap_results['slope'].iloc[0] - 1.172) < 1e-2
    assert abs(tap_results['lod (ng/mL)'].iloc[0] - 3e-4) < 1e-4

    validate_calibration_group(demo_data[demo_data['Sample'] == 'Tap water'])
    validate_calibration_group(demo_data[demo_data['Sample'] == 'Experiment water'])

def test_invalid_data(tmp_path, capsys):
    """Test validation with insufficient blanks and check stderr output."""
    invalid_data = pd.DataFrame({
        "Sample": ["Test"]*4,
        "Added_ng_mL": [0.0, 0.01, 1.0, 100.0],
        "PEC_found_ng_mL": [0.001, 0.011, 1.177, 117.287]
    })
    csv_path = tmp_path / "invalid_data.csv"
    outdir = tmp_path / "WLS_outputs"
    invalid_data.to_csv(csv_path, index=False)
    with capsys.disabled():
        with pytest.raises(ValueError, match="1 blank replicate"):
            main(csv_path=str(csv_path), outdir=str(outdir), unit="ng/mL", seed=42)
    out, err = capsys.readouterr()
    assert "Validation error" in err

def test_main_insufficient_blanks(capsys, tmp_path):
    """Test CLI path with insufficient blanks and verify stderr message."""
    bad = pd.DataFrame({
        "Sample": ["Test"]*3,
        "Added_ng_mL": [0, 1, 10],
        "PEC_found_ng_mL": [0.001, 1.1, 11.1]
    })
    csv = tmp_path / "bad.csv"
    bad.to_csv(csv, index=False)
    with capsys.disabled():
        with pytest.raises(ValueError, match="1 blank replicate"):
            main(str(csv), outdir=str(tmp_path / "out"), no_bootstrap=True)
    captured = capsys.readouterr()
    assert "1 blank replicate" in captured.err
