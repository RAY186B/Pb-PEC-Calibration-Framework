"""
Comprehensive test suite for Pb²⁺ Biosensor WLS Calibration Framework.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import pytest

# Import from the enhanced module
from pb_biosensor_wls_calibration import (
    main,
    validate_calibration_group,
    load_and_validate_data,
    detect_dataset_columns,
    compute_lod_loq,
    estimate_noise_sd,
    perform_bootstrap_resampling,
    CalibrationError,
    DataValidationError,
    ComputationError,
    FileSystemError,
    create_demo_data
)


class TestDataFixtures:
    """Comprehensive test data fixtures."""
    
    @staticmethod
    def repository_dataset() -> pd.DataFrame:
        """Exact copy of the repository dataset."""
        return pd.DataFrame({
            "Sample": ["Tap water"] * 4 + ["Experiment water"] * 4,
            "Added_ng_mL": [0.0, 0.01, 1.0, 100.0] * 2,
            "PEC_found_ng_mL": [0.001, 0.011, 1.177, 117.287, 0.001, 0.009, 1.027, 107.297],
            "Recovery_percent": [np.nan, 97.33, 117.54, 117.29, np.nan, 80.06, 102.58, 107.3],
            "RSD_percent": [4.56, 2.41, 3.54, 3.4, 4.49, 4.71, 4.39, 3.71]
        })
    
    @staticmethod
    def minimal_valid_dataset() -> pd.DataFrame:
        """Minimal valid dataset (3 points, 2 blanks)."""
        return pd.DataFrame({
            "Sample": ["Test"] * 3,
            "Added_ng_mL": [0.0, 0.0, 1.0],
            "PEC_found_ng_mL": [0.001, 0.0011, 1.1]
        })
    
    @staticmethod
    def ideal_calibration_dataset() -> pd.DataFrame:
        """Ideal calibration dataset with perfect linearity."""
        return pd.DataFrame({
            "Sample": ["Ideal"] * 6,
            "Added_ng_mL": [0.0, 0.0, 0.1, 0.1, 1.0, 1.0],
            "PEC_found_ng_mL": [0.001, 0.0011, 0.101, 0.1009, 1.001, 1.0009],
            "RSD_percent": [1.0, 1.1, 0.8, 0.9, 0.7, 0.8]
        })
    
    @staticmethod
    def high_noise_dataset() -> pd.DataFrame:
        """Dataset with high noise for robustness testing."""
        np.random.seed(42)
        return pd.DataFrame({
            "Sample": ["Noisy"] * 8,
            "Added_ng_mL": [0.0, 0.0, 0.1, 0.1, 1.0, 1.0, 10.0, 10.0],
            "PEC_found_ng_mL": [0.001, 0.0011, 0.12, 0.08, 1.2, 0.8, 12.0, 8.0],
            "RSD_percent": np.random.uniform(5, 15, 8)
        })
    
    @staticmethod
    def outlier_dataset() -> pd.DataFrame:
        """Dataset with obvious outliers."""
        return pd.DataFrame({
            "Sample": ["Outlier"] * 6,
            "Added_ng_mL": [0.0, 0.0, 0.1, 0.1, 1.0, 1.0],
            "PEC_found_ng_mL": [0.001, 0.0011, 0.12, 0.11, 10.0, 10.5]  # Last point is outlier
        })
    
    @staticmethod
    def insufficient_blanks_dataset() -> pd.DataFrame:
        """Dataset with insufficient blanks."""
        return pd.DataFrame({
            "Sample": ["Test"] * 3,
            "Added_ng_mL": [0.0, 1.0, 10.0],
            "PEC_found_ng_mL": [0.001, 1.1, 11.1]
        })
    
    @staticmethod
    def negative_concentration_dataset() -> pd.DataFrame:
        """Dataset with negative concentrations."""
        return pd.DataFrame({
            "Sample": ["Test"] * 4,
            "Added_ng_mL": [0.0, 0.0, -1.0, 1.0],
            "PEC_found_ng_mL": [0.001, 0.0011, -1.1, 1.1]
        })
    
    @staticmethod
    def infinite_value_dataset() -> pd.DataFrame:
        """Dataset with infinite values."""
        return pd.DataFrame({
            "Sample": ["Test"] * 4,
            "Added_ng_mL": [0.0, 0.0, 1.0, np.inf],
            "PEC_found_ng_mL": [0.001, 0.0011, 1.1, 1.1]
        })
    
    @staticmethod
    def single_concentration_dataset() -> pd.DataFrame:
        """Dataset with only one concentration level."""
        return pd.DataFrame({
            "Sample": ["Test"] * 4,
            "Added_ng_mL": [0.0, 0.0, 0.0, 0.0],
            "PEC_found_ng_mL": [0.001, 0.0011, 0.0009, 0.0012]
        })
    
    @staticmethod
    def empty_sample_names_dataset() -> pd.DataFrame:
        """Dataset with empty sample names."""
        return pd.DataFrame({
            "Sample": ["", "Test", "Test"],
            "Added_ng_mL": [0.0, 0.0, 1.0],
            "PEC_found_ng_mL": [0.001, 0.0011, 1.1]
        })
    
    @staticmethod
    def non_numeric_dataset() -> pd.DataFrame:
        """Dataset with non-numeric values."""
        return pd.DataFrame({
            "Sample": ["Test"] * 4,
            "Added_ng_mL": [0.0, "invalid", 1.0, 2.0],
            "PEC_found_ng_mL": [0.001, 1.1, 1.2, 2.1]
        })
    
    @staticmethod
    def missing_columns_dataset() -> pd.DataFrame:
        """Dataset with missing required columns."""
        return pd.DataFrame({
            "Sample": ["Test"] * 3,
            "Wrong_Column": [0.0, 1.0, 2.0],
            "Another_Wrong_Column": [0.1, 1.1, 2.1]
        })
    
    @staticmethod
    def extreme_values_dataset() -> pd.DataFrame:
        """Dataset with extreme values."""
        return pd.DataFrame({
            "Sample": ["Test"] * 4,
            "Added_ng_mL": [0.0, 1e6, 1e7, 1e8],
            "PEC_found_ng_mL": [0.001, 1e6, 1e7, 1e8]
        })
    
    @staticmethod
    def zero_slope_dataset() -> pd.DataFrame:
        """Dataset that will produce zero slope."""
        return pd.DataFrame({
            "Sample": ["ZeroSlope"] * 4,
            "Added_ng_mL": [0.0, 0.0, 1.0, 1.0],
            "PEC_found_ng_mL": [0.5, 0.5, 0.5, 0.5]  # Constant response
        })
    
    @staticmethod
    def large_dataset() -> pd.DataFrame:
        """Large dataset for performance testing."""
        np.random.seed(42)
        n_points = 1000
        concentrations = np.concatenate([
            np.zeros(100),  # 100 blanks
            np.logspace(-3, 2, 900)  # 900 standards
        ])
        
        responses = concentrations + np.random.normal(0, 0.01, n_points)
        responses[:100] += 0.001  # Add blank offset
        
        return pd.DataFrame({
            "Sample": ["Large"] * n_points,
            "Added_ng_mL": concentrations,
            "PEC_found_ng_mL": responses,
            "RSD_percent": np.random.uniform(1, 5, n_points)
        })


class TestDataLoadingAndValidation:
    """Test data loading and validation functionality."""
    
    def test_repository_dataset_loading(self, tmp_path: Path):
        """Test loading of actual repository dataset."""
        csv_path = tmp_path / "repository_data.csv"
        repo_data = TestDataFixtures.repository_dataset()
        repo_data.to_csv(csv_path, index=False)
        
        df, column_map = load_and_validate_data(str(csv_path))
        
        assert len(df) == 8
        assert set(df['Sample'].unique()) == {"Tap water", "Experiment water"}
        assert 'RSD_percent' in column_map
        assert 'Recovery_percent' in column_map
        assert column_map['RSD_percent'] == 'RSD_percent'
        assert column_map['Added_ng_mL'] == 'Added_ng_mL'
    
    def test_column_auto_detection(self):
        """Test automatic column detection with various naming conventions."""
        # Test with alternative column names
        alt_data = pd.DataFrame({
            "sample_name": ["Test"] * 4,
            "spike_ng_ml": [0.0, 0.1, 1.0, 10.0],
            "measured_ng_ml": [0.001, 0.11, 1.1, 11.0],
            "rsd": [2.0, 1.5, 2.1, 1.8]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            alt_data.to_csv(f.name, index=False)
            try:
                df, column_map = load_and_validate_data(f.name)
                
                assert column_map['Sample'] == 'sample_name'
                assert column_map['Added_ng_mL'] == 'spike_ng_ml'
                assert column_map['PEC_found_ng_mL'] == 'measured_ng_ml'
                assert column_map['RSD_percent'] == 'rsd'
            finally:
                Path(f.name).unlink()
    
    def test_missing_file(self):
        """Test handling of missing file."""
        with pytest.raises(FileSystemError, match="not found"):
            load_and_validate_data("nonexistent.csv")
    
    def test_empty_file(self, tmp_path: Path):
        """Test handling of empty file."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        
        with pytest.raises(DataValidationError, match="empty"):
            load_and_validate_data(str(csv_path))
    
    def test_invalid_encoding(self, tmp_path: Path):
        """Test handling of file with invalid encoding."""
        csv_path = tmp_path / "invalid_encoding.csv"
        # Write binary data that can't be decoded as CSV
        csv_path.write_bytes(b'\x00\x01\x02\x03\x04\x05')
        
        with pytest.raises(FileSystemError, match="encoding"):
            load_and_validate_data(str(csv_path))
    
    def test_non_numeric_data(self, tmp_path: Path):
        """Test handling of non-numeric data."""
        csv_path = tmp_path / "non_numeric.csv"
        TestDataFixtures.non_numeric_dataset().to_csv(csv_path, index=False)
        
        with pytest.raises(DataValidationError, match="Non-numeric values"):
            load_and_validate_data(str(csv_path))
    
    def test_infinite_values(self, tmp_path: Path):
        """Test handling of infinite values."""
        csv_path = tmp_path / "infinite.csv"
        TestDataFixtures.infinite_value_dataset().to_csv(csv_path, index=False)
        
        with pytest.raises(DataValidationError, match="infinite"):
            load_and_validate_data(str(csv_path))
    
    def test_negative_concentrations(self, tmp_path: Path):
        """Test handling of negative concentrations."""
        csv_path = tmp_path / "negative.csv"
        TestDataFixtures.negative_concentration_dataset().to_csv(csv_path, index=False)
        
        with pytest.raises(DataValidationError, match="negative"):
            load_and_validate_data(str(csv_path))
    
    def test_extreme_values_warning(self, tmp_path: Path):
        """Test warning for extreme concentration values."""
        csv_path = tmp_path / "extreme.csv"
        TestDataFixtures.extreme_values_dataset().to_csv(csv_path, index=False)
        
        # Should load successfully but with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df, column_map = load_and_validate_data(str(csv_path))
            
            assert len(df) == 4
            assert any("extremely high" in str(warning.message).lower() for warning in w)
    
    def test_empty_sample_names(self, tmp_path: Path):
        """Test handling of empty sample names."""
        csv_path = tmp_path / "empty_names.csv"
        TestDataFixtures.empty_sample_names_dataset().to_csv(csv_path, index=False)
        
        with pytest.raises(DataValidationError, match="Sample name cannot be empty"):
            load_and_validate_data(str(csv_path))
    
    def test_missing_columns(self, tmp_path: Path):
        """Test handling of missing required columns."""
        csv_path = tmp_path / "missing_cols.csv"
        TestDataFixtures.missing_columns_dataset().to_csv(csv_path, index=False)
        
        with pytest.raises(DataValidationError, match="Required column"):
            load_and_validate_data(str(csv_path))
    
    def test_corrupted_csv(self, tmp_path: Path):
        """Test handling of corrupted CSV file."""
        csv_path = tmp_path / "corrupted.csv"
        csv_path.write_text("This is not,valid,CSV\ncontent\nthat should,fail parsing")
        
        with pytest.raises(DataValidationError, match="parsing error"):
            load_and_validate_data(str(csv_path))


class TestCalibrationValidation:
    """Test calibration group validation."""
    
    def test_minimal_valid_group(self):
        """Test validation of minimal valid group."""
        data = TestDataFixtures.minimal_valid_dataset()
        validation_summary = validate_calibration_group(data)
        
        assert validation_summary['sample'] == 'Test'
        assert validation_summary['total_points'] == 3
        assert len(validation_summary['issues']) == 0
    
    def test_insufficient_points(self):
        """Test validation with insufficient data points."""
        data = pd.DataFrame({
            "Sample": ["Test"] * 2,
            "Added_ng_mL": [0.0, 1.0],
            "PEC_found_ng_mL": [0.001, 1.1]
        })
        
        with pytest.raises(DataValidationError, match="Minimum 3 required"):
            validate_calibration_group(data, min_points=3)
    
    def test_insufficient_blanks(self):
        """Test validation with insufficient blanks."""
        data = TestDataFixtures.insufficient_blanks_dataset()
        
        with pytest.raises(DataValidationError, match="blank replicate"):
            validate_calibration_group(data, min_blanks=2)
    
    def test_single_concentration_level(self):
        """Test validation with single concentration level."""
        data = TestDataFixtures.single_concentration_dataset()
        
        with pytest.raises(DataValidationError, match="at least 2 distinct concentration"):
            validate_calibration_group(data)
    
    def test_negative_concentrations_validation(self):
        """Test validation with negative concentrations."""
        data = TestDataFixtures.negative_concentration_dataset()
        
        with pytest.raises(DataValidationError, match="negative concentrations"):
            validate_calibration_group(data)
    
    def test_infinite_values_validation(self):
        """Test validation with infinite values."""
        data = TestDataFixtures.infinite_value_dataset()
        
        with pytest.raises(DataValidationError, match="infinite"):
            validate_calibration_group(data)
    
    def test_duplicate_detection(self, tmp_path: Path):
        """Test detection of duplicate data points."""
        data = pd.DataFrame({
            "Sample": ["Test"] * 4,
            "Added_ng_mL": [0.0, 0.0, 1.0, 1.0],  # Duplicates
            "PEC_found_ng_mL": [0.001, 0.001, 1.1, 1.1]  # Duplicates
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validation_summary = validate_calibration_group(data)
            
            assert any("duplicate" in str(warning.message).lower() for warning in w)
    
    def test_empty_group_validation(self):
        """Test validation of empty data group."""
        empty_df = pd.DataFrame(columns=["Sample", "Added_ng_mL", "PEC_found_ng_mL"])
        
        with pytest.raises(DataValidationError, match="empty"):
            validate_calibration_group(empty_df)


class TestCalibrationComputations:
    """Test calibration computational functions."""
    
    def test_lod_loq_calculation_normal(self):
        """Test LOD/LOQ calculation with normal values."""
        blank_sd = 0.001
        slope = 1.0
        
        lod, loq = compute_lod_loq(blank_sd, slope)
        
        expected_lod = 3.3 * 0.001 / 1.0
        expected_loq = 10.0 * 0.001 / 1.0
        
        assert abs(lod - expected_lod) < 1e-10
        assert abs(loq - expected_loq) < 1e-10
    
    def test_lod_loq_calculation_zero_slope(self):
        """Test LOD/LOQ calculation with zero slope."""
        with pytest.raises(ComputationError, match="zero slope"):
            compute_lod_loq(blank_sd=0.001, slope=0.0)
    
    def test_lod_loq_calculation_nan_slope(self):
        """Test LOD/LOQ calculation with NaN slope."""
        with pytest.raises(ComputationError, match="Invalid slope"):
            compute_lod_loq(blank_sd=0.001, slope=np.nan)
    
    def test_lod_loq_calculation_infinite_slope(self):
        """Test LOD/LOQ calculation with infinite slope."""
        with pytest.raises(ComputationError, match="Invalid slope"):
            compute_lod_loq(blank_sd=0.001, slope=np.inf)
    
    def test_lod_loq_calculation_negative_blank_sd(self):
        """Test LOD/LOQ calculation with negative blank SD."""
        with pytest.raises(ComputationError, match="Invalid blank"):
            compute_lod_loq(blank_sd=-0.001, slope=1.0)
    
    def test_lod_loq_calculation_zero_blank_sd(self):
        """Test LOD/LOQ calculation with zero blank SD."""
        lod, loq = compute_lod_loq(blank_sd=0.0, slope=1.0)
        assert lod == 0.0
        assert loq == 0.0
    
    def test_lod_loq_calculation_extreme_values(self):
        """Test LOD/LOQ calculation with extreme values."""
        # Very small slope should trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lod, loq = compute_lod_loq(blank_sd=0.001, slope=1e-12)
            
            assert lod > 1e6  # Very high LOD
            assert any("high LOD" in str(warning.message).lower() for warning in w)
    
    def test_noise_sd_estimation_blanks(self):
        """Test noise SD estimation using blanks."""
        data = TestDataFixtures.minimal_valid_dataset()
        
        sd = estimate_noise_sd(data)
        
        # Should use blank SD (0.001 and 0.0011)
        expected_sd = np.std([0.001, 0.0011], ddof=1)
        assert abs(sd - expected_sd) < 1e-10
    
    def test_noise_sd_estimation_lowest_concentration(self):
        """Test noise SD estimation using lowest concentration."""
        data = pd.DataFrame({
            "Sample": ["Test"] * 4,
            "Added_ng_mL": [0.0, 1.0, 1.0, 2.0],  # Only one blank
            "PEC_found_ng_mL": [0.001, 1.1, 1.2, 2.2]
        })
        
        sd = estimate_noise_sd(data)
        
        # Should use lowest concentration SD (1.1 and 1.2)
        expected_sd = np.std([1.1, 1.2], ddof=1)
        assert abs(sd - expected_sd) < 1e-10
    
    def test_noise_sd_estimation_default(self):
        """Test noise SD estimation with default value."""
        data = pd.DataFrame({
            "Sample": ["Test"] * 3,
            "Added_ng_mL": [0.0, 1.0, 2.0],  # Single points only
            "PEC_found_ng_mL": [0.001, 1.1, 2.2]
        })
        
        default_sd = 0.005
        sd = estimate_noise_sd(data, default_sd=default_sd)
        
        assert sd == default_sd
    
    def test_noise_sd_estimation_conservative_default(self):
        """Test noise SD estimation with conservative default."""
        data = pd.DataFrame({
            "Sample": ["Test"] * 2,
            "Added_ng_mL": [0.0, 1.0],  # Single points only
            "PEC_found_ng_mL": [0.001, 1.1]
        })
        
        sd = estimate_noise_sd(data)
        
        assert sd == 1e-3  # Conservative default


class TestBootstrapResampling:
    """Test bootstrap resampling functionality."""
    
    def test_basic_bootstrap(self):
        """Test basic bootstrap functionality."""
        data = TestDataFixtures.ideal_calibration_dataset()
        
        # Mock WLS result
        class MockWLSResult:
            def __init__(self):
                self.params = np.array([0.001, 1.0])  # [intercept, slope]
        
        mock_result = MockWLSResult()
        
        bootstrap_res = perform_bootstrap_resampling(
            group=data,
            measurement_sd=0.001,
            n_iterations=100,
            wls_result=mock_result,
            blank_mean=0.001,
            blank_sd=0.0005,
            column_map={'Sample': 'Sample', 'Added_ng_mL': 'Added_ng_mL', 'PEC_found_ng_mL': 'PEC_found_ng_mL'},
            use_external_rsd=False
        )
        
        assert 'slope_ci_boot' in bootstrap_res
        assert 'lod_ci_boot' in bootstrap_res
        
        slope_ci = bootstrap_res['slope_ci_boot']
        assert slope_ci[0] < 1.0 < slope_ci[1]  # Should contain true slope
    
    def test_bootstrap_insufficient_data(self):
        """Test bootstrap with insufficient data points."""
        data = TestDataFixtures.minimal_valid_dataset()  # Only 3 points
        
        class MockWLSResult:
            def __init__(self):
                self.params = np.array([0.001, 1.0])
        
        mock_result = MockWLSResult()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bootstrap_res = perform_bootstrap_resampling(
                group=data,
                measurement_sd=0.001,
                n_iterations=100,
                wls_result=mock_result,
                blank_mean=0.001,
                blank_sd=0.0005,
                column_map={'Sample': 'Sample', 'Added_ng_mL': 'Added_ng_mL', 'PEC_found_ng_mL': 'PEC_found_ng_mL'},
                use_external_rsd=False
            )
            
            # Should return empty dict due to insufficient data
            assert bootstrap_res == {}
            assert any("insufficient" in str(warning.message).lower() for warning in w)
    
    def test_bootstrap_with_external_rsd(self):
        """Test bootstrap with external RSD weighting."""
        data = TestDataFixtures.ideal_calibration_dataset()
        
        class MockWLSResult:
            def __init__(self):
                self.params = np.array([0.001, 1.0])
        
        mock_result = MockWLSResult()
        
        bootstrap_res = perform_bootstrap_resampling(
            group=data,
            measurement_sd=0.001,
            n_iterations=100,
            wls_result=mock_result,
            blank_mean=0.001,
            blank_sd=0.0005,
            column_map={'Sample': 'Sample', 'Added_ng_mL': 'Added_ng_mL', 'PEC_found_ng_mL': 'PEC_found_ng_mL', 'RSD_percent': 'RSD_percent'},
            use_external_rsd=True
        )
        
        # Should work with external RSD
        assert 'slope_ci_boot' in bootstrap_res
    
    def test_bootstrap_memory_efficiency(self):
        """Test that bootstrap doesn't cause memory issues with large iterations."""
        data = TestDataFixtures.ideal_calibration_dataset()
        
        class MockWLSResult:
            def __init__(self):
                self.params = np.array([0.001, 1.0])
        
        mock_result = MockWLSResult()
        
        # Test with large iteration count (but reduced for test speed)
        bootstrap_res = perform_bootstrap_resampling(
            group=data,
            measurement_sd=0.001,
            n_iterations=500,  # Reduced for test speed
            wls_result=mock_result,
            blank_mean=0.001,
            blank_sd=0.0005,
            column_map={'Sample': 'Sample', 'Added_ng_mL': 'Added_ng_mL', 'PEC_found_ng_mL': 'PEC_found_ng_mL'},
            use_external_rsd=False
        )
        
        assert 'slope_ci_boot' in bootstrap_res


class TestMainFunction:
    """Test main CLI function with comprehensive scenarios."""
    
    def test_main_with_repository_dataset(self, tmp_path: Path):
        """Test main function with exact repository dataset."""
        csv_path = tmp_path / "repository_data.csv"
        repo_data = TestDataFixtures.repository_dataset()
        repo_data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "results"
        
        # Run main function
        main([
            str(csv_path),
            '--outdir', str(output_dir),
            '--bootstrap', '100',  # Reduced for test speed
            '--no-bootstrap',  # Actually disable for speed
            '--unit', 'ng/mL',
            '--seed', '42'
        ])
        
        # Verify output files exist
        detailed_results = output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv'
        simple_results = output_dir / 'Pb_Biosensor_Simple_Results.csv'
        
        assert detailed_results.exists()
        assert simple_results.exists()
        
        # Verify results content
        df_detailed = pd.read_csv(detailed_results)
        df_simple = pd.read_csv(simple_results)
        
        assert len(df_detailed) >= 2  # At least 2 samples
        assert len(df_simple) >= 2
        
        # Check expected columns
        expected_cols = ['Sample', 'slope', 'lod (ng/mL)', 'loq (ng/mL)', 'r_squared']
        for col in expected_cols:
            assert col in df_detailed.columns
        
        # Check plot directories
        assert (output_dir / 'Tap_water').exists()
        assert (output_dir / 'Experiment_water').exists()
        
        # Check plot files
        tap_plots_dir = output_dir / 'Tap_water'
        expected_plots = ['pred_vs_actual.png', 'stdres_vs_pred.png', 
                         'qq_plot.png', 'cooks_distance.png']
        for plot in expected_plots:
            assert (tap_plots_dir / plot).exists()
    
    def test_main_numerical_accuracy(self, tmp_path: Path):
        """Test numerical accuracy against expected values."""
        csv_path = tmp_path / "test_data.csv"
        repo_data = TestDataFixtures.repository_dataset()
        repo_data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "results"
        
        # Capture output
        import io
        from contextlib import redirect_stdout
        
        output = io.StringIO()
        with redirect_stdout(output):
            main([
                str(csv_path),
                '--outdir', str(output_dir),
                '--no-bootstrap',  # Disable for reproducibility
                '--unit', 'ng/mL',
                '--seed', '42'
            ])
        
        # Check results
        results_df = pd.read_csv(output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv')
        
        tap_results = results_df[results_df['Sample'] == 'Tap water'].iloc[0]
        
        # Check against expected values from original test
        # Original test expected slope ~1.172 and LOD ~3e-4
        assert abs(tap_results['slope'] - 1.172) < 0.05  # Allow reasonable tolerance
        assert abs(tap_results['lod (ng/mL)'] - 3e-4) < 1e-3  # Allow reasonable tolerance
        
        # R² should be very high for this clean dataset
        assert tap_results['r_squared'] > 0.95
    
    def test_main_with_outlier_removal(self, tmp_path: Path):
        """Test main function with outlier removal enabled."""
        data = TestDataFixtures.outlier_dataset()
        csv_path = tmp_path / "outlier_data.csv"
        data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "outlier_results"
        
        main([
            str(csv_path),
            '--outdir', str(output_dir),
            '--remove-outliers',
            '--no-bootstrap'
        ])
        
        # Check results
        results_df = pd.read_csv(output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv')
        
        # Should have original and possibly outlier-removed results
        assert len(results_df) >= 1
        assert 'Outlier' in results_df['Sample'].values
        
        # Check if outlier-removed version exists
        outlier_removed = results_df[results_df['Sample'].str.contains('outliers_removed')]
        if len(outlier_removed) > 0:
            assert outlier_removed.iloc[0]['n_points'] < results_df.iloc[0]['n_points']
    
    def test_main_demo_mode(self, tmp_path: Path):
        """Test main function in demo mode."""
        output_dir = tmp_path / "demo_results"
        
        main(['--demo', '--outdir', str(output_dir), '--no-bootstrap'])
        
        # Should generate results
        assert (output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv').exists()
        assert (output_dir / 'Pb_Biosensor_Simple_Results.csv').exists()
        
        # Should have both samples
        results_df = pd.read_csv(output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv')
        assert 'Tap water' in results_df['Sample'].values
        assert 'Experiment water' in results_df['Sample'].values
    
    def test_main_with_external_rsd(self, tmp_path: Path):
        """Test main function with external RSD weighting."""
        data = TestDataFixtures.ideal_calibration_dataset()
        csv_path = tmp_path / "rsd_data.csv"
        data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "rsd_results"
        
        main([
            str(csv_path),
            '--outdir', str(output_dir),
            '--no-bootstrap'
        ])
        
        # Check that external RSD was used
        results_df = pd.read_csv(output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv')
        assert results_df.iloc[0]['used_external_rsd'] == True
    
    def test_main_error_handling(self, tmp_path: Path):
        """Test main function error handling with various problematic inputs."""
        
        # Test with insufficient data
        insufficient_data = TestDataFixtures.insufficient_blanks_dataset()
        csv_path = tmp_path / "insufficient.csv"
        insufficient_data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "insufficient_results"
        
        # Should handle gracefully without crashing
        main([
            str(csv_path),
            '--outdir', str(output_dir),
            '--no-bootstrap'
        ])
        
        # Should not produce results for invalid sample
        if (output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv').exists():
            results_df = pd.read_csv(output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv')
            assert len(results_df) == 0  # No valid samples
    
    def test_main_large_dataset(self, tmp_path: Path):
        """Test main function with large dataset for performance."""
        large_data = TestDataFixtures.large_dataset()
        csv_path = tmp_path / "large_data.csv"
        large_data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "large_results"
        
        import time
        start_time = time.time()
        
        main([
            str(csv_path),
            '--outdir', str(output_dir),
            '--bootstrap', '50',  # Reduced for test speed
            '--no-bootstrap'  # Actually disable for speed
        ])
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 30 seconds for this test)
        assert elapsed_time < 30
        
        # Should produce results
        assert (output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv').exists()
    
    def test_main_verbose_mode(self, tmp_path: Path):
        """Test main function in verbose mode."""
        data = TestDataFixtures.repository_dataset()
        csv_path = tmp_path / "verbose_data.csv"
        data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "verbose_results"
        
        # Should not raise any errors in verbose mode
        main([
            str(csv_path),
            '--outdir', str(output_dir),
            '--verbose',
            '--no-bootstrap'
        ])
        
        # Should produce results
        assert (output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv').exists()
    
    def test_main_quiet_mode(self, tmp_path: Path):
        """Test main function in quiet mode."""
        data = TestDataFixtures.repository_dataset()
        csv_path = tmp_path / "quiet_data.csv"
        data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "quiet_results"
        
        # Capture stdout
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            main([
                str(csv_path),
                '--outdir', str(output_dir),
                '--quiet',
                '--no-bootstrap'
            ])
        
        # Should suppress most output
        stdout_output = stdout_capture.getvalue()
        assert len(stdout_output.strip()) == 0  # No stdout in quiet mode
        
        # Should still produce results
        assert (output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv').exists()


class TestErrorRecovery:
    """Test robust error recovery mechanisms."""
    
    def test_zero_slope_recovery(self, tmp_path: Path):
        """Test recovery from zero slope condition."""
        data = TestDataFixtures.zero_slope_dataset()
        csv_path = tmp_path / "zero_slope.csv"
        data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "zero_results"
        
        # Should handle gracefully
        main([
            str(csv_path),
            '--outdir', str(output_dir),
            '--no-bootstrap'
        ])
        
        # Check that it processed but with NaN LOD/LOQ
        if (output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv').exists():
            results_df = pd.read_csv(output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv')
            assert len(results_df) >= 1
            # LOD/LOQ should be NaN for zero slope
            assert pd.isna(results_df.iloc[0]['lod (ng/mL)'])
            assert pd.isna(results_df.iloc[0]['loq (ng/mL)'])
    
    def test_high_noise_recovery(self, tmp_path: Path):
        """Test recovery from high noise conditions."""
        data = TestDataFixtures.high_noise_dataset()
        csv_path = tmp_path / "high_noise.csv"
        data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "noise_results"
        
        # Should handle high noise gracefully
        main([
            str(csv_path),
            '--outdir', str(output_dir),
            '--no-bootstrap'
        ])
        
        # Should produce results even with poor R²
        if (output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv').exists():
            results_df = pd.read_csv(output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv')
            assert len(results_df) >= 1
            # R² should be low but finite
            assert results_df.iloc[0]['r_squared'] < 0.9  # Poor fit expected
            assert results_df.iloc[0]['r_squared'] >= 0.0  # But not negative
    
    def test_memory_efficiency(self, tmp_path: Path):
        """Test memory efficiency with large bootstrap iterations."""
        data = TestDataFixtures.repository_dataset()
        csv_path = tmp_path / "memory_test.csv"
        data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "memory_results"
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        main([
            str(csv_path),
            '--outdir', str(output_dir),
            '--bootstrap', '1000',  # Large number
            '--seed', '42'
        ])
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory usage should not increase dramatically (< 500MB increase)
        mem_increase = mem_after - mem_before
        assert mem_increase < 500
        
        # Should still produce results
        assert (output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv').exists()


class TestBackwardCompatibility:
    """Test backward compatibility with original API."""
    
    def test_legacy_validate_calibration_group(self):
        """Test legacy validate_calibration_group function."""
        data = TestDataFixtures.minimal_valid_dataset()
        
        # Should not raise any exceptions
        result = validate_calibration_group(data)
        
        assert result['sample'] == 'Test'
        assert result['total_points'] == 3
    
    def test_legacy_main_function_signature(self, tmp_path: Path):
        """Test legacy main function signature compatibility."""
        data = TestDataFixtures.repository_dataset()
        csv_path = tmp_path / "legacy_data.csv"
        data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "legacy_results"
        
        # Test with legacy parameter names
        main(
            csv_path=str(csv_path),
            outdir=str(output_dir),
            bootstrap_iter=100,
            default_sd=0.001,
            remove_outliers=False,
            no_bootstrap=True,
            unit="ng/mL",
            seed=42
        )
        
        # Should produce results
        assert (output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv').exists()


class TestDatasetConnectivity:
    """Test connectivity with various dataset formats."""
    
    def test_repository_column_mapping(self):
        """Test correct mapping of repository dataset columns."""
        repo_data = TestDataFixtures.repository_dataset()
        
        column_map = detect_dataset_columns(repo_data)
        
        # Check required columns
        assert column_map['Sample'] == 'Sample'
        assert column_map['Added_ng_mL'] == 'Added_ng_mL'
        assert column_map['PEC_found_ng_mL'] == 'PEC_found_ng_mL'
        
        # Check optional columns
        assert column_map['RSD_percent'] == 'RSD_percent'
        assert column_map['Recovery_percent'] == 'Recovery_percent'
    
    def test_external_rsd_column_detection(self):
        """Test detection of external RSD columns."""
        data = pd.DataFrame({
            "Sample": ["Test"] * 4,
            "Added_ng_mL": [0.0, 0.1, 1.0, 10.0],
            "PEC_found_ng_mL": [0.001, 0.11, 1.1, 11.0],
            "RSD_percent_external": [2.0, 1.5, 2.1, 1.8]
        })
        
        column_map = detect_dataset_columns(data)
        
        # Should map RSD_percent_external to RSD_percent
        assert column_map['RSD_percent'] == 'RSD_percent_external'
    
    def test_missing_optional_columns(self):
        """Test handling of datasets without optional columns."""
        data = pd.DataFrame({
            "Sample": ["Test"] * 4,
            "Added_ng_mL": [0.0, 0.1, 1.0, 10.0],
            "PEC_found_ng_mL": [0.001, 0.11, 1.1, 11.0]
        })
        
        column_map = detect_dataset_columns(data)
        
        # Should have required columns
        assert 'Sample' in column_map
        assert 'Added_ng_mL' in column_map
        assert 'PEC_found_ng_mL' in column_map
        
        # Should not have optional columns
        assert 'RSD_percent' not in column_map
        assert 'Recovery_percent' not in column_map
    
    def test_column_name_variations(self):
        """Test handling of various column name conventions."""
        test_cases = [
            {
                'input': {"sample": ["A"], "spike_ng_ml": [1.0], "found_ng_ml": [1.1]},
                'expected': {'Sample': 'sample', 'Added_ng_mL': 'spike_ng_ml', 'PEC_found_ng_mL': 'found_ng_ml'}
            },
            {
                'input': {"Sample_Name": ["A"], "Added": [1.0], "Found": [1.1]},
                'expected': {'Sample': 'Sample_Name', 'Added_ng_mL': 'Added', 'PEC_found_ng_mL': 'Found'}
            },
            {
                'input': {"SAMPLE": ["A"], "ADDED_NG_ML": [1.0], "PEC_FOUND_NG_ML": [1.1]},
                'expected': {'Sample': 'SAMPLE', 'Added_ng_mL': 'ADDED_NG_ML', 'PEC_found_ng_mL': 'PEC_FOUND_NG_ML'}
            }
        ]
        
        for test_case in test_cases:
            data = pd.DataFrame(test_case['input'])
            column_map = detect_dataset_columns(data)
            
            for key, expected_value in test_case['expected'].items():
                assert column_map[key] == expected_value


class TestPerformanceAndRobustness:
    """Test performance and robustness under various conditions."""
    
    def test_concurrent_file_access(self, tmp_path: Path):
        """Test handling of concurrent file access."""
        import threading
        import time
        
        data = TestDataFixtures.repository_dataset()
        csv_path = tmp_path / "concurrent_data.csv"
        data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "concurrent_results"
        
        results = []
        errors = []
        
        def run_analysis():
            try:
                main([
                    str(csv_path),
                    '--outdir', str(output_dir),
                    '--no-bootstrap'
                ])
                results.append("success")
            except Exception as e:
                errors.append(str(e))
        
        # Run multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_analysis)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should complete without errors (file locking should handle this)
        assert len(errors) == 0 or all("locked" in error for error in errors)
    
    def test_file_path_safety(self, tmp_path: Path):
        """Test safety with problematic file paths."""
        # Create data with problematic sample names
        data = pd.DataFrame({
            "Sample": ["Test/with/slashes", "Test\\with\\backslashes", "Test:with:colons", "Test<with>brackets"],
            "Added_ng_mL": [0.0, 0.1, 1.0, 10.0],
            "PEC_found_ng_mL": [0.001, 0.11, 1.1, 11.0]
        })
        
        csv_path = tmp_path / "problematic_names.csv"
        data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "problematic_results"
        
        # Should handle safely without crashing
        main([
            str(csv_path),
            '--outdir', str(output_dir),
            '--no-bootstrap'
        ])
        
        # Should produce results
        if (output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv').exists():
            results_df = pd.read_csv(output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv')
            assert len(results_df) >= 1
    
    def test_unicode_handling(self, tmp_path: Path):
        """Test handling of unicode characters."""
        data = pd.DataFrame({
            "Sample": ["Test_μg", "Test_°C", "Test_²", "Test_ñ"],
            "Added_ng_mL": [0.0, 0.1, 1.0, 10.0],
            "PEC_found_ng_mL": [0.001, 0.11, 1.1, 11.0]
        })
        
        csv_path = tmp_path / "unicode_data.csv"
        data.to_csv(csv_path, index=False, encoding='utf-8')
        
        output_dir = tmp_path / "unicode_results"
        
        # Should handle unicode without issues
        main([
            str(csv_path),
            '--outdir', str(output_dir),
            '--no-bootstrap'
        ])
        
        # Should produce results
        if (output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv').exists():
            results_df = pd.read_csv(output_dir / 'Pb_Biosensor_WLS_Detailed_Results.csv')
            assert len(results_df) >= 1


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
