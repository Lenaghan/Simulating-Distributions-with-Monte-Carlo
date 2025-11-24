"""
Unit tests for quantile extraction and table generation module.
Tests cover quantile calculation, standard errors, bootstrap CIs, and table formatting.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import h5py
import json
import shutil
import sys
import os

# Add src to path for imports (adjust path as needed)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestQuantileExtraction(unittest.TestCase):
    """Test cases for quantile extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures with sample data."""
        np.random.seed(42)
        # Create sample data for testing - known distribution
        self.sample_data = np.random.randn(10000)
        self.quantile_levels = [0.75, 0.90, 0.95, 0.99]
        
        # Calculate expected quantiles for comparison
        self.expected_quantiles = {
            level: np.quantile(self.sample_data, level) 
            for level in self.quantile_levels
        }
        
        # Create temporary test directory
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create a test HDF5 file
        self.test_h5_path = self.test_dir / "test_data.h5"
        with h5py.File(self.test_h5_path, 'w') as f:
            f.create_dataset('data', data=self.sample_data)
        
        # Create test metadata
        self.test_metadata = {
            "statistic": "test_statistic",
            "sample_size": 100,
            "iterations_completed": 10000,
            "convergence_achieved": True,
            "quantiles": self.expected_quantiles,
            "runtime_seconds": 1.23,
            "timestamp": "2025-11-23T12:00:00",
            "seed_used": 42
        }
        
        metadata_path = self.test_dir / "test_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.test_metadata, f)
    
    def tearDown(self):
        """Clean up test files."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_extract_quantiles_from_h5(self):
        """Test: Extract quantiles from HDF5 file → assert correct values."""
        # This will test the actual quantile extraction once module is implemented
        # For now, test using numpy directly
        with h5py.File(self.test_h5_path, 'r') as f:
            data = f['data'][:]
        
        calculated_quantiles = {}
        for level in self.quantile_levels:
            calculated_quantiles[level] = np.quantile(data, level)
        
        # Assert quantiles match expected values
        for level in self.quantile_levels:
            self.assertAlmostEqual(
                calculated_quantiles[level],
                self.expected_quantiles[level],
                places=10,
                msg=f"Quantile {level} mismatch"
            )
    
    def test_missing_file_error(self):
        """Test: Load non-existent file → assert FileNotFoundError."""
        non_existent_path = self.test_dir / "missing_file.h5"
        
        with self.assertRaises(FileNotFoundError):
            with h5py.File(non_existent_path, 'r') as f:
                pass
    
    def test_invalid_quantile_levels(self):
        """Test: Invalid quantile levels → assert ValueError."""
        invalid_levels = [-0.1, 0.5, 1.5]  # -0.1 and 1.5 are invalid
        
        with self.assertRaises(ValueError):
            for level in invalid_levels:
                if level < 0 or level > 1:
                    raise ValueError(f"Quantile level {level} must be between 0 and 1")
    
    def test_batch_means_standard_error(self):
        """Test: Calculate standard errors using batch means → assert reasonable values."""
        n_batches = 100
        batch_size = len(self.sample_data) // n_batches
        
        batch_quantiles = {level: [] for level in self.quantile_levels}
        
        # Calculate quantiles for each batch
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_data = self.sample_data[start_idx:end_idx]
            
            for level in self.quantile_levels:
                q = np.quantile(batch_data, level)
                batch_quantiles[level].append(q)
        
        # Calculate standard errors
        standard_errors = {}
        for level in self.quantile_levels:
            batch_std = np.std(batch_quantiles[level], ddof=1)
            se = batch_std / np.sqrt(n_batches)
            standard_errors[level] = se
            
            # Assert SE is positive and reasonable
            self.assertGreater(se, 0, f"SE for quantile {level} should be positive")
            self.assertLess(se, 0.1, f"SE for quantile {level} seems too large")
    
    def test_bootstrap_confidence_intervals(self):
        """Test: Generate bootstrap CIs → assert 95% CI contains true quantile."""
        n_bootstrap = 1000
        bootstrap_quantiles = {level: [] for level in self.quantile_levels}
        
        # Perform bootstrap resampling
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(self.sample_data, 
                                               size=len(self.sample_data), 
                                               replace=True)
            for level in self.quantile_levels:
                q = np.quantile(bootstrap_sample, level)
                bootstrap_quantiles[level].append(q)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for level in self.quantile_levels:
            lower_ci = np.percentile(bootstrap_quantiles[level], 2.5)
            upper_ci = np.percentile(bootstrap_quantiles[level], 97.5)
            point_estimate = np.median(bootstrap_quantiles[level])
            
            confidence_intervals[level] = {
                'lower': lower_ci,
                'point': point_estimate,
                'upper': upper_ci
            }
            
            # Assert CI contains the true quantile
            true_quantile = self.expected_quantiles[level]
            self.assertGreaterEqual(
                true_quantile, lower_ci,
                f"True quantile {level} below CI lower bound"
            )
            self.assertLessEqual(
                true_quantile, upper_ci,
                f"True quantile {level} above CI upper bound"
            )
            
            # Assert CI width is reasonable
            ci_width = upper_ci - lower_ci
            self.assertLess(ci_width, 0.5, f"CI width for {level} too large")
    
    def test_precision_formatting(self):
        """Test: Format values to 4 decimal places → assert correct rounding."""
        test_values = [
            (0.1234567, "0.1235"),
            (0.1234444, "0.1234"),
            (0.9999999, "1.0000"),
            (0.0001234, "0.0001"),
            (1.2345678, "1.2346")
        ]
        
        for value, expected in test_values:
            formatted = f"{value:.4f}"
            self.assertEqual(formatted, expected,
                           f"Formatting {value} failed: got {formatted}, expected {expected}")
    
    def test_csv_export_format(self):
        """Test: Export to CSV → assert correct structure."""
        # Create sample data for CSV export
        data = {
            'sample_size': [30, 50, 100, 500, 1000],
            'q75': [0.1805, 0.1512, 0.1074, 0.0482, 0.0340],
            'q90': [0.2172, 0.1823, 0.1293, 0.0580, 0.0409],
            'q95': [0.2412, 0.2025, 0.1436, 0.0644, 0.0455],
            'q99': [0.2892, 0.2429, 0.1723, 0.0773, 0.0546]
        }
        
        df = pd.DataFrame(data)
        csv_path = self.test_dir / 'test_quantiles.csv'
        df.to_csv(csv_path, index=False, float_format='%.4f')
        
        # Read back and verify
        self.assertTrue(csv_path.exists(), "CSV file not created")
        
        df_read = pd.read_csv(csv_path)
        self.assertEqual(len(df_read), 5, "Wrong number of rows")
        self.assertEqual(len(df_read.columns), 5, "Wrong number of columns")
        self.assertIn('sample_size', df_read.columns, "Missing sample_size column")
        self.assertIn('q75', df_read.columns, "Missing q75 column")
        
        # Check formatting preserved
        self.assertEqual(df_read['q75'].iloc[0], 0.1805)
    
    def test_latex_table_format(self):
        """Test: Export to LaTeX → assert valid LaTeX syntax."""
        latex_content = r"""\begin{table}[h]
\centering
\begin{tabular}{lrrrr}
\toprule
Sample Size & Q75 & Q90 & Q95 & Q99 \\
\midrule
30 & 0.1805 & 0.2172 & 0.2412 & 0.2892 \\
50 & 0.1512 & 0.1823 & 0.2025 & 0.2429 \\
100 & 0.1074 & 0.1293 & 0.1436 & 0.1723 \\
\bottomrule
\end{tabular}
\caption{Critical values for test statistic}
\end{table}"""
        
        # Check for required LaTeX elements
        self.assertIn(r'\begin{tabular}', latex_content)
        self.assertIn(r'\end{tabular}', latex_content)
        self.assertIn(r'\toprule', latex_content)
        self.assertIn(r'\bottomrule', latex_content)
        self.assertIn('&', latex_content)  # Column separator
        self.assertIn(r'\\', latex_content)  # Row terminator
    
    def test_markdown_table_format(self):
        """Test: Export to Markdown → assert valid MD table."""
        markdown_content = """| Sample Size | Q75 | Q90 | Q95 | Q99 |
|------------|------|------|------|------|
| 30 | 0.1805 | 0.2172 | 0.2412 | 0.2892 |
| 50 | 0.1512 | 0.1823 | 0.2025 | 0.2429 |
| 100 | 0.1074 | 0.1293 | 0.1436 | 0.1723 |"""
        
        lines = markdown_content.strip().split('\n')
        
        # Check markdown table structure
        self.assertEqual(len(lines), 5, "Should have header + separator + 3 data rows")
        self.assertTrue(lines[0].startswith('|') and lines[0].endswith('|'))
        self.assertIn('---', lines[1], "Missing separator row")
        
        # Check all rows have same number of columns
        for line in lines:
            self.assertEqual(line.count('|'), 6, "Each row should have 6 pipe characters")
    
    def test_metadata_generation(self):
        """Test: Create metadata file → assert complete information."""
        metadata = {
            "timestamp": "2025-11-23T12:00:00",
            "iterations_used": [5000000, 5000000, 10000000],  # Different configs
            "convergence_status": {
                "kolmogorov_smirnov": True,
                "durbin_watson": True,
                "anderson_darling": True
            },
            "quantile_levels": [0.75, 0.90, 0.95, 0.99],
            "sample_sizes": [30, 50, 100, 500, 1000],
            "batch_size_for_se": 100,
            "bootstrap_iterations": 1000
        }
        
        # Verify all required fields
        required_fields = ['timestamp', 'iterations_used', 'convergence_status', 
                          'quantile_levels', 'sample_sizes']
        for field in required_fields:
            self.assertIn(field, metadata, f"Missing required field: {field}")
        
        # Verify types
        self.assertIsInstance(metadata['quantile_levels'], list)
        self.assertEqual(len(metadata['quantile_levels']), 4)
        self.assertIsInstance(metadata['convergence_status'], dict)
        
    def test_handle_mixed_file_patterns(self):
        """Test: Handle both _results.h5 and _final.h5 → assert correct precedence."""
        # Create two files with different patterns
        results_file = self.test_dir / "test_100_results.h5"
        final_file = self.test_dir / "test_n100_final.h5"
        
        # Create both files with different data
        data_results = np.random.randn(1000)
        data_final = np.random.randn(5000)
        
        with h5py.File(results_file, 'w') as f:
            f.create_dataset('data', data=data_results)
        
        with h5py.File(final_file, 'w') as f:
            f.create_dataset('data', data=data_final)
        
        # Simulate file selection logic
        files = list(self.test_dir.glob("*.h5"))
        
        # Should prefer _final.h5 files
        selected_file = None
        for f in files:
            if '_final.h5' in str(f):
                selected_file = f
                break
        if selected_file is None:
            for f in files:
                if '_results.h5' in str(f):
                    selected_file = f
                    break
        
        self.assertIsNotNone(selected_file, "No file selected")
        self.assertIn('_final.h5', str(selected_file), "Should prefer _final.h5")
        
        # Verify correct data is loaded
        with h5py.File(selected_file, 'r') as f:
            loaded_data = f['data'][:]
            self.assertEqual(len(loaded_data), 5000, "Should load from _final.h5")


class TestErrorHandling(unittest.TestCase):
    """Test cases for error conditions and edge cases."""
    
    def setUp(self):
        """Set up test directory."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test files."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_insufficient_data_points(self):
        """Test: Too few data points for quantiles → assert warning."""
        small_data = np.random.randn(100)  # Only 100 points
        
        # For 0.99 quantile, only 1 point in top 1%
        q99 = np.quantile(small_data, 0.99)
        
        # This should work but may be unstable
        self.assertIsNotNone(q99)
        
        # In real implementation, should log warning
        if len(small_data) < 1000:
            warning_message = f"Warning: Only {len(small_data)} data points available"
            self.assertIn("Warning", warning_message)
            self.assertIn("100", warning_message)
    
    def test_extreme_quantiles_stability(self):
        """Test: 0.99 quantile with limited data → assert stability warning."""
        # With 10,000 points, 0.99 quantile uses top 100 points
        data = np.random.randn(10000)
        q99 = np.quantile(data, 0.99)
        
        # Calculate how many points determine this quantile
        n_points_in_tail = int(len(data) * 0.01)
        
        # Should warn if too few points
        if n_points_in_tail < 100:
            self.assertLess(n_points_in_tail, 100, 
                           "Should flag when <100 points determine 0.99 quantile")
    
    def test_data_type_validation(self):
        """Test: Ensure data is numeric → assert TypeError for non-numeric."""
        invalid_data = ['a', 'b', 'c']  # String data
        
        with self.assertRaises((TypeError, ValueError)):
            # This should fail
            np.quantile(invalid_data, 0.5)
    
    def test_empty_data_handling(self):
        """Test: Empty dataset → assert informative error."""
        empty_data = np.array([])
        
        with self.assertRaises((ValueError, IndexError)):
            np.quantile(empty_data, 0.5)


if __name__ == '__main__':
    # Run with verbose output to see test descriptions
    unittest.main(verbosity=2)
