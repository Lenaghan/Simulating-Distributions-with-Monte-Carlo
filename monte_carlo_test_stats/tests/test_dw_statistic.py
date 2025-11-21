"""
Unit tests for Durbin-Watson statistic implementation.
"""
import pytest
import numpy as np
from statsmodels.stats.stattools import durbin_watson


class TestDurbinWatsonStatistic:
    """Test suite for DW statistic calculation"""
    
    def test_dw_against_statsmodels(self):
        """
        Test: Custom DW matches statsmodels implementation
        Expected: Results agree within 1e-10
        """
        from src.statistics.durbin_watson import durbin_watson_statistic
        
        np.random.seed(42)
        samples = [
            np.random.standard_normal(50),
            np.random.standard_normal(100),
            np.random.standard_normal(500)
        ]
        
        for sample in samples:
            custom_dw = durbin_watson_statistic(sample)
            statsmodels_dw = durbin_watson(sample)
            assert np.abs(custom_dw - statsmodels_dw) < 1e-10
    
    def test_dw_bounds(self):
        """
        Test: DW statistic stays within valid bounds
        Expected: 0 <= DW <= 4 for all inputs
        """
        from src.statistics.durbin_watson import durbin_watson_statistic
        
        np.random.seed(42)
        samples = [
            np.random.standard_normal(100),
            np.ones(100),  # Constant
            np.arange(100),  # Linear trend
        ]
        
        for sample in samples:
            dw = durbin_watson_statistic(sample)
            assert 0 <= dw <= 4
    
    def test_dw_no_autocorrelation(self):
        """
        Test: Random normal data gives DW near 2
        Expected: 1.5 < DW < 2.5 for uncorrelated data
        """
        from src.statistics.durbin_watson import durbin_watson_statistic
        
        np.random.seed(42)
        sample = np.random.standard_normal(1000)
        dw = durbin_watson_statistic(sample)
        assert 1.5 < dw < 2.5
    
    def test_dw_positive_autocorrelation(self):
        """
        Test: Positively correlated data gives DW < 2
        Expected: DW < 1.5 for positive autocorrelation
        """
        from src.statistics.durbin_watson import durbin_watson_statistic
        
        # Create positively autocorrelated series
        n = 100
        sample = np.zeros(n)
        sample[0] = np.random.randn()
        for i in range(1, n):
            sample[i] = 0.8 * sample[i-1] + 0.2 * np.random.randn()
        
        dw = durbin_watson_statistic(sample)
        assert dw < 1.5
    
    def test_dw_negative_autocorrelation(self):
        """
        Test: Negatively correlated data gives DW > 2
        Expected: DW > 2.5 for negative autocorrelation
        """
        from src.statistics.durbin_watson import durbin_watson_statistic
        
        # Create alternating series (negative autocorrelation)
        n = 100
        sample = np.zeros(n)
        for i in range(n):
            sample[i] = (-1)**i * np.random.uniform(0.5, 1.5)
        
        dw = durbin_watson_statistic(sample)
        assert dw > 2.5
    
    def test_dw_edge_cases(self):
        """
        Test: Handle edge cases appropriately
        Expected: Appropriate handling of constant arrays
        """
        from src.statistics.durbin_watson import durbin_watson_statistic
        
        # Constant array (zero denominator)
        constant = np.ones(10)
        dw = durbin_watson_statistic(constant)
        assert dw == 0.0  # Should return 0.0 for no autocorrelation
        
        # Small sample
        with pytest.raises(ValueError):
            durbin_watson_statistic(np.array([1]))


# Test specifications
"""
DW Statistic Properties:
- Range: [0, 4]
- DW ≈ 2: No autocorrelation
- DW < 2: Positive autocorrelation
- DW > 2: Negative autocorrelation
- Formula: DW = Σ(e_t - e_{t-1})² / Σe_t²
"""
