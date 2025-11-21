"""
Unit tests for Kolmogorov-Smirnov statistic implementation.
"""
import pytest
import numpy as np
from scipy import stats


class TestKolmogorovSmirnovStatistic:
    """Test suite for KS statistic calculation"""
    
    def test_ks_against_scipy(self):
        """
        Test: Custom KS matches SciPy implementation
        Expected: Results agree within 1e-10
        """
        from src.statistics.kolmogorov_smirnov import kolmogorov_smirnov_statistic
        
        np.random.seed(42)
        samples = [
            np.random.standard_normal(50),
            np.random.standard_normal(100),
            np.random.standard_normal(500)
        ]
        
        for sample in samples:
            custom_ks = kolmogorov_smirnov_statistic(sample)
            scipy_ks = stats.kstest(sample, 'norm')[0]
            assert np.abs(custom_ks - scipy_ks) < 1e-10
    
    def test_ks_bounds(self):
        """
        Test: KS statistic stays within valid bounds
        Expected: 0 <= KS <= 1 for all inputs
        """
        from src.statistics.kolmogorov_smirnov import kolmogorov_smirnov_statistic
        
        # Test various distributions
        np.random.seed(42)
        samples = [
            np.random.standard_normal(100),
            np.random.uniform(-3, 3, 100),
            np.random.exponential(1, 100)
        ]
        
        for sample in samples:
            ks = kolmogorov_smirnov_statistic(sample)
            assert 0 <= ks <= 1
    
    def test_ks_perfect_normal(self):
        """
        Test: Perfect normal quantiles give small KS
        Expected: KS < 0.05 for perfect normal data
        """
        from src.statistics.kolmogorov_smirnov import kolmogorov_smirnov_statistic
        
        # Create perfect normal quantiles
        n = 1000
        quantiles = np.linspace(0.001, 0.999, n)
        perfect_normal = stats.norm.ppf(quantiles)
        
        ks = kolmogorov_smirnov_statistic(perfect_normal)
        assert ks < 0.05
    
    def test_ks_uniform_detection(self):
        """
        Test: KS detects non-normal (uniform) data
        Expected: KS > 0.1 for uniform data
        """
        from src.statistics.kolmogorov_smirnov import kolmogorov_smirnov_statistic
        
        np.random.seed(42)
        uniform_sample = np.random.uniform(-2, 2, 100)
        ks = kolmogorov_smirnov_statistic(uniform_sample)
        assert ks > 0.1
    
    def test_ks_input_validation(self):
        """
        Test: Handles invalid inputs appropriately
        Expected: Raises appropriate errors
        """
        from src.statistics.kolmogorov_smirnov import kolmogorov_smirnov_statistic
        
        # Empty array
        with pytest.raises(ValueError):
            kolmogorov_smirnov_statistic(np.array([]))
        
        # Single value
        with pytest.raises(ValueError):
            kolmogorov_smirnov_statistic(np.array([1.0]))
        
        # Non-finite values
        with pytest.raises(ValueError):
            kolmogorov_smirnov_statistic(np.array([1, np.nan, 3]))
    
    def test_ks_calculation_steps(self):
        """
        Test: Verify D+, D- calculation logic
        Expected: max(D+, D-) equals final statistic
        """
        from src.statistics.kolmogorov_smirnov import kolmogorov_smirnov_statistic
        
        # Small sample for manual verification
        sample = np.array([-1.0, 0.0, 1.0])
        ks = kolmogorov_smirnov_statistic(sample)
        
        # Manual calculation
        sorted_sample = np.sort(sample)
        n = len(sample)
        ecdf = np.arange(1, n+1) / n
        theoretical_cdf = stats.norm.cdf(sorted_sample)
        
        d_plus = np.max(ecdf - theoretical_cdf)
        d_minus = np.max(theoretical_cdf - np.insert(ecdf[:-1], 0, 0))
        expected = max(d_plus, d_minus)
        
        assert np.abs(ks - expected) < 1e-10


# Test specifications
"""
KS Statistic Properties:
- Range: [0, 1]
- Measures maximum distance between empirical and theoretical CDF
- D = max(D+, D-) where:
  - D+ = max(ECDF(x) - F(x))
  - D- = max(F(x) - ECDF(x-))
- Should match scipy.stats.kstest for standard normal
"""
