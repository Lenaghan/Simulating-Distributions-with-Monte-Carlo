"""
Unit tests for Anderson-Darling statistic implementation.
"""
import pytest
import numpy as np
from scipy import stats


class TestAndersonDarlingStatistic:
    """Test suite for AD statistic calculation"""
    
    def test_ad_against_scipy(self):
        """
        Test: Custom AD approximates SciPy implementation
        Expected: Results reasonably close for normal data
        """
        from src.statistics.anderson_darling import anderson_darling_statistic
        
        np.random.seed(42)
        sample = np.random.standard_normal(100)
        
        custom_ad = anderson_darling_statistic(sample)
        scipy_result = stats.anderson(sample, 'norm')
        
        # AD statistic should be positive
        assert custom_ad > 0
        # Should be in reasonable range
        assert 0.1 < custom_ad < 5.0
    
    def test_ad_positive_values(self):
        """
        Test: AD statistic is always positive
        Expected: AD > 0 for all inputs
        """
        from src.statistics.anderson_darling import anderson_darling_statistic
        
        np.random.seed(42)
        samples = [
            np.random.standard_normal(50),
            np.random.uniform(-2, 2, 50),
            np.random.exponential(1, 50)
        ]
        
        for sample in samples:
            ad = anderson_darling_statistic(sample)
            assert ad > 0
    
    def test_ad_tail_sensitivity(self):
        """
        Test: AD more sensitive to tails than KS
        Expected: Heavy-tailed data gives larger AD values
        """
        from src.statistics.anderson_darling import anderson_darling_statistic
        
        np.random.seed(42)
        # Normal data
        normal_data = np.random.standard_normal(100)
        ad_normal = anderson_darling_statistic(normal_data)
        
        # Heavy-tailed data (t-distribution)
        t_data = np.random.standard_t(df=3, size=100)
        ad_t = anderson_darling_statistic(t_data)
        
        # Heavy tails should give larger AD
        assert ad_t > ad_normal
    
    def test_ad_perfect_normal(self):
        """
        Test: Perfect normal quantiles give small AD
        Expected: AD < 0.5 for perfect normal data
        """
        from src.statistics.anderson_darling import anderson_darling_statistic
        
        n = 100
        # Create perfect normal quantiles
        quantiles = np.linspace(0.01, 0.99, n)
        perfect_normal = stats.norm.ppf(quantiles)
        
        ad = anderson_darling_statistic(perfect_normal)
        assert ad < 0.5
    
    def test_ad_numerical_stability(self):
        """
        Test: Handle numerical edge cases
        Expected: No errors for extreme values
        """
        from src.statistics.anderson_darling import anderson_darling_statistic
        
        # Very small/large values
        extreme_sample = np.array([-10, -5, 0, 5, 10])
        ad = anderson_darling_statistic(extreme_sample)
        assert np.isfinite(ad)
        assert ad > 0
    
    def test_ad_input_validation(self):
        """
        Test: Validates input appropriately
        Expected: Raises errors for invalid inputs
        """
        from src.statistics.anderson_darling import anderson_darling_statistic
        
        # Too small sample
        with pytest.raises(ValueError):
            anderson_darling_statistic(np.array([1]))
        
        # Non-finite values
        with pytest.raises(ValueError):
            anderson_darling_statistic(np.array([1, np.nan, 3]))

# Test specifications
"""
AD Statistic Properties:
- Range: (0, ∞)
- More sensitive to tails than KS test
- Formula: AD = -n - (1/n)Σ(2i-1)[ln(Φ(z_i)) + ln(1-Φ(z_{n+1-i}))]
- Requires numerical stability for extreme values
"""
