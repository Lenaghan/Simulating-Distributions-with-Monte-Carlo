"""
Kolmogorov-Smirnov test statistic implementation.
"""
import numpy as np
from scipy import stats


def kolmogorov_smirnov_statistic(sample: np.ndarray) -> float:
    """
    Calculate Kolmogorov-Smirnov test statistic for standard normal distribution.
    
    The KS statistic measures the maximum distance between the empirical CDF
    of the sample and the theoretical CDF of the standard normal distribution.
    
    Args:
        sample: Array of observations
        
    Returns:
        KS test statistic D = max(D+, D-)
        
    Raises:
        ValueError: If sample is empty, has single value, or contains non-finite values
    """
    # Input validation
    if len(sample) == 0:
        raise ValueError("Sample cannot be empty")
    if len(sample) == 1:
        raise ValueError("Sample must have more than one value")
    if not np.all(np.isfinite(sample)):
        raise ValueError("Sample contains non-finite values")
    
    # Sort the sample
    sorted_sample = np.sort(sample)
    n = len(sorted_sample)
    
    # Calculate empirical CDF values
    # ECDF(x_i) = i/n for the i-th order statistic
    ecdf = np.arange(1, n + 1) / n
    
    # Calculate theoretical CDF values (standard normal)
    theoretical_cdf = stats.norm.cdf(sorted_sample, loc=0, scale=1)
    
    # Calculate D+ = max(ECDF(x) - F(x))
    d_plus = np.max(ecdf - theoretical_cdf)
    
    # Calculate D- = max(F(x) - ECDF(x-))
    # ECDF(x-) is the ECDF just before x, which is (i-1)/n
    ecdf_minus = np.arange(0, n) / n
    d_minus = np.max(theoretical_cdf - ecdf_minus)
    
    # KS statistic is the maximum of D+ and D-
    ks_statistic = max(d_plus, d_minus)
    
    return ks_statistic


def ks_critical_value(n: int, alpha: float = 0.05) -> float:
    """
    Calculate critical value for KS test using Marsaglia-Tsang-Wang approximation.
    
    Args:
        n: Sample size
        alpha: Significance level
        
    Returns:
        Critical value for KS test
    """
    # Simplified asymptotic approximation
    # Critical value ≈ c(α) / sqrt(n) where c(0.05) ≈ 1.36
    c_values = {0.01: 1.63, 0.05: 1.36, 0.10: 1.22}
    
    if alpha not in c_values:
        # Use 0.05 as default
        c = 1.36
    else:
        c = c_values[alpha]
    
    return c / np.sqrt(n)
