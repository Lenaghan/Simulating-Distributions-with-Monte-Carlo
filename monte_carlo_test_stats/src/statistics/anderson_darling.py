"""
Anderson-Darling test statistic implementation.
"""
import numpy as np
from scipy import stats


def anderson_darling_statistic(sample: np.ndarray) -> float:
    """
    Calculate Anderson-Darling test statistic for standard normal distribution.
    
    The AD statistic tests goodness-of-fit with emphasis on distribution tails.
    
    Args:
        sample: Array of observations
        
    Returns:
        AD test statistic (positive value)
        
    Raises:
        ValueError: If sample has fewer than 2 values or contains non-finite values
    """
    if len(sample) < 2:
        raise ValueError("Sample must have at least 2 values")
    
    sample = np.asarray(sample)
    
    if not np.all(np.isfinite(sample)):
        raise ValueError("Sample contains non-finite values")
    
    n = len(sample)
    
    # Standardize the data
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)
    if std == 0:
        std = 1  # Avoid division by zero
    
    z = (sample - mean) / std
    z_sorted = np.sort(z)
    
    # Calculate CDF values for sorted data
    cdf_values = stats.norm.cdf(z_sorted)
    
    # Handle numerical stability at boundaries
    # Clip values to avoid log(0) or log(1)
    epsilon = 1e-15
    cdf_values = np.clip(cdf_values, epsilon, 1 - epsilon)
    
    # Calculate the AD statistic
    # AD = -n - (1/n) * Σ(2i-1)[ln(Φ(z_i)) + ln(1-Φ(z_{n+1-i}))]
    i = np.arange(1, n + 1)
    
    # Weighted sum
    weighted_sum = np.sum(
        (2 * i - 1) * (np.log(cdf_values) + np.log(1 - cdf_values[::-1]))
    )
    
    ad_statistic = -n - weighted_sum / n
    
    return ad_statistic
