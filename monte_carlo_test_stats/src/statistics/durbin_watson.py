"""
Durbin-Watson test statistic implementation.
"""
import numpy as np


def durbin_watson_statistic(sample: np.ndarray) -> float:
    """
    Calculate Durbin-Watson test statistic for autocorrelation.
    
    The DW statistic tests for first-order autocorrelation in residuals.
    DW ≈ 2 indicates no autocorrelation, < 2 positive, > 2 negative.
    
    Args:
        sample: Array of observations (treated as residuals)
        
    Returns:
        DW statistic in range [0, 4]
        
    Raises:
        ValueError: If sample has fewer than 2 values
    """
    if len(sample) < 2:
        raise ValueError("Sample must have at least 2 values")
    
    sample = np.asarray(sample)
    
    # Calculate numerator: sum of squared first differences
    diff = sample[1:] - sample[:-1]
    numerator = np.sum(diff ** 2)
    
    # Calculate denominator: sum of squared values
    denominator = np.sum(sample ** 2)
    
    # Handle edge case of constant array
    if denominator == 0:
        return 2.0  # No autocorrelation
    
    # DW statistic
    dw = numerator / denominator
    
    return dw
