"""
Convergence monitoring for Monte Carlo simulations.
"""
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def check_convergence(
    batch1: np.ndarray,
    batch2: np.ndarray,
    quantiles: List[float],
    threshold: float = 0.0001
) -> Dict[float, bool]:
    """
    Check if quantiles have converged between two batches.
    
    Args:
        batch1: First batch of simulation results
        batch2: Second batch of simulation results
        quantiles: List of quantile levels to check
        threshold: Maximum acceptable difference
        
    Returns:
        Dictionary mapping quantile levels to convergence status
    """
    convergence_status = {}
    
    for q in quantiles:
        q1 = np.quantile(batch1, q)
        q2 = np.quantile(batch2, q)
        difference = abs(q1 - q2)
        
        convergence_status[q] = difference < threshold
        
        logger.debug(f"Quantile {q}: diff={difference:.6f}, converged={difference < threshold}")
    
    return convergence_status


def adaptive_iteration_control(
    current_iterations: int,
    convergence_status: Dict[float, bool],
    max_iterations: int = 10_000_000,
    increase_factor: float = 1.5
) -> int:
    """
    Determine new iteration count based on convergence status.
    
    Args:
        current_iterations: Current number of iterations
        convergence_status: Dictionary of quantile convergence status
        max_iterations: Maximum allowed iterations
        increase_factor: Multiplication factor for non-converged case
        
    Returns:
        Recommended number of iterations
    """
    if all(convergence_status.values()):
        # All converged - no increase needed
        return current_iterations
    
    # Calculate proportion converged
    n_converged = sum(convergence_status.values())
    n_total = len(convergence_status)
    proportion_converged = n_converged / n_total
    
    # Adaptive increase based on convergence proportion
    if proportion_converged < 0.5:
        # Less than half converged - larger increase
        new_iterations = int(current_iterations * increase_factor * 1.5)
    else:
        # More than half converged - moderate increase
        new_iterations = int(current_iterations * increase_factor)
    
    # Cap at maximum
    new_iterations = min(new_iterations, max_iterations)
    
    logger.info(f"Convergence: {n_converged}/{n_total}. Iterations: {current_iterations} -> {new_iterations}")
    
    return new_iterations


def calculate_convergence_metrics(
    batch1: np.ndarray,
    batch2: np.ndarray,
    quantiles: List[float]
) -> Dict[str, any]:
    """
    Calculate detailed convergence metrics.
    
    Args:
        batch1: First batch of results
        batch2: Second batch of results
        quantiles: Quantile levels to analyze
        
    Returns:
        Dictionary containing convergence metrics
    """
    differences = {}
    
    for q in quantiles:
        q1 = np.quantile(batch1, q)
        q2 = np.quantile(batch2, q)
        differences[q] = abs(q1 - q2)
    
    metrics = {
        'differences': differences,
        'max_difference': max(differences.values()),
        'mean_difference': np.mean(list(differences.values())),
        'batch1_size': len(batch1),
        'batch2_size': len(batch2)
    }
    
    return metrics


class ConvergenceMonitor:
    """
    Monitor and track convergence over simulation batches.
    """
    
    def __init__(
        self,
        quantiles: List[float],
        threshold: float = 0.0001,
        batch_size: int = 100_000,
        min_batches: int = 10
    ):
        """
        Initialize convergence monitor.
        
        Args:
            quantiles: Quantile levels to monitor
            threshold: Convergence threshold
            batch_size: Size of each batch
            min_batches: Minimum batches before checking convergence
        """
        self.quantiles = quantiles
        self.threshold = threshold
        self.batch_size = batch_size
        self.min_batches = min_batches
        
        self.batch_history = []
        self.quantile_history = {q: [] for q in quantiles}
        self.converged = False
    
    def add_batch(self, batch_results: np.ndarray):
        """Add new batch and check convergence."""
        self.batch_history.append(batch_results)
        
        # Calculate quantiles for this batch
        for q in self.quantiles:
            self.quantile_history[q].append(np.quantile(batch_results, q))
        
        # Check convergence if enough batches
        if len(self.batch_history) >= self.min_batches:
            self.converged = self._check_convergence()
    
    def _check_convergence(self) -> bool:
        """Check if all quantiles have converged."""
        # Compare last two windows
        if len(self.batch_history) < 2:
            return False
        
        last_batch = self.batch_history[-1]
        second_last = self.batch_history[-2]
        
        status = check_convergence(
            last_batch, 
            second_last,
            self.quantiles,
            self.threshold
        )
        
        return all(status.values())
    
    def get_current_estimates(self) -> Dict[float, float]:
        """Get current quantile estimates from all data."""
        if not self.batch_history:
            return {}
        
        all_data = np.concatenate(self.batch_history)
        estimates = {}
        
        for q in self.quantiles:
            estimates[q] = np.quantile(all_data, q)
        
        return estimates
    
    def is_converged(self) -> bool:
        """Check if simulation has converged."""
        return self.converged
