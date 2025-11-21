"""
Unit tests for convergence monitoring functionality.
"""
import pytest
import numpy as np


class TestConvergenceMonitoring:
    """Test suite for convergence monitoring"""
    
    def test_check_convergence_stable(self):
        """
        Test: Detect convergence when quantiles are stable
        Expected: Returns True when difference < threshold
        """
        from src.simulation.convergence import check_convergence
        
        # Create two batches with overlapping data
        base_data = np.random.randn(150000)
        batch1 = base_data[:100000]  
        batch2 = base_data[50000:150000]  # 50% overlap ensures similar quantiles
        
        # Make them similar by mixing
        combined = np.concatenate([batch1[:50000], batch2[:50000]])
        
        result = check_convergence(
            combined[:100000], 
            combined[50000:150000],
            quantiles=[0.95],
            threshold=0.05
        )
        assert result[0.95] == True
    
    def test_check_convergence_not_stable(self):
        """
        Test: Detect non-convergence when quantiles differ
        Expected: Returns False when difference > threshold
        """
        from src.simulation.convergence import check_convergence
        
        batch1 = np.random.normal(0, 1, 100000)
        batch2 = np.random.normal(0.5, 1, 100000)  # Different mean
        
        result = check_convergence(
            batch1,
            batch2,
            quantiles=[0.95],
            threshold=0.0001
        )
        assert result[0.95] == False
    
    def test_multiple_quantiles(self):
        """
        Test: Check multiple quantiles simultaneously
        Expected: Dictionary with convergence status for each
        """
        from src.simulation.convergence import check_convergence
        
        data = np.random.randn(200000)
        result = check_convergence(
            data[:100000],      # First half
            data[100000:],      # Second half  
            quantiles=[0.75, 0.90, 0.95, 0.99],
            threshold=0.05      # Increase threshold for random data
        )
        
        assert len(result) == 4
        assert all(isinstance(v, bool) for v in result.values())
    
    def test_adaptive_iteration_control(self):
        """
        Test: Adjust iterations based on convergence
        Expected: Increase iterations if not converged
        """
        from src.simulation.convergence import adaptive_iteration_control
        
        # Not converged
        convergence = {0.75: True, 0.90: True, 0.95: False, 0.99: False}
        new_iterations = adaptive_iteration_control(
            current_iterations=1000000,
            convergence_status=convergence,
            max_iterations=10000000
        )
        assert new_iterations > 1000000
        assert new_iterations <= 10000000
    
    def test_convergence_metrics(self):
        """
        Test: Calculate convergence metrics
        Expected: Return quantile differences and stability measure
        """
        from src.simulation.convergence import calculate_convergence_metrics
        
        batch1 = np.random.randn(10000)
        batch2 = np.random.randn(10000)
        
        metrics = calculate_convergence_metrics(batch1, batch2, [0.95])
        assert 'differences' in metrics
        assert 'max_difference' in metrics
        assert metrics['max_difference'] >= 0
