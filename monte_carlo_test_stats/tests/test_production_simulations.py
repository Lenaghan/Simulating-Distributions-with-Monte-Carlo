"""
Unit tests for production simulation orchestration.
Tests parameter grid generation, convergence-adaptive execution,
checkpoint recovery, and results storage.
"""
import pytest
import numpy as np
import os
from pathlib import Path
import tempfile
import json
import h5py
from unittest.mock import Mock, patch, MagicMock


class TestProductionSimulations:
    """Test suite for production simulation orchestration"""
    
    def test_parameter_grid_generation(self):
        """
        Test: Generate correct parameter grid for production runs
        Expected: 15 configurations (3 statistics × 5 sample sizes)
        """
        from run_full_simulations import generate_parameter_grid
        
        grid = generate_parameter_grid()
        
        # Verify total configurations
        assert len(grid) == 15
        
        # Check statistics coverage
        statistics = {config['statistic'] for config in grid}
        assert statistics == {'kolmogorov_smirnov', 'durbin_watson', 'anderson_darling'}
        
        # Check sample sizes coverage
        sample_sizes = {config['sample_size'] for config in grid}
        assert sample_sizes == {30, 50, 100, 500, 1000}
        
        # Verify each config has required fields
        for config in grid:
            assert 'statistic' in config
            assert 'sample_size' in config
            assert 'iterations' in config
            assert config['iterations'] == 10_000_000  # Default 10M
    
    def test_convergence_adaptive_execution(self):
        """
        Test: Dynamic iteration adjustment based on convergence
        Expected: Iterations increase when not converged, cap at maximum
        """
        from run_full_simulations import AdaptiveSimulator
        
        simulator = AdaptiveSimulator(
            initial_iterations=1_000_000,
            convergence_threshold=0.0001,
            max_iterations=20_000_000
        )
        
        # Mock non-converged scenario
        mock_convergence = {0.75: True, 0.90: True, 0.95: False, 0.99: False}
        
        # Test iteration increase
        new_iterations = simulator.determine_next_iterations(
            current_iterations=1_000_000,
            convergence_status=mock_convergence
        )
        
        # Should increase (1.5x factor)
        assert new_iterations == 1_500_000
        
        # Test maximum cap
        new_iterations = simulator.determine_next_iterations(
            current_iterations=15_000_000,
            convergence_status=mock_convergence
        )
        assert new_iterations <= 20_000_000
        
        # Test fully converged - no increase
        converged = {0.75: True, 0.90: True, 0.95: True, 0.99: True}
        new_iterations = simulator.determine_next_iterations(
            current_iterations=5_000_000,
            convergence_status=converged
        )
        assert new_iterations == 5_000_000
    
    def test_checkpoint_recovery_mechanism(self):
        """
        Test: Resume from checkpoint after interruption
        Expected: Load previous state and continue from last iteration
        """
        from run_full_simulations import SimulationRunner
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = SimulationRunner(checkpoint_dir=temp_dir)
            
            # Simulate interrupted run - save checkpoint
            partial_results = np.random.randn(5_000_000)
            checkpoint_metadata = {
                'statistic': 'kolmogorov_smirnov',
                'sample_size': 100,
                'iterations_completed': 5_000_000,
                'convergence_achieved': False,
                'last_quantiles': {0.95: 1.358}
            }
            
            checkpoint_path = runner.save_checkpoint(
                partial_results, 
                checkpoint_metadata
            )
            
            # Test recovery
            recovered_data, recovered_meta = runner.recover_from_checkpoint(
                statistic='kolmogorov_smirnov',
                sample_size=100
            )
            
            assert len(recovered_data) == 5_000_000
            assert recovered_meta['iterations_completed'] == 5_000_000
            assert recovered_meta['statistic'] == 'kolmogorov_smirnov'
            
            # Test continuation
            continued_results = runner.continue_from_checkpoint(
                checkpoint_path,
                target_iterations=10_000_000
            )
            
            assert len(continued_results) == 10_000_000
    
    def test_results_storage_format(self):
        """
        Test: Verify HDF5 storage format and metadata
        Expected: Proper structure with compression and metadata
        """
        from run_full_simulations import save_production_results
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock simulation results
            results = np.random.randn(10_000_000)
            metadata = {
                'statistic': 'durbin_watson',
                'sample_size': 50,
                'iterations': 10_000_000,
                'converged': True,
                'quantiles': {
                    0.75: 2.345,
                    0.90: 2.567,
                    0.95: 2.678,
                    0.99: 2.890
                },
                'runtime_seconds': 354.2,
                'seed': 42
            }
            
            filepath = Path(temp_dir) / 'dw_50_results.h5'
            save_production_results(filepath, results, metadata)
            
            # Verify file structure
            with h5py.File(filepath, 'r') as f:
                # Check data
                assert 'results' in f
                assert f['results'].shape == (10_000_000,)
                assert f['results'].compression is not None
                
                # Check metadata
                assert f.attrs['statistic'] == 'durbin_watson'
                assert f.attrs['sample_size'] == 50
                assert f.attrs['converged'] == True
                
                # Quantiles stored as JSON
                quantiles = json.loads(f.attrs['quantiles'])
                assert quantiles['0.95'] == 2.678
    
    def test_batch_processing_order(self):
        """
        Test: Verify prioritization of smaller sample sizes
        Expected: n=30, 50 processed before n=500, 1000
        """
        from run_full_simulations import prioritize_configurations
        
        configs = [
            {'statistic': 'ks', 'sample_size': 1000},
            {'statistic': 'ks', 'sample_size': 30},
            {'statistic': 'ks', 'sample_size': 500},
            {'statistic': 'ks', 'sample_size': 50},
            {'statistic': 'ks', 'sample_size': 100}
        ]
        
        prioritized = prioritize_configurations(configs)
        
        # Check ordering by sample size
        sizes = [c['sample_size'] for c in prioritized]
        assert sizes == [30, 50, 100, 500, 1000]
    
    def test_convergence_monitoring_integration(self):
        """
        Test: Integration with convergence monitoring system
        Expected: Correct convergence checks at intervals
        """
        from run_full_simulations import ProductionSimulator
        
        with patch('src.simulation.convergence.check_convergence') as mock_check:
            mock_check.return_value = {0.95: True, 0.99: False}
            
            simulator = ProductionSimulator(
                check_interval=100_000,
                convergence_threshold=0.0001
            )
            
            # Run with convergence monitoring
            simulator.run_with_monitoring(
                statistic='anderson_darling',
                sample_size=100,
                max_iterations=500_000
            )
            
            # Verify convergence checks at intervals
            # 500k iterations / 100k interval = 5 checks
            assert mock_check.call_count >= 4
    
    def test_parallel_execution_configuration(self):
        """
        Test: Verify parallel processing configuration
        Expected: Uses all CPU cores, proper batch sizing
        """
        from run_full_simulations import get_parallel_config
        
        config = get_parallel_config()
        
        assert config['n_jobs'] == -1  # All cores
        assert config['backend'] == 'loky'
        assert 'batch_size' in config
        assert config['batch_size'] > 0
    
    def test_metadata_generation(self):
        """
        Test: Complete metadata generation for each simulation
        Expected: All required fields present and correct
        """
        from run_full_simulations import generate_simulation_metadata
        
        metadata = generate_simulation_metadata(
            statistic='kolmogorov_smirnov',
            sample_size=100,
            iterations=10_000_000,
            converged=True,
            runtime=423.5,
            quantiles={0.95: 0.1358}
        )
        
        # Required fields
        assert metadata['statistic'] == 'kolmogorov_smirnov'
        assert metadata['sample_size'] == 100
        assert metadata['iterations_completed'] == 10_000_000
        assert metadata['convergence_achieved'] == True
        assert metadata['runtime_seconds'] == 423.5
        assert 'timestamp' in metadata
        assert 'seed_used' in metadata
        assert 'quantiles' in metadata
    
    def test_error_recovery_graceful(self):
        """
        Test: Graceful handling of simulation errors
        Expected: Log error, save partial results, continue with next config
        """
        from run_full_simulations import SimulationOrchestrator
        
        orchestrator = SimulationOrchestrator()
        
        # Mock a failing configuration
        with patch('src.simulation.engine.MonteCarloEngine.simulate') as mock_sim:
            mock_sim.side_effect = [
                np.random.randn(1000),  # First config succeeds
                ValueError("Simulation error"),  # Second fails
                np.random.randn(1000)  # Third succeeds
            ]
            
            configs = [
                {'statistic': 'ks', 'sample_size': 30, 'iterations': 1000},
                {'statistic': 'dw', 'sample_size': 50, 'iterations': 1000},
                {'statistic': 'ad', 'sample_size': 100, 'iterations': 1000}
            ]
            
            results = orchestrator.run_all_configurations(configs)
            
            # Should have results for 2/3 configs
            assert len(results) == 2
            assert any(r['statistic'] == 'ks' for r in results)
            assert any(r['statistic'] == 'ad' for r in results)
            
            # Error logged but not raised
            assert orchestrator.errors_logged == 1


# Test specification documentation
"""
Production Simulation Test Coverage:
1. Parameter Grid: Ensures all 15 configurations generated correctly
2. Adaptive Execution: Verifies iteration adjustment logic
3. Checkpoint Recovery: Tests interruption recovery mechanism
4. Storage Format: Validates HDF5 structure and compression
5. Batch Processing: Confirms prioritization strategy
6. Convergence Integration: Tests monitoring at intervals
7. Parallel Configuration: Verifies resource utilization
8. Metadata Generation: Ensures complete tracking information
9. Error Recovery: Tests graceful failure handling

Critical Path Tests:
- Full pipeline execution (grid -> simulate -> store)
- Convergence-based termination
- Checkpoint-based continuation
- Parallel execution correctness
"""
