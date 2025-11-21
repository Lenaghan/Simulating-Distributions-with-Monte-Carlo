"""
Production Monte Carlo Simulation Orchestration Script
Executes full-scale simulations for all test statistics and sample sizes
with convergence monitoring, checkpointing, and adaptive iteration control.
"""
import numpy as np
import pandas as pd
import h5py
import json
import yaml
import logging
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.simulation.engine import MonteCarloEngine
from src.simulation.convergence import check_convergence, ConvergenceMonitor
from src.simulation.checkpoints import CheckpointManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_parameter_grid() -> List[Dict[str, Any]]:
    """
    Generate parameter grid for all production simulations.
    
    Returns:
        List of configuration dictionaries (15 total)
    """
    statistics = ['kolmogorov_smirnov', 'durbin_watson', 'anderson_darling']
    sample_sizes = [30, 50, 100, 500, 1000]
    
    grid = []
    for statistic in statistics:
        for sample_size in sample_sizes:
            config = {
                'statistic': statistic,
                'sample_size': sample_size,
                'iterations': 10_000_000,  # Default 10M
                'quantiles': [0.75, 0.90, 0.95, 0.99],
                'convergence_threshold': 0.0001,
                'seed': 42
            }
            grid.append(config)
    
    logger.info(f"Generated parameter grid with {len(grid)} configurations")
    return grid


def prioritize_configurations(configs: List[Dict]) -> List[Dict]:
    """
    Sort configurations to process smaller sample sizes first.
    
    Args:
        configs: List of configuration dictionaries
        
    Returns:
        Sorted list with smaller sample sizes first
    """
    return sorted(configs, key=lambda x: x['sample_size'])


def get_parallel_config() -> Dict[str, Any]:
    """Get parallel processing configuration."""
    return {
        'n_jobs': -1,  # Use all CPU cores
        'backend': 'loky',
        'batch_size': 10_000,
        'verbose': 10
    }


class AdaptiveSimulator:
    """Handles convergence-adaptive simulation execution."""
    
    def __init__(
        self,
        initial_iterations: int = 1_000_000,
        convergence_threshold: float = 0.0001,
        max_iterations: int = 20_000_000,
        check_interval: int = 100_000
    ):
        self.initial_iterations = initial_iterations
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.check_interval = check_interval
    
    def determine_next_iterations(
        self,
        current_iterations: int,
        convergence_status: Dict[float, bool]
    ) -> int:
        """
        Determine next iteration count based on convergence.
        
        Args:
            current_iterations: Current number of iterations
            convergence_status: Dict of quantile -> convergence bool
            
        Returns:
            Recommended iterations for next batch
        """
        if all(convergence_status.values()):
            # All converged - no increase
            return current_iterations
        
        # Calculate proportion converged
        n_converged = sum(convergence_status.values())
        n_total = len(convergence_status)
        proportion_converged = n_converged / n_total
        
        # Adaptive increase
        if proportion_converged < 0.5:
            # Less than half converged - larger increase
            new_iterations = int(current_iterations * 1.5 * 1.5)
        else:
            # More than half converged - moderate increase
            new_iterations = int(current_iterations * 1.5)
        
        # Cap at maximum
        new_iterations = min(new_iterations, self.max_iterations)
        
        logger.info(
            f"Convergence: {n_converged}/{n_total}. "
            f"Iterations: {current_iterations} -> {new_iterations}"
        )
        
        return new_iterations


class SimulationRunner:
    """Handles individual simulation execution with checkpointing."""
    
    def __init__(self, checkpoint_dir: str = "data/interim"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=3,
            compression='gzip'
        )
    
    def save_checkpoint(
        self,
        data: np.ndarray,
        metadata: Dict[str, Any]
    ) -> str:
        """Save checkpoint with metadata."""
        name_prefix = f"checkpoint_{metadata['statistic']}_{metadata['sample_size']}"
        return self.checkpoint_manager.save_checkpoint(data, metadata, name_prefix)
    
    def recover_from_checkpoint(
        self,
        statistic: str,
        sample_size: int
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Attempt to recover from most recent checkpoint.
        
        Returns:
            (data, metadata) if checkpoint found, else (None, None)
        """
        pattern = f"checkpoint_{statistic}_{sample_size}_*.h5"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        
        if checkpoints:
            latest = checkpoints[-1]
            logger.info(f"Recovering from checkpoint: {latest}")
            return self.checkpoint_manager.load_checkpoint(str(latest))
        
        return None, None
    
    def continue_from_checkpoint(
        self,
        checkpoint_path: str,
        target_iterations: int
    ) -> np.ndarray:
        """
        Continue simulation from checkpoint to target iterations.
        
        Args:
            checkpoint_path: Path to checkpoint file
            target_iterations: Total iterations desired
            
        Returns:
            Combined results array
        """
        data, metadata = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        completed = metadata['iterations_completed']
        
        if completed >= target_iterations:
            return data[:target_iterations]
        
        # Run additional iterations
        remaining = target_iterations - completed
        logger.info(f"Running {remaining} additional iterations")
        
        # Create engine with same configuration
        engine = MonteCarloEngine(seed=metadata.get('seed', 42))
        
        # Generate new results starting from where we left off
        new_results = engine.simulate(
            test_statistic=metadata['statistic'],
            n=metadata['sample_size'],
            iterations=remaining,
            show_progress=True
        )
        
        # Combine with existing
        combined = np.concatenate([data, new_results])
        
        return combined


class ProductionSimulator:
    """Main production simulation orchestrator."""
    
    def __init__(
        self,
        check_interval: int = 100_000,
        convergence_threshold: float = 0.0001
    ):
        self.check_interval = check_interval
        self.convergence_threshold = convergence_threshold
        self.engine = MonteCarloEngine(seed=42, n_jobs=-1)
        self.adaptive_sim = AdaptiveSimulator(
            convergence_threshold=convergence_threshold
        )
        self.runner = SimulationRunner()
    
    def run_with_monitoring(
        self,
        statistic: str,
        sample_size: int,
        max_iterations: int = 10_000_000,
        quantiles: List[float] = [0.75, 0.90, 0.95, 0.99]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run simulation with convergence monitoring.
        
        Returns:
            (results array, metadata dict)
        """
        start_time = time.time()
        
        # Check for existing checkpoint
        existing_data, existing_meta = self.runner.recover_from_checkpoint(
            statistic, sample_size
        )
        
        if existing_data is not None:
            logger.info(f"Found checkpoint with {len(existing_data)} iterations")
            if existing_meta.get('convergence_achieved', False):
                # Already converged, return existing
                return existing_data, existing_meta
        
        # Initialize convergence monitor
        monitor = ConvergenceMonitor(
            quantiles=quantiles,
            threshold=self.convergence_threshold,
            batch_size=self.check_interval
        )
        
        all_results = existing_data if existing_data is not None else np.array([])
        iterations_completed = len(all_results)
        converged = False
        
        while iterations_completed < max_iterations and not converged:
            # Determine batch size
            batch_size = min(self.check_interval, max_iterations - iterations_completed)
            
            logger.info(
                f"{statistic} n={sample_size}: "
                f"Running batch of {batch_size} iterations "
                f"(total: {iterations_completed + batch_size}/{max_iterations})"
            )
            
            # Run batch
            batch_results = self.engine.simulate(
                test_statistic=statistic,
                n=sample_size,
                iterations=batch_size,
                show_progress=True
            )
            
            # Add to monitor
            monitor.add_batch(batch_results)
            
            # Combine results
            if len(all_results) == 0:
                all_results = batch_results
            else:
                all_results = np.concatenate([all_results, batch_results])
            
            iterations_completed = len(all_results)
            
            # Check convergence
            if iterations_completed >= self.adaptive_sim.initial_iterations:
                converged = monitor.is_converged()
                
                if converged:
                    logger.info(
                        f"Convergence achieved after {iterations_completed} iterations"
                    )
                    break
            
            # Save checkpoint every 500k iterations
            if iterations_completed % 500_000 == 0:
                checkpoint_meta = {
                    'statistic': statistic,
                    'sample_size': sample_size,
                    'iterations_completed': iterations_completed,
                    'convergence_achieved': converged,
                    'last_quantiles': monitor.get_current_estimates(),
                    'seed': 42
                }
                self.runner.save_checkpoint(all_results, checkpoint_meta)
        
        # Calculate final quantiles
        final_quantiles = {}
        for q in quantiles:
            final_quantiles[q] = float(np.quantile(all_results, q))
        
        # Prepare metadata
        metadata = {
            'statistic': statistic,
            'sample_size': sample_size,
            'iterations_completed': iterations_completed,
            'convergence_achieved': converged,
            'quantiles': final_quantiles,
            'runtime_seconds': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'seed_used': 42
        }
        
        return all_results, metadata


def save_production_results(
    filepath: Path,
    results: np.ndarray,
    metadata: Dict[str, Any]
):
    """
    Save results to HDF5 with compression and metadata.
    
    Args:
        filepath: Output file path
        results: Simulation results array
        metadata: Simulation metadata dictionary
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Save results with compression
        f.create_dataset(
            'results',
            data=results,
            compression='gzip',
            compression_opts=4
        )
        
        # Save metadata as attributes
        for key, value in metadata.items():
            if isinstance(value, dict):
                # Convert dict to JSON string
                f.attrs[key] = json.dumps(value)
            else:
                f.attrs[key] = value
    
    logger.info(f"Results saved to {filepath}")


def generate_simulation_metadata(
    statistic: str,
    sample_size: int,
    iterations: int,
    converged: bool,
    runtime: float,
    quantiles: Dict[float, float]
) -> Dict[str, Any]:
    """Generate complete metadata for simulation."""
    return {
        'statistic': statistic,
        'sample_size': sample_size,
        'iterations_completed': iterations,
        'convergence_achieved': converged,
        'runtime_seconds': runtime,
        'quantiles': quantiles,
        'timestamp': datetime.now().isoformat(),
        'seed_used': 42
    }


class SimulationOrchestrator:
    """Top-level orchestrator for all simulations."""
    
    def __init__(self):
        self.simulator = ProductionSimulator()
        self.errors_logged = 0
        self.results_dir = Path("data/processed")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_all_configurations(
        self,
        configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run all simulation configurations.
        
        Returns:
            List of result summaries
        """
        results = []
        total = len(configs)
        
        for i, config in enumerate(configs, 1):
            statistic = config['statistic']
            sample_size = config['sample_size']
            
            logger.info(f"\n{'='*60}")
            logger.info(
                f"Configuration {i}/{total}: {statistic}, n={sample_size}"
            )
            logger.info('='*60)
            
            try:
                # Run simulation
                sim_results, metadata = self.simulator.run_with_monitoring(
                    statistic=statistic,
                    sample_size=sample_size,
                    max_iterations=config['iterations'],
                    quantiles=config['quantiles']
                )
                
                # Save results
                filename = f"{statistic}_{sample_size}_results.h5"
                filepath = self.results_dir / filename
                save_production_results(filepath, sim_results, metadata)
                
                # Save metadata separately as JSON
                meta_filepath = self.results_dir / f"{statistic}_{sample_size}_metadata.json"
                with open(meta_filepath, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                results.append({
                    'statistic': statistic,
                    'sample_size': sample_size,
                    'status': 'completed',
                    'filepath': str(filepath),
                    'metadata': metadata
                })
                
                logger.info(f"[OK] Completed: {statistic} n={sample_size}")
                
            except Exception as e:
                logger.error(f"✗ Failed: {statistic} n={sample_size} - {str(e)}")
                self.errors_logged += 1
                
                results.append({
                    'statistic': statistic,
                    'sample_size': sample_size,
                    'status': 'failed',
                    'error': str(e)
                })
                
                # Continue with next configuration
                continue
        
        return results


def main():
    """Main execution function."""
    logger.info("Starting production Monte Carlo simulations")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Generate parameter grid
    configs = generate_parameter_grid()
    
    # Prioritize smaller sample sizes
    configs = prioritize_configurations(configs)
    
    # Create orchestrator and run
    orchestrator = SimulationOrchestrator()
    results = orchestrator.run_all_configurations(configs)
    
    # Summary
    completed = sum(1 for r in results if r['status'] == 'completed')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    logger.info(f"\n{'='*60}")
    logger.info("SIMULATION SUMMARY")
    logger.info(f"Completed: {completed}/{len(configs)}")
    logger.info(f"Failed: {failed}/{len(configs)}")
    
    # Save summary report
    summary_path = Path("reports/production_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_configurations': len(configs),
            'completed': completed,
            'failed': failed,
            'results': results
        }, f, indent=2)
    
    logger.info(f"Summary saved to {summary_path}")
    
    if failed == 0:
        logger.info("✓ All simulations completed successfully!")
        return 0
    else:
        logger.warning(f"⚠ {failed} simulations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
