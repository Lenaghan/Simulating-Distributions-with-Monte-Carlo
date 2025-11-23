"""
Production Monte Carlo Simulation Orchestration Script
Executes full-scale simulations for all test statistics and sample sizes
with convergence monitoring, checkpointing, and adaptive iteration control.

Now loads configuration from YAML files for better flexibility.
"""
import numpy as np
import pandas as pd
import h5py
import json
import yaml
import os
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

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages configuration from YAML files."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.simulation_config = {}
        self.convergence_config = {}
        
    def load_configs(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load both configuration files.
        
        Returns:
            Tuple of (simulation_config, convergence_config)
        """
        # Load simulation configuration
        sim_config_path = self.config_dir / "simulation_config.yaml"
        if sim_config_path.exists():
            with open(sim_config_path, 'r') as f:
                self.simulation_config = yaml.safe_load(f)
            logger.info(f"Loaded simulation config from {sim_config_path}")
        else:
            logger.warning(f"Simulation config not found at {sim_config_path}, using defaults")
            self.simulation_config = self._get_default_simulation_config()
        
        # Load convergence configuration
        conv_config_path = self.config_dir / "convergence_params.yaml"
        if conv_config_path.exists():
            with open(conv_config_path, 'r') as f:
                self.convergence_config = yaml.safe_load(f)
            logger.info(f"Loaded convergence config from {conv_config_path}")
        else:
            logger.warning(f"Convergence config not found at {conv_config_path}, using defaults")
            self.convergence_config = self._get_default_convergence_config()
        
        return self.simulation_config, self.convergence_config
    
    def _get_default_simulation_config(self) -> Dict[str, Any]:
        """Return default simulation configuration."""
        return {
            'test_statistics': ['kolmogorov_smirnov', 'durbin_watson', 'anderson_darling'],
            'sample_sizes': [30, 50, 100, 500, 1000],
            'iterations': {
                'initial': 1_000_000,
                'convergence_check': 5_000_000,
                'maximum': 10_000_000
            },
            'quantiles': [0.75, 0.90, 0.95, 0.99],
            'null_distribution': 'standard_normal',
            'random_seed': 42,
            'parallel': {
                'n_jobs': -1,
                'backend': 'loky',
                'batch_size': 10_000,
                'max_nbytes': '1M',
                'verbose': 10
            },
            'memory': {
                'chunk_size': 100_000,
                'enable_compression': True
            }
        }
    
    def _get_default_convergence_config(self) -> Dict[str, Any]:
        """Return default convergence configuration."""
        return {
            'quantile_stability': 0.0001,
            'batch_size': 100_000,
            'min_batches': 10,
            'confidence_level': 0.99,
            'checkpoint_interval': 100_000,
            'max_checkpoints': 50,
            'checkpoint_format': 'hdf5',
            'compression': 'snappy'
        }


def generate_parameter_grid(sim_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate parameter grid from configuration.
    
    Args:
        sim_config: Simulation configuration dictionary
        
    Returns:
        List of configuration dictionaries
    """
    statistics = sim_config['test_statistics']
    sample_sizes = sim_config['sample_sizes']
    iterations_config = sim_config['iterations']
    quantiles = sim_config['quantiles']
    seed = sim_config['random_seed']
    
    grid = []
    for statistic in statistics:
        for sample_size in sample_sizes:
            config = {
                'statistic': statistic,
                'sample_size': sample_size,
                'iterations': iterations_config['maximum'],
                'quantiles': quantiles,
                'convergence_threshold': None,  # Will be set from convergence config
                'seed': seed
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


def get_parallel_config(sim_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get parallel processing configuration.
    
    Args:
        sim_config: Simulation configuration dictionary
        
    Returns:
        Parallel processing configuration
    """
    parallel = sim_config.get('parallel', {})
    return {
        'n_jobs': parallel.get('n_jobs', -1),
        'backend': parallel.get('backend', 'loky'),
        'batch_size': parallel.get('batch_size', 10_000),
        'verbose': parallel.get('verbose', 10)
    }


class AdaptiveSimulator:
    """Handles convergence-adaptive simulation execution."""
    
    def __init__(
        self,
        sim_config: Dict[str, Any],
        conv_config: Dict[str, Any]
    ):
        """
        Initialize adaptive simulator with configuration.
        
        Args:
            sim_config: Simulation configuration
            conv_config: Convergence configuration
        """
        iterations = sim_config['iterations']
        self.initial_iterations = iterations['initial']
        self.convergence_threshold = conv_config['quantile_stability']
        self.max_iterations = iterations['maximum']
        self.check_interval = conv_config['checkpoint_interval']
    
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
    
    def __init__(
        self,
        sim_config: Dict[str, Any],
        conv_config: Dict[str, Any],
        checkpoint_dir: str = "data/interim"
    ):
        """
        Initialize simulation runner with configuration.
        
        Args:
            sim_config: Simulation configuration
            conv_config: Convergence configuration
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Get compression from convergence config
        compression = conv_config.get('compression', 'gzip')
        # Map snappy to gzip if not available in h5py
        if compression == 'snappy':
            compression = 'gzip'
            logger.info("Snappy compression not available in h5py, using gzip")
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=conv_config.get('max_checkpoints', 50),
            compression=compression,
            quantile_levels=sim_config.get('quantiles', [0.75, 0.90, 0.95, 0.99])
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
            # Updated to handle 3-value return from new checkpoint system
            data, metadata, quantiles = self.checkpoint_manager.load_checkpoint(str(latest))
            # Return only data and metadata for backward compatibility
            return data, metadata
        
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
        # Updated to handle 3-value return from new checkpoint system
        data, metadata, quantiles = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        completed = metadata['iterations_completed']
        
        if completed >= target_iterations:
            return data[:target_iterations]
        
        # Run additional iterations
        remaining = target_iterations - completed
        logger.info(f"Running {remaining} additional iterations")
        
        # Create engine with same configuration
        engine = MonteCarloEngine(
            sample_size=metadata['sample_size'],
            random_seed=metadata['seed']
        )
        
        # Continue from last iteration
        engine.current_iteration = completed
        additional = engine.generate_samples(
            statistic=metadata['statistic'],
            n_iterations=remaining
        )
        
        # Combine results
        return np.concatenate([data, additional])


class SimulationOrchestrator:
    """Orchestrates full simulation workflow."""
    
    def __init__(
        self,
        sim_config: Dict[str, Any],
        conv_config: Dict[str, Any],
        checkpoint_dir: str = "data/interim",
        output_dir: str = "data/processed"
    ):
        """
        Initialize orchestrator with configuration.
        
        Args:
            sim_config: Simulation configuration
            conv_config: Convergence configuration
            checkpoint_dir: Directory for checkpoints
            output_dir: Directory for final results
        """
        self.sim_config = sim_config
        self.conv_config = conv_config
        self.simulator = AdaptiveSimulator(sim_config, conv_config)
        self.runner = SimulationRunner(sim_config, conv_config, checkpoint_dir)
        self.parallel_config = get_parallel_config(sim_config)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_with_monitoring(
        self,
        statistic: str,
        sample_size: int,
        max_iterations: int,
        quantiles: List[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run simulation with convergence monitoring.
        
        Args:
            statistic: Test statistic name
            sample_size: Sample size
            max_iterations: Maximum iterations
            quantiles: Quantile levels to monitor
            
        Returns:
            (results array, metadata dict)
        """
        if quantiles is None:
            quantiles = self.sim_config.get('quantiles', [0.75, 0.90, 0.95, 0.99])
            
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
            threshold=self.conv_config['quantile_stability'],
            batch_size=self.conv_config['checkpoint_interval']  # Use checkpoint_interval as batch size
        )
        
        # Start with initial iterations or continue from checkpoint
        if existing_data is not None:
            all_results = existing_data
            iterations_completed = len(existing_data)
            # Add existing data to monitor in batches
            for i in range(0, iterations_completed, self.conv_config['checkpoint_interval']):
                batch = existing_data[i:i+self.conv_config['checkpoint_interval']]
                monitor.add_batch(batch)
        else:
            # Run initial batch
            logger.info(f"Starting fresh simulation with {self.simulator.initial_iterations} iterations")
            engine = MonteCarloEngine(
                sample_size=sample_size,
                random_seed=self.sim_config['random_seed']
            )
            all_results = engine.generate_samples(
                statistic=statistic,
                n_iterations=self.simulator.initial_iterations
            )
            iterations_completed = self.simulator.initial_iterations
            # Add to monitor
            monitor.add_batch(all_results)
        
        # Monitor convergence
        converged = False
        
        while iterations_completed < max_iterations and not converged:
            # Check convergence (only after initial iterations)
            if iterations_completed >= self.simulator.initial_iterations:
                converged = monitor.is_converged()
                
                if converged:
                    logger.info(f"  Convergence achieved at {iterations_completed} iterations")
                    break
            
            # Determine next batch size
            # Create convergence status for adaptive simulator
            convergence_status = {}
            if hasattr(monitor, 'quantile_converged'):
                convergence_status = monitor.quantile_converged
            else:
                # Fallback: assume not converged for all quantiles
                for q in quantiles:
                    convergence_status[q] = False
            
            next_iterations = self.simulator.determine_next_iterations(
                iterations_completed, convergence_status
            )
            
            if next_iterations == iterations_completed:
                logger.info("No iteration increase needed")
                break
            
            # Run additional iterations
            additional = next_iterations - iterations_completed
            logger.info(f"Running {additional} more iterations (total: {next_iterations})")
            
            engine = MonteCarloEngine(
                sample_size=sample_size,
                random_seed=self.sim_config['random_seed']
            )
            engine.current_iteration = iterations_completed
            
            batch_results = engine.generate_samples(
                statistic=statistic,
                n_iterations=additional
            )
            
            # Add batch to monitor
            monitor.add_batch(batch_results)
            
            # Combine results
            all_results = np.concatenate([all_results, batch_results])
            iterations_completed = next_iterations
            
            # Save checkpoint every checkpoint_interval iterations
            if iterations_completed % self.conv_config['checkpoint_interval'] == 0:
                metadata = {
                    'statistic': statistic,
                    'sample_size': sample_size,
                    'iterations_completed': iterations_completed,
                    'convergence_achieved': converged,
                    'seed': self.sim_config['random_seed'],
                    'timestamp': datetime.now().isoformat()
                }
                checkpoint_path = self.runner.save_checkpoint(all_results, metadata)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Final metadata
        metadata = {
            'statistic': statistic,
            'sample_size': sample_size,
            'iterations_completed': iterations_completed,
            'convergence_achieved': converged,
            'execution_time': time.time() - start_time,
            'seed': self.sim_config['random_seed'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save final checkpoint
        checkpoint_path = self.runner.save_checkpoint(all_results, metadata)
        
        return all_results, metadata
    
    def save_results(
        self,
        data: np.ndarray,
        metadata: Dict[str, Any],
        filename: str
    ):
        """Save final results to processed directory."""
        filepath = self.output_dir / filename
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('data', data=data, compression='gzip')
            f.attrs['metadata'] = json.dumps(metadata)
        
        logger.info(f"Results saved: {filepath}")
    
    def run_all_configurations(
        self,
        configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run all simulation configurations.
        
        Args:
            configs: List of configuration dictionaries
            
        Returns:
            Summary of results
        """
        results = {}
        total = len(configs)
        
        for i, config in enumerate(configs, 1):
            statistic = config['statistic']
            sample_size = config['sample_size']
            
            logger.info("=" * 60)
            logger.info(f"Configuration {i}/{total}: {statistic}, n={sample_size}")
            logger.info("=" * 60)
            
            try:
                # Set convergence threshold from config
                config['convergence_threshold'] = self.conv_config['quantile_stability']
                
                # Run simulation
                sim_results, metadata = self.run_with_monitoring(
                    statistic=statistic,
                    sample_size=sample_size,
                    max_iterations=config['iterations'],
                    quantiles=config['quantiles']
                )
                
                # Save results
                filename = f"{statistic}_n{sample_size}_final.h5"
                self.save_results(sim_results, metadata, filename)
                
                # Store summary
                results[f"{statistic}_n{sample_size}"] = {
                    'iterations': len(sim_results),
                    'converged': metadata['convergence_achieved'],
                    'time': metadata['execution_time']
                }
                
                logger.info(f"  Completed: {statistic} n={sample_size}")
                logger.info(f"  Iterations: {len(sim_results)}")
                logger.info(f"  Converged: {metadata['convergence_achieved']}")
                logger.info(f"  Time: {metadata['execution_time']:.2f}s")
                
            except Exception as e:
                logger.error(f"  Failed: {statistic} n={sample_size} - {str(e)}")
                results[f"{statistic}_n{sample_size}"] = {
                    'error': str(e)
                }
        
        logger.info("\n" + "=" * 60)
        logger.info("SIMULATION COMPLETE")
        logger.info("=" * 60)
        
        return results


def main():
    """Main execution function."""
    # Load configurations
    config_loader = ConfigLoader(config_dir="config")
    sim_config, conv_config = config_loader.load_configs()
    
    logger.info("Starting production Monte Carlo simulations")
    logger.info(f"Test statistics: {sim_config['test_statistics']}")
    logger.info(f"Sample sizes: {sim_config['sample_sizes']}")
    logger.info(f"Max iterations: {sim_config['iterations']['maximum']}")
    logger.info(f"Convergence threshold: {conv_config['quantile_stability']}")
    
    # Generate and prioritize configurations
    configs = generate_parameter_grid(sim_config)
    configs = prioritize_configurations(configs)
    
    # Initialize orchestrator
    orchestrator = SimulationOrchestrator(
        sim_config=sim_config,
        conv_config=conv_config,
        checkpoint_dir="data/interim",
        output_dir="data/processed"
    )
    
    # Run all simulations
    results = orchestrator.run_all_configurations(configs)
    
    # Save summary
    summary_path = Path("data/processed") / "simulation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Summary saved: {summary_path}")
    
    # Print summary
    successful = sum(1 for r in results.values() if 'error' not in r)
    failed = len(results) - successful
    
    logger.info(f"\nFinal Summary:")
    logger.info(f"  Successful: {successful}/{len(results)}")
    logger.info(f"  Failed: {failed}/{len(results)}")
    
    if failed > 0:
        logger.info("\nFailed configurations:")
        for key, result in results.items():
            if 'error' in result:
                logger.info(f"  {key}: {result['error']}")
    
    return results


if __name__ == "__main__":
    results = main()
