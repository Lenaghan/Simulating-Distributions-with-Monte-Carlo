"""
Pilot simulation to test implementation before full-scale runs.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import yaml
import time
from src.simulation.engine import MonteCarloEngine
from src.simulation.convergence import ConvergenceMonitor
from src.simulation.checkpoints import CheckpointManager
import matplotlib.pyplot as plt


def run_pilot_simulation():
    """Run small-scale pilot to validate implementation."""
    
    print("=" * 60)
    print("MONTE CARLO PILOT SIMULATION")
    print("=" * 60)
    
    # Load configuration
    with open('config/simulation_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Use smaller values for pilot
    pilot_iterations = 100_000
    pilot_sample_size = 100
    test_stat = 'kolmogorov_smirnov'
    
    print(f"\nConfiguration:")
    print(f"  Test statistic: {test_stat}")
    print(f"  Sample size: {pilot_sample_size}")
    print(f"  Iterations: {pilot_iterations:,}")
    print(f"  Random seed: {config['random_seed']}")
    
    # Initialize components
    engine = MonteCarloEngine(config=config, verbose=True)
    checkpoint_mgr = CheckpointManager()
    convergence_mon = ConvergenceMonitor(
        quantiles=[0.75, 0.90, 0.95, 0.99],
        threshold=0.0001,
        batch_size=10_000
    )
    
    # Run simulation
    print(f"\nStarting simulation...")
    start_time = time.time()
    
    results = engine.simulate(
        test_statistic=test_stat,
        n=pilot_sample_size,
        iterations=pilot_iterations,
        show_progress=True
    )
    
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.2f} seconds")
    
    # Calculate statistics
    print(f"\nResults Summary:")
    print(f"  Mean: {results.mean():.6f}")
    print(f"  Std: {results.std():.6f}")
    print(f"  Min: {results.min():.6f}")
    print(f"  Max: {results.max():.6f}")
    
    # Calculate quantiles
    quantiles = [0.75, 0.90, 0.95, 0.99]
    print(f"\nEmpirical Quantiles:")
    for q in quantiles:
        value = np.quantile(results, q)
        print(f"  {q:.0%}: {value:.6f}")
    
    # Compare to theoretical (KS asymptotic)
    n = pilot_sample_size
    critical_95 = 1.36 / np.sqrt(n)
    empirical_95 = np.quantile(results, 0.95)
    
    print(f"\nValidation (KS n={n}):")
    print(f"  Theoretical 95% critical value: {critical_95:.6f}")
    print(f"  Empirical 95% quantile: {empirical_95:.6f}")
    print(f"  Difference: {abs(empirical_95 - critical_95):.6f}")
    
    # Save checkpoint
    metadata = {
        'test_statistic': test_stat,
        'sample_size': pilot_sample_size,
        'iterations': pilot_iterations,
        'mean': float(results.mean()),
        'std': float(results.std())
    }
    checkpoint_path = checkpoint_mgr.save_checkpoint(results, metadata, 'pilot')
    print(f"\nCheckpoint saved: {checkpoint_path}")
    
    # Create visualization
    create_pilot_plots(results, test_stat, pilot_sample_size)
    
    # Estimate full-scale timing
    time_per_iter = elapsed / pilot_iterations
    full_iterations = 10_000_000
    estimated_hours = (time_per_iter * full_iterations * 3 * 5) / 3600
    
    print(f"\n" + "=" * 60)
    print(f"SCALING ESTIMATES")
    print(f"  Time per iteration: {time_per_iter*1000:.3f} ms")
    print(f"  Estimated for 10M iterations: {time_per_iter * full_iterations:.1f} seconds")
    print(f"  Full suite (3 stats × 5 sizes): {estimated_hours:.1f} hours")
    print("=" * 60)
    
    return results


def create_pilot_plots(results, test_stat, sample_size):
    """Create basic visualizations."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(results, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Test Statistic Value')
    ax.set_ylabel('Density')
    ax.set_title(f'{test_stat} Distribution (n={sample_size})')
    ax.grid(True, alpha=0.3)
    
    # Empirical CDF
    ax = axes[1]
    sorted_results = np.sort(results)
    ecdf = np.arange(1, len(results) + 1) / len(results)
    ax.plot(sorted_results, ecdf, linewidth=2)
    ax.set_xlabel('Test Statistic Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical CDF')
    ax.grid(True, alpha=0.3)
    
    # Mark quantiles
    for q in [0.75, 0.90, 0.95, 0.99]:
        value = np.quantile(results, q)
        ax.axvline(value, color='red', linestyle='--', alpha=0.5)
        ax.text(value, q, f' {q:.0%}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('reports/figures/pilot_distribution.png', dpi=100)
    print(f"\nPlot saved: reports/figures/pilot_distribution.png")
    plt.close()


if __name__ == "__main__":
    results = run_pilot_simulation()
