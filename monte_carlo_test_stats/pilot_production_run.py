"""
Pilot production run - single configuration test
Tests KS n=30 with 1M iterations
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from run_full_simulations import SimulationOrchestrator

def main():
    # Single configuration for pilot
    pilot_config = [{
        'statistic': 'kolmogorov_smirnov',
        'sample_size': 30,
        'iterations': 1_000_000,  # Reduced from 10M
        'quantiles': [0.75, 0.90, 0.95, 0.99],
        'convergence_threshold': 0.0001,
        'seed': 42
    }]
    
    orchestrator = SimulationOrchestrator()
    results = orchestrator.run_all_configurations(pilot_config)
    
    print(f"\nPilot complete: {results[0]['status']}")
    if results[0]['status'] == 'completed':
        print(f"File: {results[0]['filepath']}")
        print(f"Runtime: {results[0]['metadata']['runtime_seconds']:.1f}s")
        print(f"Converged: {results[0]['metadata']['convergence_achieved']}")

if __name__ == "__main__":
    main()
