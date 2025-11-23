"""
Cross-Validation Analysis Using Final Simulation Results
Reads simulation data from final result files in data/processed
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'cross_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CheckpointCrossValidator:
    """Cross-validation using final result files from data/processed."""
    
    def __init__(self, data_dir: Path = Path("data"), n_chains: int = 5):
        """
        Initialize cross-validator.
        
        Args:
            data_dir: Base data directory
            n_chains: Number of chains for Gelman-Rubin
        """
        self.processed_dir = data_dir / "processed"
        self.n_chains = n_chains
        self.validation_results = {}
        
        # Check if h5py is available
        try:
            import h5py
            self.h5py = h5py
            logger.info("h5py module loaded successfully")
        except ImportError:
            logger.error("h5py not installed! Install with: pip install h5py")
            raise ImportError("h5py required for reading HDF5 files")
    
    def load_simulation_data(self, statistic_name: str, sample_size: int) -> Optional[np.ndarray]:
        """
        Load simulation data from final result file.
        
        Args:
            statistic_name: Name of test statistic
            sample_size: Sample size
            
        Returns:
            Array of simulated values or None if not found
        """
        # Load from processed directory (final results)
        final_file = self.processed_dir / f"{statistic_name}_n{sample_size}_final.h5"
        
        if not final_file.exists():
            logger.warning(f"Final results file not found: {final_file.name}")
            return None
        
        logger.info(f"Loading from: {final_file.name}")
        try:
            with self.h5py.File(final_file, 'r') as f:
                if 'data' in f:
                    data = f['data'][:]
                    logger.info(f"Loaded {len(data):,} simulated values")
                    return data
                else:
                    logger.error(f"No 'data' dataset in {final_file.name}")
                    logger.info(f"Available keys: {list(f.keys())}")
                    return None
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return None
    
    def split_into_chains(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Split data into chains for Gelman-Rubin analysis.
        
        Args:
            data: Full simulation data
            
        Returns:
            List of chain arrays
        """
        n = len(data)
        chain_length = n // self.n_chains
        
        chains = []
        for i in range(self.n_chains):
            start_idx = i * chain_length
            end_idx = start_idx + chain_length if i < self.n_chains - 1 else n
            chains.append(data[start_idx:end_idx])
        
        logger.info(f"Split {n:,} values into {self.n_chains} chains of ~{chain_length:,} values each")
        return chains
    
    def calculate_gelman_rubin(self, chains: List[np.ndarray], quantile: float) -> Tuple[float, float]:
        """
        Calculate Gelman-Rubin R-hat statistic.
        
        Args:
            chains: List of chain arrays
            quantile: Quantile level to analyze
            
        Returns:
            Tuple of (R-hat, coefficient of variation)
        """
        n_chains = len(chains)
        
        # Calculate quantile for each chain
        chain_quantiles = [np.quantile(chain, quantile) for chain in chains]
        
        # Calculate between-chain variance B
        chain_means = np.array(chain_quantiles)
        overall_mean = np.mean(chain_means)
        B = np.var(chain_means, ddof=1) * len(chains[0])
        
        # Calculate within-chain variance W
        # Use batch means within each chain
        W_values = []
        for chain in chains:
            # Split chain into batches
            n_batches = min(100, len(chain) // 100)
            if n_batches > 1:
                batch_size = len(chain) // n_batches
                batch_quantiles = []
                for j in range(n_batches):
                    start = j * batch_size
                    end = start + batch_size if j < n_batches - 1 else len(chain)
                    batch_quantiles.append(np.quantile(chain[start:end], quantile))
                W_values.append(np.var(batch_quantiles, ddof=1))
        
        W = np.mean(W_values) if W_values else 0
        
        # Calculate R-hat
        if W > 0:
            var_plus = ((n_chains - 1) / n_chains) * W + (1 / n_chains) * B
            r_hat = np.sqrt(var_plus / W)
        else:
            r_hat = 1.0
        
        # Calculate coefficient of variation
        cv = np.std(chain_quantiles) / np.mean(chain_quantiles) if np.mean(chain_quantiles) != 0 else np.inf
        
        return r_hat, cv
    
    def validate_configuration(self, statistic_name: str, sample_size: int) -> Dict:
        """
        Perform cross-validation for a single configuration.
        
        Args:
            statistic_name: Name of test statistic
            sample_size: Sample size
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-validating {statistic_name} with n={sample_size}")
        logger.info(f"{'='*60}")
        
        # Load simulation data
        data = self.load_simulation_data(statistic_name, sample_size)
        
        if data is None or len(data) == 0:
            logger.error(f"No data available for {statistic_name} n={sample_size}")
            return {
                "status": "no_data",
                "statistic": statistic_name,
                "sample_size": sample_size
            }
        
        # Split into chains
        chains = self.split_into_chains(data)
        
        # Analyze each quantile
        quantiles = [0.75, 0.90, 0.95, 0.99]
        results = {
            "statistic": statistic_name,
            "sample_size": sample_size,
            "n_iterations": len(data),
            "n_chains": self.n_chains,
            "chain_lengths": [len(chain) for chain in chains],
            "quantile_diagnostics": {}
        }
        
        for q in quantiles:
            r_hat, cv = self.calculate_gelman_rubin(chains, q)
            
            # Calculate quantile estimates from each chain
            chain_estimates = [np.quantile(chain, q) for chain in chains]
            
            # Convergence criteria
            converged = r_hat < 1.1 and cv < 0.01
            
            results["quantile_diagnostics"][q] = {
                "r_hat": r_hat,
                "cv": cv,
                "converged": converged,
                "chain_estimates": chain_estimates,
                "mean_estimate": np.mean(chain_estimates),
                "std_estimate": np.std(chain_estimates)
            }
            
            logger.info(f"  Q{q:.0%}: R-hat={r_hat:.4f}, CV={cv:.4f} - {'CONVERGED' if converged else 'NOT CONVERGED'}")
        
        # Overall convergence
        results["overall_converged"] = all(
            results["quantile_diagnostics"][q]["converged"] for q in quantiles
        )
        
        logger.info(f"Overall convergence: {'YES' if results['overall_converged'] else 'NO'}")
        
        return results
    
    def validate_all_configurations(self):
        """Run validation for all standard configurations."""
        # Standard configurations as defined in run_full_simulations.py
        statistics = ['kolmogorov_smirnov', 'durbin_watson', 'anderson_darling']
        sample_sizes = [30, 50, 100, 500, 1000]
        
        # First, scan for available data
        logger.info("\n" + "="*70)
        logger.info("SCANNING FOR AVAILABLE DATA FILES")
        logger.info("="*70)
        logger.info(f"Looking in: {self.processed_dir}")
        
        available = []
        for stat_name in statistics:
            for sample_size in sample_sizes:
                final_file = self.processed_dir / f"{stat_name}_n{sample_size}_final.h5"
                if final_file.exists():
                    available.append((stat_name, sample_size))
                    logger.info(f"[OK] Found: {stat_name} n={sample_size}")
                else:
                    logger.warning(f"[FAIL] Missing: {stat_name} n={sample_size}")
        
        logger.info(f"\nTotal configurations with data: {len(available)}/15")
        
        if not available:
            logger.error("No data files found! Please run run_full_simulations.py first.")
            return
        
        # Now validate each available configuration
        logger.info("\n" + "="*70)
        logger.info("STARTING CROSS-VALIDATION")
        logger.info("="*70)
        
        for stat_name, sample_size in available:
            key = f"{stat_name}_{sample_size}"
            self.validation_results[key] = self.validate_configuration(stat_name, sample_size)
    
    def generate_summary_report(self) -> pd.DataFrame:
        """Generate summary DataFrame."""
        summary_data = []
        
        for config_key, results in self.validation_results.items():
            if results.get("status") == "no_data":
                continue
            
            parts = config_key.rsplit("_", 1)
            stat_name = parts[0].replace("_", " ").title()
            sample_size = parts[1]
            
            # Overall row
            row = {
                "Statistic": stat_name,
                "Sample Size": sample_size,
                "Iterations": results.get("n_iterations", 0),
                "Converged": "[OK]" if results.get("overall_converged", False) else "[FAIL]"
            }
            
            # Add R-hat for each quantile
            if "quantile_diagnostics" in results:
                for q in [0.75, 0.90, 0.95, 0.99]:
                    if q in results["quantile_diagnostics"]:
                        diag = results["quantile_diagnostics"][q]
                        row[f"R-hat {q:.0%}"] = f"{diag['r_hat']:.4f}"
                        row[f"CV {q:.0%}"] = f"{diag['cv']:.4f}"
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, output_dir: Path = Path("reports/cross_validation")):
        """Save all results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results as JSON (with proper type conversion)
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        json_path = output_dir / f"cross_validation_results_{timestamp}.json"
        serializable_results = convert_types(self.validation_results)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Raw results saved to {json_path}")
        
        # Save summary CSV
        summary_df = self.generate_summary_report()
        if not summary_df.empty:
            csv_path = output_dir / f"cross_validation_summary_{timestamp}.csv"
            summary_df.to_csv(csv_path, index=False)
            logger.info(f"Summary saved to {csv_path}")
        
        # Generate and save detailed report
        report = self.generate_detailed_report()
        report_path = output_dir / f"cross_validation_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
    
    def generate_detailed_report(self) -> str:
        """Generate detailed markdown report."""
        report = []
        report.append("# Cross-Validation Report with Gelman-Rubin Diagnostics\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"Data source: {self.processed_dir}\n")
        report.append(f"Number of chains: {self.n_chains}\n\n")
        
        # Summary
        total_configs = len([r for r in self.validation_results.values() 
                           if r.get("status") != "no_data"])
        converged = sum(1 for r in self.validation_results.values() 
                       if r.get("overall_converged", False))
        
        report.append("## Executive Summary\n")
        report.append(f"- Configurations analyzed: {total_configs}\n")
        report.append(f"- Fully converged: {converged}\n")
        report.append(f"- Convergence rate: {100*converged/total_configs:.1f}%\n" if total_configs > 0 else "- Convergence rate: N/A\n")
        
        # Convergence criteria
        report.append("\n## Convergence Criteria\n")
        report.append("- **R-hat < 1.1**: Chains have mixed well\n")
        report.append("- **CV < 0.01**: Low variance across chains\n\n")
        
        # Detailed results table
        report.append("## Detailed Results\n\n")
        report.append("| Configuration | n | Iterations | Q75% | Q90% | Q95% | Q99% | Status |\n")
        report.append("|---------------|---|------------|------|------|------|------|--------|\n")
        
        for key in sorted(self.validation_results.keys()):
            results = self.validation_results[key]
            
            if results.get("status") == "no_data":
                parts = key.rsplit("_", 1)
                stat = parts[0].split("_")[0].upper()
                n = parts[1]
                report.append(f"| {stat} | {n} | NO DATA | - | - | - | - | [FAIL] |\n")
                continue
            
            parts = key.rsplit("_", 1)
            stat = parts[0].split("_")[0].upper()
            n = parts[1]
            iterations = results.get("n_iterations", 0)
            
            row = f"| {stat} | {n} | {iterations:,} "
            
            if "quantile_diagnostics" in results:
                for q in [0.75, 0.90, 0.95, 0.99]:
                    if q in results["quantile_diagnostics"]:
                        r_hat = results["quantile_diagnostics"][q]["r_hat"]
                        converged = results["quantile_diagnostics"][q]["converged"]
                        symbol = "[OK]" if converged else "[FAIL]"
                        row += f"| {r_hat:.3f}{symbol} "
                    else:
                        row += "| - "
            
            overall = "[OK]" if results.get("overall_converged", False) else "[FAIL]"
            row += f"| {overall} |\n"
            report.append(row)
        
        report.append("\n## Interpretation\n")
        if converged == total_configs and total_configs > 0:
            report.append("✅ **All configurations show excellent convergence**\n")
        elif converged >= total_configs * 0.8 and total_configs > 0:
            report.append("⚠️ **Most configurations converged, some need attention**\n")
        elif total_configs > 0:
            report.append("❌ **Convergence issues detected in multiple configurations**\n")
        else:
            report.append("❌ **No data available for analysis**\n")
        
        return "".join(report)


def main():
    """Main execution function."""
    print("="*70)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*70)
    
    # Initialize validator
    try:
        validator = CheckpointCrossValidator(data_dir=Path("data"), n_chains=5)
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install h5py:")
        print("  pip install h5py")
        return None
    
    # Run validation
    print("\nStarting cross-validation analysis...")
    validator.validate_all_configurations()
    
    # Save results
    print("\nSaving results...")
    validator.save_results()
    
    # Display summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    summary_df = validator.generate_summary_report()
    if not summary_df.empty:
        print("\nSummary Results:")
        print(summary_df.to_string(index=False))
    
    # Overall statistics
    total = len([r for r in validator.validation_results.values() 
               if r.get("status") != "no_data"])
    converged = sum(1 for r in validator.validation_results.values() 
                   if r.get("overall_converged", False))
    
    if total > 0:
        print(f"\nOverall Statistics:")
        print(f"  Configurations with data: {total}")
        print(f"  Fully converged: {converged}")
        print(f"  Convergence rate: {100*converged/total:.1f}%")
    else:
        print("\n❌ No data found for analysis")
        print("\nPlease ensure that final results exist in data/processed/")
        print("Expected files format: {statistic}_n{sample_size}_final.h5")
        print("Example: kolmogorov_smirnov_n30_final.h5")
    
    print(f"\nReports saved to: reports/cross_validation/")
    
    return validator.validation_results


if __name__ == "__main__":
    results = main()
