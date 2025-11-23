"""
Statistical validation module for Monte Carlo simulation results.
Performs theoretical comparison, Type I error validation, and cross-validation.
"""
import numpy as np
import pandas as pd
import h5py
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
from scipy import stats
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results."""
    mae: float
    rmse: float
    max_error: float
    relative_errors: np.ndarray
    sample_sizes: List[int]
    quantile_levels: List[float]


class TheoreticalComparison:
    """Compare empirical quantiles to theoretical values."""
    
    def __init__(self, results_path: Path = Path("data/processed")):
        """
        Initialize theoretical comparison validator.
        
        Args:
            results_path: Path to processed simulation results
        """
        self.results_path = results_path
        self.theoretical_values = self._load_theoretical_values()
    
    def _load_theoretical_values(self) -> Dict:
        """Load known theoretical values for comparison."""
        # Theoretical values from literature
        theoretical = {
            "kolmogorov_smirnov": {
                # Asymptotic critical values: c(α) / sqrt(n)
                # c(0.01)=1.63, c(0.05)=1.36, c(0.10)=1.22
                "asymptotic": {
                    0.90: 1.22,  # α=0.10
                    0.95: 1.36,  # α=0.05
                    0.99: 1.63   # α=0.01
                },
                "finite_sample": {
                    # Published values for specific sample sizes (if available)
                    30: {0.95: 0.242},
                    50: {0.95: 0.188},
                    100: {0.95: 0.134}
                }
            },
            "durbin_watson": {
                # DW theoretical center for no autocorrelation
                "center": 2.0,
                "bounds": (0.0, 4.0),
                # Critical values depend on number of regressors (k) and sample size
                # Using k=0 (no regressors) for pure autocorrelation test
                "critical_values": {
                    30: {"dL_05": 1.35, "dU_05": 1.49},
                    50: {"dL_05": 1.50, "dU_05": 1.59},
                    100: {"dL_05": 1.65, "dU_05": 1.69}
                }
            },
            "anderson_darling": {
                # Critical values from Stephens (1974) for normal distribution
                # These are for the modified statistic AD*
                "critical_values": {
                    0.90: 0.631,  # α=0.10
                    0.95: 0.752,  # α=0.05
                    0.99: 1.035   # α=0.01
                },
                # Finite sample adjustment: AD* = AD * (1 + 0.75/n + 2.25/n²)
                "adjustment_formula": lambda ad, n: ad * (1 + 0.75/n + 2.25/(n**2))
            }
        }
        return theoretical
    
    def compare_ks_theoretical(self, empirical_quantiles: Dict, 
                              sample_sizes: List[int]) -> ValidationResult:
        """
        Compare KS empirical quantiles to asymptotic formula.
        
        Args:
            empirical_quantiles: Dict with structure {sample_size: {quantile: value}}
            sample_sizes: List of sample sizes to validate
            
        Returns:
            ValidationResult with comparison metrics
        """
        mae_values = []
        squared_errors = []
        relative_errors = []
        
        for n in sample_sizes:
            if n not in empirical_quantiles:
                logger.warning(f"No empirical data for n={n}")
                continue
                
            for q_level in [0.90, 0.95, 0.99]:
                if q_level not in empirical_quantiles[n]:
                    continue
                    
                # Calculate theoretical value using asymptotic formula
                c_alpha = self.theoretical_values["kolmogorov_smirnov"]["asymptotic"].get(q_level)
                if c_alpha:
                    theoretical = c_alpha / np.sqrt(n)
                    empirical = empirical_quantiles[n][q_level]
                    
                    error = abs(empirical - theoretical)
                    mae_values.append(error)
                    squared_errors.append(error ** 2)
                    relative_errors.append(error / theoretical)
                    
                    logger.info(f"KS n={n}, q={q_level}: empirical={empirical:.4f}, "
                              f"theoretical={theoretical:.4f}, error={error:.4f}")
        
        mae = np.mean(mae_values) if mae_values else np.nan
        rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else np.nan
        max_error = np.max(mae_values) if mae_values else np.nan
        
        return ValidationResult(
            mae=mae,
            rmse=rmse,
            max_error=max_error,
            relative_errors=np.array(relative_errors),
            sample_sizes=sample_sizes,
            quantile_levels=[0.90, 0.95, 0.99]
        )
    
    def compare_dw_theoretical(self, empirical_stats: Dict) -> ValidationResult:
        """
        Compare DW empirical distribution to theoretical properties.
        
        Args:
            empirical_stats: Dict with distribution statistics
            
        Returns:
            ValidationResult with comparison metrics
        """
        theoretical_center = self.theoretical_values["durbin_watson"]["center"]
        bounds = self.theoretical_values["durbin_watson"]["bounds"]
        
        errors = []
        for n, stats in empirical_stats.items():
            mean = stats.get("mean", np.nan)
            if not np.isnan(mean):
                error = abs(mean - theoretical_center)
                errors.append(error)
                
                # Check if values are within bounds
                min_val = stats.get("min", 0)
                max_val = stats.get("max", 4)
                
                if min_val < bounds[0] or max_val > bounds[1]:
                    logger.warning(f"DW values outside theoretical bounds for n={n}")
                
                logger.info(f"DW n={n}: mean={mean:.4f}, theoretical={theoretical_center}, "
                          f"error={error:.4f}")
        
        mae = np.mean(errors) if errors else np.nan
        rmse = np.sqrt(np.mean([e**2 for e in errors])) if errors else np.nan
        
        return ValidationResult(
            mae=mae,
            rmse=rmse,
            max_error=np.max(errors) if errors else np.nan,
            relative_errors=np.array([e/theoretical_center for e in errors]),
            sample_sizes=list(empirical_stats.keys()),
            quantile_levels=[]
        )
    
    def compare_ad_theoretical(self, empirical_quantiles: Dict,
                              sample_sizes: List[int]) -> ValidationResult:
        """
        Compare AD empirical quantiles to Stephens critical values.
        
        Args:
            empirical_quantiles: Dict with structure {sample_size: {quantile: value}}
            sample_sizes: List of sample sizes to validate
            
        Returns:
            ValidationResult with comparison metrics
        """
        mae_values = []
        squared_errors = []
        relative_errors = []
        adjustment = self.theoretical_values["anderson_darling"]["adjustment_formula"]
        
        for n in sample_sizes:
            if n not in empirical_quantiles:
                continue
                
            for q_level in [0.90, 0.95, 0.99]:
                if q_level not in empirical_quantiles[n]:
                    continue
                    
                # Get theoretical critical value
                theoretical = self.theoretical_values["anderson_darling"]["critical_values"].get(q_level)
                if theoretical:
                    # Apply finite sample adjustment to empirical value
                    empirical_raw = empirical_quantiles[n][q_level]
                    empirical_adjusted = adjustment(empirical_raw, n)
                    
                    error = abs(empirical_adjusted - theoretical)
                    mae_values.append(error)
                    squared_errors.append(error ** 2)
                    relative_errors.append(error / theoretical)
                    
                    logger.info(f"AD n={n}, q={q_level}: empirical_adj={empirical_adjusted:.4f}, "
                              f"theoretical={theoretical:.4f}, error={error:.4f}")
        
        mae = np.mean(mae_values) if mae_values else np.nan
        rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else np.nan
        
        return ValidationResult(
            mae=mae,
            rmse=rmse,
            max_error=np.max(mae_values) if mae_values else np.nan,
            relative_errors=np.array(relative_errors),
            sample_sizes=sample_sizes,
            quantile_levels=[0.90, 0.95, 0.99]
        )


class TypeIErrorValidator:
    """Validate Type I error rates for test statistics."""
    
    def __init__(self, statistics_modules: Dict):
        """
        Initialize Type I error validator.
        
        Args:
            statistics_modules: Dict mapping statistic names to compute functions
        """
        self.statistics_modules = statistics_modules
        
    def generate_null_samples(self, n_samples: int, sample_size: int,
                            seed: int = 42) -> np.ndarray:
        """
        Generate samples under null hypothesis (standard normal).
        
        Args:
            n_samples: Number of samples to generate
            sample_size: Size of each sample
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_samples, sample_size)
        """
        rng = np.random.RandomState(seed)
        return rng.standard_normal((n_samples, sample_size))
    
    def calculate_rejection_rates(self, test_statistics: np.ndarray,
                                 critical_values: Dict[float, float]) -> Dict:
        """
        Calculate empirical rejection rates at each significance level.
        
        Args:
            test_statistics: Array of test statistics from null samples
            critical_values: Dict mapping quantile levels to critical values
            
        Returns:
            Dict with nominal vs empirical rates and confidence intervals
        """
        n = len(test_statistics)
        results = {}
        
        for quantile_level, critical_value in critical_values.items():
            # Count rejections (test stat > critical value)
            n_rejections = np.sum(test_statistics > critical_value)
            empirical_rate = n_rejections / n
            
            # Calculate confidence interval using Wilson score method
            ci_lower, ci_upper = self._wilson_score_interval(n_rejections, n)
            
            # Nominal rate is 1 - quantile_level
            nominal_rate = 1 - quantile_level
            
            results[quantile_level] = {
                "nominal": nominal_rate,
                "empirical": empirical_rate,
                "n_rejections": n_rejections,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "within_tolerance": abs(empirical_rate - nominal_rate) <= 0.005
            }
            
            logger.info(f"Quantile {quantile_level}: nominal={nominal_rate:.3f}, "
                       f"empirical={empirical_rate:.3f} "
                       f"[{ci_lower:.3f}, {ci_upper:.3f}]")
        
        return results
    
    def _wilson_score_interval(self, successes: int, n: int,
                              confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval for a proportion.
        
        Args:
            successes: Number of successes (rejections)
            n: Total number of trials
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if n == 0:
            return (0, 1)
            
        z = stats.norm.ppf((1 + confidence) / 2)
        p_hat = successes / n
        
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
    
    def validate_type_i_error(self, statistic_name: str, sample_size: int,
                            empirical_quantiles: Dict[float, float],
                            n_validation_samples: int = 10000) -> Dict:
        """
        Complete Type I error validation for a test statistic.
        
        Args:
            statistic_name: Name of test statistic
            sample_size: Sample size for validation
            empirical_quantiles: Critical values from simulation
            n_validation_samples: Number of validation samples
            
        Returns:
            Dict with validation results
        """
        logger.info(f"Starting Type I error validation for {statistic_name}, n={sample_size}")
        
        # Generate null samples
        null_samples = self.generate_null_samples(n_validation_samples, sample_size)
        
        # Calculate test statistics
        compute_stat = self.statistics_modules.get(statistic_name)
        if not compute_stat:
            raise ValueError(f"Unknown statistic: {statistic_name}")
            
        test_statistics = []
        for i in range(n_validation_samples):
            stat = compute_stat(null_samples[i])
            test_statistics.append(stat)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Computed {i+1}/{n_validation_samples} test statistics")
        
        test_statistics = np.array(test_statistics)
        
        # Calculate rejection rates
        results = self.calculate_rejection_rates(test_statistics, empirical_quantiles)
        
        return {
            "statistic": statistic_name,
            "sample_size": sample_size,
            "n_samples": n_validation_samples,
            "rejection_rates": results,
            "test_stat_summary": {
                "mean": np.mean(test_statistics),
                "std": np.std(test_statistics),
                "min": np.min(test_statistics),
                "max": np.max(test_statistics)
            }
        }


class CrossValidator:
    """Perform cross-validation using Gelman-Rubin diagnostic."""
    
    def split_chains(self, simulated_values: np.ndarray,
                     n_chains: int = 5) -> List[np.ndarray]:
        """
        Split simulation into independent chains.
        
        Args:
            simulated_values: Full simulation results
            n_chains: Number of chains to create
            
        Returns:
            List of arrays, one per chain
        """
        n = len(simulated_values)
        chain_length = n // n_chains
        
        chains = []
        for i in range(n_chains):
            start_idx = i * chain_length
            end_idx = start_idx + chain_length if i < n_chains - 1 else n
            chains.append(simulated_values[start_idx:end_idx])
            
        return chains
    
    def gelman_rubin_statistic(self, chains: List[np.ndarray],
                              quantile: float = 0.95) -> float:
        """
        Calculate Gelman-Rubin R-hat statistic for convergence.
        
        Args:
            chains: List of chains
            quantile: Quantile level to assess
            
        Returns:
            R-hat statistic (should be < 1.1 for convergence)
        """
        n_chains = len(chains)
        
        # Calculate quantile for each chain
        chain_quantiles = [np.quantile(chain, quantile) for chain in chains]
        
        # Calculate between-chain variance B
        chain_means = np.array(chain_quantiles)
        overall_mean = np.mean(chain_means)
        B = np.sum((chain_means - overall_mean)**2) / (n_chains - 1)
        
        # Calculate within-chain variance W
        # For quantiles, we use batch means within each chain
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
                W_values.append(np.var(batch_quantiles))
        
        W = np.mean(W_values) if W_values else 0
        
        # Calculate R-hat
        if W > 0:
            var_plus = ((n_chains - 1) / n_chains) * W + B
            r_hat = np.sqrt(var_plus / W)
        else:
            r_hat = 1.0
            
        return r_hat
    
    def coefficient_of_variation(self, values: np.ndarray) -> float:
        """
        Calculate coefficient of variation.
        
        Args:
            values: Array of values (e.g., quantile estimates)
            
        Returns:
            CV = std / mean
        """
        mean = np.mean(values)
        if mean == 0:
            return np.inf
        return np.std(values) / abs(mean)
    
    def validate_convergence(self, simulated_values: np.ndarray,
                           quantiles: List[float] = [0.75, 0.90, 0.95, 0.99],
                           n_chains: int = 5) -> Dict:
        """
        Complete cross-validation analysis.
        
        Args:
            simulated_values: Full simulation results
            quantiles: Quantile levels to validate
            n_chains: Number of chains for splitting
            
        Returns:
            Dict with convergence diagnostics
        """
        chains = self.split_chains(simulated_values, n_chains)
        
        results = {
            "n_chains": n_chains,
            "chain_lengths": [len(chain) for chain in chains],
            "quantile_diagnostics": {}
        }
        
        for q in quantiles:
            # Calculate quantile for each chain
            chain_quantiles = [np.quantile(chain, q) for chain in chains]
            
            # Calculate R-hat
            r_hat = self.gelman_rubin_statistic(chains, q)
            
            # Calculate CV
            cv = self.coefficient_of_variation(np.array(chain_quantiles))
            
            # Determine convergence status
            converged = r_hat < 1.1 and cv < 0.01
            
            results["quantile_diagnostics"][q] = {
                "chain_estimates": chain_quantiles,
                "mean": np.mean(chain_quantiles),
                "std": np.std(chain_quantiles),
                "r_hat": r_hat,
                "cv": cv,
                "converged": converged
            }
            
            logger.info(f"Quantile {q}: R-hat={r_hat:.4f}, CV={cv:.4f}, "
                       f"Converged={converged}")
        
        return results


class ValidationReportGenerator:
    """Generate validation reports in multiple formats."""
    
    def __init__(self, output_dir: Path = Path("reports")):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for report output
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_summary_table(self, validation_results: Dict) -> pd.DataFrame:
        """
        Generate summary validation table.
        
        Args:
            validation_results: Dict with all validation results
            
        Returns:
            DataFrame with formatted summary
        """
        summary_data = []
        
        for stat_name, results in validation_results.items():
            row = {
                "Statistic": stat_name.replace("_", " ").title(),
                "MAE": results.get("mae", np.nan),
                "RMSE": results.get("rmse", np.nan),
                "Max Error": results.get("max_error", np.nan),
                "Status": "✓" if results.get("mae", 1) < 0.002 else "⚠"
            }
            summary_data.append(row)
            
        df = pd.DataFrame(summary_data)
        df = df.round(4)
        return df
    
    def format_type_i_error_table(self, type_i_results: Dict) -> pd.DataFrame:
        """
        Format Type I error validation results.
        
        Args:
            type_i_results: Dict with Type I error validation data
            
        Returns:
            Formatted DataFrame
        """
        data = []
        
        for quantile, rates in type_i_results.items():
            row = {
                "Quantile": f"{quantile:.0%}",
                "Nominal α": f"{rates['nominal']:.3f}",
                "Empirical": f"{rates['empirical']:.3f}",
                "95% CI": f"[{rates['ci_lower']:.3f}, {rates['ci_upper']:.3f}]",
                "Status": "✓" if rates['within_tolerance'] else "✗"
            }
            data.append(row)
            
        return pd.DataFrame(data)
    
    def generate_markdown_report(self, all_results: Dict) -> str:
        """
        Generate complete markdown validation report.
        
        Args:
            all_results: Complete validation results
            
        Returns:
            Markdown formatted report string
        """
        report = []
        report.append("# Statistical Validation Report\n")
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        report.append("Monte Carlo simulations validated against theoretical values ")
        report.append("and Type I error rates.\n\n")
        
        # Theoretical Comparison
        report.append("## 1. Theoretical Value Comparison\n")
        report.append("### Summary Statistics\n")
        
        if "theoretical_comparison" in all_results:
            df = self.generate_summary_table(all_results["theoretical_comparison"])
            report.append(df.to_markdown(index=False))
            report.append("\n\n")
            
            # Add interpretation
            report.append("**Interpretation:**\n")
            report.append("- Target MAE < 0.002 for all statistics\n")
            report.append("- ✓ indicates passing validation\n")
            report.append("- ⚠ indicates further investigation needed\n\n")
        
        # Type I Error Validation
        report.append("## 2. Type I Error Rate Validation\n")
        
        if "type_i_error" in all_results:
            for stat_name, results in all_results["type_i_error"].items():
                report.append(f"### {stat_name.replace('_', ' ').title()}\n")
                
                if "rejection_rates" in results:
                    df = self.format_type_i_error_table(results["rejection_rates"])
                    report.append(df.to_markdown(index=False))
                    report.append("\n\n")
        
        # Convergence Diagnostics
        report.append("## 3. Convergence Diagnostics\n")
        
        if "cross_validation" in all_results:
            report.append("### Gelman-Rubin Statistics\n")
            report.append("| Statistic | Sample Size | Quantile | R-hat | CV | Status |\n")
            report.append("|-----------|------------|----------|-------|-----|--------|\n")
            
            for config, results in all_results["cross_validation"].items():
                if "quantile_diagnostics" in results:
                    for q, diag in results["quantile_diagnostics"].items():
                        stat_name = config.split("_")[0]
                        n = config.split("_")[1]
                        status = "✓" if diag["converged"] else "✗"
                        report.append(f"| {stat_name} | {n} | {q:.0%} | "
                                    f"{diag['r_hat']:.3f} | {diag['cv']:.3f} | {status} |\n")
            report.append("\n")
        
        # Recommendations
        report.append("## 4. Recommendations\n")
        report.append("Based on validation results:\n\n")
        
        # Check for issues
        issues = []
        if "theoretical_comparison" in all_results:
            for stat, results in all_results["theoretical_comparison"].items():
                if results.get("mae", 1) > 0.002:
                    issues.append(f"- {stat}: MAE exceeds target threshold")
        
        if issues:
            report.append("### Issues Requiring Attention:\n")
            for issue in issues:
                report.append(f"{issue}\n")
        else:
            report.append("- All validations passed successfully\n")
            report.append("- Empirical quantiles suitable for production use\n")
        
        return "\n".join(report)
    
    def save_report(self, report_content: str, filename: str = "validation_report.md"):
        """Save report to file."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write(report_content)
        logger.info(f"Report saved to {output_path}")
        
    def export_tables(self, tables: Dict[str, pd.DataFrame]):
        """Export tables to CSV and LaTeX formats."""
        for name, df in tables.items():
            # CSV export
            csv_path = self.output_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            
            # LaTeX export
            latex_path = self.output_dir / f"{name}.tex"
            with open(latex_path, "w") as f:
                f.write(df.to_latex(index=False))
                
            logger.info(f"Exported {name} to CSV and LaTeX")


# Utility functions for loading simulation data
def load_simulation_results(filepath: Path) -> Dict:
    """Load simulation results from HDF5 file."""
    results = {}
    with h5py.File(filepath, 'r') as f:
        if 'simulated_values' in f:
            results['values'] = f['simulated_values'][:]
        if 'quantiles' in f:
            results['quantiles'] = {
                float(k): v for k, v in f['quantiles'].items()
            }
    return results


def load_all_results(results_dir: Path = Path("data/processed")) -> Dict:
    """Load all simulation results from directory."""
    all_results = {}
    
    for h5_file in results_dir.glob("*.h5"):
        if "_metadata" not in h5_file.name:
            # Parse filename: statistic_samplesize_results.h5
            parts = h5_file.stem.split("_")
            if len(parts) >= 3:
                stat_name = "_".join(parts[:-2])
                sample_size = int(parts[-2]) if parts[-2].isdigit() else parts[-2]
                
                if stat_name not in all_results:
                    all_results[stat_name] = {}
                    
                all_results[stat_name][sample_size] = load_simulation_results(h5_file)
                logger.info(f"Loaded {stat_name} n={sample_size}")
    
    return all_results


if __name__ == "__main__":
    # Example usage
    logger.info("Starting validation pipeline...")
    
    # Load simulation results
    results = load_all_results()
    
    # Run theoretical comparison
    comparator = TheoreticalComparison()
    
    # Generate report
    reporter = ValidationReportGenerator()
    
    logger.info("Validation complete")
