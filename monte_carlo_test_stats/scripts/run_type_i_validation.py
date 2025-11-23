"""
Type I Error Validation - Standalone Production Script
Complete validation with proper sample size (10000) and fixed JSON serialization
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import h5py
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import logging

# Import validation module and test statistics
from src.analysis.validation import TypeIErrorValidator
from src.statistics.kolmogorov_smirnov import kolmogorov_smirnov_statistic
from src.statistics.durbin_watson import durbin_watson_statistic
from src.statistics.anderson_darling import anderson_darling_statistic

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'type_i_validation_production_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TypeIErrorValidationPipeline:
    """Pipeline for complete Type I error validation across all configurations."""
    
    def __init__(self, results_dir: Path = Path("data/processed"),
                 n_validation_samples: int = 10000):
        """
        Initialize validation pipeline.
        
        Args:
            results_dir: Directory containing simulation results
            n_validation_samples: Number of samples for validation
        """
        self.results_dir = results_dir
        self.n_validation_samples = n_validation_samples
        
        # Map statistic names to compute functions
        self.statistics_functions = {
            "kolmogorov_smirnov": kolmogorov_smirnov_statistic,
            "durbin_watson": durbin_watson_statistic,
            "anderson_darling": anderson_darling_statistic
        }
        
        # Initialize validator
        self.validator = TypeIErrorValidator(self.statistics_functions)
        
        # Results storage
        self.validation_results = {}
        
    def load_empirical_quantiles(self, statistic_name: str, 
                                sample_size: int) -> Dict[float, float]:
        """
        Load empirical quantiles from simulation results.
        
        Args:
            statistic_name: Name of test statistic
            sample_size: Sample size
            
        Returns:
            Dict mapping quantile levels to critical values
        """
        # Try different filename patterns
        patterns = [
            f"{statistic_name}_{sample_size}_results.h5",
            f"{statistic_name}_n{sample_size}_final.h5",
            f"{statistic_name}_{sample_size}_final.h5"
        ]
        
        quantiles = {}
        file_found = False
        
        for pattern in patterns:
            filepath = self.results_dir / pattern
            if filepath.exists():
                logger.info(f"Loading quantiles from {filepath}")
                try:
                    with h5py.File(filepath, 'r') as f:
                        # Check for different data structures
                        if 'quantiles' in f:
                            # New format with pre-calculated quantiles
                            quantiles_group = f['quantiles']
                            for q_str in quantiles_group.keys():
                                q_float = float(q_str)
                                value = quantiles_group[q_str]
                                if hasattr(value, 'shape') and value.shape == ():
                                    # Scalar dataset
                                    quantiles[q_float] = float(value[()])
                                else:
                                    # Direct value
                                    quantiles[q_float] = float(value)
                            file_found = True
                            break
                        elif 'simulated_values' in f:
                            # Old format - calculate quantiles from raw data
                            values = f['simulated_values'][:]
                            for q in [0.75, 0.90, 0.95, 0.99]:
                                quantiles[q] = float(np.quantile(values, q))
                            file_found = True
                            break
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")
                    continue
        
        if not file_found:
            logger.warning(f"No simulation results found for {statistic_name} n={sample_size}")
            # Try to load from metadata JSON as fallback
            metadata_path = self.results_dir / f"{statistic_name}_{sample_size}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if 'quantiles' in metadata:
                        quantiles = {float(k): float(v) for k, v in metadata['quantiles'].items()}
                        logger.info(f"Loaded quantiles from metadata for {statistic_name} n={sample_size}")
        
        if not quantiles:
            logger.error(f"Could not load quantiles for {statistic_name} n={sample_size}")
        else:
            logger.info(f"Loaded quantiles: {quantiles}")
            
        return quantiles
    
    def validate_single_configuration(self, statistic_name: str,
                                     sample_size: int) -> Dict:
        """
        Run Type I error validation for a single configuration.
        
        Args:
            statistic_name: Name of test statistic
            sample_size: Sample size
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Validating {statistic_name} with n={sample_size}")
        logger.info(f"{'='*60}")
        
        # Load empirical quantiles
        empirical_quantiles = self.load_empirical_quantiles(statistic_name, sample_size)
        
        if not empirical_quantiles:
            logger.error(f"Skipping validation for {statistic_name} n={sample_size} - no quantiles")
            return {
                "status": "failed",
                "reason": "No empirical quantiles available"
            }
        
        # Run validation
        try:
            results = self.validator.validate_type_i_error(
                statistic_name=statistic_name,
                sample_size=sample_size,
                empirical_quantiles=empirical_quantiles,
                n_validation_samples=self.n_validation_samples
            )
            
            # Add summary statistics
            results["summary"] = self._summarize_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Validation failed for {statistic_name} n={sample_size}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _summarize_results(self, results: Dict) -> Dict:
        """
        Generate summary statistics for validation results.
        
        Args:
            results: Raw validation results
            
        Returns:
            Summary dictionary
        """
        summary = {
            "all_within_tolerance": True,
            "max_deviation": 0,
            "mean_absolute_deviation": 0
        }
        
        if "rejection_rates" not in results:
            return summary
        
        deviations = []
        for q_level, rates in results["rejection_rates"].items():
            deviation = abs(rates["empirical"] - rates["nominal"])
            deviations.append(deviation)
            
            if not rates["within_tolerance"]:
                summary["all_within_tolerance"] = False
        
        summary["max_deviation"] = max(deviations) if deviations else 0
        summary["mean_absolute_deviation"] = np.mean(deviations) if deviations else 0
        
        return summary
    
    def validate_all_configurations(self):
        """Run validation for all available configurations (sequential for Windows)."""
        configurations = [
            ("kolmogorov_smirnov", 30),
            ("kolmogorov_smirnov", 50),
            ("kolmogorov_smirnov", 100),
            ("kolmogorov_smirnov", 500),
            ("kolmogorov_smirnov", 1000),
            ("durbin_watson", 30),
            ("durbin_watson", 50),
            ("durbin_watson", 100),
            ("durbin_watson", 500),
            ("durbin_watson", 1000),
            ("anderson_darling", 30),
            ("anderson_darling", 50),
            ("anderson_darling", 100),
            ("anderson_darling", 500),
            ("anderson_darling", 1000),
        ]
        
        logger.info(f"Starting Type I error validation for {len(configurations)} configurations")
        logger.info(f"Using {self.n_validation_samples} validation samples per configuration")
        logger.info("Running in sequential mode (Windows-compatible)")
        
        for i, (stat_name, sample_size) in enumerate(configurations, 1):
            logger.info(f"\nProgress: {i}/{len(configurations)}")
            key = f"{stat_name}_{sample_size}"
            self.validation_results[key] = self.validate_single_configuration(
                stat_name, sample_size
            )
            
            # Log summary
            if "summary" in self.validation_results[key]:
                summary = self.validation_results[key]["summary"]
                status = "PASS" if summary["all_within_tolerance"] else "[FAIL]"
                logger.info(f"{key}: {status} (max deviation: {summary['max_deviation']:.4f})")


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    Handles the bool serialization issue.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):  # Handle numpy bool explicitly
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int_, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    return obj


def save_results_safely(validation_results, output_dir: Path = Path("reports/type_i_validation")):
    """
    Save validation results with proper error handling.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert and save JSON
    try:
        json_path = output_dir / f"type_i_validation_results_{timestamp}.json"
        serializable_results = convert_numpy_types(validation_results)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Raw results saved to {json_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        # Save as pickle as backup
        import pickle
        pickle_path = output_dir / f"type_i_validation_results_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(validation_results, f)
        logger.info(f"Results saved as pickle to {pickle_path}")
    
    # Generate summary DataFrame
    summary_data = []
    for config_key, results in validation_results.items():
        if results.get("status") in ["failed", "error"]:
            continue
            
        parts = config_key.rsplit("_", 1)
        stat_name = parts[0]
        sample_size = parts[1]
        
        row = {
            "Statistic": stat_name.replace("_", " ").title(),
            "Sample Size": sample_size,
        }
        
        if "rejection_rates" in results:
            for q_level, rates in results["rejection_rates"].items():
                alpha = 1 - q_level
                row[f"α={alpha:.2f}"] = f"{rates['empirical']:.3f}"
                row[f"α={alpha:.2f} Pass"] = "[OK]" if rates["within_tolerance"] else "[FAIL]"
        
        if "summary" in results:
            row["Overall"] = "[OK]" if results["summary"]["all_within_tolerance"] else "[FAIL]"
            row["Max Dev"] = f"{results['summary']['max_deviation']:.4f}"
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary CSV
    csv_path = output_dir / f"type_i_validation_summary_{timestamp}.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Summary saved to {csv_path}")
    
    # Generate markdown report
    generate_detailed_report(validation_results, output_dir, timestamp)
    
    return summary_df


def generate_detailed_report(validation_results, output_dir: Path, timestamp: str):
    """Generate detailed markdown report with interpretation."""
    report = []
    report.append("# Type I Error Validation Report (Production Run)\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"Validation samples per configuration: 10,000\n")
    report.append("Tolerance: ±0.005 from nominal rate\n\n")
    
    # Overall summary
    report.append("## Executive Summary\n")
    
    total_configs = len(validation_results)
    passed = sum(1 for r in validation_results.values() 
                if r.get("summary", {}).get("all_within_tolerance", False))
    
    report.append(f"- **Total configurations tested:** {total_configs}\n")
    report.append(f"- **Configurations passing:** {passed}/{total_configs}\n")
    report.append(f"- **Pass rate:** {100*passed/total_configs:.1f}%\n\n")
    
    if passed == total_configs:
        report.append("**Result: All configurations maintain proper Type I error rates.**\n\n")
    elif passed >= total_configs * 0.8:
        report.append("**Result: Most configurations pass, minor adjustments may be needed.**\n\n")
    else:
        report.append("**Result: Significant deviations detected, review required.**\n\n")
    
    # Detailed results by statistic
    for stat_name in ["kolmogorov_smirnov", "durbin_watson", "anderson_darling"]:
        report.append(f"## {stat_name.replace('_', ' ').title()}\n\n")
        
        # Create summary table
        report.append("| n | α=0.25 | α=0.10 | α=0.05 | α=0.01 | Overall |\n")
        report.append("|---|--------|--------|--------|--------|----------|\n")
        
        for sample_size in [30, 50, 100, 500, 1000]:
            key = f"{stat_name}_{sample_size}"
            if key not in validation_results:
                continue
                
            results = validation_results[key]
            if results.get("status") in ["failed", "error"]:
                report.append(f"| {sample_size} | ERROR | ERROR | ERROR | ERROR | [FAIL] |\n")
                continue
            
            row = [f"| {sample_size} "]
            
            if "rejection_rates" in results:
                for q_level in [0.75, 0.90, 0.95, 0.99]:
                    if q_level in results["rejection_rates"]:
                        rates = results["rejection_rates"][q_level]
                        emp = rates["empirical"]
                        nom = rates["nominal"]
                        diff = emp - nom
                        symbol = "[OK]" if rates["within_tolerance"] else "[FAIL]"
                        # Show difference for clarity
                        row.append(f"| {emp:.3f} ({diff:+.3f}) {symbol} ")
                    else:
                        row.append("| - ")
            
            status = "[OK]" if results.get("summary", {}).get("all_within_tolerance", False) else "[FAIL]"
            row.append(f"| {status} |\n")
            report.append("".join(row))
        
        report.append("\n")
    
    # Interpretation section
    report.append("## Interpretation\n\n")
    report.append("### Reading the Results:\n")
    report.append("- Each cell shows: `empirical (difference) status`\n")
    report.append("- [OK] = within ±0.005 tolerance\n")
    report.append("- [FAIL] = outside tolerance\n")
    report.append("- Difference = empirical - nominal rate\n\n")
    
    # Pattern analysis
    report.append("### Observed Patterns:\n\n")
    
    # Analyze by sample size
    small_n_pass = 0
    large_n_pass = 0
    small_n_total = 0
    large_n_total = 0
    
    for key, results in validation_results.items():
        if results.get("status") in ["failed", "error"]:
            continue
        
        parts = key.rsplit("_", 1)
        sample_size = int(parts[1])
        
        if sample_size <= 50:
            small_n_total += 1
            if results.get("summary", {}).get("all_within_tolerance", False):
                small_n_pass += 1
        else:
            large_n_total += 1
            if results.get("summary", {}).get("all_within_tolerance", False):
                large_n_pass += 1
    
    if small_n_total > 0:
        report.append(f"- **Small samples (n≤50):** {small_n_pass}/{small_n_total} passed "
                     f"({100*small_n_pass/small_n_total:.0f}%)\n")
    if large_n_total > 0:
        report.append(f"- **Large samples (n>50):** {large_n_pass}/{large_n_total} passed "
                     f"({100*large_n_pass/large_n_total:.0f}%)\n")
    
    # Recommendations
    report.append("\n### Recommendations:\n\n")
    
    if passed == total_configs:
        report.append("1. Empirical critical values are validated and ready for use\n")
        report.append("2. Type I error rates are well-controlled across all configurations\n")
        report.append("3. No adjustments needed to the simulation results\n")
    elif passed >= total_configs * 0.8:
        report.append("1. Most configurations are within acceptable tolerances\n")
        report.append("2. Consider re-running simulations for failed configurations with more iterations\n")
        report.append("3. Small deviations may be due to Monte Carlo variability\n")
    else:
        report.append("1. Review simulation methodology for potential issues\n")
        report.append("2. Consider increasing simulation iterations (currently 10M)\n")
        report.append("3. Verify test statistic implementations against reference software\n")
    
    # Technical notes
    report.append("\n### Technical Notes:\n\n")
    report.append("- Validation uses 10,000 independent samples under null hypothesis\n")
    report.append("- Confidence intervals calculated using Wilson score method\n")
    report.append("- Tolerance of ±0.005 is standard for Monte Carlo validation\n")
    report.append("- Some variation is expected due to finite sample effects\n")
    
    # Save report
    report_path = output_dir / f"type_i_validation_report_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("".join(report))
    logger.info(f"Detailed report saved to {report_path}")


def main():
    """Main execution function for production run."""
    logger.info("="*70)
    logger.info("TYPE I ERROR VALIDATION - PRODUCTION RUN")
    logger.info("="*70)
    
    # Use full 10000 samples for production validation
    n_validation_samples = 10000
    
    logger.info(f"Running production validation with {n_validation_samples} samples")
    logger.info("This will take approximately 2-3 minutes to complete")
    
    # Initialize pipeline
    pipeline = TypeIErrorValidationPipeline(
        results_dir=Path("data/processed"),
        n_validation_samples=n_validation_samples
    )
    
    # Run validation
    logger.info("Starting validation of all 15 configurations...")
    pipeline.validate_all_configurations()
    
    # Save results with fixed JSON serialization
    logger.info("\nSaving results...")
    summary_df = save_results_safely(pipeline.validation_results)
    
    # Display summary
    logger.info("\n" + "="*70)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*70)
    
    print("\nType I Error Validation Summary:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    
    # Overall assessment
    total = len(pipeline.validation_results)
    passed = sum(1 for r in pipeline.validation_results.values() 
                if r.get("summary", {}).get("all_within_tolerance", False))
    
    print(f"\nOverall Results:")
    print(f"  Configurations Tested: {total}")
    print(f"  Configurations Passed: {passed}")
    print(f"  Pass Rate: {100*passed/total:.1f}%")
    
    if passed == total:
        print("\n SUCCESS: All configurations maintain proper Type I error rates!")
        print("The empirical critical values are validated for use.")
    elif passed >= total * 0.8:
        print("\nMOSTLY PASSED: Most configurations are within tolerance.")
        print("Minor adjustments may improve the remaining configurations.")
    else:
        print("\n ATTENTION NEEDED: Several configurations show deviations.")
        print("Review the detailed report for specific issues.")
    
    print(f"\nReports saved to: reports/type_i_validation/")
    print("Check the markdown report for detailed analysis and recommendations.")
    
    return pipeline.validation_results


if __name__ == "__main__":
    results = main()
