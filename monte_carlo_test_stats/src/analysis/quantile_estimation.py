"""
Quantile Extraction and Table Generation Module

This module processes Monte Carlo simulation results to:
1. Extract quantiles from HDF5 files
2. Calculate standard errors using batch means
3. Generate bootstrap confidence intervals
4. Create formatted tables in CSV, LaTeX, and Markdown
"""

import numpy as np
import pandas as pd
import h5py
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantileExtractor:
    """Extracts quantiles from Monte Carlo simulation results."""
    
    def __init__(self, 
                 data_dir: str = "data/processed",
                 output_dir: str = "reports/quantile_tables",
                 quantile_levels: Optional[List[float]] = None):
        """
        Initialize the quantile extractor.
        
        Args:
            data_dir: Directory containing HDF5 simulation results
            output_dir: Directory for output tables
            quantile_levels: Quantile levels to extract (default: [0.75, 0.90, 0.95, 0.99])
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.quantile_levels = quantile_levels or [0.75, 0.90, 0.95, 0.99]
        
        # Create output directories
        for subdir in ['csv', 'latex', 'markdown']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Statistics and sample sizes
        self.statistics = ['kolmogorov_smirnov', 'durbin_watson', 'anderson_darling']
        self.sample_sizes = [30, 50, 100, 500, 1000]
        
        logger.info(f"Initialized QuantileExtractor with data_dir={data_dir}, output_dir={output_dir}")
    
    def extract_quantiles_from_h5(self, file_path: Path) -> Dict[float, float]:
        """
        Extract quantiles from an HDF5 file.
        
        Args:
            file_path: Path to HDF5 file
            
        Returns:
            Dictionary mapping quantile levels to values
        """
        if not file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            if 'data' not in f:
                raise KeyError(f"'data' dataset not found in {file_path}")
            
            data = f['data'][:]
            logger.info(f"Loaded {len(data):,} values from {file_path.name}")
        
        # Calculate quantiles
        quantiles = {}
        for level in self.quantile_levels:
            if level < 0 or level > 1:
                raise ValueError(f"Quantile level {level} must be between 0 and 1")
            quantiles[level] = np.quantile(data, level)
        
        return quantiles, data
    
    def calculate_batch_means_se(self, 
                                 data: np.ndarray, 
                                 n_batches: int = 100) -> Dict[float, float]:
        """
        Calculate standard errors using batch means method.
        
        Args:
            data: Simulation data
            n_batches: Number of batches
            
        Returns:
            Dictionary mapping quantile levels to standard errors
        """
        batch_size = len(data) // n_batches
        if batch_size < 100:
            warnings.warn(f"Small batch size ({batch_size}), SE estimates may be unstable")
        
        batch_quantiles = {level: [] for level in self.quantile_levels}
        
        # Calculate quantiles for each batch
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < n_batches - 1 else len(data)
            batch_data = data[start_idx:end_idx]
            
            for level in self.quantile_levels:
                q = np.quantile(batch_data, level)
                batch_quantiles[level].append(q)
        
        # Calculate standard errors
        standard_errors = {}
        for level in self.quantile_levels:
            batch_std = np.std(batch_quantiles[level], ddof=1)
            se = batch_std / np.sqrt(n_batches)
            standard_errors[level] = se
            
            logger.debug(f"SE for quantile {level}: {se:.6f}")
        
        return standard_errors
    
    def calculate_bootstrap_ci(self, 
                               data: np.ndarray, 
                               n_bootstrap: int = 1000,
                               ci_level: float = 0.95) -> Dict[float, Dict[str, float]]:
        """
        Calculate bootstrap confidence intervals for quantiles.
        
        Args:
            data: Simulation data
            n_bootstrap: Number of bootstrap iterations
            ci_level: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Dictionary with confidence intervals for each quantile
        """
        alpha = 1 - ci_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        bootstrap_quantiles = {level: [] for level in self.quantile_levels}
        
        # Perform bootstrap resampling
        np.random.seed(42)  # For reproducibility
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            for level in self.quantile_levels:
                q = np.quantile(bootstrap_sample, level)
                bootstrap_quantiles[level].append(q)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for level in self.quantile_levels:
            lower_ci = np.percentile(bootstrap_quantiles[level], lower_percentile)
            upper_ci = np.percentile(bootstrap_quantiles[level], upper_percentile)
            point_estimate = np.median(bootstrap_quantiles[level])
            
            confidence_intervals[level] = {
                'lower': lower_ci,
                'point': point_estimate,
                'upper': upper_ci,
                'width': upper_ci - lower_ci
            }
            
            logger.debug(f"CI for quantile {level}: [{lower_ci:.6f}, {upper_ci:.6f}]")
        
        return confidence_intervals
    
    def process_single_configuration(self, 
                                    statistic: str, 
                                    sample_size: int) -> Dict[str, Any]:
        """
        Process a single statistic/sample_size configuration.
        
        Args:
            statistic: Name of the test statistic
            sample_size: Sample size
            
        Returns:
            Dictionary with quantiles, SEs, and CIs
        """
        # Find the appropriate file
        file_patterns = [
            f"{statistic}_n{sample_size}_final.h5",
            f"{statistic}_{sample_size}_results.h5"
        ]
        
        file_path = None
        for pattern in file_patterns:
            candidate = self.data_dir / pattern
            if candidate.exists():
                file_path = candidate
                break
        
        if file_path is None:
            logger.warning(f"No data file found for {statistic} n={sample_size}")
            return None
        
        logger.info(f"Processing {statistic} with n={sample_size} from {file_path.name}")
        
        # Extract quantiles and get data
        quantiles, data = self.extract_quantiles_from_h5(file_path)
        
        # Calculate standard errors
        standard_errors = self.calculate_batch_means_se(data)
        
        # Calculate bootstrap confidence intervals
        confidence_intervals = self.calculate_bootstrap_ci(data)
        
        # Load metadata if available
        metadata_path = file_path.with_suffix('.json')
        if not metadata_path.exists():
            # Try alternative naming
            metadata_path = self.data_dir / f"{statistic}_{sample_size}_metadata.json"
        
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Validate against pre-calculated quantiles if available
            if 'quantiles' in metadata:
                for level_str, expected_val in metadata['quantiles'].items():
                    level = float(level_str)
                    if level in quantiles:
                        calc_val = quantiles[level]
                        diff = abs(calc_val - expected_val)
                        if diff > 0.0001:
                            logger.warning(
                                f"Quantile mismatch for {level}: "
                                f"calculated={calc_val:.6f}, metadata={expected_val:.6f}"
                            )
        
        return {
            'quantiles': quantiles,
            'standard_errors': standard_errors,
            'confidence_intervals': confidence_intervals,
            'metadata': metadata,
            'n_iterations': len(data)
        }
    
    def aggregate_all_configurations(self) -> Dict[str, Dict[int, Dict]]:
        """
        Process all statistic/sample_size configurations.
        
        Returns:
            Nested dictionary with results for all configurations
        """
        results = {}
        
        for statistic in self.statistics:
            results[statistic] = {}
            
            for sample_size in self.sample_sizes:
                config_results = self.process_single_configuration(statistic, sample_size)
                
                if config_results is not None:
                    results[statistic][sample_size] = config_results
                    logger.info(f" [OK] Processed {statistic} n={sample_size}")
                else:
                    logger.error(f"[FAIL] Failed to process {statistic} n={sample_size}")
        
        return results
    
    def format_quantile_table(self, 
                              statistic: str, 
                              results: Dict[int, Dict]) -> pd.DataFrame:
        """
        Format quantile results as a pandas DataFrame.
        
        Args:
            statistic: Name of the test statistic
            results: Results dictionary for the statistic
            
        Returns:
            DataFrame with formatted quantiles
        """
        data = []
        
        for sample_size in self.sample_sizes:
            if sample_size not in results:
                continue
            
            row = {'sample_size': sample_size}
            quantiles = results[sample_size]['quantiles']
            
            for level in self.quantile_levels:
                col_name = f'q{int(level*100)}'
                row[col_name] = quantiles[level]
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by sample size
        df = df.sort_values('sample_size')
        
        return df
    
    def format_se_table(self, 
                       statistic: str, 
                       results: Dict[int, Dict]) -> pd.DataFrame:
        """
        Format standard errors as a pandas DataFrame.
        
        Args:
            statistic: Name of the test statistic
            results: Results dictionary for the statistic
            
        Returns:
            DataFrame with formatted standard errors
        """
        data = []
        
        for sample_size in self.sample_sizes:
            if sample_size not in results:
                continue
            
            row = {'sample_size': sample_size}
            standard_errors = results[sample_size]['standard_errors']
            
            for level in self.quantile_levels:
                col_name = f'se_q{int(level*100)}'
                row[col_name] = standard_errors[level]
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('sample_size')
        
        return df
    
    def export_to_csv(self, 
                     df: pd.DataFrame, 
                     filename: str,
                     precision: int = 4):
        """Export DataFrame to CSV format."""
        output_path = self.output_dir / 'csv' / filename
        df.to_csv(output_path, index=False, float_format=f'%.{precision}f')
        logger.info(f"Exported CSV: {output_path}")
        return output_path
    
    def export_to_latex(self, 
                       df: pd.DataFrame, 
                       filename: str,
                       caption: str = "",
                       precision: int = 4):
        """Export DataFrame to LaTeX format."""
        output_path = self.output_dir / 'latex' / filename
        
        # Format the DataFrame for LaTeX
        float_format = lambda x: f'{x:.{precision}f}'
        
        latex_content = df.to_latex(
            index=False,
            float_format=float_format,
            caption=caption,
            label=f"tab:{filename.replace('.tex', '')}",
            column_format='l' + 'r' * (len(df.columns) - 1)
        )
        
        # Add booktabs commands
        latex_content = latex_content.replace('\\toprule', '\\toprule\n\\midrule')
        latex_content = latex_content.replace('\\bottomrule', '\\midrule\n\\bottomrule')
        
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"Exported LaTeX: {output_path}")
        return output_path
    
    def export_to_markdown(self, 
                          df: pd.DataFrame, 
                          filename: str,
                          precision: int = 4):
        """Export DataFrame to Markdown format."""
        output_path = self.output_dir / 'markdown' / filename
        
        # Create markdown table
        lines = []
        
        # Header
        headers = df.columns.tolist()
        lines.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
        lines.append('|' + '|'.join(['---' if h in ['sample_size', 'statistic'] else '------'
                                     for h in headers]) + '|')
        
        # Data rows
        for _, row in df.iterrows():
            formatted_row = []
            for col in headers:
                value = row[col]
                if col == 'sample_size':
                    formatted_row.append(str(int(value)))
                elif col == 'statistic' or isinstance(value, str):
                    formatted_row.append(str(value))
                else:
                    formatted_row.append(f'{value:.{precision}f}')
            lines.append('| ' + ' | '.join(formatted_row) + ' |')
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Exported Markdown: {output_path}")
        return output_path
    
    def generate_all_tables(self):
        """Generate quantile tables for all configurations in all formats."""
        logger.info("Starting quantile table generation...")
        
        # Process all configurations
        all_results = self.aggregate_all_configurations()
        
        # Generate tables for each statistic
        for statistic in self.statistics:
            if statistic not in all_results:
                logger.error(f"No results for {statistic}")
                continue
            
            results = all_results[statistic]
            
            # Create quantile table
            quantile_df = self.format_quantile_table(statistic, results)
            
            # Export in all formats
            self.export_to_csv(quantile_df, f'{statistic}_quantiles.csv')
            self.export_to_latex(
                quantile_df, 
                f'{statistic}_quantiles.tex',
                caption=f"Critical values for {statistic.replace('_', ' ').title()}"
            )
            self.export_to_markdown(quantile_df, f'{statistic}_quantiles.md')
            
            # Create standard error table
            se_df = self.format_se_table(statistic, results)
            
            # Export SE tables
            self.export_to_csv(se_df, f'{statistic}_standard_errors.csv')
            self.export_to_latex(
                se_df,
                f'{statistic}_standard_errors.tex',
                caption=f"Standard errors for {statistic.replace('_', ' ').title()}"
            )
            self.export_to_markdown(se_df, f'{statistic}_standard_errors.md')
        
        # Create master summary table
        self.create_master_summary(all_results)
        
        # Generate metadata file
        self.generate_metadata(all_results)
        
        logger.info("[OK] Quantile table generation complete!")
    
    def create_master_summary(self, all_results: Dict):
        """Create a master summary table combining all statistics."""
        summary_data = []
        
        for statistic in self.statistics:
            if statistic not in all_results:
                continue
            
            for sample_size in self.sample_sizes:
                if sample_size not in all_results[statistic]:
                    continue
                
                row = {
                    'statistic': statistic.replace('_', ' ').title(),
                    'sample_size': sample_size
                }
                
                quantiles = all_results[statistic][sample_size]['quantiles']
                for level in self.quantile_levels:
                    col_name = f'q{int(level*100)}'
                    row[col_name] = quantiles[level]
                
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Export master summary
        self.export_to_csv(summary_df, 'master_quantile_summary.csv')
        self.export_to_latex(
            summary_df,
            'master_quantile_summary.tex',
            caption="Critical values for all test statistics"
        )
        self.export_to_markdown(summary_df, 'master_quantile_summary.md')
        
        logger.info("Created master summary table")
    
    def generate_metadata(self, all_results: Dict):
        """Generate metadata file documenting the analysis."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'quantile_levels': self.quantile_levels,
            'sample_sizes': self.sample_sizes,
            'statistics': self.statistics,
            'batch_size_for_se': 100,
            'bootstrap_iterations': 1000,
            'configurations_processed': {},
            'convergence_status': {}
        }
        
        # Add details for each configuration
        for statistic in self.statistics:
            if statistic not in all_results:
                continue
            
            metadata['convergence_status'][statistic] = {}
            
            for sample_size in self.sample_sizes:
                if sample_size not in all_results[statistic]:
                    continue
                
                config_key = f"{statistic}_n{sample_size}"
                config_data = all_results[statistic][sample_size]
                
                metadata['configurations_processed'][config_key] = {
                    'n_iterations': config_data['n_iterations'],
                    'converged': config_data['metadata'].get('convergence_achieved', None)
                }
                
                metadata['convergence_status'][statistic][sample_size] = \
                    config_data['metadata'].get('convergence_achieved', None)
        
        # Save metadata
        metadata_path = self.output_dir / 'quantile_analysis_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated metadata: {metadata_path}")
        
        return metadata


def main():
    """Main function to run quantile extraction and table generation."""
    # Initialize extractor
    extractor = QuantileExtractor(
        data_dir="data/processed",
        output_dir="reports/quantile_tables"
    )
    
    # Generate all tables
    extractor.generate_all_tables()
    
    print("\n" + "="*60)
    print("QUANTILE TABLE GENERATION COMPLETE")
    print("="*60)
    print("\nGenerated files in reports/quantile_tables/:")
    print("  - CSV files in csv/")
    print("  - LaTeX files in latex/")
    print("  - Markdown files in markdown/")
    print("  - Metadata in quantile_analysis_metadata.json")


if __name__ == "__main__":
    main()
