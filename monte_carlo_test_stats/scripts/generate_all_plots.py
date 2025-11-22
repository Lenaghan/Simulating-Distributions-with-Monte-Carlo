"""
Batch generation script for all production simulation plots.
Generates publication-quality visualizations for all test statistics and sample sizes.
"""
import sys
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization.distribution_plots import (
    plot_distribution, plot_empirical_cdf, plot_qq,
    plot_convergence, plot_sample_size_comparison,
    plot_critical_value_heatmap, plot_theoretical_comparison,
    create_faceted_plot, save_publication_figure
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("data/interim")
OUTPUT_DIR = Path("reports/figures")
DPI = 300

# Theoretical values for comparison (at 95% quantile)
THEORETICAL_VALUES = {
    'kolmogorov_smirnov': {
        30: 0.248,
        50: 0.192,
        100: 0.136,
        500: 0.061,
        1000: 0.043
    },
    'anderson_darling': {
        30: 0.787 * (1 + 4/30 - 25/900),  # Finite sample adjustment
        50: 0.787 * (1 + 4/50 - 25/2500),
        100: 0.787 * (1 + 4/100 - 25/10000),
        500: 0.787 * (1 + 4/500 - 25/250000),
        1000: 0.787 * (1 + 4/1000 - 25/1000000)
    },
    'durbin_watson': {
        # Centered around 2 for no autocorrelation
        30: 2.0,
        50: 2.0,
        100: 2.0,
        500: 2.0,
        1000: 2.0
    }
}


def load_simulation_results(filepath: Path) -> Tuple[np.ndarray, Dict]:
    """Load results and metadata from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        results = f['results'][:]
        metadata = {}
        for key in f.attrs.keys():
            value = f.attrs[key]
            if isinstance(value, str) and value.startswith('{'):
                metadata[key] = json.loads(value)
            else:
                metadata[key] = value
    return results, metadata


def load_checkpoint_history(statistic: str, sample_size: int) -> Dict:
    """Load convergence history from checkpoints if available."""
    pattern = f"checkpoint_{statistic}_{sample_size}_*.h5"
    checkpoints = sorted(CHECKPOINT_DIR.glob(pattern))
    
    if not checkpoints:
        return {}
    
    history = {'iterations': [], 'quantiles': {}}
    
    for checkpoint_file in checkpoints:
        try:
            with h5py.File(checkpoint_file, 'r') as f:
                iterations = f.attrs.get('iterations_completed', 0)
                if 'last_quantiles' in f.attrs:
                    quantiles = json.loads(f.attrs['last_quantiles'])
                    history['iterations'].append(iterations)
                    for q, val in quantiles.items():
                        q_float = float(q)
                        if q_float not in history['quantiles']:
                            history['quantiles'][q_float] = []
                        history['quantiles'][q_float].append(val)
        except Exception as e:
            logger.warning(f"Error loading checkpoint {checkpoint_file}: {e}")
            continue
    
    return history


def generate_individual_plots(statistic: str, sample_size: int, results: np.ndarray, metadata: Dict):
    """Generate all plot types for a single configuration."""
    logger.info(f"Generating plots for {statistic} n={sample_size}")
    
    # 1. Distribution plot (histogram + KDE)
    fig, ax = plot_distribution(results, statistic, sample_size)
    save_publication_figure(
        fig,
        OUTPUT_DIR / f"{statistic}_{sample_size}_distribution.png",
        dpi=DPI
    )
    
    # 2. ECDF with quantiles
    quantiles = [0.75, 0.90, 0.95, 0.99]
    fig, ax = plot_empirical_cdf(results, quantiles)
    ax.set_title(f'{statistic.replace("_", " ").title()} (n={sample_size}) - ECDF')
    save_publication_figure(
        fig,
        OUTPUT_DIR / f"{statistic}_{sample_size}_ecdf.png",
        dpi=DPI
    )
    
    # 3. Q-Q plot
    fig, ax = plot_qq(results)
    ax.set_title(f'{statistic.replace("_", " ").title()} (n={sample_size}) - Q-Q Plot')
    save_publication_figure(
        fig,
        OUTPUT_DIR / f"{statistic}_{sample_size}_qq.png",
        dpi=DPI
    )
    
    # 4. Convergence plot (if checkpoint data available)
    history = load_checkpoint_history(statistic, sample_size)
    if history and 'iterations' in history and history['iterations']:
        fig, ax = plot_convergence(
            np.array(history['iterations']),
            history['quantiles'],
            threshold=0.0001
        )
        ax.set_title(f'{statistic.replace("_", " ").title()} (n={sample_size}) - Convergence')
        save_publication_figure(
            fig,
            OUTPUT_DIR / f"{statistic}_{sample_size}_convergence.png",
            dpi=DPI
        )


def generate_comparison_plots():
    """Generate comparative visualizations across configurations."""
    logger.info("Generating comparison plots")
    
    # Load all results
    all_results = {}
    all_metadata = {}
    
    for filepath in DATA_DIR.glob("*_results.h5"):
        # Parse filename
        parts = filepath.stem.split('_')
        if len(parts) >= 3:
            statistic = '_'.join(parts[:-2])
            sample_size = int(parts[-2])
            
            results, metadata = load_simulation_results(filepath)
            
            if statistic not in all_results:
                all_results[statistic] = {}
                all_metadata[statistic] = {}
            
            all_results[statistic][sample_size] = results
            all_metadata[statistic][sample_size] = metadata
    
    # Generate comparison plots for each statistic
    for statistic in all_results.keys():
        # 1. Sample size comparison (KDE overlay)
        fig, ax = plot_sample_size_comparison(
            all_results[statistic],
            statistic,
            plot_type='kde'
        )
        save_publication_figure(
            fig,
            OUTPUT_DIR / f"{statistic}_sample_size_comparison_kde.png",
            dpi=DPI
        )
        
        # 2. Sample size comparison (ECDF overlay)
        fig, ax = plot_sample_size_comparison(
            all_results[statistic],
            statistic,
            plot_type='ecdf'
        )
        save_publication_figure(
            fig,
            OUTPUT_DIR / f"{statistic}_sample_size_comparison_ecdf.png",
            dpi=DPI
        )
        
        # 3. Critical value heatmap
        critical_values = {}
        for sample_size, meta in all_metadata[statistic].items():
            if 'quantiles' in meta:
                critical_values[sample_size] = meta['quantiles']
        
        if critical_values:
            fig, ax = plot_critical_value_heatmap(critical_values)
            ax.set_title(f'{statistic.replace("_", " ").title()} - Critical Values')
            save_publication_figure(
                fig,
                OUTPUT_DIR / f"{statistic}_critical_values_heatmap.png",
                dpi=DPI
            )
        
        # 4. Theoretical comparison (if available)
        if statistic in THEORETICAL_VALUES:
            sample_sizes = sorted(all_results[statistic].keys())
            empirical = []
            theoretical = []
            
            for n in sample_sizes:
                # Get 95% quantile
                if n in all_metadata[statistic] and 'quantiles' in all_metadata[statistic][n]:
                    q95 = all_metadata[statistic][n]['quantiles'].get('0.95', 
                           all_metadata[statistic][n]['quantiles'].get(0.95, np.nan))
                    empirical.append(float(q95))
                    theoretical.append(THEORETICAL_VALUES[statistic][n])
            
            if empirical and theoretical:
                fig, ax = plot_theoretical_comparison(
                    sample_sizes,
                    empirical,
                    theoretical,
                    statistic
                )
                save_publication_figure(
                    fig,
                    OUTPUT_DIR / f"{statistic}_theoretical_comparison.png",
                    dpi=DPI
                )
    
    # 5. Create faceted overview plot
    logger.info("Creating faceted overview plot")
    facet_data = {}
    for statistic in all_results.keys():
        for sample_size, results in all_results[statistic].items():
            if sample_size in [30, 100, 1000]:  # Representative sizes
                facet_data[(statistic, sample_size)] = results
    
    if facet_data:
        fig = create_faceted_plot(facet_data, ncols=3)
        save_publication_figure(
            fig,
            OUTPUT_DIR / "all_statistics_faceted.png",
            dpi=DPI
        )
    
    # 6. Create summary comparison at n=100
    logger.info("Creating summary comparison plot")
    n100_data = {}
    for statistic in all_results.keys():
        if 100 in all_results[statistic]:
            n100_data[statistic] = all_results[statistic][100]
    
    if n100_data:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (statistic, data) in enumerate(n100_data.items()):
            ax = axes[idx]
            ax.hist(data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{statistic.replace("_", " ").title()}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('All Test Statistics at n=100', fontsize=14, y=1.02)
        plt.tight_layout()
        save_publication_figure(
            fig,
            OUTPUT_DIR / "summary_n100_comparison.png",
            dpi=DPI
        )


def main():
    """Main execution function."""
    logger.info(f"Starting batch plot generation")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all result files
    result_files = list(DATA_DIR.glob("*_results.h5"))
    logger.info(f"Found {len(result_files)} result files")
    
    # Generate individual plots
    for filepath in result_files:
        try:
            # Parse filename
            parts = filepath.stem.split('_')
            if len(parts) >= 3:
                statistic = '_'.join(parts[:-2])
                sample_size = int(parts[-2])
                
                # Load and process
                results, metadata = load_simulation_results(filepath)
                generate_individual_plots(statistic, sample_size, results, metadata)
                
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            continue
    
    # Generate comparison plots
    generate_comparison_plots()
    
    # Count generated plots
    plot_count = len(list(OUTPUT_DIR.glob("*.png")))
    logger.info(f"✓ Generated {plot_count} plots in {OUTPUT_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
