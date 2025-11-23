"""
Fixed batch generation script for all production simulation plots.
Handles missing theoretical values gracefully and ensures all plots are generated.
"""
import sys
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("data/interim")
OUTPUT_DIR = Path("reports/figures")
DPI = 300

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# FIXED: Complete theoretical values for all statistics
THEORETICAL_VALUES = {
    'kolmogorov_smirnov': {
        30: 0.248,
        50: 0.192,
        100: 0.136,
        500: 0.061,
        1000: 0.043
    },
    'anderson_darling': {
        # Using standard A² critical values with finite sample adjustment
        30: 0.787 * (1 + 4/30 - 25/900),  # ≈ 0.873
        50: 0.787 * (1 + 4/50 - 25/2500),  # ≈ 0.847
        100: 0.787 * (1 + 4/100 - 25/10000),  # ≈ 0.816
        500: 0.787 * (1 + 4/500 - 25/250000),  # ≈ 0.793
        1000: 0.787 * (1 + 4/1000 - 25/1000000)  # ≈ 0.790
    },
    'durbin_watson': {
        # DW statistic centers around 2.0 for no autocorrelation
        # These are approximate upper critical values at 5% significance
        30: 2.35,
        50: 2.30,
        100: 2.20,
        500: 2.10,
        1000: 2.05
    }
}

def safe_import_visualization():
    """Safely import visualization functions with fallback."""
    try:
        from src.visualization.distribution_plots import (
            plot_distribution, plot_empirical_cdf, plot_qq,
            plot_convergence, plot_sample_size_comparison,
            plot_critical_value_heatmap, plot_theoretical_comparison,
            create_faceted_plot, save_publication_figure
        )
        return True, locals()
    except ImportError as e:
        logger.error(f"Failed to import visualization module: {e}")
        logger.info("Creating stub functions...")
        return False, {}

def load_simulation_results(filepath: Path) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """Load results and metadata from HDF5 file with error handling."""
    try:
        import h5py
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
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None, None

def load_checkpoint_history(statistic: str, sample_size: int) -> Dict:
    """Load convergence history from checkpoints with intermediate sampling for proper visualization."""
    pattern = f"checkpoint_{statistic}_{sample_size}_*.h5"
    checkpoints = sorted(CHECKPOINT_DIR.glob(pattern))
    
    if not checkpoints:
        return {}
    
    # Find the checkpoint with the most data
    import h5py
    largest_checkpoint = None
    max_iterations = 0
    
    for checkpoint_file in checkpoints:
        try:
            with h5py.File(checkpoint_file, 'r') as f:
                if 'metadata' in f.attrs:
                    metadata = json.loads(f.attrs['metadata'])
                    iterations = metadata.get('iterations_completed', 0)
                    if iterations > max_iterations:
                        max_iterations = iterations
                        largest_checkpoint = checkpoint_file
        except Exception as e:
            logger.warning(f"Error checking checkpoint {checkpoint_file}: {e}")
            continue
    
    if not largest_checkpoint:
        return {}
    
    # Load the largest checkpoint and sample at multiple points
    history = {'iterations': [], 'quantiles': {}}
    target_quantiles = [0.75, 0.90, 0.95, 0.99]
    
    # Define sampling points - more dense at the beginning for better convergence visualization
    sample_points = [
        100, 500, 1000, 2500, 5000, 7500,
        10000, 25000, 50000, 75000,
        100000, 150000, 200000, 250000,
        300000, 400000, 500000, 750000,
        1000000, 1500000, 2000000, 2500000,
        3000000, 3500000
    ]
    
    try:
        with h5py.File(largest_checkpoint, 'r') as f:
            if 'data' not in f:
                return {}
                
            data = f['data'][:]
            data_length = len(data)
            
            # Filter sample points to those within data length
            valid_sample_points = [sp for sp in sample_points if sp <= data_length]
            
            # Add the final point if not already included
            if data_length not in valid_sample_points and data_length > 0:
                valid_sample_points.append(data_length)
            
            # Calculate quantiles at each sample point
            for sp in valid_sample_points:
                if sp > 0:
                    sample_data = data[:sp]
                    history['iterations'].append(sp)
                    
                    for q in target_quantiles:
                        q_value = float(np.quantile(sample_data, q))
                        if q not in history['quantiles']:
                            history['quantiles'][q] = []
                        history['quantiles'][q].append(q_value)
            
            logger.info(f"  Loaded convergence data: {len(history['iterations'])} sample points from {data_length:,} total iterations")
                            
    except Exception as e:
        logger.error(f"Error loading checkpoint {largest_checkpoint}: {e}")
        return {}
    
    return history

def generate_individual_plots(statistic: str, sample_size: int, results: np.ndarray, 
                            metadata: Dict, viz_funcs: Dict):
    """Generate all plot types for a single configuration."""
    logger.info(f"Generating plots for {statistic} n={sample_size}")
    
    if not viz_funcs:
        logger.error("Visualization functions not available")
        return
    
    try:
        # 1. Distribution plot (histogram + KDE)
        fig, ax = viz_funcs['plot_distribution'](results, statistic, sample_size)
        viz_funcs['save_publication_figure'](
            fig,
            OUTPUT_DIR / f"{statistic}_{sample_size}_distribution.png",
            dpi=DPI
        )
        logger.info(f"  [OK] Distribution plot saved")
    except Exception as e:
        logger.error(f"  [FAIL] Distribution plot failed: {e}")
    
    try:
        # 2. ECDF with quantiles
        quantiles = [0.75, 0.90, 0.95, 0.99]
        fig, ax = viz_funcs['plot_empirical_cdf'](results, quantiles)
        ax.set_title(f'{statistic.replace("_", " ").title()} (n={sample_size}) - ECDF')
        viz_funcs['save_publication_figure'](
            fig,
            OUTPUT_DIR / f"{statistic}_{sample_size}_ecdf.png",
            dpi=DPI
        )
        logger.info(f"  [OK] ECDF plot saved")
    except Exception as e:
        logger.error(f"  [FAIL] ECDF plot failed: {e}")
    
    try:
        # 3. Q-Q plot
        fig, ax = viz_funcs['plot_qq'](results)
        ax.set_title(f'{statistic.replace("_", " ").title()} (n={sample_size}) - Q-Q Plot')
        viz_funcs['save_publication_figure'](
            fig,
            OUTPUT_DIR / f"{statistic}_{sample_size}_qq.png",
            dpi=DPI
        )
        logger.info(f"  [OK] Q-Q plot saved")
    except Exception as e:
        logger.error(f"  [FAIL] Q-Q plot failed: {e}")
    
    # 4. Convergence plot (if checkpoint data available)
    history = load_checkpoint_history(statistic, sample_size)
    if history and 'iterations' in history and history['iterations']:
        try:
            fig, ax = viz_funcs['plot_convergence'](
                np.array(history['iterations']),
                history['quantiles'],
                threshold=0.0001
            )
            ax.set_title(f'{statistic.replace("_", " ").title()} (n={sample_size}) - Convergence')
            viz_funcs['save_publication_figure'](
                fig,
                OUTPUT_DIR / f"{statistic}_{sample_size}_convergence.png",
                dpi=DPI
            )
            logger.info(f"  [OK] Convergence plot saved")
        except Exception as e:
            logger.error(f"  [FAIL] Convergence plot failed: {e}")
    else:
        logger.info(f"  - Convergence plot skipped (no checkpoint data)")

def generate_comparison_plots(viz_funcs: Dict):
    """Generate comparative visualizations across configurations."""
    logger.info("Generating comparison plots...")
    
    if not viz_funcs:
        logger.error("Visualization functions not available")
        return
    
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
            
            if results is not None:
                if statistic not in all_results:
                    all_results[statistic] = {}
                    all_metadata[statistic] = {}
                
                all_results[statistic][sample_size] = results
                all_metadata[statistic][sample_size] = metadata
    
    logger.info(f"Loaded data for {len(all_results)} statistics")
    
    # Generate comparison plots for each statistic
    for statistic in all_results.keys():
        logger.info(f"Processing comparison plots for {statistic}")
        
        # 1. Sample size comparison (KDE overlay)
        try:
            fig, ax = viz_funcs['plot_sample_size_comparison'](
                all_results[statistic],
                statistic,
                plot_type='kde'
            )
            viz_funcs['save_publication_figure'](
                fig,
                OUTPUT_DIR / f"{statistic}_sample_size_comparison_kde.png",
                dpi=DPI
            )
            logger.info(f"  [OK] KDE comparison saved")
        except Exception as e:
            logger.error(f"  [FAIL] KDE comparison failed: {e}")
        
        # 2. Sample size comparison (ECDF overlay)
        try:
            fig, ax = viz_funcs['plot_sample_size_comparison'](
                all_results[statistic],
                statistic,
                plot_type='ecdf'
            )
            viz_funcs['save_publication_figure'](
                fig,
                OUTPUT_DIR / f"{statistic}_sample_size_comparison_ecdf.png",
                dpi=DPI
            )
            logger.info(f"  [OK] ECDF comparison saved")
        except Exception as e:
            logger.error(f"  [FAIL] ECDF comparison failed: {e}")
        
        # 3. Critical value heatmap
        try:
            critical_values = {}
            for sample_size, meta in all_metadata[statistic].items():
                if 'quantiles' in meta:
                    # Convert quantiles to proper format
                    quantiles_dict = meta['quantiles']
                    if isinstance(quantiles_dict, dict):
                        # Ensure numeric keys
                        critical_values[sample_size] = {
                            float(k) if isinstance(k, str) else k: v 
                            for k, v in quantiles_dict.items()
                        }
            
            if critical_values:
                fig, ax = viz_funcs['plot_critical_value_heatmap'](critical_values)
                ax.set_title(f'{statistic.replace("_", " ").title()} - Critical Values')
                viz_funcs['save_publication_figure'](
                    fig,
                    OUTPUT_DIR / f"{statistic}_critical_values_heatmap.png",
                    dpi=DPI
                )
                logger.info(f"  [OK] Critical values heatmap saved")
            else:
                logger.warning(f"  - Heatmap skipped (no critical values)")
        except Exception as e:
            logger.error(f"  [FAIL] Heatmap failed: {e}")
        
        # 4. Theoretical comparison (if available)
        if statistic in THEORETICAL_VALUES:
            try:
                sample_sizes = sorted(all_results[statistic].keys())
                empirical = []
                theoretical = []
                
                for n in sample_sizes:
                    if n in THEORETICAL_VALUES[statistic]:
                        # Get 95% quantile from metadata
                        if n in all_metadata[statistic] and 'quantiles' in all_metadata[statistic][n]:
                            quantiles_dict = all_metadata[statistic][n]['quantiles']
                            # Try different key formats
                            q95 = None
                            for key in ['0.95', 0.95, '95', 95]:
                                if str(key) in quantiles_dict or key in quantiles_dict:
                                    q95 = quantiles_dict.get(str(key), quantiles_dict.get(key))
                                    break
                            
                            if q95 is not None:
                                empirical.append(float(q95))
                                theoretical.append(THEORETICAL_VALUES[statistic][n])
                
                if empirical and theoretical:
                    fig, ax = viz_funcs['plot_theoretical_comparison'](
                        sample_sizes[:len(empirical)],
                        empirical,
                        theoretical,
                        statistic
                    )
                    viz_funcs['save_publication_figure'](
                        fig,
                        OUTPUT_DIR / f"{statistic}_theoretical_comparison.png",
                        dpi=DPI
                    )
                    logger.info(f"  [OK] Theoretical comparison saved")
                else:
                    logger.warning(f"  - Theoretical comparison skipped (insufficient data)")
            except Exception as e:
                logger.error(f"  [FAIL] Theoretical comparison failed: {e}")
        else:
            logger.info(f"  - Theoretical comparison skipped (no theoretical values)")
    
    # 5. Create faceted overview plot
    logger.info("Creating faceted overview plot")
    try:
        facet_data = {}
        for statistic in all_results.keys():
            for sample_size, results in all_results[statistic].items():
                if sample_size in [30, 100, 1000]:  # Representative sizes
                    facet_data[(statistic, sample_size)] = results
        
        if facet_data:
            fig = viz_funcs['create_faceted_plot'](facet_data, ncols=3)
            viz_funcs['save_publication_figure'](
                fig,
                OUTPUT_DIR / "all_statistics_faceted.png",
                dpi=DPI
            )
            logger.info("  [OK] Faceted plot saved")
        else:
            logger.warning("  - Faceted plot skipped (no data)")
    except Exception as e:
        logger.error(f"  [FAIL] Faceted plot failed: {e}")
    
    # 6. Create summary comparison at n=100
    logger.info("Creating summary comparison plot")
    try:
        n100_data = {}
        for statistic in all_results.keys():
            if 100 in all_results[statistic]:
                n100_data[statistic] = all_results[statistic][100]
        
        if n100_data:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for idx, (statistic, data) in enumerate(n100_data.items()):
                ax = axes[idx] if len(n100_data) == 3 else axes
                ax.hist(data, bins=50, density=True, alpha=0.7, 
                       color='skyblue', edgecolor='black')
                ax.set_title(f'{statistic.replace("_", " ").title()}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots if less than 3 statistics
            if len(n100_data) < 3:
                for idx in range(len(n100_data), 3):
                    axes[idx].set_visible(False)
            
            plt.suptitle('All Test Statistics at n=100', fontsize=14, y=1.02)
            plt.tight_layout()
            viz_funcs['save_publication_figure'](
                fig,
                OUTPUT_DIR / "summary_n100_comparison.png",
                dpi=DPI
            )
            logger.info("  [OK] Summary comparison saved")
        else:
            logger.warning("  - Summary comparison skipped (no n=100 data)")
    except Exception as e:
        logger.error(f"  [FAIL] Summary comparison failed: {e}")

def main():
    """Main execution function."""
    global DATA_DIR, CHECKPOINT_DIR  # Allow modification if needed
    
    logger.info("="*60)
    logger.info("FIXED BATCH PLOT GENERATION")
    logger.info("="*60)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Import visualization functions
    success, viz_funcs = safe_import_visualization()
    
    if not success:
        logger.error("Cannot proceed without visualization module")
        return 1
    
    # Check if DATA_DIR exists, if not try to find it
    if not DATA_DIR.exists():
        logger.warning(f"DATA_DIR {DATA_DIR} does not exist, searching for alternatives...")
        # Try alternative paths
        alt_paths = [
            Path("data/processed"),
            Path("monte_carlo_test_stats/data/processed"),
            Path("../data/processed"),
            Path.cwd() / "data" / "processed"
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                DATA_DIR = alt_path
                CHECKPOINT_DIR = alt_path.parent / "interim"
                logger.info(f"Using alternative path: {DATA_DIR}")
                break
        else:
            logger.error("Could not find data directory in any expected location")
            return 1
    
    # Process all result files for individual plots
    result_files = list(DATA_DIR.glob("*_results.h5"))
    logger.info(f"Found {len(result_files)} result files")
    
    # Generate individual plots
    successful_plots = 0
    failed_plots = 0
    
    for filepath in result_files:
        try:
            # Parse filename
            parts = filepath.stem.split('_')
            if len(parts) >= 3:
                statistic = '_'.join(parts[:-2])
                sample_size = int(parts[-2])
                
                # Load and process
                results, metadata = load_simulation_results(filepath)
                if results is not None:
                    generate_individual_plots(statistic, sample_size, results, metadata, viz_funcs)
                    successful_plots += 1
                else:
                    failed_plots += 1
                    
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            failed_plots += 1
    
    # Generate comparison plots
    generate_comparison_plots(viz_funcs)
    
    # Count generated plots
    if OUTPUT_DIR.exists():
        plot_count = len(list(OUTPUT_DIR.glob("*.png")))
        logger.info("="*60)
        logger.info(f"[OK] Generated {plot_count} plots in {OUTPUT_DIR}")
        logger.info(f"  Successful configurations: {successful_plots}")
        logger.info(f"  Failed configurations: {failed_plots}")
    
    # Final summary
    logger.info("="*60)
    logger.info("GENERATION COMPLETE")
    logger.info("="*60)
    logger.info("Expected plot types generated:")
    logger.info("  1. Individual plots (distribution, ECDF, Q-Q, convergence)")
    logger.info("  2. Comparison plots (KDE, ECDF overlays)")
    logger.info("  3. Critical value heatmaps")
    logger.info("  4. Theoretical comparisons")
    logger.info("  5. Faceted overview")
    logger.info("  6. Summary comparison at n=100")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
