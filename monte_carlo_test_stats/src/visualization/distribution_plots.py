"""
Distribution visualization module for Monte Carlo simulation results.
Provides publication-quality plots for distributions, convergence, and comparisons.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from pathlib import Path

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_palette("colorblind")


def get_accessible_colors(n_colors: int = 5) -> List[str]:
    """Get colorblind-friendly color palette."""
    return sns.color_palette("colorblind", n_colors).as_hex()


def plot_distribution(
    data: np.ndarray,
    statistic_name: str,
    sample_size: int,
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create histogram with KDE overlay.
    
    Args:
        data: Simulation results
        statistic_name: Name of test statistic
        sample_size: Sample size n
        bins: Number of histogram bins
        figsize: Figure dimensions
        
    Returns:
        (figure, axes) tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram
    n, bins_edges, _ = ax.hist(
        data, bins=bins, density=True, 
        alpha=0.7, color='skyblue', edgecolor='black'
    )
    
    # KDE overlay
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 500)
    kde_values = kde(x_range)
    ax.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
    
    # Labels
    ax.set_xlabel(f'{statistic_name.replace("_", " ").title()} Statistic')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {statistic_name} (n={sample_size})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_empirical_cdf(
    data: np.ndarray,
    quantiles: List[float] = [0.75, 0.90, 0.95, 0.99],
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate empirical CDF plot with quantile markers.
    
    Args:
        data: Simulation results
        quantiles: Quantile levels to mark
        figsize: Figure dimensions
        
    Returns:
        (figure, axes) tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort data for CDF
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Plot ECDF
    ax.plot(sorted_data, ecdf, 'b-', linewidth=2, label='Empirical CDF')
    
    # Mark quantiles
    colors = get_accessible_colors(len(quantiles))
    for q, color in zip(quantiles, colors):
        q_value = np.quantile(data, q)
        ax.axvline(x=q_value, linestyle='--', alpha=0.7, 
                  color=color, label=f'{q:.0%} quantile: {q_value:.4f}')
        ax.axhline(y=q, linestyle=':', alpha=0.5, color=color)
    
    ax.set_xlabel('Statistic Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical CDF with Quantiles')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    return fig, ax


def plot_qq(
    data: np.ndarray,
    distribution: str = 'norm',
    figsize: Tuple[int, int] = (8, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create Q-Q plot for distribution assessment.
    
    Args:
        data: Simulation results
        distribution: Theoretical distribution ('norm', 'uniform', etc.)
        figsize: Figure dimensions
        
    Returns:
        (figure, axes) tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Q-Q plot
    stats.probplot(data, dist=distribution, plot=ax)
    
    # Customize
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title(f'Q-Q Plot ({distribution.title()} Distribution)')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_convergence(
    iterations: np.ndarray,
    quantile_history: Dict[float, np.ndarray],
    threshold: float = 0.0001,
    figsize: Tuple[int, int] = (12, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot quantile convergence over iterations.
    
    Args:
        iterations: Iteration counts
        quantile_history: Dict of quantile level -> values over time
        threshold: Convergence threshold
        figsize: Figure dimensions
        
    Returns:
        (figure, axes) tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = get_accessible_colors(len(quantile_history))
    
    for (q_level, values), color in zip(quantile_history.items(), colors):
        # Plot quantile evolution
        ax.plot(iterations, values, label=f'{q_level:.0%} quantile', 
               color=color, linewidth=2)
        
        # Add convergence band
        final_value = values[-1]
        ax.fill_between(iterations, 
                       final_value - threshold, 
                       final_value + threshold,
                       alpha=0.2, color=color)
        ax.axhline(y=final_value, linestyle=':', alpha=0.5, color=color)
    
    ax.set_xscale('log')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Quantile Value')
    ax.set_title('Quantile Convergence Monitoring')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_sample_size_comparison(
    data_dict: Dict[int, np.ndarray],
    statistic_name: str,
    plot_type: str = 'kde',
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare distributions across sample sizes.
    
    Args:
        data_dict: Dict of sample_size -> simulation results
        statistic_name: Name of test statistic
        plot_type: 'kde' or 'ecdf'
        figsize: Figure dimensions
        
    Returns:
        (figure, axes) tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = get_accessible_colors(len(data_dict))
    
    for (n, data), color in zip(sorted(data_dict.items()), colors):
        if plot_type == 'kde':
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(
                min(d.min() for d in data_dict.values()),
                max(d.max() for d in data_dict.values()),
                500
            )
            ax.plot(x_range, kde(x_range), label=f'n={n}', 
                   color=color, linewidth=2)
        else:  # ecdf
            sorted_data = np.sort(data)
            ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, ecdf, label=f'n={n}', 
                   color=color, linewidth=2)
    
    ax.set_xlabel(f'{statistic_name.replace("_", " ").title()} Statistic')
    ax.set_ylabel('Density' if plot_type == 'kde' else 'Cumulative Probability')
    ax.set_title(f'{statistic_name}: Effect of Sample Size')
    ax.legend(title='Sample Size')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_critical_value_heatmap(
    critical_values: Dict[int, Dict[float, float]],
    figsize: Tuple[int, int] = (10, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create heatmap of critical values.
    
    Args:
        critical_values: Nested dict: sample_size -> quantile -> value
        figsize: Figure dimensions
        
    Returns:
        (figure, axes) tuple
    """
    # Convert to DataFrame
    df = pd.DataFrame(critical_values).T
    df.index.name = 'Sample Size'
    df.columns = [f'{q:.0%}' for q in df.columns]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(df, annot=True, fmt='.4f', cmap='YlOrRd',
                cbar_kws={'label': 'Critical Value'},
                ax=ax)
    
    ax.set_title('Critical Values by Sample Size and Significance Level')
    ax.set_xlabel('Significance Level (α)')
    ax.set_ylabel('Sample Size (n)')
    
    return fig, ax


def plot_theoretical_comparison(
    sample_sizes: List[int],
    empirical: List[float],
    theoretical: List[float],
    statistic_name: str,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot empirical vs theoretical values.
    
    Args:
        sample_sizes: List of sample sizes
        empirical: Empirical quantile values
        theoretical: Theoretical values
        statistic_name: Name of test statistic
        figsize: Figure dimensions
        
    Returns:
        (figure, axes) tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot empirical
    ax.scatter(sample_sizes, empirical, s=100, alpha=0.7, 
              label='Empirical', zorder=5)
    
    # Plot theoretical
    ax.plot(sample_sizes, theoretical, 'r--', linewidth=2, 
           label='Theoretical', zorder=4)
    
    # Calculate errors
    mae = np.mean(np.abs(np.array(empirical) - np.array(theoretical)))
    rmse = np.sqrt(np.mean((np.array(empirical) - np.array(theoretical))**2))
    
    # Add error text
    ax.text(0.95, 0.05, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Critical Value')
    ax.set_title(f'{statistic_name}: Empirical vs Theoretical')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def create_faceted_plot(
    data_dict: Dict[Tuple[str, int], np.ndarray],
    ncols: int = 3,
    figsize: Tuple[int, int] = (15, 12)
) -> plt.Figure:
    """
    Create grid of subplots for multiple configurations.
    
    Args:
        data_dict: Dict of (statistic, sample_size) -> data
        ncols: Number of columns in grid
        figsize: Figure dimensions
        
    Returns:
        Figure object
    """
    nconfigs = len(data_dict)
    nrows = (nconfigs + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nconfigs > 1 else [axes]
    
    for idx, ((stat, n), data) in enumerate(data_dict.items()):
        if idx >= nconfigs:
            break
        
        ax = axes[idx]
        
        # Create histogram with KDE
        ax.hist(data, bins=30, density=True, alpha=0.7, 
               color='skyblue', edgecolor='black')
        
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2)
        
        ax.set_title(f'{stat} (n={n})')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(nconfigs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def save_publication_figure(
    fig: plt.Figure,
    filepath: Path,
    dpi: int = 300,
    bbox_inches: str = 'tight',
    **kwargs
):
    """
    Save figure at publication quality.
    
    Args:
        fig: Figure to save
        filepath: Output path
        dpi: Resolution (dots per inch)
        bbox_inches: Bounding box setting
        **kwargs: Additional savefig arguments
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(
        filepath,
        dpi=dpi,
        bbox_inches=bbox_inches,
        **kwargs
    )
    plt.close(fig)
