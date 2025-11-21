"""
Unit tests for distribution visualization module.
Tests histogram/KDE plots, convergence tracking, Q-Q plots,
comparative visualizations, and publication-quality output.
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch


class TestDistributionVisualization:
    """Test suite for distribution visualization functions"""
    
    def test_histogram_with_kde(self):
        """
        Test: Create histogram with KDE overlay
        Expected: Figure with histogram bars and smooth KDE line
        """
        from src.visualization.distribution_plots import plot_distribution
        
        # Generate test data
        np.random.seed(42)
        data = np.random.standard_normal(10000)
        
        fig, ax = plot_distribution(
            data,
            statistic_name='kolmogorov_smirnov',
            sample_size=100,
            bins=50
        )
        
        # Check figure properties
        assert fig is not None
        assert ax is not None
        
        # Verify histogram exists
        patches = ax.patches
        assert len(patches) == 50  # 50 bins
        
        # Verify KDE line exists
        lines = ax.lines
        assert len(lines) >= 1  # At least KDE line
        
        # Check labels
        assert ax.get_xlabel() != ''
        assert ax.get_ylabel() == 'Density'
        assert ax.get_title() != ''
        
        plt.close(fig)
    
    def test_empirical_cdf_plot(self):
        """
        Test: Generate empirical CDF plot
        Expected: Monotonic increasing function from 0 to 1
        """
        from src.visualization.distribution_plots import plot_empirical_cdf
        
        data = np.random.standard_normal(1000)
        quantiles = [0.75, 0.90, 0.95, 0.99]
        
        fig, ax = plot_empirical_cdf(data, quantiles=quantiles)
        
        # Check CDF properties
        lines = ax.lines
        assert len(lines) >= 1
        
        # Verify quantile markers
        vlines = [l for l in ax.lines if l.get_linestyle() == '--']
        assert len(vlines) == len(quantiles)
        
        # Check y-axis range
        assert ax.get_ylim()[0] >= 0
        assert ax.get_ylim()[1] <= 1.1
        
        plt.close(fig)
    
    def test_qq_plot_normality(self):
        """
        Test: Q-Q plot for assessing normality
        Expected: Points follow diagonal line for normal data
        """
        from src.visualization.distribution_plots import plot_qq
        
        # Normal data should align with theoretical line
        normal_data = np.random.standard_normal(1000)
        fig, ax = plot_qq(normal_data, distribution='norm')
        
        # Check for scatter points and reference line
        assert len(ax.collections) >= 1  # Scatter points
        assert len(ax.lines) >= 1  # Reference line
        
        # Verify axes labels
        assert 'Theoretical' in ax.get_xlabel()
        assert 'Sample' in ax.get_ylabel()
        
        plt.close(fig)
    
    def test_convergence_plot(self):
        """
        Test: Plot quantile convergence over iterations
        Expected: Lines showing quantile evolution with convergence bands
        """
        from src.visualization.distribution_plots import plot_convergence
        
        # Simulate convergence data
        iterations = np.arange(10000, 1000000, 10000)
        quantile_history = {
            0.95: np.random.normal(1.36, 0.01, len(iterations)),
            0.99: np.random.normal(1.63, 0.01, len(iterations))
        }
        
        fig, ax = plot_convergence(
            iterations,
            quantile_history,
            threshold=0.0001
        )
        
        # Check lines for each quantile
        assert len(ax.lines) >= 2
        
        # Verify convergence bands
        collections = ax.collections
        assert len(collections) >= 2  # Fill_between for bands
        
        # Check log scale on x-axis
        assert ax.get_xscale() == 'log'
        
        plt.close(fig)
    
    def test_multi_sample_comparison(self):
        """
        Test: Compare distributions across sample sizes
        Expected: Overlay plot with different colors/styles per sample size
        """
        from src.visualization.distribution_plots import plot_sample_size_comparison
        
        # Generate data for different sample sizes
        data_dict = {
            30: np.random.standard_normal(10000),
            50: np.random.standard_normal(10000) * 0.9,
            100: np.random.standard_normal(10000) * 0.8,
            500: np.random.standard_normal(10000) * 0.7,
            1000: np.random.standard_normal(10000) * 0.6
        }
        
        fig, ax = plot_sample_size_comparison(
            data_dict,
            statistic_name='kolmogorov_smirnov'
        )
        
        # Check for multiple lines (one per sample size)
        assert len(ax.lines) >= 5
        
        # Verify legend
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 5
        
        plt.close(fig)
    
    def test_critical_value_heatmap(self):
        """
        Test: Create heatmap of critical values
        Expected: Color-coded matrix with annotations
        """
        from src.visualization.distribution_plots import plot_critical_value_heatmap
        
        # Mock critical values
        critical_values = {
            30: {0.90: 0.234, 0.95: 0.242, 0.99: 0.290},
            50: {0.90: 0.180, 0.95: 0.188, 0.99: 0.226},
            100: {0.90: 0.127, 0.95: 0.136, 0.99: 0.163}
        }
        
        fig, ax = plot_critical_value_heatmap(critical_values)
        
        # Check for heatmap image
        images = ax.images
        assert len(images) == 1
        
        # Verify colorbar exists
        assert fig.axes[-1].get_ylabel() != ''  # Colorbar label
        
        # Check annotations
        texts = ax.texts
        assert len(texts) == 9  # 3x3 grid
        
        plt.close(fig)
    
    def test_theoretical_comparison_plot(self):
        """
        Test: Plot empirical vs theoretical values
        Expected: Scatter plot with theoretical line and error metrics
        """
        from src.visualization.distribution_plots import plot_theoretical_comparison
        
        sample_sizes = [30, 50, 100, 500, 1000]
        empirical = [0.242, 0.188, 0.136, 0.061, 0.043]
        theoretical = [0.248, 0.192, 0.136, 0.061, 0.043]
        
        fig, ax = plot_theoretical_comparison(
            sample_sizes,
            empirical,
            theoretical,
            statistic_name='kolmogorov_smirnov'
        )
        
        # Check for scatter and line
        assert len(ax.collections) >= 1  # Empirical scatter
        assert len(ax.lines) >= 1  # Theoretical line
        
        # Verify error metrics in text
        text_present = any('MAE' in t.get_text() for t in ax.texts)
        assert text_present
        
        plt.close(fig)
    
    def test_publication_quality_export(self):
        """
        Test: Export figures at publication quality
        Expected: High DPI, correct dimensions, saved to file
        """
        from src.visualization.distribution_plots import save_publication_figure
        
        # Create simple figure
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot([0, 1], [0, 1])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_figure.png'
            
            save_publication_figure(
                fig,
                filepath,
                dpi=300,
                bbox_inches='tight'
            )
            
            # Verify file created
            assert filepath.exists()
            assert filepath.stat().st_size > 0
            
            # Check file is readable
            from PIL import Image
            img = Image.open(filepath)
            assert img.size[0] > 1000  # Width > 1000px at 300 DPI
        
        plt.close(fig)
    
    def test_color_palette_accessibility(self):
        """
        Test: Use colorblind-friendly palette
        Expected: Distinct colors that work for color vision deficiencies
        """
        from src.visualization.distribution_plots import get_accessible_colors
        
        colors = get_accessible_colors(n_colors=5)
        
        # Should return 5 distinct colors
        assert len(colors) == 5
        assert len(set(colors)) == 5  # All unique
        
        # Should be valid matplotlib colors
        for color in colors:
            assert plt.colors.is_color_like(color)
    
    def test_faceted_visualization(self):
        """
        Test: Create faceted plots for all statistics
        Expected: Grid of subplots, one per configuration
        """
        from src.visualization.distribution_plots import create_faceted_plot
        
        # Mock data for multiple configurations
        configs = [
            ('kolmogorov_smirnov', 30),
            ('kolmogorov_smirnov', 100),
            ('durbin_watson', 30),
            ('durbin_watson', 100)
        ]
        
        data_dict = {config: np.random.randn(1000) for config in configs}
        
        fig = create_faceted_plot(data_dict, ncols=2)
        
        # Check subplot count
        assert len(fig.axes) == 4
        
        # Verify each has content
        for ax in fig.axes:
            assert len(ax.lines) > 0 or len(ax.patches) > 0
        
        plt.close(fig)


# Test specifications
"""
Visualization Module Test Coverage:
1. Distribution plots: Histogram, KDE, ECDF
2. Convergence tracking: Quantile evolution plots
3. Normality assessment: Q-Q plots
4. Comparative views: Multi-sample overlays
5. Critical values: Heatmaps with annotations
6. Theoretical comparison: Empirical vs expected
7. Export quality: 300 DPI publication figures
8. Accessibility: Colorblind-friendly palettes
9. Layout: Faceted grids for multiple configs

Required functionality:
- Matplotlib/Seaborn integration
- Statistical plot types
- Customizable aesthetics
- High-quality export
- Interactive notebook support
"""
