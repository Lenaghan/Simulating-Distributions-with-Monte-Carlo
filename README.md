# Monte Carlo Test Statistic Distributions
> Empirical critical value tables for Kolmogorov-Smirnov, Durbin-Watson, and Anderson-Darling test statistics via large-scale Monte Carlo simulation

This project generates complete probability distributions and critical values for three widely-used test statistics that lack closed-form analytical solutions. Using 10⁷ iterations per configuration, we produce quantile estimates with four-decimal precision across sample sizes n ∈ {30, 50, 100, 500, 1000}, enabling researchers to obtain exact critical values without interpolating from incomplete published tables.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/numpy-≥1.21.0-orange.svg)](https://numpy.org/)

## Table of Contents
- [🔬 Overview](#-overview)
- [📊 Results](#-results)
- [⚡ Quick Start](#-quick-start)
- [🛠️ Installation](#️-installation)
- [📁 Project Structure](#-project-structure)
- [🚀 Usage](#-usage)
- [⚙️ Configuration](#️-configuration)
- [🧪 Experiments](#-experiments)
- [📈 Validation](#-validation)
- [🤝 Contributing](#-contributing)
- [📚 Citation](#-citation)
- [📜 License](#-license)

---

## 🔬 Overview

### Problem Statement

Statistical hypothesis testing relies on critical values derived from test statistic distributions. The Kolmogorov-Smirnov (K-S), Durbin-Watson (D-W), and Anderson-Darling (A-D) test statistics lack closed-form distributions, forcing researchers to use incomplete lookup tables or imprecise p-values. Published tables cover only specific sample sizes and significance levels, creating gaps in clinical trials, quality control, and econometric modeling applications.

### Our Contribution

- **Complete empirical distributions**: 10⁷ Monte Carlo iterations per configuration for three test statistics
- **Precise critical values**: Quantile estimates at 0.75, 0.90, 0.95, and 0.99 levels with four-decimal precision
- **Comprehensive coverage**: Five sample sizes (n = 30, 50, 100, 500, 1000) producing 15 total configurations
- **Validated accuracy**: Mean absolute error < 0.002; Type I error rates within ±0.005 of nominal levels
- **Reproducible framework**: Checkpoint recovery, convergence monitoring, and deterministic random number generation

### Test Statistics

| Statistic | Formula | Application |
|-----------|---------|-------------|
| **Kolmogorov-Smirnov** | D_n = sup\|F_n(x) - F(x)\| | Goodness-of-fit testing, distribution validation |
| **Durbin-Watson** | DW = Σ(e_t - e_{t-1})² / Σe_t² | Autocorrelation detection in regression residuals |
| **Anderson-Darling** | A² = -n - (1/n)Σ(2i-1)[ln(Φ(z_i)) + ln(1-Φ(z_{n+1-i}))] | Tail-sensitive distributional assessment |

### Assumptions and Limitations

- Null distribution assumes standard normal N(0,1) samples
- D-W implementation treats samples as residuals (no regression predictors)
- Single-sample variants only; two-sample K-S and A-D require separate distributions
- Critical values apply to the specific null hypothesis formulation implemented

---

## 📊 Results

### Critical Value Tables

| Statistic | n | q₇₅ | q₉₀ | q₉₅ | q₉₉ |
|-----------|---|-----|-----|-----|-----|
| Kolmogorov-Smirnov | 30 | 0.1805 | 0.2172 | 0.2412 | 0.2892 |
| Kolmogorov-Smirnov | 50 | 0.1409 | 0.1694 | 0.1877 | 0.2243 |
| Kolmogorov-Smirnov | 100 | 0.1002 | 0.1206 | 0.1340 | 0.1604 |
| Kolmogorov-Smirnov | 500 | 0.0452 | 0.0543 | 0.0604 | 0.0726 |
| Kolmogorov-Smirnov | 1000 | 0.0321 | 0.0386 | 0.0427 | 0.0513 |
| Durbin-Watson | 30 | 2.1763 | 2.3900 | 2.5167 | 2.7496 |
| Durbin-Watson | 50 | 2.1505 | 2.3208 | 2.4187 | 2.6007 |
| Durbin-Watson | 100 | 2.1159 | 2.2367 | 2.3086 | 2.4404 |
| Durbin-Watson | 500 | 2.0567 | 2.1109 | 2.1433 | 2.2034 |
| Durbin-Watson | 1000 | 2.0410 | 2.0796 | 2.1024 | 2.1461 |
| Anderson-Darling | 30 | 0.4591 | 0.6174 | 0.7359 | 1.0091 |
| Anderson-Darling | 50 | 0.4630 | 0.6212 | 0.7376 | 1.0130 |
| Anderson-Darling | 100 | 0.4662 | 0.6266 | 0.7435 | 1.0206 |
| Anderson-Darling | 500 | 0.4702 | 0.6286 | 0.7462 | 1.0323 |
| Anderson-Darling | 1000 | 0.4702 | 0.6318 | 0.7503 | 1.0285 |

### Key Findings

- **K-S statistics**: Right-skewed, bounded [0, 1], variance decreases with n (critical values scale as ~1/√n)
- **D-W statistics**: Centered near 2.0 with approximate symmetry, bounded [0, 4]
- **A-D statistics**: Heavy right tails, unbounded above, relatively stable across sample sizes
- **Convergence**: All 15 configurations achieved R̂ < 1.05 with quantile stability < 0.0001

### Computational Performance

- Total runtime: ~54 minutes for all configurations
- Throughput: ~47,600 iterations/second
- Storage: ~1.2 GB for full simulation data (HDF5 compressed)

---

## ⚡ Quick Start

```bash
# Clone repository
git clone https://github.com/username/monte_carlo_test_stats.git
cd monte_carlo_test_stats

# Create environment
conda env create -f config/environment.yml
conda activate monte_carlo_test_stats

# Run quick simulation (reduced iterations for testing)
python scripts/run_full_simulations.py --quick

# View results
python scripts/generate_all_plots.py
```

**Expected output** (quick mode, ~2 minutes):
```
Starting production Monte Carlo simulations
Test statistics: ['kolmogorov_smirnov', 'durbin_watson', 'anderson_darling']
Sample sizes: [30, 50, 100, 500, 1000]
...
Configuration 1/15: kolmogorov_smirnov, n=30
  Convergence achieved at 100000 iterations
  Time: 8.42s
...
SIMULATION COMPLETE
  Successful: 15/15
```

---

## 🛠️ Installation

### System Requirements

- Python 3.9 or higher
- 8 GB RAM minimum (16 GB recommended for full simulations)
- Multi-core CPU recommended (parallelization uses all available cores)
- 5 GB free disk space for checkpoints and results

### Environment Setup

**Option 1: Conda (Recommended)**
```bash
conda env create -f config/environment.yml
conda activate monte_carlo_test_stats
```

**Option 2: pip**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21.0 | Array operations, random number generation |
| scipy | ≥1.9.0 | Statistical distributions, CDF calculations |
| statsmodels | ≥0.13.0 | Additional statistical functions |
| pandas | ≥1.5.0 | Data manipulation, table generation |
| h5py | ≥3.7.0 | HDF5 checkpoint storage |
| matplotlib | ≥3.5.0 | Visualization |
| seaborn | ≥0.11.0 | Statistical plots |
| joblib | ≥1.1.0 | Parallel processing |
| tqdm | ≥4.64.0 | Progress bars |
| PyYAML | ≥6.0 | Configuration loading |

### Verification

```bash
python -c "from src.simulation.engine import MonteCarloEngine; print('Installation successful')"
```

---

## 📁 Project Structure

```
monte_carlo_test_stats/
├── config/
│   ├── convergence_params.yaml    # Convergence monitoring settings
│   ├── environment.yml            # Conda environment specification
│   └── simulation_config.yaml     # Main simulation parameters
├── data/
│   ├── interim/                   # Checkpoints during simulation
│   ├── processed/                 # Final simulation results (HDF5)
│   └── validation/                # Validation test outputs
├── logs/                          # Execution logs
├── reports/
│   ├── cross_validation/          # Cross-validation reports
│   ├── figures/                   # Generated plots
│   ├── master_validation/         # Comprehensive validation reports
│   ├── quantile_tables/
│   │   ├── csv/                   # CSV format tables
│   │   ├── latex/                 # LaTeX format tables
│   │   └── markdown/              # Markdown format tables
│   └── type_i_validation/         # Type I error validation
├── scripts/
│   ├── generate_all_plots.py      # Visualization generation
│   ├── generate_master_validation_report.py
│   ├── run_cross_validation.py    # Cross-validation experiments
│   ├── run_full_simulations.py    # Main simulation entry point
│   └── run_type_i_validation.py   # Type I error validation
├── src/
│   ├── analysis/
│   │   ├── quantile_estimation.py # Quantile extraction and tables
│   │   └── validation.py          # Validation utilities
│   ├── simulation/
│   │   ├── checkpoints.py         # Checkpoint save/load
│   │   ├── convergence.py         # Convergence monitoring
│   │   ├── engine.py              # Core Monte Carlo engine
│   │   └── parallel.py            # Parallel RNG management
│   ├── statistics/
│   │   ├── anderson_darling.py    # A-D statistic implementation
│   │   ├── durbin_watson.py       # D-W statistic implementation
│   │   └── kolmogorov_smirnov.py  # K-S statistic implementation
│   └── visualization/
│       └── distribution_plots.py  # Plotting functions
└── requirements.txt
```

---

## 🚀 Usage

### Running Full Simulations

```bash
# Production run (10M iterations per configuration)
python scripts/run_full_simulations.py

# Custom configuration
python scripts/run_full_simulations.py --config config/custom_config.yaml

# Resume from checkpoint
python scripts/run_full_simulations.py --resume
```

### Generating Quantile Tables

```bash
# Generate tables in all formats (CSV, LaTeX, Markdown)
python -c "
from src.analysis.quantile_estimation import QuantileExtractor
extractor = QuantileExtractor()
extractor.process_all_statistics()
"
```

### Creating Visualizations

```bash
# Generate all plots
python scripts/generate_all_plots.py

# Specific plot types
python scripts/generate_all_plots.py --type distribution
python scripts/generate_all_plots.py --type convergence
python scripts/generate_all_plots.py --type comparison
```

### Using as a Library

```python
from src.simulation.engine import MonteCarloEngine
from src.statistics.kolmogorov_smirnov import kolmogorov_smirnov_statistic
import numpy as np

# Initialize engine
engine = MonteCarloEngine(seed=42, n_jobs=-1, verbose=True)

# Run simulation
results = engine.simulate(
    test_statistic='kolmogorov_smirnov',
    n=100,
    iterations=1_000_000
)

# Extract quantiles
quantiles = {q: np.quantile(results, q) for q in [0.90, 0.95, 0.99]}
print(f"Critical values (n=100): {quantiles}")
```

### Checkpoint Recovery

```python
from src.simulation.checkpoints import CheckpointManager

manager = CheckpointManager(checkpoint_dir="data/interim")

# Load latest checkpoint
data, metadata, quantiles = manager.load_latest_checkpoint()
print(f"Loaded {len(data):,} iterations")
print(f"Pre-calculated quantiles: {quantiles}")

# Merge multiple checkpoints
merged_data, merged_quantiles = manager.merge_checkpoints(pattern="ks_*.h5")
```

---

## ⚙️ Configuration

### Simulation Configuration (`config/simulation_config.yaml`)

```yaml
test_statistics:
  - kolmogorov_smirnov
  - durbin_watson
  - anderson_darling

sample_sizes: [30, 50, 100, 500, 1000]

iterations:
  initial: 1_000_000      # Starting iterations
  convergence_check: 5_000_000
  maximum: 10_000_000     # Hard cap

quantiles: [0.75, 0.90, 0.95, 0.99]

null_distribution: standard_normal
random_seed: 42

parallel:
  n_jobs: -1              # -1 = all CPU cores
  backend: loky
  batch_size: 10_000

memory:
  chunk_size: 100_000
  enable_compression: true
```

### Convergence Parameters (`config/convergence_params.yaml`)

```yaml
quantile_stability: 0.0001    # Max change between batches
batch_size: 100_000           # Iterations per convergence check
min_batches: 10               # Minimum before checking
confidence_level: 0.99

checkpoint_interval: 100_000
max_checkpoints: 50
checkpoint_format: hdf5
compression: snappy
```

### Environment Variables

```bash
export MONTE_CARLO_SEED=42           # Override random seed
export MONTE_CARLO_NJOBS=4           # Limit parallel workers
export MONTE_CARLO_DATA_DIR=/path/to/data
```

---

## 🧪 Experiments

### Reproducing Paper Results

```bash
# Run complete simulation suite
python scripts/run_full_simulations.py

# Generate validation report
python scripts/generate_master_validation_report.py

# All outputs saved to reports/
```

### Cross-Validation

```bash
# Split simulations into independent chains
python scripts/run_cross_validation.py --chains 5 --iterations 2_000_000

# Results include:
# - Gelman-Rubin R̂ diagnostic
# - Coefficient of variation across chains
# - Quantile agreement statistics
```

### Type I Error Validation

```bash
# Validate rejection rates at nominal α levels
python scripts/run_type_i_validation.py --samples 10_000

# Checks:
# - α = 0.01: rejection rate ∈ [0.005, 0.015]
# - α = 0.05: rejection rate ∈ [0.045, 0.055]
# - α = 0.10: rejection rate ∈ [0.095, 0.105]
```

### Custom Experiments

```python
# Test alternative sample sizes
from src.simulation.engine import MonteCarloEngine

engine = MonteCarloEngine(seed=42)
custom_sizes = [20, 75, 200, 2000]

for n in custom_sizes:
    results = engine.simulate('kolmogorov_smirnov', n=n, iterations=1_000_000)
    q95 = np.quantile(results, 0.95)
    print(f"n={n}: q95 = {q95:.4f}")
```

---

## 📈 Validation

### Validation Framework

The framework validates results through three mechanisms:

1. **Theoretical Comparison**: Compare empirical quantiles against known asymptotic approximations (where available)
2. **Type I Error Validation**: Generate independent null samples and verify rejection rates match nominal α levels
3. **Cross-Validation**: Split into independent chains and check Gelman-Rubin R̂ convergence diagnostic

### Acceptance Criteria

| Metric | Target | Achieved |
|--------|--------|----------|
| Mean Absolute Error (quantiles) | < 0.002 | ✓ |
| Type I Error Deviation | ±0.005 | ✓ |
| Gelman-Rubin R̂ | < 1.1 | ✓ (all < 1.05) |
| Coefficient of Variation | < 0.01 | ✓ |
| Quantile Stability | < 0.0001 | ✓ |

### Running Validation Suite

```bash
# Full validation
python scripts/generate_master_validation_report.py

# Output: reports/master_validation/validation_report.html
```

### Validation Metrics

```python
from src.analysis.validation import ValidationSuite

validator = ValidationSuite(data_dir="data/processed")

# Run all validation checks
results = validator.run_full_validation()

print(f"Type I Error Check: {'PASS' if results['type_i_passed'] else 'FAIL'}")
print(f"Gelman-Rubin R-hat: {results['gelman_rubin']:.4f}")
print(f"Max Quantile Difference: {results['max_quantile_diff']:.6f}")
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Code Standards

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src/ scripts/
isort src/ scripts/

# Run tests
pytest tests/ -v

# Type checking
mypy src/
```

### Contribution Areas

- **New test statistics**: Implement in `src/statistics/` following existing patterns
- **Additional sample sizes**: Extend configuration and run simulations
- **Alternative null distributions**: Modify `MonteCarloEngine` to support non-normal nulls
- **Performance optimizations**: Improve parallel efficiency or memory usage
- **Documentation**: Improve docstrings, add tutorials, fix typos

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-statistic`)
3. Add tests for new functionality
4. Ensure all tests pass and code is formatted
5. Submit PR with description of changes

---

## 📚 Citation

If you use this code or the critical value tables in your research, please cite:

```bibtex
@misc{lenaghan2024montecarlo,
  title={Monte Carlo Test Statistic Distributions: Kolmogorov-Smirnov, 
         Durbin-Watson, Anderson-Darling},
  author={Lenaghan, Terrence P.},
  year={2024},
  note={Project Group 293}
}
```

### Related Work

- Banack, H. R., et al. (2021). Monte Carlo simulation approaches for quantitative bias analysis. *Epidemiologic Reviews*, 43(1), 106-117.
- Cowan, G. (2024). Monte Carlo techniques. In *Review of Particle Physics*. Particle Data Group.
- van Kesteren, E.-J. (2025). Tidy simulation: Designing robust, reproducible, and scalable Monte Carlo simulations. *arXiv:2509.11741*.

### Acknowledgments

- NumPy and SciPy communities for foundational statistical computing tools
- Joblib developers for parallel processing infrastructure

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

### Data Attribution

- Simulation methodology follows established Monte Carlo practices from statistical literature
- Test statistic formulations are standard implementations from scipy.stats where applicable
- Generated critical value tables are released under the same MIT license for unrestricted use

---

## Appendix: Mathematical Formulations

### Kolmogorov-Smirnov Statistic

```
D_n = max(D⁺, D⁻)

where:
  D⁺ = max_i [F̂_n(x_i) - F(x_i)]
  D⁻ = max_i [F(x_i) - F̂_n(x_{i-1})]
  F̂_n = empirical CDF
  F = theoretical CDF (standard normal)
```

### Durbin-Watson Statistic

```
DW = Σ_{t=2}^{n} (e_t - e_{t-1})² / Σ_{t=1}^{n} e_t²

Range: [0, 4]
  DW ≈ 2: no autocorrelation
  DW < 2: positive autocorrelation
  DW > 2: negative autocorrelation
```

### Anderson-Darling Statistic

```
A² = -n - (1/n) Σ_{i=1}^{n} (2i-1)[ln(Φ(z_i)) + ln(1-Φ(z_{n+1-i}))]

where:
  z_i = (x_{(i)} - x̄) / s (standardized order statistics)
  Φ = standard normal CDF
```
