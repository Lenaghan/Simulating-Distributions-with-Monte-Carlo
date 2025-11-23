#!/usr/bin/env python
"""
Test script to verify YAML configuration loading.
Run this to check if your YAML files are being loaded correctly.
"""
import yaml
import json
from pathlib import Path
import sys


def test_yaml_loading():
    """Test loading and parsing of YAML configuration files."""
    
    print("=" * 60)
    print("YAML Configuration Test")
    print("=" * 60)
    
    config_dir = Path("config")
    
    # Test simulation_config.yaml
    print("\n1. Testing simulation_config.yaml...")
    sim_config_path = config_dir / "simulation_config.yaml"
    
    if not sim_config_path.exists():
        print(f"   [FAIL] File not found: {sim_config_path}")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Looking in: {sim_config_path.absolute()}")
    else:
        try:
            with open(sim_config_path, 'r') as f:
                sim_config = yaml.safe_load(f)
            print(f"   [OK] Successfully loaded simulation_config.yaml")
            print(f"   Test statistics: {sim_config.get('test_statistics', 'NOT FOUND')}")
            print(f"   Sample sizes: {sim_config.get('sample_sizes', 'NOT FOUND')}")
            print(f"   Max iterations: {sim_config.get('iterations', {}).get('maximum', 'NOT FOUND')}")
            print(f"   Quantiles: {sim_config.get('quantiles', 'NOT FOUND')}")
            print(f"   Parallel jobs: {sim_config.get('parallel', {}).get('n_jobs', 'NOT FOUND')}")
        except Exception as e:
            print(f"   [FAIL] Error loading file: {e}")
    
    # Test convergence_params.yaml
    print("\n2. Testing convergence_params.yaml...")
    conv_config_path = config_dir / "convergence_params.yaml"
    
    if not conv_config_path.exists():
        print(f"   [FAIL] File not found: {conv_config_path}")
        print(f"   Looking in: {conv_config_path.absolute()}")
    else:
        try:
            with open(conv_config_path, 'r') as f:
                conv_config = yaml.safe_load(f)
            print(f"   [OK] Successfully loaded convergence_params.yaml")
            print(f"   Quantile stability: {conv_config.get('quantile_stability', 'NOT FOUND')}")
            print(f"   Batch size: {conv_config.get('batch_size', 'NOT FOUND')}")
            print(f"   Checkpoint interval: {conv_config.get('checkpoint_interval', 'NOT FOUND')}")
            print(f"   Max checkpoints: {conv_config.get('max_checkpoints', 'NOT FOUND')}")
            print(f"   Compression: {conv_config.get('compression', 'NOT FOUND')}")
        except Exception as e:
            print(f"   [FAIL] Error loading file: {e}")
    
    # Test parameter grid generation
    print("\n3. Testing parameter grid generation...")
    if sim_config_path.exists():
        try:
            with open(sim_config_path, 'r') as f:
                sim_config = yaml.safe_load(f)
            
            statistics = sim_config.get('test_statistics', [])
            sample_sizes = sim_config.get('sample_sizes', [])
            total_configs = len(statistics) * len(sample_sizes)
            
            print(f"   Statistics: {len(statistics)}")
            print(f"   Sample sizes: {len(sample_sizes)}")
            print(f"   Total configurations: {total_configs}")
            
            if total_configs > 0:
                print(f"   [OK] Will generate {total_configs} simulation configurations")
            else:
                print(f"   [FAIL] No configurations will be generated")
        except Exception as e:
            print(f"   [FAIL] Error: {e}")
    
    # Test imports
    print("\n4. Testing required imports...")
    imports_ok = True
    
    required_modules = ['yaml', 'numpy', 'h5py', 'pandas']
    for module in required_modules:
        try:
            __import__(module)
            print(f"   [OK] {module} imported successfully")
        except ImportError:
            print(f"   [FAIL] Failed to import {module}")
            imports_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_ok = True
    if not sim_config_path.exists():
        print("[FAIL] simulation_config.yaml not found")
        all_ok = False
    else:
        print("[OK] simulation_config.yaml found and loaded")
    
    if not conv_config_path.exists():
        print("[FAIL] convergence_params.yaml not found")
        all_ok = False
    else:
        print("[OK] convergence_params.yaml found and loaded")
    
    if not imports_ok:
        print("[FAIL] Some required modules are missing")
        all_ok = False
    else:
        print("[OK] All required modules available")
    
    if all_ok:
        print("\n[OK] All tests passed! Your YAML configuration is ready to use.")
    else:
        print("\n[FAIL] Some issues found. Please check the messages above.")

if __name__ == "__main__":
    success = test_yaml_loading()
    sys.exit(0 if success else 1)
