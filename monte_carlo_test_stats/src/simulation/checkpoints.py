"""
Checkpoint system for saving/loading simulation state with pre-calculated quantiles.
"""
import h5py
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint saving and loading for simulations."""
    
    def __init__(
        self,
        checkpoint_dir: str = "data/interim",
        max_checkpoints: int = 50,
        compression: str = "gzip",
        quantile_levels: List[float] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            compression: Compression type for HDF5
            quantile_levels: Quantile levels to pre-calculate (default: [0.75, 0.90, 0.95, 0.99])
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.compression = compression
        self.quantile_levels = quantile_levels or [0.75, 0.90, 0.95, 0.99]
    
    def save_checkpoint(
        self,
        data: np.ndarray,
        metadata: Dict[str, Any],
        name_prefix: str = "checkpoint"
    ) -> str:
        """
        Save simulation data to checkpoint with pre-calculated quantiles.
        
        Args:
            data: Simulation results array
            metadata: Dictionary of metadata
            name_prefix: Prefix for checkpoint filename
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name_prefix}_{timestamp}.h5"
        filepath = self.checkpoint_dir / filename
        
        # Calculate quantiles from the data
        quantiles_dict = self._calculate_quantiles(data)
        
        with h5py.File(filepath, 'w') as f:
            # Save data with compression
            f.create_dataset(
                'data',
                data=data,
                compression=self.compression,
                compression_opts=4
            )
            
            # Save metadata as attributes
            f.attrs['metadata'] = json.dumps(metadata)
            f.attrs['timestamp'] = timestamp
            f.attrs['shape'] = data.shape
            
            # Save pre-calculated quantiles as JSON string
            f.attrs['last_quantiles'] = json.dumps(quantiles_dict)
            f.attrs['quantile_levels'] = json.dumps(self.quantile_levels)
        
        logger.info(f"Checkpoint saved: {filepath}")
        logger.debug(f"Pre-calculated quantiles: {quantiles_dict}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints(name_prefix)
        
        return str(filepath)
    
    def _calculate_quantiles(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate quantiles from data array.
        
        Args:
            data: Simulation results array
            
        Returns:
            Dictionary mapping quantile level strings to values
        """
        quantiles_dict = {}
        
        for level in self.quantile_levels:
            # Calculate quantile value
            q_value = np.quantile(data, level)
            # Store with string key for JSON serialization
            quantiles_dict[f"{level:.2f}"] = float(q_value)
        
        return quantiles_dict
    
    def load_latest_checkpoint(self) -> Tuple[np.ndarray, Dict[str, Any], Optional[Dict[str, float]]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Tuple of (data array, metadata dict, quantiles dict)
        """
        checkpoints = sorted(self.checkpoint_dir.glob("*.h5"))
        
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
        
        latest = checkpoints[-1]
        return self.load_checkpoint(str(latest))
    
    def load_checkpoint(self, filepath: str) -> Tuple[np.ndarray, Dict[str, Any], Optional[Dict[str, float]]]:
        """
        Load specific checkpoint file with quantiles.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Tuple of (data array, metadata dict, quantiles dict)
        """
        with h5py.File(filepath, 'r') as f:
            data = f['data'][:]
            metadata = json.loads(f.attrs['metadata'])
            
            # Load pre-calculated quantiles if available
            quantiles = None
            if 'last_quantiles' in f.attrs:
                quantiles = json.loads(f.attrs['last_quantiles'])
        
        logger.info(f"Checkpoint loaded: {filepath}")
        if quantiles:
            logger.debug(f"Loaded pre-calculated quantiles: {quantiles}")
        
        return data, metadata, quantiles
    
    def get_quantiles_from_checkpoint(self, filepath: str) -> Optional[Dict[str, float]]:
        """
        Extract only the quantiles from a checkpoint without loading data.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Dictionary of quantiles or None if not available
        """
        try:
            with h5py.File(filepath, 'r') as f:
                if 'last_quantiles' in f.attrs:
                    return json.loads(f.attrs['last_quantiles'])
        except Exception as e:
            logger.error(f"Error reading quantiles from {filepath}: {e}")
        
        return None
    
    def _cleanup_old_checkpoints(self, name_prefix: str):
        """Remove old checkpoints keeping only max_checkpoints."""
        pattern = f"{name_prefix}_*.h5"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        
        if len(checkpoints) > self.max_checkpoints:
            to_remove = checkpoints[:-self.max_checkpoints]
            for checkpoint in to_remove:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")
    
    def merge_checkpoints(
        self,
        pattern: str = "*.h5",
        recalculate_quantiles: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
        """
        Merge data from multiple checkpoints.
        
        Args:
            pattern: Glob pattern for checkpoint files
            recalculate_quantiles: Whether to calculate quantiles for merged data
            
        Returns:
            Tuple of (concatenated data array, quantiles dict if calculated)
        """
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            raise FileNotFoundError("No checkpoints to merge")
        
        all_data = []
        for checkpoint in checkpoints:
            data, _, _ = self.load_checkpoint(str(checkpoint))
            all_data.append(data)
        
        merged = np.concatenate(all_data)
        logger.info(f"Merged {len(checkpoints)} checkpoints, total size: {len(merged)}")
        
        # Calculate quantiles for merged data if requested
        quantiles = None
        if recalculate_quantiles:
            quantiles = self._calculate_quantiles(merged)
            logger.debug(f"Calculated quantiles for merged data: {quantiles}")
        
        return merged, quantiles
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata and quantiles."""
        checkpoints = []
        
        for filepath in sorted(self.checkpoint_dir.glob("*.h5")):
            try:
                with h5py.File(filepath, 'r') as f:
                    info = {
                        'path': str(filepath),
                        'timestamp': f.attrs.get('timestamp', 'unknown'),
                        'shape': f.attrs.get('shape', []),
                        'size_mb': filepath.stat().st_size / (1024 * 1024),
                        'has_quantiles': 'last_quantiles' in f.attrs
                    }
                    
                    # Include quantiles if available
                    if 'last_quantiles' in f.attrs:
                        info['quantiles'] = json.loads(f.attrs['last_quantiles'])
                    
                    checkpoints.append(info)
            except:
                logger.warning(f"Could not read checkpoint: {filepath}")
        
        return checkpoints
    
    def update_checkpoint_quantiles(self, filepath: str) -> Dict[str, float]:
        """
        Update an existing checkpoint to add quantiles if missing.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Dictionary of calculated quantiles
        """
        # Load the data
        with h5py.File(filepath, 'r') as f:
            data = f['data'][:]
            has_quantiles = 'last_quantiles' in f.attrs
        
        if has_quantiles:
            logger.info(f"Checkpoint already has quantiles: {filepath}")
            return self.get_quantiles_from_checkpoint(filepath)
        
        # Calculate quantiles
        quantiles_dict = self._calculate_quantiles(data)
        
        # Update the checkpoint file
        with h5py.File(filepath, 'r+') as f:
            f.attrs['last_quantiles'] = json.dumps(quantiles_dict)
            f.attrs['quantile_levels'] = json.dumps(self.quantile_levels)
        
        logger.info(f"Updated checkpoint with quantiles: {filepath}")
        return quantiles_dict


# Example usage for convergence plotting
def get_quantiles_for_convergence_plot(checkpoint_manager: CheckpointManager, checkpoint_path: str) -> Dict[str, float]:
    """
    Helper function to get quantiles for convergence plotting.
    
    Args:
        checkpoint_manager: CheckpointManager instance
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary of quantiles ready for plotting
    """
    # Try to get pre-calculated quantiles first
    quantiles = checkpoint_manager.get_quantiles_from_checkpoint(checkpoint_path)
    
    if quantiles is None:
        # If not available, load data and calculate
        logger.info("Quantiles not found in checkpoint, calculating...")
        data, _, _ = checkpoint_manager.load_checkpoint(checkpoint_path)
        quantiles = checkpoint_manager._calculate_quantiles(data)
        
        # Optionally update the checkpoint with quantiles
        checkpoint_manager.update_checkpoint_quantiles(checkpoint_path)
    
    return quantiles


if __name__ == "__main__":
    # Example demonstration
    import tempfile
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir=tmpdir,
            quantile_levels=[0.75, 0.90, 0.95, 0.99]
        )
        
        # Generate sample data
        np.random.seed(42)
        sample_data = np.random.normal(0, 1, 10000)
        
        # Save checkpoint with quantiles
        metadata = {
            'test_type': 'ks',
            'n_iterations': 10000,
            'sample_size': 100
        }
        
        checkpoint_path = manager.save_checkpoint(
            data=sample_data,
            metadata=metadata,
            name_prefix='test_checkpoint'
        )
        
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Load checkpoint with quantiles
        data, meta, quantiles = manager.load_checkpoint(checkpoint_path)
        
        print(f"Loaded data shape: {data.shape}")
        print(f"Loaded metadata: {meta}")
        print(f"Pre-calculated quantiles: {quantiles}")
        
        # Demonstrate getting quantiles without loading data
        quantiles_only = manager.get_quantiles_from_checkpoint(checkpoint_path)
        print(f"Quantiles retrieved without loading data: {quantiles_only}")
        
        # List checkpoints with quantile info
        checkpoint_list = manager.list_checkpoints()
        for cp in checkpoint_list:
            print(f"\nCheckpoint info:")
            for key, value in cp.items():
                print(f"  {key}: {value}")
