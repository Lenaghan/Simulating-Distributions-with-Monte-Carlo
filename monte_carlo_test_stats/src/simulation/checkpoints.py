"""
Checkpoint system for saving/loading simulation state.
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
        max_checkpoints: int = 3,
        compression: str = "gzip"
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            compression: Compression type for HDF5
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.compression = compression
    
    def save_checkpoint(
        self,
        data: np.ndarray,
        metadata: Dict[str, Any],
        name_prefix: str = "checkpoint"
    ) -> str:
        """
        Save simulation data to checkpoint.
        
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
        
        logger.info(f"Checkpoint saved: {filepath}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints(name_prefix)
        
        return str(filepath)
    
    def load_latest_checkpoint(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Tuple of (data array, metadata dict)
        """
        checkpoints = sorted(self.checkpoint_dir.glob("*.h5"))
        
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
        
        latest = checkpoints[-1]
        return self.load_checkpoint(str(latest))
    
    def load_checkpoint(self, filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load specific checkpoint file.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Tuple of (data array, metadata dict)
        """
        with h5py.File(filepath, 'r') as f:
            data = f['data'][:]
            metadata = json.loads(f.attrs['metadata'])
        
        logger.info(f"Checkpoint loaded: {filepath}")
        return data, metadata
    
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
        pattern: str = "*.h5"
    ) -> np.ndarray:
        """
        Merge data from multiple checkpoints.
        
        Args:
            pattern: Glob pattern for checkpoint files
            
        Returns:
            Concatenated data array
        """
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            raise FileNotFoundError("No checkpoints to merge")
        
        all_data = []
        for checkpoint in checkpoints:
            data, _ = self.load_checkpoint(str(checkpoint))
            all_data.append(data)
        
        merged = np.concatenate(all_data)
        logger.info(f"Merged {len(checkpoints)} checkpoints, total size: {len(merged)}")
        
        return merged
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []
        
        for filepath in sorted(self.checkpoint_dir.glob("*.h5")):
            try:
                with h5py.File(filepath, 'r') as f:
                    info = {
                        'path': str(filepath),
                        'timestamp': f.attrs.get('timestamp', 'unknown'),
                        'shape': f.attrs.get('shape', []),
                        'size_mb': filepath.stat().st_size / (1024 * 1024)
                    }
                    checkpoints.append(info)
            except:
                logger.warning(f"Could not read checkpoint: {filepath}")
        
        return checkpoints
