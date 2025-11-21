"""
Parallel Random Number Generation Management
Ensures independent, reproducible random streams for parallel workers.
"""
import numpy as np
from typing import List, Optional, Dict, Any


class ParallelRNGManager:
    """
    Manages independent random number generators for parallel processing.
    Uses PCG64's jump-ahead functionality to ensure non-overlapping sequences.
    """
    
    def __init__(self, base_seed: int = 42, n_streams: Optional[int] = None):
        """
        Initialize RNG manager with base seed.
        
        Args:
            base_seed: Base random seed for reproducibility
            n_streams: Number of parallel streams to prepare
        """
        self.base_seed = base_seed
        self.base_rng = np.random.PCG64(base_seed)
        self.n_streams = n_streams
        self._streams = {}
        self._generators = {}
        self._states = {}
        
        if n_streams is not None:
            self._initialize_streams(n_streams)
    
    def _initialize_streams(self, n_streams: int):
        """Initialize the specified number of RNG streams."""
        # Use jump-ahead to create independent streams
        for i in range(n_streams):
            # Each stream jumps ahead by 2^128 steps to ensure independence
            stream_rng = np.random.PCG64(self.base_seed).jumped(i)
            self._streams[i] = stream_rng
            self._generators[i] = np.random.Generator(stream_rng)
    
    def get_streams(self) -> List[np.random.Generator]:
        """
        Get list of independent RNG streams.
        
        Returns:
            List of numpy Generator objects
        """
        if not self._generators:
            raise RuntimeError("No streams initialized. Call with n_streams first.")
        
        return [self._generators[i] for i in range(self.n_streams)]
    
    def get_generator(self, worker_id: int) -> np.random.Generator:
        """
        Get RNG generator for specific worker.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            Independent numpy Generator for this worker
        """
        if worker_id not in self._generators:
            # Create on-demand if not exists
            stream_rng = np.random.PCG64(self.base_seed).jumped(worker_id)
            self._streams[worker_id] = stream_rng
            self._generators[worker_id] = np.random.Generator(stream_rng)
        
        return self._generators[worker_id]
    
    def generate_worker_seeds(self, n_workers: int) -> np.ndarray:
        """
        Generate unique seeds for each worker.
        
        Args:
            n_workers: Number of workers
            
        Returns:
            Array of unique seeds
        """
        # Use base RNG to generate worker seeds deterministically
        seed_gen = np.random.PCG64(self.base_seed)
        seed_generator = np.random.Generator(seed_gen)
        
        # Generate raw random integers for seeds
        seeds = seed_generator.integers(0, 2**63, size=n_workers)
        
        return seeds
    
    def create_independent_generator(self, worker_id: int) -> np.random.Generator:
        """
        Create an independent generator using jump-ahead.
        
        Args:
            worker_id: Unique identifier for the worker
            
        Returns:
            Independent Generator instance
        """
        # Create new PCG64 instance and jump ahead
        rng = np.random.PCG64(self.base_seed)
        # Jump by worker_id * 2^128 steps
        jumped_rng = rng.jumped(worker_id)
        
        return np.random.Generator(jumped_rng)
    
    def save_state(self, stream_id: int) -> Dict[str, Any]:
        """
        Save the state of a specific stream.
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            State dictionary
        """
        if stream_id not in self._streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        # PCG64 state includes the internal state and position
        state = self._streams[stream_id].state
        self._states[stream_id] = state
        
        return state
    
    def restore_state(self, stream_id: int, state: Dict[str, Any]):
        """
        Restore a stream to a previously saved state.
        
        Args:
            stream_id: ID of the stream
            state: Previously saved state dictionary
        """
        if stream_id not in self._streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        self._streams[stream_id].state = state
        # Recreate generator with restored state
        self._generators[stream_id] = np.random.Generator(self._streams[stream_id])
    
    def verify_independence(self, n_samples: int = 10000, threshold: float = 0.05) -> bool:
        """
        Verify statistical independence of streams.
        
        Args:
            n_samples: Number of samples to generate for testing
            threshold: Maximum acceptable correlation
            
        Returns:
            True if streams are independent
        """
        if self.n_streams is None or self.n_streams < 2:
            return True
        
        # Generate samples from each stream
        samples = []
        for i in range(min(self.n_streams, 5)):  # Test first 5 streams
            gen = self.get_generator(i)
            samples.append(gen.standard_normal(n_samples))
        
        # Check pairwise correlations
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                corr = np.corrcoef(samples[i], samples[j])[0, 1]
                if abs(corr) > threshold:
                    return False
        
        return True


def create_parallel_seeds(base_seed: int, n_workers: int, method: str = "jump") -> List[Any]:
    """
    Create seeds for parallel workers using specified method.
    
    Args:
        base_seed: Base seed for reproducibility
        n_workers: Number of parallel workers
        method: Method for seed generation ('jump' or 'random')
        
    Returns:
        List of RNG objects or seeds
    """
    if method == "jump":
        # Use PCG64 jump-ahead
        base_rng = np.random.PCG64(base_seed)
        rngs = [base_rng.jumped(i) for i in range(n_workers)]
        return rngs
    
    elif method == "random":
        # Generate random seeds
        rng = np.random.PCG64(base_seed)
        generator = np.random.Generator(rng)
        seeds = generator.integers(0, 2**63, size=n_workers)
        return seeds.tolist()
    
    else:
        raise ValueError(f"Unknown method: {method}")
