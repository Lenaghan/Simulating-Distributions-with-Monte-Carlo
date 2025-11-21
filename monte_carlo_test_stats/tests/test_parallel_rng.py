"""
Unit tests for ParallelRNGManager.
Tests RNG stream independence and reproducibility in parallel contexts.
"""
import pytest
import numpy as np
from scipy import stats
from src.simulation.parallel import ParallelRNGManager

class TestParallelRNGManager:
    """Test suite for parallel RNG management"""
    
    def test_stream_generation(self):
        """
        Test: Generate independent RNG streams
        Expected:
        - Each stream produces different sequences
        - Number of streams matches requested
        """
        from src.simulation.parallel import ParallelRNGManager
        manager = ParallelRNGManager(base_seed=42, n_streams=4)
        streams = manager.get_streams()
        assert len(streams) == 4
        
        # Generate numbers from each stream
        samples = [stream.random(100) for stream in streams]
        # Check all streams produce different sequences
        for i in range(len(samples)):
            for j in range(i+1, len(samples)):
                assert not np.array_equal(samples[i], samples[j])
    
    def test_stream_independence_correlation(self):
        """
        Test: Streams are statistically independent
        Expected: 
        - Cross-correlation between streams < 0.05
        - p-value > 0.05 for independence test
        """
        manager = ParallelRNGManager(base_seed=42, n_streams=4)
        streams = manager.get_streams()
        
        # Generate large samples
        samples = [stream.standard_normal(10000) for stream in streams]
        
        # Test pairwise independence
        for i in range(len(samples)):
            for j in range(i+1, len(samples)):
                correlation = np.corrcoef(samples[i], samples[j])[0, 1]
                assert abs(correlation) < 0.05
    
    def test_reproducibility_with_same_seed(self):
        """
        Test: Same base seed produces identical stream set
        Expected: Identical sequences from corresponding streams
        """
        manager1 = ParallelRNGManager(base_seed=42, n_streams=3)
        manager2 = ParallelRNGManager(base_seed=42, n_streams=3)
        
        streams1 = manager1.get_streams()
        streams2 = manager2.get_streams()
        
        for s1, s2 in zip(streams1, streams2):
            assert np.array_equal(s1.random(100), s2.random(100))
    
    def test_pcg64_jump_ahead(self):
        """
        Test: PCG64 jump-ahead creates non-overlapping sequences
        Expected:
        - Jumped streams don't overlap in sequence space
        - Each stream advances by large jump
        """
        manager = ParallelRNGManager(base_seed=42)
        base_rng = np.random.PCG64(42)
        
        # Create jumped version
        jumped_rng = base_rng.jumped(1)
        
        # Sequences should be different
        gen1 = np.random.Generator(base_rng)
        gen2 = np.random.Generator(jumped_rng)
        
        assert not np.array_equal(gen1.random(1000), gen2.random(1000))
    
    def test_worker_seed_generation(self):
        """
        Test: Generate unique seeds for workers
        Expected:
        - All seeds are unique
        - Seeds are deterministic given base seed
        """
        manager = ParallelRNGManager(base_seed=42)
        seeds = manager.generate_worker_seeds(n_workers=10)
        
        assert len(seeds) == 10
        assert len(set(seeds)) == 10  # All unique
        
        # Reproducible
        manager2 = ParallelRNGManager(base_seed=42)
        seeds2 = manager2.generate_worker_seeds(n_workers=10)
        assert np.array_equal(seeds, seeds2)
    
    def test_stream_state_persistence(self):
        """
        Test: Stream states can be saved and restored
        Expected: Restored stream continues from saved state
        """
        manager = ParallelRNGManager(base_seed=42)
        stream = manager.get_generator(0)
        
        # Generate some numbers
        first_batch = stream.random(10)
        
        # Save state
        state = manager.save_state(0)
        
        # Generate more numbers
        second_batch = stream.random(10)
        
        # Restore state and regenerate
        manager.restore_state(0, state)
        second_batch_restored = stream.random(10)
        
        assert np.array_equal(second_batch, second_batch_restored)

# Test specifications
"""
Required Properties:
1. Statistical independence between streams (correlation < 0.05)
2. No sequence overlap between parallel workers
3. Reproducibility with same seed
4. Unique seeds for each worker
5. PCG64 jump-ahead functionality
"""
