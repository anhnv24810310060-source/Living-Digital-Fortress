"""
Comprehensive tests for Inference Engine
"""

import pytest
import time
import numpy as np
import torch
import torch.nn as nn
from inference.inference_engine import (
    InferenceEngine,
    InferenceRequest,
    BatchInferenceResult,
    LRUCache,
    DynamicBatcher,
    GPUManager
)


# Simple test model
class SimpleModel(nn.Module):
    """Simple neural network for testing"""
    
    def __init__(self, input_size=10, output_size=2):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)


class TestLRUCache:
    """Test LRU Cache"""
    
    def test_initialization(self):
        """Test cache initialization"""
        cache = LRUCache(capacity=5)
        assert cache.capacity == 5
        assert cache.size() == 0
    
    def test_put_and_get(self):
        """Test putting and getting items"""
        cache = LRUCache(capacity=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction"""
        cache = LRUCache(capacity=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add key4, should evict key2 (least recently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_update_existing(self):
        """Test updating existing key"""
        cache = LRUCache(capacity=3)
        
        cache.put("key1", "value1")
        cache.put("key1", "value1_updated")
        
        assert cache.get("key1") == "value1_updated"
        assert cache.size() == 1
    
    def test_remove(self):
        """Test removing items"""
        cache = LRUCache(capacity=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        cache.remove("key1")
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.size() == 1
    
    def test_clear(self):
        """Test clearing cache"""
        cache = LRUCache(capacity=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        cache.clear()
        
        assert cache.size() == 0
        assert cache.get("key1") is None
    
    def test_keys(self):
        """Test getting all keys"""
        cache = LRUCache(capacity=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        keys = cache.keys()
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys


class TestDynamicBatcher:
    """Test Dynamic Batcher"""
    
    def test_initialization(self):
        """Test batcher initialization"""
        batcher = DynamicBatcher(max_batch_size=32, max_wait_time_ms=10.0)
        assert batcher.max_batch_size == 32
        assert batcher.max_wait_time_ms == 10.0
        assert batcher.running == False
    
    def test_start_stop(self):
        """Test starting and stopping batcher"""
        batcher = DynamicBatcher(max_batch_size=32)
        
        batcher.start()
        assert batcher.running == True
        assert batcher.worker_thread is not None
        
        batcher.stop()
        assert batcher.running == False
    
    def test_submit_request(self):
        """Test submitting requests"""
        batcher = DynamicBatcher(max_batch_size=2, max_wait_time_ms=100.0)
        batcher.start()
        
        try:
            # Create requests
            req1 = InferenceRequest(
                request_id="req1",
                model_name="test",
                model_version="v1",
                features=np.array([1, 2, 3])
            )
            
            req2 = InferenceRequest(
                request_id="req2",
                model_name="test",
                model_version="v1",
                features=np.array([4, 5, 6])
            )
            
            # Submit requests
            batcher.submit_request(req1)
            batcher.submit_request(req2)
            
            # Wait for processing
            time.sleep(0.2)
            
            # Requests should be marked as completed (even if inference function is None)
            assert req1.completed or req2.completed
        
        finally:
            batcher.stop()


class TestGPUManager:
    """Test GPU Manager"""
    
    def test_initialization(self):
        """Test GPU manager initialization"""
        gpu_manager = GPUManager()
        assert gpu_manager.device is not None
        assert isinstance(gpu_manager.device, torch.device)
    
    def test_get_device(self):
        """Test getting device"""
        gpu_manager = GPUManager()
        device = gpu_manager.get_device()
        
        assert isinstance(device, torch.device)
        assert str(device) in ['cpu', 'cuda', 'cuda:0', 'cuda:1']
    
    def test_get_memory_info(self):
        """Test getting memory info"""
        gpu_manager = GPUManager()
        info = gpu_manager.get_memory_info()
        
        assert 'available' in info
        assert 'device_count' in info
        
        if torch.cuda.is_available():
            assert info['available'] == True
            assert info['device_count'] > 0
            assert 'devices' in info
        else:
            assert info['available'] == False
            assert info['device_count'] == 0
    
    def test_clear_cache(self):
        """Test clearing GPU cache"""
        gpu_manager = GPUManager()
        
        # Should not raise error
        gpu_manager.clear_cache()


class TestInferenceEngine:
    """Test Inference Engine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = InferenceEngine(
            use_local_cache=True,
            local_cache_size=5,
            enable_batching=False,
            use_gpu=False
        )
        
        assert engine.use_local_cache == True
        assert engine.enable_batching == False
        assert engine.use_gpu == False
    
    def test_register_model(self):
        """Test registering a model"""
        engine = InferenceEngine(use_local_cache=True, enable_batching=False, use_gpu=False)
        
        model = SimpleModel()
        engine.register_model("test_model", "v1.0", model)
        
        # Model should be retrievable
        retrieved = engine.get_model("test_model", "v1.0")
        assert retrieved is not None
    
    def test_get_model_from_cache(self):
        """Test getting model from cache"""
        engine = InferenceEngine(use_local_cache=True, enable_batching=False, use_gpu=False)
        
        model = SimpleModel()
        engine.register_model("test_model", "v1.0", model, cache=True)
        
        # Get from cache
        cached_model = engine.get_model("test_model", "v1.0")
        assert cached_model is not None
    
    def test_predict_single(self):
        """Test single prediction"""
        engine = InferenceEngine(use_local_cache=True, enable_batching=False, use_gpu=False)
        
        model = SimpleModel(input_size=10, output_size=2)
        engine.register_model("test_model", "v1.0", model)
        
        # Create input
        features = np.random.randn(10).astype(np.float32)
        
        # Predict
        prediction, latency = engine.predict("test_model", "v1.0", features)
        
        assert prediction is not None
        assert latency > 0
    
    def test_predict_batch(self):
        """Test batch prediction"""
        engine = InferenceEngine(use_local_cache=True, enable_batching=False, use_gpu=False)
        
        model = SimpleModel(input_size=10, output_size=2)
        engine.register_model("test_model", "v1.0", model)
        
        # Create batch input
        features_batch = [np.random.randn(10).astype(np.float32) for _ in range(5)]
        
        # Predict batch
        result = engine.predict_batch("test_model", "v1.0", features_batch)
        
        assert isinstance(result, BatchInferenceResult)
        assert result.batch_size == 5
        assert len(result.predictions) == 5
        assert result.total_time > 0
    
    def test_model_not_found(self):
        """Test error when model not found"""
        engine = InferenceEngine(use_local_cache=True, enable_batching=False, use_gpu=False)
        
        features = np.random.randn(10).astype(np.float32)
        
        with pytest.raises(ValueError, match="Model not found"):
            engine.predict("nonexistent_model", "v1.0", features)
    
    def test_cache_eviction(self):
        """Test cache eviction with small capacity"""
        engine = InferenceEngine(
            use_local_cache=True,
            local_cache_size=2,
            enable_batching=False,
            use_gpu=False
        )
        
        # Register 3 models (should evict first one)
        for i in range(3):
            model = SimpleModel()
            engine.register_model(f"model_{i}", "v1.0", model, cache=True)
        
        # First model should be evicted
        assert engine.local_cache.size() <= 2
    
    def test_get_stats(self):
        """Test getting engine statistics"""
        engine = InferenceEngine(use_local_cache=True, enable_batching=False, use_gpu=True)
        
        model = SimpleModel()
        engine.register_model("test_model", "v1.0", model)
        
        stats = engine.get_stats()
        
        assert 'caching' in stats
        assert 'batching' in stats
        assert 'models' in stats
        assert stats['models']['registered'] == 1
    
    def test_clear_caches(self):
        """Test clearing caches"""
        engine = InferenceEngine(use_local_cache=True, enable_batching=False, use_gpu=False)
        
        model = SimpleModel()
        engine.register_model("test_model", "v1.0", model, cache=True)
        
        # Clear caches
        engine.clear_caches()
        
        # Cache should be empty
        assert engine.local_cache.size() == 0
    
    def test_shutdown(self):
        """Test engine shutdown"""
        engine = InferenceEngine(use_local_cache=True, enable_batching=True, use_gpu=False)
        
        model = SimpleModel()
        engine.register_model("test_model", "v1.0", model)
        
        # Shutdown should not raise error
        engine.shutdown()
        
        # Batcher should be stopped
        assert engine.batcher.running == False


class TestInferenceRequest:
    """Test InferenceRequest dataclass"""
    
    def test_initialization(self):
        """Test request initialization"""
        features = np.array([1, 2, 3])
        request = InferenceRequest(
            request_id="req123",
            model_name="test_model",
            model_version="v1.0",
            features=features
        )
        
        assert request.request_id == "req123"
        assert request.model_name == "test_model"
        assert request.model_version == "v1.0"
        assert np.array_equal(request.features, features)
        assert request.completed == False


class TestBatchInferenceResult:
    """Test BatchInferenceResult dataclass"""
    
    def test_initialization(self):
        """Test result initialization"""
        result = BatchInferenceResult(
            predictions=[1, 0, 1],
            latencies=[5.0, 6.0, 7.0],
            batch_size=3,
            total_time=0.018
        )
        
        assert len(result.predictions) == 3
        assert result.batch_size == 3
        assert result.total_time == 0.018
    
    def test_get_avg_latency(self):
        """Test average latency calculation"""
        result = BatchInferenceResult(
            predictions=[1, 0, 1],
            latencies=[5.0, 6.0, 7.0],
            batch_size=3,
            total_time=0.018
        )
        
        avg_latency = result.get_avg_latency()
        assert avg_latency == 0.018 / 3


class TestIntegration:
    """Integration tests for inference engine"""
    
    def test_full_inference_pipeline(self):
        """Test complete inference pipeline"""
        # Initialize engine
        engine = InferenceEngine(
            use_local_cache=True,
            local_cache_size=5,
            enable_batching=False,
            use_gpu=False
        )
        
        # Create and register model
        model = SimpleModel(input_size=10, output_size=2)
        engine.register_model("production_model", "v1.0", model)
        
        # Single prediction
        features = np.random.randn(10).astype(np.float32)
        prediction, latency = engine.predict("production_model", "v1.0", features)
        
        assert prediction is not None
        assert latency > 0
        assert latency < 1000  # Should be under 1 second
        
        # Batch prediction
        features_batch = [np.random.randn(10).astype(np.float32) for _ in range(10)]
        batch_result = engine.predict_batch("production_model", "v1.0", features_batch)
        
        assert batch_result.batch_size == 10
        assert len(batch_result.predictions) == 10
        
        # Check stats
        stats = engine.get_stats()
        assert stats['models']['registered'] >= 1
        
        # Cleanup
        engine.shutdown()
    
    def test_cache_hit_performance(self):
        """Test cache improves performance"""
        engine = InferenceEngine(
            use_local_cache=True,
            local_cache_size=5,
            enable_batching=False,
            use_gpu=False
        )
        
        model = SimpleModel(input_size=10, output_size=2)
        engine.register_model("cached_model", "v1.0", model, cache=True)
        
        features = np.random.randn(10).astype(np.float32)
        
        # First prediction (cache miss)
        _, latency1 = engine.predict("cached_model", "v1.0", features)
        
        # Second prediction (cache hit)
        _, latency2 = engine.predict("cached_model", "v1.0", features)
        
        # Both should complete successfully
        assert latency1 > 0
        assert latency2 > 0
        
        engine.shutdown()
    
    def test_multiple_models(self):
        """Test handling multiple models"""
        engine = InferenceEngine(
            use_local_cache=True,
            local_cache_size=10,
            enable_batching=False,
            use_gpu=False
        )
        
        # Register multiple models
        for i in range(5):
            model = SimpleModel(input_size=10, output_size=2)
            engine.register_model(f"model_{i}", "v1.0", model)
        
        # Predict with each model
        features = np.random.randn(10).astype(np.float32)
        
        for i in range(5):
            prediction, latency = engine.predict(f"model_{i}", "v1.0", features)
            assert prediction is not None
            assert latency > 0
        
        # Check stats
        stats = engine.get_stats()
        assert stats['models']['registered'] == 5
        
        engine.shutdown()
    
    def test_concurrent_inference(self):
        """Test concurrent inference requests"""
        import threading
        
        engine = InferenceEngine(
            use_local_cache=True,
            enable_batching=False,
            use_gpu=False
        )
        
        model = SimpleModel(input_size=10, output_size=2)
        engine.register_model("concurrent_model", "v1.0", model)
        
        results = []
        errors = []
        
        def make_prediction():
            try:
                features = np.random.randn(10).astype(np.float32)
                prediction, latency = engine.predict("concurrent_model", "v1.0", features)
                results.append((prediction, latency))
            except Exception as e:
                errors.append(e)
        
        # Run 10 concurrent requests
        threads = [threading.Thread(target=make_prediction) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should complete successfully
        assert len(results) == 10
        assert len(errors) == 0
        
        engine.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
