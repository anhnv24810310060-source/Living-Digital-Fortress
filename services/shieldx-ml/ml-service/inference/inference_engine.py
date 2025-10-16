"""
Optimized Inference Engine
Provides model caching, dynamic batching, and GPU optimization for production inference
"""

import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field
import numpy as np
import torch
import redis
import pickle
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Single inference request"""
    request_id: str
    model_name: str
    model_version: str
    features: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    # Response handling
    response: Optional[Any] = None
    error: Optional[str] = None
    completed: bool = False


@dataclass
class BatchInferenceResult:
    """Result of batch inference"""
    predictions: List[Any]
    latencies: List[float]
    batch_size: int
    total_time: float
    
    def get_avg_latency(self) -> float:
        """Get average latency per request"""
        return self.total_time / self.batch_size if self.batch_size > 0 else 0.0


class LRUCache:
    """Least Recently Used cache for models"""
    
    def __init__(self, capacity: int = 10):
        """
        Initialize LRU cache
        
        Args:
            capacity: Maximum number of models to cache
        """
        self.capacity = capacity
        self.cache = OrderedDict()
        self._lock = threading.Lock()
        
        logger.info(f"LRU Cache initialized with capacity {capacity}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            if key not in self.cache:
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self._lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                # Add new
                if len(self.cache) >= self.capacity:
                    # Remove least recently used
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    logger.info(f"Evicted model from cache: {oldest_key}")
                
                self.cache[key] = value
                logger.info(f"Added model to cache: {key}")
    
    def remove(self, key: str):
        """Remove item from cache"""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                logger.info(f"Removed model from cache: {key}")
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def size(self) -> int:
        """Get current cache size"""
        with self._lock:
            return len(self.cache)
    
    def keys(self) -> List[str]:
        """Get all cache keys"""
        with self._lock:
            return list(self.cache.keys())


class RedisModelCache:
    """Redis-based distributed model cache"""
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 ttl_seconds: int = 3600):
        """
        Initialize Redis cache
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
            ttl_seconds: Time-to-live for cached models
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False
        )
        self.ttl_seconds = ttl_seconds
        
        logger.info(f"Redis cache connected to {redis_host}:{redis_port}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get model from Redis"""
        try:
            data = self.redis_client.get(f"model:{key}")
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def put(self, key: str, value: Any):
        """Put model in Redis"""
        try:
            data = pickle.dumps(value)
            self.redis_client.setex(
                f"model:{key}",
                self.ttl_seconds,
                data
            )
            logger.info(f"Cached model in Redis: {key}")
        except Exception as e:
            logger.error(f"Redis put error: {e}")
    
    def remove(self, key: str):
        """Remove model from Redis"""
        try:
            self.redis_client.delete(f"model:{key}")
            logger.info(f"Removed model from Redis: {key}")
        except Exception as e:
            logger.error(f"Redis remove error: {e}")
    
    def clear(self):
        """Clear all cached models"""
        try:
            pattern = "model:*"
            for key in self.redis_client.scan_iter(pattern):
                self.redis_client.delete(key)
            logger.info("Redis cache cleared")
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def ttl(self, key: str) -> int:
        """Get time-to-live for key"""
        try:
            return self.redis_client.ttl(f"model:{key}")
        except Exception as e:
            logger.error(f"Redis ttl error: {e}")
            return -1


class DynamicBatcher:
    """Dynamic batching for inference requests"""
    
    def __init__(self,
                 max_batch_size: int = 32,
                 max_wait_time_ms: float = 10.0,
                 model_inference_fn: Optional[Callable] = None):
        """
        Initialize dynamic batcher
        
        Args:
            max_batch_size: Maximum batch size
            max_wait_time_ms: Maximum wait time in milliseconds
            model_inference_fn: Model inference function
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.model_inference_fn = model_inference_fn
        
        self.request_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        
        logger.info(f"Dynamic batcher initialized: max_batch={max_batch_size}, max_wait={max_wait_time_ms}ms")
    
    def start(self):
        """Start batcher worker thread"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()
        logger.info("Dynamic batcher started")
    
    def stop(self):
        """Stop batcher worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Dynamic batcher stopped")
    
    def submit_request(self, request: InferenceRequest):
        """Submit inference request"""
        self.request_queue.put(request)
    
    def _batch_worker(self):
        """Worker thread for batch processing"""
        while self.running:
            try:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
            except Exception as e:
                logger.error(f"Batch worker error: {e}")
    
    def _collect_batch(self) -> List[InferenceRequest]:
        """Collect requests into batch"""
        batch = []
        deadline = time.time() + (self.max_wait_time_ms / 1000.0)
        
        while len(batch) < self.max_batch_size and time.time() < deadline:
            timeout = max(0.001, deadline - time.time())
            try:
                request = self.request_queue.get(timeout=timeout)
                batch.append(request)
            except queue.Empty:
                break
        
        return batch
    
    def _process_batch(self, batch: List[InferenceRequest]):
        """Process batch of requests"""
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Stack features
            features = np.stack([req.features for req in batch])
            
            # Run inference
            if self.model_inference_fn:
                predictions = self.model_inference_fn(features)
            else:
                predictions = [None] * len(batch)
            
            # Assign results
            for i, request in enumerate(batch):
                request.response = predictions[i] if i < len(predictions) else None
                request.completed = True
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            for request in batch:
                request.error = str(e)
                request.completed = True
        
        batch_time = time.time() - start_time
        logger.debug(f"Processed batch: size={len(batch)}, time={batch_time*1000:.2f}ms")


class GPUManager:
    """GPU resource management"""
    
    def __init__(self):
        self.device = self._select_device()
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        logger.info(f"GPU Manager initialized: device={self.device}, count={self.device_count}")
    
    def _select_device(self) -> torch.device:
        """Select best available device"""
        if torch.cuda.is_available():
            # Select GPU with most free memory
            max_free = 0
            best_device = 0
            
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                free, total = torch.cuda.mem_get_info()
                if free > max_free:
                    max_free = free
                    best_device = i
            
            return torch.device(f"cuda:{best_device}")
        else:
            return torch.device("cpu")
    
    def get_device(self) -> torch.device:
        """Get current device"""
        return self.device
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information"""
        if not torch.cuda.is_available():
            return {
                'available': False,
                'device_count': 0
            }
        
        info = {
            'available': True,
            'device_count': self.device_count,
            'devices': []
        }
        
        for i in range(self.device_count):
            torch.cuda.set_device(i)
            free, total = torch.cuda.mem_get_info()
            
            info['devices'].append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory_mb': total / (1024 ** 2),
                'free_memory_mb': free / (1024 ** 2),
                'used_memory_mb': (total - free) / (1024 ** 2),
                'utilization_pct': ((total - free) / total) * 100
            })
        
        return info
    
    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")


class InferenceEngine:
    """Optimized inference engine with caching and batching"""
    
    def __init__(self,
                 use_local_cache: bool = True,
                 local_cache_size: int = 10,
                 use_redis_cache: bool = False,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 enable_batching: bool = True,
                 max_batch_size: int = 32,
                 max_wait_time_ms: float = 10.0,
                 use_gpu: bool = True):
        """
        Initialize inference engine
        
        Args:
            use_local_cache: Enable local LRU cache
            local_cache_size: Local cache capacity
            use_redis_cache: Enable Redis distributed cache
            redis_host: Redis host
            redis_port: Redis port
            enable_batching: Enable dynamic batching
            max_batch_size: Maximum batch size
            max_wait_time_ms: Maximum wait time for batching
            use_gpu: Use GPU if available
        """
        # Caching
        self.use_local_cache = use_local_cache
        self.use_redis_cache = use_redis_cache
        
        if use_local_cache:
            self.local_cache = LRUCache(capacity=local_cache_size)
        
        if use_redis_cache:
            self.redis_cache = RedisModelCache(
                redis_host=redis_host,
                redis_port=redis_port
            )
        
        # Batching
        self.enable_batching = enable_batching
        if enable_batching:
            self.batcher = DynamicBatcher(
                max_batch_size=max_batch_size,
                max_wait_time_ms=max_wait_time_ms
            )
            self.batcher.start()
        
        # GPU
        self.use_gpu = use_gpu
        if use_gpu:
            self.gpu_manager = GPUManager()
        
        # Model registry
        self.models: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        logger.info("Inference engine initialized")
    
    def register_model(self, 
                      model_name: str,
                      model_version: str,
                      model: Any,
                      cache: bool = True):
        """
        Register a model for inference
        
        Args:
            model_name: Model name
            model_version: Model version
            model: Model object
            cache: Whether to cache the model
        """
        key = f"{model_name}:{model_version}"
        
        with self._lock:
            self.models[key] = model
        
        # Cache if enabled
        if cache:
            if self.use_local_cache:
                self.local_cache.put(key, model)
            
            if self.use_redis_cache:
                self.redis_cache.put(key, model)
        
        logger.info(f"Registered model: {key}")
    
    def get_model(self, model_name: str, model_version: str) -> Optional[Any]:
        """Get model from cache or registry"""
        key = f"{model_name}:{model_version}"
        
        # Try local cache first
        if self.use_local_cache:
            model = self.local_cache.get(key)
            if model is not None:
                logger.debug(f"Model found in local cache: {key}")
                return model
        
        # Try Redis cache
        if self.use_redis_cache:
            model = self.redis_cache.get(key)
            if model is not None:
                logger.debug(f"Model found in Redis cache: {key}")
                # Update local cache
                if self.use_local_cache:
                    self.local_cache.put(key, model)
                return model
        
        # Try registry
        with self._lock:
            model = self.models.get(key)
            if model is not None:
                logger.debug(f"Model found in registry: {key}")
                return model
        
        logger.warning(f"Model not found: {key}")
        return None
    
    def predict(self,
                model_name: str,
                model_version: str,
                features: np.ndarray,
                timeout: float = 30.0) -> Tuple[Any, float]:
        """
        Run inference
        
        Args:
            model_name: Model name
            model_version: Model version
            features: Input features
            timeout: Timeout in seconds
        
        Returns:
            Tuple of (prediction, latency_ms)
        """
        start_time = time.time()
        
        # Get model
        model = self.get_model(model_name, model_version)
        if model is None:
            raise ValueError(f"Model not found: {model_name}:{model_version}")
        
        # Create request
        request = InferenceRequest(
            request_id=self._generate_request_id(),
            model_name=model_name,
            model_version=model_version,
            features=features
        )
        
        # Use batching if enabled
        if self.enable_batching:
            self.batcher.submit_request(request)
            
            # Wait for completion
            deadline = time.time() + timeout
            while not request.completed and time.time() < deadline:
                time.sleep(0.001)
            
            if not request.completed:
                raise TimeoutError("Inference timeout")
            
            if request.error:
                raise RuntimeError(f"Inference error: {request.error}")
            
            prediction = request.response
        else:
            # Direct inference
            prediction = self._run_inference(model, features)
        
        latency_ms = (time.time() - start_time) * 1000
        return prediction, latency_ms
    
    def predict_batch(self,
                     model_name: str,
                     model_version: str,
                     features_batch: List[np.ndarray]) -> BatchInferenceResult:
        """
        Run batch inference
        
        Args:
            model_name: Model name
            model_version: Model version
            features_batch: List of feature arrays
        
        Returns:
            Batch inference result
        """
        start_time = time.time()
        
        # Get model
        model = self.get_model(model_name, model_version)
        if model is None:
            raise ValueError(f"Model not found: {model_name}:{model_version}")
        
        # Stack features
        features = np.stack(features_batch)
        
        # Run inference
        predictions = self._run_inference(model, features)
        
        total_time = time.time() - start_time
        
        # Individual latencies (approximate)
        avg_latency = total_time / len(features_batch)
        latencies = [avg_latency] * len(features_batch)
        
        return BatchInferenceResult(
            predictions=predictions if isinstance(predictions, list) else predictions.tolist(),
            latencies=latencies,
            batch_size=len(features_batch),
            total_time=total_time
        )
    
    def _run_inference(self, model: Any, features: np.ndarray) -> Any:
        """Run model inference"""
        # Convert to tensor if PyTorch model
        if isinstance(model, torch.nn.Module):
            features_tensor = torch.from_numpy(features).float()
            
            # Move to GPU if available
            if self.use_gpu and torch.cuda.is_available():
                features_tensor = features_tensor.to(self.gpu_manager.get_device())
                model = model.to(self.gpu_manager.get_device())
            
            # Run inference
            with torch.no_grad():
                output = model(features_tensor)
            
            # Convert back to numpy
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()
            
            return output
        else:
            # Scikit-learn or other models
            return model.predict(features)
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        return f"{int(time.time() * 1000000)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        stats = {
            'caching': {
                'local_enabled': self.use_local_cache,
                'redis_enabled': self.use_redis_cache
            },
            'batching': {
                'enabled': self.enable_batching
            },
            'models': {
                'registered': len(self.models)
            }
        }
        
        if self.use_local_cache:
            stats['caching']['local_size'] = self.local_cache.size()
            stats['caching']['local_keys'] = self.local_cache.keys()
        
        if self.use_gpu:
            stats['gpu'] = self.gpu_manager.get_memory_info()
        
        return stats
    
    def clear_caches(self):
        """Clear all caches"""
        if self.use_local_cache:
            self.local_cache.clear()
        
        if self.use_redis_cache:
            self.redis_cache.clear()
        
        if self.use_gpu:
            self.gpu_manager.clear_cache()
        
        logger.info("All caches cleared")
    
    def shutdown(self):
        """Shutdown inference engine"""
        if self.enable_batching:
            self.batcher.stop()
        
        self.clear_caches()
        logger.info("Inference engine shutdown")


# Global inference engine instance
_global_engine: Optional[InferenceEngine] = None


def get_global_engine() -> InferenceEngine:
    """Get or create global inference engine"""
    global _global_engine
    if _global_engine is None:
        _global_engine = InferenceEngine()
    return _global_engine


def set_global_engine(engine: InferenceEngine):
    """Set global inference engine"""
    global _global_engine
    _global_engine = engine
