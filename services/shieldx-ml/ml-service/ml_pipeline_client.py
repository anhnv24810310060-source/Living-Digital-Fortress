"""
ML Model Registry Client - Python Interface
Integrates with Go-based model registry for model management
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import redis
import logging

logger = logging.getLogger(__name__)


class ModelStatus:
    DRAFT = "draft"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelRegistryClient:
    """Client for interacting with the ML model registry"""
    
    def __init__(self, redis_url: str, storage_dir: str = "/var/ml/models"):
        self.redis_client = redis.from_url(redis_url)
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def register_model(
        self,
        name: str,
        version: str,
        model_path: str,
        algorithm: str,
        framework: str = "scikit-learn",
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register a trained model in the registry
        
        Returns:
            model_id: Unique identifier for the registered model
        """
        # Generate model ID
        model_id = self._generate_model_id(name, version)
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(model_path)
        file_size = os.path.getsize(model_path)
        
        # Create metadata
        metadata = {
            "model_id": model_id,
            "name": name,
            "version": version,
            "algorithm": algorithm,
            "framework": framework,
            "status": ModelStatus.DRAFT,
            "metrics": metrics or {},
            "parameters": parameters or {},
            "tags": tags or [],
            "file_path": model_path,
            "file_hash": file_hash,
            "file_size": file_size,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "created_by": "ml-service"
        }
        
        # Store in Redis
        key = f"ml:model:{model_id}"
        self.redis_client.set(key, json.dumps(metadata))
        
        logger.info(f"Model registered: {model_id} ({name} v{version})")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata"""
        key = f"ml:model:{model_id}"
        data = self.redis_client.get(key)
        
        if data is None:
            return None
        
        return json.loads(data)
    
    def list_models(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all models, optionally filtered by status"""
        pattern = "ml:model:*"
        models = []
        
        for key in self.redis_client.scan_iter(pattern):
            data = self.redis_client.get(key)
            if data:
                metadata = json.loads(data)
                if status is None or metadata.get("status") == status:
                    models.append(metadata)
        
        return models
    
    def promote_model(self, model_id: str, new_status: str) -> bool:
        """Promote model to new status"""
        metadata = self.get_model(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        # Validate transition
        if not self._is_valid_transition(metadata["status"], new_status):
            raise ValueError(
                f"Invalid status transition: {metadata['status']} -> {new_status}"
            )
        
        metadata["status"] = new_status
        metadata["updated_at"] = datetime.utcnow().isoformat()
        
        key = f"ml:model:{model_id}"
        self.redis_client.set(key, json.dumps(metadata))
        
        logger.info(f"Model {model_id} promoted to {new_status}")
        return True
    
    def get_production_model(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the current production model for a given name"""
        models = self.list_models(status=ModelStatus.PRODUCTION)
        
        # Filter by name and get latest version
        matching = [m for m in models if m["name"] == name]
        if not matching:
            return None
        
        # Sort by version (descending) and return first
        matching.sort(key=lambda m: m["version"], reverse=True)
        return matching[0]
    
    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.utcnow().isoformat()
        hash_input = f"{name}_{version}_{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of model file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _is_valid_transition(self, from_status: str, to_status: str) -> bool:
        """Validate status transition"""
        valid_transitions = {
            ModelStatus.DRAFT: [ModelStatus.TESTING, ModelStatus.ARCHIVED],
            ModelStatus.TESTING: [ModelStatus.STAGING, ModelStatus.DRAFT, ModelStatus.ARCHIVED],
            ModelStatus.STAGING: [ModelStatus.PRODUCTION, ModelStatus.TESTING, ModelStatus.ARCHIVED],
            ModelStatus.PRODUCTION: [ModelStatus.ARCHIVED],
            ModelStatus.ARCHIVED: [ModelStatus.TESTING]
        }
        
        allowed = valid_transitions.get(from_status, [])
        return to_status in allowed


class ABTestingClient:
    """Client for A/B testing experiments"""
    
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
    
    def get_model_assignment(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Get model assignment for a user in an experiment"""
        # Check for sticky assignment
        assignment_key = f"ab:assignment:{experiment_id}:{user_id}"
        assignment = self.redis_client.get(assignment_key)
        
        if assignment:
            data = json.loads(assignment)
            return data["model_id"]
        
        # Get experiment config
        exp_key = f"ab:experiment:{experiment_id}"
        exp_data = self.redis_client.get(exp_key)
        
        if not exp_data:
            return None
        
        experiment = json.loads(exp_data)
        
        if experiment["status"] != "running":
            return None
        
        # Assign based on hash
        model_id = self._assign_model(experiment, user_id)
        
        # Store sticky assignment
        if experiment.get("sticky_assignment", True):
            assignment_data = {
                "experiment_id": experiment_id,
                "user_id": user_id,
                "model_id": model_id,
                "assigned_at": datetime.utcnow().isoformat()
            }
            self.redis_client.setex(
                assignment_key,
                30 * 24 * 3600,  # 30 days
                json.dumps(assignment_data)
            )
        
        return model_id
    
    def record_metric(
        self,
        experiment_id: str,
        model_id: str,
        metric_name: str,
        value: float
    ) -> bool:
        """Record a metric value for a model variant"""
        exp_key = f"ab:experiment:{experiment_id}"
        exp_data = self.redis_client.get(exp_key)
        
        if not exp_data:
            return False
        
        experiment = json.loads(exp_data)
        
        # Update metrics
        if model_id not in experiment["metrics"]:
            experiment["metrics"][model_id] = {
                "model_id": model_id,
                "sample_count": 0,
                "metrics": {},
                "distributions": {}
            }
        
        metric_data = experiment["metrics"][model_id]
        metric_data["sample_count"] += 1
        
        # Running average
        current = metric_data["metrics"].get(metric_name, 0.0)
        n = metric_data["sample_count"]
        new_value = (current * (n - 1) + value) / n
        metric_data["metrics"][metric_name] = new_value
        
        # Update distributions (keep last 1000)
        if metric_name not in metric_data["distributions"]:
            metric_data["distributions"][metric_name] = []
        
        metric_data["distributions"][metric_name].append(value)
        if len(metric_data["distributions"][metric_name]) > 1000:
            metric_data["distributions"][metric_name] = \
                metric_data["distributions"][metric_name][-1000:]
        
        metric_data["last_updated"] = datetime.utcnow().isoformat()
        experiment["updated_at"] = datetime.utcnow().isoformat()
        
        # Persist
        self.redis_client.set(exp_key, json.dumps(experiment))
        return True
    
    def _assign_model(self, experiment: Dict, user_id: str) -> str:
        """Assign user to model variant using consistent hashing"""
        # Hash user ID
        hash_bytes = hashlib.sha256(
            f"{experiment['id']}{user_id}".encode()
        ).digest()
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        
        # Convert to percentage
        percentage = (hash_int % 10000) / 100.0
        
        # Assign based on traffic split
        cumulative = 0.0
        for variant in experiment["models"]:
            cumulative += variant["traffic_pct"]
            if percentage < cumulative:
                return variant["model_id"]
        
        # Fallback to control
        for variant in experiment["models"]:
            if variant.get("is_control"):
                return variant["model_id"]
        
        return experiment["models"][0]["model_id"]


class DriftDetectorClient:
    """Client for feature drift detection"""
    
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
    
    def record_feature(self, feature_name: str, value: float):
        """Record a feature value for drift monitoring"""
        key = f"drift:feature:{feature_name}"
        
        # Store in a time-series list (keep last 1000)
        self.redis_client.lpush(key, json.dumps({
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        }))
        self.redis_client.ltrim(key, 0, 999)
    
    def get_drift_alerts(self, since_timestamp: Optional[str] = None) -> List[Dict]:
        """Get drift alerts since a specific timestamp"""
        pattern = "drift:alert:*"
        alerts = []
        
        for key in self.redis_client.scan_iter(pattern):
            data = self.redis_client.get(key)
            if data:
                alert = json.loads(data)
                
                if since_timestamp is None or alert["detected_at"] > since_timestamp:
                    alerts.append(alert)
        
        return sorted(alerts, key=lambda a: a["detected_at"], reverse=True)


# Singleton instances
_model_registry = None
_ab_testing = None
_drift_detector = None


def get_model_registry(redis_url: str = None) -> ModelRegistryClient:
    """Get singleton model registry client"""
    global _model_registry
    if _model_registry is None:
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        _model_registry = ModelRegistryClient(redis_url)
    return _model_registry


def get_ab_testing(redis_url: str = None) -> ABTestingClient:
    """Get singleton A/B testing client"""
    global _ab_testing
    if _ab_testing is None:
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        _ab_testing = ABTestingClient(redis_url)
    return _ab_testing


def get_drift_detector(redis_url: str = None) -> DriftDetectorClient:
    """Get singleton drift detector client"""
    global _drift_detector
    if _drift_detector is None:
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        _drift_detector = DriftDetectorClient(redis_url)
    return _drift_detector
