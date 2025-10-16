"""
ML-Specific Monitoring System
Tracks model performance, data quality, drift, and system health in production
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import threading
import json

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Single metric measurement at a point in time"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'labels': self.labels
        }


@dataclass
class ModelPrediction:
    """Single prediction record for monitoring"""
    timestamp: float
    model_name: str
    model_version: str
    prediction: Any
    confidence: float
    features: Dict[str, float]
    true_label: Optional[Any] = None
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'prediction': str(self.prediction),
            'confidence': self.confidence,
            'features': self.features,
            'true_label': str(self.true_label) if self.true_label else None,
            'latency_ms': self.latency_ms
        }


class MetricTracker:
    """Track a single metric over time with statistics"""
    
    def __init__(self, name: str, window_size: int = 1000):
        self.name = name
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self._lock = threading.Lock()
        
    def record(self, value: float, timestamp: Optional[float] = None):
        """Record a new metric value"""
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
            self.values.append(value)
            self.timestamps.append(timestamp)
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistical summary of recent values"""
        with self._lock:
            if not self.values:
                return {
                    'count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'p50': 0.0,
                    'p95': 0.0,
                    'p99': 0.0
                }
            
            arr = np.array(self.values)
            return {
                'count': len(arr),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'p50': float(np.percentile(arr, 50)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99))
            }
    
    def get_recent(self, seconds: int = 60) -> List[Tuple[float, float]]:
        """Get values from last N seconds"""
        with self._lock:
            cutoff = time.time() - seconds
            result = []
            for ts, val in zip(self.timestamps, self.values):
                if ts >= cutoff:
                    result.append((ts, val))
            return result


class AccuracyTracker:
    """Track model accuracy over time"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.correct = deque(maxlen=window_size)
        self._lock = threading.Lock()
        
    def record(self, prediction: Any, true_label: Any):
        """Record a prediction and its correctness"""
        with self._lock:
            self.predictions.append(prediction)
            self.correct.append(prediction == true_label)
    
    def get_accuracy(self) -> float:
        """Get current accuracy"""
        with self._lock:
            if not self.correct:
                return 0.0
            return sum(self.correct) / len(self.correct)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed accuracy statistics"""
        with self._lock:
            if not self.correct:
                return {
                    'accuracy': 0.0,
                    'total_predictions': 0,
                    'correct_predictions': 0
                }
            
            return {
                'accuracy': sum(self.correct) / len(self.correct),
                'total_predictions': len(self.correct),
                'correct_predictions': sum(self.correct)
            }


class LatencyTracker(MetricTracker):
    """Track inference latency"""
    
    def __init__(self, window_size: int = 1000):
        super().__init__('latency', window_size)
    
    def get_sla_compliance(self, sla_ms: float = 10.0) -> Dict[str, Any]:
        """Check if latency meets SLA"""
        stats = self.get_stats()
        return {
            'sla_target_ms': sla_ms,
            'p99_latency_ms': stats['p99'],
            'meets_sla': stats['p99'] <= sla_ms,
            'violations': sum(1 for v in self.values if v > sla_ms),
            'total_requests': stats['count']
        }


class ThroughputTracker:
    """Track request throughput"""
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.timestamps = deque()
        self._lock = threading.Lock()
        
    def record_request(self, timestamp: Optional[float] = None):
        """Record a single request"""
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
            self.timestamps.append(timestamp)
            
            # Remove old timestamps
            cutoff = timestamp - self.window_seconds
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.popleft()
    
    def get_throughput(self) -> float:
        """Get current throughput (requests per second)"""
        with self._lock:
            if not self.timestamps:
                return 0.0
            
            # Clean old timestamps
            cutoff = time.time() - self.window_seconds
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.popleft()
            
            if not self.timestamps:
                return 0.0
            
            time_span = self.timestamps[-1] - self.timestamps[0]
            if time_span == 0:
                return 0.0
            
            return len(self.timestamps) / time_span
    
    def get_stats(self) -> Dict[str, Any]:
        """Get throughput statistics"""
        throughput = self.get_throughput()
        return {
            'requests_per_second': throughput,
            'total_requests': len(self.timestamps),
            'window_seconds': self.window_seconds
        }


class DriftTracker:
    """Track feature and prediction drift"""
    
    def __init__(self, baseline_window: int = 1000, current_window: int = 100):
        self.baseline_window = baseline_window
        self.current_window = current_window
        self.baseline_features = {}  # feature_name -> deque of values
        self.current_features = {}   # feature_name -> deque of values
        self.baseline_predictions = deque(maxlen=baseline_window)
        self.current_predictions = deque(maxlen=current_window)
        self._lock = threading.Lock()
        
    def record_baseline(self, features: Dict[str, float], prediction: Any):
        """Record baseline data"""
        with self._lock:
            for name, value in features.items():
                if name not in self.baseline_features:
                    self.baseline_features[name] = deque(maxlen=self.baseline_window)
                self.baseline_features[name].append(value)
            
            self.baseline_predictions.append(prediction)
    
    def record_current(self, features: Dict[str, float], prediction: Any):
        """Record current production data"""
        with self._lock:
            for name, value in features.items():
                if name not in self.current_features:
                    self.current_features[name] = deque(maxlen=self.current_window)
                self.current_features[name].append(value)
            
            self.current_predictions.append(prediction)
    
    def detect_feature_drift(self, threshold: float = 0.05) -> Dict[str, Any]:
        """Detect feature drift using KS test"""
        with self._lock:
            drift_results = {}
            
            for name in self.baseline_features:
                if name not in self.current_features:
                    continue
                
                baseline = list(self.baseline_features[name])
                current = list(self.current_features[name])
                
                if len(baseline) < 30 or len(current) < 30:
                    continue
                
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(baseline, current)
                
                drift_results[name] = {
                    'ks_statistic': float(statistic),
                    'p_value': float(p_value),
                    'has_drift': p_value < threshold,
                    'baseline_mean': float(np.mean(baseline)),
                    'current_mean': float(np.mean(current)),
                    'mean_shift': float(np.mean(current) - np.mean(baseline))
                }
            
            return drift_results
    
    def detect_prediction_drift(self) -> Dict[str, Any]:
        """Detect prediction distribution drift"""
        with self._lock:
            if len(self.baseline_predictions) < 30 or len(self.current_predictions) < 30:
                return {
                    'has_drift': False,
                    'message': 'Insufficient data for drift detection'
                }
            
            baseline = np.array(self.baseline_predictions)
            current = np.array(self.current_predictions)
            
            # Chi-square test for categorical predictions
            baseline_counts = np.bincount(baseline.astype(int))
            current_counts = np.bincount(current.astype(int))
            
            # Ensure same length
            max_len = max(len(baseline_counts), len(current_counts))
            baseline_counts = np.pad(baseline_counts, (0, max_len - len(baseline_counts)))
            current_counts = np.pad(current_counts, (0, max_len - len(current_counts)))
            
            chi2, p_value = stats.chisquare(current_counts + 1, baseline_counts + 1)
            
            return {
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'has_drift': p_value < 0.05,
                'baseline_distribution': baseline_counts.tolist(),
                'current_distribution': current_counts.tolist()
            }


class FairnessMetrics:
    """Track model fairness across protected groups"""
    
    def __init__(self):
        self.predictions_by_group = {}  # group_name -> list of (prediction, true_label)
        self._lock = threading.Lock()
        
    def record(self, group: str, prediction: Any, true_label: Any):
        """Record prediction for a protected group"""
        with self._lock:
            if group not in self.predictions_by_group:
                self.predictions_by_group[group] = []
            self.predictions_by_group[group].append((prediction, true_label))
    
    def get_fairness_metrics(self) -> Dict[str, Any]:
        """Calculate fairness metrics across groups"""
        with self._lock:
            metrics = {}
            
            for group, predictions in self.predictions_by_group.items():
                if not predictions:
                    continue
                
                correct = sum(1 for pred, true in predictions if pred == true)
                total = len(predictions)
                accuracy = correct / total if total > 0 else 0.0
                
                # Count positive predictions
                positive_rate = sum(1 for pred, _ in predictions if pred == 1) / total if total > 0 else 0.0
                
                metrics[group] = {
                    'accuracy': accuracy,
                    'positive_rate': positive_rate,
                    'total_predictions': total
                }
            
            # Calculate disparate impact
            if len(metrics) >= 2:
                rates = [m['positive_rate'] for m in metrics.values()]
                disparate_impact = min(rates) / max(rates) if max(rates) > 0 else 1.0
                metrics['_overall'] = {
                    'disparate_impact': disparate_impact,
                    'is_fair': disparate_impact >= 0.8  # 80% rule
                }
            
            return metrics


class MLMonitor:
    """Comprehensive ML system monitoring"""
    
    def __init__(self, 
                 accuracy_window: int = 1000,
                 latency_window: int = 1000,
                 throughput_window: int = 60,
                 drift_baseline_window: int = 1000,
                 drift_current_window: int = 100):
        """
        Initialize ML monitoring system
        
        Args:
            accuracy_window: Number of predictions to track for accuracy
            latency_window: Number of requests to track for latency
            throughput_window: Time window in seconds for throughput
            drift_baseline_window: Baseline data window for drift detection
            drift_current_window: Current data window for drift detection
        """
        self.accuracy = AccuracyTracker(accuracy_window)
        self.latency = LatencyTracker(latency_window)
        self.throughput = ThroughputTracker(throughput_window)
        self.drift = DriftTracker(drift_baseline_window, drift_current_window)
        self.fairness = FairnessMetrics()
        
        # Custom metrics
        self.custom_metrics = {}  # metric_name -> MetricTracker
        
        # Prediction log
        self.prediction_log = deque(maxlen=10000)
        self._lock = threading.Lock()
        
        logger.info("ML Monitor initialized")
    
    def record_prediction(self, 
                         model_name: str,
                         model_version: str,
                         prediction: Any,
                         confidence: float,
                         features: Dict[str, float],
                         true_label: Optional[Any] = None,
                         latency_ms: Optional[float] = None,
                         protected_group: Optional[str] = None,
                         is_baseline: bool = False):
        """
        Record a model prediction
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            prediction: Model prediction
            confidence: Prediction confidence score
            features: Input features
            true_label: True label if available
            latency_ms: Inference latency in milliseconds
            protected_group: Protected group identifier for fairness
            is_baseline: Whether this is baseline data for drift detection
        """
        timestamp = time.time()
        
        # Record latency
        if latency_ms is not None:
            self.latency.record(latency_ms, timestamp)
        
        # Record throughput
        self.throughput.record_request(timestamp)
        
        # Record accuracy
        if true_label is not None:
            self.accuracy.record(prediction, true_label)
            
            # Record fairness
            if protected_group is not None:
                self.fairness.record(protected_group, prediction, true_label)
        
        # Record drift
        if is_baseline:
            self.drift.record_baseline(features, prediction)
        else:
            self.drift.record_current(features, prediction)
        
        # Log prediction
        pred_record = ModelPrediction(
            timestamp=timestamp,
            model_name=model_name,
            model_version=model_version,
            prediction=prediction,
            confidence=confidence,
            features=features,
            true_label=true_label,
            latency_ms=latency_ms
        )
        
        with self._lock:
            self.prediction_log.append(pred_record)
        
        logger.debug(f"Recorded prediction: {model_name}@{model_version} -> {prediction}")
    
    def add_custom_metric(self, name: str, window_size: int = 1000):
        """Add a custom metric tracker"""
        if name not in self.custom_metrics:
            self.custom_metrics[name] = MetricTracker(name, window_size)
            logger.info(f"Added custom metric: {name}")
    
    def record_custom_metric(self, name: str, value: float):
        """Record a custom metric value"""
        if name not in self.custom_metrics:
            self.add_custom_metric(name)
        self.custom_metrics[name].record(value)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        accuracy_stats = self.accuracy.get_stats()
        latency_stats = self.latency.get_stats()
        throughput_stats = self.throughput.get_stats()
        drift_results = self.drift.detect_feature_drift()
        
        # Determine health status
        issues = []
        
        if accuracy_stats['accuracy'] < 0.90:
            issues.append(f"Low accuracy: {accuracy_stats['accuracy']:.2%}")
        
        if latency_stats['p99'] > 10.0:
            issues.append(f"High latency: {latency_stats['p99']:.2f}ms")
        
        if throughput_stats['requests_per_second'] < 100:
            issues.append(f"Low throughput: {throughput_stats['requests_per_second']:.1f} req/s")
        
        drift_count = sum(1 for r in drift_results.values() if r.get('has_drift', False))
        if drift_count > 0:
            issues.append(f"Feature drift detected in {drift_count} features")
        
        status = 'healthy' if not issues else 'degraded' if len(issues) < 3 else 'unhealthy'
        
        return {
            'status': status,
            'timestamp': time.time(),
            'issues': issues,
            'metrics': {
                'accuracy': accuracy_stats,
                'latency': latency_stats,
                'throughput': throughput_stats
            },
            'drift': {
                'features_with_drift': drift_count,
                'total_features': len(drift_results)
            }
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report"""
        return {
            'timestamp': time.time(),
            'accuracy': self.accuracy.get_stats(),
            'latency': {
                **self.latency.get_stats(),
                'sla_compliance': self.latency.get_sla_compliance()
            },
            'throughput': self.throughput.get_stats(),
            'drift': {
                'features': self.drift.detect_feature_drift(),
                'predictions': self.drift.detect_prediction_drift()
            },
            'fairness': self.fairness.get_fairness_metrics(),
            'custom_metrics': {
                name: tracker.get_stats()
                for name, tracker in self.custom_metrics.items()
            },
            'health': self.get_health_status()
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        report = self.get_comprehensive_report()
        lines = []
        
        # Accuracy
        acc = report['accuracy']['accuracy']
        lines.append(f"# HELP ml_model_accuracy Model prediction accuracy")
        lines.append(f"# TYPE ml_model_accuracy gauge")
        lines.append(f"ml_model_accuracy {acc}")
        
        # Latency
        lat = report['latency']
        lines.append(f"# HELP ml_inference_latency_ms Inference latency in milliseconds")
        lines.append(f"# TYPE ml_inference_latency_ms summary")
        lines.append(f'ml_inference_latency_ms{{quantile="0.5"}} {lat["p50"]}')
        lines.append(f'ml_inference_latency_ms{{quantile="0.95"}} {lat["p95"]}')
        lines.append(f'ml_inference_latency_ms{{quantile="0.99"}} {lat["p99"]}')
        
        # Throughput
        thr = report['throughput']['requests_per_second']
        lines.append(f"# HELP ml_requests_per_second Request throughput")
        lines.append(f"# TYPE ml_requests_per_second gauge")
        lines.append(f"ml_requests_per_second {thr}")
        
        # Drift
        drift_count = report['drift']['features']['features_with_drift']
        lines.append(f"# HELP ml_features_with_drift Number of features showing drift")
        lines.append(f"# TYPE ml_features_with_drift gauge")
        lines.append(f"ml_features_with_drift {drift_count}")
        
        return "\n".join(lines)
    
    def get_recent_predictions(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent predictions"""
        with self._lock:
            recent = list(self.prediction_log)[-count:]
            return [pred.to_dict() for pred in recent]
    
    def reset_baseline(self):
        """Reset drift detection baseline with current data"""
        with self._lock:
            self.drift.baseline_features = self.drift.current_features.copy()
            self.drift.baseline_predictions = self.drift.current_predictions.copy()
            self.drift.current_features.clear()
            self.drift.current_predictions.clear()
        
        logger.info("Drift detection baseline reset")


# Global monitor instance
_global_monitor: Optional[MLMonitor] = None


def get_global_monitor() -> MLMonitor:
    """Get or create global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MLMonitor()
    return _global_monitor


def set_global_monitor(monitor: MLMonitor):
    """Set global monitor instance"""
    global _global_monitor
    _global_monitor = monitor
