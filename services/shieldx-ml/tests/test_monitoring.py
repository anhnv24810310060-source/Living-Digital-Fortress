"""
Comprehensive tests for ML monitoring system
"""

import pytest
import time
import numpy as np
from monitoring.ml_monitor import (
    MLMonitor,
    MetricTracker,
    AccuracyTracker,
    LatencyTracker,
    ThroughputTracker,
    DriftTracker,
    FairnessMetrics,
    MetricSnapshot,
    ModelPrediction
)


class TestMetricTracker:
    """Test MetricTracker class"""
    
    def test_initialization(self):
        """Test tracker initialization"""
        tracker = MetricTracker('test_metric', window_size=100)
        assert tracker.name == 'test_metric'
        assert tracker.window_size == 100
        assert len(tracker.values) == 0
    
    def test_record_value(self):
        """Test recording metric values"""
        tracker = MetricTracker('test')
        tracker.record(10.0)
        tracker.record(20.0)
        tracker.record(30.0)
        
        assert len(tracker.values) == 3
        assert list(tracker.values) == [10.0, 20.0, 30.0]
    
    def test_get_stats(self):
        """Test statistical summary"""
        tracker = MetricTracker('test')
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for v in values:
            tracker.record(float(v))
        
        stats = tracker.get_stats()
        assert stats['count'] == 10
        assert stats['mean'] == 5.5
        assert stats['min'] == 1.0
        assert stats['max'] == 10.0
        assert stats['p50'] == 5.5
    
    def test_window_size_limit(self):
        """Test window size limit"""
        tracker = MetricTracker('test', window_size=5)
        for i in range(10):
            tracker.record(float(i))
        
        assert len(tracker.values) == 5
        assert list(tracker.values) == [5.0, 6.0, 7.0, 8.0, 9.0]
    
    def test_get_recent(self):
        """Test getting recent values"""
        tracker = MetricTracker('test')
        now = time.time()
        
        # Add values with timestamps
        tracker.record(1.0, now - 100)
        tracker.record(2.0, now - 50)
        tracker.record(3.0, now - 10)
        tracker.record(4.0, now - 5)
        
        # Get last 60 seconds
        recent = tracker.get_recent(seconds=60)
        assert len(recent) == 3  # Only last 3 within 60 seconds
    
    def test_empty_stats(self):
        """Test stats on empty tracker"""
        tracker = MetricTracker('test')
        stats = tracker.get_stats()
        
        assert stats['count'] == 0
        assert stats['mean'] == 0.0


class TestAccuracyTracker:
    """Test AccuracyTracker class"""
    
    def test_initialization(self):
        """Test tracker initialization"""
        tracker = AccuracyTracker(window_size=100)
        assert tracker.window_size == 100
        assert len(tracker.predictions) == 0
    
    def test_record_predictions(self):
        """Test recording predictions"""
        tracker = AccuracyTracker()
        tracker.record(1, 1)  # Correct
        tracker.record(0, 1)  # Incorrect
        tracker.record(1, 1)  # Correct
        
        assert len(tracker.predictions) == 3
        assert len(tracker.correct) == 3
    
    def test_get_accuracy(self):
        """Test accuracy calculation"""
        tracker = AccuracyTracker()
        
        # 70% accuracy
        for i in range(100):
            tracker.record(1 if i < 70 else 0, 1)
        
        accuracy = tracker.get_accuracy()
        assert abs(accuracy - 0.70) < 0.01
    
    def test_get_stats(self):
        """Test accuracy statistics"""
        tracker = AccuracyTracker()
        tracker.record(1, 1)
        tracker.record(1, 1)
        tracker.record(0, 1)
        
        stats = tracker.get_stats()
        assert stats['total_predictions'] == 3
        assert stats['correct_predictions'] == 2
        assert abs(stats['accuracy'] - 0.6667) < 0.001
    
    def test_empty_accuracy(self):
        """Test accuracy on empty tracker"""
        tracker = AccuracyTracker()
        assert tracker.get_accuracy() == 0.0


class TestLatencyTracker:
    """Test LatencyTracker class"""
    
    def test_initialization(self):
        """Test tracker initialization"""
        tracker = LatencyTracker(window_size=100)
        assert tracker.name == 'latency'
        assert tracker.window_size == 100
    
    def test_sla_compliance(self):
        """Test SLA compliance checking"""
        tracker = LatencyTracker()
        
        # Add latencies: most under 10ms, some over
        latencies = [5, 6, 7, 8, 9, 15, 20, 25]
        for lat in latencies:
            tracker.record(float(lat))
        
        sla = tracker.get_sla_compliance(sla_ms=10.0)
        assert sla['sla_target_ms'] == 10.0
        assert sla['meets_sla'] == False  # p99 > 10ms
        assert sla['violations'] == 3
        assert sla['total_requests'] == 8
    
    def test_sla_met(self):
        """Test when SLA is met"""
        tracker = LatencyTracker()
        
        # All under 10ms
        for i in range(100):
            tracker.record(float(5 + i % 3))  # 5, 6, 7, 5, 6, 7, ...
        
        sla = tracker.get_sla_compliance(sla_ms=10.0)
        assert sla['meets_sla'] == True


class TestThroughputTracker:
    """Test ThroughputTracker class"""
    
    def test_initialization(self):
        """Test tracker initialization"""
        tracker = ThroughputTracker(window_seconds=60)
        assert tracker.window_seconds == 60
        assert len(tracker.timestamps) == 0
    
    def test_record_requests(self):
        """Test recording requests"""
        tracker = ThroughputTracker(window_seconds=10)
        now = time.time()
        
        # Record 10 requests
        for i in range(10):
            tracker.record_request(now + i * 0.1)
        
        assert len(tracker.timestamps) == 10
    
    def test_get_throughput(self):
        """Test throughput calculation"""
        tracker = ThroughputTracker(window_seconds=10)
        now = time.time()
        
        # 100 requests over 10 seconds = 10 req/s
        for i in range(100):
            tracker.record_request(now + i * 0.1)
        
        throughput = tracker.get_throughput()
        assert abs(throughput - 10.0) < 1.0
    
    def test_window_cleanup(self):
        """Test old timestamps are removed"""
        tracker = ThroughputTracker(window_seconds=5)
        now = time.time()
        
        # Add old timestamps
        tracker.record_request(now - 10)
        tracker.record_request(now - 8)
        tracker.record_request(now - 2)
        tracker.record_request(now - 1)
        
        # Should only count recent ones
        throughput = tracker.get_throughput()
        # Only 2 requests in last 5 seconds
        assert len(tracker.timestamps) == 2
    
    def test_get_stats(self):
        """Test throughput statistics"""
        tracker = ThroughputTracker(window_seconds=10)
        now = time.time()
        
        for i in range(50):
            tracker.record_request(now + i * 0.1)
        
        stats = tracker.get_stats()
        assert 'requests_per_second' in stats
        assert stats['total_requests'] == 50
        assert stats['window_seconds'] == 10


class TestDriftTracker:
    """Test DriftTracker class"""
    
    def test_initialization(self):
        """Test tracker initialization"""
        tracker = DriftTracker(baseline_window=1000, current_window=100)
        assert tracker.baseline_window == 1000
        assert tracker.current_window == 100
        assert len(tracker.baseline_features) == 0
    
    def test_record_baseline(self):
        """Test recording baseline data"""
        tracker = DriftTracker()
        features = {'feature1': 1.0, 'feature2': 2.0}
        tracker.record_baseline(features, prediction=1)
        
        assert 'feature1' in tracker.baseline_features
        assert 'feature2' in tracker.baseline_features
        assert len(tracker.baseline_predictions) == 1
    
    def test_record_current(self):
        """Test recording current data"""
        tracker = DriftTracker()
        features = {'feature1': 1.5, 'feature2': 2.5}
        tracker.record_current(features, prediction=0)
        
        assert 'feature1' in tracker.current_features
        assert 'feature2' in tracker.current_features
        assert len(tracker.current_predictions) == 1
    
    def test_detect_no_drift(self):
        """Test drift detection when no drift"""
        tracker = DriftTracker()
        
        # Same distribution
        np.random.seed(42)
        for i in range(100):
            val = np.random.normal(0, 1)
            tracker.record_baseline({'feature1': val}, prediction=1)
        
        np.random.seed(43)
        for i in range(100):
            val = np.random.normal(0, 1)
            tracker.record_current({'feature1': val}, prediction=1)
        
        drift = tracker.detect_feature_drift(threshold=0.05)
        assert 'feature1' in drift
        # Should not detect drift (same distribution)
        assert drift['feature1']['p_value'] > 0.05
    
    def test_detect_drift(self):
        """Test drift detection when drift exists"""
        tracker = DriftTracker()
        
        # Different distributions
        for i in range(100):
            val = np.random.normal(0, 1)
            tracker.record_baseline({'feature1': val}, prediction=1)
        
        for i in range(100):
            val = np.random.normal(5, 1)  # Shifted mean
            tracker.record_current({'feature1': val}, prediction=1)
        
        drift = tracker.detect_feature_drift(threshold=0.05)
        assert 'feature1' in drift
        # Should detect drift
        assert drift['feature1']['has_drift'] == True
        assert drift['feature1']['p_value'] < 0.05
    
    def test_insufficient_data(self):
        """Test drift detection with insufficient data"""
        tracker = DriftTracker()
        
        # Only 10 samples (need 30+)
        for i in range(10):
            tracker.record_baseline({'feature1': float(i)}, prediction=1)
            tracker.record_current({'feature1': float(i + 1)}, prediction=1)
        
        drift = tracker.detect_feature_drift()
        assert len(drift) == 0  # No results due to insufficient data


class TestFairnessMetrics:
    """Test FairnessMetrics class"""
    
    def test_initialization(self):
        """Test metrics initialization"""
        metrics = FairnessMetrics()
        assert len(metrics.predictions_by_group) == 0
    
    def test_record_predictions(self):
        """Test recording predictions by group"""
        metrics = FairnessMetrics()
        metrics.record('group_a', 1, 1)
        metrics.record('group_b', 0, 1)
        
        assert 'group_a' in metrics.predictions_by_group
        assert 'group_b' in metrics.predictions_by_group
    
    def test_fairness_calculation(self):
        """Test fairness metrics calculation"""
        metrics = FairnessMetrics()
        
        # Group A: 80% accuracy
        for i in range(100):
            metrics.record('group_a', 1 if i < 80 else 0, 1)
        
        # Group B: 70% accuracy
        for i in range(100):
            metrics.record('group_b', 1 if i < 70 else 0, 1)
        
        fairness = metrics.get_fairness_metrics()
        assert 'group_a' in fairness
        assert 'group_b' in fairness
        assert abs(fairness['group_a']['accuracy'] - 0.80) < 0.01
        assert abs(fairness['group_b']['accuracy'] - 0.70) < 0.01
    
    def test_disparate_impact(self):
        """Test disparate impact calculation"""
        metrics = FairnessMetrics()
        
        # Group A: 90% positive rate
        for i in range(100):
            metrics.record('group_a', 1 if i < 90 else 0, 1)
        
        # Group B: 75% positive rate
        for i in range(100):
            metrics.record('group_b', 1 if i < 75 else 0, 1)
        
        fairness = metrics.get_fairness_metrics()
        assert '_overall' in fairness
        
        # Disparate impact = 0.75 / 0.90 = 0.833 (> 0.8 = fair)
        di = fairness['_overall']['disparate_impact']
        assert abs(di - 0.833) < 0.01
        assert fairness['_overall']['is_fair'] == True


class TestMLMonitor:
    """Test MLMonitor main class"""
    
    def test_initialization(self):
        """Test monitor initialization"""
        monitor = MLMonitor()
        assert monitor.accuracy is not None
        assert monitor.latency is not None
        assert monitor.throughput is not None
        assert monitor.drift is not None
        assert monitor.fairness is not None
    
    def test_record_prediction(self):
        """Test recording a prediction"""
        monitor = MLMonitor()
        
        monitor.record_prediction(
            model_name='test_model',
            model_version='v1.0',
            prediction=1,
            confidence=0.95,
            features={'feature1': 1.0, 'feature2': 2.0},
            true_label=1,
            latency_ms=5.0
        )
        
        assert len(monitor.prediction_log) == 1
        assert monitor.accuracy.get_accuracy() == 1.0
    
    def test_record_multiple_predictions(self):
        """Test recording multiple predictions"""
        monitor = MLMonitor()
        
        for i in range(100):
            monitor.record_prediction(
                model_name='test_model',
                model_version='v1.0',
                prediction=i % 2,
                confidence=0.9,
                features={'feature1': float(i)},
                true_label=i % 2,
                latency_ms=5.0 + i % 5
            )
        
        assert len(monitor.prediction_log) == 100
        assert monitor.accuracy.get_accuracy() == 1.0
    
    def test_custom_metrics(self):
        """Test custom metrics"""
        monitor = MLMonitor()
        
        monitor.add_custom_metric('custom_score', window_size=100)
        monitor.record_custom_metric('custom_score', 0.85)
        monitor.record_custom_metric('custom_score', 0.90)
        
        assert 'custom_score' in monitor.custom_metrics
        stats = monitor.custom_metrics['custom_score'].get_stats()
        assert stats['count'] == 2
    
    def test_get_health_status(self):
        """Test health status check"""
        monitor = MLMonitor()
        
        # Add good predictions
        for i in range(100):
            monitor.record_prediction(
                model_name='test_model',
                model_version='v1.0',
                prediction=1,
                confidence=0.95,
                features={'feature1': 1.0},
                true_label=1,
                latency_ms=5.0
            )
        
        health = monitor.get_health_status()
        assert health['status'] == 'healthy'
        assert len(health['issues']) == 0
    
    def test_get_health_status_degraded(self):
        """Test degraded health status"""
        monitor = MLMonitor()
        
        # Add predictions with low accuracy
        for i in range(100):
            monitor.record_prediction(
                model_name='test_model',
                model_version='v1.0',
                prediction=1,
                confidence=0.95,
                features={'feature1': 1.0},
                true_label=0,  # Wrong labels
                latency_ms=5.0
            )
        
        health = monitor.get_health_status()
        assert health['status'] in ['degraded', 'unhealthy']
        assert len(health['issues']) > 0
    
    def test_comprehensive_report(self):
        """Test comprehensive report generation"""
        monitor = MLMonitor()
        
        # Add some data
        for i in range(50):
            monitor.record_prediction(
                model_name='test_model',
                model_version='v1.0',
                prediction=i % 2,
                confidence=0.9,
                features={'feature1': float(i)},
                true_label=i % 2,
                latency_ms=5.0
            )
        
        report = monitor.get_comprehensive_report()
        assert 'timestamp' in report
        assert 'accuracy' in report
        assert 'latency' in report
        assert 'throughput' in report
        assert 'drift' in report
        assert 'fairness' in report
        assert 'health' in report
    
    def test_prometheus_export(self):
        """Test Prometheus metrics export"""
        monitor = MLMonitor()
        
        # Add some data
        for i in range(50):
            monitor.record_prediction(
                model_name='test_model',
                model_version='v1.0',
                prediction=1,
                confidence=0.9,
                features={'feature1': 1.0},
                true_label=1,
                latency_ms=5.0
            )
        
        metrics = monitor.export_prometheus_metrics()
        assert 'ml_model_accuracy' in metrics
        assert 'ml_inference_latency_ms' in metrics
        assert 'ml_requests_per_second' in metrics
        assert 'ml_features_with_drift' in metrics
    
    def test_get_recent_predictions(self):
        """Test getting recent predictions"""
        monitor = MLMonitor()
        
        for i in range(200):
            monitor.record_prediction(
                model_name='test_model',
                model_version='v1.0',
                prediction=i % 2,
                confidence=0.9,
                features={'feature1': float(i)},
                true_label=i % 2,
                latency_ms=5.0
            )
        
        recent = monitor.get_recent_predictions(count=100)
        assert len(recent) == 100
        assert all(isinstance(pred, dict) for pred in recent)
    
    def test_reset_baseline(self):
        """Test resetting drift baseline"""
        monitor = MLMonitor()
        
        # Add baseline data
        for i in range(100):
            monitor.record_prediction(
                model_name='test_model',
                model_version='v1.0',
                prediction=1,
                confidence=0.9,
                features={'feature1': float(i)},
                is_baseline=True
            )
        
        # Add current data
        for i in range(50):
            monitor.record_prediction(
                model_name='test_model',
                model_version='v1.0',
                prediction=1,
                confidence=0.9,
                features={'feature1': float(i)},
                is_baseline=False
            )
        
        # Reset baseline
        monitor.reset_baseline()
        
        # Current features should be cleared
        assert len(monitor.drift.current_features) == 0
    
    def test_thread_safety(self):
        """Test thread safety of monitoring"""
        import threading
        
        monitor = MLMonitor()
        
        def record_predictions():
            for i in range(100):
                monitor.record_prediction(
                    model_name='test_model',
                    model_version='v1.0',
                    prediction=i % 2,
                    confidence=0.9,
                    features={'feature1': float(i)},
                    true_label=i % 2,
                    latency_ms=5.0
                )
        
        # Run multiple threads
        threads = [threading.Thread(target=record_predictions) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have 500 predictions total
        assert len(monitor.prediction_log) == 500


class TestModelPrediction:
    """Test ModelPrediction dataclass"""
    
    def test_initialization(self):
        """Test prediction record initialization"""
        pred = ModelPrediction(
            timestamp=time.time(),
            model_name='test_model',
            model_version='v1.0',
            prediction=1,
            confidence=0.95,
            features={'feature1': 1.0},
            true_label=1,
            latency_ms=5.0
        )
        
        assert pred.model_name == 'test_model'
        assert pred.model_version == 'v1.0'
        assert pred.prediction == 1
        assert pred.confidence == 0.95
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        pred = ModelPrediction(
            timestamp=time.time(),
            model_name='test_model',
            model_version='v1.0',
            prediction=1,
            confidence=0.95,
            features={'feature1': 1.0},
            true_label=1,
            latency_ms=5.0
        )
        
        pred_dict = pred.to_dict()
        assert isinstance(pred_dict, dict)
        assert pred_dict['model_name'] == 'test_model'
        assert pred_dict['prediction'] == '1'
        assert pred_dict['confidence'] == 0.95


class TestMetricSnapshot:
    """Test MetricSnapshot dataclass"""
    
    def test_initialization(self):
        """Test snapshot initialization"""
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            value=0.95,
            labels={'model': 'test', 'version': 'v1'}
        )
        
        assert snapshot.value == 0.95
        assert snapshot.labels['model'] == 'test'
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            value=0.95,
            labels={'model': 'test'}
        )
        
        snapshot_dict = snapshot.to_dict()
        assert isinstance(snapshot_dict, dict)
        assert snapshot_dict['value'] == 0.95
        assert 'timestamp' in snapshot_dict


class TestIntegration:
    """Integration tests for full monitoring pipeline"""
    
    def test_full_monitoring_pipeline(self):
        """Test complete monitoring workflow"""
        monitor = MLMonitor()
        
        # Simulate production workload
        np.random.seed(42)
        for i in range(1000):
            # Generate features
            features = {
                'feature1': np.random.normal(0, 1),
                'feature2': np.random.normal(0, 1)
            }
            
            # Simulate model prediction
            prediction = 1 if features['feature1'] > 0 else 0
            true_label = prediction if np.random.rand() > 0.1 else 1 - prediction
            
            # Record
            monitor.record_prediction(
                model_name='production_model',
                model_version='v2.0',
                prediction=prediction,
                confidence=0.85 + np.random.rand() * 0.15,
                features=features,
                true_label=true_label,
                latency_ms=5.0 + np.random.rand() * 5.0,
                is_baseline=i < 500
            )
        
        # Get comprehensive report
        report = monitor.get_comprehensive_report()
        
        # Verify all components working
        assert report['accuracy']['accuracy'] > 0.8
        assert report['latency']['p99'] < 15.0
        assert report['throughput']['requests_per_second'] > 0
        assert report['health']['status'] in ['healthy', 'degraded', 'unhealthy']
        
        # Export Prometheus metrics
        prom_metrics = monitor.export_prometheus_metrics()
        assert len(prom_metrics) > 0
        assert 'ml_model_accuracy' in prom_metrics
    
    def test_drift_detection_pipeline(self):
        """Test drift detection in production scenario"""
        monitor = MLMonitor()
        
        # Baseline data (normal distribution)
        np.random.seed(42)
        for i in range(500):
            features = {'feature1': np.random.normal(0, 1)}
            monitor.record_prediction(
                model_name='test',
                model_version='v1',
                prediction=1,
                confidence=0.9,
                features=features,
                is_baseline=True
            )
        
        # Current data (shifted distribution - drift!)
        for i in range(100):
            features = {'feature1': np.random.normal(3, 1)}  # Mean shifted
            monitor.record_prediction(
                model_name='test',
                model_version='v1',
                prediction=1,
                confidence=0.9,
                features=features,
                is_baseline=False
            )
        
        # Check for drift
        report = monitor.get_comprehensive_report()
        drift_results = report['drift']['features']
        
        assert 'feature1' in drift_results
        assert drift_results['feature1']['has_drift'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
