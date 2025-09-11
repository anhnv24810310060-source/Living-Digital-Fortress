import unittest
import json
import numpy as np
from datetime import datetime
from feature_store import FeatureStore, MLPipeline, PluginOutput
import tempfile
import os

class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        # Use in-memory SQLite for testing
        self.db_url = "sqlite:///:memory:"
        self.redis_url = "redis://localhost:6379/1"  # Use different DB for testing
        
        try:
            self.feature_store = FeatureStore(self.db_url, self.redis_url)
            self.ml_pipeline = MLPipeline(self.feature_store)
        except Exception as e:
            self.skipTest(f"Cannot connect to test databases: {e}")
    
    def test_plugin_output_normalization(self):
        """Test plugin output normalization to features"""
        plugin_output = PluginOutput(
            plugin_id="test_plugin",
            artifact_id="test_artifact",
            success=True,
            results={"is_malware": True, "threat_type": "trojan", "score": 0.9},
            confidence=0.85,
            tags=["malware", "trojan", "executable"],
            indicators=[
                {"type": "hash", "value": "abc123", "confidence": 0.9, "context": "SHA256"},
                {"type": "url", "value": "http://malicious.com", "confidence": 0.8, "context": "C2"}
            ],
            execution_time=150,
            timestamp=datetime.now()
        )
        
        features = self.feature_store.normalize_plugin_output(plugin_output)
        
        # Verify feature structure
        self.assertEqual(features.artifact_id, "test_artifact")
        self.assertEqual(features.plugin_id, "test_plugin")
        self.assertIsInstance(features.feature_vector, np.ndarray)
        self.assertIsInstance(features.feature_names, list)
        self.assertGreater(len(features.feature_names), 0)
        
        # Verify feature values
        self.assertGreater(len(features.feature_vector), 10)  # Should have multiple features
        
    def test_result_feature_extraction(self):
        """Test extraction of features from plugin results"""
        results = {
            "is_malware": True,
            "confidence_score": 0.9,
            "threat_family": "trojan",
            "nested": {"detection": True, "score": 0.8}
        }
        
        features = self.feature_store._extract_result_features(results)
        
        # Check extracted features
        self.assertEqual(features["result_is_malware"], 1.0)
        self.assertEqual(features["result_confidence_score"], 0.9)
        self.assertIn("result_threat_family_hash", features)
        self.assertEqual(features["result_nested_detection"], 1.0)
        self.assertEqual(features["result_nested_score"], 0.8)
    
    def test_tag_feature_extraction(self):
        """Test tag feature extraction and one-hot encoding"""
        tags = ["malware", "trojan", "executable", "custom_tag"]
        
        features = self.feature_store._extract_tag_features(tags)
        
        # Check known tags
        self.assertEqual(features["tag_malware"], 1.0)
        self.assertEqual(features["tag_trojan"], 1.0)
        self.assertEqual(features["tag_executable"], 1.0)
        self.assertEqual(features["tag_benign"], 0.0)
        
        # Check diversity score
        self.assertEqual(features["tag_diversity"], 1.0)  # All unique tags
    
    def test_indicator_feature_extraction(self):
        """Test indicator feature extraction"""
        indicators = [
            {"type": "hash", "confidence": 0.9},
            {"type": "url", "confidence": 0.8},
            {"type": "hash", "confidence": 0.7}
        ]
        
        features = self.feature_store._extract_indicator_features(indicators)
        
        # Check aggregated statistics
        self.assertAlmostEqual(features["avg_indicator_confidence"], 0.8, places=2)
        self.assertEqual(features["max_indicator_confidence"], 0.9)
        self.assertEqual(features["min_indicator_confidence"], 0.7)
        
        # Check type distribution
        self.assertAlmostEqual(features["indicator_type_hash"], 2/3, places=2)
        self.assertAlmostEqual(features["indicator_type_url"], 1/3, places=2)
    
    def test_feature_vectorization(self):
        """Test feature dictionary to vector conversion"""
        features = {
            "feature_a": 0.5,
            "feature_b": 1.0,
            "feature_c": 0.0
        }
        
        vector, names = self.feature_store._vectorize_features(features)
        
        # Check vector properties
        self.assertEqual(len(vector), 3)
        self.assertEqual(len(names), 3)
        self.assertEqual(names, ["feature_a", "feature_b", "feature_c"])
        self.assertTrue(np.array_equal(vector, [0.5, 1.0, 0.0]))
    
    def test_ml_pipeline_processing(self):
        """Test end-to-end ML pipeline processing"""
        plugin_output_json = json.dumps({
            "plugin_id": "test_plugin",
            "artifact_id": "test_artifact",
            "success": True,
            "results": {"is_malware": True},
            "confidence": 0.8,
            "tags": ["malware"],
            "indicators": [],
            "execution_time": 100,
            "timestamp": datetime.now().isoformat()
        })
        
        result = self.ml_pipeline.process_plugin_output(plugin_output_json)
        
        # Check processing result
        self.assertTrue(result["success"])
        self.assertIn("feature_count", result)
        self.assertIn("process_id", result)
        self.assertGreater(result["feature_count"], 0)
    
    def test_feature_validation(self):
        """Test feature validation and error handling"""
        # Test with invalid JSON
        invalid_json = "invalid json"
        result = self.ml_pipeline.process_plugin_output(invalid_json)
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        
        # Test with missing required fields
        incomplete_json = json.dumps({
            "plugin_id": "test",
            # Missing other required fields
        })
        result = self.ml_pipeline.process_plugin_output(incomplete_json)
        self.assertFalse(result["success"])
    
    def test_feature_consistency(self):
        """Test that same input produces consistent features"""
        plugin_output = PluginOutput(
            plugin_id="consistency_test",
            artifact_id="test_artifact",
            success=True,
            results={"test": True},
            confidence=0.5,
            tags=["test"],
            indicators=[],
            execution_time=100,
            timestamp=datetime.now()
        )
        
        # Generate features twice
        features1 = self.feature_store.normalize_plugin_output(plugin_output)
        features2 = self.feature_store.normalize_plugin_output(plugin_output)
        
        # Should be identical
        self.assertTrue(np.array_equal(features1.feature_vector, features2.feature_vector))
        self.assertEqual(features1.feature_names, features2.feature_names)

class TestFeatureEngineering(unittest.TestCase):
    def test_hash_consistency(self):
        """Test that feature hashing is consistent"""
        feature_store = FeatureStore("sqlite:///:memory:", "redis://localhost:6379/1")
        
        features1 = {"test": "value"}
        features2 = {"test": "value"}
        features3 = {"test": "different"}
        
        hash1 = feature_store._compute_feature_hash(features1)
        hash2 = feature_store._compute_feature_hash(features2)
        hash3 = feature_store._compute_feature_hash(features3)
        
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
    
    def test_nan_handling(self):
        """Test NaN value handling in feature vectors"""
        feature_store = FeatureStore("sqlite:///:memory:", "redis://localhost:6379/1")
        
        features = {"valid": 1.0, "invalid": float('nan'), "zero": 0.0}
        vector, names = feature_store._vectorize_features(features)
        
        # NaN should be converted to 0
        self.assertFalse(np.isnan(vector).any())
        self.assertEqual(vector[1], 0.0)  # NaN converted to 0

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)