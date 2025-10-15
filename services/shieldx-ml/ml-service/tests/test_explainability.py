"""
Tests for explainability modules
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from explainability.shap_explainer import SHAPExplainer, ModelAgnosticExplainer
from explainability.lime_explainer import LIMEExplainer, LIMETextExplainer
from explainability.counterfactual import CounterfactualExplainer, ActionableInsights


class SimpleClassifier(nn.Module):
    """Simple classifier for testing"""
    def __init__(self, input_dim=10, hidden_dim=20, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TestSHAPExplainer(unittest.TestCase):
    """Test SHAP explainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 10
        self.num_samples = 100
        
        # Create model
        self.model = SimpleClassifier(self.input_dim)
        self.model.eval()
        
        # Generate background data
        self.background_data = np.random.randn(self.num_samples, self.input_dim).astype(np.float32)
        
    def test_shap_explainer_initialization(self):
        """Test SHAP explainer can be initialized"""
        explainer = SHAPExplainer(self.model, self.background_data)
        self.assertIsNotNone(explainer)
        self.assertIsNotNone(explainer.explainer)
        
    def test_explain_instance(self):
        """Test explaining a single instance"""
        explainer = SHAPExplainer(self.model, self.background_data)
        instance = np.random.randn(self.input_dim).astype(np.float32)
        
        shap_values = explainer.explain_instance(instance)
        
        self.assertEqual(shap_values.shape[0], self.input_dim)
        self.assertTrue(np.isfinite(shap_values).all())
        
    def test_explain_batch(self):
        """Test explaining multiple instances"""
        explainer = SHAPExplainer(self.model, self.background_data)
        instances = np.random.randn(5, self.input_dim).astype(np.float32)
        
        shap_values = explainer.explain_batch(instances)
        
        self.assertEqual(shap_values.shape[0], 5)
        self.assertEqual(shap_values.shape[1], self.input_dim)
        self.assertTrue(np.isfinite(shap_values).all())
        
    def test_get_feature_importance(self):
        """Test getting feature importance"""
        explainer = SHAPExplainer(self.model, self.background_data)
        instance = np.random.randn(self.input_dim).astype(np.float32)
        
        shap_values = explainer.explain_instance(instance)
        importance = explainer.get_feature_importance(shap_values)
        
        self.assertIn('features', importance)
        self.assertIn('importance', importance)
        self.assertEqual(len(importance['features']), self.input_dim)
        self.assertEqual(len(importance['importance']), self.input_dim)
        
        # Check sorted by importance
        importance_values = importance['importance']
        self.assertTrue(all(importance_values[i] >= importance_values[i+1] 
                           for i in range(len(importance_values)-1)))


class TestModelAgnosticExplainer(unittest.TestCase):
    """Test model-agnostic SHAP explainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 10
        self.model = SimpleClassifier(self.input_dim)
        self.model.eval()
        
    def test_explainer_initialization(self):
        """Test explainer can be initialized"""
        explainer = ModelAgnosticExplainer(self.model)
        self.assertIsNotNone(explainer)
        
    def test_explain_instance(self):
        """Test explaining a single instance"""
        explainer = ModelAgnosticExplainer(self.model)
        instance = np.random.randn(self.input_dim).astype(np.float32)
        
        shap_values = explainer.explain_instance(instance, n_samples=100)
        
        self.assertEqual(shap_values.shape[0], self.input_dim)
        self.assertTrue(np.isfinite(shap_values).all())


class TestLIMEExplainer(unittest.TestCase):
    """Test LIME explainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 10
        self.model = SimpleClassifier(self.input_dim)
        self.model.eval()
        
        self.feature_names = [f"feature_{i}" for i in range(self.input_dim)]
        
    def test_lime_explainer_initialization(self):
        """Test LIME explainer can be initialized"""
        explainer = LIMEExplainer(self.model, feature_names=self.feature_names)
        self.assertIsNotNone(explainer)
        self.assertIsNotNone(explainer.explainer)
        
    def test_explain_instance(self):
        """Test explaining a single instance"""
        explainer = LIMEExplainer(self.model, feature_names=self.feature_names)
        instance = np.random.randn(self.input_dim).astype(np.float32)
        
        explanation = explainer.explain_instance(instance, num_features=5)
        
        self.assertIn('features', explanation)
        self.assertIn('weights', explanation)
        self.assertIn('prediction', explanation)
        self.assertIn('intercept', explanation)
        
        self.assertLessEqual(len(explanation['features']), 5)
        self.assertEqual(len(explanation['features']), len(explanation['weights']))
        
    def test_explain_batch(self):
        """Test explaining multiple instances"""
        explainer = LIMEExplainer(self.model, feature_names=self.feature_names)
        instances = np.random.randn(3, self.input_dim).astype(np.float32)
        
        explanations = explainer.explain_batch(instances, num_features=5)
        
        self.assertEqual(len(explanations), 3)
        for exp in explanations:
            self.assertIn('features', exp)
            self.assertIn('weights', exp)
            
    def test_get_feature_importance_summary(self):
        """Test getting feature importance summary"""
        explainer = LIMEExplainer(self.model, feature_names=self.feature_names)
        instances = np.random.randn(10, self.input_dim).astype(np.float32)
        
        explanations = explainer.explain_batch(instances, num_features=5)
        summary = explainer.get_feature_importance_summary(explanations)
        
        self.assertIn('features', summary)
        self.assertIn('avg_importance', summary)
        self.assertIn('frequency', summary)
        
        self.assertEqual(len(summary['features']), len(summary['avg_importance']))
        self.assertEqual(len(summary['features']), len(summary['frequency']))


class TestLIMETextExplainer(unittest.TestCase):
    """Test LIME text explainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple text classifier
        class TextClassifier(nn.Module):
            def __init__(self, vocab_size=100, embedding_dim=20, hidden_dim=10):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.fc1 = nn.Linear(embedding_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, 2)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # x is token indices
                if isinstance(x, np.ndarray):
                    x = torch.LongTensor(x)
                x = self.embedding(x)
                x = x.mean(dim=1)  # Average pooling
                x = self.relu(self.fc1(x))
                return self.fc2(x)
        
        self.model = TextClassifier()
        self.model.eval()
        
    def test_text_explainer_initialization(self):
        """Test text explainer can be initialized"""
        explainer = LIMETextExplainer(self.model, class_names=['safe', 'malicious'])
        self.assertIsNotNone(explainer)
        self.assertIsNotNone(explainer.explainer)
        
    def test_explain_log_entry(self):
        """Test explaining a log entry"""
        explainer = LIMETextExplainer(self.model, class_names=['safe', 'malicious'])
        log_entry = "User login from IP 192.168.1.1"
        
        explanation = explainer.explain_log_entry(log_entry, num_features=5)
        
        self.assertIn('words', explanation)
        self.assertIn('weights', explanation)
        self.assertIn('prediction_probs', explanation)
        
        self.assertLessEqual(len(explanation['words']), 5)
        self.assertEqual(len(explanation['words']), len(explanation['weights']))


class TestCounterfactualExplainer(unittest.TestCase):
    """Test counterfactual explainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 10
        self.model = SimpleClassifier(self.input_dim)
        self.model.eval()
        
    def test_explainer_initialization(self):
        """Test explainer can be initialized"""
        explainer = CounterfactualExplainer(self.model, target_class=1)
        self.assertIsNotNone(explainer)
        self.assertEqual(explainer.target_class, 1)
        
    def test_generate_counterfactual(self):
        """Test generating counterfactual"""
        explainer = CounterfactualExplainer(self.model, target_class=1)
        
        # Create instance classified as class 0
        instance = np.random.randn(self.input_dim).astype(np.float32)
        
        counterfactual = explainer.generate_counterfactual(
            instance, 
            max_iterations=50,
            lr=0.1
        )
        
        self.assertEqual(counterfactual.shape, instance.shape)
        self.assertTrue(np.isfinite(counterfactual).all())
        
        # Counterfactual should be different from original
        self.assertFalse(np.allclose(counterfactual, instance))
        
    def test_generate_with_constraints(self):
        """Test generating counterfactual with constraints"""
        explainer = CounterfactualExplainer(self.model, target_class=1)
        instance = np.random.randn(self.input_dim).astype(np.float32)
        
        # Define constraints (first 3 features are immutable)
        feature_constraints = {
            i: {'mutable': False} for i in range(3)
        }
        
        counterfactual = explainer.generate_counterfactual(
            instance,
            feature_constraints=feature_constraints,
            max_iterations=50
        )
        
        # First 3 features should be unchanged
        self.assertTrue(np.allclose(counterfactual[:3], instance[:3]))
        
        # Other features can change
        self.assertFalse(np.allclose(counterfactual[3:], instance[3:]))
        
    def test_batch_generation(self):
        """Test generating multiple counterfactuals"""
        explainer = CounterfactualExplainer(self.model, target_class=1)
        instances = np.random.randn(3, self.input_dim).astype(np.float32)
        
        counterfactuals = explainer.generate_batch(instances, max_iterations=50)
        
        self.assertEqual(counterfactuals.shape, instances.shape)
        self.assertTrue(np.isfinite(counterfactuals).all())


class TestActionableInsights(unittest.TestCase):
    """Test actionable insights generator"""
    
    def test_generate_recommendations(self):
        """Test generating recommendations"""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        counterfactual = np.array([1.0, 2.5, 3.0, 3.0, 6.0])
        feature_names = [f"feature_{i}" for i in range(5)]
        
        insights = ActionableInsights.generate_recommendations(
            original,
            counterfactual,
            feature_names=feature_names
        )
        
        self.assertIn('num_changes', insights)
        self.assertIn('changes', insights)
        self.assertIn('total_distance', insights)
        
        # Should detect 3 changes (features 1, 3, 4)
        self.assertEqual(insights['num_changes'], 3)
        self.assertEqual(len(insights['changes']), 3)
        
        # Check change details
        for change in insights['changes']:
            self.assertIn('feature', change)
            self.assertIn('original_value', change)
            self.assertIn('suggested_value', change)
            self.assertIn('change', change)
            self.assertIn('change_magnitude', change)
            
    def test_generate_recommendations_no_names(self):
        """Test generating recommendations without feature names"""
        original = np.array([1.0, 2.0, 3.0])
        counterfactual = np.array([1.0, 2.5, 3.0])
        
        insights = ActionableInsights.generate_recommendations(
            original,
            counterfactual
        )
        
        self.assertEqual(insights['num_changes'], 1)
        self.assertEqual(insights['changes'][0]['feature'], 'feature_1')
        
    def test_sort_by_magnitude(self):
        """Test sorting changes by magnitude"""
        original = np.array([1.0, 2.0, 3.0])
        counterfactual = np.array([1.5, 3.0, 4.5])
        
        insights = ActionableInsights.generate_recommendations(
            original,
            counterfactual,
            sort_by_magnitude=True
        )
        
        # Changes should be sorted by magnitude
        magnitudes = [change['change_magnitude'] for change in insights['changes']]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))


class TestExplainabilityIntegration(unittest.TestCase):
    """Integration tests for explainability modules"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 10
        self.model = SimpleClassifier(self.input_dim)
        self.model.eval()
        
        self.instance = np.random.randn(self.input_dim).astype(np.float32)
        self.background_data = np.random.randn(50, self.input_dim).astype(np.float32)
        
    def test_multiple_explainers_same_instance(self):
        """Test using multiple explainers on same instance"""
        # SHAP
        shap_explainer = SHAPExplainer(self.model, self.background_data)
        shap_values = shap_explainer.explain_instance(self.instance)
        
        # LIME
        lime_explainer = LIMEExplainer(self.model)
        lime_explanation = lime_explainer.explain_instance(self.instance)
        
        # Counterfactual
        cf_explainer = CounterfactualExplainer(self.model, target_class=1)
        counterfactual = cf_explainer.generate_counterfactual(self.instance, max_iterations=30)
        
        # All should produce results
        self.assertIsNotNone(shap_values)
        self.assertIsNotNone(lime_explanation)
        self.assertIsNotNone(counterfactual)
        
        # Results should have correct shapes
        self.assertEqual(shap_values.shape[0], self.input_dim)
        self.assertIn('features', lime_explanation)
        self.assertEqual(counterfactual.shape, self.instance.shape)
        
    def test_explainer_consistency(self):
        """Test that explainers produce consistent results"""
        explainer = SHAPExplainer(self.model, self.background_data)
        
        # Explain same instance twice
        shap_values1 = explainer.explain_instance(self.instance)
        shap_values2 = explainer.explain_instance(self.instance)
        
        # Results should be very similar (may have small numerical differences)
        self.assertTrue(np.allclose(shap_values1, shap_values2, rtol=0.1))


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSHAPExplainer))
    suite.addTests(loader.loadTestsFromTestCase(TestModelAgnosticExplainer))
    suite.addTests(loader.loadTestsFromTestCase(TestLIMEExplainer))
    suite.addTests(loader.loadTestsFromTestCase(TestLIMETextExplainer))
    suite.addTests(loader.loadTestsFromTestCase(TestCounterfactualExplainer))
    suite.addTests(loader.loadTestsFromTestCase(TestActionableInsights))
    suite.addTests(loader.loadTestsFromTestCase(TestExplainabilityIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
