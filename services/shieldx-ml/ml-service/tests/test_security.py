"""
Tests for security module (adversarial defense and poisoning detection)
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.adversarial import (
    FGSM, PGD, CarliniWagner,
    AdversarialDefense, EnsembleDefense, InputTransformDefense
)
from security.poisoning_detector import (
    DataValidator, ClusteringDetector, GradientAnalyzer,
    BackdoorDetector, PoisoningDetector
)


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, input_dim=10, hidden_dim=20, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TestFGSM(unittest.TestCase):
    """Test FGSM attack"""
    
    def setUp(self):
        self.model = SimpleModel()
        self.model.eval()
        self.epsilon = 0.1
        
    def test_fgsm_initialization(self):
        """Test FGSM can be initialized"""
        attack = FGSM(self.model, self.epsilon)
        self.assertEqual(attack.epsilon, self.epsilon)
        
    def test_generate_adversarial_examples(self):
        """Test generating adversarial examples"""
        attack = FGSM(self.model, self.epsilon)
        
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        
        x_adv = attack.generate(x, y)
        
        self.assertEqual(x_adv.shape, x.shape)
        self.assertFalse(torch.allclose(x_adv, x))
        
        # Perturbation should be bounded by epsilon
        perturbation = (x_adv - x).abs().max().item()
        self.assertLessEqual(perturbation, self.epsilon + 0.01)


class TestPGD(unittest.TestCase):
    """Test PGD attack"""
    
    def setUp(self):
        self.model = SimpleModel()
        self.model.eval()
        self.epsilon = 0.1
        
    def test_pgd_initialization(self):
        """Test PGD can be initialized"""
        attack = PGD(self.model, self.epsilon, num_iter=10)
        self.assertEqual(attack.epsilon, self.epsilon)
        self.assertEqual(attack.num_iter, 10)
        
    def test_generate_adversarial_examples(self):
        """Test generating adversarial examples"""
        attack = PGD(self.model, self.epsilon, num_iter=10)
        
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        
        x_adv = attack.generate(x, y)
        
        self.assertEqual(x_adv.shape, x.shape)
        self.assertFalse(torch.allclose(x_adv, x))
        
        # Perturbation should be bounded
        perturbation = (x_adv - x).abs().max().item()
        self.assertLessEqual(perturbation, self.epsilon + 0.01)


class TestCarliniWagner(unittest.TestCase):
    """Test C&W attack"""
    
    def setUp(self):
        self.model = SimpleModel()
        self.model.eval()
        self.epsilon = 0.3
        
    def test_cw_initialization(self):
        """Test C&W can be initialized"""
        attack = CarliniWagner(self.model, self.epsilon, num_iter=20)
        self.assertEqual(attack.epsilon, self.epsilon)
        self.assertEqual(attack.num_iter, 20)
        
    def test_generate_adversarial_examples(self):
        """Test generating adversarial examples"""
        attack = CarliniWagner(self.model, self.epsilon, num_iter=20)
        
        x = torch.randn(3, 10)
        y = torch.randint(0, 2, (3,))
        
        x_adv = attack.generate(x, y)
        
        self.assertEqual(x_adv.shape, x.shape)
        self.assertTrue(torch.isfinite(x_adv).all())


class TestAdversarialDefense(unittest.TestCase):
    """Test adversarial defense"""
    
    def setUp(self):
        self.model = SimpleModel()
        self.epsilon = 0.1
        
    def test_defense_initialization(self):
        """Test defense can be initialized"""
        defense = AdversarialDefense(self.model, attack_type='fgsm', epsilon=self.epsilon)
        self.assertIsNotNone(defense.attack)
        
    def test_generate_adversarial_batch(self):
        """Test generating mixed batch"""
        defense = AdversarialDefense(self.model, attack_type='pgd', epsilon=self.epsilon, mix_ratio=0.5)
        
        x = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        
        x_mixed, y_mixed = defense.generate_adversarial_batch(x, y)
        
        self.assertEqual(x_mixed.shape, x.shape)
        self.assertEqual(y_mixed.shape, y.shape)
        
    def test_adversarial_training_step(self):
        """Test adversarial training step"""
        defense = AdversarialDefense(self.model, attack_type='fgsm', epsilon=self.epsilon)
        
        x = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        metrics = defense.adversarial_training_step(x, y, optimizer, criterion)
        
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertGreaterEqual(metrics['loss'], 0)
        
    def test_evaluate_robustness(self):
        """Test robustness evaluation"""
        defense = AdversarialDefense(self.model, attack_type='fgsm', epsilon=self.epsilon)
        
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        
        metrics = defense.evaluate_robustness(x, y)
        
        self.assertIn('clean_accuracy', metrics)
        self.assertIn('adversarial_accuracy', metrics)
        self.assertIn('robustness_drop', metrics)
        self.assertIn('avg_perturbation', metrics)


class TestEnsembleDefense(unittest.TestCase):
    """Test ensemble defense"""
    
    def setUp(self):
        self.models = [SimpleModel() for _ in range(3)]
        
    def test_ensemble_initialization(self):
        """Test ensemble can be initialized"""
        defense = EnsembleDefense(self.models)
        self.assertEqual(len(defense.models), 3)
        
    def test_ensemble_prediction(self):
        """Test ensemble prediction"""
        defense = EnsembleDefense(self.models)
        
        x = torch.randn(5, 10)
        pred = defense.predict(x, method='voting')
        
        self.assertEqual(pred.shape[0], x.shape[0])
        
    def test_different_aggregation_methods(self):
        """Test different aggregation methods"""
        defense = EnsembleDefense(self.models)
        x = torch.randn(5, 10)
        
        pred_voting = defense.predict(x, method='voting')
        pred_average = defense.predict(x, method='average')
        pred_max = defense.predict(x, method='max')
        
        self.assertEqual(pred_voting.shape, pred_average.shape)
        self.assertEqual(pred_average.shape, pred_max.shape)


class TestInputTransformDefense(unittest.TestCase):
    """Test input transform defense"""
    
    def setUp(self):
        self.defense = InputTransformDefense()
        
    def test_bit_depth_reduction(self):
        """Test bit depth reduction"""
        x = torch.randn(5, 10)
        x_reduced = self.defense.bit_depth_reduction(x, bits=5)
        
        self.assertEqual(x_reduced.shape, x.shape)
        self.assertTrue(torch.isfinite(x_reduced).all())
        
    def test_jpeg_compression(self):
        """Test JPEG compression simulation"""
        x = torch.randn(5, 20)
        x_compressed = self.defense.jpeg_compression(x, quality=75)
        
        self.assertEqual(x_compressed.shape, x.shape)
        
    def test_apply_transforms(self):
        """Test applying all transforms"""
        x = torch.randn(5, 20)
        x_transformed = self.defense.apply_transforms(x)
        
        self.assertEqual(x_transformed.shape, x.shape)
        self.assertTrue(torch.isfinite(x_transformed).all())


class TestDataValidator(unittest.TestCase):
    """Test data validator"""
    
    def setUp(self):
        self.validator = DataValidator(contamination=0.1)
        
    def test_validator_initialization(self):
        """Test validator can be initialized"""
        self.assertEqual(self.validator.contamination, 0.1)
        self.assertFalse(self.validator.fitted)
        
    def test_fit_and_detect(self):
        """Test fitting and detection"""
        # Create clean data
        X_clean = np.random.randn(100, 10)
        
        # Fit validator
        self.validator.fit(X_clean)
        self.assertTrue(self.validator.fitted)
        
        # Add outliers
        X_test = np.vstack([
            X_clean[:90],
            np.random.randn(10, 10) * 5  # Outliers
        ])
        
        outliers = self.validator.detect_outliers(X_test)
        
        self.assertEqual(len(outliers), 100)
        self.assertTrue(outliers.sum() > 0)
        
    def test_validate_batch(self):
        """Test batch validation"""
        X = np.random.randn(100, 10)
        self.validator.fit(X)
        
        report = self.validator.validate_batch(X)
        
        self.assertIn('total_samples', report)
        self.assertIn('outliers_detected', report)
        self.assertIn('outlier_ratio', report)


class TestClusteringDetector(unittest.TestCase):
    """Test clustering detector"""
    
    def setUp(self):
        self.detector = ClusteringDetector(eps=0.5, min_samples=5)
        
    def test_detector_initialization(self):
        """Test detector can be initialized"""
        self.assertIsNotNone(self.detector.dbscan)
        
    def test_detect_poisoned_clusters(self):
        """Test detecting poisoned clusters"""
        # Create data with distinct clusters
        X1 = np.random.randn(50, 10) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        X2 = np.random.randn(50, 10) + np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        X = np.vstack([X1, X2])
        
        # Labels (mostly consistent per cluster)
        y = np.concatenate([np.zeros(50), np.ones(50)])
        
        report = self.detector.detect_poisoned_clusters(X, y)
        
        self.assertIn('num_clusters', report)
        self.assertIn('clusters', report)
        self.assertIn('suspicious_samples', report)


class TestGradientAnalyzer(unittest.TestCase):
    """Test gradient analyzer"""
    
    def setUp(self):
        self.model = SimpleModel()
        self.analyzer = GradientAnalyzer(self.model, threshold=3.0)
        
    def test_analyzer_initialization(self):
        """Test analyzer can be initialized"""
        self.assertEqual(self.analyzer.threshold, 3.0)
        
    def test_compute_sample_gradients(self):
        """Test computing sample gradients"""
        x = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        criterion = nn.CrossEntropyLoss()
        
        grad_norms = self.analyzer.compute_sample_gradients(x, y, criterion)
        
        self.assertEqual(len(grad_norms), 10)
        self.assertTrue(torch.isfinite(grad_norms).all())
        
    def test_detect_anomalous_gradients(self):
        """Test detecting anomalous gradients"""
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        criterion = nn.CrossEntropyLoss()
        
        report = self.analyzer.detect_anomalous_gradients(x, y, criterion)
        
        self.assertIn('mean_gradient', report)
        self.assertIn('num_outliers', report)
        self.assertIn('gradient_norms', report)


class TestBackdoorDetector(unittest.TestCase):
    """Test backdoor detector"""
    
    def setUp(self):
        self.model = SimpleModel()
        self.detector = BackdoorDetector(self.model, target_class=1)
        
    def test_detector_initialization(self):
        """Test detector can be initialized"""
        self.assertEqual(self.detector.target_class, 1)
        
    def test_reverse_engineer_trigger(self):
        """Test reverse engineering trigger"""
        x = torch.randn(10, 10)
        
        trigger = self.detector.reverse_engineer_trigger(x, num_iterations=20)
        
        self.assertEqual(trigger.shape, (10,))
        self.assertTrue(torch.isfinite(trigger).all())
        
    def test_detect_backdoor(self):
        """Test backdoor detection"""
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        
        report = self.detector.detect_backdoor(x, y)
        
        self.assertIn('backdoor_detected', report)
        self.assertIn('success_rate', report)
        self.assertIn('trigger_pattern', report)


class TestPoisoningDetector(unittest.TestCase):
    """Test comprehensive poisoning detector"""
    
    def setUp(self):
        self.model = SimpleModel()
        self.detector = PoisoningDetector(self.model, contamination=0.1)
        
    def test_detector_initialization(self):
        """Test detector can be initialized"""
        self.assertIsNotNone(self.detector.data_validator)
        self.assertIsNotNone(self.detector.clustering_detector)
        self.assertIsNotNone(self.detector.gradient_analyzer)
        
    def test_validate_training_data(self):
        """Test validating training data"""
        # Create mostly clean data with some outliers
        X_clean = np.random.randn(90, 10)
        X_poison = np.random.randn(10, 10) * 5
        X = np.vstack([X_clean, X_poison])
        
        y = np.concatenate([np.zeros(90), np.ones(10)])
        
        report = self.detector.validate_training_data(X, y)
        
        self.assertIn('total_samples', report)
        self.assertIn('total_suspicious', report)
        self.assertIn('recommendation', report)
        
    def test_monitor_training_batch(self):
        """Test monitoring training batch"""
        x = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        criterion = nn.CrossEntropyLoss()
        
        report = self.detector.monitor_training_batch(x, y, criterion)
        
        self.assertIn('batch_size', report)
        self.assertIn('anomalous_gradients', report)
        self.assertIn('alert', report)
        
    def test_detect_backdoors(self):
        """Test detecting backdoors for all classes"""
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        
        report = self.detector.detect_backdoors(x, y, num_classes=2)
        
        self.assertIn('backdoor_detected', report)
        self.assertIn('per_class_results', report)
        self.assertIn('recommendation', report)
        self.assertEqual(len(report['per_class_results']), 2)


class TestSecurityIntegration(unittest.TestCase):
    """Integration tests for security module"""
    
    def setUp(self):
        self.model = SimpleModel()
        
    def test_adversarial_defense_pipeline(self):
        """Test complete adversarial defense pipeline"""
        # Create defense
        defense = AdversarialDefense(self.model, attack_type='pgd', epsilon=0.1)
        
        # Training data
        x = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Training step
        metrics = defense.adversarial_training_step(x, y, optimizer, criterion)
        self.assertIn('loss', metrics)
        
        # Evaluate robustness
        robustness = defense.evaluate_robustness(x, y)
        self.assertIn('clean_accuracy', robustness)
        self.assertIn('adversarial_accuracy', robustness)
        
    def test_poisoning_detection_pipeline(self):
        """Test complete poisoning detection pipeline"""
        # Create detector
        detector = PoisoningDetector(self.model)
        
        # Training data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Validate data
        validation_report = detector.validate_training_data(X, y)
        self.assertIn('recommendation', validation_report)
        
        # Monitor training
        x_batch = torch.FloatTensor(X[:10])
        y_batch = torch.LongTensor(y[:10])
        criterion = nn.CrossEntropyLoss()
        
        monitoring_report = detector.monitor_training_batch(x_batch, y_batch, criterion)
        self.assertIn('alert', monitoring_report)


def run_tests():
    """Run all security tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFGSM))
    suite.addTests(loader.loadTestsFromTestCase(TestPGD))
    suite.addTests(loader.loadTestsFromTestCase(TestCarliniWagner))
    suite.addTests(loader.loadTestsFromTestCase(TestAdversarialDefense))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleDefense))
    suite.addTests(loader.loadTestsFromTestCase(TestInputTransformDefense))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestClusteringDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestGradientAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestBackdoorDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestPoisoningDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
