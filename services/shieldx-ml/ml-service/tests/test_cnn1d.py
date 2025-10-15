"""
Unit tests for CNN-1D models
"""

import unittest
import numpy as np
import torch
import tempfile
import os

from models.cnn1d import CNN1DClassifier, PacketThreatDetector


class TestCNN1DClassifier(unittest.TestCase):
    """Test CNN1DClassifier model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 10
        self.num_classes = 6
        self.seq_len = 100
        self.batch_size = 16
        
        self.model = CNN1DClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            max_seq_len=self.seq_len
        )
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.num_classes, self.num_classes)
    
    def test_forward_pass(self):
        """Test forward pass"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = self.model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_different_architectures(self):
        """Test different filter configurations"""
        configs = [
            ([32, 64], [3, 5]),
            ([64, 128, 256], [3, 5, 7]),
            ([128], [5]),
        ]
        
        for num_filters, kernel_sizes in configs:
            model = CNN1DClassifier(
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                num_filters=num_filters,
                kernel_sizes=kernel_sizes
            )
            
            x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
            output = model(x)
            
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))


class TestPacketThreatDetector(unittest.TestCase):
    """Test PacketThreatDetector system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 15
        self.num_classes = 6
        self.seq_len = 50
        self.n_samples = 300
        
        # Generate synthetic packet data
        np.random.seed(42)
        self.X = np.random.randn(self.n_samples, self.seq_len, self.input_dim)
        
        # Generate random labels
        self.y = np.random.randint(0, self.num_classes, size=self.n_samples)
        
        self.detector = PacketThreatDetector(
            input_dim=self.input_dim,
            num_classes=self.num_classes
        )
    
    def test_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.optimizer)
        self.assertEqual(self.detector.num_classes, self.num_classes)
    
    def test_fit(self):
        """Test model training"""
        self.detector.fit(
            self.X,
            self.y,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            verbose=False
        )
        
        # Check that losses are recorded
        self.assertGreater(len(self.detector.train_losses), 0)
        self.assertGreater(len(self.detector.val_losses), 0)
        self.assertGreater(len(self.detector.train_accuracies), 0)
    
    def test_predict(self):
        """Test prediction"""
        self.detector.fit(self.X[:200], self.y[:200], epochs=3, verbose=False)
        
        test_X = self.X[200:250]
        predictions = self.detector.predict(test_X)
        
        self.assertEqual(len(predictions), 50)
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions < self.num_classes))
    
    def test_predict_proba(self):
        """Test probability prediction"""
        self.detector.fit(self.X[:200], self.y[:200], epochs=3, verbose=False)
        
        test_X = self.X[200:250]
        probs = self.detector.predict_proba(test_X)
        
        self.assertEqual(probs.shape, (50, self.num_classes))
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))
        # Each row should sum to ~1
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(50), decimal=5)
    
    def test_save_load(self):
        """Test model save and load"""
        self.detector.fit(self.X[:200], self.y[:200], epochs=3, verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_cnn_model.pt")
            self.detector.save(model_path)
            
            # Create new detector and load
            new_detector = PacketThreatDetector(
                input_dim=self.input_dim,
                num_classes=self.num_classes
            )
            new_detector.load(model_path)
            
            # Check that predictions are consistent
            test_X = self.X[200:210]
            pred1 = self.detector.predict(test_X)
            pred2 = new_detector.predict(test_X)
            np.testing.assert_array_equal(pred1, pred2)
    
    def test_early_stopping(self):
        """Test early stopping"""
        detector = PacketThreatDetector(
            input_dim=self.input_dim,
            num_classes=self.num_classes
        )
        
        detector.fit(
            self.X,
            self.y,
            epochs=100,
            early_stopping_patience=3,
            verbose=False
        )
        
        # Should stop before 100 epochs
        self.assertLess(len(detector.train_losses), 100)
    
    def test_invalid_input_shape_raises_error(self):
        """Test that invalid input shape raises error"""
        detector = PacketThreatDetector(
            input_dim=self.input_dim,
            num_classes=self.num_classes
        )
        
        # 2D input should raise error
        invalid_X = np.random.randn(100, self.input_dim)
        invalid_y = np.random.randint(0, self.num_classes, size=100)
        
        with self.assertRaises(ValueError):
            detector.fit(invalid_X, invalid_y, epochs=1)
    
    def test_class_names(self):
        """Test class names"""
        self.assertEqual(len(self.detector.class_names), self.num_classes)
        self.assertIn('Normal', self.detector.class_names)
        self.assertIn('DDoS', self.detector.class_names)


if __name__ == '__main__':
    unittest.main()
