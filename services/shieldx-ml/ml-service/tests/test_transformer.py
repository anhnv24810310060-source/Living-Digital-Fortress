"""
Unit tests for Transformer models
"""

import unittest
import numpy as np
import torch
import tempfile
import os

from models.transformer import TransformerEncoder, TransformerThreatDetector, PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):
    """Test Positional Encoding"""
    
    def test_positional_encoding(self):
        """Test positional encoding"""
        d_model = 64
        max_len = 100
        batch_size = 8
        seq_len = 50
        
        pe = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))


class TestTransformerEncoder(unittest.TestCase):
    """Test TransformerEncoder model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 10
        self.num_classes = 6
        self.d_model = 64
        self.nhead = 4
        self.seq_len = 30
        self.batch_size = 16
        
        self.model = TransformerEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=2,
            num_classes=self.num_classes
        )
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.d_model, self.d_model)
        self.assertEqual(self.model.num_classes, self.num_classes)
    
    def test_forward_pass(self):
        """Test forward pass"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = self.model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_forward_with_mask(self):
        """Test forward pass with mask"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        mask = torch.zeros(self.seq_len, self.seq_len)
        
        output = self.model(x, mask=mask)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_different_architectures(self):
        """Test different transformer configurations"""
        configs = [
            (128, 8, 4),  # d_model, nhead, num_layers
            (256, 8, 6),
            (64, 4, 2),
        ]
        
        for d_model, nhead, num_layers in configs:
            model = TransformerEncoder(
                input_dim=self.input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                num_classes=self.num_classes
            )
            
            x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
            output = model(x)
            
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))


class TestTransformerThreatDetector(unittest.TestCase):
    """Test TransformerThreatDetector system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 15
        self.num_classes = 6
        self.seq_len = 40
        self.n_samples = 300
        
        # Generate synthetic data
        np.random.seed(42)
        self.X = np.random.randn(self.n_samples, self.seq_len, self.input_dim)
        self.y = np.random.randint(0, self.num_classes, size=self.n_samples)
        
        self.detector = TransformerThreatDetector(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            d_model=64,
            nhead=4,
            num_layers=2
        )
    
    def test_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.optimizer)
        self.assertIsNotNone(self.detector.scheduler)
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
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(50), decimal=5)
    
    def test_save_load(self):
        """Test model save and load"""
        self.detector.fit(self.X[:200], self.y[:200], epochs=3, verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_transformer_model.pt")
            self.detector.save(model_path)
            
            # Create new detector and load
            new_detector = TransformerThreatDetector(
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
        detector = TransformerThreatDetector(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            d_model=64,
            nhead=4,
            num_layers=2
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
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduler"""
        initial_lr = self.detector.optimizer.param_groups[0]['lr']
        
        self.detector.fit(
            self.X[:200],
            self.y[:200],
            epochs=10,
            verbose=False
        )
        
        final_lr = self.detector.optimizer.param_groups[0]['lr']
        
        # LR should change due to CosineAnnealingLR
        self.assertNotEqual(initial_lr, final_lr)
    
    def test_invalid_input_shape_raises_error(self):
        """Test that invalid input shape raises error"""
        detector = TransformerThreatDetector(
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
        self.assertIn('SQL_Injection', self.detector.class_names)


if __name__ == '__main__':
    unittest.main()
