"""
Unit tests for LSTM Autoencoder models
"""

import unittest
import numpy as np
import torch
import tempfile
import os

from models.lstm_autoencoder import LSTMAutoencoder, SequentialAnomalyDetector


class TestLSTMAutoencoder(unittest.TestCase):
    """Test LSTMAutoencoder model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 10
        self.hidden_dim = 32
        self.latent_dim = 16
        self.num_layers = 2
        self.batch_size = 16
        self.seq_len = 20
        
        self.model = LSTMAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers
        )
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.latent_dim, self.latent_dim)
        self.assertEqual(self.model.num_layers, self.num_layers)
    
    def test_forward_pass(self):
        """Test forward pass"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = self.model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.input_dim))
    
    def test_encode(self):
        """Test encoding"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        latent = self.model.encode(x)
        
        self.assertEqual(latent.shape, (self.batch_size, self.latent_dim))
    
    def test_decode(self):
        """Test decoding"""
        z = torch.randn(self.batch_size, self.latent_dim)
        reconstruction = self.model.decode(z, self.seq_len)
        
        self.assertEqual(reconstruction.shape, (self.batch_size, self.seq_len, self.input_dim))
    
    def test_bidirectional_model(self):
        """Test bidirectional LSTM"""
        model = LSTMAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            bidirectional=True
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.input_dim))


class TestSequentialAnomalyDetector(unittest.TestCase):
    """Test SequentialAnomalyDetector system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 5
        self.seq_len = 30
        self.n_samples = 500
        self.n_anomalies = 25
        
        # Generate synthetic sequential data
        np.random.seed(42)
        
        # Normal sequences (sinusoidal patterns)
        t = np.linspace(0, 4 * np.pi, self.seq_len)
        self.normal_data = np.array([
            np.sin(t[:, None] + np.random.randn(1, self.input_dim) * 0.5) + \
            np.random.randn(self.seq_len, self.input_dim) * 0.1
            for _ in range(self.n_samples)
        ])
        
        # Anomalous sequences (random walk with large steps)
        self.anomalies = np.array([
            np.cumsum(np.random.randn(self.seq_len, self.input_dim) * 2, axis=0)
            for _ in range(self.n_anomalies)
        ])
        
        self.detector = SequentialAnomalyDetector(
            input_dim=self.input_dim,
            hidden_dim=32,
            latent_dim=16,
            num_layers=2
        )
    
    def test_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.optimizer)
        self.assertIsNone(self.detector.threshold)
    
    def test_fit(self):
        """Test model training"""
        self.detector.fit(
            self.normal_data,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=False
        )
        
        # Check that threshold is set
        self.assertIsNotNone(self.detector.threshold)
        self.assertGreater(self.detector.threshold, 0)
        
        # Check that losses are recorded
        self.assertGreater(len(self.detector.train_losses), 0)
        self.assertGreater(len(self.detector.val_losses), 0)
    
    def test_reconstruction_error(self):
        """Test reconstruction error calculation"""
        self.detector.fit(self.normal_data[:400], epochs=5, verbose=False)
        
        test_data = self.normal_data[400:]
        errors = self.detector.reconstruction_error(test_data)
        
        self.assertEqual(len(errors), len(test_data))
        self.assertTrue(np.all(errors >= 0))
    
    def test_predict(self):
        """Test anomaly prediction"""
        # Train on normal data
        self.detector.fit(self.normal_data, epochs=10, verbose=False)
        
        # Test on normal data
        pred_normal = self.detector.predict(self.normal_data[:50])
        
        # Test on anomalies
        pred_anomaly = self.detector.predict(self.anomalies)
        
        # Most normal samples should be classified as normal (0)
        normal_rate = np.sum(pred_normal == 0) / len(pred_normal)
        self.assertGreater(normal_rate, 0.7)
        
        # Some anomalies should be detected
        anomaly_rate = np.sum(pred_anomaly == 1) / len(pred_anomaly)
        self.assertGreater(anomaly_rate, 0.2)
    
    def test_predict_proba(self):
        """Test probability prediction"""
        self.detector.fit(self.normal_data, epochs=10, verbose=False)
        
        scores = self.detector.predict_proba(self.normal_data[:50])
        
        self.assertEqual(len(scores), 50)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))
    
    def test_save_load(self):
        """Test model save and load"""
        # Train model
        self.detector.fit(self.normal_data, epochs=5, verbose=False)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_lstm_model.pt")
            self.detector.save(model_path)
            
            # Create new detector and load
            new_detector = SequentialAnomalyDetector(input_dim=self.input_dim)
            new_detector.load(model_path)
            
            # Check that threshold is preserved
            self.assertEqual(new_detector.threshold, self.detector.threshold)
            
            # Check that predictions are consistent
            pred1 = self.detector.predict(self.normal_data[:10])
            pred2 = new_detector.predict(self.normal_data[:10])
            np.testing.assert_array_equal(pred1, pred2)
    
    def test_early_stopping(self):
        """Test early stopping"""
        detector = SequentialAnomalyDetector(
            input_dim=self.input_dim,
            hidden_dim=32,
            latent_dim=16
        )
        
        detector.fit(
            self.normal_data,
            epochs=100,
            early_stopping_patience=3,
            verbose=False
        )
        
        # Should stop before 100 epochs
        self.assertLess(len(detector.train_losses), 100)
    
    def test_bidirectional_detector(self):
        """Test bidirectional LSTM detector"""
        detector = SequentialAnomalyDetector(
            input_dim=self.input_dim,
            hidden_dim=32,
            latent_dim=16,
            bidirectional=True
        )
        
        # Should train without errors
        detector.fit(self.normal_data[:400], epochs=5, verbose=False)
        
        # Should make predictions
        pred = detector.predict(self.normal_data[400:450])
        self.assertEqual(len(pred), 50)
    
    def test_different_sequence_lengths(self):
        """Test handling of different sequence lengths"""
        # Create sequences of different lengths
        seq_lengths = [10, 20, 30]
        
        for seq_len in seq_lengths:
            data = np.random.randn(100, seq_len, self.input_dim)
            
            detector = SequentialAnomalyDetector(
                input_dim=self.input_dim,
                hidden_dim=16,
                latent_dim=8
            )
            
            # Should train without errors
            detector.fit(data, epochs=3, verbose=False)
            
            # Should make predictions
            pred = detector.predict(data[:10])
            self.assertEqual(len(pred), 10)
    
    def test_invalid_input_shape_raises_error(self):
        """Test that invalid input shape raises error"""
        detector = SequentialAnomalyDetector(input_dim=self.input_dim)
        
        # 2D input should raise error (expecting 3D)
        invalid_data = np.random.randn(100, self.input_dim)
        
        with self.assertRaises(ValueError):
            detector.fit(invalid_data, epochs=1)
    
    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error before fitting"""
        detector = SequentialAnomalyDetector(input_dim=self.input_dim)
        
        with self.assertRaises(ValueError):
            detector.predict(self.normal_data[:10])


if __name__ == '__main__':
    unittest.main()
