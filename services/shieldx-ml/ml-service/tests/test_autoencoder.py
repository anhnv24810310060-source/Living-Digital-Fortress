"""
Unit tests for Autoencoder models
"""

import unittest
import numpy as np
import torch
import tempfile
import os

from models.autoencoder import BasicAutoencoder, AnomalyDetectionAE


class TestBasicAutoencoder(unittest.TestCase):
    """Test BasicAutoencoder model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 50
        self.latent_dim = 10
        self.hidden_dims = [32, 16]
        self.batch_size = 32
        
        self.model = BasicAutoencoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        )
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.latent_dim, self.latent_dim)
        self.assertEqual(self.model.hidden_dims, self.hidden_dims)
    
    def test_forward_pass(self):
        """Test forward pass"""
        x = torch.randn(self.batch_size, self.input_dim)
        output = self.model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
    
    def test_encode(self):
        """Test encoding"""
        x = torch.randn(self.batch_size, self.input_dim)
        latent = self.model.encode(x)
        
        self.assertEqual(latent.shape, (self.batch_size, self.latent_dim))
    
    def test_decode(self):
        """Test decoding"""
        z = torch.randn(self.batch_size, self.latent_dim)
        reconstruction = self.model.decode(z)
        
        self.assertEqual(reconstruction.shape, (self.batch_size, self.input_dim))
    
    def test_encode_decode_cycle(self):
        """Test encode-decode cycle"""
        x = torch.randn(self.batch_size, self.input_dim)
        latent = self.model.encode(x)
        reconstruction = self.model.decode(latent)
        
        self.assertEqual(reconstruction.shape, x.shape)


class TestAnomalyDetectionAE(unittest.TestCase):
    """Test AnomalyDetectionAE system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 20
        self.n_samples = 1000
        self.n_anomalies = 50
        
        # Generate synthetic data
        np.random.seed(42)
        self.normal_data = np.random.randn(self.n_samples, self.input_dim)
        
        # Add some anomalies (with larger magnitude)
        self.anomalies = np.random.randn(self.n_anomalies, self.input_dim) * 5
        
        self.detector = AnomalyDetectionAE(
            input_dim=self.input_dim,
            latent_dim=8,
            hidden_dims=[16, 12]
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
            batch_size=128,
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
        self.detector.fit(self.normal_data[:800], epochs=5, verbose=False)
        
        test_data = self.normal_data[800:]
        errors = self.detector.reconstruction_error(test_data)
        
        self.assertEqual(len(errors), len(test_data))
        self.assertTrue(np.all(errors >= 0))
    
    def test_predict(self):
        """Test anomaly prediction"""
        # Train on normal data
        self.detector.fit(self.normal_data, epochs=10, verbose=False)
        
        # Test on normal data
        pred_normal = self.detector.predict(self.normal_data[:100])
        
        # Test on anomalies
        pred_anomaly = self.detector.predict(self.anomalies)
        
        # Most normal samples should be classified as normal (0)
        normal_rate = np.sum(pred_normal == 0) / len(pred_normal)
        self.assertGreater(normal_rate, 0.8)
        
        # Most anomalies should be detected (1)
        anomaly_rate = np.sum(pred_anomaly == 1) / len(pred_anomaly)
        self.assertGreater(anomaly_rate, 0.5)
    
    def test_predict_proba(self):
        """Test probability prediction"""
        self.detector.fit(self.normal_data, epochs=10, verbose=False)
        
        scores = self.detector.predict_proba(self.normal_data[:100])
        
        self.assertEqual(len(scores), 100)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))
    
    def test_save_load(self):
        """Test model save and load"""
        # Train model
        self.detector.fit(self.normal_data, epochs=5, verbose=False)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pt")
            self.detector.save(model_path)
            
            # Create new detector and load
            new_detector = AnomalyDetectionAE(input_dim=self.input_dim)
            new_detector.load(model_path)
            
            # Check that threshold is preserved
            self.assertEqual(new_detector.threshold, self.detector.threshold)
            
            # Check that predictions are consistent
            pred1 = self.detector.predict(self.normal_data[:10])
            pred2 = new_detector.predict(self.normal_data[:10])
            np.testing.assert_array_equal(pred1, pred2)
    
    def test_early_stopping(self):
        """Test early stopping"""
        detector = AnomalyDetectionAE(
            input_dim=self.input_dim,
            latent_dim=8
        )
        
        detector.fit(
            self.normal_data,
            epochs=100,
            early_stopping_patience=3,
            verbose=False
        )
        
        # Should stop before 100 epochs
        self.assertLess(len(detector.train_losses), 100)
    
    def test_different_architectures(self):
        """Test different hidden dimensions"""
        architectures = [
            [32, 16],
            [64, 32, 16],
            [128],
        ]
        
        for hidden_dims in architectures:
            detector = AnomalyDetectionAE(
                input_dim=self.input_dim,
                latent_dim=8,
                hidden_dims=hidden_dims
            )
            
            # Should train without errors
            detector.fit(self.normal_data[:500], epochs=5, verbose=False)
            
            # Should make predictions
            pred = detector.predict(self.normal_data[500:550])
            self.assertEqual(len(pred), 50)
    
    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error before fitting"""
        detector = AnomalyDetectionAE(input_dim=self.input_dim)
        
        with self.assertRaises(ValueError):
            detector.predict(self.normal_data[:10])


if __name__ == '__main__':
    unittest.main()
