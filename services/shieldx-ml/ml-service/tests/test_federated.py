"""
Tests for enhanced federated learning
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from federated.enhanced_fl import (
    FederatedConfig,
    DifferentialPrivacy,
    SecureAggregator,
    ByzantineDefense,
    ModelCompressor,
    FederatedLearningServer
)


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestDifferentialPrivacy(unittest.TestCase):
    """Test differential privacy mechanism"""
    
    def setUp(self):
        self.dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        
    def test_dp_initialization(self):
        """Test DP can be initialized"""
        self.assertEqual(self.dp.epsilon, 1.0)
        self.assertEqual(self.dp.delta, 1e-5)
        self.assertEqual(self.dp.clip_norm, 1.0)
        
    def test_clip_gradients(self):
        """Test gradient clipping"""
        # Large gradient
        gradients = torch.randn(100) * 10
        clipped = self.dp.clip_gradients(gradients)
        
        # Norm should be <= clip_norm
        norm = torch.norm(clipped).item()
        self.assertLessEqual(norm, self.dp.clip_norm + 0.01)
        
    def test_add_noise(self):
        """Test noise addition"""
        tensor = torch.ones(50)
        noised = self.dp.add_noise(tensor)
        
        # Should be different
        self.assertFalse(torch.allclose(tensor, noised))
        
        # Should still be close (with high probability)
        diff = torch.norm(tensor - noised).item()
        self.assertLess(diff, 10.0)  # Reasonable bound
        
    def test_privatize_gradients(self):
        """Test full privatization"""
        gradients = torch.randn(100) * 5
        private = self.dp.privatize_gradients(gradients)
        
        # Should be clipped and noised
        self.assertFalse(torch.allclose(gradients, private))
        self.assertTrue(torch.isfinite(private).all())


class TestSecureAggregator(unittest.TestCase):
    """Test secure aggregation"""
    
    def setUp(self):
        self.aggregator = SecureAggregator(num_clients=5)
        
    def test_aggregator_initialization(self):
        """Test aggregator can be initialized"""
        self.assertEqual(self.aggregator.num_clients, 5)
        
    def test_generate_keypair(self):
        """Test keypair generation"""
        private_key, public_key = self.aggregator.generate_client_keypair('client_1')
        
        self.assertIsNotNone(private_key)
        self.assertIsNotNone(public_key)
        self.assertIn('client_1', self.aggregator.client_keys)
        
    def test_create_secret_shares(self):
        """Test secret sharing"""
        value = torch.tensor([1.0, 2.0, 3.0])
        shares = self.aggregator.create_secret_shares(value, num_shares=3)
        
        self.assertEqual(len(shares), 3)
        
        # Sum of shares should equal original
        reconstructed = sum(shares)
        self.assertTrue(torch.allclose(reconstructed, value))
        
    def test_aggregate_shares(self):
        """Test share aggregation"""
        shares = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([5.0, 6.0])
        ]
        
        aggregated = self.aggregator.aggregate_shares(shares)
        expected = torch.tensor([9.0, 12.0])
        
        self.assertTrue(torch.allclose(aggregated, expected))
        
    def test_secure_aggregate(self):
        """Test secure aggregation"""
        client_updates = {
            'client_1': torch.tensor([1.0, 2.0]),
            'client_2': torch.tensor([3.0, 4.0]),
            'client_3': torch.tensor([5.0, 6.0])
        }
        
        aggregated = self.aggregator.secure_aggregate(client_updates)
        expected = torch.tensor([3.0, 4.0])  # Average
        
        self.assertTrue(torch.allclose(aggregated, expected))


class TestByzantineDefense(unittest.TestCase):
    """Test Byzantine-robust aggregation"""
    
    def setUp(self):
        self.defense = ByzantineDefense(malicious_threshold=0.3)
        
    def test_defense_initialization(self):
        """Test defense can be initialized"""
        self.assertEqual(self.defense.malicious_threshold, 0.3)
        
    def test_compute_update_distances(self):
        """Test distance computation"""
        updates = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.1, 2.1]),
            torch.tensor([10.0, 20.0])  # Outlier
        ]
        
        distances = self.defense.compute_update_distances(updates)
        
        self.assertEqual(distances.shape, (3, 3))
        # Distance to self should be 0
        self.assertEqual(distances[0, 0], 0)
        # Symmetric
        self.assertAlmostEqual(distances[0, 1], distances[1, 0])
        
    def test_krum(self):
        """Test Krum aggregation"""
        # Create updates with one Byzantine
        updates = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.1, 2.1]),
            torch.tensor([1.2, 1.9]),
            torch.tensor([100.0, 200.0])  # Byzantine
        ]
        
        selected = self.defense.krum(updates, f=1)
        
        # Should not select the Byzantine update
        self.assertFalse(torch.allclose(selected, updates[-1]))
        
    def test_multi_krum(self):
        """Test Multi-Krum aggregation"""
        updates = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.1, 2.1]),
            torch.tensor([1.2, 1.9]),
            torch.tensor([100.0, 200.0])  # Byzantine
        ]
        
        aggregated = self.defense.multi_krum(updates, f=1, m=2)
        
        # Should average good updates
        self.assertTrue(torch.isfinite(aggregated).all())
        
    def test_median(self):
        """Test median aggregation"""
        updates = [
            torch.tensor([1.0, 5.0]),
            torch.tensor([2.0, 6.0]),
            torch.tensor([3.0, 7.0]),
            torch.tensor([100.0, 200.0])  # Byzantine
        ]
        
        median = self.defense.median(updates)
        
        # Median should be robust to outlier
        self.assertLess(median[0].item(), 10.0)
        self.assertLess(median[1].item(), 20.0)
        
    def test_trimmed_mean(self):
        """Test trimmed mean aggregation"""
        updates = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
            torch.tensor([4.0]),
            torch.tensor([100.0])  # Outlier
        ]
        
        trimmed = self.defense.trimmed_mean(updates, trim_ratio=0.2)
        
        # Should remove outlier
        self.assertLess(trimmed.item(), 10.0)


class TestModelCompressor(unittest.TestCase):
    """Test model compression"""
    
    def setUp(self):
        self.compressor = ModelCompressor(compression_ratio=0.1)
        
    def test_compressor_initialization(self):
        """Test compressor can be initialized"""
        self.assertEqual(self.compressor.compression_ratio, 0.1)
        
    def test_quantize(self):
        """Test quantization"""
        tensor = torch.randn(100)
        quantized, scale, zero_point = self.compressor.quantize(tensor, bits=8)
        
        # Should be uint8
        self.assertEqual(quantized.dtype, torch.uint8)
        
        # Dequantize
        dequantized = self.compressor.dequantize(quantized, scale, zero_point)
        
        # Should be approximately equal
        error = torch.abs(tensor - dequantized).mean().item()
        self.assertLess(error, 0.1)
        
    def test_sparsify(self):
        """Test sparsification"""
        tensor = torch.randn(100)
        sparse, mask = self.compressor.sparsify(tensor, sparsity=0.9)
        
        # Should have 10% non-zero values
        nonzero_ratio = (mask != 0).float().mean().item()
        self.assertAlmostEqual(nonzero_ratio, 0.1, delta=0.01)
        
    def test_compress_decompress(self):
        """Test compression and decompression"""
        tensor = torch.randn(50, 50)
        
        # Compress
        compressed = self.compressor.compress(tensor)
        
        self.assertIn('quantized', compressed)
        self.assertIn('mask', compressed)
        self.assertIn('scale', compressed)
        
        # Decompress
        decompressed = self.compressor.decompress(compressed)
        
        # Should have same shape
        self.assertEqual(decompressed.shape, tensor.shape)
        
        # Check size reduction
        original_size = tensor.numel() * 4  # float32
        compressed_size = compressed['quantized'].numel() + compressed['mask'].numel()
        reduction = compressed_size / original_size
        
        self.assertLess(reduction, 0.2)  # At least 80% reduction


class TestFederatedLearningServer(unittest.TestCase):
    """Test FL server"""
    
    def setUp(self):
        self.model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=2)
        self.config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            num_rounds=5,
            local_epochs=2,
            learning_rate=0.01,
            epsilon=1.0,
            secure_aggregation=True,
            byzantine_robust=True
        )
        self.server = FederatedLearningServer(self.model, self.config)
        
    def test_server_initialization(self):
        """Test server can be initialized"""
        self.assertIsNotNone(self.server.model)
        self.assertIsNotNone(self.server.dp)
        self.assertIsNotNone(self.server.secure_agg)
        self.assertIsNotNone(self.server.byzantine_defense)
        
    def test_get_global_model(self):
        """Test getting global model"""
        global_model = self.server.get_global_model()
        
        self.assertIsInstance(global_model, dict)
        self.assertGreater(len(global_model), 0)
        
    def test_aggregate_updates(self):
        """Test update aggregation"""
        client_updates = {}
        
        for i in range(3):
            updates = {}
            for name, param in self.model.named_parameters():
                updates[name] = param.data.clone() + torch.randn_like(param.data) * 0.1
            client_updates[f'client_{i}'] = updates
        
        aggregated = self.server.aggregate_updates(client_updates)
        
        self.assertIsInstance(aggregated, dict)
        self.assertEqual(len(aggregated), len(list(self.model.named_parameters())))
        
    def test_update_global_model(self):
        """Test global model update"""
        old_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        new_params = {}
        for name, param in self.model.named_parameters():
            new_params[name] = param.data + torch.randn_like(param.data) * 0.1
        
        self.server.update_global_model(new_params)
        
        # Model should be updated
        for name, param in self.model.named_parameters():
            self.assertFalse(torch.allclose(param.data, old_params[name]))
        
    def test_train_round(self):
        """Test training round"""
        # Generate dummy client data
        client_data = {}
        for i in range(5):
            X = torch.randn(20, 10)
            y = torch.randint(0, 2, (20,))
            client_data[f'client_{i}'] = (X, y)
        
        # Run training round
        metrics = self.server.train_round(client_data)
        
        self.assertIn('round', metrics)
        self.assertIn('num_clients', metrics)
        self.assertEqual(metrics['num_clients'], 3)  # clients_per_round


class TestFederatedIntegration(unittest.TestCase):
    """Integration tests for federated learning"""
    
    def test_full_fl_pipeline(self):
        """Test complete FL training pipeline"""
        # Setup
        model = SimpleModel(input_dim=10)
        config = FederatedConfig(
            num_clients=5,
            clients_per_round=3,
            num_rounds=3,
            local_epochs=2
        )
        server = FederatedLearningServer(model, config)
        
        # Generate client data
        client_data = {}
        for i in range(5):
            X = torch.randn(20, 10)
            y = torch.randint(0, 2, (20,))
            client_data[f'client_{i}'] = (X, y)
        
        # Run multiple rounds
        for round_num in range(3):
            metrics = server.train_round(client_data)
            self.assertEqual(metrics['round'], round_num + 1)
        
        # Model should be updated
        self.assertEqual(server.global_round, 3)


def run_tests():
    """Run all FL tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDifferentialPrivacy))
    suite.addTests(loader.loadTestsFromTestCase(TestSecureAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestByzantineDefense))
    suite.addTests(loader.loadTestsFromTestCase(TestModelCompressor))
    suite.addTests(loader.loadTestsFromTestCase(TestFederatedLearningServer))
    suite.addTests(loader.loadTestsFromTestCase(TestFederatedIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
