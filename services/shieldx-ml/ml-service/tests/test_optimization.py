"""
Tests for model optimization
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.model_optimizer import (
    ModelQuantizer,
    ModelPruner,
    KnowledgeDistillation,
    ONNXExporter,
    ModelOptimizer
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


class DummyDataLoader:
    """Dummy data loader for testing"""
    def __init__(self, num_batches=5, batch_size=10, input_dim=10):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.input_dim = input_dim
        
    def __iter__(self):
        for _ in range(self.num_batches):
            X = torch.randn(self.batch_size, self.input_dim)
            y = torch.randint(0, 2, (self.batch_size,))
            yield X, y
            
    def __len__(self):
        return self.num_batches


class TestModelQuantizer(unittest.TestCase):
    """Test model quantization"""
    
    def setUp(self):
        self.model = SimpleModel()
        self.quantizer = ModelQuantizer()
        
    def test_quantizer_initialization(self):
        """Test quantizer can be initialized"""
        self.assertEqual(self.quantizer.backend, 'fbgemm')
        
    def test_dynamic_quantization(self):
        """Test dynamic quantization"""
        quantized = self.quantizer.quantize_dynamic(self.model)
        
        # Should return a model
        self.assertIsInstance(quantized, nn.Module)
        
        # Test inference
        x = torch.randn(5, 10)
        output = quantized(x)
        self.assertEqual(output.shape, (5, 2))
        
    def test_measure_size(self):
        """Test model size measurement"""
        metrics = self.quantizer.measure_size(self.model)
        
        self.assertIn('num_parameters', metrics)
        self.assertIn('size_mb', metrics)
        self.assertGreater(metrics['num_parameters'], 0)
        self.assertGreater(metrics['size_mb'], 0)
        
    def test_fp16_conversion(self):
        """Test FP16 conversion"""
        fp16_model = self.quantizer.convert_to_fp16(self.model)
        
        # Check parameter dtype
        for param in fp16_model.parameters():
            self.assertEqual(param.dtype, torch.float16)


class TestModelPruner(unittest.TestCase):
    """Test model pruning"""
    
    def setUp(self):
        self.model = SimpleModel()
        self.pruner = ModelPruner(target_sparsity=0.5)
        
    def test_pruner_initialization(self):
        """Test pruner can be initialized"""
        self.assertEqual(self.pruner.target_sparsity, 0.5)
        
    def test_unstructured_pruning(self):
        """Test unstructured pruning"""
        original_params = sum(p.numel() for p in self.model.parameters())
        
        pruned = self.pruner.prune_unstructured(self.model, method='l1')
        
        # Should still have same number of parameters
        pruned_params = sum(p.numel() for p in pruned.parameters())
        self.assertEqual(pruned_params, original_params)
        
        # But many should be zero
        sparsity = self.pruner.measure_sparsity(pruned)
        self.assertGreater(sparsity, 0.4)  # Close to 50%
        
    def test_structured_pruning(self):
        """Test structured pruning"""
        pruned = self.pruner.prune_structured(self.model, dim=0)
        
        sparsity = self.pruner.measure_sparsity(pruned)
        self.assertGreater(sparsity, 0)
        
    def test_measure_sparsity(self):
        """Test sparsity measurement"""
        # Unpruned model should have low sparsity
        sparsity = self.pruner.measure_sparsity(self.model)
        self.assertLess(sparsity, 0.1)
        
        # Pruned model should have high sparsity
        pruned = self.pruner.prune_unstructured(self.model, method='l1')
        sparsity = self.pruner.measure_sparsity(pruned)
        self.assertGreater(sparsity, 0.4)


class TestKnowledgeDistillation(unittest.TestCase):
    """Test knowledge distillation"""
    
    def setUp(self):
        self.teacher = SimpleModel(input_dim=10, hidden_dim=50, output_dim=2)
        self.student = SimpleModel(input_dim=10, hidden_dim=10, output_dim=2)
        self.kd = KnowledgeDistillation(self.teacher, self.student)
        
    def test_kd_initialization(self):
        """Test KD can be initialized"""
        self.assertEqual(self.kd.temperature, 3.0)
        self.assertEqual(self.kd.alpha, 0.5)
        
    def test_distillation_loss(self):
        """Test distillation loss calculation"""
        student_logits = torch.randn(5, 2)
        teacher_logits = torch.randn(5, 2)
        labels = torch.randint(0, 2, (5,))
        
        loss = self.kd.distillation_loss(student_logits, teacher_logits, labels)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))
        
    def test_train_student(self):
        """Test student training"""
        train_loader = DummyDataLoader(num_batches=3)
        
        initial_params = {name: param.clone() for name, param in self.student.named_parameters()}
        
        trained_student = self.kd.train_student(train_loader, num_epochs=2)
        
        # Parameters should be updated
        params_changed = False
        for name, param in trained_student.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break
        
        self.assertTrue(params_changed)


class TestONNXExporter(unittest.TestCase):
    """Test ONNX export"""
    
    def setUp(self):
        self.model = SimpleModel()
        self.exporter = ONNXExporter()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_exporter_initialization(self):
        """Test exporter can be initialized"""
        self.assertIsNotNone(self.exporter)
        
    def test_export(self):
        """Test ONNX export"""
        dummy_input = torch.randn(1, 10)
        output_path = os.path.join(self.temp_dir, 'model.onnx')
        
        success = self.exporter.export(self.model, dummy_input, output_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
    def test_export_with_dynamic_axes(self):
        """Test ONNX export with dynamic batch size"""
        dummy_input = torch.randn(1, 10)
        output_path = os.path.join(self.temp_dir, 'model_dynamic.onnx')
        
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        success = self.exporter.export(
            self.model, dummy_input, output_path,
            dynamic_axes=dynamic_axes
        )
        
        self.assertTrue(success)


class TestModelOptimizer(unittest.TestCase):
    """Test unified model optimizer"""
    
    def setUp(self):
        self.model = SimpleModel()
        self.optimizer = ModelOptimizer()
        
    def test_optimizer_initialization(self):
        """Test optimizer can be initialized"""
        self.assertIsNotNone(self.optimizer.quantizer)
        self.assertIsNotNone(self.optimizer.pruner)
        self.assertIsNotNone(self.optimizer.exporter)
        
    def test_quantization_strategy(self):
        """Test quantization optimization"""
        optimized = self.optimizer.optimize_pipeline(
            self.model,
            strategy='quantization'
        )
        
        self.assertIsNotNone(optimized)
        
        # Test inference
        x = torch.randn(5, 10)
        output = optimized(x)
        self.assertEqual(output.shape, (5, 2))
        
    def test_pruning_strategy(self):
        """Test pruning optimization"""
        optimized = self.optimizer.optimize_pipeline(
            self.model,
            strategy='pruning'
        )
        
        # Check sparsity
        sparsity = self.optimizer.pruner.measure_sparsity(optimized)
        self.assertGreater(sparsity, 0.4)
        
    def test_combined_strategy(self):
        """Test combined quantization + pruning"""
        optimized = self.optimizer.optimize_pipeline(
            self.model,
            strategy='quantization_pruning'
        )
        
        self.assertIsNotNone(optimized)
        
    def test_fp16_strategy(self):
        """Test FP16 optimization"""
        optimized = self.optimizer.optimize_pipeline(
            self.model,
            strategy='fp16'
        )
        
        # Check dtype
        for param in optimized.parameters():
            self.assertEqual(param.dtype, torch.float16)


class TestOptimizationIntegration(unittest.TestCase):
    """Integration tests for optimization"""
    
    def test_full_optimization_pipeline(self):
        """Test complete optimization workflow"""
        # Create model
        model = SimpleModel(input_dim=20, hidden_dim=50, output_dim=3)
        
        # Measure original
        quantizer = ModelQuantizer()
        original_size = quantizer.measure_size(model)
        
        # Optimize
        optimizer = ModelOptimizer()
        optimized = optimizer.optimize_pipeline(model, strategy='pruning')
        
        # Measure optimized
        optimized_params = sum(p.numel() for p in optimized.parameters())
        original_params = sum(p.numel() for p in model.parameters())
        
        # Should have same number of params (but many zeros)
        self.assertEqual(optimized_params, original_params)
        
        # Test inference
        x = torch.randn(10, 20)
        output = optimized(x)
        self.assertEqual(output.shape, (10, 3))
        
    def test_quantization_accuracy(self):
        """Test quantized model maintains reasonable accuracy"""
        model = SimpleModel()
        model.eval()
        
        # Test data
        x = torch.randn(20, 10)
        
        # Original output
        with torch.no_grad():
            original_output = model(x)
        
        # Quantize
        quantizer = ModelQuantizer()
        quantized = quantizer.quantize_dynamic(model)
        
        # Quantized output
        with torch.no_grad():
            quantized_output = quantized(x)
        
        # Outputs should be similar
        diff = torch.abs(original_output - quantized_output).mean().item()
        self.assertLess(diff, 0.5)  # Reasonable tolerance


def run_tests():
    """Run all optimization tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelQuantizer))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPruner))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeDistillation))
    suite.addTests(loader.loadTestsFromTestCase(TestONNXExporter))
    suite.addTests(loader.loadTestsFromTestCase(TestModelOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizationIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
