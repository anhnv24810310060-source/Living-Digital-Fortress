"""
Model Optimization for Production
Implements quantization, pruning, knowledge distillation, and ONNX export
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import copy

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    Quantize models to reduce size and improve inference speed
    Supports INT8 and FP16 quantization
    """
    
    def __init__(self, backend: str = 'fbgemm'):
        """
        Initialize quantizer
        
        Args:
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        """
        self.backend = backend
        torch.backends.quantized.engine = backend
        
        logger.info(f"Initialized quantizer with backend={backend}")
        
    def quantize_dynamic(self, model: nn.Module, 
                        dtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        Dynamic quantization (weights only)
        Fast and easy, good for LSTMs/Transformers
        
        Args:
            model: Model to quantize
            dtype: Quantization dtype
            
        Returns:
            Quantized model
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=dtype
        )
        
        logger.info("Applied dynamic quantization")
        return quantized_model
        
    def quantize_static(self, model: nn.Module, 
                       calibration_data: torch.Tensor,
                       dtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        Static quantization (weights + activations)
        Best accuracy but requires calibration data
        
        Args:
            model: Model to quantize
            calibration_data: Data for calibration
            dtype: Quantization dtype
            
        Returns:
            Quantized model
        """
        model.eval()
        
        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig(self.backend)
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with sample data
        with torch.no_grad():
            for batch in calibration_data:
                model(batch)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        logger.info("Applied static quantization with calibration")
        return quantized_model
        
    def quantize_qat(self, model: nn.Module, train_loader: Any,
                    num_epochs: int = 5, lr: float = 0.001) -> nn.Module:
        """
        Quantization-aware training (QAT)
        Best accuracy, trains with quantization in mind
        
        Args:
            model: Model to quantize
            train_loader: Training data loader
            num_epochs: Training epochs
            lr: Learning rate
            
        Returns:
            Quantized model
        """
        model.train()
        
        # Prepare for QAT
        model.qconfig = torch.quantization.get_default_qat_qconfig(self.backend)
        torch.quantization.prepare_qat(model, inplace=True)
        
        # Train with quantization
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
            logger.info(f"QAT Epoch {epoch + 1}/{num_epochs} completed")
        
        # Convert to quantized model
        model.eval()
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        logger.info("Applied quantization-aware training")
        return quantized_model
        
    def convert_to_fp16(self, model: nn.Module) -> nn.Module:
        """
        Convert model to FP16 (half precision)
        
        Args:
            model: Model to convert
            
        Returns:
            FP16 model
        """
        return model.half()
        
    def measure_size(self, model: nn.Module) -> Dict[str, float]:
        """
        Measure model size
        
        Args:
            model: Model to measure
            
        Returns:
            Size metrics
        """
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Estimate size in MB
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return {
            'num_parameters': num_params,
            'size_mb': size_mb,
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024
        }


class ModelPruner:
    """
    Prune models to remove unnecessary weights
    Reduces model size and increases sparsity
    """
    
    def __init__(self, target_sparsity: float = 0.5):
        """
        Initialize pruner
        
        Args:
            target_sparsity: Target sparsity ratio (0.5 = 50% zeros)
        """
        self.target_sparsity = target_sparsity
        
        logger.info(f"Initialized pruner with target_sparsity={target_sparsity}")
        
    def prune_unstructured(self, model: nn.Module, 
                          method: str = 'l1') -> nn.Module:
        """
        Unstructured pruning (prune individual weights)
        
        Args:
            model: Model to prune
            method: Pruning method ('l1', 'random', 'ln')
            
        Returns:
            Pruned model
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if method == 'l1':
                    prune.l1_unstructured(module, name='weight', 
                                        amount=self.target_sparsity)
                elif method == 'random':
                    prune.random_unstructured(module, name='weight',
                                            amount=self.target_sparsity)
                elif method == 'ln':
                    prune.ln_structured(module, name='weight', 
                                      amount=self.target_sparsity, n=2, dim=0)
                
                # Make pruning permanent
                prune.remove(module, 'weight')
        
        logger.info(f"Applied {method} unstructured pruning")
        return model
        
    def prune_structured(self, model: nn.Module, 
                        dim: int = 0) -> nn.Module:
        """
        Structured pruning (prune entire channels/neurons)
        
        Args:
            model: Model to prune
            dim: Dimension to prune along
            
        Returns:
            Pruned model
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                prune.ln_structured(module, name='weight',
                                  amount=self.target_sparsity, n=2, dim=dim)
                prune.remove(module, 'weight')
        
        logger.info("Applied structured pruning")
        return model
        
    def iterative_pruning(self, model: nn.Module, train_loader: Any,
                         num_iterations: int = 5) -> nn.Module:
        """
        Iterative magnitude pruning
        Prune gradually while fine-tuning
        
        Args:
            model: Model to prune
            train_loader: Training data
            num_iterations: Number of pruning iterations
            
        Returns:
            Pruned model
        """
        sparsity_schedule = np.linspace(0, self.target_sparsity, num_iterations)
        
        for iteration, sparsity in enumerate(sparsity_schedule):
            # Prune
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
            
            # Fine-tune
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx > 10:  # Quick fine-tune
                    break
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            logger.info(f"Iteration {iteration + 1}/{num_iterations}: sparsity={sparsity:.2f}")
        
        # Make pruning permanent
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                prune.remove(module, 'weight')
        
        logger.info("Completed iterative pruning")
        return model
        
    def measure_sparsity(self, model: nn.Module) -> float:
        """
        Measure model sparsity
        
        Args:
            model: Model to measure
            
        Returns:
            Sparsity ratio
        """
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        return sparsity


class KnowledgeDistillation:
    """
    Knowledge distillation (teacher-student training)
    Transfer knowledge from large model to small model
    """
    
    def __init__(self, teacher: nn.Module, student: nn.Module,
                 temperature: float = 3.0, alpha: float = 0.5):
        """
        Initialize knowledge distillation
        
        Args:
            teacher: Large teacher model
            student: Small student model
            temperature: Softmax temperature for distillation
            alpha: Weight for distillation loss (1-alpha for hard labels)
        """
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher.eval()
        
        logger.info(f"Initialized KD with T={temperature}, α={alpha}")
        
    def distillation_loss(self, student_logits: torch.Tensor,
                         teacher_logits: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate distillation loss
        
        Args:
            student_logits: Student predictions
            teacher_logits: Teacher predictions
            labels: True labels
            
        Returns:
            Combined loss
        """
        # Soft targets from teacher
        soft_targets = torch.softmax(teacher_logits / self.temperature, dim=1)
        soft_predictions = torch.log_softmax(student_logits / self.temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distill_loss = nn.KLDivLoss(reduction='batchmean')(
            soft_predictions, soft_targets
        ) * (self.temperature ** 2)
        
        # Hard label loss
        hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
        
        # Combined loss
        loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return loss
        
    def train_student(self, train_loader: Any, num_epochs: int = 10,
                     lr: float = 0.001) -> nn.Module:
        """
        Train student model using teacher
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs
            lr: Learning rate
            
        Returns:
            Trained student model
        """
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        
        self.student.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Get teacher predictions
                with torch.no_grad():
                    teacher_logits = self.teacher(data)
                
                # Get student predictions
                student_logits = self.student(data)
                
                # Calculate loss
                loss = self.distillation_loss(student_logits, teacher_logits, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}")
        
        logger.info("Completed knowledge distillation training")
        return self.student


class ONNXExporter:
    """
    Export PyTorch models to ONNX format
    Enables deployment across different frameworks
    """
    
    def __init__(self):
        """Initialize ONNX exporter"""
        logger.info("Initialized ONNX exporter")
        
    def export(self, model: nn.Module, dummy_input: torch.Tensor,
              output_path: str, opset_version: int = 11,
              dynamic_axes: Optional[Dict] = None) -> bool:
        """
        Export model to ONNX
        
        Args:
            model: Model to export
            dummy_input: Example input for tracing
            output_path: Output file path
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes for variable batch size
            
        Returns:
            Success status
        """
        try:
            model.eval()
            
            # Default dynamic axes for batch dimension
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            
            # Export
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            
            logger.info(f"Exported model to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export ONNX: {e}")
            return False
            
    def verify_export(self, model: nn.Module, onnx_path: str,
                     test_input: torch.Tensor) -> bool:
        """
        Verify ONNX export by comparing outputs
        
        Args:
            model: Original PyTorch model
            onnx_path: Path to ONNX model
            test_input: Test input
            
        Returns:
            Verification status
        """
        try:
            import onnx
            import onnxruntime as ort
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX runtime session
            ort_session = ort.InferenceSession(onnx_path)
            
            # PyTorch inference
            model.eval()
            with torch.no_grad():
                pytorch_output = model(test_input).numpy()
            
            # ONNX inference
            onnx_input = {ort_session.get_inputs()[0].name: test_input.numpy()}
            onnx_output = ort_session.run(None, onnx_input)[0]
            
            # Compare
            max_diff = np.abs(pytorch_output - onnx_output).max()
            
            if max_diff < 1e-5:
                logger.info(f"ONNX export verified (max_diff={max_diff:.2e})")
                return True
            else:
                logger.warning(f"ONNX output differs (max_diff={max_diff:.2e})")
                return False
                
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return False


class ModelOptimizer:
    """
    Unified model optimization pipeline
    Combines quantization, pruning, distillation
    """
    
    def __init__(self):
        """Initialize optimizer"""
        self.quantizer = ModelQuantizer()
        self.pruner = ModelPruner()
        self.exporter = ONNXExporter()
        
        logger.info("Initialized model optimizer")
        
    def optimize_pipeline(self, model: nn.Module, strategy: str,
                         train_loader: Optional[Any] = None,
                         calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Run optimization pipeline
        
        Args:
            model: Model to optimize
            strategy: Optimization strategy
            train_loader: Training data (if needed)
            calibration_data: Calibration data (if needed)
            
        Returns:
            Optimized model
        """
        original_size = self.quantizer.measure_size(model)
        logger.info(f"Original model: {original_size['size_mb']:.2f} MB")
        
        optimized = model
        
        if strategy == 'quantization':
            if calibration_data is not None:
                optimized = self.quantizer.quantize_static(model, calibration_data)
            else:
                optimized = self.quantizer.quantize_dynamic(model)
                
        elif strategy == 'pruning':
            optimized = self.pruner.prune_unstructured(model, method='l1')
            
        elif strategy == 'quantization_pruning':
            # Prune first, then quantize
            optimized = self.pruner.prune_unstructured(model, method='l1')
            if calibration_data is not None:
                optimized = self.quantizer.quantize_static(optimized, calibration_data)
            else:
                optimized = self.quantizer.quantize_dynamic(optimized)
                
        elif strategy == 'fp16':
            optimized = self.quantizer.convert_to_fp16(model)
        
        # Measure improvement
        try:
            optimized_size = self.quantizer.measure_size(optimized)
            reduction = (1 - optimized_size['size_mb'] / original_size['size_mb']) * 100
            logger.info(f"Optimized model: {optimized_size['size_mb']:.2f} MB ({reduction:.1f}% reduction)")
        except:
            logger.info("Optimized model (size measurement not available for quantized)")
        
        return optimized
