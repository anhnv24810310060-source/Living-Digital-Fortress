"""
Adversarial Defense Module
Implements defenses against adversarial attacks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AdversarialAttack:
    """Base class for adversarial attacks"""
    
    def __init__(self, model: nn.Module, epsilon: float = 0.1):
        """
        Initialize adversarial attack
        
        Args:
            model: Target model
            epsilon: Maximum perturbation magnitude
        """
        self.model = model
        self.epsilon = epsilon
        self.model.eval()
        
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial example"""
        raise NotImplementedError


class FGSM(AdversarialAttack):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014)
    Single-step attack using gradient sign
    """
    
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate FGSM adversarial examples
        
        Args:
            x: Input samples [batch_size, features]
            y: True labels [batch_size]
            
        Returns:
            Adversarial examples
        """
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(x_adv)
        
        # Calculate loss
        if output.dim() == 1:
            # Binary classification or regression
            loss = F.mse_loss(output, y.float())
        else:
            # Multi-class classification
            loss = F.cross_entropy(output, y.long())
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial example
        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + self.epsilon * grad_sign
        
        return x_adv.detach()


class PGD(AdversarialAttack):
    """
    Projected Gradient Descent (Madry et al., 2017)
    Multi-step iterative attack with projection
    """
    
    def __init__(self, model: nn.Module, epsilon: float = 0.1, 
                 alpha: float = 0.01, num_iter: int = 40):
        """
        Initialize PGD attack
        
        Args:
            model: Target model
            epsilon: Maximum perturbation magnitude
            alpha: Step size
            num_iter: Number of iterations
        """
        super().__init__(model, epsilon)
        self.alpha = alpha
        self.num_iter = num_iter
        
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate PGD adversarial examples
        
        Args:
            x: Input samples [batch_size, features]
            y: True labels [batch_size]
            
        Returns:
            Adversarial examples
        """
        # Random initialization within epsilon ball
        x_adv = x.clone().detach()
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, x.min(), x.max())
        
        for i in range(self.num_iter):
            x_adv.requires_grad = True
            
            # Forward pass
            output = self.model(x_adv)
            
            # Calculate loss
            if output.dim() == 1:
                loss = F.mse_loss(output, y.float())
            else:
                loss = F.cross_entropy(output, y.long())
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial example
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv.detach() + self.alpha * grad_sign
            
            # Project back to epsilon ball
            perturbation = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + perturbation
            x_adv = torch.clamp(x_adv, x.min(), x.max())
        
        return x_adv


class CarliniWagner(AdversarialAttack):
    """
    Carlini & Wagner (C&W) Attack (Carlini & Wagner, 2017)
    Optimization-based attack with high success rate
    """
    
    def __init__(self, model: nn.Module, epsilon: float = 0.1,
                 c: float = 1.0, kappa: float = 0.0, num_iter: int = 100,
                 learning_rate: float = 0.01):
        """
        Initialize C&W attack
        
        Args:
            model: Target model
            epsilon: Maximum perturbation magnitude
            c: Weight of adversarial loss
            kappa: Confidence parameter
            num_iter: Number of iterations
            learning_rate: Optimization learning rate
        """
        super().__init__(model, epsilon)
        self.c = c
        self.kappa = kappa
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate C&W adversarial examples
        
        Args:
            x: Input samples [batch_size, features]
            y: True labels [batch_size]
            
        Returns:
            Adversarial examples
        """
        # Initialize perturbation variable
        delta = torch.zeros_like(x, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.learning_rate)
        
        for i in range(self.num_iter):
            # Generate adversarial example
            x_adv = x + delta
            
            # Forward pass
            output = self.model(x_adv)
            
            # C&W loss: ||delta||^2 + c * f(x+delta)
            l2_loss = torch.norm(delta, p=2)
            
            if output.dim() == 1:
                # Binary classification
                adv_loss = torch.clamp(output - y.float(), min=-self.kappa).sum()
            else:
                # Multi-class classification
                real = output.gather(1, y.view(-1, 1)).squeeze()
                other = output.clone()
                other.scatter_(1, y.view(-1, 1), -float('inf'))
                other, _ = other.max(1)
                adv_loss = torch.clamp(other - real + self.kappa, min=0).sum()
            
            loss = l2_loss + self.c * adv_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Project to epsilon ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
        
        return (x + delta).detach()


class AdversarialDefense:
    """
    Adversarial defense using adversarial training
    Improves model robustness by training on adversarial examples
    """
    
    def __init__(self, model: nn.Module, attack_type: str = 'pgd',
                 epsilon: float = 0.1, mix_ratio: float = 0.5):
        """
        Initialize adversarial defense
        
        Args:
            model: Model to defend
            attack_type: Type of attack ('fgsm', 'pgd', 'cw')
            epsilon: Perturbation magnitude
            mix_ratio: Ratio of adversarial to clean examples
        """
        self.model = model
        self.epsilon = epsilon
        self.mix_ratio = mix_ratio
        
        # Initialize attack generator
        if attack_type == 'fgsm':
            self.attack = FGSM(model, epsilon)
        elif attack_type == 'pgd':
            self.attack = PGD(model, epsilon)
        elif attack_type == 'cw':
            self.attack = CarliniWagner(model, epsilon)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        logger.info(f"Initialized {attack_type.upper()} adversarial defense")
        
    def generate_adversarial_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mixed batch of clean and adversarial examples
        
        Args:
            x: Clean input samples
            y: Labels
            
        Returns:
            Mixed samples and labels
        """
        batch_size = x.size(0)
        num_adv = int(batch_size * self.mix_ratio)
        
        # Generate adversarial examples for subset
        x_adv = self.attack.generate(x[:num_adv], y[:num_adv])
        
        # Mix clean and adversarial
        x_mixed = torch.cat([x_adv, x[num_adv:]], dim=0)
        y_mixed = torch.cat([y[:num_adv], y[num_adv:]], dim=0)
        
        # Shuffle
        perm = torch.randperm(batch_size)
        x_mixed = x_mixed[perm]
        y_mixed = y_mixed[perm]
        
        return x_mixed, y_mixed
        
    def adversarial_training_step(self, x: torch.Tensor, y: torch.Tensor,
                                   optimizer: torch.optim.Optimizer,
                                   criterion: nn.Module) -> Dict[str, float]:
        """
        Single training step with adversarial examples
        
        Args:
            x: Input samples
            y: Labels
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        # Generate mixed batch
        x_mixed, y_mixed = self.generate_adversarial_batch(x, y)
        
        # Forward pass
        output = self.model(x_mixed)
        
        # Calculate loss
        if output.dim() == 1:
            loss = criterion(output, y_mixed.float())
        else:
            loss = criterion(output, y_mixed.long())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        if output.dim() > 1:
            _, predicted = output.max(1)
            correct = predicted.eq(y_mixed.long()).sum().item()
            accuracy = 100.0 * correct / y_mixed.size(0)
        else:
            accuracy = 0.0
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
        
    def evaluate_robustness(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model robustness against adversarial attacks
        
        Args:
            x: Clean input samples
            y: True labels
            
        Returns:
            Robustness metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Clean accuracy
            output_clean = self.model(x)
            if output_clean.dim() > 1:
                _, pred_clean = output_clean.max(1)
                clean_accuracy = 100.0 * pred_clean.eq(y.long()).sum().item() / y.size(0)
            else:
                clean_accuracy = 0.0
        
        # Generate adversarial examples
        x_adv = self.attack.generate(x, y)
        
        with torch.no_grad():
            # Adversarial accuracy
            output_adv = self.model(x_adv)
            if output_adv.dim() > 1:
                _, pred_adv = output_adv.max(1)
                adv_accuracy = 100.0 * pred_adv.eq(y.long()).sum().item() / y.size(0)
            else:
                adv_accuracy = 0.0
        
        # Calculate perturbation statistics
        perturbation = (x_adv - x).abs()
        avg_perturbation = perturbation.mean().item()
        max_perturbation = perturbation.max().item()
        
        return {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'robustness_drop': clean_accuracy - adv_accuracy,
            'avg_perturbation': avg_perturbation,
            'max_perturbation': max_perturbation
        }


class EnsembleDefense:
    """
    Ensemble-based defense using multiple models
    Improves robustness through diversity
    """
    
    def __init__(self, models: List[nn.Module]):
        """
        Initialize ensemble defense
        
        Args:
            models: List of models in ensemble
        """
        self.models = models
        for model in self.models:
            model.eval()
            
        logger.info(f"Initialized ensemble defense with {len(models)} models")
        
    def predict(self, x: torch.Tensor, method: str = 'voting') -> torch.Tensor:
        """
        Make prediction using ensemble
        
        Args:
            x: Input samples
            method: Aggregation method ('voting', 'average', 'max')
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                predictions.append(output)
        
        predictions = torch.stack(predictions)
        
        if method == 'voting':
            # Majority voting
            if predictions.dim() == 3:  # [num_models, batch_size, num_classes]
                _, votes = predictions.max(dim=2)
                ensemble_pred, _ = torch.mode(votes, dim=0)
            else:
                ensemble_pred = predictions.mean(dim=0)
        elif method == 'average':
            # Average predictions
            ensemble_pred = predictions.mean(dim=0)
        elif method == 'max':
            # Max predictions
            ensemble_pred = predictions.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return ensemble_pred
        
    def evaluate_ensemble_robustness(self, x: torch.Tensor, y: torch.Tensor,
                                     attack: AdversarialAttack) -> Dict[str, float]:
        """
        Evaluate ensemble robustness
        
        Args:
            x: Clean input samples
            y: True labels
            attack: Adversarial attack
            
        Returns:
            Robustness metrics
        """
        # Clean accuracy
        pred_clean = self.predict(x, method='voting')
        if pred_clean.dim() > 1:
            _, pred_clean_class = pred_clean.max(1)
            clean_accuracy = 100.0 * pred_clean_class.eq(y.long()).sum().item() / y.size(0)
        else:
            clean_accuracy = 0.0
        
        # Generate adversarial examples (using first model)
        x_adv = attack.generate(x, y)
        
        # Adversarial accuracy
        pred_adv = self.predict(x_adv, method='voting')
        if pred_adv.dim() > 1:
            _, pred_adv_class = pred_adv.max(1)
            adv_accuracy = 100.0 * pred_adv_class.eq(y.long()).sum().item() / y.size(0)
        else:
            adv_accuracy = 0.0
        
        return {
            'ensemble_clean_accuracy': clean_accuracy,
            'ensemble_adv_accuracy': adv_accuracy,
            'ensemble_robustness': adv_accuracy / clean_accuracy if clean_accuracy > 0 else 0.0
        }


class InputTransformDefense:
    """
    Defense using input transformations
    Removes adversarial perturbations through preprocessing
    """
    
    def __init__(self):
        """Initialize input transform defense"""
        logger.info("Initialized input transform defense")
        
    def bit_depth_reduction(self, x: torch.Tensor, bits: int = 5) -> torch.Tensor:
        """
        Reduce bit depth to remove small perturbations
        
        Args:
            x: Input samples
            bits: Number of bits to keep
            
        Returns:
            Transformed samples
        """
        # Normalize to [0, 1]
        x_min, x_max = x.min(), x.max()
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        
        # Quantize
        levels = 2 ** bits
        x_quant = torch.round(x_norm * levels) / levels
        
        # Denormalize
        x_out = x_quant * (x_max - x_min) + x_min
        
        return x_out
        
    def jpeg_compression(self, x: torch.Tensor, quality: int = 75) -> torch.Tensor:
        """
        Simulate JPEG compression effect
        
        Args:
            x: Input samples
            quality: Compression quality (0-100)
            
        Returns:
            Compressed samples
        """
        # Simple approximation: quantization + smoothing
        x_quant = self.bit_depth_reduction(x, bits=6)
        
        # Apply smoothing (blur effect)
        kernel_size = max(3, int(10 * (100 - quality) / 100))
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        x_smooth = F.avg_pool1d(
            x_quant.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        ).squeeze(1)
        
        return x_smooth
        
    def random_resizing(self, x: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        """
        Random resizing and rescaling
        
        Args:
            x: Input samples
            scale_range: Range of random scales
            
        Returns:
            Transformed samples
        """
        scale = torch.empty(1).uniform_(*scale_range).item()
        
        # Interpolate
        x_scaled = F.interpolate(
            x.unsqueeze(1),
            scale_factor=scale,
            mode='linear',
            align_corners=False
        )
        
        # Resize back
        x_out = F.interpolate(
            x_scaled,
            size=x.size(-1),
            mode='linear',
            align_corners=False
        ).squeeze(1)
        
        return x_out
        
    def apply_transforms(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all transformations
        
        Args:
            x: Input samples
            
        Returns:
            Transformed samples
        """
        x = self.bit_depth_reduction(x)
        x = self.jpeg_compression(x)
        x = self.random_resizing(x)
        
        return x
