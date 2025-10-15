"""
Counterfactual Explanations Generator
Provides "what-if" explanations by finding minimal changes to input
that would change the prediction
"""

import numpy as np
import torch
import torch.optim as optim
from typing import Optional, Dict, Any, List, Callable
import logging

logger = logging.getLogger(__name__)


class CounterfactualExplainer:
    """
    Generate counterfactual explanations
    
    Finds minimal perturbations to input that flip the prediction
    to a desired target class. Useful for actionable recommendations.
    """
    
    def __init__(
        self,
        model: Any,
        feature_ranges: Optional[Dict[int, tuple]] = None,
        categorical_features: Optional[List[int]] = None
    ):
        """
        Args:
            model: Model to explain
            feature_ranges: Dict mapping feature indices to (min, max) ranges
            categorical_features: Indices of categorical features
        """
        self.model = model
        self.feature_ranges = feature_ranges or {}
        self.categorical_features = categorical_features or []
        
        # Get PyTorch model if wrapped
        if hasattr(model, 'model'):
            self.pytorch_model = model.model
        else:
            self.pytorch_model = model
    
    def generate(
        self,
        instance: np.ndarray,
        target_class: int,
        max_iterations: int = 1000,
        learning_rate: float = 0.1,
        distance_weight: float = 1.0,
        sparsity_weight: float = 0.1,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanation
        
        Args:
            instance: Original instance
            target_class: Desired class
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for gradient descent
            distance_weight: Weight for L2 distance loss
            sparsity_weight: Weight for L1 sparsity loss
            verbose: Print progress
            
        Returns:
            Dictionary with counterfactual and changes
        """
        # Convert to tensor
        instance_tensor = torch.FloatTensor(instance).unsqueeze(0)
        instance_tensor.requires_grad = False
        
        # Initialize counterfactual as copy of instance
        counterfactual = instance_tensor.clone().detach()
        counterfactual.requires_grad = True
        
        # Optimizer
        optimizer = optim.Adam([counterfactual], lr=learning_rate)
        
        best_counterfactual = None
        best_distance = float('inf')
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Get prediction
            self.pytorch_model.eval()
            with torch.set_grad_enabled(True):
                output = self.pytorch_model(counterfactual)
                
                # Classification loss (want to maximize target class probability)
                if len(output.shape) > 1 and output.shape[1] > 1:
                    # Multi-class
                    class_loss = -output[0, target_class]
                else:
                    # Binary
                    class_loss = -output[0]
                
                # Distance loss (L2 norm)
                distance = torch.norm(counterfactual - instance_tensor, p=2)
                
                # Sparsity loss (L1 norm) - encourage few changes
                sparsity = torch.norm(counterfactual - instance_tensor, p=1)
                
                # Total loss
                loss = class_loss + distance_weight * distance + sparsity_weight * sparsity
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            # Apply constraints (feature ranges)
            with torch.no_grad():
                for feat_idx, (min_val, max_val) in self.feature_ranges.items():
                    if len(counterfactual.shape) == 2:
                        # Tabular data
                        counterfactual[0, feat_idx] = torch.clamp(
                            counterfactual[0, feat_idx],
                            min_val,
                            max_val
                        )
                    elif len(counterfactual.shape) == 3:
                        # Sequential data
                        counterfactual[0, :, feat_idx] = torch.clamp(
                            counterfactual[0, :, feat_idx],
                            min_val,
                            max_val
                        )
            
            # Check if target class achieved
            with torch.no_grad():
                pred = self.pytorch_model(counterfactual)
                if len(pred.shape) > 1 and pred.shape[1] > 1:
                    predicted_class = torch.argmax(pred, dim=1).item()
                else:
                    predicted_class = (pred > 0.5).int().item()
                
                current_distance = distance.item()
                
                if predicted_class == target_class:
                    if current_distance < best_distance:
                        best_distance = current_distance
                        best_counterfactual = counterfactual.clone().detach()
                    
                    if verbose and iteration % 100 == 0:
                        logger.info(
                            f"Iter {iteration}: Target reached, distance={current_distance:.4f}"
                        )
                elif verbose and iteration % 100 == 0:
                    logger.info(
                        f"Iter {iteration}: Class={predicted_class}, distance={current_distance:.4f}"
                    )
        
        if best_counterfactual is None:
            logger.warning("Failed to find counterfactual")
            best_counterfactual = counterfactual.detach()
        
        # Calculate changes
        changes = self._calculate_changes(
            instance_tensor.numpy(),
            best_counterfactual.numpy()
        )
        
        return {
            'original': instance,
            'counterfactual': best_counterfactual.numpy()[0],
            'changes': changes,
            'distance': best_distance,
            'target_class': target_class,
            'success': best_distance < float('inf')
        }
    
    def _calculate_changes(
        self,
        original: np.ndarray,
        counterfactual: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Calculate feature-level changes
        
        Args:
            original: Original instance
            counterfactual: Counterfactual instance
            
        Returns:
            List of changes per feature
        """
        changes = []
        
        diff = counterfactual - original
        
        if len(diff.shape) == 2:
            # Tabular data
            for i in range(diff.shape[1]):
                change = float(diff[0, i])
                if abs(change) > 1e-6:  # Ignore negligible changes
                    changes.append({
                        'feature_idx': i,
                        'original_value': float(original[0, i]),
                        'new_value': float(counterfactual[0, i]),
                        'change': change,
                        'change_percent': (change / (original[0, i] + 1e-9)) * 100
                    })
        elif len(diff.shape) == 3:
            # Sequential data
            for i in range(diff.shape[2]):
                total_change = np.sum(np.abs(diff[0, :, i]))
                if total_change > 1e-6:
                    changes.append({
                        'feature_idx': i,
                        'original_values': original[0, :, i].tolist(),
                        'new_values': counterfactual[0, :, i].tolist(),
                        'total_change': float(total_change)
                    })
        
        # Sort by absolute change
        if changes and 'change' in changes[0]:
            changes.sort(key=lambda x: abs(x['change']), reverse=True)
        elif changes and 'total_change' in changes[0]:
            changes.sort(key=lambda x: x['total_change'], reverse=True)
        
        return changes
    
    def generate_diverse_counterfactuals(
        self,
        instance: np.ndarray,
        target_class: int,
        num_counterfactuals: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple diverse counterfactuals
        
        Args:
            instance: Original instance
            target_class: Target class
            num_counterfactuals: Number to generate
            **kwargs: Additional arguments for generate()
            
        Returns:
            List of counterfactual explanations
        """
        counterfactuals = []
        
        for i in range(num_counterfactuals):
            # Add randomness for diversity
            perturbed_instance = instance + np.random.randn(*instance.shape) * 0.01
            
            cf = self.generate(
                perturbed_instance,
                target_class,
                **kwargs
            )
            
            if cf['success']:
                counterfactuals.append(cf)
        
        return counterfactuals


class ActionableInsights:
    """
    Generate actionable recommendations from counterfactual explanations
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        feature_descriptions: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            feature_names: Names of features
            feature_descriptions: Human-readable descriptions of features
        """
        self.feature_names = feature_names or []
        self.feature_descriptions = feature_descriptions or {}
    
    def generate_recommendations(
        self,
        counterfactual_explanation: Dict[str, Any],
        top_k: int = 5
    ) -> List[str]:
        """
        Generate actionable recommendations from counterfactual
        
        Args:
            counterfactual_explanation: Output from CounterfactualExplainer
            top_k: Number of top recommendations
            
        Returns:
            List of recommendation strings
        """
        changes = counterfactual_explanation['changes'][:top_k]
        recommendations = []
        
        for change in changes:
            feature_idx = change['feature_idx']
            
            if feature_idx < len(self.feature_names):
                feature_name = self.feature_names[feature_idx]
            else:
                feature_name = f"Feature {feature_idx}"
            
            if 'change' in change:
                # Tabular data
                original = change['original_value']
                new_value = change['new_value']
                change_val = change['change']
                
                if change_val > 0:
                    direction = "increase"
                else:
                    direction = "decrease"
                
                recommendation = (
                    f"{direction.capitalize()} {feature_name} from "
                    f"{original:.2f} to {new_value:.2f} "
                    f"({abs(change['change_percent']):.1f}% change)"
                )
            else:
                # Sequential data
                recommendation = f"Modify {feature_name} pattern"
            
            # Add description if available
            if feature_name in self.feature_descriptions:
                recommendation += f" - {self.feature_descriptions[feature_name]}"
            
            recommendations.append(recommendation)
        
        return recommendations
