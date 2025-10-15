"""
SHAP (SHapley Additive exPlanations) integration for model explainability
Provides global and local feature importance explanations
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, List, Callable
import logging

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


class SHAPExplainer:
    """
    SHAP-based explainer for PyTorch models
    
    Provides feature importance explanations using Shapley values:
    - Global feature importance (averaged across all samples)
    - Local explanations (per prediction)
    - Interaction effects between features
    """
    
    def __init__(
        self,
        model: Any,
        background_data: Optional[np.ndarray] = None,
        model_type: str = 'deep',
        feature_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: PyTorch model or any model with predict method
            background_data: Background dataset for SHAP (training data sample)
            model_type: Type of explainer ('deep', 'gradient', 'kernel')
            feature_names: Names of input features
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        self.model = model
        self.background_data = background_data
        self.model_type = model_type
        self.feature_names = feature_names
        self.explainer = None
        
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type"""
        if self.model_type == 'deep':
            # For deep learning models
            if hasattr(self.model, 'model'):
                # Our detector classes wrap the actual PyTorch model
                pytorch_model = self.model.model
            else:
                pytorch_model = self.model
            
            if self.background_data is not None:
                background_tensor = torch.FloatTensor(self.background_data[:100])
                self.explainer = shap.DeepExplainer(pytorch_model, background_tensor)
                logger.info("Initialized DeepExplainer")
            else:
                logger.warning("No background data provided for DeepExplainer")
                
        elif self.model_type == 'gradient':
            # Gradient-based explainer
            if hasattr(self.model, 'model'):
                pytorch_model = self.model.model
            else:
                pytorch_model = self.model
            
            if self.background_data is not None:
                background_tensor = torch.FloatTensor(self.background_data[:100])
                self.explainer = shap.GradientExplainer(pytorch_model, background_tensor)
                logger.info("Initialized GradientExplainer")
                
        elif self.model_type == 'kernel':
            # Model-agnostic kernel SHAP
            def predict_fn(x):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(x)
                elif hasattr(self.model, 'predict'):
                    return self.model.predict(x)
                else:
                    raise ValueError("Model must have predict or predict_proba method")
            
            if self.background_data is not None:
                self.explainer = shap.KernelExplainer(predict_fn, self.background_data[:100])
                logger.info("Initialized KernelExplainer")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def explain(
        self,
        X: np.ndarray,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for input samples
        
        Args:
            X: Input samples (n_samples, n_features) or (n_samples, seq_len, n_features)
            max_samples: Maximum number of samples to explain
            
        Returns:
            Dictionary containing SHAP values and visualizations
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Provide background_data.")
        
        # Limit number of samples for computational efficiency
        if max_samples is not None and len(X) > max_samples:
            X = X[:max_samples]
        
        try:
            # Calculate SHAP values
            if self.model_type in ['deep', 'gradient']:
                X_tensor = torch.FloatTensor(X)
                shap_values = self.explainer.shap_values(X_tensor)
            else:
                shap_values = self.explainer.shap_values(X)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class output
                shap_values_array = np.array(shap_values)
            else:
                shap_values_array = shap_values
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(shap_values_array)
            
            return {
                'shap_values': shap_values_array,
                'feature_importance': feature_importance,
                'base_values': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else None,
                'data': X,
                'feature_names': self.feature_names
            }
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            raise
    
    def _calculate_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate global feature importance from SHAP values
        
        Args:
            shap_values: SHAP values array
            
        Returns:
            Dictionary mapping feature names/indices to importance scores
        """
        # Average absolute SHAP values across samples
        if len(shap_values.shape) == 3:
            # Multi-class: (n_classes, n_samples, n_features)
            importance = np.abs(shap_values).mean(axis=(0, 1))
        elif len(shap_values.shape) == 2:
            # Binary or regression: (n_samples, n_features)
            importance = np.abs(shap_values).mean(axis=0)
        elif len(shap_values.shape) == 4:
            # Sequential data: (n_classes, n_samples, seq_len, n_features)
            importance = np.abs(shap_values).mean(axis=(0, 1, 2))
        else:
            importance = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dict
        if self.feature_names is not None:
            importance_dict = {
                name: float(imp) 
                for name, imp in zip(self.feature_names, importance)
            }
        else:
            importance_dict = {
                f'feature_{i}': float(imp)
                for i, imp in enumerate(importance)
            }
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance_dict
    
    def explain_instance(
        self,
        instance: np.ndarray,
        class_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single instance
        
        Args:
            instance: Single sample to explain
            class_idx: Class index for multi-class models
            
        Returns:
            Dictionary with SHAP values and feature contributions
        """
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        elif len(instance.shape) == 2 and instance.shape[0] > 1:
            logger.warning("Multiple instances provided, using first one")
            instance = instance[:1]
        
        explanation = self.explain(instance, max_samples=1)
        shap_values = explanation['shap_values']
        
        # Get SHAP values for specific class if multi-class
        if len(shap_values.shape) == 3 and class_idx is not None:
            shap_values = shap_values[class_idx]
        
        # Create feature contributions
        contributions = {}
        if self.feature_names is not None:
            for i, name in enumerate(self.feature_names):
                contributions[name] = float(shap_values[0, i])
        else:
            for i in range(shap_values.shape[-1]):
                contributions[f'feature_{i}'] = float(shap_values[0, i])
        
        # Sort by absolute contribution
        contributions = dict(sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return {
            'shap_values': shap_values,
            'contributions': contributions,
            'instance': instance,
            'class_idx': class_idx
        }
    
    def get_top_features(
        self,
        X: np.ndarray,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Get top k most important features
        
        Args:
            X: Input samples
            top_k: Number of top features to return
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        explanation = self.explain(X)
        importance = explanation['feature_importance']
        
        top_features = list(importance.items())[:top_k]
        return top_features
    
    def summary_plot_data(
        self,
        X: np.ndarray,
        max_display: int = 20
    ) -> Dict[str, Any]:
        """
        Prepare data for SHAP summary plot
        
        Args:
            X: Input samples
            max_display: Maximum features to display
            
        Returns:
            Dictionary with data for plotting
        """
        explanation = self.explain(X)
        
        return {
            'shap_values': explanation['shap_values'],
            'features': X,
            'feature_names': self.feature_names,
            'max_display': max_display
        }


class ModelAgnosticExplainer:
    """
    Model-agnostic explainer using sampling-based methods
    Works with any model that has a predict method
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: Any model with predict or predict_proba method
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
    
    def permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Calculate permutation-based feature importance
        
        Args:
            X: Input features
            y: True labels
            n_repeats: Number of permutation repeats
            random_state: Random seed
            
        Returns:
            Dictionary of feature importances
        """
        from sklearn.metrics import accuracy_score
        
        np.random.seed(random_state)
        
        # Baseline score
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(X)
            baseline_score = accuracy_score(y, y_pred)
        else:
            raise ValueError("Model must have predict method")
        
        n_features = X.shape[-1]
        importances = np.zeros(n_features)
        
        # Permute each feature
        for feature_idx in range(n_features):
            feature_scores = []
            
            for _ in range(n_repeats):
                X_permuted = X.copy()
                
                # Permute the feature
                if len(X.shape) == 2:
                    np.random.shuffle(X_permuted[:, feature_idx])
                elif len(X.shape) == 3:
                    np.random.shuffle(X_permuted[:, :, feature_idx])
                
                # Calculate score with permuted feature
                y_pred_permuted = self.model.predict(X_permuted)
                permuted_score = accuracy_score(y, y_pred_permuted)
                feature_scores.append(baseline_score - permuted_score)
            
            importances[feature_idx] = np.mean(feature_scores)
        
        # Create importance dict
        if self.feature_names is not None:
            importance_dict = {
                name: float(imp)
                for name, imp in zip(self.feature_names, importances)
            }
        else:
            importance_dict = {
                f'feature_{i}': float(imp)
                for i, imp in enumerate(importances)
            }
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance_dict
