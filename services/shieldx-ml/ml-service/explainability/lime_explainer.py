"""
LIME (Local Interpretable Model-agnostic Explanations) integration
Provides local explanations by approximating model with interpretable models
"""

import numpy as np
from typing import Optional, Dict, Any, List, Callable
import logging

logger = logging.getLogger(__name__)

try:
    from lime import lime_tabular
    from lime import lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")


class LIMEExplainer:
    """
    LIME explainer for tabular data
    
    Explains individual predictions by learning a local linear model
    around the prediction point.
    """
    
    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        mode: str = 'classification'
    ):
        """
        Args:
            training_data: Training data for understanding feature distributions
            feature_names: Names of features
            class_names: Names of classes
            categorical_features: Indices of categorical features
            mode: 'classification' or 'regression'
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Install with: pip install lime")
        
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        
        # Flatten if sequential data
        if len(training_data.shape) == 3:
            # (n_samples, seq_len, n_features) -> (n_samples, seq_len * n_features)
            self.training_data_flat = training_data.reshape(training_data.shape[0], -1)
            logger.info("Flattened sequential data for LIME")
        else:
            self.training_data_flat = training_data
        
        # Initialize LIME explainer
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.training_data_flat,
            feature_names=feature_names,
            class_names=class_names,
            categorical_features=categorical_features,
            mode=mode,
            verbose=False
        )
        
        logger.info(f"Initialized LIME explainer with {len(self.training_data_flat)} samples")
    
    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: Callable,
        num_features: int = 10,
        num_samples: int = 5000,
        top_labels: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Explain a single prediction
        
        Args:
            instance: Sample to explain
            predict_fn: Prediction function (must return probabilities)
            num_features: Number of features to include in explanation
            num_samples: Number of samples for local approximation
            top_labels: Number of top labels to explain
            
        Returns:
            Dictionary with explanation details
        """
        # Flatten if needed
        if len(instance.shape) > 1:
            instance_flat = instance.flatten()
        else:
            instance_flat = instance
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            data_row=instance_flat,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=top_labels
        )
        
        # Extract feature weights for each class
        explanations_by_class = {}
        
        if self.mode == 'classification':
            available_labels = explanation.available_labels()
            
            for label in available_labels:
                feature_weights = explanation.as_list(label=label)
                explanations_by_class[label] = {
                    'feature_weights': feature_weights,
                    'score': explanation.score if hasattr(explanation, 'score') else None
                }
        else:
            feature_weights = explanation.as_list()
            explanations_by_class[0] = {
                'feature_weights': feature_weights,
                'score': explanation.score if hasattr(explanation, 'score') else None
            }
        
        # Get prediction probabilities
        pred_proba = predict_fn(instance_flat.reshape(1, -1))
        
        return {
            'instance': instance,
            'explanations': explanations_by_class,
            'prediction_probabilities': pred_proba,
            'local_prediction': explanation.local_pred if hasattr(explanation, 'local_pred') else None,
            'intercept': explanation.intercept if hasattr(explanation, 'intercept') else None
        }
    
    def explain_batch(
        self,
        X: np.ndarray,
        predict_fn: Callable,
        num_features: int = 10,
        max_samples: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Explain multiple instances
        
        Args:
            X: Samples to explain
            predict_fn: Prediction function
            num_features: Number of features per explanation
            max_samples: Maximum samples to explain
            
        Returns:
            List of explanation dictionaries
        """
        if len(X) > max_samples:
            logger.warning(f"Explaining only first {max_samples} samples")
            X = X[:max_samples]
        
        explanations = []
        
        for i, instance in enumerate(X):
            try:
                exp = self.explain_instance(
                    instance,
                    predict_fn,
                    num_features=num_features
                )
                explanations.append(exp)
            except Exception as e:
                logger.error(f"Error explaining instance {i}: {e}")
                explanations.append(None)
        
        return explanations
    
    def get_feature_importance_summary(
        self,
        explanations: List[Dict[str, Any]],
        class_idx: int = 0
    ) -> Dict[str, float]:
        """
        Aggregate feature importance across multiple explanations
        
        Args:
            explanations: List of explanations from explain_batch
            class_idx: Class index to aggregate
            
        Returns:
            Dictionary of aggregated feature importances
        """
        feature_importance = {}
        
        for exp in explanations:
            if exp is None:
                continue
            
            class_explanation = exp['explanations'].get(class_idx)
            if class_explanation is None:
                continue
            
            for feature_name, weight in class_explanation['feature_weights']:
                if feature_name not in feature_importance:
                    feature_importance[feature_name] = []
                feature_importance[feature_name].append(abs(weight))
        
        # Average importance
        avg_importance = {
            name: np.mean(weights)
            for name, weights in feature_importance.items()
        }
        
        # Sort by importance
        avg_importance = dict(sorted(
            avg_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return avg_importance


class LIMETextExplainer:
    """
    LIME explainer for text data (e.g., log analysis)
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            class_names: Names of classes
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed")
        
        self.class_names = class_names
        self.explainer = lime_text.LimeTextExplainer(class_names=class_names)
        
        logger.info("Initialized LIME text explainer")
    
    def explain_instance(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict[str, Any]:
        """
        Explain a text prediction
        
        Args:
            text: Text to explain
            predict_fn: Prediction function
            num_features: Number of words to include
            num_samples: Number of samples for approximation
            
        Returns:
            Explanation dictionary
        """
        explanation = self.explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Extract explanations for each class
        explanations_by_class = {}
        
        for label in explanation.available_labels():
            word_weights = explanation.as_list(label=label)
            explanations_by_class[label] = {
                'word_weights': word_weights,
                'score': explanation.score if hasattr(explanation, 'score') else None
            }
        
        return {
            'text': text,
            'explanations': explanations_by_class,
            'prediction_probabilities': predict_fn([text])[0]
        }
