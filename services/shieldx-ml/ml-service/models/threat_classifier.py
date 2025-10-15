"""
Unified Threat Classification System
Combines multiple deep learning models for comprehensive threat detection
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging

from .autoencoder import AnomalyDetectionAE
from .lstm_autoencoder import SequentialAnomalyDetector
from .cnn1d import PacketThreatDetector
from .transformer import TransformerThreatDetector

logger = logging.getLogger(__name__)


class ThreatClassifier:
    """
    Multi-model threat classification system
    
    Combines different models for comprehensive threat detection:
    - Autoencoder for anomaly detection
    - LSTM Autoencoder for sequential anomalies
    - CNN-1D for packet-level threats
    - Transformer for complex pattern detection
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 6,
        enable_autoencoder: bool = True,
        enable_lstm: bool = True,
        enable_cnn: bool = True,
        enable_transformer: bool = True,
        device: Optional[str] = None
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Number of threat classes
            enable_autoencoder: Use autoencoder
            enable_lstm: Use LSTM autoencoder
            enable_cnn: Use CNN-1D
            enable_transformer: Use Transformer
            device: Device (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Initialize models
        self.models: Dict[str, any] = {}
        
        if enable_autoencoder:
            self.models['autoencoder'] = AnomalyDetectionAE(
                input_dim=input_dim,
                latent_dim=32,
                device=self.device
            )
            logger.info("Initialized Autoencoder")
        
        if enable_lstm:
            self.models['lstm'] = SequentialAnomalyDetector(
                input_dim=input_dim,
                hidden_dim=64,
                latent_dim=32,
                device=self.device
            )
            logger.info("Initialized LSTM Autoencoder")
        
        if enable_cnn:
            self.models['cnn'] = PacketThreatDetector(
                input_dim=input_dim,
                num_classes=num_classes,
                device=self.device
            )
            logger.info("Initialized CNN-1D")
        
        if enable_transformer:
            self.models['transformer'] = TransformerThreatDetector(
                input_dim=input_dim,
                num_classes=num_classes,
                device=self.device
            )
            logger.info("Initialized Transformer")
        
        # Ensemble weights (learned or configured)
        self.ensemble_weights = {
            'autoencoder': 0.2,
            'lstm': 0.2,
            'cnn': 0.3,
            'transformer': 0.3
        }
        
        # Class names
        self.class_names = [
            'Normal', 'SQL_Injection', 'XSS',
            'DDoS', 'Port_Scan', 'Malware'
        ][:num_classes]
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> 'ThreatClassifier':
        """
        Train models
        
        Args:
            X: Training data
            y: Labels (required for supervised models)
            model_name: Specific model to train (None = train all)
            **kwargs: Additional training parameters
            
        Returns:
            self
        """
        models_to_train = [model_name] if model_name else self.models.keys()
        
        for name in models_to_train:
            if name not in self.models:
                logger.warning(f"Model {name} not found, skipping")
                continue
            
            logger.info(f"Training {name}...")
            
            try:
                if name in ['autoencoder', 'lstm']:
                    # Unsupervised models - train on normal data only
                    if y is not None:
                        normal_mask = y == 0
                        X_normal = X[normal_mask]
                        self.models[name].fit(X_normal, **kwargs)
                    else:
                        self.models[name].fit(X, **kwargs)
                else:
                    # Supervised models - need labels
                    if y is None:
                        raise ValueError(f"Model {name} requires labels (y)")
                    self.models[name].fit(X, y, **kwargs)
                
                logger.info(f"Successfully trained {name}")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        strategy: str = 'weighted_voting',
        return_individual: bool = False
    ) -> np.ndarray:
        """
        Predict threat classes
        
        Args:
            X: Input data
            strategy: Ensemble strategy ('weighted_voting', 'majority', 'max_confidence')
            return_individual: Return predictions from each model
            
        Returns:
            Predicted classes (and individual predictions if requested)
        """
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                if name in ['autoencoder', 'lstm']:
                    # Anomaly detectors - binary output (0=normal, 1=anomaly)
                    pred = model.predict(X)
                    predictions[name] = pred
                    probabilities[name] = model.predict_proba(X)
                else:
                    # Classifiers - multi-class output
                    pred = model.predict(X)
                    predictions[name] = pred
                    probabilities[name] = model.predict_proba(X)
            except Exception as e:
                logger.warning(f"Error getting predictions from {name}: {e}")
        
        # Ensemble predictions
        if strategy == 'weighted_voting':
            ensemble_pred = self._weighted_voting(predictions, probabilities)
        elif strategy == 'majority':
            ensemble_pred = self._majority_voting(predictions)
        elif strategy == 'max_confidence':
            ensemble_pred = self._max_confidence(predictions, probabilities)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        if return_individual:
            return ensemble_pred, predictions
        return ensemble_pred
    
    def predict_proba(
        self,
        X: np.ndarray,
        strategy: str = 'weighted_voting'
    ) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input data
            strategy: Ensemble strategy
            
        Returns:
            Class probabilities (n_samples, num_classes)
        """
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                if name in ['autoencoder', 'lstm']:
                    # Convert anomaly scores to binary probabilities
                    scores = model.predict_proba(X)
                    probs = np.zeros((len(scores), self.num_classes))
                    probs[:, 0] = 1 - scores  # Normal probability
                    probs[:, 1] = scores  # Anomaly probability (generic)
                    probabilities[name] = probs
                else:
                    probabilities[name] = model.predict_proba(X)
            except Exception as e:
                logger.warning(f"Error getting probabilities from {name}: {e}")
        
        if not probabilities:
            raise ValueError("No valid model predictions available")
        
        # Weighted average of probabilities
        ensemble_probs = np.zeros((X.shape[0], self.num_classes))
        total_weight = 0
        
        for name, probs in probabilities.items():
            weight = self.ensemble_weights.get(name, 1.0)
            ensemble_probs += weight * probs
            total_weight += weight
        
        ensemble_probs /= total_weight
        
        return ensemble_probs
    
    def _weighted_voting(
        self,
        predictions: Dict[str, np.ndarray],
        probabilities: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Weighted voting ensemble"""
        ensemble_probs = self.predict_proba(
            np.zeros((len(next(iter(predictions.values()))), self.input_dim))
        )
        return np.argmax(ensemble_probs, axis=1)
    
    def _majority_voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Majority voting ensemble"""
        n_samples = len(next(iter(predictions.values())))
        votes = np.zeros((n_samples, self.num_classes))
        
        for name, pred in predictions.items():
            for i, p in enumerate(pred):
                if p < self.num_classes:
                    votes[i, p] += 1
        
        return np.argmax(votes, axis=1)
    
    def _max_confidence(
        self,
        predictions: Dict[str, np.ndarray],
        probabilities: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Max confidence ensemble"""
        n_samples = len(next(iter(predictions.values())))
        ensemble_pred = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            max_conf = 0
            best_pred = 0
            
            for name, probs in probabilities.items():
                conf = np.max(probs[i])
                pred = np.argmax(probs[i])
                
                if conf > max_conf:
                    max_conf = conf
                    best_pred = pred
            
            ensemble_pred[i] = best_pred
        
        return ensemble_pred
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strategy: str = 'weighted_voting'
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance
        
        Args:
            X: Test data
            y: True labels
            strategy: Ensemble strategy
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        # Get predictions
        y_pred = self.predict(X, strategy=strategy)
        y_proba = self.predict_proba(X, strategy=strategy)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        cm = confusion_matrix(y, y_pred)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        logger.info(f"Ensemble Accuracy: {accuracy:.4f}")
        logger.info(f"Ensemble F1-Score: {f1:.4f}")
        
        return metrics
    
    def save(self, base_path: str) -> None:
        """Save all models"""
        for name, model in self.models.items():
            model_path = f"{base_path}_{name}.pt"
            model.save(model_path)
            logger.info(f"Saved {name} to {model_path}")
    
    def load(self, base_path: str) -> None:
        """Load all models"""
        for name, model in self.models.items():
            model_path = f"{base_path}_{name}.pt"
            try:
                model.load(model_path)
                logger.info(f"Loaded {name} from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
    
    def get_model_names(self) -> List[str]:
        """Get names of available models"""
        return list(self.models.keys())
    
    def get_model(self, name: str):
        """Get specific model by name"""
        return self.models.get(name)
