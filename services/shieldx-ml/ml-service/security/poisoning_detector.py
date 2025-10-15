"""
Model Poisoning Detection
Detects data poisoning and backdoor attacks in training data
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates training data for poisoning attacks
    Uses statistical methods to detect outliers
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize data validator
        
        Args:
            contamination: Expected proportion of outliers
        """
        self.contamination = contamination
        self.detector = EllipticEnvelope(contamination=contamination)
        self.fitted = False
        
        logger.info(f"Initialized data validator with contamination={contamination}")
        
    def fit(self, X: np.ndarray):
        """
        Fit validator on clean data
        
        Args:
            X: Training data [n_samples, n_features]
        """
        self.detector.fit(X)
        self.fitted = True
        logger.info(f"Fitted data validator on {X.shape[0]} samples")
        
    def detect_outliers(self, X: np.ndarray) -> np.ndarray:
        """
        Detect outliers in data
        
        Args:
            X: Data to check [n_samples, n_features]
            
        Returns:
            Boolean mask (True = outlier)
        """
        if not self.fitted:
            raise ValueError("Validator must be fitted before detection")
        
        predictions = self.detector.predict(X)
        outliers = predictions == -1
        
        logger.info(f"Detected {outliers.sum()} outliers in {X.shape[0]} samples")
        
        return outliers
        
    def validate_batch(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Validate a batch of data
        
        Args:
            X: Data batch [n_samples, n_features]
            
        Returns:
            Validation report
        """
        outliers = self.detect_outliers(X)
        
        # Calculate statistics
        decision_scores = self.detector.decision_function(X)
        
        return {
            'total_samples': X.shape[0],
            'outliers_detected': int(outliers.sum()),
            'outlier_ratio': float(outliers.sum() / X.shape[0]),
            'min_score': float(decision_scores.min()),
            'max_score': float(decision_scores.max()),
            'mean_score': float(decision_scores.mean()),
            'outlier_indices': np.where(outliers)[0].tolist()
        }


class ClusteringDetector:
    """
    Detects poisoning using clustering analysis
    Identifies suspicious clusters that may contain poisoned data
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Initialize clustering detector
        
        Args:
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples parameter
        """
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        
        logger.info(f"Initialized clustering detector with eps={eps}, min_samples={min_samples}")
        
    def detect_poisoned_clusters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Detect suspicious clusters
        
        Args:
            X: Feature data [n_samples, n_features]
            y: Labels [n_samples]
            
        Returns:
            Detection report
        """
        # Apply DBSCAN
        clusters = self.dbscan.fit_predict(X)
        
        # Analyze clusters
        unique_clusters = np.unique(clusters)
        cluster_info = []
        
        suspicious_samples = []
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Noise points
                continue
                
            cluster_mask = clusters == cluster_id
            cluster_samples = X[cluster_mask]
            cluster_labels = y[cluster_mask]
            
            # Check label consistency
            unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
            label_entropy = -np.sum((label_counts / len(cluster_labels)) * 
                                   np.log2(label_counts / len(cluster_labels) + 1e-10))
            
            # Suspicious if high label entropy (mixed labels)
            is_suspicious = label_entropy > 0.5
            
            cluster_info.append({
                'cluster_id': int(cluster_id),
                'size': int(cluster_mask.sum()),
                'label_entropy': float(label_entropy),
                'unique_labels': unique_labels.tolist(),
                'is_suspicious': bool(is_suspicious)
            })
            
            if is_suspicious:
                suspicious_indices = np.where(cluster_mask)[0]
                suspicious_samples.extend(suspicious_indices.tolist())
        
        # Noise points are also suspicious
        noise_mask = clusters == -1
        noise_indices = np.where(noise_mask)[0]
        suspicious_samples.extend(noise_indices.tolist())
        
        return {
            'num_clusters': len(unique_clusters) - (1 if -1 in unique_clusters else 0),
            'num_noise_points': int(noise_mask.sum()),
            'clusters': cluster_info,
            'suspicious_samples': suspicious_samples,
            'total_suspicious': len(suspicious_samples)
        }


class GradientAnalyzer:
    """
    Analyzes gradients during training to detect poisoning
    Poisoned samples often have unusual gradient patterns
    """
    
    def __init__(self, model: nn.Module, threshold: float = 3.0):
        """
        Initialize gradient analyzer
        
        Args:
            model: Model being trained
            threshold: Standard deviations for outlier detection
        """
        self.model = model
        self.threshold = threshold
        self.gradient_history = []
        
        logger.info(f"Initialized gradient analyzer with threshold={threshold}")
        
    def compute_sample_gradients(self, X: torch.Tensor, y: torch.Tensor,
                                 criterion: nn.Module) -> torch.Tensor:
        """
        Compute gradient for each sample
        
        Args:
            X: Input samples [batch_size, features]
            y: Labels [batch_size]
            criterion: Loss function
            
        Returns:
            Gradient norms [batch_size]
        """
        self.model.train()
        gradient_norms = []
        
        for i in range(X.size(0)):
            x_sample = X[i:i+1]
            y_sample = y[i:i+1]
            
            # Forward pass
            output = self.model(x_sample)
            
            # Calculate loss
            if output.dim() == 1:
                loss = criterion(output, y_sample.float())
            else:
                loss = criterion(output, y_sample.long())
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Compute gradient norm
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = np.sqrt(grad_norm)
            
            gradient_norms.append(grad_norm)
        
        return torch.tensor(gradient_norms)
        
    def detect_anomalous_gradients(self, X: torch.Tensor, y: torch.Tensor,
                                   criterion: nn.Module) -> Dict[str, Any]:
        """
        Detect samples with anomalous gradients
        
        Args:
            X: Input samples
            y: Labels
            criterion: Loss function
            
        Returns:
            Detection report
        """
        # Compute gradients
        gradient_norms = self.compute_sample_gradients(X, y, criterion)
        
        # Statistical analysis
        mean_grad = gradient_norms.mean().item()
        std_grad = gradient_norms.std().item()
        
        # Detect outliers
        z_scores = (gradient_norms - mean_grad) / (std_grad + 1e-10)
        outliers = torch.abs(z_scores) > self.threshold
        
        # Store history
        self.gradient_history.append({
            'mean': mean_grad,
            'std': std_grad,
            'num_outliers': int(outliers.sum())
        })
        
        return {
            'mean_gradient': mean_grad,
            'std_gradient': std_grad,
            'gradient_norms': gradient_norms.tolist(),
            'z_scores': z_scores.tolist(),
            'outlier_indices': torch.where(outliers)[0].tolist(),
            'num_outliers': int(outliers.sum())
        }


class BackdoorDetector:
    """
    Detects backdoor triggers in models
    Identifies patterns that cause misclassification
    """
    
    def __init__(self, model: nn.Module, target_class: int):
        """
        Initialize backdoor detector
        
        Args:
            model: Model to analyze
            target_class: Target class for backdoor
        """
        self.model = model
        self.target_class = target_class
        self.model.eval()
        
        logger.info(f"Initialized backdoor detector for target class {target_class}")
        
    def reverse_engineer_trigger(self, X_clean: torch.Tensor,
                                 num_iterations: int = 100,
                                 lr: float = 0.1) -> torch.Tensor:
        """
        Reverse engineer backdoor trigger
        
        Args:
            X_clean: Clean samples
            num_iterations: Optimization iterations
            lr: Learning rate
            
        Returns:
            Potential trigger pattern
        """
        # Initialize trigger as small perturbation
        trigger = torch.zeros_like(X_clean[0], requires_grad=True)
        optimizer = torch.optim.Adam([trigger], lr=lr)
        
        target = torch.tensor([self.target_class])
        
        for i in range(num_iterations):
            # Apply trigger to samples
            X_triggered = X_clean + trigger.unsqueeze(0)
            
            # Forward pass
            output = self.model(X_triggered)
            
            # Loss: maximize target class probability
            if output.dim() > 1:
                loss = -F.cross_entropy(output, target.expand(output.size(0)).long())
            else:
                loss = -F.mse_loss(output, target.float().expand_as(output))
            
            # Regularization: minimize trigger size
            loss = loss + 0.01 * trigger.norm()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return trigger.detach()
        
    def detect_backdoor(self, X_test: torch.Tensor, y_test: torch.Tensor,
                       trigger_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Detect if model has backdoor
        
        Args:
            X_test: Test samples
            y_test: Test labels
            trigger_threshold: Success rate threshold for backdoor
            
        Returns:
            Detection report
        """
        # Reverse engineer trigger
        trigger = self.reverse_engineer_trigger(X_test)
        
        # Test trigger effectiveness
        X_triggered = X_test + trigger.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(X_triggered)
            
            if output.dim() > 1:
                _, predictions = output.max(1)
                success_rate = (predictions == self.target_class).float().mean().item()
            else:
                success_rate = 0.0
        
        # Check if backdoor detected
        backdoor_detected = success_rate > trigger_threshold
        
        # Compute trigger statistics
        trigger_norm = trigger.norm().item()
        trigger_max = trigger.abs().max().item()
        
        return {
            'backdoor_detected': bool(backdoor_detected),
            'success_rate': float(success_rate),
            'trigger_norm': float(trigger_norm),
            'trigger_max': float(trigger_max),
            'trigger_pattern': trigger.tolist(),
            'target_class': int(self.target_class)
        }


class PoisoningDetector:
    """
    Comprehensive poisoning detection system
    Combines multiple detection methods
    """
    
    def __init__(self, model: nn.Module, contamination: float = 0.1):
        """
        Initialize poisoning detector
        
        Args:
            model: Model to protect
            contamination: Expected contamination rate
        """
        self.model = model
        self.data_validator = DataValidator(contamination=contamination)
        self.clustering_detector = ClusteringDetector()
        self.gradient_analyzer = GradientAnalyzer(model)
        
        logger.info("Initialized comprehensive poisoning detector")
        
    def validate_training_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive validation of training data
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Validation report
        """
        # Fit validator on data
        self.data_validator.fit(X)
        
        # Statistical validation
        stat_report = self.data_validator.validate_batch(X)
        
        # Clustering analysis
        cluster_report = self.clustering_detector.detect_poisoned_clusters(X, y)
        
        # Combine results
        all_suspicious = set(stat_report['outlier_indices'] + 
                           cluster_report['suspicious_samples'])
        
        return {
            'total_samples': X.shape[0],
            'statistical_outliers': stat_report['outliers_detected'],
            'clustering_suspicious': cluster_report['total_suspicious'],
            'total_suspicious': len(all_suspicious),
            'suspicious_ratio': len(all_suspicious) / X.shape[0],
            'suspicious_indices': sorted(list(all_suspicious)),
            'statistical_report': stat_report,
            'clustering_report': cluster_report,
            'recommendation': 'REMOVE_SUSPICIOUS' if len(all_suspicious) > 0 else 'DATA_CLEAN'
        }
        
    def monitor_training_batch(self, X: torch.Tensor, y: torch.Tensor,
                              criterion: nn.Module) -> Dict[str, Any]:
        """
        Monitor a training batch for poisoning
        
        Args:
            X: Input batch
            y: Labels
            criterion: Loss function
            
        Returns:
            Monitoring report
        """
        # Gradient analysis
        grad_report = self.gradient_analyzer.detect_anomalous_gradients(X, y, criterion)
        
        return {
            'batch_size': X.size(0),
            'anomalous_gradients': grad_report['num_outliers'],
            'mean_gradient': grad_report['mean_gradient'],
            'gradient_report': grad_report,
            'alert': grad_report['num_outliers'] > 0
        }
        
    def detect_backdoors(self, X_test: torch.Tensor, y_test: torch.Tensor,
                        num_classes: int) -> Dict[str, Any]:
        """
        Detect backdoors for all classes
        
        Args:
            X_test: Test samples
            y_test: Test labels
            num_classes: Number of classes
            
        Returns:
            Detection report for all classes
        """
        results = []
        
        for target_class in range(num_classes):
            detector = BackdoorDetector(self.model, target_class)
            report = detector.detect_backdoor(X_test, y_test)
            results.append(report)
        
        # Aggregate results
        backdoor_detected = any(r['backdoor_detected'] for r in results)
        max_success_rate = max(r['success_rate'] for r in results)
        
        return {
            'backdoor_detected': backdoor_detected,
            'max_success_rate': max_success_rate,
            'per_class_results': results,
            'recommendation': 'RETRAIN_MODEL' if backdoor_detected else 'MODEL_CLEAN'
        }
