"""
Incremental Learning Module

Real-time learning with concept drift adaptation and online feature selection.
Supports streaming data processing and continuous model updates.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class DriftDetectionMethod(Enum):
    """Drift detection methods"""
    DDM = "ddm"  # Drift Detection Method
    EDDM = "eddm"  # Early Drift Detection Method
    ADWIN = "adwin"  # Adaptive Windowing
    PAGE_HINKLEY = "page_hinkley"


@dataclass
class DriftDetectionResult:
    """Result from drift detection"""
    drift_detected: bool
    warning_detected: bool
    drift_point: Optional[int]
    confidence: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DDMDetector:
    """
    Drift Detection Method (DDM)
    
    Monitors error rate and triggers alarm when statistical properties change
    """
    
    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_instances: int = 30
    ):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_instances = min_instances
        
        self.error_rate = 0.0
        self.std_dev = 0.0
        self.min_error_rate = float('inf')
        self.min_std_dev = float('inf')
        self.num_instances = 0
        
        self.drift_detected = False
        self.warning_detected = False
        
        logger.info(f"DDM detector initialized: warning={warning_level}, drift={drift_level}")
    
    def update(self, prediction_correct: bool) -> DriftDetectionResult:
        """
        Update detector with new prediction result
        
        Args:
            prediction_correct: Whether prediction was correct
        
        Returns:
            Drift detection result
        """
        self.num_instances += 1
        error = 0 if prediction_correct else 1
        
        # Update error rate (moving average)
        self.error_rate = ((self.num_instances - 1) * self.error_rate + error) / self.num_instances
        
        # Update standard deviation
        self.std_dev = np.sqrt(self.error_rate * (1 - self.error_rate) / self.num_instances)
        
        self.drift_detected = False
        self.warning_detected = False
        
        if self.num_instances >= self.min_instances:
            # Check for drift
            if self.error_rate + self.std_dev <= self.min_error_rate + self.min_std_dev:
                self.min_error_rate = self.error_rate
                self.min_std_dev = self.std_dev
            
            # Drift detection
            if self.error_rate + self.std_dev >= self.min_error_rate + self.drift_level * self.min_std_dev:
                self.drift_detected = True
                logger.warning(f"Drift detected at instance {self.num_instances}")
            
            # Warning detection
            elif self.error_rate + self.std_dev >= self.min_error_rate + self.warning_level * self.min_std_dev:
                self.warning_detected = True
                logger.info(f"Drift warning at instance {self.num_instances}")
        
        confidence = 1.0 - (self.error_rate + self.std_dev)
        
        return DriftDetectionResult(
            drift_detected=self.drift_detected,
            warning_detected=self.warning_detected,
            drift_point=self.num_instances if self.drift_detected else None,
            confidence=max(0.0, confidence)
        )
    
    def reset(self):
        """Reset detector state"""
        self.error_rate = 0.0
        self.std_dev = 0.0
        self.min_error_rate = float('inf')
        self.min_std_dev = float('inf')
        self.num_instances = 0
        self.drift_detected = False
        self.warning_detected = False


class ADWINDetector:
    """
    Adaptive Windowing (ADWIN) drift detector
    
    Maintains adaptive sliding window and detects distribution changes
    """
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta  # Confidence level
        self.window = deque()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        
        self.drift_detected = False
        
        logger.info(f"ADWIN detector initialized: delta={delta}")
    
    def update(self, value: float) -> DriftDetectionResult:
        """
        Update detector with new value
        
        Args:
            value: New observation value
        
        Returns:
            Drift detection result
        """
        self.window.append(value)
        self.width += 1
        self.total += value
        
        self.drift_detected = False
        drift_point = None
        
        # Check for drift by splitting window
        if self.width >= 2:
            for i in range(1, self.width):
                # Split window at position i
                left_sum = sum(list(self.window)[:i])
                right_sum = self.total - left_sum
                left_mean = left_sum / i
                right_mean = right_sum / (self.width - i)
                
                # Hoeffding bound
                m = 1 / (1/i + 1/(self.width - i))
                epsilon = np.sqrt((1 / (2*m)) * np.log(2 / self.delta))
                
                if abs(left_mean - right_mean) > epsilon:
                    # Drift detected, remove old values
                    for _ in range(i):
                        removed = self.window.popleft()
                        self.total -= removed
                        self.width -= 1
                    
                    self.drift_detected = True
                    drift_point = self.width
                    logger.warning(f"ADWIN drift detected, window reset to {self.width}")
                    break
        
        confidence = 1.0 - abs(self.total / self.width - 0.5) if self.width > 0 else 0.5
        
        return DriftDetectionResult(
            drift_detected=self.drift_detected,
            warning_detected=False,
            drift_point=drift_point,
            confidence=confidence
        )
    
    def reset(self):
        """Reset detector"""
        self.window.clear()
        self.total = 0.0
        self.width = 0
        self.drift_detected = False


class OnlineFeatureSelector:
    """
    Online feature selection with importance tracking
    """
    
    def __init__(
        self,
        num_features: int,
        selection_ratio: float = 0.8,
        update_frequency: int = 100
    ):
        self.num_features = num_features
        self.selection_ratio = selection_ratio
        self.update_frequency = update_frequency
        
        self.feature_importance = np.ones(num_features) / num_features
        self.selected_features = list(range(num_features))
        self.num_updates = 0
        
        logger.info(f"Online feature selector: {num_features} features, "
                   f"selection_ratio={selection_ratio}")
    
    def update_importance(self, gradients: np.ndarray):
        """
        Update feature importance based on gradients
        
        Args:
            gradients: Feature gradients [num_features]
        """
        # Exponential moving average
        alpha = 0.1
        self.feature_importance = (
            alpha * np.abs(gradients) +
            (1 - alpha) * self.feature_importance
        )
        
        self.num_updates += 1
        
        # Periodically update selected features
        if self.num_updates % self.update_frequency == 0:
            self._update_selected_features()
    
    def _update_selected_features(self):
        """Update selected feature set"""
        num_select = int(self.num_features * self.selection_ratio)
        
        # Select top-k most important features
        top_indices = np.argsort(self.feature_importance)[-num_select:]
        self.selected_features = sorted(top_indices.tolist())
        
        logger.info(f"Updated feature selection: {len(self.selected_features)} features")
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features by selecting important ones
        
        Args:
            features: Input features [batch_size, num_features]
        
        Returns:
            Selected features [batch_size, num_selected]
        """
        return features[:, self.selected_features]
    
    def get_selected_features(self) -> List[int]:
        """Get list of selected feature indices"""
        return self.selected_features.copy()


class IncrementalNeuralNetwork(nn.Module):
    """
    Incremental neural network with online learning
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        learning_rate: float = 0.001
    ):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Incremental NN: input={input_size}, hidden={hidden_sizes}, "
                   f"output={output_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)
    
    def partial_fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """
        Perform one step of incremental learning
        
        Args:
            X: Input features [batch_size, input_size]
            y: Target labels [batch_size]
        
        Returns:
            Loss value
        """
        self.train()
        
        self.optimizer.zero_grad()
        outputs = self.forward(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features [batch_size, input_size]
        
        Returns:
            Predicted labels [batch_size]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X)
            _, predicted = outputs.max(1)
        return predicted.numpy()


class RealtimeLearner:
    """
    Real-time learning system with drift detection and adaptation
    
    Features:
    - Incremental model updates
    - Concept drift detection
    - Online feature selection
    - Adaptive learning rate
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [128, 64],
        buffer_size: int = 1000,
        drift_detection_method: str = "ddm",
        feature_selection: bool = True
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.buffer_size = buffer_size
        
        # Incremental model
        self.model = IncrementalNeuralNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size
        )
        
        # Drift detector
        if drift_detection_method == "ddm":
            self.drift_detector = DDMDetector()
        elif drift_detection_method == "adwin":
            self.drift_detector = ADWINDetector()
        else:
            raise ValueError(f"Unknown drift detection method: {drift_detection_method}")
        
        # Feature selector
        self.feature_selector = None
        if feature_selection:
            self.feature_selector = OnlineFeatureSelector(input_size)
        
        # Data buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Statistics
        self.num_updates = 0
        self.num_drifts = 0
        self.accuracy_history = deque(maxlen=100)
        
        logger.info(f"Realtime learner initialized: input={input_size}, "
                   f"output={output_size}, drift_method={drift_detection_method}")
    
    def update(
        self,
        features: np.ndarray,
        label: int,
        retrain_on_drift: bool = True
    ) -> Dict[str, Any]:
        """
        Update model with new sample
        
        Args:
            features: Feature vector [input_size]
            label: True label
            retrain_on_drift: Whether to retrain model on drift detection
        
        Returns:
            Update result with drift info
        """
        # Make prediction first
        X = torch.FloatTensor(features).unsqueeze(0)
        prediction = self.model.predict(X)[0]
        is_correct = (prediction == label)
        
        # Update drift detector
        drift_result = self.drift_detector.update(is_correct)
        
        # Add to buffer
        self.buffer.append((features, label))
        
        # Incremental update
        y = torch.LongTensor([label])
        loss = self.model.partial_fit(X, y)
        
        self.num_updates += 1
        self.accuracy_history.append(1.0 if is_correct else 0.0)
        
        # Handle drift
        if drift_result.drift_detected:
            self.num_drifts += 1
            logger.warning(f"Drift #{self.num_drifts} detected at update {self.num_updates}")
            
            if retrain_on_drift and len(self.buffer) > 10:
                self._retrain_on_buffer()
        
        # Update feature importance
        if self.feature_selector is not None and self.num_updates % 10 == 0:
            # Get gradients for feature importance
            X.requires_grad = True
            outputs = self.model(X)
            outputs[0, label].backward()
            gradients = X.grad.abs().squeeze().numpy()
            self.feature_selector.update_importance(gradients)
        
        return {
            'prediction': int(prediction),
            'is_correct': bool(is_correct),
            'loss': loss,
            'drift_detected': drift_result.drift_detected,
            'drift_confidence': drift_result.confidence,
            'num_updates': self.num_updates,
            'num_drifts': self.num_drifts,
            'accuracy': np.mean(self.accuracy_history) if self.accuracy_history else 0.0
        }
    
    def _retrain_on_buffer(self):
        """Retrain model on buffer data"""
        if len(self.buffer) < 10:
            return
        
        logger.info(f"Retraining on buffer ({len(self.buffer)} samples)")
        
        # Convert buffer to tensors
        features = np.array([x[0] for x in self.buffer])
        labels = np.array([x[1] for x in self.buffer])
        
        X = torch.FloatTensor(features)
        y = torch.LongTensor(labels)
        
        # Mini-batch training
        batch_size = 32
        num_batches = (len(self.buffer) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.buffer))
            
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            self.model.partial_fit(X_batch, y_batch)
        
        # Reset drift detector after retraining
        self.drift_detector.reset()
        
        logger.info("Retraining completed")
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction
        
        Args:
            features: Feature vector [input_size]
        
        Returns:
            Prediction result
        """
        # Apply feature selection if enabled
        if self.feature_selector is not None:
            features = self.feature_selector.transform(features.reshape(1, -1))
            features = features.flatten()
        
        X = torch.FloatTensor(features).unsqueeze(0)
        prediction = self.model.predict(X)[0]
        
        # Get confidence
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
            confidence = probabilities[0, prediction].item()
        
        return {
            'prediction': int(prediction),
            'confidence': confidence,
            'num_updates': self.num_updates,
            'num_drifts': self.num_drifts
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'num_updates': self.num_updates,
            'num_drifts': self.num_drifts,
            'recent_accuracy': np.mean(self.accuracy_history) if self.accuracy_history else 0.0,
            'buffer_size': len(self.buffer),
            'selected_features': self.feature_selector.get_selected_features() if self.feature_selector else None
        }
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_updates': self.num_updates,
            'num_drifts': self.num_drifts,
            'buffer': list(self.buffer)
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.num_updates = checkpoint['num_updates']
        self.num_drifts = checkpoint['num_drifts']
        self.buffer = deque(checkpoint['buffer'], maxlen=self.buffer_size)
        logger.info(f"Model loaded from {filepath}")


class StreamProcessor:
    """
    Process streaming data with real-time learning
    """
    
    def __init__(self, learner: RealtimeLearner):
        self.learner = learner
        self.start_time = None
        self.num_processed = 0
        
    def process_stream(
        self,
        data_stream: Any,
        label_func: Callable,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process data stream
        
        Args:
            data_stream: Iterator of data samples
            label_func: Function to get labels
            max_samples: Maximum samples to process
        
        Returns:
            Processing statistics
        """
        self.start_time = time.time()
        self.num_processed = 0
        
        results = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'drifts_detected': 0
        }
        
        for i, sample in enumerate(data_stream):
            if max_samples is not None and i >= max_samples:
                break
            
            features = sample['features']
            label = label_func(sample)
            
            update_result = self.learner.update(features, label)
            
            if update_result['is_correct']:
                results['correct_predictions'] += 1
            results['total_predictions'] += 1
            
            if update_result['drift_detected']:
                results['drifts_detected'] += 1
            
            self.num_processed += 1
            
            # Log progress
            if (i + 1) % 1000 == 0:
                accuracy = results['correct_predictions'] / results['total_predictions']
                logger.info(f"Processed {i+1} samples, accuracy={accuracy:.4f}")
        
        elapsed_time = time.time() - self.start_time
        results['elapsed_time'] = elapsed_time
        results['throughput'] = self.num_processed / elapsed_time if elapsed_time > 0 else 0
        results['accuracy'] = results['correct_predictions'] / results['total_predictions']
        
        return results
