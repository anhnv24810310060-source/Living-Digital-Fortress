"""
CNN-1D models for packet-level threat detection
Suitable for analyzing network traffic patterns and packet sequences
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CNN1DClassifier(nn.Module):
    """
    1D Convolutional Neural Network for packet classification
    
    Architecture:
    - Multiple Conv1D layers with different kernel sizes
    - Max pooling for dimensionality reduction
    - Fully connected layers for classification
    - Suitable for packet-level threat detection
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_filters: List[int] = None,
        kernel_sizes: List[int] = None,
        dropout: float = 0.3,
        max_seq_len: int = 1000
    ):
        """
        Args:
            input_dim: Input feature dimension per timestep
            num_classes: Number of threat classes to predict
            num_filters: Number of filters for each conv layer
            kernel_sizes: Kernel sizes for each conv layer
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super(CNN1DClassifier, self).__init__()
        
        if num_filters is None:
            num_filters = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_dim
        
        for n_filters, kernel_size in zip(num_filters, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.ReLU(),
                nn.BatchNorm1d(n_filters),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout)
            ])
            in_channels = n_filters
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate output size after convolutions
        self._calculate_conv_output_size(max_seq_len)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def _calculate_conv_output_size(self, seq_len: int):
        """Calculate output size after conv layers"""
        # Create dummy input
        dummy_input = torch.randn(1, self.input_dim, seq_len)
        dummy_output = self.conv_layers(dummy_input)
        self.conv_output_size = dummy_output.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Class logits (batch_size, num_classes)
        """
        # Transpose for Conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        logits = self.fc_layers(x)
        
        return logits


class PacketThreatDetector:
    """
    CNN-1D based packet threat detection system
    
    Classifies network packets into threat categories:
    - Normal traffic
    - SQL Injection
    - XSS attacks
    - DDoS patterns
    - Port scanning
    - Malware communication
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 6,
        num_filters: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Number of threat classes
            num_filters: Conv filters per layer
            kernel_sizes: Kernel sizes per layer
            learning_rate: Learning rate
            device: Device (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.num_classes = num_classes
        self.model = CNN1DClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        
        # Class names for interpretation
        self.class_names = [
            'Normal', 'SQL_Injection', 'XSS', 
            'DDoS', 'Port_Scan', 'Malware'
        ][:num_classes]
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> 'PacketThreatDetector':
        """
        Train the model
        
        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            y: Training labels (n_samples,)
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation fraction
            early_stopping_patience: Early stopping patience
            verbose: Print progress
            
        Returns:
            self
        """
        # Validate input
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D input (n_samples, seq_len, n_features), got {X.shape}")
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i + batch_size]
                batch_y = y_train_tensor[i:i + batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = epoch_loss / (len(X_train_tensor) // batch_size + 1)
            train_accuracy = correct / total
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_tensor)
                val_loss = self.criterion(val_logits, y_val_tensor).item()
                
                _, val_predicted = torch.max(val_logits, 1)
                val_accuracy = (val_predicted == y_val_tensor).sum().item() / len(y_val_tensor)
                
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict threat classes
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            _, predicted = torch.max(logits, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            
        Returns:
            Class probabilities (n_samples, num_classes)
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'input_dim': self.model.input_dim,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = CNN1DClassifier(
            input_dim=checkpoint['input_dim'],
            num_classes=checkpoint['num_classes']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_classes = checkpoint['num_classes']
        self.class_names = checkpoint['class_names']
        
        logger.info(f"Model loaded from {path}")
