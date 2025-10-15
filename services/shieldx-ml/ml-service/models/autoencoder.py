"""
Autoencoder models for anomaly detection using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class BasicAutoencoder(nn.Module):
    """
    Basic Autoencoder for anomaly detection
    
    Architecture:
    - Encoder: input_dim -> hidden_dims -> latent_dim
    - Decoder: latent_dim -> hidden_dims (reversed) -> input_dim
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2
    ):
        """
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent representation dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
        """
        super(BasicAutoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder"""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)


class AnomalyDetectionAE:
    """
    Autoencoder-based anomaly detection system
    
    Uses reconstruction error as anomaly score.
    Higher reconstruction error indicates anomaly.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent representation dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for training
            device: Device to use (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = BasicAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.threshold: Optional[float] = None
        
        # Training metrics
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
    
    def fit(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> 'AnomalyDetectionAE':
        """
        Train the autoencoder
        
        Args:
            X: Training data (n_samples, n_features)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            early_stopping_patience: Patience for early stopping
            verbose: Print training progress
            
        Returns:
            self
        """
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train = X[:split_idx]
        X_val = X[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch = X_train_tensor[i:i + batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                reconstruction = self.model(batch)
                loss = self.criterion(reconstruction, batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_reconstruction = self.model(X_val_tensor)
                val_loss = self.criterion(val_reconstruction, X_val_tensor).item()
                self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
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
        
        # Calculate threshold from validation set
        self._calculate_threshold(X_val)
        
        return self
    
    def _calculate_threshold(
        self,
        X: np.ndarray,
        percentile: float = 95.0
    ) -> None:
        """
        Calculate anomaly threshold from reconstruction errors
        
        Args:
            X: Normal data samples
            percentile: Percentile for threshold (e.g., 95 means 5% false positive)
        """
        errors = self.reconstruction_error(X)
        self.threshold = np.percentile(errors, percentile)
        logger.info(f"Anomaly threshold set to {self.threshold:.6f} ({percentile}th percentile)")
    
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for samples
        
        Args:
            X: Input samples (n_samples, n_features)
            
        Returns:
            Reconstruction errors (n_samples,)
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstruction = self.model(X_tensor)
            
            # MSE per sample
            errors = torch.mean((X_tensor - reconstruction) ** 2, dim=1)
            return errors.cpu().numpy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (1 = anomaly, 0 = normal)
        
        Args:
            X: Input samples (n_samples, n_features)
            
        Returns:
            Binary predictions (n_samples,)
        """
        if self.threshold is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        errors = self.reconstruction_error(X)
        return (errors > self.threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly probability scores
        
        Args:
            X: Input samples (n_samples, n_features)
            
        Returns:
            Anomaly scores (n_samples,), higher = more anomalous
        """
        errors = self.reconstruction_error(X)
        
        # Normalize by threshold
        if self.threshold is not None:
            scores = errors / self.threshold
        else:
            scores = errors
        
        # Clip to [0, 1] and return as probability
        return np.clip(scores, 0, 1)
    
    def save(self, path: str) -> None:
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'threshold': self.threshold,
            'input_dim': self.model.input_dim,
            'latent_dim': self.model.latent_dim,
            'hidden_dims': self.model.hidden_dims,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate model with saved architecture
        self.model = BasicAutoencoder(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim'],
            hidden_dims=checkpoint['hidden_dims']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.threshold = checkpoint['threshold']
        
        logger.info(f"Model loaded from {path}")
