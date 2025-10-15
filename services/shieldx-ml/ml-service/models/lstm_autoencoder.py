"""
LSTM Autoencoder for sequential anomaly detection using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time-series anomaly detection
    
    Architecture:
    - Encoder: LSTM layers to encode sequence into fixed representation
    - Decoder: LSTM layers to reconstruct sequence from representation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            latent_dim: Latent representation dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Latent projection
        encoder_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.encoder_projection = nn.Linear(encoder_output_dim, latent_dim)
        
        # Decoder projection
        self.decoder_projection = nn.Linear(latent_dim, hidden_dim)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence to latent representation
        
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            
        Returns:
            Latent representation (batch_size, latent_dim)
        """
        # LSTM encoding
        _, (hidden, _) = self.encoder_lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # Project to latent space
        latent = self.encoder_projection(hidden)
        return latent
    
    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent representation to sequence
        
        Args:
            z: Latent representation (batch_size, latent_dim)
            seq_len: Length of sequence to generate
            
        Returns:
            Reconstructed sequence (batch_size, seq_len, input_dim)
        """
        batch_size = z.size(0)
        
        # Project latent to decoder input
        decoder_input = self.decoder_projection(z)
        
        # Repeat for each timestep
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_len, 1)
        
        # LSTM decoding
        decoder_output, _ = self.decoder_lstm(decoder_input)
        
        # Project to output dimension
        reconstruction = self.output_layer(decoder_output)
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder
        
        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            
        Returns:
            Reconstructed sequence (batch_size, seq_len, input_dim)
        """
        seq_len = x.size(1)
        latent = self.encode(x)
        reconstruction = self.decode(latent, seq_len)
        return reconstruction


class SequentialAnomalyDetector:
    """
    LSTM Autoencoder-based sequential anomaly detection system
    
    Detects anomalies in time-series or sequential data using
    reconstruction error as anomaly score.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        bidirectional: bool = False,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            latent_dim: Latent representation dimension
            num_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
            learning_rate: Learning rate for training
            device: Device to use (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            bidirectional=bidirectional
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
        batch_size: int = 64,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> 'SequentialAnomalyDetector':
        """
        Train the LSTM autoencoder
        
        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            early_stopping_patience: Patience for early stopping
            verbose: Print training progress
            
        Returns:
            self
        """
        # Validate input shape
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D input (n_samples, seq_len, n_features), got {X.shape}")
        
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
            X: Normal sequences
            percentile: Percentile for threshold
        """
        errors = self.reconstruction_error(X)
        self.threshold = np.percentile(errors, percentile)
        logger.info(f"Anomaly threshold set to {self.threshold:.6f} ({percentile}th percentile)")
    
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for sequences
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            
        Returns:
            Reconstruction errors (n_samples,)
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstruction = self.model(X_tensor)
            
            # MSE per sample (averaged over sequence and features)
            errors = torch.mean((X_tensor - reconstruction) ** 2, dim=(1, 2))
            return errors.cpu().numpy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (1 = anomaly, 0 = normal)
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            
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
            X: Input sequences (n_samples, seq_len, n_features)
            
        Returns:
            Anomaly scores (n_samples,)
        """
        errors = self.reconstruction_error(X)
        
        # Normalize by threshold
        if self.threshold is not None:
            scores = errors / self.threshold
        else:
            scores = errors
        
        # Clip to [0, 1]
        return np.clip(scores, 0, 1)
    
    def save(self, path: str) -> None:
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'threshold': self.threshold,
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
            'latent_dim': self.model.latent_dim,
            'num_layers': self.model.num_layers,
            'bidirectional': self.model.bidirectional,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate model with saved architecture
        self.model = LSTMAutoencoder(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            latent_dim=checkpoint['latent_dim'],
            num_layers=checkpoint['num_layers'],
            bidirectional=checkpoint['bidirectional']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.threshold = checkpoint['threshold']
        
        logger.info(f"Model loaded from {path}")
