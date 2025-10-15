"""
Transformer-based models for threat detection
Using self-attention mechanism for capturing complex patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, List
import logging
import math

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for threat detection
    
    Uses multi-head self-attention to capture complex relationships
    in sequential threat patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 6
    ):
        """
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension (embedding size)
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            num_classes: Number of output classes
        """
        super(TransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mask: Attention mask (optional)
            
        Returns:
            Class logits (batch_size, num_classes)
        """
        # Project input to d_model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, mask=mask)
        
        # Global average pooling over sequence dimension
        pooled = encoded.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class TransformerThreatDetector:
    """
    Transformer-based threat detection system
    
    Uses self-attention mechanism to detect complex threat patterns
    in sequential data (network flows, API calls, log sequences, etc.)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 6,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        learning_rate: float = 0.0001,
        device: Optional[str] = None
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Number of threat classes
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            learning_rate: Learning rate
            device: Device (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.num_classes = num_classes
        self.model = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100
        )
        
        # Training metrics
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        
        # Class names
        self.class_names = [
            'Normal', 'SQL_Injection', 'XSS',
            'DDoS', 'Port_Scan', 'Malware'
        ][:num_classes]
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> 'TransformerThreatDetector':
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
            num_batches = 0
            
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
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
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
            
            # Learning rate scheduling
            self.scheduler.step()
            
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
        """Predict threat classes"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            _, predicted = torch.max(logits, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()
    
    def get_attention_weights(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Extract attention weights for interpretability
        
        Returns list of attention weights from each transformer layer
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Need to hook into transformer layers to extract attention
            attention_weights = []
            
            def hook_fn(module, input, output):
                # Extract attention weights if available
                if hasattr(output, 'attention_weights'):
                    attention_weights.append(output.attention_weights.cpu().numpy())
            
            # Register hooks (simplified - full implementation would hook each layer)
            hooks = []
            for layer in self.model.transformer_encoder.layers:
                hook = layer.self_attn.register_forward_hook(hook_fn)
                hooks.append(hook)
            
            # Forward pass
            _ = self.model(X_tensor)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            return attention_weights
    
    def save(self, path: str) -> None:
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'input_dim': self.model.input_dim,
            'd_model': self.model.d_model,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = TransformerEncoder(
            input_dim=checkpoint['input_dim'],
            d_model=checkpoint['d_model'],
            num_classes=checkpoint['num_classes']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_classes = checkpoint['num_classes']
        self.class_names = checkpoint['class_names']
        
        logger.info(f"Model loaded from {path}")
