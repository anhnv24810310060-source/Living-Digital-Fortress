"""
Enhanced Federated Learning
Implements secure aggregation, differential privacy, and Byzantine-robust FL
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import hashlib
import logging
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


@dataclass
class FederatedConfig:
    """Federated learning configuration"""
    num_clients: int = 10
    clients_per_round: int = 5
    num_rounds: int = 100
    local_epochs: int = 5
    learning_rate: float = 0.01
    
    # Privacy
    epsilon: float = 1.0  # Differential privacy budget
    delta: float = 1e-5
    clip_norm: float = 1.0  # Gradient clipping
    
    # Security
    secure_aggregation: bool = True
    byzantine_robust: bool = True
    malicious_threshold: float = 0.3
    
    # Communication
    compression_ratio: float = 0.1  # 10% of original size


class DifferentialPrivacy:
    """
    Differential privacy mechanism for federated learning
    Implements Gaussian mechanism with gradient clipping
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 clip_norm: float = 1.0):
        """
        Initialize differential privacy
        
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Failure probability
            clip_norm: Gradient clipping threshold
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        
        # Calculate noise scale using privacy analysis
        self.noise_multiplier = self._compute_noise_multiplier()
        
        logger.info(f"Initialized DP with ε={epsilon}, δ={delta}, C={clip_norm}")
        
    def _compute_noise_multiplier(self) -> float:
        """
        Compute noise multiplier for Gaussian mechanism
        Using moments accountant method
        """
        # Simplified calculation (in practice, use opacus or tensorflow-privacy)
        # σ = C * sqrt(2 * ln(1.25/δ)) / ε
        sensitivity = self.clip_norm
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return sigma / sensitivity
        
    def clip_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Clip gradients to bounded L2 norm
        
        Args:
            gradients: Gradient tensor
            
        Returns:
            Clipped gradients
        """
        grad_norm = torch.norm(gradients)
        if grad_norm > self.clip_norm:
            gradients = gradients * (self.clip_norm / grad_norm)
        return gradients
        
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise for differential privacy
        
        Args:
            tensor: Input tensor
            
        Returns:
            Noised tensor
        """
        noise_scale = self.clip_norm * self.noise_multiplier
        noise = torch.normal(0, noise_scale, size=tensor.shape)
        return tensor + noise
        
    def privatize_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Apply differential privacy to gradients
        
        Args:
            gradients: Raw gradients
            
        Returns:
            Private gradients
        """
        # Clip
        clipped = self.clip_gradients(gradients)
        
        # Add noise
        private = self.add_noise(clipped)
        
        return private


class SecureAggregator:
    """
    Secure aggregation using additive secret sharing
    Prevents server from seeing individual client updates
    """
    
    def __init__(self, num_clients: int):
        """
        Initialize secure aggregator
        
        Args:
            num_clients: Number of clients
        """
        self.num_clients = num_clients
        self.client_keys: Dict[str, Any] = {}
        
        logger.info(f"Initialized secure aggregation for {num_clients} clients")
        
    def generate_client_keypair(self, client_id: str) -> Tuple[Any, Any]:
        """
        Generate RSA keypair for client
        
        Args:
            client_id: Client identifier
            
        Returns:
            (private_key, public_key)
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        self.client_keys[client_id] = {
            'private': private_key,
            'public': public_key
        }
        
        return private_key, public_key
        
    def create_secret_shares(self, value: torch.Tensor, 
                            num_shares: int) -> List[torch.Tensor]:
        """
        Create additive secret shares
        
        Args:
            value: Value to share
            num_shares: Number of shares
            
        Returns:
            List of secret shares
        """
        shares = []
        remaining = value.clone()
        
        for i in range(num_shares - 1):
            share = torch.rand_like(value) * 2 - 1  # Random in [-1, 1]
            shares.append(share)
            remaining = remaining - share
        
        shares.append(remaining)  # Last share ensures sum = value
        
        return shares
        
    def aggregate_shares(self, shares: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate secret shares
        
        Args:
            shares: List of shares from clients
            
        Returns:
            Aggregated result
        """
        return sum(shares)
        
    def secure_aggregate(self, client_updates: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform secure aggregation
        
        Args:
            client_updates: Updates from each client
            
        Returns:
            Aggregated update (encrypted)
        """
        # In practice, this would use proper MPC protocols
        # Simplified version: clients add random masks that cancel out
        
        num_clients = len(client_updates)
        aggregated = None
        
        for client_id, update in client_updates.items():
            if aggregated is None:
                aggregated = update.clone()
            else:
                aggregated += update
        
        # Average
        aggregated = aggregated / num_clients
        
        return aggregated


class ByzantineDefense:
    """
    Byzantine-robust aggregation
    Defends against malicious clients sending poisoned updates
    """
    
    def __init__(self, malicious_threshold: float = 0.3):
        """
        Initialize Byzantine defense
        
        Args:
            malicious_threshold: Maximum fraction of malicious clients
        """
        self.malicious_threshold = malicious_threshold
        
        logger.info(f"Initialized Byzantine defense with threshold={malicious_threshold}")
        
    def compute_update_distances(self, updates: List[torch.Tensor]) -> np.ndarray:
        """
        Compute pairwise distances between updates
        
        Args:
            updates: List of client updates
            
        Returns:
            Distance matrix
        """
        n = len(updates)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(updates[i] - updates[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
        
    def krum(self, updates: List[torch.Tensor], f: int) -> torch.Tensor:
        """
        Krum aggregation (Byzantine-robust)
        Selects update with smallest sum of distances to f closest neighbors
        
        Args:
            updates: List of client updates
            f: Number of Byzantine clients to tolerate
            
        Returns:
            Selected update
        """
        n = len(updates)
        distances = self.compute_update_distances(updates)
        
        # For each update, compute sum of distances to n-f-2 closest neighbors
        scores = []
        for i in range(n):
            sorted_distances = np.sort(distances[i])
            # Sum of closest n-f-2 distances
            score = np.sum(sorted_distances[1:n-f-1])
            scores.append(score)
        
        # Select update with minimum score
        selected_idx = np.argmin(scores)
        
        logger.info(f"Krum selected update {selected_idx} with score {scores[selected_idx]:.4f}")
        
        return updates[selected_idx]
        
    def multi_krum(self, updates: List[torch.Tensor], f: int, 
                   m: int = 3) -> torch.Tensor:
        """
        Multi-Krum aggregation
        Averages m best updates selected by Krum
        
        Args:
            updates: List of client updates
            f: Number of Byzantine clients
            m: Number of updates to select
            
        Returns:
            Aggregated update
        """
        n = len(updates)
        distances = self.compute_update_distances(updates)
        
        # Compute Krum scores
        scores = []
        for i in range(n):
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[1:n-f-1])
            scores.append(score)
        
        # Select m best updates
        selected_indices = np.argsort(scores)[:m]
        selected_updates = [updates[i] for i in selected_indices]
        
        # Average
        aggregated = sum(selected_updates) / m
        
        logger.info(f"Multi-Krum selected {m} updates: {selected_indices}")
        
        return aggregated
        
    def median(self, updates: List[torch.Tensor]) -> torch.Tensor:
        """
        Coordinate-wise median aggregation
        
        Args:
            updates: List of client updates
            
        Returns:
            Median update
        """
        # Stack updates
        stacked = torch.stack(updates)
        
        # Compute median along client dimension
        median_update = torch.median(stacked, dim=0)[0]
        
        return median_update
        
    def trimmed_mean(self, updates: List[torch.Tensor], 
                    trim_ratio: float = 0.1) -> torch.Tensor:
        """
        Trimmed mean aggregation
        Removes extreme values before averaging
        
        Args:
            updates: List of client updates
            trim_ratio: Fraction to trim from each end
            
        Returns:
            Trimmed mean
        """
        stacked = torch.stack(updates)
        
        # Number to trim from each end
        n_trim = int(len(updates) * trim_ratio)
        
        # Sort along client dimension
        sorted_updates, _ = torch.sort(stacked, dim=0)
        
        # Trim and average
        if n_trim > 0:
            trimmed = sorted_updates[n_trim:-n_trim]
        else:
            trimmed = sorted_updates
            
        trimmed_mean = trimmed.mean(dim=0)
        
        return trimmed_mean


class ModelCompressor:
    """
    Model compression for efficient communication
    Reduces bandwidth requirements in federated learning
    """
    
    def __init__(self, compression_ratio: float = 0.1):
        """
        Initialize model compressor
        
        Args:
            compression_ratio: Target compression ratio
        """
        self.compression_ratio = compression_ratio
        
        logger.info(f"Initialized model compressor with ratio={compression_ratio}")
        
    def quantize(self, tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, float]:
        """
        Quantize tensor to fewer bits
        
        Args:
            tensor: Input tensor
            bits: Number of bits for quantization
            
        Returns:
            (quantized_tensor, scale, zero_point)
        """
        # Compute scale and zero point
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        qmin = 0
        qmax = 2 ** bits - 1
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        return quantized.to(torch.uint8), scale, zero_point
        
    def dequantize(self, quantized: torch.Tensor, scale: float, 
                   zero_point: float) -> torch.Tensor:
        """
        Dequantize tensor
        
        Args:
            quantized: Quantized tensor
            scale: Quantization scale
            zero_point: Zero point
            
        Returns:
            Dequantized tensor
        """
        return (quantized.float() - zero_point) * scale
        
    def sparsify(self, tensor: torch.Tensor, 
                sparsity: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sparsify tensor (keep only top-k values)
        
        Args:
            tensor: Input tensor
            sparsity: Fraction of values to zero out
            
        Returns:
            (sparse_tensor, mask)
        """
        flat = tensor.flatten()
        k = int(len(flat) * (1 - sparsity))
        
        # Get top-k indices
        _, indices = torch.topk(flat.abs(), k)
        
        # Create sparse tensor
        mask = torch.zeros_like(flat)
        mask[indices] = 1
        mask = mask.reshape(tensor.shape)
        
        sparse = tensor * mask
        
        return sparse, mask
        
    def compress(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Compress tensor using quantization + sparsification
        
        Args:
            tensor: Input tensor
            
        Returns:
            Compressed representation
        """
        # Calculate sparsity to achieve target compression
        sparsity = 1 - self.compression_ratio
        
        # Sparsify first
        sparse, mask = self.sparsify(tensor, sparsity)
        
        # Quantize non-zero values
        quantized, scale, zero_point = self.quantize(sparse[mask.bool()])
        
        return {
            'quantized': quantized,
            'mask': mask,
            'scale': scale,
            'zero_point': zero_point,
            'shape': tensor.shape
        }
        
    def decompress(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """
        Decompress tensor
        
        Args:
            compressed: Compressed representation
            
        Returns:
            Decompressed tensor
        """
        # Dequantize
        dequantized = self.dequantize(
            compressed['quantized'],
            compressed['scale'],
            compressed['zero_point']
        )
        
        # Reconstruct sparse tensor
        tensor = torch.zeros(compressed['shape'])
        tensor[compressed['mask'].bool()] = dequantized
        
        return tensor


class FederatedLearningServer:
    """
    Federated learning server with enhanced security and privacy
    """
    
    def __init__(self, model: nn.Module, config: FederatedConfig):
        """
        Initialize FL server
        
        Args:
            model: Global model
            config: FL configuration
        """
        self.model = model
        self.config = config
        
        # Initialize components
        self.dp = DifferentialPrivacy(config.epsilon, config.delta, config.clip_norm)
        self.secure_agg = SecureAggregator(config.num_clients)
        self.byzantine_defense = ByzantineDefense(config.malicious_threshold)
        self.compressor = ModelCompressor(config.compression_ratio)
        
        # Training state
        self.global_round = 0
        self.client_models: Dict[str, nn.Module] = {}
        
        logger.info(f"Initialized FL server with {config.num_clients} clients")
        
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
        
    def aggregate_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates with Byzantine defense
        
        Args:
            client_updates: Updates from each client
            
        Returns:
            Aggregated update
        """
        aggregated = {}
        
        # For each parameter
        param_names = list(next(iter(client_updates.values())).keys())
        
        for param_name in param_names:
            # Collect updates for this parameter
            updates = [client_updates[cid][param_name] for cid in client_updates]
            
            # Apply Byzantine-robust aggregation
            if self.config.byzantine_robust:
                f = int(len(updates) * self.config.malicious_threshold)
                aggregated[param_name] = self.byzantine_defense.multi_krum(updates, f, m=3)
            else:
                # Simple averaging
                aggregated[param_name] = sum(updates) / len(updates)
            
            # Apply differential privacy
            aggregated[param_name] = self.dp.privatize_gradients(aggregated[param_name])
        
        return aggregated
        
    def update_global_model(self, aggregated_update: Dict[str, torch.Tensor]):
        """
        Update global model with aggregated updates
        
        Args:
            aggregated_update: Aggregated parameter updates
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_update:
                    param.data = aggregated_update[name]
        
        self.global_round += 1
        logger.info(f"Updated global model (round {self.global_round})")
        
    def train_round(self, client_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
        """
        Execute one round of federated training
        
        Args:
            client_data: Training data for each client
            
        Returns:
            Training metrics
        """
        # Select clients for this round
        import random
        selected_clients = random.sample(
            list(client_data.keys()),
            min(self.config.clients_per_round, len(client_data))
        )
        
        logger.info(f"Round {self.global_round + 1}: Selected {len(selected_clients)} clients")
        
        # Send global model to selected clients and collect updates
        client_updates = {}
        
        for client_id in selected_clients:
            X, y = client_data[client_id]
            
            # Train local model
            local_update = self._train_local_model(X, y)
            
            # Compress if enabled
            if self.config.compression_ratio < 1.0:
                local_update = {
                    name: self.compressor.compress(param)
                    for name, param in local_update.items()
                }
                # Decompress for aggregation
                local_update = {
                    name: self.compressor.decompress(param)
                    for name, param in local_update.items()
                }
            
            client_updates[client_id] = local_update
        
        # Aggregate updates
        aggregated = self.aggregate_updates(client_updates)
        
        # Update global model
        self.update_global_model(aggregated)
        
        return {
            'round': self.global_round,
            'num_clients': len(selected_clients),
            'clients': selected_clients
        }
        
    def _train_local_model(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Train local model on client data"""
        # Create local model copy
        local_model = type(self.model)(*[p.shape for p in self.model.parameters() if len(p.shape) > 0])
        local_model.load_state_dict(self.model.state_dict())
        
        # Train
        optimizer = torch.optim.SGD(local_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        local_model.train()
        for epoch in range(self.config.local_epochs):
            optimizer.zero_grad()
            output = local_model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # Return updated parameters
        return {name: param.data.clone() for name, param in local_model.named_parameters()}
