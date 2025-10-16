"""
Transfer Learning Module

Pre-trained model integration and fine-tuning for cybersecurity tasks.
Supports BERT, ResNet, and custom model adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)


class PretrainedModel(Enum):
    """Available pre-trained models"""
    BERT_BASE = "bert-base-uncased"
    BERT_LARGE = "bert-large-uncased"
    DISTILBERT = "distilbert-base-uncased"
    ROBERTA_BASE = "roberta-base"
    ROBERTA_LARGE = "roberta-large"
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    VIT_BASE = "vit-base-patch16-224"


class FineTuningStrategy(Enum):
    """Fine-tuning strategies"""
    FULL = "full"  # Fine-tune all layers
    FEATURE_EXTRACTION = "feature_extraction"  # Freeze base, train head only
    GRADUAL_UNFREEZING = "gradual_unfreezing"  # Progressively unfreeze layers
    DISCRIMINATIVE_LR = "discriminative_lr"  # Different LR for different layers
    ADAPTER = "adapter"  # Add adapter layers


@dataclass
class TransferConfig:
    """Configuration for transfer learning"""
    base_model: str
    num_classes: int
    strategy: str
    freeze_layers: Optional[List[str]] = None
    learning_rate: float = 1e-4
    base_lr_multiplier: float = 0.1  # For discriminative LR
    max_seq_length: int = 512  # For text models
    dropout: float = 0.1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TransferConfig':
        """Create from dictionary"""
        return cls(**data)


class BERTForSequenceClassification(nn.Module):
    """
    BERT-based sequence classification for log analysis
    """
    
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Simulated BERT encoder (in production, use transformers library)
        self.embedding = nn.Embedding(30522, hidden_size)  # BERT vocab size
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=3072,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=12
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        logger.info(f"Initialized BERT classifier: hidden={hidden_size}, "
                   f"classes={num_classes}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # Embedding
        embeddings = self.embedding(input_ids)
        
        # Encode
        if attention_mask is not None:
            # Convert attention mask to additive mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        encoded = self.encoder(embeddings, src_key_padding_mask=attention_mask)
        
        # Pool (use [CLS] token representation)
        pooled = encoded[:, 0, :]
        
        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits


class ResNetForImageClassification(nn.Module):
    """
    ResNet-based image classification
    """
    
    def __init__(
        self,
        num_classes: int,
        layers: List[int] = [2, 2, 2, 2],  # ResNet-18
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, layers[0])
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        
        logger.info(f"Initialized ResNet classifier: layers={layers}, "
                   f"classes={num_classes}")
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer"""
        layers = []
        
        # First block (may have stride > 1)
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images [batch_size, 3, H, W]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for ResNet"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class AdapterLayer(nn.Module):
    """Adapter layer for parameter-efficient fine-tuning"""
    
    def __init__(self, hidden_size: int, adapter_size: int = 64):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Down-project
        adapter_input = self.down_project(x)
        adapter_input = self.activation(adapter_input)
        
        # Up-project
        adapter_output = self.up_project(adapter_input)
        
        # Residual connection
        return x + adapter_output


class TransferLearningManager:
    """
    Manage transfer learning pipelines
    
    Features:
    - Pre-trained model loading
    - Fine-tuning strategies
    - Domain adaptation
    - Layer freezing/unfreezing
    """
    
    def __init__(self):
        self.models: Dict[str, nn.Module] = {}
        self.configs: Dict[str, TransferConfig] = {}
        
        logger.info("Transfer learning manager initialized")
    
    def load_pretrained_model(
        self,
        model_name: str,
        config: TransferConfig
    ) -> nn.Module:
        """
        Load pre-trained model
        
        Args:
            model_name: Model identifier
            config: Transfer learning configuration
        
        Returns:
            Loaded model
        """
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return self.models[model_name]
        
        # Load appropriate model based on type
        if "bert" in config.base_model.lower():
            model = BERTForSequenceClassification(
                num_classes=config.num_classes,
                dropout=config.dropout
            )
        elif "resnet" in config.base_model.lower():
            model = ResNetForImageClassification(
                num_classes=config.num_classes,
                dropout=config.dropout
            )
        else:
            raise ValueError(f"Unsupported model: {config.base_model}")
        
        # Apply fine-tuning strategy
        self._apply_strategy(model, config)
        
        self.models[model_name] = model
        self.configs[model_name] = config
        
        logger.info(f"Loaded pre-trained model: {config.base_model}, "
                   f"strategy={config.strategy}")
        
        return model
    
    def _apply_strategy(self, model: nn.Module, config: TransferConfig):
        """Apply fine-tuning strategy"""
        strategy = FineTuningStrategy(config.strategy)
        
        if strategy == FineTuningStrategy.FEATURE_EXTRACTION:
            # Freeze all layers except classification head
            self._freeze_base_layers(model)
            logger.info("Applied feature extraction strategy (frozen base)")
        
        elif strategy == FineTuningStrategy.FULL:
            # Train all layers
            logger.info("Applied full fine-tuning strategy")
        
        elif strategy == FineTuningStrategy.GRADUAL_UNFREEZING:
            # Start with frozen base, will unfreeze progressively
            self._freeze_base_layers(model)
            logger.info("Applied gradual unfreezing strategy (initial freeze)")
        
        elif strategy == FineTuningStrategy.ADAPTER:
            # Add adapter layers
            self._add_adapter_layers(model)
            logger.info("Applied adapter strategy")
    
    def _freeze_base_layers(self, model: nn.Module):
        """Freeze all layers except classification head"""
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classification head
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
    
    def _add_adapter_layers(self, model: nn.Module):
        """Add adapter layers to model"""
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False
        
        # Add adapters (simplified - in production, add to each layer)
        if hasattr(model, 'encoder'):
            # For transformer models
            model.adapter = AdapterLayer(768)  # BERT hidden size
            model.adapter.requires_grad_(True)
    
    def fine_tune(
        self,
        model_name: str,
        train_data: torch.utils.data.DataLoader,
        val_data: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 3,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Fine-tune model on task data
        
        Args:
            model_name: Model identifier
            train_data: Training data loader
            val_data: Validation data loader
            num_epochs: Number of training epochs
            device: Training device
        
        Returns:
            Training history
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        config = self.configs[model_name]
        
        model = model.to(device)
        model.train()
        
        # Setup optimizer with discriminative learning rates
        optimizer = self._get_optimizer(model, config)
        criterion = nn.CrossEntropyLoss()
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        logger.info(f"Starting fine-tuning for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self._train_epoch(
                model, train_data, optimizer, criterion, device
            )
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            if val_data is not None:
                val_loss, val_acc = self._validate(
                    model, val_data, criterion, device
                )
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                           f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
            
            # Gradual unfreezing
            if config.strategy == FineTuningStrategy.GRADUAL_UNFREEZING.value:
                self._unfreeze_layer(model, epoch)
        
        logger.info("Fine-tuning completed")
        return history
    
    def _get_optimizer(
        self,
        model: nn.Module,
        config: TransferConfig
    ) -> torch.optim.Optimizer:
        """Get optimizer with discriminative learning rates"""
        if config.strategy == FineTuningStrategy.DISCRIMINATIVE_LR.value:
            # Different learning rates for base and head
            base_params = []
            head_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'classifier' in name or 'fc' in name:
                        head_params.append(param)
                    else:
                        base_params.append(param)
            
            optimizer = torch.optim.AdamW([
                {'params': base_params, 'lr': config.learning_rate * config.base_lr_multiplier},
                {'params': head_params, 'lr': config.learning_rate}
            ])
            
            logger.info(f"Using discriminative LR: base={config.learning_rate * config.base_lr_multiplier:.6f}, "
                       f"head={config.learning_rate:.6f}")
        else:
            # Single learning rate
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.learning_rate
            )
        
        return optimizer
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str
    ) -> Tuple[float, float]:
        """Train one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_data):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(train_data)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(
        self,
        model: nn.Module,
        val_data: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: str
    ) -> Tuple[float, float]:
        """Validate model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_data:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(val_data)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _unfreeze_layer(self, model: nn.Module, epoch: int):
        """Progressively unfreeze layers"""
        # Simplified: unfreeze one layer group per epoch
        layers = []
        if hasattr(model, 'layer4'):
            layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        elif hasattr(model, 'encoder'):
            # For transformer models, unfreeze encoder layers
            if hasattr(model.encoder, 'layers'):
                layers = list(reversed(model.encoder.layers))
        
        if epoch < len(layers):
            for param in layers[epoch].parameters():
                param.requires_grad = True
            logger.info(f"Unfroze layer group {epoch+1}")
    
    def domain_adaptation(
        self,
        model_name: str,
        source_data: torch.utils.data.DataLoader,
        target_data: torch.utils.data.DataLoader,
        num_epochs: int = 5,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Perform domain adaptation
        
        Uses simple adversarial training approach
        
        Args:
            model_name: Model identifier
            source_data: Source domain data
            target_data: Target domain data
            num_epochs: Training epochs
            device: Training device
        
        Returns:
            Adaptation history
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        model = model.to(device)
        
        # Domain classifier (binary: source vs target)
        feature_dim = 768 if 'bert' in model_name.lower() else 512
        domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(domain_classifier.parameters()),
            lr=1e-4
        )
        
        task_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()
        
        history = {'task_loss': [], 'domain_loss': []}
        
        logger.info(f"Starting domain adaptation for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            model.train()
            domain_classifier.train()
            
            epoch_task_loss = 0
            epoch_domain_loss = 0
            
            # Train on both source and target
            for (source_inputs, source_labels), (target_inputs, _) in zip(source_data, target_data):
                source_inputs = source_inputs.to(device)
                source_labels = source_labels.to(device)
                target_inputs = target_inputs.to(device)
                
                # Task loss on source
                source_outputs = model(source_inputs)
                task_loss = task_criterion(source_outputs, source_labels)
                
                # Domain loss (adversarial)
                source_features = source_outputs.detach()
                target_features = model(target_inputs).detach()
                
                source_domain = torch.zeros(source_features.size(0), dtype=torch.long).to(device)
                target_domain = torch.ones(target_features.size(0), dtype=torch.long).to(device)
                
                domain_pred_source = domain_classifier(source_features)
                domain_pred_target = domain_classifier(target_features)
                
                domain_loss = (
                    domain_criterion(domain_pred_source, source_domain) +
                    domain_criterion(domain_pred_target, target_domain)
                )
                
                # Total loss
                loss = task_loss + 0.5 * domain_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_task_loss += task_loss.item()
                epoch_domain_loss += domain_loss.item()
            
            avg_task_loss = epoch_task_loss / len(source_data)
            avg_domain_loss = epoch_domain_loss / len(source_data)
            
            history['task_loss'].append(avg_task_loss)
            history['domain_loss'].append(avg_domain_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"task_loss={avg_task_loss:.4f}, "
                       f"domain_loss={avg_domain_loss:.4f}")
        
        logger.info("Domain adaptation completed")
        return history
    
    def export_model(self, model_name: str, filepath: str):
        """Export fine-tuned model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        config = self.configs[model_name]
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config.to_dict()
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model exported to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load fine-tuned model"""
        checkpoint = torch.load(filepath)
        config = TransferConfig.from_dict(checkpoint['config'])
        
        model = self.load_pretrained_model(model_name, config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        return model
