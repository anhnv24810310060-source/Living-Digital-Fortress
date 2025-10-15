# 🛠️ ML Master Level - Implementation Guide

Chi tiết kỹ thuật cho việc triển khai từng component trong lộ trình ML Master.

---

## 📋 Table of Contents

1. [Advanced Anomaly Detection](#1-advanced-anomaly-detection)
2. [Deep Learning Models](#2-deep-learning-models)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Explainability](#4-model-explainability)
5. [Adversarial Defense](#5-adversarial-defense)
6. [Performance Optimization](#6-performance-optimization)

---

## 1. Advanced Anomaly Detection

### 1.1 Local Outlier Factor (LOF)

#### Implementation
```go
// pkg/ml/lof_detector.go
package ml

import (
    "math"
    "sort"
)

type LOFDetector struct {
    k          int       // Number of neighbors
    trained    bool
    dataPoints [][]float64
    distances  [][]float64
}

func NewLOFDetector(k int) *LOFDetector {
    return &LOFDetector{
        k: k,
    }
}

func (lof *LOFDetector) Train(data [][]float64) error {
    lof.dataPoints = make([][]float64, len(data))
    copy(lof.dataPoints, data)
    lof.trained = true
    return nil
}

func (lof *LOFDetector) Detect(point []float64) (bool, float64) {
    if !lof.trained {
        return false, 0.0
    }
    
    // 1. Find k-nearest neighbors
    neighbors := lof.findKNearestNeighbors(point, lof.k)
    
    // 2. Calculate k-distance
    kDistance := lof.calculateKDistance(point, neighbors)
    
    // 3. Calculate reachability distance
    reachDist := lof.calculateReachabilityDistance(point, neighbors)
    
    // 4. Calculate Local Reachability Density (LRD)
    lrd := lof.calculateLRD(point, neighbors)
    
    // 5. Calculate LOF score
    lofScore := 0.0
    for _, neighbor := range neighbors {
        neighborLRD := lof.calculateLRD(neighbor, lof.findKNearestNeighbors(neighbor, lof.k))
        lofScore += neighborLRD / lrd
    }
    lofScore /= float64(len(neighbors))
    
    // Normalize score
    normalizedScore := math.Min(lofScore/10.0, 1.0)
    isAnomaly := lofScore > 1.5 // Threshold
    
    return isAnomaly, normalizedScore
}

func (lof *LOFDetector) findKNearestNeighbors(point []float64, k int) [][]float64 {
    type distancePoint struct {
        distance float64
        point    []float64
    }
    
    distances := make([]distancePoint, 0, len(lof.dataPoints))
    for _, dataPoint := range lof.dataPoints {
        dist := euclideanDistance(point, dataPoint)
        distances = append(distances, distancePoint{dist, dataPoint})
    }
    
    sort.Slice(distances, func(i, j int) bool {
        return distances[i].distance < distances[j].distance
    })
    
    neighbors := make([][]float64, 0, k)
    for i := 0; i < k && i < len(distances); i++ {
        neighbors = append(neighbors, distances[i].point)
    }
    
    return neighbors
}

func (lof *LOFDetector) calculateKDistance(point []float64, neighbors [][]float64) float64 {
    if len(neighbors) == 0 {
        return 0
    }
    return euclideanDistance(point, neighbors[len(neighbors)-1])
}

func (lof *LOFDetector) calculateReachabilityDistance(point []float64, neighbors [][]float64) float64 {
    sum := 0.0
    for _, neighbor := range neighbors {
        dist := euclideanDistance(point, neighbor)
        kDist := lof.calculateKDistance(neighbor, lof.findKNearestNeighbors(neighbor, lof.k))
        sum += math.Max(dist, kDist)
    }
    return sum / float64(len(neighbors))
}

func (lof *LOFDetector) calculateLRD(point []float64, neighbors [][]float64) float64 {
    reachDist := lof.calculateReachabilityDistance(point, neighbors)
    if reachDist == 0 {
        return math.Inf(1)
    }
    return 1.0 / reachDist
}

func euclideanDistance(a, b []float64) float64 {
    sum := 0.0
    for i := range a {
        diff := a[i] - b[i]
        sum += diff * diff
    }
    return math.Sqrt(sum)
}
```

#### Testing
```go
// pkg/ml/lof_detector_test.go
package ml

import (
    "testing"
)

func TestLOFDetector(t *testing.T) {
    // Create training data (normal points)
    trainData := [][]float64{
        {1.0, 1.0}, {1.1, 1.0}, {0.9, 1.1},
        {1.0, 0.9}, {1.2, 1.1}, {0.8, 0.9},
    }
    
    detector := NewLOFDetector(3)
    err := detector.Train(trainData)
    if err != nil {
        t.Fatalf("Training failed: %v", err)
    }
    
    // Test normal point
    normal := []float64{1.05, 1.05}
    isAnomaly, score := detector.Detect(normal)
    if isAnomaly {
        t.Errorf("Normal point detected as anomaly, score: %f", score)
    }
    
    // Test outlier
    outlier := []float64{10.0, 10.0}
    isAnomaly, score = detector.Detect(outlier)
    if !isAnomaly {
        t.Errorf("Outlier not detected, score: %f", score)
    }
}

func BenchmarkLOFDetect(b *testing.B) {
    trainData := generateRandomData(1000, 10)
    detector := NewLOFDetector(5)
    detector.Train(trainData)
    
    testPoint := make([]float64, 10)
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        detector.Detect(testPoint)
    }
}
```

### 1.2 One-Class SVM

#### Implementation (via CGO to libsvm)
```go
// pkg/ml/ocsvm_detector.go
package ml

/*
#cgo LDFLAGS: -lsvm
#include <svm.h>
#include <stdlib.h>
*/
import "C"
import (
    "fmt"
    "unsafe"
)

type OneClassSVMDetector struct {
    model   *C.struct_svm_model
    param   *C.struct_svm_parameter
    trained bool
    nu      float64 // Anomaly fraction
    gamma   float64 // RBF kernel parameter
}

func NewOneClassSVMDetector(nu, gamma float64) *OneClassSVMDetector {
    param := &C.struct_svm_parameter{}
    param.svm_type = C.ONE_CLASS
    param.kernel_type = C.RBF
    param.nu = C.double(nu)
    param.gamma = C.double(gamma)
    param.cache_size = 100
    param.eps = 0.001
    param.shrinking = 1
    param.probability = 0
    
    return &OneClassSVMDetector{
        param: param,
        nu:    nu,
        gamma: gamma,
    }
}

func (svm *OneClassSVMDetector) Train(data [][]float64) error {
    if len(data) == 0 {
        return fmt.Errorf("no training data")
    }
    
    // Convert Go data to libsvm format
    problem := svm.createProblem(data)
    defer C.free(unsafe.Pointer(problem))
    
    // Check parameters
    errorMsg := C.svm_check_parameter(problem, svm.param)
    if errorMsg != nil {
        return fmt.Errorf("parameter error: %s", C.GoString(errorMsg))
    }
    
    // Train model
    svm.model = C.svm_train(problem, svm.param)
    svm.trained = true
    
    return nil
}

func (svm *OneClassSVMDetector) Detect(point []float64) (bool, float64) {
    if !svm.trained {
        return false, 0.0
    }
    
    // Convert point to libsvm format
    node := svm.createNodes(point)
    defer C.free(unsafe.Pointer(node))
    
    // Predict
    decision := C.svm_predict(svm.model, node)
    
    // decision > 0 = normal, < 0 = anomaly
    isAnomaly := decision < 0
    score := float64(decision)
    
    return isAnomaly, math.Abs(score)
}

func (svm *OneClassSVMDetector) createProblem(data [][]float64) *C.struct_svm_problem {
    // Implementation details...
    return nil
}

func (svm *OneClassSVMDetector) createNodes(point []float64) *C.struct_svm_node {
    // Implementation details...
    return nil
}
```

---

## 2. Deep Learning Models

### 2.1 LSTM-based Threat Detection

#### PyTorch Implementation
```python
# services/shieldx-ml/ml-service/models/lstm_detector.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class LSTMThreatDetector(nn.Module):
    """
    LSTM-based threat detection for sequential patterns.
    
    Architecture:
    - Bidirectional LSTM layers
    - Attention mechanism
    - Fully connected classifier
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(LSTMThreatDetector, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_size * self.num_directions)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            logits: Class logits (batch_size, num_classes)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_size * num_directions)
        
        # Apply attention
        attended, attention_weights = self.attention(lstm_out)
        # attended: (batch_size, hidden_size * num_directions)
        
        # Classification
        logits = self.fc(attended)
        
        return logits, attention_weights
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over classes."""
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=1)


class AttentionLayer(nn.Module):
    """
    Attention mechanism for sequence processing.
    """
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: LSTM output (batch_size, seq_len, hidden_size)
        
        Returns:
            attended: Weighted sum (batch_size, hidden_size)
            attention_weights: Weights (batch_size, seq_len)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_out).squeeze(-1)
        # attention_scores: (batch_size, seq_len)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights: (batch_size, seq_len)
        
        # Weighted sum
        attended = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_out
        ).squeeze(1)
        # attended: (batch_size, hidden_size)
        
        return attended, attention_weights


# Training script
class ThreatDetectorTrainer:
    """
    Training pipeline for LSTM threat detector.
    """
    
    def __init__(
        self,
        model: LSTMThreatDetector,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(data)
            loss = self.criterion(logits, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                logits, _ = self.model(data)
                loss = self.criterion(logits, target)
                
                total_loss += loss.item()
                
                pred = logits.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
```

#### Integration with Go
```go
// pkg/ml/lstm_wrapper.go
package ml

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
)

type LSTMDetector struct {
    endpoint string
    client   *http.Client
}

type LSTMRequest struct {
    Sequences [][]float64 `json:"sequences"`
}

type LSTMResponse struct {
    Predictions       []string             `json:"predictions"`
    Probabilities     [][]float64          `json:"probabilities"`
    AttentionWeights  [][]float64          `json:"attention_weights"`
}

func NewLSTMDetector(endpoint string) *LSTMDetector {
    return &LSTMDetector{
        endpoint: endpoint,
        client:   &http.Client{Timeout: 5 * time.Second},
    }
}

func (lstm *LSTMDetector) Predict(sequences [][]float64) (*LSTMResponse, error) {
    req := &LSTMRequest{Sequences: sequences}
    
    body, err := json.Marshal(req)
    if err != nil {
        return nil, fmt.Errorf("marshal request: %w", err)
    }
    
    resp, err := lstm.client.Post(
        lstm.endpoint+"/predict",
        "application/json",
        bytes.NewReader(body),
    )
    if err != nil {
        return nil, fmt.Errorf("http request: %w", err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("prediction failed: %s", string(body))
    }
    
    var result LSTMResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, fmt.Errorf("decode response: %w", err)
    }
    
    return &result, nil
}
```

---

## 3. Feature Engineering

### 3.1 Network Flow Features

```python
# services/shieldx-ml/ml-service/features/network_features.py

import numpy as np
from typing import Dict, List
from scapy.all import IP, TCP, UDP, ICMP

class NetworkFeatureExtractor:
    """
    Extract features from network traffic.
    """
    
    @staticmethod
    def extract_flow_features(packets: List) -> Dict[str, float]:
        """
        Extract statistical features from packet flow.
        
        Features:
        - Basic: packet count, byte count, duration
        - Statistical: mean, std, min, max of packet sizes
        - Behavioral: inter-arrival times, flow rate
        - Protocol: TCP flags, protocol distribution
        """
        if not packets:
            return {}
        
        features = {}
        
        # Basic features
        features['packet_count'] = len(packets)
        features['total_bytes'] = sum(len(pkt) for pkt in packets)
        
        # Timing features
        timestamps = [pkt.time for pkt in packets]
        duration = timestamps[-1] - timestamps[0]
        features['duration'] = duration
        features['packets_per_second'] = len(packets) / max(duration, 0.001)
        features['bytes_per_second'] = features['total_bytes'] / max(duration, 0.001)
        
        # Packet size features
        sizes = [len(pkt) for pkt in packets]
        features['size_mean'] = np.mean(sizes)
        features['size_std'] = np.std(sizes)
        features['size_min'] = np.min(sizes)
        features['size_max'] = np.max(sizes)
        features['size_median'] = np.median(sizes)
        
        # Inter-arrival time features
        if len(timestamps) > 1:
            iat = np.diff(timestamps)
            features['iat_mean'] = np.mean(iat)
            features['iat_std'] = np.std(iat)
            features['iat_min'] = np.min(iat)
            features['iat_max'] = np.max(iat)
        
        # Protocol features
        protocols = [pkt.__class__.__name__ for pkt in packets]
        protocol_counts = {}
        for proto in set(protocols):
            protocol_counts[proto] = protocols.count(proto)
        
        features['protocol_diversity'] = len(protocol_counts)
        for proto, count in protocol_counts.items():
            features[f'protocol_{proto}_ratio'] = count / len(packets)
        
        # TCP-specific features
        tcp_packets = [pkt for pkt in packets if TCP in pkt]
        if tcp_packets:
            flags = [pkt[TCP].flags for pkt in tcp_packets]
            features['tcp_syn_count'] = sum(1 for f in flags if f & 0x02)
            features['tcp_ack_count'] = sum(1 for f in flags if f & 0x10)
            features['tcp_fin_count'] = sum(1 for f in flags if f & 0x01)
            features['tcp_rst_count'] = sum(1 for f in flags if f & 0x04)
            features['tcp_psh_count'] = sum(1 for f in flags if f & 0x08)
            
            # TCP flags entropy
            flag_counts = np.bincount(flags)
            flag_probs = flag_counts / len(flags)
            features['tcp_flags_entropy'] = -np.sum(
                flag_probs * np.log2(flag_probs + 1e-10)
            )
        
        # Payload entropy (measure of randomness)
        payloads = [bytes(pkt.payload) for pkt in packets if hasattr(pkt, 'payload')]
        if payloads:
            all_bytes = b''.join(payloads)
            byte_counts = np.bincount(list(all_bytes), minlength=256)
            byte_probs = byte_counts / len(all_bytes)
            features['payload_entropy'] = -np.sum(
                byte_probs * np.log2(byte_probs + 1e-10)
            )
        
        # Direction features (if available)
        if hasattr(packets[0], 'direction'):
            directions = [pkt.direction for pkt in packets]
            features['forward_ratio'] = directions.count('forward') / len(directions)
            features['backward_ratio'] = directions.count('backward') / len(directions)
        
        return features
    
    @staticmethod
    def extract_connection_features(connection: Dict) -> Dict[str, float]:
        """
        Extract features from connection metadata.
        """
        features = {}
        
        # Basic connection info
        features['src_port'] = connection.get('src_port', 0)
        features['dst_port'] = connection.get('dst_port', 0)
        features['protocol'] = NetworkFeatureExtractor._encode_protocol(
            connection.get('protocol', 'unknown')
        )
        
        # Port-based features
        features['is_well_known_port'] = 1 if features['dst_port'] < 1024 else 0
        features['is_registered_port'] = 1 if 1024 <= features['dst_port'] < 49152 else 0
        features['is_dynamic_port'] = 1 if features['dst_port'] >= 49152 else 0
        
        # Connection state
        features['connection_state'] = NetworkFeatureExtractor._encode_state(
            connection.get('state', 'unknown')
        )
        
        return features
    
    @staticmethod
    def _encode_protocol(protocol: str) -> int:
        """Encode protocol as integer."""
        protocols = {'tcp': 6, 'udp': 17, 'icmp': 1}
        return protocols.get(protocol.lower(), 0)
    
    @staticmethod
    def _encode_state(state: str) -> int:
        """Encode connection state as integer."""
        states = {
            'established': 1,
            'syn_sent': 2,
            'syn_recv': 3,
            'fin_wait1': 4,
            'fin_wait2': 5,
            'time_wait': 6,
            'close': 7,
            'close_wait': 8,
            'last_ack': 9,
            'listen': 10,
            'closing': 11,
        }
        return states.get(state.lower(), 0)
```

---

## 4. Model Explainability

### 4.1 SHAP Integration

```python
# services/shieldx-ml/ml-service/explainability/shap_explainer.py

import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for model interpretability.
    """
    
    def __init__(self, model, model_type: str = 'tree'):
        """
        Args:
            model: Trained model
            model_type: 'tree', 'deep', 'linear', or 'kernel'
        """
        self.model = model
        self.model_type = model_type
        self.explainer = self._create_explainer(model, model_type)
    
    def _create_explainer(self, model, model_type: str):
        """Create appropriate SHAP explainer based on model type."""
        if model_type == 'tree':
            return shap.TreeExplainer(model)
        elif model_type == 'deep':
            return shap.DeepExplainer(model)
        elif model_type == 'linear':
            return shap.LinearExplainer(model)
        elif model_type == 'kernel':
            return shap.KernelExplainer(model)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def explain_instance(
        self,
        instance: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Returns:
            Dictionary containing:
            - shap_values: SHAP values for each feature
            - base_value: Base prediction value
            - feature_importance: Feature importance ranking
            - prediction: Model prediction
        """
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance)
        
        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
        else:
            base_value = 0.0
        
        # Calculate feature importance
        if isinstance(shap_values, list):
            # Multi-class case
            abs_shap = [np.abs(sv) for sv in shap_values]
            importance = [sv.mean(axis=0) for sv in abs_shap]
        else:
            # Binary or regression case
            importance = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dict
        if feature_names:
            feature_importance = {
                name: float(imp)
                for name, imp in zip(feature_names, importance)
            }
            # Sort by importance
            feature_importance = dict(
                sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )
        else:
            feature_importance = {
                f"feature_{i}": float(imp)
                for i, imp in enumerate(importance)
            }
        
        # Get prediction
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(instance)
        else:
            prediction = None
        
        return {
            'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            'base_value': float(base_value) if np.isscalar(base_value) else base_value.tolist(),
            'feature_importance': feature_importance,
            'prediction': prediction.tolist() if prediction is not None else None,
        }
    
    def plot_waterfall(
        self,
        instance: np.ndarray,
        feature_names: List[str] = None,
        max_display: int = 20
    ):
        """
        Create waterfall plot showing how features contribute to prediction.
        """
        shap_values = self.explainer.shap_values(instance)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary
        
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=instance[0],
                feature_names=feature_names
            ),
            max_display=max_display
        )
    
    def plot_force(
        self,
        instance: np.ndarray,
        feature_names: List[str] = None
    ):
        """
        Create force plot visualizing feature contributions.
        """
        shap_values = self.explainer.shap_values(instance)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap.force_plot(
            self.explainer.expected_value,
            shap_values[0],
            instance[0],
            feature_names=feature_names
        )
    
    def global_importance(
        self,
        X: np.ndarray,
        feature_names: List[str] = None,
        top_k: int = 20
    ) -> Dict[str, float]:
        """
        Calculate global feature importance across dataset.
        """
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values[1]).mean(axis=0)
        else:
            shap_values = np.abs(shap_values).mean(axis=0)
        
        if feature_names:
            importance = {
                name: float(val)
                for name, val in zip(feature_names, shap_values)
            }
        else:
            importance = {
                f"feature_{i}": float(val)
                for i, val in enumerate(shap_values)
            }
        
        # Sort and return top-k
        sorted_importance = dict(
            sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
        )
        
        return sorted_importance
```

---

## 5. Adversarial Defense

### 5.1 FGSM Attack & Defense

```python
# services/shieldx-ml/ml-service/security/adversarial.py

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class AdversarialDefense:
    """
    Adversarial attack generation and defense.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        model.to(device)
    
    def fgsm_attack(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        epsilon: float = 0.3
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack.
        
        Args:
            data: Input data
            target: True labels
            epsilon: Perturbation magnitude
        
        Returns:
            Adversarial examples
        """
        data.requires_grad = True
        
        # Forward pass
        output = self.model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Create adversarial example
        data_grad = data.grad.data
        perturbed_data = data + epsilon * data_grad.sign()
        
        # Clip to valid range
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        return perturbed_data
    
    def pgd_attack(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        epsilon: float = 0.3,
        alpha: float = 0.01,
        num_iter: int = 40
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack.
        More powerful than FGSM.
        """
        perturbed_data = data.clone()
        
        for i in range(num_iter):
            perturbed_data.requires_grad = True
            
            output = self.model(perturbed_data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            self.model.zero_grad()
            loss.backward()
            
            # Update perturbation
            data_grad = perturbed_data.grad.data
            perturbed_data = perturbed_data + alpha * data_grad.sign()
            
            # Project back to epsilon ball
            eta = torch.clamp(perturbed_data - data, -epsilon, epsilon)
            perturbed_data = torch.clamp(data + eta, 0, 1).detach()
        
        return perturbed_data
    
    def adversarial_training(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epsilon: float = 0.3,
        attack_ratio: float = 0.5
    ) -> float:
        """
        Train model with adversarial examples.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            epsilon: Perturbation magnitude
            attack_ratio: Ratio of adversarial examples in batch
        
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            batch_size = data.size(0)
            num_adv = int(batch_size * attack_ratio)
            
            # Generate adversarial examples for part of batch
            if num_adv > 0:
                adv_data = self.fgsm_attack(
                    data[:num_adv],
                    target[:num_adv],
                    epsilon
                )
                
                # Combine clean and adversarial
                mixed_data = torch.cat([data[num_adv:], adv_data], dim=0)
                mixed_target = torch.cat([target[num_adv:], target[:num_adv]], dim=0)
            else:
                mixed_data = data
                mixed_target = target
            
            # Train on mixed batch
            optimizer.zero_grad()
            output = self.model(mixed_data)
            loss = nn.CrossEntropyLoss()(output, mixed_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate_robustness(
        self,
        test_loader: torch.utils.data.DataLoader,
        epsilon: float = 0.3
    ) -> Tuple[float, float]:
        """
        Evaluate model robustness against adversarial attacks.
        
        Returns:
            clean_accuracy, adversarial_accuracy
        """
        self.model.eval()
        
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Clean accuracy
                output = self.model(data)
                pred = output.argmax(dim=1)
                clean_correct += (pred == target).sum().item()
                
                # Adversarial accuracy
                adv_data = self.fgsm_attack(data, target, epsilon)
                adv_output = self.model(adv_data)
                adv_pred = adv_output.argmax(dim=1)
                adv_correct += (adv_pred == target).sum().item()
                
                total += target.size(0)
        
        clean_acc = clean_correct / total
        adv_acc = adv_correct / total
        
        return clean_acc, adv_acc
```

---

## 6. Performance Optimization

### 6.1 Model Quantization

```python
# services/shieldx-ml/ml-service/optimization/quantization.py

import torch
import torch.nn as nn
import torch.quantization as quantization

class ModelQuantizer:
    """
    Quantize models for faster inference and smaller size.
    """
    
    @staticmethod
    def quantize_dynamic(
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Dynamic quantization (weights only).
        Best for LSTM, Transformer models.
        """
        quantized_model = quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=dtype
        )
        return quantized_model
    
    @staticmethod
    def quantize_static(
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader
    ) -> nn.Module:
        """
        Static quantization (weights + activations).
        Requires calibration data.
        """
        # Prepare model
        model.eval()
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(model, inplace=True)
        
        # Calibrate
        with torch.no_grad():
            for data, _ in calibration_loader:
                model(data)
        
        # Convert
        quantized_model = quantization.convert(model, inplace=True)
        return quantized_model
    
    @staticmethod
    def quantize_qat(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10
    ) -> nn.Module:
        """
        Quantization-Aware Training (QAT).
        Train model with fake quantization.
        """
        # Prepare model for QAT
        model.train()
        model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
        quantization.prepare_qat(model, inplace=True)
        
        # Train with quantization
        criterion = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Convert to quantized model
        model.eval()
        quantized_model = quantization.convert(model, inplace=True)
        return quantized_model
```

---

## 📊 Performance Benchmarks

### Expected Performance After Implementation

| Metric | Before | After (Master) | Improvement |
|--------|--------|----------------|-------------|
| Accuracy | 85% | 96%+ | +11% |
| FPR | 5% | <2% | -60% |
| Latency (p99) | 50ms | <10ms | -80% |
| Model Size | 500MB | 100MB | -80% |
| Throughput | 1K/s | 10K/s | +900% |

---

**Next Steps**: Implement each component in order, test thoroughly, and integrate into production pipeline.
