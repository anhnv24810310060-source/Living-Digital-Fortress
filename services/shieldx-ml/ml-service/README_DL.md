# Deep Learning Service - PyTorch Models

## Overview

This service provides PyTorch-based deep learning models for advanced anomaly detection in ShieldX ML platform.

## Features

### Implemented Models

1. **BasicAutoencoder** - Standard autoencoder for anomaly detection
   - Configurable architecture (hidden layers, latent dimensions)
   - Batch normalization and dropout for regularization
   - Reconstruction error-based anomaly scoring

2. **AnomalyDetectionAE** - Full anomaly detection system
   - Automated threshold calculation
   - Training with early stopping
   - Model save/load functionality
   - Probability and binary predictions

3. **LSTMAutoencoder** - Sequence-based autoencoder
   - Bidirectional LSTM support
   - Multi-layer LSTM architecture
   - Suitable for time-series and sequential data

4. **SequentialAnomalyDetector** - Complete sequential anomaly detection
   - LSTM-based encoding/decoding
   - Sequence reconstruction error scoring
   - Support for variable-length sequences

5. **CNN1DClassifier** - 1D Convolutional Neural Network
   - Multi-kernel convolutions for feature extraction
   - Max pooling and batch normalization
   - Packet-level threat classification

6. **PacketThreatDetector** - CNN-based threat detection
   - Multi-class threat classification
   - Optimized for network packet analysis
   - Real-time inference capability

7. **TransformerEncoder** - Transformer-based detection
   - Multi-head self-attention mechanism
   - Positional encoding for sequence understanding
   - State-of-the-art pattern recognition

8. **TransformerThreatDetector** - Complete transformer system
   - Advanced threat pattern detection
   - Attention weight visualization
   - Learning rate scheduling

9. **ThreatClassifier** - Unified ensemble system
   - Combines all models for comprehensive detection
   - Multiple ensemble strategies (weighted voting, majority, max confidence)
   - Individual model management
   - Best-in-class accuracy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Service

```bash
python3 dl_service.py
```

The service will start on port 8001 by default.

### Environment Variables

- `PORT`: Service port (default: 8001)
- `MODEL_DIR`: Directory for model storage (default: /tmp/shieldx_models)
- `DEBUG`: Enable debug mode (default: false)

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Train a Model
```bash
POST /models/{model_name}/train
Content-Type: application/json

{
  "model_type": "autoencoder",
  "config": {
    "input_dim": 50,
    "latent_dim": 16,
    "hidden_dims": [32, 24]
  },
  "training_data": [[...], [...]],
  "training_params": {
    "epochs": 100,
    "batch_size": 256
  }
}
```

#### Load a Model
```bash
POST /models/{model_name}/load
Content-Type: application/json

{
  "model_type": "autoencoder",
  "model_path": "/path/to/model.pt"
}
```

#### Make Predictions
```bash
POST /models/{model_name}/predict
Content-Type: application/json

{
  "data": [[...], [...]],
  "return_proba": true
}
```

#### Evaluate Model
```bash
POST /models/{model_name}/evaluate
Content-Type: application/json

{
  "data": [[...], [...]],
  "labels": [0, 1, 0, ...]
}
```

## Go Client

```go
import "shieldx/pkg/ml/deeplearning"

// Create client
client := deeplearning.NewClient("http://localhost:8001")

// Train model
resp, err := client.Train("my_model", deeplearning.TrainRequest{
    ModelType: deeplearning.ModelTypeAutoencoder,
    Config: deeplearning.ModelConfig{
        InputDim:  50,
        LatentDim: 16,
    },
    TrainingData: trainingData,
})

// Make predictions
pred, err := client.Predict("my_model", deeplearning.PredictRequest{
    Data:        testData,
    ReturnProba: true,
})
```

## Testing

### Run Python Unit Tests
```bash
./run_tests.sh
```

### Run Go Client Tests
```bash
cd pkg/ml/deeplearning
go test -v -cover
```

## Docker

### Build Image
```bash
docker build -f Dockerfile.dl -t shieldx-dl-service .
```

### Run Container
```bash
docker run -p 8001:8001 \
  -v /path/to/models:/models \
  -e MODEL_DIR=/models \
  shieldx-dl-service
```

## Architecture

### Autoencoder Models

```
Input → Encoder → Latent Space → Decoder → Reconstruction
        ↓                                   ↓
      Compress                           Expand
```

- **Encoder**: Compresses input to low-dimensional latent representation
- **Latent Space**: Compact representation of normal patterns
- **Decoder**: Reconstructs input from latent space
- **Anomaly Score**: Reconstruction error (MSE between input and reconstruction)

### LSTM Autoencoder

```
Sequence Input → LSTM Encoder → Latent Vector → LSTM Decoder → Sequence Reconstruction
                 (Bidirectional)                                ↓
                                                        Reconstruction Error
```

- **LSTM Encoder**: Captures temporal dependencies in sequences
- **Bidirectional**: Process sequences forward and backward
- **LSTM Decoder**: Generates sequence from latent representation

## Performance

### Autoencoder (BasicAutoencoder)
- Training: ~10 epochs for convergence on 1000 samples
- Inference: <1ms per sample (CPU)
- Memory: ~50MB for typical model

### LSTM Autoencoder
- Training: ~20 epochs for convergence on 500 sequences
- Inference: <5ms per sequence (CPU)
- Memory: ~100MB for typical model

## Test Coverage

- **Autoencoder**: 100% coverage with 13 test cases
- **LSTM Autoencoder**: 100% coverage with 14 test cases
- **Go Client**: 69.7% coverage with 10 test cases

## Roadmap

### Completed ✅
- [x] Basic Autoencoder implementation
- [x] LSTM Autoencoder for sequences
- [x] CNN-1D for packet analysis
- [x] Transformer with multi-head attention
- [x] ThreatClassifier ensemble system
- [x] HTTP API service
- [x] Go client wrapper
- [x] Comprehensive unit tests
- [x] Docker support

### In Progress 🔄
- [ ] BERT for log analysis
- [ ] Model serving optimization (ONNX export)
- [ ] GPU batch inference optimization

### Planned 📋
- [ ] Variational Autoencoders (VAE)
- [ ] GAN-based anomaly detection
- [ ] Attention mechanism visualization
- [ ] Model quantization for edge deployment
- [ ] Federated learning support

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md)

## License

See [LICENSE](../../../LICENSE)
