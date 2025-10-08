 
-----

# 🛡️ ShieldX ML Service

[](https://www.python.org/)
[](https://golang.org)
[](https://opensource.org/licenses/Apache-2.0)
[](https://www.docker.com/)

**ShieldX ML Service** is the brain of the system, providing the capability to detect and predict security threats using advanced Machine Learning models.

## 📋 Table of Contents

  - [🎯 Overview](https://www.google.com/search?q=%23-overview)
      - [Key Features](https://www.google.com/search?q=%23key-features)
      - [Technology Stack](https://www.google.com/search?q=%23technology-stack)
  - [🏗️ System Architecture](https://www.google.com/search?q=%23%EF%B8%8F-system-architecture)
  - [🚀 Quick Start](https://www.google.com/search?q=%23-quick-start)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation & Startup](https://www.google.com/search?q=%23installation--startup)
  - [📡 API Reference](https://www.google.com/search?q=%23-api-reference)
      - [Prediction Endpoints](https://www.google.com/search?q=%23prediction-endpoints)
      - [Model Management](https://www.google.com/search?q=%23model-management)
      - [Training Endpoints](https://www.google.com/search?q=%23training-endpoints)
  - [🤖 ML Models](https://www.google.com/search?q=%23-ml-models)
  - [💻 Development Guide](https://www.google.com/search?q=%23-development-guide)
      - [Project Structure](https://www.google.com/search?q=%23project-structure)
      - [Creating a Custom Model](https://www.google.com/search?q=%23creating-a-custom-model)
  - [🧪 Testing](https://www.google.com/search?q=%23-testing)
  - [📊 Monitoring](https://www.google.com/search?q=%23-monitoring)
  - [🔧 Troubleshooting](https://www.google.com/search?q=%23-troubleshooting)
  - [📚 References](https://www.google.com/search?q=%23-references)
  - [📄 License](https://www.google.com/search?q=%23-license)

-----

## 🎯 Overview

### Key Features

  - **Anomaly Detection**: Identifies user or system behaviors that deviate from established norms.
  - **Threat Classification**: Automatically classifies attack types (e.g., SQL Injection, XSS, Brute-force).
  - **Behavioral Analysis**: Builds user behavior profiles (UEBA - User and Entity Behavior Analytics) to detect insider threats.
  - **Predictive Analytics**: Forecasts potential attacks based on initial indicators.
  - **Model Training and Tuning**: Provides APIs to retrain models with new data.
  - **Feature Engineering**: Automatically extracts meaningful features from raw data (logs, network packets).

### Technology Stack

  - **Language**: Python 3.11+, Go 1.25+ (API Wrapper)
  - **ML Frameworks**: TensorFlow, PyTorch, Scikit-learn, XGBoost
  - **API Frameworks**: FastAPI (Python), Gin (Go Wrapper)
  - **Model Storage**: MinIO / AWS S3
  - **Feature Store**: Redis, PostgreSQL
  - **Queue**: RabbitMQ / Redis (for asynchronous training and prediction tasks)

-----

## 🏗️ System Architecture

```plaintext
┌─────────────────────────────────────────────────────┐
│ ShieldX ML Service (Port 5003)                      │
├─────────────────────────────────────────────────────┤
│ API Layer (FastAPI)                                 │
│ - Prediction endpoints (real-time & batch)          │
│ - Training endpoints                                │
│ - Model management (deploy, rollback)               │
├─────────────────────────────────────────────────────┤
│ ML Engine                                           │
│ - Model Inference                                   │
│ - Feature Extraction                                │
│ - Preprocessing Pipeline                            │
├─────────────────────────────────────────────────────┤
│ Model Registry                                      │
│ - Model versioning                                  │
│ - A/B testing support                               │
│ - Rollback to previous model versions               │
├─────────────────────────────────────────────────────┤
│ Data Layer                                          │
│ - Feature Store (Redis)                             │
│ - Training Data (PostgreSQL)                        │
│ - Model Storage (MinIO/S3)                          │
└─────────────────────────────────────────────────────┘
       │                      │                      │
┌──────▼──────┐        ┌──────▼──────┐        ┌──────▼──────┐
│  PostgreSQL │        │    Redis    │        │    MinIO    │
└─────────────┘        └─────────────┘        └─────────────┘
```

-----

## 🚀 Quick Start

### Prerequisites

  - Python `3.11` or newer
  - Go `1.25` or newer (optional)
  - PostgreSQL `15`, Redis `7`, MinIO
  - (Optional) CUDA & cuDNN for GPU acceleration

### Installation & Startup

```bash
# 1. Clone the repository
git clone https://github.com/shieldx-bot/shieldx.git
cd shieldx/services/shieldx-ml

# 2. Create and activate a Python virtual environment
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
# venv\Scripts\activate

# 3. Install necessary Python libraries
pip install -r requirements.txt
pip install tensorflow torch scikit-learn xgboost

# 4. Start background services (PostgreSQL, Redis, MinIO) using Docker
# (Commands are similar to other services)
# docker run ...

# 5. Configure environment variables
export ML_PORT=5003
export ML_DB_HOST=localhost
export ML_REDIS_HOST=localhost
export ML_S3_ENDPOINT=http://localhost:9000
export ML_S3_ACCESS_KEY=minioadmin
export ML_S3_SECRET_KEY=minioadmin

# 6. Run database migrations (using Alembic)
alembic upgrade head

# 7. Start the API service
uvicorn src.main:app --host 0.0.0.0 --port 5003 --reload

# 8. Verify the service status
# You should receive {"status": "ok"} if successful
curl http://localhost:5003/health
```

-----

## 📡 API Reference

**Base URL**: `http://localhost:5003/api/v1`

### Prediction Endpoints

#### 1\. Predict Threat

`POST /api/v1/predict/threat`

\<details\>
\<summary\>View Request Example\</summary\>

```json
{
  "request_data": {
    "method": "POST",
    "path": "/api/users",
    "headers": {
      "User-Agent": "sqlmap/1.5.5",
      "Content-Type": "application/json"
    },
    "body": "{\"username\":\"' OR 1=1 --\"}",
    "source_ip": "203.0.113.45"
  }
}
```

\</details\>
\<details\>
\<summary\>View Response Example (200 OK)\</summary\>

```json
{
  "prediction_id": "pred-456",
  "threat_detected": true,
  "threat_type": "sql_injection",
  "confidence": 0.985,
  "risk_score": 92,
  "recommended_action": "block_ip",
  "features_used": [
    "sql_keywords_count",
    "special_chars_ratio",
    "request_body_entropy"
  ],
  "model_version": "sql_detector_v2.1.0",
  "inference_time_ms": 12
}
```

\</details\>

#### 2\. Detect Anomaly

`POST /api/v1/predict/anomaly`

\<details\>
\<summary\>View Request Example\</summary\>

```json
{
  "user_id": "user-prod-123",
  "behavioral_data": {
    "login_timestamp": "2025-10-08T03:15:00Z",
    "login_geolocation": "Unknown",
    "device_fingerprint": "new_device_hash_xyz",
    "actions": ["view_dashboard", "export_user_data", "delete_records"]
  }
}
```

\</details\>
\<details\>
\<summary\>View Response Example\</summary\>

```json
{
  "is_anomaly": true,
  "anomaly_score": 0.91,
  "anomaly_reasons": [
    "Unusual login time (3 AM)",
    "Unrecognized device",
    "Execution of high-risk action (delete_records)"
  ],
  "recommended_action": "force_logout_and_require_mfa"
}
```

\</details\>

### Model Management

#### 1\. List Models

`GET /api/v1/models?status=active`

\<details\>
\<summary\>View Response Example\</summary\>

```json
{
  "models": [
    {
      "model_id": "sql_detector_v2.1.0",
      "name": "sql_injection_detector",
      "version": "v2.1.0",
      "status": "active",
      "metrics": { "accuracy": 0.98, "f1_score": 0.97 },
      "deployed_at": "2025-10-08T10:00:00Z"
    }
  ]
}
```

\</details\>

#### 2\. Deploy a Model

`POST /api/v1/models/{model_id}/deploy`

#### 3\. Rollback to a Previous Version

`POST /api/v1/models/{model_id}/rollback`

### Training Endpoints

#### 1\. Start a Training Job

`POST /api/v1/training/start`

\<details\>
\<summary\>View Request Example\</summary\>

```json
{
  "model_type": "sql_injection_detector",
  "training_data_source": "s3://training-data/sql-injection-2025-q3.csv",
  "hyperparameters": {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 64
  }
}
```

\</details\>
\<details\>
\<summary\>View Response Example\</summary\>

```json
{
  "training_job_id": "job-789",
  "status": "queued",
  "submitted_at": "2025-10-08T14:30:00Z"
}
```

\</details\>

#### 2\. Get Training Job Status

`GET /api/v1/training/{job_id}`

\<details\>
\<summary\>View Response Example\</summary\>

```json
{
  "job_id": "job-789",
  "status": "completed",
  "progress": 1.0,
  "final_metrics": {
    "accuracy": 0.985,
    "validation_accuracy": 0.981
  },
  "new_model_id": "sql_detector_v2.2.0"
}
```

\</details\>

-----

## 🤖 ML Models

#### 1\. SQL Injection Detector

  - **Type**: Binary Classification
  - **Algorithm**: LSTM (Long Short-Term Memory) combined with Random Forest
  - **Features**: SQL keyword density, special character ratio, string entropy, tokenization patterns.

#### 2\. User Behavior Anomaly Detector

  - **Type**: Unsupervised Learning
  - **Algorithm**: Isolation Forest and Autoencoder
  - **Features**: Login time patterns, geolocation, device fingerprints, sequence of actions.

#### 3\. XSS Detector

  - **Type**: Binary Classification
  - **Algorithm**: CNN (Convolutional Neural Network) with an Attention mechanism
  - **Features**: HTML/JS tag patterns, script injection patterns, detection of event handlers.

-----

## 💻 Development Guide

### Project Structure

```
shieldx-ml/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── api/                    # Route and schema definitions
│   ├── ml/
│   │   ├── models/             # Logic for each model type
│   │   ├── feature_engineering.py
│   │   └── model_registry.py   # Model lifecycle management
│   ├── training/
│   │   ├── trainer.py          # Training logic
│   │   └── evaluator.py        # Evaluation logic
│   └── core/                   # Configuration, core components
├── models/                     # Directory for storing trained model files
├── notebooks/                  # Jupyter notebooks for experimentation
├── tests/                      # Test cases
└── requirements.txt
```

### Creating a Custom Model

```python
# src/ml/models/custom_detector.py
from src.ml.models.base import BaseModel

class CustomThreatDetector(BaseModel):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = self.load_model(model_path)
    
    def extract_features(self, data: dict) -> list:
        # Custom feature extraction logic
        features = []
        # ...
        return features

    def predict(self, data: dict) -> dict:
        features = self.extract_features(data)
        prediction = self.model.predict(features)
        confidence = float(prediction[0])
        return {
            "threat_detected": confidence > 0.5,
            "confidence": confidence
        }
```

-----

## 🧪 Testing

The system uses `pytest` for testing and `locust` for load testing.

```bash
# Run all unit tests
pytest tests/ -v

# Run tests for a specific model
pytest tests/ml/models/test_sql_injection.py

# Run tests with a code coverage report
pytest --cov=src tests/

# Run load testing with Locust
locust -f tests/load_testing/locustfile.py --host=http://localhost:5003
```

-----

## 📊 Monitoring

### Prometheus Metrics

```
shieldx_ml_predictions_total{model_name,threat_detected}  # Total number of predictions
shieldx_ml_inference_duration_seconds{model_name}         # Inference latency
shieldx_ml_model_accuracy{model_name,model_version}       # Accuracy of the active model
shieldx_ml_training_jobs_total{status}                    # Status of training jobs
```

-----

## 🔧 Troubleshooting

#### High Inference Latency

  - **Cause**: The model is too complex, the input data is large, or there is no GPU acceleration.
  - **Solution**:
    1.  Check logs to see which step is most time-consuming (preprocessing, inference).
    2.  Optimize the model (quantization, pruning).
    3.  Use a GPU if available.
    4.  Consider using a simpler model for real-time tasks.

#### Poor Model Performance on Real Data

  - **Cause**: Data drift (real-world data differs from training data), overfitting.
  - **Solution**:
    1.  Schedule periodic retraining of the model with new data.
    2.  Monitor model performance metrics (accuracy, F1-score) over time.
    3.  Use data augmentation techniques to enrich the training data.

-----

## 📚 References

  - [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
  - [Introduction to Machine Learning for Security](https://www.google.com/search?q=https://www.oreilly.com/library/view/machine-learning-and/9781492032284/)
  - [MLOps: Concepts, Tools, and Platforms](https://www.google.com/search?q=https://www.manning.com/books/manning-liveproject-building-a-reproducible-model-workflow)

-----

## 📄 License

This project is licensed under the [Apache License 2.0](https://github.com/shieldx-bot/shieldx/blob/main/LICENSE).