"""
Enhanced ML Trainer with MLflow Integration
Supports model versioning, A/B testing, and drift detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging
from datetime import datetime
import os
from typing import Dict, Optional, Tuple, Any
import mlflow
import mlflow.sklearn
from feature_store import FeatureStore
from ml_pipeline_client import get_model_registry, get_drift_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedMLTrainer:
    """
    Production-grade ML trainer with:
    - Model versioning via registry
    - MLflow experiment tracking
    - Cross-validation
    - Hyperparameter tracking
    - Drift detection integration
    """
    
    def __init__(
        self,
        feature_store: FeatureStore,
        mlflow_uri: Optional[str] = None,
        redis_url: Optional[str] = None
    ):
        self.feature_store = feature_store
        self.model = None
        self.scaler = StandardScaler()
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize MLflow
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        exp_name = os.getenv("MLFLOW_EXPERIMENT", "threat-detection")
        mlflow.set_experiment(exp_name)
        
        # Initialize model registry client
        self.registry = get_model_registry(redis_url)
        self.drift_detector = get_drift_detector(redis_url)
        
        # Model storage
        self.models_dir = os.getenv("ML_MODEL_STORAGE_DIR", "/var/ml/models")
        os.makedirs(self.models_dir, exist_ok=True)
    
    def train_model(
        self,
        limit: int = 50000,
        algorithm: str = "random_forest",
        hyperparameters: Optional[Dict[str, Any]] = None,
        register: bool = True
    ) -> Tuple[str, Dict[str, float]]:
        """
        Train ML model with full production pipeline
        
        Args:
            limit: Maximum number of training samples
            algorithm: Algorithm to use (random_forest, gradient_boosting)
            hyperparameters: Model hyperparameters
            register: Whether to register model in registry
        
        Returns:
            (model_id, metrics): Model identifier and performance metrics
        """
        logger.info(f"Starting model training: {algorithm}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"train_{self.model_version}"):
            # Log parameters
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("max_samples", limit)
            mlflow.log_param("version", self.model_version)
            
            # Get training data
            X, y, feature_names = self.feature_store.get_training_data(limit)
            
            if len(X) == 0:
                logger.error("No training data available")
                raise ValueError("No training data available")
            
            logger.info(f"Training data: {len(X)} samples, {len(feature_names)} features")
            mlflow.log_param("sample_count", len(X))
            mlflow.log_param("feature_count", len(feature_names))
            
            # Prepare data
            X_scaled = self.scaler.fit_transform(X)
            
            # Convert to binary classification
            y_binary = np.where(y > 0.7, 1, np.where(y < 0.3, 0, -1))
            mask = y_binary != -1
            X_filtered = X_scaled[mask]
            y_filtered = y_binary[mask]
            
            if len(X_filtered) < 100:
                raise ValueError("Insufficient labeled data for training")
            
            logger.info(f"Filtered data: {len(X_filtered)} samples")
            class_dist = np.bincount(y_filtered)
            logger.info(f"Class distribution: {class_dist}")
            mlflow.log_param("class_0_count", int(class_dist[0]))
            mlflow.log_param("class_1_count", int(class_dist[1]))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered,
                test_size=0.2,
                random_state=42,
                stratify=y_filtered
            )
            
            # Initialize model
            self.model = self._create_model(algorithm, hyperparameters)
            
            # Log hyperparameters
            if hyperparameters:
                for key, value in hyperparameters.items():
                    mlflow.log_param(key, value)
            
            # Train model
            logger.info("Training model...")
            train_start = datetime.now()
            self.model.fit(X_train, y_train)
            train_duration = (datetime.now() - train_start).total_seconds()
            
            mlflow.log_metric("training_time_sec", train_duration)
            
            # Evaluate model
            metrics = self._evaluate_model(X_train, y_train, X_test, y_test)
            
            # Log metrics to MLflow
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_filtered, y_filtered, cv=5, scoring='accuracy'
            )
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
            mlflow.log_metric("cv_accuracy_std", cv_scores.std())
            
            logger.info(f"Model metrics: {json.dumps(metrics, indent=2)}")
            
            # Save model
            model_filename = f"model_{self.model_version}.joblib"
            model_path = os.path.join(self.models_dir, model_filename)
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': feature_names,
                'metadata': {
                    'version': self.model_version,
                    'algorithm': algorithm,
                    'trained_at': datetime.utcnow().isoformat(),
                    'metrics': metrics
                }
            }, model_path)
            
            logger.info(f"Model saved: {model_path}")
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name=f"threat-detector-{algorithm}"
            )
            
            # Register in model registry
            model_id = None
            if register:
                try:
                    model_id = self.registry.register_model(
                        name="threat-detector",
                        version=self.model_version,
                        model_path=model_path,
                        algorithm=algorithm,
                        framework="scikit-learn",
                        metrics=metrics,
                        parameters=hyperparameters or {},
                        description=f"Threat detection model trained on {len(X_filtered)} samples",
                        tags=[algorithm, "production-ready"]
                    )
                    logger.info(f"Model registered: {model_id}")
                    mlflow.log_param("model_id", model_id)
                except Exception as e:
                    logger.error(f"Failed to register model: {e}")
            
            # Set baseline for drift detection
            try:
                baseline_features = {
                    name: X_filtered[:, i].tolist()
                    for i, name in enumerate(feature_names[:10])  # First 10 features
                }
                # Note: This would be called via Go API in production
                logger.info("Baseline features prepared for drift detection")
            except Exception as e:
                logger.error(f"Failed to set drift baseline: {e}")
            
            return model_id or model_filename, metrics
    
    def _create_model(
        self,
        algorithm: str,
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        """Create model instance based on algorithm"""
        params = hyperparameters or {}
        
        if algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 5),
                min_samples_leaf=params.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
        elif algorithm == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 5),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def _evaluate_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        # Training metrics
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Test metrics
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
            'precision': float(precision_score(y_test, y_test_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_test_pred, average='weighted')),
            'f1_score': float(f1_score(y_test, y_test_pred, average='weighted')),
            'roc_auc': float(roc_auc_score(y_test, y_test_proba))
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            top_features = np.argsort(self.model.feature_importances_)[-10:]
            logger.info(f"Top 10 feature indices: {top_features.tolist()}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Classification report
        report = classification_report(y_test, y_test_pred)
        logger.info(f"Classification Report:\n{report}")
        
        return metrics
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model from disk"""
        try:
            loaded = joblib.load(model_path)
            self.model = loaded['model']
            self.scaler = loaded['scaler']
            logger.info(f"Model loaded: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with the trained model"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        return predictions, probabilities


if __name__ == "__main__":
    # Example training run
    from feature_store import FeatureStore
    
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ml")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    mlflow_uri = os.getenv("MLFLOW_URI", "http://localhost:5000")
    
    feature_store = FeatureStore(db_url, redis_url)
    trainer = EnhancedMLTrainer(feature_store, mlflow_uri, redis_url)
    
    # Train with random forest
    model_id, metrics = trainer.train_model(
        limit=10000,
        algorithm="random_forest",
        hyperparameters={
            'n_estimators': 150,
            'max_depth': 12,
            'min_samples_split': 3
        }
    )
    
    print(f"Training complete!")
    print(f"Model ID: {model_id}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
