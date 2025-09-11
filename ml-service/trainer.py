import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging
from datetime import datetime
import os
from feature_store import FeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLTrainer:
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.model = None
        self.scaler = StandardScaler()
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def train_model(self, limit: int = 50000):
        """Train ML model on plugin output features"""
        logger.info(f"Starting model training with up to {limit} samples")
        
        # Get training data
        X, y, feature_names = self.feature_store.get_training_data(limit)
        
        if len(X) == 0:
            logger.error("No training data available")
            return False
            
        logger.info(f"Training data loaded: {len(X)} samples, {len(feature_names)} features")
        
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert continuous confidence scores to binary classification
        # High confidence (>0.7) = malware, low confidence (<0.3) = benign
        y_binary = np.where(y > 0.7, 1, np.where(y < 0.3, 0, -1))
        
        # Remove uncertain samples (-1)
        mask = y_binary != -1
        X_filtered = X_scaled[mask]
        y_filtered = y_binary[mask]
        
        if len(X_filtered) < 100:
            logger.error("Insufficient labeled data for training")
            return False
            
        logger.info(f"Filtered training data: {len(X_filtered)} samples")
        logger.info(f"Class distribution: {np.bincount(y_filtered)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(feature_names),
            'model_version': self.model_version
        }
        
        logger.info(f"Model training completed. Metrics: {metrics}")
        
        # Save model and metadata
        self.save_model(feature_names, metrics)
        
        return True
    
    def save_model(self, feature_names, metrics):
        """Save trained model and metadata"""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = f"{model_dir}/threat_classifier_{self.model_version}.joblib"
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = f"{model_dir}/scaler_{self.model_version}.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_version': self.model_version,
            'feature_names': feature_names,
            'metrics': metrics,
            'model_type': 'RandomForestClassifier',
            'created_at': datetime.now().isoformat(),
            'model_path': model_path,
            'scaler_path': scaler_path
        }
        
        metadata_path = f"{model_dir}/metadata_{self.model_version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Update latest model symlink
        latest_model_path = f"{model_dir}/latest_model.joblib"
        latest_scaler_path = f"{model_dir}/latest_scaler.joblib"
        latest_metadata_path = f"{model_dir}/latest_metadata.json"
        
        if os.path.exists(latest_model_path):
            os.remove(latest_model_path)
        if os.path.exists(latest_scaler_path):
            os.remove(latest_scaler_path)
        if os.path.exists(latest_metadata_path):
            os.remove(latest_metadata_path)
            
        os.symlink(os.path.basename(model_path), latest_model_path)
        os.symlink(os.path.basename(scaler_path), latest_scaler_path)
        os.symlink(os.path.basename(metadata_path), latest_metadata_path)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Metadata saved: {metadata_path}")
    
    def predict(self, features):
        """Make prediction on new features"""
        if self.model is None:
            raise ValueError("Model not trained")
            
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'probability_benign': float(probability[0]),
            'probability_malware': float(probability[1]),
            'confidence': float(max(probability))
        }

def main():
    """Main training loop"""
    DB_URL = os.getenv("DB_URL", "postgresql://ml_user:ml_pass2024@localhost:5432/ml_features")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Initialize feature store
    feature_store = FeatureStore(DB_URL, REDIS_URL)
    
    # Initialize trainer
    trainer = MLTrainer(feature_store)
    
    # Train model
    success = trainer.train_model()
    
    if success:
        logger.info("Model training completed successfully")
    else:
        logger.error("Model training failed")
        exit(1)

if __name__ == "__main__":
    main()