"""
PyTorch Deep Learning Service
Provides HTTP API for deep learning model inference
"""

from flask import Flask, request, jsonify
import numpy as np
import logging
import os
from typing import Dict, Any

from models.autoencoder import AnomalyDetectionAE
from models.lstm_autoencoder import SequentialAnomalyDetector
from models.cnn1d import PacketThreatDetector
from models.transformer import TransformerThreatDetector
from models.threat_classifier import ThreatClassifier
from explainability.shap_explainer import SHAPExplainer, ModelAgnosticExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.counterfactual import CounterfactualExplainer, ActionableInsights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model registry
models: Dict[str, Any] = {}

# Explainer registry
explainers: Dict[str, Dict[str, Any]] = {}

# Model storage path
MODEL_DIR = os.getenv('MODEL_DIR', '/tmp/shieldx_models')
os.makedirs(MODEL_DIR, exist_ok=True)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'loaded_models': list(models.keys())
    })


@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'models': [
            {
                'name': name,
                'type': type(model).__name__,
                'status': 'loaded'
            }
            for name, model in models.items()
        ]
    })


@app.route('/models/<model_name>/train', methods=['POST'])
def train_model(model_name: str):
    """
    Train a new model
    
    Request body:
    {
        "model_type": "autoencoder" | "lstm_autoencoder" | "cnn1d" | "transformer" | "threat_classifier",
        "config": {
            "input_dim": 50,
            "latent_dim": 32,
            "hidden_dim": 64,  # for LSTM
            "num_classes": 6,  # for classifiers
            ...
        },
        "training_data": [[...], [...], ...],  # numpy array as list
        "training_labels": [0, 1, ...],  # for supervised models
        "training_params": {
            "epochs": 100,
            "batch_size": 256,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        
        model_type = data.get('model_type')
        config = data.get('config', {})
        training_data = np.array(data.get('training_data'))
        training_labels = data.get('training_labels')
        if training_labels is not None:
            training_labels = np.array(training_labels)
        training_params = data.get('training_params', {})
        
        # Create model based on type
        if model_type == 'autoencoder':
            model = AnomalyDetectionAE(
                input_dim=config.get('input_dim'),
                latent_dim=config.get('latent_dim', 32),
                hidden_dims=config.get('hidden_dims'),
                learning_rate=config.get('learning_rate', 0.001)
            )
        elif model_type == 'lstm_autoencoder':
            model = SequentialAnomalyDetector(
                input_dim=config.get('input_dim'),
                hidden_dim=config.get('hidden_dim', 64),
                latent_dim=config.get('latent_dim', 32),
                num_layers=config.get('num_layers', 2),
                bidirectional=config.get('bidirectional', False),
                learning_rate=config.get('learning_rate', 0.001)
            )
        elif model_type == 'cnn1d':
            model = PacketThreatDetector(
                input_dim=config.get('input_dim'),
                num_classes=config.get('num_classes', 6),
                num_filters=config.get('num_filters'),
                kernel_sizes=config.get('kernel_sizes'),
                learning_rate=config.get('learning_rate', 0.001)
            )
        elif model_type == 'transformer':
            model = TransformerThreatDetector(
                input_dim=config.get('input_dim'),
                num_classes=config.get('num_classes', 6),
                d_model=config.get('d_model', 256),
                nhead=config.get('nhead', 8),
                num_layers=config.get('num_layers', 6),
                learning_rate=config.get('learning_rate', 0.0001)
            )
        elif model_type == 'threat_classifier':
            model = ThreatClassifier(
                input_dim=config.get('input_dim'),
                num_classes=config.get('num_classes', 6),
                enable_autoencoder=config.get('enable_autoencoder', True),
                enable_lstm=config.get('enable_lstm', True),
                enable_cnn=config.get('enable_cnn', True),
                enable_transformer=config.get('enable_transformer', True)
            )
        else:
            return jsonify({'error': f'Unknown model type: {model_type}'}), 400
        
        # Train model
        logger.info(f"Training {model_type} model: {model_name}")
        if model_type in ['cnn1d', 'transformer', 'threat_classifier']:
            # Supervised models need labels
            if training_labels is None:
                return jsonify({'error': f'{model_type} requires training_labels'}), 400
            model.fit(training_data, training_labels, **training_params)
        else:
            # Unsupervised models
            model.fit(training_data, **training_params)
            )
        else:
            return jsonify({'error': f'Unknown model type: {model_type}'}), 400
        
        # Train model
        logger.info(f"Training {model_type} model: {model_name}")
        model.fit(
            training_data,
            epochs=training_params.get('epochs', 100),
            batch_size=training_params.get('batch_size', 256 if model_type == 'autoencoder' else 64),
            validation_split=training_params.get('validation_split', 0.2),
            early_stopping_patience=training_params.get('early_stopping_patience', 10)
        )
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
        model.save(model_path)
        
        # Register model
        models[model_name] = model
        
        return jsonify({
            'status': 'success',
            'model_name': model_name,
            'model_type': model_type,
            'model_path': model_path,
            'threshold': float(model.threshold) if model.threshold else None,
            'train_losses': [float(x) for x in model.train_losses[-10:]],  # last 10
            'val_losses': [float(x) for x in model.val_losses[-10:]]
        })
        
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/models/<model_name>/load', methods=['POST'])
def load_model(model_name: str):
    """
    Load a trained model
    
    Request body:
    {
        "model_type": "autoencoder" | "lstm_autoencoder",
        "model_path": "/path/to/model.pt"
    }
    """
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        model_path = data.get('model_path', os.path.join(MODEL_DIR, f"{model_name}.pt"))
        
        # Create model instance
        if model_type == 'autoencoder':
            model = AnomalyDetectionAE(input_dim=1)  # Will be overwritten by load
        elif model_type == 'lstm_autoencoder':
            model = SequentialAnomalyDetector(input_dim=1)
        else:
            return jsonify({'error': f'Unknown model type: {model_type}'}), 400
        
        # Load model
        model.load(model_path)
        models[model_name] = model
        
        logger.info(f"Loaded model {model_name} from {model_path}")
        
        return jsonify({
            'status': 'success',
            'model_name': model_name,
            'model_type': model_type,
            'threshold': float(model.threshold) if model.threshold else None
        })
        
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/models/<model_name>/predict', methods=['POST'])
def predict(model_name: str):
    """
    Make predictions with a loaded model
    
    Request body:
    {
        "data": [[...], [...], ...],  # numpy array as list
        "return_proba": true  # optional, return probabilities instead of binary
    }
    """
    try:
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not loaded'}), 404
        
        data = request.get_json()
        X = np.array(data.get('data'))
        return_proba = data.get('return_proba', False)
        
        model = models[model_name]
        
        # Make predictions
        if return_proba:
            predictions = model.predict_proba(X)
        else:
            predictions = model.predict(X)
        
        # Calculate reconstruction errors for analysis
        errors = model.reconstruction_error(X)
        
        return jsonify({
            'predictions': predictions.tolist(),
            'reconstruction_errors': errors.tolist(),
            'threshold': float(model.threshold) if model.threshold else None,
            'num_anomalies': int(np.sum(predictions > 0.5)) if return_proba else int(np.sum(predictions))
        })
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/models/<model_name>/evaluate', methods=['POST'])
def evaluate(model_name: str):
    """
    Evaluate model performance
    
    Request body:
    {
        "data": [[...], [...], ...],
        "labels": [0, 1, 0, ...]  # true labels
    }
    """
    try:
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not loaded'}), 404
        
        data = request.get_json()
        X = np.array(data.get('data'))
        y_true = np.array(data.get('labels'))
        
        model = models[model_name]
        
        # Make predictions
        y_pred = model.predict(X)
        y_scores = model.predict_proba(X)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        }
        
        # ROC AUC if we have both classes
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores))
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics['confusion_matrix'] = {
            'true_negatives': int(cm[0, 0]) if cm.shape[0] > 0 else 0,
            'false_positives': int(cm[0, 1]) if cm.shape[0] > 1 else 0,
            'false_negatives': int(cm[1, 0]) if cm.shape[0] > 1 else 0,
            'true_positives': int(cm[1, 1]) if cm.shape[0] > 1 else 0
        }
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/models/<model_name>/unload', methods=['POST'])
def unload_model(model_name: str):
    """Unload a model from memory"""
    if model_name in models:
        del models[model_name]
        logger.info(f"Unloaded model: {model_name}")
        return jsonify({'status': 'success', 'message': f'Model {model_name} unloaded'})
    else:
        return jsonify({'error': f'Model {model_name} not found'}), 404


@app.route('/models/<model_name>/explain', methods=['POST'])
def explain_prediction(model_name: str):
    """Generate explanation for model prediction"""
    try:
        data = request.json
        explainer_type = data.get('explainer_type', 'shap')
        instance = np.array(data['instance'])
        background_data = data.get('background_data')
        
        # Check if model exists
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not loaded'}), 404
        
        model = models[model_name]
        
        # Initialize or get explainer
        explainer_key = f"{model_name}_{explainer_type}"
        if explainer_key not in explainers:
            if explainer_type == 'shap':
                if background_data is not None:
                    bg_data = np.array(background_data)
                    explainers[explainer_key] = {
                        'type': 'shap',
                        'explainer': SHAPExplainer(model, bg_data)
                    }
                else:
                    # Use model-agnostic SHAP
                    explainers[explainer_key] = {
                        'type': 'shap_agnostic',
                        'explainer': ModelAgnosticExplainer(model)
                    }
            elif explainer_type == 'lime':
                feature_names = data.get('feature_names')
                explainers[explainer_key] = {
                    'type': 'lime',
                    'explainer': LIMEExplainer(model, feature_names=feature_names)
                }
            elif explainer_type == 'counterfactual':
                target_class = data.get('target_class', 0)
                explainers[explainer_key] = {
                    'type': 'counterfactual',
                    'explainer': CounterfactualExplainer(model, target_class=target_class)
                }
            else:
                return jsonify({'error': f'Unknown explainer type: {explainer_type}'}), 400
        
        explainer_info = explainers[explainer_key]
        explainer = explainer_info['explainer']
        
        # Generate explanation
        if explainer_type == 'shap':
            if isinstance(explainer, SHAPExplainer):
                shap_values = explainer.explain_instance(instance)
                feature_importance = explainer.get_feature_importance(shap_values)
                
                return jsonify({
                    'explainer_type': 'shap',
                    'shap_values': shap_values.tolist(),
                    'feature_importance': {
                        'features': feature_importance['features'].tolist(),
                        'importance': feature_importance['importance'].tolist()
                    },
                    'base_value': float(explainer.explainer.expected_value)
                })
            else:
                # Model-agnostic SHAP
                shap_values = explainer.explain_instance(instance)
                return jsonify({
                    'explainer_type': 'shap_agnostic',
                    'shap_values': shap_values.tolist()
                })
                
        elif explainer_type == 'lime':
            explanation = explainer.explain_instance(instance)
            return jsonify({
                'explainer_type': 'lime',
                'explanation': explanation
            })
            
        elif explainer_type == 'counterfactual':
            counterfactual = explainer.generate_counterfactual(instance)
            insights = ActionableInsights.generate_recommendations(
                instance, 
                counterfactual,
                feature_names=data.get('feature_names')
            )
            
            return jsonify({
                'explainer_type': 'counterfactual',
                'original': instance.tolist(),
                'counterfactual': counterfactual.tolist(),
                'insights': insights
            })
    
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/models/<model_name>/batch-explain', methods=['POST'])
def batch_explain(model_name: str):
    """Generate explanations for batch of instances"""
    try:
        data = request.json
        explainer_type = data.get('explainer_type', 'shap')
        instances = np.array(data['instances'])
        
        # Check if model exists
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not loaded'}), 404
        
        model = models[model_name]
        
        # For LIME, use batch explanation
        if explainer_type == 'lime':
            explainer_key = f"{model_name}_lime"
            if explainer_key not in explainers:
                feature_names = data.get('feature_names')
                explainers[explainer_key] = {
                    'type': 'lime',
                    'explainer': LIMEExplainer(model, feature_names=feature_names)
                }
            
            explainer = explainers[explainer_key]['explainer']
            explanations = explainer.explain_batch(instances)
            
            return jsonify({
                'explainer_type': 'lime',
                'explanations': explanations
            })
        
        # For other explainers, explain each instance
        results = []
        for instance in instances:
            # Call single explain endpoint
            response = explain_prediction(model_name)
            if isinstance(response, tuple):
                return response
            results.append(response.json)
        
        return jsonify({
            'explainer_type': explainer_type,
            'batch_size': len(instances),
            'explanations': results
        })
    
    except Exception as e:
        logger.error(f"Error generating batch explanations: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8001))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting PyTorch Deep Learning Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
