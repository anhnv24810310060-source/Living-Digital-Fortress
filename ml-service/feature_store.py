import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import json
import logging
from dataclasses import dataclass, asdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter as PromCounter, Histogram as PromHistogram, generate_latest, CONTENT_TYPE_LATEST
import hashlib

@dataclass
class PluginOutput:
    plugin_id: str
    artifact_id: str
    success: bool
    results: Dict[str, Any]
    confidence: float
    tags: List[str]
    indicators: List[Dict[str, Any]]
    execution_time: int
    timestamp: datetime

@dataclass
class NormalizedFeatures:
    artifact_id: str
    plugin_id: str
    feature_vector: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime

class FeatureStore:
    def __init__(self, db_url: str, redis_url: str):
        self.db_url = db_url
        self.redis_client = redis.from_url(redis_url)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_schema = self._load_feature_schema()
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS ml_features (
                            id SERIAL PRIMARY KEY,
                            artifact_id VARCHAR(255) NOT NULL,
                            plugin_id VARCHAR(255) NOT NULL,
                            feature_vector JSONB NOT NULL,
                            feature_names JSONB NOT NULL,
                            metadata JSONB NOT NULL,
                            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        );
                        
                        CREATE INDEX IF NOT EXISTS idx_ml_features_artifact ON ml_features(artifact_id);
                        CREATE INDEX IF NOT EXISTS idx_ml_features_plugin ON ml_features(plugin_id);
                        CREATE INDEX IF NOT EXISTS idx_ml_features_timestamp ON ml_features(timestamp);
                        
                        CREATE TABLE IF NOT EXISTS ml_training_jobs (
                            id SERIAL PRIMARY KEY,
                            job_id VARCHAR(255) UNIQUE NOT NULL,
                            status VARCHAR(50) NOT NULL,
                            feature_count INTEGER,
                            model_metrics JSONB,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            completed_at TIMESTAMP WITH TIME ZONE
                        );
                    """)
                    conn.commit()
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
            raise
        
    def normalize_plugin_output(self, plugin_output: PluginOutput) -> NormalizedFeatures:
        """Normalize plugin output to standardized feature vector"""
        
        # Extract base features
        base_features = {
            'confidence': plugin_output.confidence,
            'execution_time_ms': plugin_output.execution_time,
            'success': 1.0 if plugin_output.success else 0.0,
            'tag_count': len(plugin_output.tags),
            'indicator_count': len(plugin_output.indicators)
        }
        
        # Extract result features
        result_features = self._extract_result_features(plugin_output.results)
        
        # Extract tag features (one-hot encoding)
        tag_features = self._extract_tag_features(plugin_output.tags)
        
        # Extract indicator features
        indicator_features = self._extract_indicator_features(plugin_output.indicators)
        
        # Combine all features
        all_features = {**base_features, **result_features, **tag_features, **indicator_features}
        
        # Convert to feature vector
        feature_vector, feature_names = self._vectorize_features(all_features)
        
        return NormalizedFeatures(
            artifact_id=plugin_output.artifact_id,
            plugin_id=plugin_output.plugin_id,
            feature_vector=feature_vector,
            feature_names=feature_names,
            metadata={
                'original_tags': plugin_output.tags,
                'original_confidence': plugin_output.confidence,
                'plugin_version': self._get_plugin_version(plugin_output.plugin_id),
                'feature_hash': self._compute_feature_hash(all_features)
            },
            timestamp=plugin_output.timestamp
        )
    
    def _extract_result_features(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from plugin results"""
        features = {}
        
        for key, value in results.items():
            if isinstance(value, bool):
                features[f'result_{key}'] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                features[f'result_{key}'] = float(value)
            elif isinstance(value, str):
                # Hash string values to numerical
                features[f'result_{key}_hash'] = abs(hash(value)) % 1000000 / 1000000.0
            elif isinstance(value, dict):
                # Recursively extract from nested dicts
                nested = self._extract_result_features(value)
                for nested_key, nested_value in nested.items():
                    features[f'result_{key}_{nested_key}'] = nested_value
                    
        return features
    
    def _extract_tag_features(self, tags: List[str]) -> Dict[str, float]:
        """Convert tags to one-hot encoded features"""
        features = {}
        
        # Known tag categories
        known_tags = [
            'malware', 'benign', 'suspicious', 'trojan', 'virus', 'worm',
            'ransomware', 'adware', 'spyware', 'rootkit', 'backdoor',
            'executable', 'document', 'archive', 'script', 'image'
        ]
        
        for tag in known_tags:
            features[f'tag_{tag}'] = 1.0 if tag in tags else 0.0
            
        # Tag diversity score
        features['tag_diversity'] = len(set(tags)) / max(len(tags), 1)
        
        return features
    
    def _extract_indicator_features(self, indicators: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract features from indicators"""
        features = {}
        
        if not indicators:
            return {'indicator_count': 0.0, 'avg_indicator_confidence': 0.0}
        
        # Aggregate indicator statistics
        confidences = [ind.get('confidence', 0.0) for ind in indicators]
        features['avg_indicator_confidence'] = np.mean(confidences)
        features['max_indicator_confidence'] = np.max(confidences)
        features['min_indicator_confidence'] = np.min(confidences)
        features['std_indicator_confidence'] = np.std(confidences)
        
        # Indicator type distribution
        indicator_types = [ind.get('type', 'unknown') for ind in indicators]
        type_counts = {}
        for itype in ['hash', 'url', 'ip', 'domain', 'file', 'registry', 'network']:
            type_counts[f'indicator_type_{itype}'] = indicator_types.count(itype)
            
        # Normalize by total count
        total_indicators = len(indicators)
        for key, count in type_counts.items():
            features[key] = count / total_indicators
            
        return features
    
    def _vectorize_features(self, features: Dict[str, float]) -> tuple:
        """Convert feature dict to numpy array"""
        
        # Ensure consistent feature ordering
        sorted_features = sorted(features.items())
        feature_names = [name for name, _ in sorted_features]
        feature_values = np.array([value for _, value in sorted_features])
        
        # Handle NaN values
        feature_values = np.nan_to_num(feature_values, nan=0.0)
        
        return feature_values, feature_names
    
    def store_features(self, features: NormalizedFeatures) -> bool:
        """Store normalized features in database"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Store in features table
                    cur.execute("""
                        INSERT INTO ml_features 
                        (artifact_id, plugin_id, feature_vector, feature_names, metadata, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        features.artifact_id,
                        features.plugin_id,
                        json.dumps(features.feature_vector.tolist()),
                        json.dumps(features.feature_names),
                        json.dumps(features.metadata),
                        features.timestamp
                    ))
                    
                    # Cache in Redis for real-time access
                    cache_key = f"features:{features.artifact_id}:{features.plugin_id}"
                    cache_data = {
                        'vector': features.feature_vector.tolist(),
                        'names': features.feature_names,
                        'timestamp': features.timestamp.isoformat()
                    }
                    self.redis_client.setex(cache_key, 3600, json.dumps(cache_data))
                    
            return True
            
        except Exception as e:
            logging.error(f"Failed to store features: {e}")
            return False
    
    def get_training_data(self, limit: int = 10000) -> tuple:
        """Get feature data for model training"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT feature_vector, metadata, feature_names
                        FROM ml_features 
                        WHERE timestamp > NOW() - INTERVAL '30 days'
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (limit,))
                    
                    rows = cur.fetchall()
                    
            if not rows:
                return np.array([]), np.array([]), []
                
            # Extract features and labels
            X = np.array([json.loads(row['feature_vector']) for row in rows])
            feature_names = json.loads(rows[0]['feature_names']) if rows else []
            
            # Extract labels from metadata
            y = []
            for row in rows:
                metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                y.append(metadata.get('original_confidence', 0.5))
                
            return X, np.array(y), feature_names
            
        except Exception as e:
            logging.error(f"Failed to get training data: {e}")
            return np.array([]), np.array([]), []
    
    def _compute_feature_hash(self, features: Dict[str, float]) -> str:
        """Compute hash of feature vector for deduplication"""
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.sha256(feature_str.encode()).hexdigest()[:16]
    
    def _load_feature_schema(self) -> Dict[str, Any]:
        """Load feature schema configuration"""
        return {
            'version': '1.0',
            'base_features': ['confidence', 'execution_time_ms', 'success', 'tag_count', 'indicator_count'],
            'max_features': 1000,
            'normalization': 'standard'
        }
    
    def _get_plugin_version(self, plugin_id: str) -> str:
        """Get plugin version from cache or database"""
        try:
            cached = self.redis_client.get(f"plugin_version:{plugin_id}")
            if cached:
                return cached.decode('utf-8')
        except:
            pass
        return "unknown"

class MLPipeline:
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        
    def process_plugin_output(self, plugin_output_json: str) -> Dict[str, Any]:
        """Process plugin output through ML pipeline"""
        try:
            # Parse plugin output
            data = json.loads(plugin_output_json)
            plugin_output = PluginOutput(
                plugin_id=data['plugin_id'],
                artifact_id=data['artifact_id'],
                success=data['success'],
                results=data['results'],
                confidence=data['confidence'],
                tags=data['tags'],
                indicators=data['indicators'],
                execution_time=data['execution_time'],
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            )
            
            # Normalize to features
            features = self.feature_store.normalize_plugin_output(plugin_output)
            
            # Store features
            success = self.feature_store.store_features(features)
            
            return {
                'success': success,
                'feature_count': len(features.feature_names),
                'process_id': f"ml_{int(datetime.now().timestamp())}"
            }
            
        except Exception as e:
            logging.error(f"Failed to process plugin output: {e}")
            return {'success': False, 'error': str(e)}

# Flask API
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize components
DB_URL = "postgresql://ml_user:ml_pass2024@localhost:5432/ml_features"
REDIS_URL = "redis://localhost:6379/0"

feature_store = FeatureStore(DB_URL, REDIS_URL)
ml_pipeline = MLPipeline(feature_store)

# Prometheus metrics
REQUESTS_TOTAL = PromCounter('mlservice_http_requests_total', 'Total HTTP requests', ['endpoint', 'method', 'status'])
REQ_DURATION = PromHistogram('mlservice_http_request_duration_seconds', 'HTTP request duration seconds', ['endpoint', 'method'])

def track_metrics(endpoint):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            method = request.method
            start = datetime.now()
            try:
                resp = fn(*args, **kwargs)
                # Flask handlers can return (body, status)
                status = 200
                if isinstance(resp, tuple) and len(resp) >= 2:
                    status = resp[1]
                REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status=status).inc()
                return resp
            finally:
                REQ_DURATION.labels(endpoint=endpoint, method=method).observe((datetime.now() - start).total_seconds())
        return wrapper
    return decorator

@app.route('/process', methods=['POST'])
@track_metrics('process')
def process_plugin_output():
    try:
        plugin_output_json = request.get_json()
        if not plugin_output_json:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
            
        result = ml_pipeline.process_plugin_output(json.dumps(plugin_output_json))
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logging.error(f"API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/training-data', methods=['GET'])
@track_metrics('training_data')
def get_training_data():
    try:
        limit = int(request.args.get('limit', 10000))
        X, y, feature_names = feature_store.get_training_data(limit)
        
        return jsonify({
            'success': True,
            'feature_matrix': X.tolist(),
            'labels': y.tolist(),
            'feature_names': feature_names,
            'count': len(X)
        }), 200
        
    except Exception as e:
        logging.error(f"Training data error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
@track_metrics('health')
def health_check():
@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
    return jsonify({
        'status': 'healthy',
        'service': 'ml-feature-store',
        'timestamp': datetime.now().isoformat()
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)