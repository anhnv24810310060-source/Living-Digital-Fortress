import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class BehavioralScorer:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.scaler = StandardScaler()
        self.keystroke_model = IsolationForest(contamination=0.1, random_state=42)
        self.mouse_model = IsolationForest(contamination=0.1, random_state=42)
        self.access_model = IsolationForest(contamination=0.1, random_state=42)
        
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained behavioral models"""
        try:
            self.keystroke_model = joblib.load('models/keystroke_model.pkl')
            self.mouse_model = joblib.load('models/mouse_model.pkl')
            self.access_model = joblib.load('models/access_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            logging.info("Loaded pre-trained behavioral models")
        except FileNotFoundError:
            logging.info("No pre-trained models found, using default models")
    
    def calculate_keystroke_risk(self, keystroke_data: list, user_id: str) -> float:
        """Calculate keystroke dynamics risk score"""
        if len(keystroke_data) < 5:
            return 0.5
        
        features = self._extract_keystroke_features(keystroke_data)
        baseline = self._get_user_baseline(user_id, 'keystroke')
        
        if baseline is None:
            return 0.3
        
        deviation_score = self._calculate_deviation(features, baseline)
        
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        anomaly_score = self.keystroke_model.decision_function(feature_vector)[0]
        
        risk_score = (deviation_score + abs(anomaly_score)) / 2
        return min(risk_score, 1.0)
    
    def _extract_keystroke_features(self, keystroke_data: list) -> dict:
        """Extract statistical features from keystroke data"""
        if not keystroke_data:
            return {}
        
        intervals = []
        durations = []
        pressures = []
        
        for i in range(1, len(keystroke_data)):
            interval = keystroke_data[i]['timestamp'] - keystroke_data[i-1]['timestamp']
            intervals.append(interval)
            durations.append(keystroke_data[i]['duration'])
            if 'pressure' in keystroke_data[i]:
                pressures.append(keystroke_data[i]['pressure'])
        
        features = {
            'avg_interval': np.mean(intervals) if intervals else 0,
            'std_interval': np.std(intervals) if intervals else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'std_duration': np.std(durations) if durations else 0,
            'typing_speed': len(keystroke_data) / (max(intervals) / 1000) if intervals else 0,
        }
        
        if pressures:
            features.update({
                'avg_pressure': np.mean(pressures),
                'std_pressure': np.std(pressures)
            })
        
        return features
    
    def calculate_mouse_risk(self, mouse_data: list, user_id: str) -> float:
        """Calculate mouse dynamics risk score"""
        if len(mouse_data) < 10:
            return 0.3
        
        features = self._extract_mouse_features(mouse_data)
        baseline = self._get_user_baseline(user_id, 'mouse')
        
        if baseline is None:
            return 0.3
        
        deviation_score = self._calculate_deviation(features, baseline)
        
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        anomaly_score = self.mouse_model.decision_function(feature_vector)[0]
        
        risk_score = (deviation_score + abs(anomaly_score)) / 2
        return min(risk_score, 1.0)
    
    def _extract_mouse_features(self, mouse_data: list) -> dict:
        """Extract statistical features from mouse movement data"""
        if not mouse_data:
            return {}
        
        velocities = []
        accelerations = []
        distances = []
        
        for i in range(1, len(mouse_data)):
            prev = mouse_data[i-1]
            curr = mouse_data[i]
            
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            distance = np.sqrt(dx*dx + dy*dy)
            distances.append(distance)
            
            dt = (curr['timestamp'] - prev['timestamp']) / 1000.0
            if dt > 0:
                velocity = distance / dt
                velocities.append(velocity)
                
                if i > 1 and len(velocities) > 1:
                    acceleration = (velocities[-1] - velocities[-2]) / dt
                    accelerations.append(acceleration)
        
        features = {
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'std_velocity': np.std(velocities) if velocities else 0,
            'avg_acceleration': np.mean(accelerations) if accelerations else 0,
            'std_acceleration': np.std(accelerations) if accelerations else 0,
            'avg_distance': np.mean(distances) if distances else 0,
            'total_distance': sum(distances) if distances else 0,
        }
        
        return features
    
    def calculate_access_pattern_risk(self, access_patterns: list, user_id: str) -> float:
        """Calculate access pattern risk score"""
        if not access_patterns:
            return 0.5
        
        features = self._extract_access_features(access_patterns)
        baseline = self._get_user_baseline(user_id, 'access')
        
        if baseline is None:
            return 0.2
        
        deviation_score = self._calculate_deviation(features, baseline)
        
        failure_rate = features.get('failure_rate', 0)
        if failure_rate > 0.3:
            deviation_score += 0.4
        
        return min(deviation_score, 1.0)
    
    def _extract_access_features(self, access_patterns: list) -> dict:
        """Extract features from access patterns"""
        if not access_patterns:
            return {}
        
        total_accesses = len(access_patterns)
        failed_accesses = sum(1 for access in access_patterns if not access.get('success', True))
        
        resources = [access.get('resource', '') for access in access_patterns]
        unique_resources = len(set(resources))
        
        timestamps = [datetime.fromisoformat(access['timestamp'].replace('Z', '+00:00')) 
                     for access in access_patterns if 'timestamp' in access]
        
        time_features = {}
        if timestamps:
            hours = [ts.hour for ts in timestamps]
            time_features = {
                'avg_hour': np.mean(hours),
                'std_hour': np.std(hours),
                'night_access_ratio': sum(1 for h in hours if h < 6 or h > 22) / len(hours)
            }
        
        features = {
            'failure_rate': failed_accesses / total_accesses if total_accesses > 0 else 0,
            'resource_diversity': unique_resources / total_accesses if total_accesses > 0 else 0,
            'access_frequency': total_accesses,
            **time_features
        }
        
        return features
    
    def _get_user_baseline(self, user_id: str, feature_type: str) -> dict:
        """Get user behavioral baseline from database"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT baseline_data FROM user_baselines 
                        WHERE user_id = %s AND feature_type = %s
                        ORDER BY last_updated DESC LIMIT 1
                    """, (user_id, feature_type))
                    
                    row = cur.fetchone()
                    if row:
                        return json.loads(row['baseline_data'])
        except Exception as e:
            logging.error(f"Failed to get user baseline: {e}")
        
        return None
    
    def _calculate_deviation(self, current_features: dict, baseline: dict) -> float:
        """Calculate deviation from baseline features"""
        if not baseline or not current_features:
            return 0.5
        
        deviations = []
        for key in current_features:
            if key in baseline and baseline[key] != 0:
                deviation = abs(current_features[key] - baseline[key]) / abs(baseline[key])
                deviations.append(min(deviation, 2.0))
        
        return np.mean(deviations) if deviations else 0.5
    
    def update_user_baseline(self, user_id: str, feature_type: str, features: dict):
        """Update user behavioral baseline"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO user_baselines (user_id, feature_type, baseline_data, last_updated)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (user_id, feature_type) 
                        DO UPDATE SET baseline_data = EXCLUDED.baseline_data, 
                                     last_updated = EXCLUDED.last_updated
                    """, (user_id, feature_type, json.dumps(features)))
                    conn.commit()
        except Exception as e:
            logging.error(f"Failed to update user baseline: {e}")

# Initialize scorer
DB_URL = os.getenv("DATABASE_URL", "postgresql://contauth_user:contauth_pass2024@localhost:5432/contauth")
scorer = BehavioralScorer(DB_URL)

@app.route('/score/keystroke', methods=['POST'])
def score_keystroke():
    """Score keystroke dynamics"""
    try:
        data = request.get_json()
        
        if not data or 'keystroke_data' not in data or 'user_id' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        risk_score = scorer.calculate_keystroke_risk(
            data['keystroke_data'], 
            data['user_id']
        )
        
        return jsonify({
            'success': True,
            'risk_score': risk_score,
            'feature_type': 'keystroke'
        })
        
    except Exception as e:
        logging.error(f"Keystroke scoring error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/score/mouse', methods=['POST'])
def score_mouse():
    """Score mouse dynamics"""
    try:
        data = request.get_json()
        
        if not data or 'mouse_data' not in data or 'user_id' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        risk_score = scorer.calculate_mouse_risk(
            data['mouse_data'], 
            data['user_id']
        )
        
        return jsonify({
            'success': True,
            'risk_score': risk_score,
            'feature_type': 'mouse'
        })
        
    except Exception as e:
        logging.error(f"Mouse scoring error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/score/access', methods=['POST'])
def score_access():
    """Score access patterns"""
    try:
        data = request.get_json()
        
        if not data or 'access_patterns' not in data or 'user_id' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        risk_score = scorer.calculate_access_pattern_risk(
            data['access_patterns'], 
            data['user_id']
        )
        
        return jsonify({
            'success': True,
            'risk_score': risk_score,
            'feature_type': 'access'
        })
        
    except Exception as e:
        logging.error(f"Access pattern scoring error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/baseline/update', methods=['POST'])
def update_baseline():
    """Update user behavioral baseline"""
    try:
        data = request.get_json()
        
        required_fields = ['user_id', 'feature_type', 'features']
        if not data or not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        scorer.update_user_baseline(
            data['user_id'],
            data['feature_type'],
            data['features']
        )
        
        return jsonify({
            'success': True,
            'message': 'Baseline updated successfully'
        })
        
    except Exception as e:
        logging.error(f"Baseline update error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'behavioral-scorer',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=False)