#!/usr/bin/env python3
"""
Mock Services for ShieldX Gateway Testing
Simulates Zero Trust, AI Analyzer, and other backend services
"""

import json
import time
import random
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

# Mock Zero Trust Service (Port 8091)
@app.route('/evaluate', methods=['POST'])
def zero_trust_evaluate():
    """Mock Zero Trust evaluation"""
    data = request.get_json()
    client_ip = data.get('client_ip', '')
    path = data.get('path', '')
    user_agent = data.get('user_agent', '')
    
    # Simulate trust scoring
    trust_score = 0.8
    
    # Lower trust for suspicious indicators
    if 'admin' in path.lower():
        trust_score -= 0.3
    if 'bot' in user_agent.lower() or 'curl' in user_agent.lower():
        trust_score -= 0.2
    if '..' in path:
        trust_score -= 0.4
    if any(x in path.lower() for x in ['union', 'select', 'script']):
        trust_score -= 0.5
    
    trust_score = max(0.0, min(1.0, trust_score))
    
    # Add some realistic delay
    time.sleep(random.uniform(0.001, 0.01))
    
    return jsonify({
        'trust_score': trust_score,
        'allow': trust_score >= 0.5,
        'require_additional_auth': trust_score < 0.7,
        'require_deception': trust_score < 0.3,
        'block': trust_score < 0.1,
        'risk_level': 'HIGH' if trust_score < 0.3 else 'MEDIUM' if trust_score < 0.7 else 'LOW',
        'factors': {
            'ip_reputation': 0.8,
            'user_agent': 0.7 if 'bot' not in user_agent.lower() else 0.3,
            'path_risk': 0.9 if '..' not in path else 0.2,
            'method_risk': 0.8
        }
    })

# Mock AI Analyzer Service (Port 8087) 
@app.route('/analyze', methods=['POST'])
def ai_analyzer_analyze():
    """Mock AI analysis"""
    data = request.get_json()
    features = data.get('features', [])
    
    # Simulate ML analysis
    threat_score = random.uniform(0.1, 0.4)  # Base threat score
    
    # Increase threat score based on features
    if len(features) > 0:
        # Simulate feature analysis
        if features[0] > 100:  # Path length
            threat_score += 0.2
        if features[1] > 5:    # Path depth
            threat_score += 0.1
        if features[4] > 0.5:  # Suspicious path score
            threat_score += 0.3
    
    threat_score = min(1.0, threat_score)
    is_anomaly = threat_score > 0.6
    
    # Add processing delay
    time.sleep(random.uniform(0.005, 0.02))
    
    return jsonify({
        'success': True,
        'is_anomaly': is_anomaly,
        'score': threat_score,
        'confidence': random.uniform(0.7, 0.95),
        'explanation': 'ML model analysis completed',
        'threat_score': threat_score,
        'behavior_pattern': 'SUSPICIOUS' if is_anomaly else 'NORMAL',
        'attack_vector': 'PATH_TRAVERSAL' if '..' in str(features) else 'UNKNOWN'
    })

# Mock Isolation Vault Service (Port 8085)
@app.route('/isolate', methods=['POST'])
def isolation_vault_isolate():
    """Mock isolation vault"""
    data = request.get_json()
    threat_level = data.get('threat_level', 'MEDIUM')
    
    # Simulate isolation configuration
    config = {
        'isolation_level': threat_level,
        'network_restrictions': ['10.0.0.0/8', '172.16.0.0/12'],
        'resource_limits': {
            'memory': '256MB' if threat_level == 'HIGH' else '512MB',
            'cpu': '0.5' if threat_level == 'HIGH' else '1.0',
            'disk_io': 'limited' if threat_level == 'HIGH' else 'normal'
        },
        'monitoring_level': threat_level,
        'sandbox_id': f"vault_{random.randint(1000, 9999)}"
    }
    
    time.sleep(random.uniform(0.002, 0.008))
    
    return jsonify(config)

# Mock Deception Engine Service (Port 8084)
@app.route('/select-decoy', methods=['POST'])
def deception_engine_select():
    """Mock deception engine"""
    data = request.get_json()
    attack_vector = data.get('attack_vector', 'UNKNOWN')
    
    # Select appropriate decoy based on attack vector
    decoy_types = {
        'SQL_INJECTION': 'database_honeypot',
        'XSS': 'web_honeypot', 
        'PATH_TRAVERSAL': 'file_honeypot',
        'UNKNOWN': 'generic_honeypot'
    }
    
    decoy_type = decoy_types.get(attack_vector, 'generic_honeypot')
    decoy_id = f"decoy_{random.randint(1000, 9999)}"
    
    time.sleep(random.uniform(0.003, 0.01))
    
    return jsonify({
        'decoy_id': decoy_id,
        'decoy_type': decoy_type,
        'endpoint': f'http://localhost:8083/{decoy_type}',
        'effectiveness': random.uniform(0.7, 0.95),
        'estimated_engagement_time': random.randint(30, 300)
    })

# Health endpoints for all services
@app.route('/health', methods=['GET'])
def health_check():
    """Health check for all mock services"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'service': 'mock-services',
        'uptime': time.time() - start_time
    })

# Metrics endpoint
@app.route('/metrics', methods=['GET'])
def metrics():
    """Mock metrics endpoint"""
    return """
# HELP mock_requests_total Total requests processed
# TYPE mock_requests_total counter
mock_requests_total 1234

# HELP mock_request_duration_seconds Request duration
# TYPE mock_request_duration_seconds histogram
mock_request_duration_seconds_bucket{le="0.005"} 100
mock_request_duration_seconds_bucket{le="0.01"} 200
mock_request_duration_seconds_bucket{le="0.025"} 300
mock_request_duration_seconds_bucket{le="+Inf"} 400
"""

def run_mock_service(port, service_name):
    """Run mock service on specific port"""
    print(f"Starting {service_name} on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == "__main__":
    start_time = time.time()
    
    print("Starting Mock Services for ShieldX Gateway Testing")
    print("=" * 60)
    
    # Start multiple services on different ports
    services = [
        (8091, "Zero Trust Engine"),
        (8087, "AI Analyzer"), 
        (8085, "Isolation Vault"),
        (8084, "Deception Engine")
    ]
    
    threads = []
    
    for port, name in services:
        thread = threading.Thread(
            target=run_mock_service,
            args=(port, name),
            daemon=True
        )
        thread.start()
        threads.append(thread)
        time.sleep(0.5)  # Stagger startup
    
    print("\nAll mock services started!")
    print("Services available:")
    for port, name in services:
        print(f"  - {name}: http://localhost:{port}")
    
    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping mock services...")
        print("Mock services stopped.")