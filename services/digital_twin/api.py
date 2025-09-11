from flask import Flask, request, jsonify
from runner import DigitalTwinRunner
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

DB_URL = os.getenv("DATABASE_URL", "postgresql://twin_user:twin_pass2024@localhost:5432/digital_twin")
runner = DigitalTwinRunner(DB_URL)

@app.route('/snapshots', methods=['POST'])
def create_snapshot():
    """Create new network snapshot"""
    try:
        data = request.get_json()
        
        if not data or not all(k in data for k in ['name', 'topology', 'services']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        snapshot_id = runner.create_snapshot(
            name=data['name'],
            topology=data['topology'],
            services=data['services'],
            vulnerabilities=data.get('vulnerabilities', [])
        )
        
        return jsonify({
            'success': True, 
            'snapshot_id': snapshot_id,
            'message': f'Snapshot {snapshot_id} created successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/snapshots/<snapshot_id>', methods=['GET'])
def get_snapshot(snapshot_id):
    """Get network snapshot details"""
    try:
        snapshot = runner.load_snapshot(snapshot_id)
        
        if not snapshot:
            return jsonify({'error': 'Snapshot not found'}), 404
        
        return jsonify({
            'snapshot_id': snapshot.snapshot_id,
            'name': snapshot.name,
            'topology': snapshot.topology,
            'services': snapshot.services,
            'vulnerabilities': snapshot.vulnerabilities,
            'created_at': snapshot.created_at.isoformat(),
            'metadata': snapshot.metadata
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scenarios', methods=['POST'])
def create_scenario():
    """Create new attack scenario"""
    try:
        data = request.get_json()
        
        required_fields = ['name', 'description', 'attack_steps', 'expected_outcomes', 'target_services']
        if not data or not all(k in data for k in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        scenario_id = runner.create_scenario(
            name=data['name'],
            description=data['description'],
            attack_steps=data['attack_steps'],
            expected_outcomes=data['expected_outcomes'],
            target_services=data['target_services'],
            difficulty=data.get('difficulty', 'medium')
        )
        
        return jsonify({
            'success': True, 
            'scenario_id': scenario_id,
            'message': f'Scenario {scenario_id} created successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scenarios/<scenario_id>', methods=['GET'])
def get_scenario(scenario_id):
    """Get attack scenario details"""
    try:
        scenario = runner.load_scenario(scenario_id)
        
        if not scenario:
            return jsonify({'error': 'Scenario not found'}), 404
        
        return jsonify({
            'scenario_id': scenario.scenario_id,
            'name': scenario.name,
            'description': scenario.description,
            'attack_steps': scenario.attack_steps,
            'expected_outcomes': scenario.expected_outcomes,
            'target_services': scenario.target_services,
            'difficulty': scenario.difficulty,
            'created_at': scenario.created_at.isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulations', methods=['POST'])
def run_simulation():
    """Run attack simulation"""
    try:
        data = request.get_json()
        
        if not data or not all(k in data for k in ['snapshot_id', 'scenario_id']):
            return jsonify({'error': 'Missing snapshot_id or scenario_id'}), 400
        
        result = runner.run_simulation(
            snapshot_id=data['snapshot_id'],
            scenario_id=data['scenario_id']
        )
        
        return jsonify({
            'success': True,
            'simulation_id': result.simulation_id,
            'status': result.status,
            'success_rate': result.success_rate,
            'precision_score': result.precision_score,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat() if result.end_time else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulations/<simulation_id>', methods=['GET'])
def get_simulation_result(simulation_id):
    """Get simulation result"""
    try:
        result = runner.get_simulation_result(simulation_id)
        
        if not result:
            return jsonify({'error': 'Simulation not found'}), 404
        
        return jsonify({
            'simulation_id': result.simulation_id,
            'snapshot_id': result.snapshot_id,
            'scenario_id': result.scenario_id,
            'status': result.status,
            'success_rate': result.success_rate,
            'precision_score': result.precision_score,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'execution_log': result.execution_log,
            'detected_attacks': result.detected_attacks,
            'report': result.report_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'digital-twin',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)