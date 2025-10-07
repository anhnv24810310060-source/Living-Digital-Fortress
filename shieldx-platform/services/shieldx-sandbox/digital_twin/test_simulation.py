import unittest
import json
from datetime import datetime
from runner import DigitalTwinRunner, NetworkSnapshot, AttackScenario

class TestDigitalTwin(unittest.TestCase):
    def setUp(self):
        self.db_url = "postgresql://test:test@localhost:5432/test_twin"
        try:
            self.runner = DigitalTwinRunner(self.db_url)
        except:
            self.skipTest("Database not available")
    
    def test_snapshot_creation(self):
        """Test network snapshot creation"""
        snapshot_id = self.runner.create_snapshot(
            name="Test Network",
            topology={"subnets": [{"name": "test", "cidr": "10.0.1.0/24"}]},
            services=[{"name": "web", "type": "web_server"}],
            vulnerabilities=[{"type": "test_vuln", "severity": "low"}]
        )
        
        self.assertIsNotNone(snapshot_id)
        self.assertTrue(snapshot_id.startswith("snap_"))
        
        # Verify snapshot can be loaded
        snapshot = self.runner.load_snapshot(snapshot_id)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.name, "Test Network")
    
    def test_scenario_creation(self):
        """Test attack scenario creation"""
        scenario_id = self.runner.create_scenario(
            name="Test Attack",
            description="Test scenario",
            attack_steps=[{"name": "test_step", "type": "port_scan"}],
            expected_outcomes=["test_outcome"],
            target_services=["web_server"]
        )
        
        self.assertIsNotNone(scenario_id)
        self.assertTrue(scenario_id.startswith("scenario_"))
        
        # Verify scenario can be loaded
        scenario = self.runner.load_scenario(scenario_id)
        self.assertIsNotNone(scenario)
        self.assertEqual(scenario.name, "Test Attack")
    
    def test_simulation_precision_calculation(self):
        """Test simulation precision score calculation"""
        scenario = AttackScenario(
            scenario_id="test",
            name="test",
            description="test",
            attack_steps=[
                {"should_detect": True},
                {"should_detect": True},
                {"should_detect": False}
            ],
            expected_outcomes=[],
            target_services=[],
            difficulty="medium",
            created_at=datetime.now()
        )
        
        attack_results = {
            "detected_attacks": [{"step_number": 1}, {"step_number": 2}]
        }
        
        precision = self.runner._calculate_precision_score(scenario, attack_results)
        self.assertEqual(precision, 1.0)  # 2 detected out of 2 expected
    
    def test_port_scan_simulation(self):
        """Test port scan simulation"""
        containers = {
            "web_server": {
                "container": None,
                "service_config": {"type": "web_server"},
                "ip_address": "10.0.1.100"
            }
        }
        
        step = {
            "target": "web_server",
            "ports": [80, 443, 22, 3389]
        }
        
        result = self.runner._simulate_port_scan(step, containers)
        
        self.assertTrue(result["success"])
        self.assertIn("open_ports", result)
        self.assertIn("target_ip", result)
        self.assertEqual(result["target_ip"], "10.0.1.100")
    
    def test_exploit_simulation(self):
        """Test vulnerability exploitation simulation"""
        containers = {
            "web_server": {
                "container": None,
                "service_config": {"type": "web_server"},
                "ip_address": "10.0.1.100"
            }
        }
        
        step = {
            "target": "web_server",
            "vulnerability": "sql_injection"
        }
        
        result = self.runner._simulate_exploit(step, containers)
        
        self.assertIn("success", result)
        self.assertIn("detected", result)
        self.assertIn("vulnerability", result)
        self.assertEqual(result["vulnerability"], "sql_injection")
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        scenario = AttackScenario(
            scenario_id="test",
            name="test",
            description="test",
            attack_steps=[{}, {}, {}],  # 3 steps
            expected_outcomes=[],
            target_services=[],
            difficulty="medium",
            created_at=datetime.now()
        )
        
        attack_results = {
            "execution_log": [
                {"status": "success"},
                {"status": "success"},
                {"status": "failed"}
            ]
        }
        
        success_rate = self.runner._calculate_success_rate(scenario, attack_results)
        self.assertAlmostEqual(success_rate, 2/3, places=2)
    
    def test_recommendation_generation(self):
        """Test security recommendation generation"""
        attack_results = {
            "detected_attacks": [],
            "execution_log": [
                {"step_name": "port_scan_step"},
                {"step_name": "lateral_movement_step"},
                {"step_name": "normal_step"}
            ]
        }
        
        recommendations = self.runner._generate_recommendations(attack_results)
        
        self.assertIsInstance(recommendations, list)
        self.assertTrue(any("monitoring" in rec.lower() for rec in recommendations))
        self.assertTrue(any("intrusion detection" in rec.lower() for rec in recommendations))
        self.assertTrue(any("segmentation" in rec.lower() for rec in recommendations))

if __name__ == '__main__':
    unittest.main(verbosity=2)