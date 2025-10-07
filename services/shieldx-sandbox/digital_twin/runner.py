import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
import docker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NetworkSnapshot:
    snapshot_id: str
    name: str
    topology: Dict[str, Any]
    services: List[Dict[str, Any]]
    vulnerabilities: List[Dict[str, Any]]
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class AttackScenario:
    scenario_id: str
    name: str
    description: str
    attack_steps: List[Dict[str, Any]]
    expected_outcomes: List[str]
    target_services: List[str]
    difficulty: str
    created_at: datetime

@dataclass
class SimulationResult:
    simulation_id: str
    snapshot_id: str
    scenario_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    execution_log: List[Dict[str, Any]]
    detected_attacks: List[Dict[str, Any]]
    success_rate: float
    precision_score: float
    report_data: Dict[str, Any]

class DigitalTwinRunner:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.docker_client = docker.from_env()
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS network_snapshots (
                            snapshot_id VARCHAR(255) PRIMARY KEY,
                            name VARCHAR(255) NOT NULL,
                            topology JSONB NOT NULL,
                            services JSONB NOT NULL,
                            vulnerabilities JSONB NOT NULL,
                            metadata JSONB NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        );
                        
                        CREATE TABLE IF NOT EXISTS attack_scenarios (
                            scenario_id VARCHAR(255) PRIMARY KEY,
                            name VARCHAR(255) NOT NULL,
                            description TEXT,
                            attack_steps JSONB NOT NULL,
                            expected_outcomes JSONB NOT NULL,
                            target_services JSONB NOT NULL,
                            difficulty VARCHAR(50) NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        );
                        
                        CREATE TABLE IF NOT EXISTS simulation_results (
                            simulation_id VARCHAR(255) PRIMARY KEY,
                            snapshot_id VARCHAR(255),
                            scenario_id VARCHAR(255),
                            status VARCHAR(50) NOT NULL,
                            start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                            end_time TIMESTAMP WITH TIME ZONE,
                            execution_log JSONB NOT NULL,
                            detected_attacks JSONB NOT NULL,
                            success_rate FLOAT NOT NULL,
                            precision_score FLOAT NOT NULL,
                            report_data JSONB NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        );
                    """)
                    conn.commit()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def load_snapshot(self, snapshot_id: str) -> Optional[NetworkSnapshot]:
        """Load network snapshot from database"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM network_snapshots WHERE snapshot_id = %s", (snapshot_id,))
                    row = cur.fetchone()
                    if not row:
                        return None
                    
                    return NetworkSnapshot(
                        snapshot_id=row['snapshot_id'],
                        name=row['name'],
                        topology=row['topology'],
                        services=row['services'],
                        vulnerabilities=row['vulnerabilities'],
                        created_at=row['created_at'],
                        metadata=row['metadata']
                    )
        except Exception as e:
            logger.error(f"Failed to load snapshot {snapshot_id}: {e}")
            return None
    
    def create_snapshot(self, name: str, topology: Dict[str, Any], 
                       services: List[Dict[str, Any]], 
                       vulnerabilities: List[Dict[str, Any]]) -> str:
        """Create new network snapshot"""
        snapshot_id = f"snap_{uuid.uuid4().hex[:8]}"
        
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO network_snapshots 
                        (snapshot_id, name, topology, services, vulnerabilities, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        snapshot_id, name, 
                        json.dumps(topology),
                        json.dumps(services),
                        json.dumps(vulnerabilities),
                        json.dumps({"created_by": "digital_twin_runner"})
                    ))
                    conn.commit()
            
            logger.info(f"Created snapshot {snapshot_id}: {name}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise
    
    def load_scenario(self, scenario_id: str) -> Optional[AttackScenario]:
        """Load attack scenario from database"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM attack_scenarios WHERE scenario_id = %s", (scenario_id,))
                    row = cur.fetchone()
                    if not row:
                        return None
                    
                    return AttackScenario(
                        scenario_id=row['scenario_id'],
                        name=row['name'],
                        description=row['description'],
                        attack_steps=row['attack_steps'],
                        expected_outcomes=row['expected_outcomes'],
                        target_services=row['target_services'],
                        difficulty=row['difficulty'],
                        created_at=row['created_at']
                    )
        except Exception as e:
            logger.error(f"Failed to load scenario {scenario_id}: {e}")
            return None
    
    def create_scenario(self, name: str, description: str, 
                       attack_steps: List[Dict[str, Any]],
                       expected_outcomes: List[str],
                       target_services: List[str],
                       difficulty: str = "medium") -> str:
        """Create new attack scenario"""
        scenario_id = f"scenario_{uuid.uuid4().hex[:8]}"
        
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO attack_scenarios 
                        (scenario_id, name, description, attack_steps, expected_outcomes, target_services, difficulty)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        scenario_id, name, description,
                        json.dumps(attack_steps),
                        json.dumps(expected_outcomes),
                        json.dumps(target_services),
                        difficulty
                    ))
                    conn.commit()
            
            logger.info(f"Created scenario {scenario_id}: {name}")
            return scenario_id
            
        except Exception as e:
            logger.error(f"Failed to create scenario: {e}")
            raise
    
    def run_simulation(self, snapshot_id: str, scenario_id: str) -> SimulationResult:
        """Run attack simulation on network snapshot"""
        simulation_id = f"sim_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting simulation {simulation_id}: snapshot={snapshot_id}, scenario={scenario_id}")
        
        snapshot = self.load_snapshot(snapshot_id)
        scenario = self.load_scenario(scenario_id)
        
        if not snapshot or not scenario:
            raise ValueError("Invalid snapshot or scenario ID")
        
        result = SimulationResult(
            simulation_id=simulation_id,
            snapshot_id=snapshot_id,
            scenario_id=scenario_id,
            status="running",
            start_time=start_time,
            end_time=None,
            execution_log=[],
            detected_attacks=[],
            success_rate=0.0,
            precision_score=0.0,
            report_data={}
        )
        
        try:
            self._store_simulation_result(result)
            
            # Deploy network topology
            network_containers = self._deploy_network_topology(snapshot, simulation_id)
            result.execution_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "deploy_topology",
                "status": "success",
                "details": f"Deployed {len(network_containers)} containers"
            })
            
            # Execute attack scenario
            attack_results = self._execute_attack_scenario(scenario, network_containers, simulation_id)
            result.execution_log.extend(attack_results["execution_log"])
            result.detected_attacks = attack_results["detected_attacks"]
            
            # Calculate metrics
            result.success_rate = self._calculate_success_rate(scenario, attack_results)
            result.precision_score = self._calculate_precision_score(scenario, attack_results)
            
            # Generate report
            result.report_data = self._generate_simulation_report(snapshot, scenario, attack_results)
            
            result.status = "completed"
            result.end_time = datetime.now(timezone.utc)
            
            logger.info(f"Simulation {simulation_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Simulation {simulation_id} failed: {e}")
            result.status = "failed"
            result.end_time = datetime.now(timezone.utc)
            result.execution_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "simulation_error",
                "status": "error",
                "details": str(e)
            })
        
        finally:
            self._cleanup_simulation(simulation_id)
            self._store_simulation_result(result)
        
        return result
    
    def _deploy_network_topology(self, snapshot: NetworkSnapshot, simulation_id: str) -> Dict[str, Any]:
        """Deploy network topology using Docker containers"""
        containers = {}
        network_name = f"sim_network_{simulation_id}"
        
        try:
            network = self.docker_client.networks.create(network_name, driver="bridge")
            
            for service in snapshot.services:
                container_name = f"sim_{simulation_id}_{service['name']}"
                image = self._get_service_image(service)
                
                container = self.docker_client.containers.run(
                    image,
                    name=container_name,
                    network=network_name,
                    detach=True,
                    environment=service.get("environment", {}),
                    labels={"simulation_id": simulation_id}
                )
                
                containers[service["name"]] = {
                    "container": container,
                    "service_config": service,
                    "ip_address": self._get_container_ip(container, network_name)
                }
                
                logger.info(f"Deployed service {service['name']} as {container_name}")
            
            time.sleep(5)  # Wait for services to be ready
            return containers
            
        except Exception as e:
            logger.error(f"Failed to deploy topology: {e}")
            self._cleanup_simulation(simulation_id)
            raise
    
    def _execute_attack_scenario(self, scenario: AttackScenario, 
                                containers: Dict[str, Any], 
                                simulation_id: str) -> Dict[str, Any]:
        """Execute attack scenario steps"""
        execution_log = []
        detected_attacks = []
        
        for i, step in enumerate(scenario.attack_steps):
            step_start = datetime.now(timezone.utc)
            
            logger.info(f"Executing attack step {i+1}: {step['name']}")
            
            try:
                step_result = self._execute_attack_step(step, containers, simulation_id)
                
                execution_log.append({
                    "step_number": i + 1,
                    "step_name": step["name"],
                    "timestamp": step_start.isoformat(),
                    "duration_seconds": (datetime.now(timezone.utc) - step_start).total_seconds(),
                    "status": "success",
                    "result": step_result,
                    "target": step.get("target", "unknown")
                })
                
                if step_result.get("detected", False):
                    detected_attacks.append({
                        "step_number": i + 1,
                        "attack_type": step.get("type", "unknown"),
                        "target": step.get("target", "unknown"),
                        "detection_method": step_result.get("detection_method", "unknown"),
                        "confidence": step_result.get("confidence", 0.0),
                        "timestamp": step_start.isoformat()
                    })
                
                time.sleep(step.get("delay", 2))
                
            except Exception as e:
                logger.error(f"Attack step {i+1} failed: {e}")
                execution_log.append({
                    "step_number": i + 1,
                    "step_name": step["name"],
                    "timestamp": step_start.isoformat(),
                    "duration_seconds": (datetime.now(timezone.utc) - step_start).total_seconds(),
                    "status": "failed",
                    "error": str(e),
                    "target": step.get("target", "unknown")
                })
        
        return {"execution_log": execution_log, "detected_attacks": detected_attacks}
    
    def _execute_attack_step(self, step: Dict[str, Any], 
                           containers: Dict[str, Any], 
                           simulation_id: str) -> Dict[str, Any]:
        """Execute individual attack step"""
        step_type = step.get("type", "unknown")
        target = step.get("target", "")
        
        if step_type == "port_scan":
            return self._simulate_port_scan(step, containers)
        elif step_type == "exploit_vulnerability":
            return self._simulate_exploit(step, containers)
        elif step_type == "lateral_movement":
            return self._simulate_lateral_movement(step, containers)
        elif step_type == "data_exfiltration":
            return self._simulate_data_exfiltration(step, containers)
        else:
            return {
                "success": True,
                "detected": False,
                "details": f"Simulated {step_type} attack on {target}"
            }
    
    def _simulate_port_scan(self, step: Dict[str, Any], containers: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate port scanning attack"""
        target = step.get("target", "")
        ports = step.get("ports", [80, 443, 22])
        
        if target not in containers:
            return {"success": False, "error": f"Target {target} not found"}
        
        target_ip = containers[target]["ip_address"]
        open_ports = [port for port in ports if port in [80, 443, 22]]
        detected = len(ports) > 10 or 22 in ports
        
        return {
            "success": True,
            "detected": detected,
            "detection_method": "IDS signature" if detected else None,
            "confidence": 0.8 if detected else 0.0,
            "open_ports": open_ports,
            "target_ip": target_ip
        }
    
    def _simulate_exploit(self, step: Dict[str, Any], containers: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate vulnerability exploitation"""
        target = step.get("target", "")
        vulnerability = step.get("vulnerability", "")
        
        if target not in containers:
            return {"success": False, "error": f"Target {target} not found"}
        
        success_rate = {
            "buffer_overflow": 0.7,
            "sql_injection": 0.8,
            "rce": 0.9,
            "privilege_escalation": 0.6
        }.get(vulnerability, 0.5)
        
        success = success_rate > 0.5
        detected = success and success_rate > 0.7
        
        return {
            "success": success,
            "detected": detected,
            "detection_method": "behavioral analysis" if detected else None,
            "confidence": 0.9 if detected else 0.0,
            "vulnerability": vulnerability,
            "impact": "high" if success else "none"
        }
    
    def _simulate_lateral_movement(self, step: Dict[str, Any], containers: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate lateral movement"""
        source = step.get("source", "")
        target = step.get("target", "")
        method = step.get("method", "ssh")
        
        success = True
        detected = method in ["rdp", "wmi"]
        
        return {
            "success": success,
            "detected": detected,
            "detection_method": "network monitoring" if detected else None,
            "confidence": 0.7 if detected else 0.0,
            "method": method,
            "path": f"{source} -> {target}"
        }
    
    def _simulate_data_exfiltration(self, step: Dict[str, Any], containers: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate data exfiltration"""
        target = step.get("target", "")
        data_type = step.get("data_type", "files")
        size_mb = step.get("size_mb", 100)
        
        detected = size_mb > 500 or data_type == "database"
        
        return {
            "success": True,
            "detected": detected,
            "detection_method": "DLP system" if detected else None,
            "confidence": 0.85 if detected else 0.0,
            "data_type": data_type,
            "size_mb": size_mb
        }
    
    def _calculate_success_rate(self, scenario: AttackScenario, attack_results: Dict[str, Any]) -> float:
        """Calculate attack scenario success rate"""
        total_steps = len(scenario.attack_steps)
        successful_steps = sum(1 for log in attack_results["execution_log"] if log["status"] == "success")
        return successful_steps / total_steps if total_steps > 0 else 0.0
    
    def _calculate_precision_score(self, scenario: AttackScenario, attack_results: Dict[str, Any]) -> float:
        """Calculate detection precision score"""
        expected_detections = len([step for step in scenario.attack_steps if step.get("should_detect", False)])
        actual_detections = len(attack_results["detected_attacks"])
        
        if expected_detections == 0:
            return 1.0 if actual_detections == 0 else 0.0
        
        return min(actual_detections / expected_detections, 1.0)
    
    def _generate_simulation_report(self, snapshot: NetworkSnapshot, 
                                  scenario: AttackScenario, 
                                  attack_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive simulation report"""
        return {
            "summary": {
                "snapshot_name": snapshot.name,
                "scenario_name": scenario.name,
                "total_steps": len(scenario.attack_steps),
                "successful_steps": sum(1 for log in attack_results["execution_log"] if log["status"] == "success"),
                "detected_attacks": len(attack_results["detected_attacks"]),
                "simulation_duration": sum(log.get("duration_seconds", 0) for log in attack_results["execution_log"])
            },
            "attack_chain": [
                {
                    "step": log["step_number"],
                    "name": log["step_name"],
                    "status": log["status"],
                    "detected": any(d["step_number"] == log["step_number"] for d in attack_results["detected_attacks"])
                }
                for log in attack_results["execution_log"]
            ],
            "detection_analysis": {
                "total_detections": len(attack_results["detected_attacks"]),
                "detection_methods": list(set(d.get("detection_method", "unknown") for d in attack_results["detected_attacks"])),
                "high_confidence_detections": len([d for d in attack_results["detected_attacks"] if d.get("confidence", 0) > 0.8])
            },
            "recommendations": self._generate_recommendations(attack_results)
        }
    
    def _generate_recommendations(self, attack_results: Dict[str, Any]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        detected_count = len(attack_results["detected_attacks"])
        total_steps = len(attack_results["execution_log"])
        
        if detected_count < total_steps * 0.5:
            recommendations.append("Consider implementing additional network monitoring solutions")
        
        if any("port_scan" in log.get("step_name", "") for log in attack_results["execution_log"]):
            recommendations.append("Deploy intrusion detection system to catch port scanning activities")
        
        if any("lateral_movement" in log.get("step_name", "") for log in attack_results["execution_log"]):
            recommendations.append("Implement network segmentation to limit lateral movement")
        
        return recommendations
    
    def _get_service_image(self, service: Dict[str, Any]) -> str:
        """Get Docker image for service type"""
        service_images = {
            "web_server": "nginx:alpine",
            "database": "postgres:13-alpine",
            "ssh_server": "linuxserver/openssh-server",
            "ftp_server": "stilliard/pure-ftpd"
        }
        return service_images.get(service.get("type", "web_server"), "alpine:latest")
    
    def _get_container_ip(self, container, network_name: str) -> str:
        """Get container IP address"""
        try:
            container.reload()
            return container.attrs["NetworkSettings"]["Networks"][network_name]["IPAddress"]
        except:
            return "unknown"
    
    def _cleanup_simulation(self, simulation_id: str):
        """Cleanup simulation containers and networks"""
        try:
            containers = self.docker_client.containers.list(
                all=True, filters={"label": f"simulation_id={simulation_id}"}
            )
            
            for container in containers:
                try:
                    container.remove(force=True)
                    logger.info(f"Removed container {container.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove container: {e}")
            
            try:
                network = self.docker_client.networks.get(f"sim_network_{simulation_id}")
                network.remove()
                logger.info(f"Removed network sim_network_{simulation_id}")
            except Exception as e:
                logger.warning(f"Failed to remove network: {e}")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def _store_simulation_result(self, result: SimulationResult):
        """Store simulation result in database"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO simulation_results 
                        (simulation_id, snapshot_id, scenario_id, status, start_time, end_time,
                         execution_log, detected_attacks, success_rate, precision_score, report_data)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (simulation_id) DO UPDATE SET
                            status = EXCLUDED.status,
                            end_time = EXCLUDED.end_time,
                            execution_log = EXCLUDED.execution_log,
                            detected_attacks = EXCLUDED.detected_attacks,
                            success_rate = EXCLUDED.success_rate,
                            precision_score = EXCLUDED.precision_score,
                            report_data = EXCLUDED.report_data
                    """, (
                        result.simulation_id, result.snapshot_id, result.scenario_id,
                        result.status, result.start_time, result.end_time,
                        json.dumps(result.execution_log),
                        json.dumps(result.detected_attacks),
                        result.success_rate, result.precision_score,
                        json.dumps(result.report_data)
                    ))
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to store simulation result: {e}")
    
    def get_simulation_result(self, simulation_id: str) -> Optional[SimulationResult]:
        """Get simulation result from database"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM simulation_results WHERE simulation_id = %s", (simulation_id,))
                    row = cur.fetchone()
                    if not row:
                        return None
                    
                    return SimulationResult(
                        simulation_id=row['simulation_id'],
                        snapshot_id=row['snapshot_id'],
                        scenario_id=row['scenario_id'],
                        status=row['status'],
                        start_time=row['start_time'],
                        end_time=row['end_time'],
                        execution_log=row['execution_log'],
                        detected_attacks=row['detected_attacks'],
                        success_rate=row['success_rate'],
                        precision_score=row['precision_score'],
                        report_data=row['report_data']
                    )
        except Exception as e:
            logger.error(f"Failed to get simulation result: {e}")
            return None