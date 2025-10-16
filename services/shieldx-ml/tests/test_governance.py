"""
Comprehensive tests for Model Governance System
"""

import pytest
import time
import json
import tempfile
from pathlib import Path
from governance.model_governance import (
    ModelGovernance,
    ModelLineageTracker,
    ComplianceChecker,
    AuditLogger,
    ModelMetadata,
    AuditEvent,
    ComplianceStandard,
    ModelStage,
    AuditEventType
)


class TestModelMetadata:
    """Test ModelMetadata dataclass"""
    
    def test_initialization(self):
        """Test metadata initialization"""
        metadata = ModelMetadata(
            model_id="model_001",
            name="test_model",
            version="v1.0",
            framework="pytorch",
            algorithm="isolation_forest"
        )
        
        assert metadata.model_id == "model_001"
        assert metadata.name == "test_model"
        assert metadata.version == "v1.0"
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        metadata = ModelMetadata(
            model_id="model_001",
            name="test_model",
            version="v1.0",
            framework="pytorch",
            algorithm="isolation_forest"
        )
        
        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data['model_id'] == "model_001"
    
    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            'model_id': "model_001",
            'name': "test_model",
            'version': "v1.0",
            'framework': "pytorch",
            'algorithm': "isolation_forest",
            'training_data_id': "",
            'training_code_hash': "",
            'created_by': "system",
            'stage': "development"
        }
        
        metadata = ModelMetadata.from_dict(data)
        assert metadata.model_id == "model_001"
        assert metadata.name == "test_model"


class TestModelLineageTracker:
    """Test ModelLineageTracker class"""
    
    def test_initialization(self):
        """Test tracker initialization"""
        tracker = ModelLineageTracker()
        assert len(tracker.models) == 0
        assert len(tracker.lineage_graph) == 0
    
    def test_register_model(self):
        """Test registering a model"""
        tracker = ModelLineageTracker()
        
        metadata = ModelMetadata(
            model_id="model_001",
            name="test_model",
            version="v1.0",
            framework="pytorch",
            algorithm="cnn"
        )
        
        tracker.register_model(metadata)
        assert "model_001" in tracker.models
    
    def test_get_model(self):
        """Test retrieving a model"""
        tracker = ModelLineageTracker()
        
        metadata = ModelMetadata(
            model_id="model_001",
            name="test_model",
            version="v1.0",
            framework="pytorch",
            algorithm="cnn"
        )
        
        tracker.register_model(metadata)
        retrieved = tracker.get_model("model_001")
        
        assert retrieved is not None
        assert retrieved.model_id == "model_001"
    
    def test_lineage_tracking(self):
        """Test lineage tracking"""
        tracker = ModelLineageTracker()
        
        # Parent model
        parent = ModelMetadata(
            model_id="model_parent",
            name="parent_model",
            version="v1.0",
            framework="pytorch",
            algorithm="base"
        )
        tracker.register_model(parent)
        
        # Child model
        child = ModelMetadata(
            model_id="model_child",
            name="child_model",
            version="v2.0",
            framework="pytorch",
            algorithm="improved",
            parent_model_id="model_parent"
        )
        tracker.register_model(child)
        
        # Check lineage graph
        assert "model_parent" in tracker.lineage_graph
        assert "model_child" in tracker.lineage_graph["model_parent"]
    
    def test_get_ancestors(self):
        """Test getting ancestor models"""
        tracker = ModelLineageTracker()
        
        # Create lineage: grandparent -> parent -> child
        grandparent = ModelMetadata(
            model_id="model_gp",
            name="grandparent",
            version="v1.0",
            framework="pytorch",
            algorithm="base"
        )
        tracker.register_model(grandparent)
        
        parent = ModelMetadata(
            model_id="model_p",
            name="parent",
            version="v2.0",
            framework="pytorch",
            algorithm="improved",
            parent_model_id="model_gp"
        )
        tracker.register_model(parent)
        
        child = ModelMetadata(
            model_id="model_c",
            name="child",
            version="v3.0",
            framework="pytorch",
            algorithm="best",
            parent_model_id="model_p"
        )
        tracker.register_model(child)
        
        ancestors = tracker.get_ancestors("model_c")
        assert len(ancestors) == 2
        assert ancestors[0].model_id == "model_p"
        assert ancestors[1].model_id == "model_gp"
    
    def test_get_descendants(self):
        """Test getting descendant models"""
        tracker = ModelLineageTracker()
        
        # Create lineage
        parent = ModelMetadata(
            model_id="model_p",
            name="parent",
            version="v1.0",
            framework="pytorch",
            algorithm="base"
        )
        tracker.register_model(parent)
        
        child1 = ModelMetadata(
            model_id="model_c1",
            name="child1",
            version="v2.0",
            framework="pytorch",
            algorithm="variant1",
            parent_model_id="model_p"
        )
        tracker.register_model(child1)
        
        child2 = ModelMetadata(
            model_id="model_c2",
            name="child2",
            version="v2.1",
            framework="pytorch",
            algorithm="variant2",
            parent_model_id="model_p"
        )
        tracker.register_model(child2)
        
        descendants = tracker.get_descendants("model_p")
        assert len(descendants) == 2
        assert any(d.model_id == "model_c1" for d in descendants)
        assert any(d.model_id == "model_c2" for d in descendants)
    
    def test_get_lineage_path(self):
        """Test getting lineage path"""
        tracker = ModelLineageTracker()
        
        # Create lineage chain
        for i in range(3):
            model = ModelMetadata(
                model_id=f"model_{i}",
                name=f"model_{i}",
                version=f"v{i}.0",
                framework="pytorch",
                algorithm="base",
                parent_model_id=f"model_{i-1}" if i > 0 else None
            )
            tracker.register_model(model)
        
        path = tracker.get_lineage_path("model_2")
        assert len(path) == 3
        assert path == ["model_0", "model_1", "model_2"]
    
    def test_get_lineage_tree(self):
        """Test getting lineage tree structure"""
        tracker = ModelLineageTracker()
        
        # Create tree structure
        parent = ModelMetadata(
            model_id="parent",
            name="parent",
            version="v1.0",
            framework="pytorch",
            algorithm="base"
        )
        tracker.register_model(parent)
        
        child = ModelMetadata(
            model_id="child",
            name="child",
            version="v2.0",
            framework="pytorch",
            algorithm="improved",
            parent_model_id="parent"
        )
        tracker.register_model(child)
        
        tree = tracker.get_lineage_tree("parent")
        assert tree['model_id'] == "parent"
        assert len(tree['children']) == 1
        assert tree['children'][0]['model_id'] == "child"


class TestComplianceChecker:
    """Test ComplianceChecker class"""
    
    def test_initialization(self):
        """Test checker initialization"""
        checker = ComplianceChecker()
        assert ComplianceStandard.GDPR.value in checker.rules
        assert ComplianceStandard.SOC2.value in checker.rules
        assert ComplianceStandard.HIPAA.value in checker.rules
    
    def test_gdpr_compliance(self):
        """Test GDPR compliance checking"""
        checker = ComplianceChecker()
        
        # Model with good GDPR compliance
        model = ModelMetadata(
            model_id="model_001",
            name="compliant_model",
            version="v1.0",
            framework="pytorch",
            algorithm="cnn",
            data_privacy_level="high",
            tags=['explainable', 'shap', 'consent_tracked'],
            feature_names=['feature1', 'feature2']
        )
        
        result = checker.check_compliance(model, ComplianceStandard.GDPR)
        assert result['standard'] == ComplianceStandard.GDPR.value
        assert result['passed'] == True
        assert result['score'] > 0.8
    
    def test_gdpr_non_compliance(self):
        """Test GDPR non-compliance detection"""
        checker = ComplianceChecker()
        
        # Model with poor GDPR compliance
        model = ModelMetadata(
            model_id="model_002",
            name="non_compliant_model",
            version="v1.0",
            framework="pytorch",
            algorithm="cnn",
            data_privacy_level="",
            tags=[],
            feature_names=[]
        )
        
        result = checker.check_compliance(model, ComplianceStandard.GDPR)
        assert result['passed'] == False
        assert result['score'] < 0.5
    
    def test_soc2_compliance(self):
        """Test SOC2 compliance checking"""
        checker = ComplianceChecker()
        
        model = ModelMetadata(
            model_id="model_003",
            name="secure_model",
            version="v1.0",
            framework="pytorch",
            algorithm="cnn",
            tags=['access_logged', 'encrypted', 'backup_enabled', 'incident_response']
        )
        
        result = checker.check_compliance(model, ComplianceStandard.SOC2)
        assert result['standard'] == ComplianceStandard.SOC2.value
        assert result['passed'] == True
    
    def test_hipaa_compliance(self):
        """Test HIPAA compliance checking"""
        checker = ComplianceChecker()
        
        model = ModelMetadata(
            model_id="model_004",
            name="healthcare_model",
            version="v1.0",
            framework="pytorch",
            algorithm="cnn",
            data_privacy_level="high",
            tags=['audit_enabled', 'access_control', 'tls_required']
        )
        
        result = checker.check_compliance(model, ComplianceStandard.HIPAA)
        assert result['standard'] == ComplianceStandard.HIPAA.value
        # May not pass all checks but should return result
        assert 'checks' in result
        assert len(result['checks']) > 0


class TestAuditLogger:
    """Test AuditLogger class"""
    
    def test_initialization(self):
        """Test logger initialization"""
        logger = AuditLogger()
        assert len(logger.events) == 0
    
    def test_log_event(self):
        """Test logging an event"""
        logger = AuditLogger()
        
        event = logger.log_event(
            event_type=AuditEventType.MODEL_CREATED,
            actor="test_user",
            model_id="model_001",
            model_version="v1.0",
            action="create_model"
        )
        
        assert event is not None
        assert event.event_type == AuditEventType.MODEL_CREATED.value
        assert event.actor == "test_user"
        assert len(logger.events) == 1
    
    def test_log_to_file(self):
        """Test logging to file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            log_file = f.name
        
        try:
            logger = AuditLogger(log_file=log_file)
            
            logger.log_event(
                event_type=AuditEventType.MODEL_DEPLOYED,
                actor="deployer",
                model_id="model_001",
                action="deploy"
            )
            
            # Read back from file
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 1
            event_data = json.loads(lines[0])
            assert event_data['event_type'] == AuditEventType.MODEL_DEPLOYED.value
        finally:
            Path(log_file).unlink()
    
    def test_query_events(self):
        """Test querying events"""
        logger = AuditLogger()
        
        # Log multiple events
        logger.log_event(AuditEventType.MODEL_CREATED, "user1", model_id="model_001")
        logger.log_event(AuditEventType.MODEL_DEPLOYED, "user2", model_id="model_001")
        logger.log_event(AuditEventType.MODEL_CREATED, "user1", model_id="model_002")
        
        # Query by model_id
        events = logger.get_events(model_id="model_001")
        assert len(events) == 2
        
        # Query by actor
        events = logger.get_events(actor="user1")
        assert len(events) == 2
        
        # Query by event_type
        events = logger.get_events(event_type=AuditEventType.MODEL_CREATED)
        assert len(events) == 2
    
    def test_event_filtering(self):
        """Test event filtering by time"""
        logger = AuditLogger()
        
        now = time.time()
        logger.log_event(AuditEventType.MODEL_CREATED, "user1", model_id="model_001")
        time.sleep(0.1)
        logger.log_event(AuditEventType.MODEL_DEPLOYED, "user2", model_id="model_002")
        
        # Query events after first event
        events = logger.get_events(start_time=now + 0.05)
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.MODEL_DEPLOYED.value
    
    def test_get_event_summary(self):
        """Test event summary"""
        logger = AuditLogger()
        
        # Log various events
        logger.log_event(AuditEventType.MODEL_CREATED, "user1", success=True)
        logger.log_event(AuditEventType.MODEL_CREATED, "user2", success=True)
        logger.log_event(AuditEventType.MODEL_DEPLOYED, "user1", success=False)
        
        summary = logger.get_event_summary()
        assert summary['total_events'] == 3
        assert summary['failed_events'] == 1
        assert summary['success_rate'] == 2/3
        assert AuditEventType.MODEL_CREATED.value in summary['events_by_type']


class TestModelGovernance:
    """Test ModelGovernance main class"""
    
    def test_initialization(self):
        """Test governance initialization"""
        governance = ModelGovernance()
        assert governance.lineage is not None
        assert governance.compliance is not None
        assert governance.audit is not None
    
    def test_register_model(self):
        """Test registering a model"""
        governance = ModelGovernance()
        
        metadata = ModelMetadata(
            model_id="model_001",
            name="test_model",
            version="v1.0",
            framework="pytorch",
            algorithm="cnn"
        )
        
        model_id = governance.register_model(metadata, actor="test_user")
        assert model_id == "model_001"
        
        # Check audit log
        events = governance.audit.get_events(model_id="model_001")
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.MODEL_CREATED.value
    
    def test_check_compliance(self):
        """Test checking compliance"""
        governance = ModelGovernance()
        
        metadata = ModelMetadata(
            model_id="model_001",
            name="test_model",
            version="v1.0",
            framework="pytorch",
            algorithm="cnn",
            data_privacy_level="high",
            tags=['explainable', 'shap', 'consent_tracked'],
            feature_names=['feature1']
        )
        
        governance.register_model(metadata)
        
        result = governance.check_compliance(
            "model_001",
            [ComplianceStandard.GDPR],
            actor="compliance_officer"
        )
        
        assert 'overall_passed' in result
        assert ComplianceStandard.GDPR.value in result['standards']
    
    def test_promote_model(self):
        """Test promoting a model"""
        governance = ModelGovernance()
        
        metadata = ModelMetadata(
            model_id="model_001",
            name="test_model",
            version="v1.0",
            framework="pytorch",
            algorithm="cnn",
            stage=ModelStage.DEVELOPMENT.value,
            data_privacy_level="high",
            tags=['explainable', 'shap', 'consent_tracked', 'access_logged', 
                  'encrypted', 'backup_enabled', 'incident_response'],
            feature_names=['feature1']
        )
        
        governance.register_model(metadata)
        
        # Promote to staging (no compliance required)
        success = governance.promote_model(
            "model_001",
            ModelStage.STAGING,
            actor="engineer",
            require_compliance=False
        )
        assert success == True
        
        # Check stage updated
        model = governance.lineage.get_model("model_001")
        assert model.stage == ModelStage.STAGING.value
    
    def test_promote_model_with_compliance(self):
        """Test promoting model with compliance check"""
        governance = ModelGovernance()
        
        # Non-compliant model
        metadata = ModelMetadata(
            model_id="model_002",
            name="test_model",
            version="v1.0",
            framework="pytorch",
            algorithm="cnn",
            stage=ModelStage.DEVELOPMENT.value
        )
        
        governance.register_model(metadata)
        
        # Try to promote to production (should fail)
        success = governance.promote_model(
            "model_002",
            ModelStage.PRODUCTION,
            actor="engineer",
            require_compliance=True
        )
        assert success == False
    
    def test_get_model_history(self):
        """Test getting model history"""
        governance = ModelGovernance()
        
        # Parent model
        parent = ModelMetadata(
            model_id="model_parent",
            name="parent_model",
            version="v1.0",
            framework="pytorch",
            algorithm="base"
        )
        governance.register_model(parent, actor="user1")
        
        # Child model
        child = ModelMetadata(
            model_id="model_child",
            name="child_model",
            version="v2.0",
            framework="pytorch",
            algorithm="improved",
            parent_model_id="model_parent"
        )
        governance.register_model(child, actor="user2")
        
        # Get history
        history = governance.get_model_history("model_child")
        
        assert 'model' in history
        assert 'lineage' in history
        assert 'audit_trail' in history
        assert len(history['lineage']['ancestors']) == 1
    
    def test_generate_model_card(self):
        """Test generating model card"""
        governance = ModelGovernance()
        
        metadata = ModelMetadata(
            model_id="model_001",
            name="production_model",
            version="v1.0",
            framework="pytorch",
            algorithm="transformer",
            description="Advanced threat detection model",
            use_cases=["malware detection", "anomaly detection"],
            limitations=["Requires labeled data", "High compute cost"],
            accuracy=0.95,
            precision=0.93,
            recall=0.97,
            f1_score=0.95,
            hyperparameters={'learning_rate': 0.001, 'epochs': 100},
            feature_names=['feature1', 'feature2', 'feature3']
        )
        
        governance.register_model(metadata)
        
        card = governance.generate_model_card("model_001")
        
        assert 'model_details' in card
        assert 'intended_use' in card
        assert 'performance' in card
        assert 'technical_details' in card
        assert 'limitations' in card
        assert 'compliance' in card
        assert 'lineage' in card
        
        assert card['model_details']['name'] == "production_model"
        assert card['performance']['accuracy'] == 0.95
        assert len(card['limitations']) == 2


class TestIntegration:
    """Integration tests for governance system"""
    
    def test_full_governance_workflow(self):
        """Test complete governance workflow"""
        governance = ModelGovernance()
        
        # 1. Register initial model
        model_v1 = ModelMetadata(
            model_id="threat_detector_v1",
            name="threat_detector",
            version="v1.0",
            framework="pytorch",
            algorithm="cnn",
            description="Initial threat detection model",
            use_cases=["malware detection"],
            accuracy=0.85,
            data_privacy_level="high",
            tags=['explainable', 'shap'],
            feature_names=['packet_size', 'protocol', 'port']
        )
        governance.register_model(model_v1, actor="ml_engineer")
        
        # 2. Check compliance
        compliance = governance.check_compliance(
            "threat_detector_v1",
            [ComplianceStandard.GDPR],
            actor="compliance_officer"
        )
        assert compliance['overall_passed'] == True
        
        # 3. Promote to staging
        governance.promote_model(
            "threat_detector_v1",
            ModelStage.STAGING,
            actor="ml_engineer",
            require_compliance=False
        )
        
        # 4. Register improved model
        model_v2 = ModelMetadata(
            model_id="threat_detector_v2",
            name="threat_detector",
            version="v2.0",
            framework="pytorch",
            algorithm="transformer",
            description="Improved threat detection model",
            use_cases=["malware detection", "anomaly detection"],
            accuracy=0.95,
            parent_model_id="threat_detector_v1",
            data_privacy_level="high",
            tags=['explainable', 'shap', 'consent_tracked', 'access_logged',
                  'encrypted', 'backup_enabled', 'incident_response'],
            feature_names=['packet_size', 'protocol', 'port', 'payload_hash']
        )
        governance.register_model(model_v2, actor="ml_engineer")
        
        # 5. Check lineage
        history = governance.get_model_history("threat_detector_v2")
        assert len(history['lineage']['ancestors']) == 1
        assert history['lineage']['ancestors'][0]['model_id'] == "threat_detector_v1"
        
        # 6. Generate model card
        card = governance.generate_model_card("threat_detector_v2")
        assert card['model_details']['version'] == "v2.0"
        assert card['performance']['accuracy'] == 0.95
        
        # 7. Promote to production
        success = governance.promote_model(
            "threat_detector_v2",
            ModelStage.PRODUCTION,
            actor="lead_engineer",
            require_compliance=True
        )
        assert success == True
        
        # 8. Verify audit trail
        events = governance.audit.get_events(model_id="threat_detector_v2")
        assert len(events) >= 3  # Created, compliance check, promoted


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
