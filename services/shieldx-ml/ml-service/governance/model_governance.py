"""
Model Governance System
Provides model lineage tracking, compliance validation, audit logging, and documentation
"""

import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Compliance standards"""
    GDPR = "GDPR"  # EU General Data Protection Regulation
    SOC2 = "SOC2"  # Service Organization Control 2
    HIPAA = "HIPAA"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "PCI_DSS"  # Payment Card Industry Data Security Standard
    ISO_27001 = "ISO_27001"  # Information Security Management
    CCPA = "CCPA"  # California Consumer Privacy Act


class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class AuditEventType(Enum):
    """Types of audit events"""
    MODEL_CREATED = "model_created"
    MODEL_TRAINED = "model_trained"
    MODEL_EVALUATED = "model_evaluated"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_PROMOTED = "model_promoted"
    MODEL_DEPRECATED = "model_deprecated"
    MODEL_ACCESSED = "model_accessed"
    MODEL_MODIFIED = "model_modified"
    MODEL_DELETED = "model_deleted"
    DATA_ACCESSED = "data_accessed"
    COMPLIANCE_CHECK = "compliance_check"
    SECURITY_SCAN = "security_scan"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    name: str
    version: str
    framework: str  # pytorch, tensorflow, sklearn
    algorithm: str
    
    # Lineage
    parent_model_id: Optional[str] = None
    training_data_id: str = ""
    training_code_hash: str = ""
    
    # Performance
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    created_by: str = "system"
    stage: str = ModelStage.DEVELOPMENT.value
    tags: List[str] = field(default_factory=list)
    
    # Compliance
    compliance_standards: List[str] = field(default_factory=list)
    data_privacy_level: str = "medium"  # low, medium, high
    
    # Documentation
    description: str = ""
    use_cases: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Technical details
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_metrics: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class AuditEvent:
    """Audit log event"""
    event_id: str
    timestamp: float
    event_type: str
    actor: str  # User or service performing action
    
    # Event details
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    resource_type: str = ""
    resource_id: str = ""
    action: str = ""
    
    # Context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Result
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary"""
        return cls(**data)


class ModelLineageTracker:
    """Track model lineage and provenance"""
    
    def __init__(self):
        self.models: Dict[str, ModelMetadata] = {}
        self.lineage_graph: Dict[str, List[str]] = {}  # parent_id -> child_ids
        self._lock = threading.Lock()
        
    def register_model(self, metadata: ModelMetadata):
        """Register a new model"""
        with self._lock:
            self.models[metadata.model_id] = metadata
            
            # Update lineage graph
            if metadata.parent_model_id:
                if metadata.parent_model_id not in self.lineage_graph:
                    self.lineage_graph[metadata.parent_model_id] = []
                self.lineage_graph[metadata.parent_model_id].append(metadata.model_id)
            
            logger.info(f"Registered model: {metadata.name}@{metadata.version}")
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata"""
        with self._lock:
            return self.models.get(model_id)
    
    def get_ancestors(self, model_id: str) -> List[ModelMetadata]:
        """Get all ancestor models"""
        ancestors = []
        current_id = model_id
        
        with self._lock:
            while current_id:
                model = self.models.get(current_id)
                if not model:
                    break
                
                if model.parent_model_id:
                    parent = self.models.get(model.parent_model_id)
                    if parent:
                        ancestors.append(parent)
                    current_id = model.parent_model_id
                else:
                    break
        
        return ancestors
    
    def get_descendants(self, model_id: str) -> List[ModelMetadata]:
        """Get all descendant models"""
        descendants = []
        queue = [model_id]
        
        with self._lock:
            while queue:
                current_id = queue.pop(0)
                children_ids = self.lineage_graph.get(current_id, [])
                
                for child_id in children_ids:
                    child = self.models.get(child_id)
                    if child:
                        descendants.append(child)
                        queue.append(child_id)
        
        return descendants
    
    def get_lineage_path(self, model_id: str) -> List[str]:
        """Get full lineage path from root to model"""
        path = []
        current_id = model_id
        
        with self._lock:
            while current_id:
                model = self.models.get(current_id)
                if not model:
                    break
                
                path.insert(0, current_id)
                current_id = model.parent_model_id
        
        return path
    
    def get_lineage_tree(self, model_id: str) -> Dict[str, Any]:
        """Get lineage tree structure"""
        with self._lock:
            model = self.models.get(model_id)
            if not model:
                return {}
            
            tree = {
                'model_id': model_id,
                'name': model.name,
                'version': model.version,
                'stage': model.stage,
                'children': []
            }
            
            # Add children recursively
            children_ids = self.lineage_graph.get(model_id, [])
            for child_id in children_ids:
                child_tree = self.get_lineage_tree(child_id)
                if child_tree:
                    tree['children'].append(child_tree)
            
            return tree


class ComplianceChecker:
    """Check model compliance with regulations"""
    
    def __init__(self):
        self.rules: Dict[str, List[callable]] = {}
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Setup default compliance rules"""
        # GDPR rules
        self.rules[ComplianceStandard.GDPR.value] = [
            self._check_data_privacy,
            self._check_right_to_explanation,
            self._check_data_minimization,
            self._check_consent_tracking
        ]
        
        # SOC2 rules
        self.rules[ComplianceStandard.SOC2.value] = [
            self._check_access_logging,
            self._check_data_encryption,
            self._check_backup_procedures,
            self._check_incident_response
        ]
        
        # HIPAA rules
        self.rules[ComplianceStandard.HIPAA.value] = [
            self._check_phi_protection,
            self._check_audit_controls,
            self._check_access_controls,
            self._check_transmission_security
        ]
    
    def check_compliance(self, 
                        model: ModelMetadata,
                        standard: ComplianceStandard) -> Dict[str, Any]:
        """Check model compliance with standard"""
        rules = self.rules.get(standard.value, [])
        results = []
        
        for rule in rules:
            try:
                result = rule(model)
                results.append(result)
            except Exception as e:
                logger.error(f"Compliance check failed: {rule.__name__}: {e}")
                results.append({
                    'rule': rule.__name__,
                    'passed': False,
                    'message': str(e)
                })
        
        passed = all(r['passed'] for r in results)
        
        return {
            'standard': standard.value,
            'model_id': model.model_id,
            'model_version': model.version,
            'passed': passed,
            'timestamp': time.time(),
            'checks': results,
            'score': sum(1 for r in results if r['passed']) / len(results) if results else 0.0
        }
    
    # GDPR compliance checks
    def _check_data_privacy(self, model: ModelMetadata) -> Dict[str, Any]:
        """Check data privacy requirements"""
        has_privacy_level = bool(model.data_privacy_level)
        return {
            'rule': 'data_privacy',
            'passed': has_privacy_level,
            'message': 'Data privacy level documented' if has_privacy_level else 'Missing data privacy level'
        }
    
    def _check_right_to_explanation(self, model: ModelMetadata) -> Dict[str, Any]:
        """Check right to explanation (explainability)"""
        # Check if model has explainability features
        has_explanation = 'explainable' in model.tags or 'shap' in model.tags or 'lime' in model.tags
        return {
            'rule': 'right_to_explanation',
            'passed': has_explanation,
            'message': 'Model supports explanations' if has_explanation else 'Model lacks explainability'
        }
    
    def _check_data_minimization(self, model: ModelMetadata) -> Dict[str, Any]:
        """Check data minimization principle"""
        # Check if feature list is documented
        has_features = len(model.feature_names) > 0
        return {
            'rule': 'data_minimization',
            'passed': has_features,
            'message': f'{len(model.feature_names)} features documented' if has_features else 'Features not documented'
        }
    
    def _check_consent_tracking(self, model: ModelMetadata) -> Dict[str, Any]:
        """Check consent tracking"""
        has_consent = 'consent_tracked' in model.tags
        return {
            'rule': 'consent_tracking',
            'passed': has_consent,
            'message': 'Consent tracking enabled' if has_consent else 'Consent tracking not documented'
        }
    
    # SOC2 compliance checks
    def _check_access_logging(self, model: ModelMetadata) -> Dict[str, Any]:
        """Check access logging"""
        has_logging = 'access_logged' in model.tags
        return {
            'rule': 'access_logging',
            'passed': has_logging,
            'message': 'Access logging enabled' if has_logging else 'Access logging not configured'
        }
    
    def _check_data_encryption(self, model: ModelMetadata) -> Dict[str, Any]:
        """Check data encryption"""
        has_encryption = 'encrypted' in model.tags
        return {
            'rule': 'data_encryption',
            'passed': has_encryption,
            'message': 'Data encryption enabled' if has_encryption else 'Data encryption not configured'
        }
    
    def _check_backup_procedures(self, model: ModelMetadata) -> Dict[str, Any]:
        """Check backup procedures"""
        has_backup = 'backup_enabled' in model.tags
        return {
            'rule': 'backup_procedures',
            'passed': has_backup,
            'message': 'Backup procedures configured' if has_backup else 'No backup procedures'
        }
    
    def _check_incident_response(self, model: ModelMetadata) -> Dict[str, Any]:
        """Check incident response plan"""
        has_response = 'incident_response' in model.tags
        return {
            'rule': 'incident_response',
            'passed': has_response,
            'message': 'Incident response plan exists' if has_response else 'No incident response plan'
        }
    
    # HIPAA compliance checks
    def _check_phi_protection(self, model: ModelMetadata) -> Dict[str, Any]:
        """Check PHI (Protected Health Information) protection"""
        has_phi_protection = model.data_privacy_level == 'high'
        return {
            'rule': 'phi_protection',
            'passed': has_phi_protection,
            'message': 'PHI protection enabled' if has_phi_protection else 'PHI protection insufficient'
        }
    
    def _check_audit_controls(self, model: ModelMetadata) -> Dict[str, Any]:
        """Check audit controls"""
        has_audit = 'audit_enabled' in model.tags
        return {
            'rule': 'audit_controls',
            'passed': has_audit,
            'message': 'Audit controls enabled' if has_audit else 'Audit controls not configured'
        }
    
    def _check_access_controls(self, model: ModelMetadata) -> Dict[str, Any]:
        """Check access controls"""
        has_access_control = 'access_control' in model.tags
        return {
            'rule': 'access_controls',
            'passed': has_access_control,
            'message': 'Access controls configured' if has_access_control else 'Access controls missing'
        }
    
    def _check_transmission_security(self, model: ModelMetadata) -> Dict[str, Any]:
        """Check transmission security"""
        has_tls = 'tls_required' in model.tags
        return {
            'rule': 'transmission_security',
            'passed': has_tls,
            'message': 'TLS encryption required' if has_tls else 'TLS not required'
        }


class AuditLogger:
    """Audit logging system"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.events: List[AuditEvent] = []
        self._lock = threading.Lock()
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(self,
                  event_type: AuditEventType,
                  actor: str,
                  model_id: Optional[str] = None,
                  model_version: Optional[str] = None,
                  action: str = "",
                  metadata: Optional[Dict[str, Any]] = None,
                  success: bool = True,
                  error_message: Optional[str] = None):
        """Log an audit event"""
        event_id = self._generate_event_id()
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=time.time(),
            event_type=event_type.value,
            actor=actor,
            model_id=model_id,
            model_version=model_version,
            action=action,
            metadata=metadata or {},
            success=success,
            error_message=error_message
        )
        
        with self._lock:
            self.events.append(event)
            
            # Write to file if configured
            if self.log_file:
                self._write_to_file(event)
        
        logger.info(f"Audit event: {event_type.value} by {actor}")
        
        return event
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        data = f"{time.time()}{id(self)}".encode()
        return hashlib.sha256(data).hexdigest()[:16]
    
    def _write_to_file(self, event: AuditEvent):
        """Write event to log file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_events(self,
                   model_id: Optional[str] = None,
                   actor: Optional[str] = None,
                   event_type: Optional[AuditEventType] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   limit: int = 100) -> List[AuditEvent]:
        """Query audit events"""
        with self._lock:
            filtered = self.events
            
            if model_id:
                filtered = [e for e in filtered if e.model_id == model_id]
            
            if actor:
                filtered = [e for e in filtered if e.actor == actor]
            
            if event_type:
                filtered = [e for e in filtered if e.event_type == event_type.value]
            
            if start_time:
                filtered = [e for e in filtered if e.timestamp >= start_time]
            
            if end_time:
                filtered = [e for e in filtered if e.timestamp <= end_time]
            
            # Sort by timestamp descending
            filtered.sort(key=lambda e: e.timestamp, reverse=True)
            
            return filtered[:limit]
    
    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary of audit events"""
        with self._lock:
            total = len(self.events)
            by_type = {}
            by_actor = {}
            failed = 0
            
            for event in self.events:
                # Count by type
                by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
                
                # Count by actor
                by_actor[event.actor] = by_actor.get(event.actor, 0) + 1
                
                # Count failures
                if not event.success:
                    failed += 1
            
            return {
                'total_events': total,
                'failed_events': failed,
                'success_rate': (total - failed) / total if total > 0 else 0.0,
                'events_by_type': by_type,
                'events_by_actor': by_actor
            }


class ModelGovernance:
    """Comprehensive model governance system"""
    
    def __init__(self, audit_log_file: Optional[str] = None):
        """
        Initialize model governance system
        
        Args:
            audit_log_file: Path to audit log file
        """
        self.lineage = ModelLineageTracker()
        self.compliance = ComplianceChecker()
        self.audit = AuditLogger(audit_log_file)
        
        logger.info("Model Governance system initialized")
    
    def register_model(self,
                      metadata: ModelMetadata,
                      actor: str = "system") -> str:
        """Register a new model"""
        # Register in lineage tracker
        self.lineage.register_model(metadata)
        
        # Log audit event
        self.audit.log_event(
            event_type=AuditEventType.MODEL_CREATED,
            actor=actor,
            model_id=metadata.model_id,
            model_version=metadata.version,
            action="register_model",
            metadata={
                'name': metadata.name,
                'framework': metadata.framework,
                'algorithm': metadata.algorithm
            }
        )
        
        return metadata.model_id
    
    def check_compliance(self,
                        model_id: str,
                        standards: List[ComplianceStandard],
                        actor: str = "system") -> Dict[str, Any]:
        """Check model compliance"""
        model = self.lineage.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        
        results = {}
        all_passed = True
        
        for standard in standards:
            result = self.compliance.check_compliance(model, standard)
            results[standard.value] = result
            
            if not result['passed']:
                all_passed = False
            
            # Log audit event
            self.audit.log_event(
                event_type=AuditEventType.COMPLIANCE_CHECK,
                actor=actor,
                model_id=model_id,
                model_version=model.version,
                action=f"compliance_check_{standard.value}",
                metadata=result,
                success=result['passed']
            )
        
        return {
            'model_id': model_id,
            'model_version': model.version,
            'overall_passed': all_passed,
            'standards': results,
            'timestamp': time.time()
        }
    
    def promote_model(self,
                     model_id: str,
                     new_stage: ModelStage,
                     actor: str,
                     require_compliance: bool = True) -> bool:
        """Promote model to new stage"""
        model = self.lineage.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        
        # Check compliance if required
        if require_compliance and new_stage == ModelStage.PRODUCTION:
            compliance_result = self.check_compliance(
                model_id,
                [ComplianceStandard.GDPR, ComplianceStandard.SOC2],
                actor
            )
            
            if not compliance_result['overall_passed']:
                self.audit.log_event(
                    event_type=AuditEventType.MODEL_PROMOTED,
                    actor=actor,
                    model_id=model_id,
                    model_version=model.version,
                    action=f"promote_to_{new_stage.value}",
                    success=False,
                    error_message="Compliance checks failed"
                )
                return False
        
        # Update stage
        model.stage = new_stage.value
        
        # Log audit event
        self.audit.log_event(
            event_type=AuditEventType.MODEL_PROMOTED,
            actor=actor,
            model_id=model_id,
            model_version=model.version,
            action=f"promote_to_{new_stage.value}",
            metadata={'old_stage': model.stage, 'new_stage': new_stage.value}
        )
        
        logger.info(f"Model {model_id} promoted to {new_stage.value}")
        return True
    
    def get_model_history(self, model_id: str) -> Dict[str, Any]:
        """Get complete model history"""
        model = self.lineage.get_model(model_id)
        if not model:
            return {}
        
        # Get lineage
        ancestors = self.lineage.get_ancestors(model_id)
        descendants = self.lineage.get_descendants(model_id)
        lineage_tree = self.lineage.get_lineage_tree(model_id)
        
        # Get audit events
        events = self.audit.get_events(model_id=model_id, limit=1000)
        
        return {
            'model': model.to_dict(),
            'lineage': {
                'ancestors': [a.to_dict() for a in ancestors],
                'descendants': [d.to_dict() for d in descendants],
                'tree': lineage_tree
            },
            'audit_trail': [e.to_dict() for e in events],
            'total_events': len(events)
        }
    
    def generate_model_card(self, model_id: str) -> Dict[str, Any]:
        """Generate model card documentation"""
        model = self.lineage.get_model(model_id)
        if not model:
            return {}
        
        return {
            'model_details': {
                'name': model.name,
                'version': model.version,
                'model_id': model.model_id,
                'framework': model.framework,
                'algorithm': model.algorithm,
                'created_at': datetime.fromtimestamp(model.created_at).isoformat(),
                'created_by': model.created_by
            },
            'intended_use': {
                'use_cases': model.use_cases,
                'description': model.description,
                'stage': model.stage
            },
            'performance': {
                'accuracy': model.accuracy,
                'precision': model.precision,
                'recall': model.recall,
                'f1_score': model.f1_score,
                'training_metrics': model.training_metrics
            },
            'technical_details': {
                'hyperparameters': model.hyperparameters,
                'features': model.feature_names,
                'training_data': model.training_data_id
            },
            'limitations': model.limitations,
            'compliance': {
                'standards': model.compliance_standards,
                'data_privacy_level': model.data_privacy_level
            },
            'lineage': {
                'parent_model': model.parent_model_id,
                'training_code_hash': model.training_code_hash
            }
        }


# Global governance instance
_global_governance: Optional[ModelGovernance] = None


def get_global_governance() -> ModelGovernance:
    """Get or create global governance instance"""
    global _global_governance
    if _global_governance is None:
        _global_governance = ModelGovernance()
    return _global_governance


def set_global_governance(governance: ModelGovernance):
    """Set global governance instance"""
    global _global_governance
    _global_governance = governance
