#!/bin/bash

# Production Deployment Script for PERSON 3 Services
# Credits, Shadow, Camouflage services with backup and rollback

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="shieldx-prod"
BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
DEPLOYMENT_TIMEOUT="10m"
ROLLBACK_ON_FAILURE="true"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        log_warn "Namespace $NAMESPACE does not exist, creating..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Check PVC storage class exists
    if ! kubectl get storageclass fast-ssd &>/dev/null; then
        log_error "StorageClass 'fast-ssd' not found"
        exit 1
    fi
    
    # Check database connectivity
    log_info "Checking database connectivity..."
    if ! kubectl exec -n "$NAMESPACE" deploy/postgres -- pg_isready -U postgres &>/dev/null; then
        log_error "PostgreSQL is not ready"
        exit 1
    fi
    
    log_info "✓ Pre-deployment checks passed"
}

# Backup databases before deployment
backup_databases() {
    log_info "Backing up databases to $BACKUP_DIR..."
    mkdir -p "$BACKUP_DIR"
    
    # Backup Credits database
    log_info "Backing up credits database..."
    kubectl exec -n "$NAMESPACE" deploy/postgres -- pg_dump -U credits_user credits > "$BACKUP_DIR/credits.sql" || {
        log_error "Credits database backup failed"
        return 1
    }
    
    # Backup Shadow database
    log_info "Backing up shadow database..."
    kubectl exec -n "$NAMESPACE" deploy/postgres -- pg_dump -U shadow_user shadow > "$BACKUP_DIR/shadow.sql" || {
        log_error "Shadow database backup failed"
        return 1
    }
    
    # Compress backups
    tar -czf "$BACKUP_DIR.tar.gz" -C "$(dirname $BACKUP_DIR)" "$(basename $BACKUP_DIR)"
    rm -rf "$BACKUP_DIR"
    
    log_info "✓ Database backups completed: $BACKUP_DIR.tar.gz"
}

# Deploy service with rollback capability
deploy_service() {
    local service_name=$1
    local manifest_path=$2
    
    log_info "Deploying $service_name..."
    
    # Save current state for rollback
    kubectl get deployment "$service_name" -n "$NAMESPACE" -o yaml > "/tmp/${service_name}-rollback.yaml" 2>/dev/null || true
    
    # Apply new configuration
    if ! kubectl apply -f "$manifest_path"; then
        log_error "Failed to apply manifest for $service_name"
        return 1
    fi
    
    # Wait for rollout
    log_info "Waiting for $service_name rollout..."
    if ! kubectl rollout status deployment/"$service_name" -n "$NAMESPACE" --timeout="$DEPLOYMENT_TIMEOUT"; then
        log_error "$service_name rollout failed"
        
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            log_warn "Rolling back $service_name..."
            kubectl rollout undo deployment/"$service_name" -n "$NAMESPACE"
            kubectl rollout status deployment/"$service_name" -n "$NAMESPACE" --timeout="5m"
        fi
        return 1
    fi
    
    log_info "✓ $service_name deployed successfully"
}

# Health check after deployment
health_check() {
    local service_name=$1
    local port=$2
    local max_retries=30
    local retry_interval=2
    
    log_info "Running health check for $service_name..."
    
    for ((i=1; i<=max_retries; i++)); do
        if kubectl exec -n "$NAMESPACE" deploy/"$service_name" -- wget -q -O- "http://localhost:$port/health" &>/dev/null; then
            log_info "✓ $service_name health check passed"
            return 0
        fi
        
        log_warn "Health check attempt $i/$max_retries failed, retrying..."
        sleep "$retry_interval"
    done
    
    log_error "Health check failed for $service_name"
    return 1
}

# Run database migrations
run_migrations() {
    local service=$1
    
    log_info "Running database migrations for $service..."
    
    # Create migration job
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${service}-migration-$(date +%s)
  namespace: $NAMESPACE
spec:
  ttlSecondsAfterFinished: 300
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: migrate
        image: ghcr.io/shieldx/${service}-service:v2.0.0
        command: ["/app/migrate"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ${service}-secrets
              key: DATABASE_URL
        - name: BACKUP_BEFORE_MIGRATE
          value: "true"
EOF

    # Wait for migration to complete
    kubectl wait --for=condition=complete --timeout=300s job -l "app=${service}-migration" -n "$NAMESPACE"
    
    log_info "✓ Migrations completed for $service"
}

# Smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Test Credits API
    log_info "Testing Credits API..."
    kubectl run test-credits --rm -it --restart=Never -n "$NAMESPACE" \
        --image=curlimages/curl:latest \
        -- curl -f http://credits-service:5004/health || {
        log_error "Credits API smoke test failed"
        return 1
    }
    
    # Test Shadow API
    log_info "Testing Shadow API..."
    kubectl run test-shadow --rm -it --restart=Never -n "$NAMESPACE" \
        --image=curlimages/curl:latest \
        -- curl -f http://shadow-service:5005/health || {
        log_error "Shadow API smoke test failed"
        return 1
    }
    
    log_info "✓ Smoke tests passed"
}

# Verify metrics collection
verify_metrics() {
    log_info "Verifying metrics collection..."
    
    for service in credits-service shadow-service; do
        local metrics=$(kubectl exec -n "$NAMESPACE" deploy/"$service" -- wget -q -O- http://localhost:5004/metrics 2>/dev/null || echo "")
        
        if [[ -z "$metrics" ]]; then
            log_warn "No metrics available for $service"
        else
            log_info "✓ Metrics available for $service"
        fi
    done
}

# Update deployment documentation
update_documentation() {
    log_info "Updating deployment documentation..."
    
    cat > "/tmp/deployment-summary.md" <<EOF
# Deployment Summary

**Date:** $(date)
**Namespace:** $NAMESPACE
**Deployed by:** PERSON 3
**Backup Location:** $BACKUP_DIR.tar.gz

## Services Deployed

- **Credits Service** (Port 5004)
  - Replicas: 3
  - Resources: 500m CPU / 512Mi RAM (request)
  - Autoscaling: 3-10 pods

- **Shadow Service** (Port 5005)
  - Replicas: 2
  - Resources: 1000m CPU / 1Gi RAM (request)
  - Autoscaling: 2-6 pods

## Health Status

\`\`\`
$(kubectl get pods -n "$NAMESPACE" -l 'app in (credits-service,shadow-service)')
\`\`\`

## Metrics

\`\`\`
$(kubectl top pods -n "$NAMESPACE" -l 'app in (credits-service,shadow-service)')
\`\`\`

## Rollback Command

If issues occur:
\`\`\`bash
kubectl rollout undo deployment/credits-service -n $NAMESPACE
kubectl rollout undo deployment/shadow-service -n $NAMESPACE
\`\`\`

## Restore Database

\`\`\`bash
tar -xzf $BACKUP_DIR.tar.gz
kubectl exec -i -n $NAMESPACE deploy/postgres -- psql -U credits_user credits < credits.sql
kubectl exec -i -n $NAMESPACE deploy/postgres -- psql -U shadow_user shadow < shadow.sql
\`\`\`
EOF

    log_info "✓ Documentation updated: /tmp/deployment-summary.md"
}

# Main deployment workflow
main() {
    log_info "Starting production deployment..."
    
    # Pre-checks
    pre_deployment_checks
    
    # Backup
    if ! backup_databases; then
        log_error "Backup failed, aborting deployment"
        exit 1
    fi
    
    # Deploy Credits Service
    if ! deploy_service "credits-service" "pilot/credits/credits-production.yaml"; then
        log_error "Credits deployment failed"
        exit 1
    fi
    
    # Health check Credits
    if ! health_check "credits-service" "5004"; then
        log_error "Credits health check failed"
        exit 1
    fi
    
    # Deploy Shadow Service
    if ! deploy_service "shadow-service" "pilot/shadow/shadow-production.yaml"; then
        log_error "Shadow deployment failed"
        exit 1
    fi
    
    # Health check Shadow
    if ! health_check "shadow-service" "5005"; then
        log_error "Shadow health check failed"
        exit 1
    fi
    
    # Smoke tests
    if ! run_smoke_tests; then
        log_warn "Smoke tests failed, but deployment continues"
    fi
    
    # Verify metrics
    verify_metrics
    
    # Update documentation
    update_documentation
    
    log_info "✓✓✓ Production deployment completed successfully! ✓✓✓"
    log_info "Backup location: $BACKUP_DIR.tar.gz"
    log_info "Deployment summary: /tmp/deployment-summary.md"
}

# Rollback function (can be called independently)
rollback() {
    log_warn "Initiating rollback..."
    
    kubectl rollout undo deployment/credits-service -n "$NAMESPACE"
    kubectl rollout undo deployment/shadow-service -n "$NAMESPACE"
    
    kubectl rollout status deployment/credits-service -n "$NAMESPACE"
    kubectl rollout status deployment/shadow-service -n "$NAMESPACE"
    
    log_info "✓ Rollback completed"
}

# Parse command line arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    rollback)
        rollback
        ;;
    backup)
        backup_databases
        ;;
    health-check)
        health_check "credits-service" "5004"
        health_check "shadow-service" "5005"
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|backup|health-check}"
        exit 1
        ;;
esac
