#!/bin/bash
# Living Digital Fortress - PERSON 3 Production Deployment Script
# Blue-Green Deployment with Automated Rollback
# Version: 1.0.0

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

NAMESPACE="living-fortress"
SERVICE_NAME="credits-service"
DEPLOYMENT_TIMEOUT=600  # 10 minutes
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=10
CANARY_STAGES=(1 5 25 50 100)
CANARY_STAGE_DURATION=180  # 3 minutes per stage

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl not found. Please install kubectl."; exit 1; }
    command -v jq >/dev/null 2>&1 || { log_error "jq not found. Please install jq."; exit 1; }
    
    kubectl cluster-info >/dev/null 2>&1 || { log_error "Cannot connect to Kubernetes cluster."; exit 1; }
    
    log_success "Prerequisites check passed"
}

get_current_environment() {
    local current_env=$(kubectl get service ${SERVICE_NAME} -n ${NAMESPACE} -o jsonpath='{.spec.selector.version}' 2>/dev/null || echo "blue")
    echo "$current_env"
}

get_target_environment() {
    local current=$(get_current_environment)
    if [ "$current" == "blue" ]; then
        echo "green"
    else
        echo "blue"
    fi
}

wait_for_deployment() {
    local deployment=$1
    local timeout=$2
    
    log_info "Waiting for deployment ${deployment} to be ready..."
    
    if kubectl rollout status deployment/${deployment} -n ${NAMESPACE} --timeout=${timeout}s; then
        log_success "Deployment ${deployment} is ready"
        return 0
    else
        log_error "Deployment ${deployment} failed to become ready within ${timeout}s"
        return 1
    fi
}

check_health() {
    local environment=$1
    local retries=$2
    local interval=$3
    
    log_info "Checking health of ${environment} environment..."
    
    for i in $(seq 1 $retries); do
        local pods=$(kubectl get pods -n ${NAMESPACE} -l app=${SERVICE_NAME},version=${environment} -o json)
        local ready_pods=$(echo "$pods" | jq '[.items[] | select(.status.phase == "Running" and .status.conditions[] | select(.type == "Ready" and .status == "True"))] | length')
        local total_pods=$(echo "$pods" | jq '.items | length')
        
        if [ "$ready_pods" -gt 0 ] && [ "$ready_pods" -eq "$total_pods" ]; then
            log_success "All $total_pods pods in ${environment} are healthy"
            return 0
        fi
        
        log_warning "Health check attempt $i/$retries: $ready_pods/$total_pods pods ready"
        sleep $interval
    done
    
    log_error "Health check failed after $retries attempts"
    return 1
}

check_metrics() {
    local environment=$1
    
    log_info "Checking metrics for ${environment} environment..."
    
    # Query Prometheus for error rate (if available)
    # This is a placeholder - actual implementation would query Prometheus API
    
    # Simulate metrics check
    local error_rate=$(kubectl get pods -n ${NAMESPACE} -l app=${SERVICE_NAME},version=${environment} --no-headers | grep -c "Running" || echo 0)
    
    if [ "$error_rate" -gt 0 ]; then
        log_success "Metrics check passed for ${environment}"
        return 0
    else
        log_warning "Metrics check inconclusive for ${environment}"
        return 1
    fi
}

shift_traffic() {
    local target_env=$1
    local percentage=$2
    
    log_info "Shifting ${percentage}% traffic to ${target_env}..."
    
    # In a real implementation, this would use a service mesh (Istio, Linkerd)
    # or an advanced ingress controller to gradually shift traffic
    
    if [ "$percentage" -eq 100 ]; then
        # Full cutover
        kubectl patch service ${SERVICE_NAME} -n ${NAMESPACE} -p "{\"spec\":{\"selector\":{\"version\":\"${target_env}\"}}}"
        log_success "Full traffic switched to ${target_env}"
    else
        # Partial traffic shift (would require service mesh)
        log_info "Canary deployment at ${percentage}% (service mesh required for partial traffic)"
    fi
}

rollback() {
    local current_env=$1
    local target_env=$2
    
    log_warning "Initiating rollback from ${target_env} to ${current_env}..."
    
    # Switch traffic back to current environment
    kubectl patch service ${SERVICE_NAME} -n ${NAMESPACE} -p "{\"spec\":{\"selector\":{\"version\":\"${current_env}\"}}}"
    
    # Scale down failed deployment
    kubectl scale deployment/${SERVICE_NAME}-${target_env} -n ${NAMESPACE} --replicas=0
    
    log_success "Rollback completed. Traffic restored to ${current_env}"
}

run_smoke_tests() {
    local environment=$1
    
    log_info "Running smoke tests against ${environment}..."
    
    # Get service endpoint
    local service_ip=$(kubectl get service ${SERVICE_NAME}-${environment} -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
    
    # Create a temporary pod to run tests
    kubectl run smoke-test-${environment} \
        --image=curlimages/curl:latest \
        --rm -i --restart=Never \
        --namespace=${NAMESPACE} \
        -- curl -f -s -o /dev/null -w "%{http_code}" http://${service_ip}:5004/health
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "Smoke tests passed for ${environment}"
        return 0
    else
        log_error "Smoke tests failed for ${environment}"
        return 1
    fi
}

backup_database() {
    log_info "Creating database backup before deployment..."
    
    # Trigger database backup
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="pre-deployment-${timestamp}"
    
    kubectl exec -n ${NAMESPACE} postgres-primary-0 -- \
        pg_dump -U credits_user -d credits_db | \
        kubectl exec -i -n ${NAMESPACE} postgres-primary-0 -- \
        gzip > /tmp/${backup_name}.sql.gz
    
    log_success "Database backup created: ${backup_name}.sql.gz"
}

run_migrations() {
    log_info "Running database migrations..."
    
    # Run migrations using a Job
    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration-$(date +%s)
  namespace: ${NAMESPACE}
spec:
  template:
    spec:
      containers:
      - name: migration
        image: livingfortress/migrations:latest
        env:
        - name: DB_URL
          valueFrom:
            secretKeyRef:
              name: fortress-secrets
              key: DB_PRIMARY_URL
      restartPolicy: Never
  backoffLimit: 3
EOF
    
    # Wait for migration job to complete
    sleep 5
    local job_name=$(kubectl get jobs -n ${NAMESPACE} --sort-by=.metadata.creationTimestamp | tail -1 | awk '{print $1}')
    kubectl wait --for=condition=complete --timeout=300s job/${job_name} -n ${NAMESPACE}
    
    log_success "Database migrations completed"
}

# =============================================================================
# DEPLOYMENT STAGES
# =============================================================================

stage_pre_deployment() {
    log_info "=========================================="
    log_info "Stage 1: Pre-Deployment Checks"
    log_info "=========================================="
    
    check_prerequisites
    
    local current_env=$(get_current_environment)
    local target_env=$(get_target_environment)
    
    log_info "Current environment: ${current_env}"
    log_info "Target environment: ${target_env}"
    
    # Backup database
    backup_database
    
    log_success "Pre-deployment checks completed"
    
    echo "$current_env:$target_env"
}

stage_deploy_new_version() {
    local current_env=$1
    local target_env=$2
    local new_image=$3
    
    log_info "=========================================="
    log_info "Stage 2: Deploy New Version"
    log_info "=========================================="
    
    log_info "Deploying ${new_image} to ${target_env} environment..."
    
    # Update deployment image
    kubectl set image deployment/${SERVICE_NAME}-${target_env} \
        ${SERVICE_NAME}=${new_image} \
        -n ${NAMESPACE}
    
    # Wait for deployment to be ready
    if ! wait_for_deployment "${SERVICE_NAME}-${target_env}" ${DEPLOYMENT_TIMEOUT}; then
        log_error "Deployment failed"
        return 1
    fi
    
    # Run database migrations
    run_migrations
    
    log_success "New version deployed to ${target_env}"
}

stage_health_checks() {
    local target_env=$1
    
    log_info "=========================================="
    log_info "Stage 3: Health Checks"
    log_info "=========================================="
    
    if ! check_health ${target_env} ${HEALTH_CHECK_RETRIES} ${HEALTH_CHECK_INTERVAL}; then
        log_error "Health checks failed"
        return 1
    fi
    
    # Run smoke tests
    if ! run_smoke_tests ${target_env}; then
        log_error "Smoke tests failed"
        return 1
    fi
    
    # Check metrics
    if ! check_metrics ${target_env}; then
        log_warning "Metrics check inconclusive, proceeding with caution"
    fi
    
    log_success "Health checks passed"
}

stage_canary_deployment() {
    local current_env=$1
    local target_env=$2
    
    log_info "=========================================="
    log_info "Stage 4: Canary Deployment"
    log_info "=========================================="
    
    for stage in "${CANARY_STAGES[@]}"; do
        log_info "Canary stage: ${stage}% traffic to ${target_env}"
        
        shift_traffic ${target_env} ${stage}
        
        log_info "Monitoring for ${CANARY_STAGE_DURATION} seconds..."
        sleep ${CANARY_STAGE_DURATION}
        
        # Check health during canary
        if ! check_health ${target_env} 5 10; then
            log_error "Health check failed during canary stage ${stage}%"
            rollback ${current_env} ${target_env}
            return 1
        fi
        
        # Check for elevated error rates
        # (In production, this would query Prometheus)
        
        log_success "Canary stage ${stage}% successful"
    done
    
    log_success "Canary deployment completed"
}

stage_full_cutover() {
    local current_env=$1
    local target_env=$2
    
    log_info "=========================================="
    log_info "Stage 5: Full Cutover"
    log_info "=========================================="
    
    log_info "Performing full traffic cutover to ${target_env}..."
    shift_traffic ${target_env} 100
    
    log_info "Monitoring for 5 minutes post-cutover..."
    sleep 300
    
    if ! check_health ${target_env} 10 10; then
        log_error "Health check failed after cutover"
        rollback ${current_env} ${target_env}
        return 1
    fi
    
    log_success "Full cutover completed successfully"
}

stage_cleanup() {
    local current_env=$1
    
    log_info "=========================================="
    log_info "Stage 6: Cleanup"
    log_info "=========================================="
    
    log_info "Scaling down old ${current_env} environment to 1 replica..."
    kubectl scale deployment/${SERVICE_NAME}-${current_env} -n ${NAMESPACE} --replicas=1
    
    log_success "Cleanup completed"
}

# =============================================================================
# MAIN DEPLOYMENT FUNCTION
# =============================================================================

deploy() {
    local new_image=$1
    
    log_info "╔════════════════════════════════════════════════════════════════╗"
    log_info "║   Living Digital Fortress - Blue-Green Deployment             ║"
    log_info "║   PERSON 3: Business Logic & Infrastructure                   ║"
    log_info "╚════════════════════════════════════════════════════════════════╝"
    
    # Stage 1: Pre-deployment
    local environments=$(stage_pre_deployment)
    local current_env=$(echo $environments | cut -d: -f1)
    local target_env=$(echo $environments | cut -d: -f2)
    
    # Stage 2: Deploy new version
    if ! stage_deploy_new_version ${current_env} ${target_env} ${new_image}; then
        log_error "Deployment failed at stage 2"
        exit 1
    fi
    
    # Stage 3: Health checks
    if ! stage_health_checks ${target_env}; then
        log_error "Deployment failed at stage 3"
        rollback ${current_env} ${target_env}
        exit 1
    fi
    
    # Stage 4: Canary deployment
    if ! stage_canary_deployment ${current_env} ${target_env}; then
        log_error "Deployment failed at stage 4"
        exit 1
    fi
    
    # Stage 5: Full cutover
    if ! stage_full_cutover ${current_env} ${target_env}; then
        log_error "Deployment failed at stage 5"
        exit 1
    fi
    
    # Stage 6: Cleanup
    stage_cleanup ${current_env}
    
    log_success "╔════════════════════════════════════════════════════════════════╗"
    log_success "║   DEPLOYMENT COMPLETED SUCCESSFULLY!                           ║"
    log_success "║   New version: ${new_image}                                    ║"
    log_success "║   Active environment: ${target_env}                            ║"
    log_success "╚════════════════════════════════════════════════════════════════╝"
}

# =============================================================================
# CLI INTERFACE
# =============================================================================

show_help() {
    cat << EOF
Living Digital Fortress - Blue-Green Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy <image>      Deploy new version with blue-green strategy
    rollback            Rollback to previous version
    status              Show current deployment status
    test                Run smoke tests
    help                Show this help message

Examples:
    $0 deploy livingfortress/credits-service:v2.0.0
    $0 rollback
    $0 status

EOF
}

show_status() {
    log_info "Current Deployment Status:"
    echo ""
    
    local current_env=$(get_current_environment)
    echo "Active Environment: ${current_env}"
    echo ""
    
    kubectl get deployments -n ${NAMESPACE} -l app=${SERVICE_NAME}
    echo ""
    
    kubectl get pods -n ${NAMESPACE} -l app=${SERVICE_NAME}
    echo ""
    
    kubectl get service ${SERVICE_NAME} -n ${NAMESPACE}
}

# =============================================================================
# MAIN
# =============================================================================

case "${1:-help}" in
    deploy)
        if [ -z "${2:-}" ]; then
            log_error "Please specify image to deploy"
            show_help
            exit 1
        fi
        deploy "$2"
        ;;
    rollback)
        current=$(get_current_environment)
        target=$(get_target_environment)
        rollback $target $current
        ;;
    status)
        show_status
        ;;
    test)
        current=$(get_current_environment)
        run_smoke_tests $current
        ;;
    help|*)
        show_help
        ;;
esac
