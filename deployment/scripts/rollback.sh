#!/bin/bash

# Code Intelligence System Rollback Script
# This script handles rollback operations for different deployment types

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"

# Default values
ENVIRONMENT="production"
DEPLOY_TYPE="kubernetes"
ROLLBACK_STEPS=1
DRY_RUN=false
VERBOSE=false
FORCE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Code Intelligence System Rollback Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV    Target environment (dev, staging, production) [default: production]
    -t, --type TYPE         Deployment type (docker, kubernetes, terraform) [default: kubernetes]
    -s, --steps STEPS       Number of rollback steps [default: 1]
    -d, --dry-run           Show what would be rolled back without executing
    -f, --force             Force rollback without confirmation
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Examples:
    $0 --environment production --type kubernetes
    $0 --environment staging --steps 2 --dry-run
    $0 --environment dev --type docker --force

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--type)
                DEPLOY_TYPE="$2"
                shift 2
                ;;
            -s|--steps)
                ROLLBACK_STEPS="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        dev|staging|production)
            log_info "Rolling back environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: dev, staging, production"
            exit 1
            ;;
    esac
}

# Confirm rollback
confirm_rollback() {
    if [ "$FORCE" = true ] || [ "$DRY_RUN" = true ]; then
        return 0
    fi
    
    echo ""
    log_warning "You are about to rollback the $ENVIRONMENT environment"
    log_warning "Deployment type: $DEPLOY_TYPE"
    log_warning "Rollback steps: $ROLLBACK_STEPS"
    echo ""
    
    read -p "Are you sure you want to proceed? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Rollback cancelled"
        exit 0
    fi
}

# Get current deployment status
get_deployment_status() {
    log_info "Getting current deployment status..."
    
    case $DEPLOY_TYPE in
        docker)
            if command -v docker-compose >/dev/null 2>&1; then
                cd "$PROJECT_ROOT"
                docker-compose ps
            else
                log_warning "docker-compose not available"
            fi
            ;;
        kubernetes)
            if command -v kubectl >/dev/null 2>&1; then
                kubectl get deployments -n code-intelligence
                kubectl get pods -n code-intelligence
            else
                log_warning "kubectl not available"
            fi
            ;;
        terraform)
            if command -v terraform >/dev/null 2>&1; then
                cd "$DEPLOYMENT_DIR/terraform"
                terraform show -json | jq -r '.values.root_module.resources[] | select(.type == "aws_instance") | .values.tags.Name'
            else
                log_warning "terraform not available"
            fi
            ;;
    esac
}

# Rollback Docker deployment
rollback_docker() {
    log_info "Rolling back Docker deployment..."
    
    cd "$PROJECT_ROOT"
    
    local compose_file="docker-compose.yml"
    if [ "$ENVIRONMENT" = "dev" ]; then
        compose_file="docker-compose.dev.yml"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would stop services: docker-compose -f $compose_file down"
        log_info "[DRY RUN] Would remove volumes: docker-compose -f $compose_file down -v"
        return 0
    fi
    
    # Stop current services
    log_info "Stopping current services..."
    docker-compose -f "$compose_file" down
    
    # Get previous image tags
    local previous_tag
    if [ -f ".deployment_history" ]; then
        previous_tag=$(tail -n "$((ROLLBACK_STEPS + 1))" .deployment_history | head -n 1)
        log_info "Rolling back to image tag: $previous_tag"
        
        # Update environment variable for previous tag
        export IMAGE_TAG="$previous_tag"
    else
        log_warning "No deployment history found, using latest tag"
        export IMAGE_TAG="${ENVIRONMENT}-latest"
    fi
    
    # Start services with previous version
    log_info "Starting services with previous version..."
    docker-compose -f "$compose_file" up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Verify rollback
    if docker-compose -f "$compose_file" ps | grep -q "Up"; then
        log_success "Docker rollback completed successfully"
    else
        log_error "Docker rollback failed"
        return 1
    fi
}

# Rollback Kubernetes deployment
rollback_kubernetes() {
    log_info "Rolling back Kubernetes deployment..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would rollback deployments:"
        kubectl get deployments -n code-intelligence -o name
        return 0
    fi
    
    # Get list of deployments
    local deployments
    deployments=$(kubectl get deployments -n code-intelligence -o name)
    
    if [ -z "$deployments" ]; then
        log_error "No deployments found in code-intelligence namespace"
        return 1
    fi
    
    # Rollback each deployment
    for deployment in $deployments; do
        local deployment_name
        deployment_name=$(echo "$deployment" | cut -d'/' -f2)
        
        log_info "Rolling back deployment: $deployment_name"
        
        # Check rollout history
        if kubectl rollout history "$deployment" -n code-intelligence | grep -q "REVISION"; then
            # Rollback to previous revision
            kubectl rollout undo "$deployment" -n code-intelligence --to-revision=$(($(kubectl rollout history "$deployment" -n code-intelligence | tail -n 1 | awk '{print $1}') - ROLLBACK_STEPS))
            
            # Wait for rollout to complete
            kubectl rollout status "$deployment" -n code-intelligence --timeout=300s
        else
            log_warning "No rollout history found for $deployment_name"
        fi
    done
    
    # Verify rollback
    log_info "Verifying rollback..."
    kubectl get pods -n code-intelligence
    
    # Check if all pods are running
    local failed_pods
    failed_pods=$(kubectl get pods -n code-intelligence --field-selector=status.phase!=Running -o name)
    
    if [ -n "$failed_pods" ]; then
        log_error "Some pods are not running after rollback:"
        echo "$failed_pods"
        return 1
    fi
    
    log_success "Kubernetes rollback completed successfully"
}

# Rollback Terraform deployment
rollback_terraform() {
    log_info "Rolling back Terraform deployment..."
    
    cd "$DEPLOYMENT_DIR/terraform"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would show Terraform state:"
        terraform show
        log_info "[DRY RUN] Would apply previous state"
        return 0
    fi
    
    # Check if state backup exists
    local state_backup="terraform.tfstate.backup"
    if [ ! -f "$state_backup" ]; then
        log_error "No Terraform state backup found"
        log_error "Cannot perform automatic rollback"
        return 1
    fi
    
    # Create current state backup
    cp terraform.tfstate "terraform.tfstate.pre-rollback.$(date +%Y%m%d_%H%M%S)"
    
    # Restore previous state
    log_info "Restoring previous Terraform state..."
    cp "$state_backup" terraform.tfstate
    
    # Apply the previous state
    log_info "Applying previous configuration..."
    terraform apply -auto-approve -var="environment=$ENVIRONMENT"
    
    # Verify rollback
    terraform show
    
    log_success "Terraform rollback completed successfully"
}

# Rollback database migrations
rollback_database() {
    log_info "Rolling back database migrations..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would rollback database migrations"
        return 0
    fi
    
    # This would depend on your migration system
    # Example for Alembic (Python):
    # alembic downgrade -$ROLLBACK_STEPS
    
    # Example for custom migration system:
    cd "$PROJECT_ROOT"
    
    if [ -f "src/code_intelligence/database/migrations.py" ]; then
        python -m src.code_intelligence.database.migrations rollback --steps "$ROLLBACK_STEPS"
    else
        log_warning "No database migration system found"
    fi
}

# Verify rollback
verify_rollback() {
    log_info "Verifying rollback..."
    
    case $DEPLOY_TYPE in
        docker)
            # Check Docker services
            cd "$PROJECT_ROOT"
            if docker-compose ps | grep -q "Up"; then
                log_success "Docker services are running after rollback"
            else
                log_error "Docker services are not running after rollback"
                return 1
            fi
            ;;
        kubernetes)
            # Check Kubernetes deployment
            if kubectl get pods -n code-intelligence | grep -q "Running"; then
                log_success "Kubernetes pods are running after rollback"
            else
                log_error "Kubernetes pods are not running after rollback"
                return 1
            fi
            ;;
        terraform)
            # Check Terraform state
            cd "$DEPLOYMENT_DIR/terraform"
            if terraform show | grep -q "resource"; then
                log_success "Terraform resources are deployed after rollback"
            else
                log_error "Terraform rollback verification failed"
                return 1
            fi
            ;;
    esac
    
    # Health check API endpoint
    local api_url
    case $DEPLOY_TYPE in
        docker)
            api_url="http://localhost:8000/api/v1/health/"
            ;;
        kubernetes)
            # Use port-forward for testing
            kubectl port-forward service/code-intelligence-api-service 8080:8000 -n code-intelligence &
            local port_forward_pid=$!
            sleep 5
            api_url="http://localhost:8080/api/v1/health/"
            ;;
        terraform)
            api_url="https://api.${ENVIRONMENT}.code-intelligence.example.com/api/v1/health/"
            ;;
    esac
    
    log_info "Checking API health at: $api_url"
    
    # Wait for API to be available
    local retries=10
    while [ $retries -gt 0 ]; do
        if curl -f -s "$api_url" > /dev/null; then
            log_success "API is healthy after rollback"
            break
        fi
        
        log_info "Waiting for API to be available... ($retries retries left)"
        sleep 10
        ((retries--))
    done
    
    # Clean up port-forward if used
    if [ "$DEPLOY_TYPE" = "kubernetes" ] && [ -n "${port_forward_pid:-}" ]; then
        kill $port_forward_pid 2>/dev/null || true
    fi
    
    if [ $retries -eq 0 ]; then
        log_error "API health check failed after rollback"
        return 1
    fi
    
    log_success "Rollback verification completed"
}

# Main rollback function
main() {
    log_info "Starting Code Intelligence System rollback"
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment type: $DEPLOY_TYPE"
    log_info "Rollback steps: $ROLLBACK_STEPS"
    
    # Set verbose mode
    if [ "$VERBOSE" = true ]; then
        set -x
    fi
    
    # Validate inputs
    validate_environment
    
    # Get current status
    get_deployment_status
    
    # Confirm rollback
    confirm_rollback
    
    # Perform rollback based on type
    case $DEPLOY_TYPE in
        docker)
            rollback_docker
            ;;
        kubernetes)
            rollback_kubernetes
            ;;
        terraform)
            rollback_terraform
            ;;
        *)
            log_error "Invalid deployment type: $DEPLOY_TYPE"
            exit 1
            ;;
    esac
    
    # Rollback database if needed
    rollback_database
    
    # Verify rollback
    if ! verify_rollback; then
        log_error "Rollback verification failed"
        exit 1
    fi
    
    log_success "Rollback completed successfully!"
    
    # Log rollback event
    echo "$(date): Rollback completed for $ENVIRONMENT environment ($DEPLOY_TYPE)" >> "$PROJECT_ROOT/.rollback_history"
}

# Parse arguments and run main function
parse_args "$@"
main