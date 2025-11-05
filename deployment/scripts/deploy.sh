#!/bin/bash

# Code Intelligence System Deployment Script
# This script automates the deployment process for different environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"

# Default values
ENVIRONMENT="production"
DEPLOY_TYPE="kubernetes"
SKIP_BUILD=false
SKIP_TESTS=false
DRY_RUN=false
VERBOSE=false

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
Code Intelligence System Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV    Target environment (dev, staging, production) [default: production]
    -t, --type TYPE         Deployment type (docker, kubernetes, terraform) [default: kubernetes]
    -s, --skip-build        Skip building Docker images
    -T, --skip-tests        Skip running tests
    -d, --dry-run           Show what would be deployed without executing
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

Examples:
    $0 --environment dev --type docker
    $0 --environment production --type kubernetes --skip-tests
    $0 --environment staging --dry-run

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
            -s|--skip-build)
                SKIP_BUILD=true
                shift
                ;;
            -T|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
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
            log_info "Deploying to environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: dev, staging, production"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools based on deployment type
    case $DEPLOY_TYPE in
        docker)
            command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
            command -v docker-compose >/dev/null 2>&1 || missing_tools+=("docker-compose")
            ;;
        kubernetes)
            command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
            command -v helm >/dev/null 2>&1 || missing_tools+=("helm")
            ;;
        terraform)
            command -v terraform >/dev/null 2>&1 || missing_tools+=("terraform")
            command -v aws >/dev/null 2>&1 || missing_tools+=("aws")
            ;;
    esac
    
    # Common tools
    command -v git >/dev/null 2>&1 || missing_tools+=("git")
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    log_success "All prerequisites satisfied"
}

# Run tests
run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        log_warning "Skipping tests"
        return 0
    fi
    
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would run: python -m pytest tests/ -v"
        return 0
    fi
    
    # Run tests
    python -m pytest tests/ -v --tb=short
    
    log_success "Tests passed"
}

# Build Docker images
build_images() {
    if [ "$SKIP_BUILD" = true ]; then
        log_warning "Skipping image build"
        return 0
    fi
    
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    local image_tag="${ENVIRONMENT}-$(git rev-parse --short HEAD)"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would build images with tag: $image_tag"
        return 0
    fi
    
    # Build main API image
    docker build -t "code-intelligence/api:${image_tag}" \
                 -t "code-intelligence/api:${ENVIRONMENT}-latest" \
                 --target production .
    
    # Build web interface image
    if [ -f "src/code_intelligence/web/Dockerfile" ]; then
        docker build -t "code-intelligence/web:${image_tag}" \
                     -t "code-intelligence/web:${ENVIRONMENT}-latest" \
                     src/code_intelligence/web/
    fi
    
    log_success "Images built successfully"
    
    # Push images if not local development
    if [ "$ENVIRONMENT" != "dev" ]; then
        push_images "$image_tag"
    fi
}

# Push Docker images
push_images() {
    local image_tag="$1"
    
    log_info "Pushing Docker images..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would push images with tag: $image_tag"
        return 0
    fi
    
    # Push to registry (configure your registry here)
    docker push "code-intelligence/api:${image_tag}"
    docker push "code-intelligence/api:${ENVIRONMENT}-latest"
    
    if docker images | grep -q "code-intelligence/web"; then
        docker push "code-intelligence/web:${image_tag}"
        docker push "code-intelligence/web:${ENVIRONMENT}-latest"
    fi
    
    log_success "Images pushed successfully"
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    local compose_file="docker-compose.yml"
    if [ "$ENVIRONMENT" = "dev" ]; then
        compose_file="docker-compose.dev.yml"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would run: docker-compose -f $compose_file up -d"
        return 0
    fi
    
    # Set environment variables
    export ENVIRONMENT="$ENVIRONMENT"
    export IMAGE_TAG="${ENVIRONMENT}-latest"
    
    # Deploy services
    docker-compose -f "$compose_file" up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    if docker-compose -f "$compose_file" ps | grep -q "unhealthy"; then
        log_error "Some services are unhealthy"
        docker-compose -f "$compose_file" ps
        exit 1
    fi
    
    log_success "Docker deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    cd "$DEPLOYMENT_DIR/kubernetes"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would deploy Kubernetes manifests"
        kubectl apply --dry-run=client -f .
        return 0
    fi
    
    # Apply namespace first
    kubectl apply -f namespace.yaml
    
    # Apply configurations
    kubectl apply -f configmap.yaml
    kubectl apply -f secrets.yaml
    
    # Deploy databases
    kubectl apply -f databases.yaml
    
    # Wait for databases to be ready
    log_info "Waiting for databases to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n code-intelligence --timeout=300s
    kubectl wait --for=condition=ready pod -l app=neo4j -n code-intelligence --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis -n code-intelligence --timeout=300s
    
    # Deploy application
    kubectl apply -f api-deployment.yaml
    
    # Wait for API to be ready
    kubectl wait --for=condition=available deployment/code-intelligence-api -n code-intelligence --timeout=300s
    
    # Deploy ingress
    kubectl apply -f ingress.yaml
    
    log_success "Kubernetes deployment completed"
    
    # Show deployment status
    kubectl get pods -n code-intelligence
    kubectl get services -n code-intelligence
}

# Deploy with Terraform
deploy_terraform() {
    log_info "Deploying infrastructure with Terraform..."
    
    cd "$DEPLOYMENT_DIR/terraform"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would run Terraform plan"
        terraform plan -var="environment=$ENVIRONMENT"
        return 0
    fi
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -var="environment=$ENVIRONMENT" -out=tfplan
    
    # Apply deployment
    log_info "Applying Terraform configuration..."
    terraform apply tfplan
    
    # Output important information
    terraform output
    
    log_success "Terraform deployment completed"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    case $DEPLOY_TYPE in
        docker)
            # Check Docker services
            if docker-compose ps | grep -q "Up"; then
                log_success "Docker services are running"
            else
                log_error "Some Docker services are not running"
                return 1
            fi
            ;;
        kubernetes)
            # Check Kubernetes deployment
            if kubectl get pods -n code-intelligence | grep -q "Running"; then
                log_success "Kubernetes pods are running"
            else
                log_error "Some Kubernetes pods are not running"
                return 1
            fi
            ;;
        terraform)
            # Check Terraform state
            cd "$DEPLOYMENT_DIR/terraform"
            if terraform show | grep -q "resource"; then
                log_success "Terraform resources are deployed"
            else
                log_error "Terraform deployment verification failed"
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
            api_url="http://$(kubectl get service code-intelligence-api-service -n code-intelligence -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8000/api/v1/health/"
            ;;
        terraform)
            api_url="https://api.${ENVIRONMENT}.code-intelligence.example.com/api/v1/health/"
            ;;
    esac
    
    if [ "$DRY_RUN" = false ]; then
        log_info "Checking API health at: $api_url"
        
        # Wait for API to be available
        local retries=30
        while [ $retries -gt 0 ]; do
            if curl -f -s "$api_url" > /dev/null; then
                log_success "API is healthy"
                break
            fi
            
            log_info "Waiting for API to be available... ($retries retries left)"
            sleep 10
            ((retries--))
        done
        
        if [ $retries -eq 0 ]; then
            log_error "API health check failed"
            return 1
        fi
    fi
    
    log_success "Deployment verification completed"
}

# Rollback deployment
rollback_deployment() {
    log_error "Deployment failed. Initiating rollback..."
    
    case $DEPLOY_TYPE in
        docker)
            docker-compose down
            ;;
        kubernetes)
            kubectl rollout undo deployment/code-intelligence-api -n code-intelligence
            ;;
        terraform)
            cd "$DEPLOYMENT_DIR/terraform"
            terraform destroy -auto-approve -var="environment=$ENVIRONMENT"
            ;;
    esac
    
    log_warning "Rollback completed"
}

# Main deployment function
main() {
    log_info "Starting Code Intelligence System deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment type: $DEPLOY_TYPE"
    
    # Set verbose mode
    if [ "$VERBOSE" = true ]; then
        set -x
    fi
    
    # Validate inputs
    validate_environment
    
    # Check prerequisites
    check_prerequisites
    
    # Run tests
    run_tests
    
    # Build images (if needed)
    if [ "$DEPLOY_TYPE" != "terraform" ]; then
        build_images
    fi
    
    # Deploy based on type
    case $DEPLOY_TYPE in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        terraform)
            deploy_terraform
            ;;
        *)
            log_error "Invalid deployment type: $DEPLOY_TYPE"
            exit 1
            ;;
    esac
    
    # Verify deployment
    if ! verify_deployment; then
        rollback_deployment
        exit 1
    fi
    
    log_success "Deployment completed successfully!"
    
    # Show access information
    case $DEPLOY_TYPE in
        docker)
            echo ""
            log_info "Access URLs:"
            log_info "  API: http://localhost:8000"
            log_info "  Web: http://localhost:3000"
            log_info "  Neo4j: http://localhost:7474"
            ;;
        kubernetes)
            echo ""
            log_info "Access URLs:"
            log_info "  Use 'kubectl port-forward' or configure ingress"
            ;;
        terraform)
            echo ""
            log_info "Access URLs:"
            cd "$DEPLOYMENT_DIR/terraform"
            terraform output api_url
            terraform output web_url
            ;;
    esac
}

# Parse arguments and run main function
parse_args "$@"
main