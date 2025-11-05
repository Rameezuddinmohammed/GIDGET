# Deployment Guide

This guide covers deploying the Code Intelligence System in various environments, from development to production.

## Overview

The Code Intelligence System supports multiple deployment options:

- **Docker Compose**: Simple single-machine deployment
- **Kubernetes**: Scalable container orchestration
- **Cloud Deployment**: AWS, GCP, Azure with Terraform
- **Hybrid**: Mix of cloud and on-premises components

## Prerequisites

### Common Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Git
- 4GB+ RAM
- 20GB+ disk space

### Production Requirements
- Kubernetes 1.24+
- Load balancer (nginx, HAProxy, or cloud LB)
- SSL certificates
- Monitoring stack (Prometheus, Grafana)
- Backup solution

## Development Deployment

### Quick Start with Docker Compose

1. **Clone and configure:**
   ```bash
   git clone https://github.com/your-org/code-intelligence.git
   cd code-intelligence
   cp .env.example .env
   ```

2. **Start development environment:**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

3. **Verify deployment:**
   ```bash
   curl http://localhost:8000/api/v1/health/
   ```

### Development Configuration

Edit `.env` for development:
```env
# Environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration
NEO4J_PASSWORD=dev-password
POSTGRES_PASSWORD=dev-password

# Development Features
ENABLE_DEBUG_ENDPOINTS=true
ENABLE_CORS=true
```

### Development Services

The development stack includes:
- **API Server**: `http://localhost:8000`
- **Web Interface**: `http://localhost:3000`
- **Neo4j Browser**: `http://localhost:7474`
- **Grafana**: `http://localhost:3001`

## Production Deployment

### Docker Compose Production

1. **Configure production environment:**
   ```bash
   cp .env.example .env.production
   ```

   Edit `.env.production`:
   ```env
   # Environment
   ENVIRONMENT=production
   LOG_LEVEL=INFO

   # Security
   NEO4J_PASSWORD=secure-random-password
   POSTGRES_PASSWORD=secure-random-password
   JWT_SECRET=your-jwt-secret-key
   ENCRYPTION_KEY=your-encryption-key

   # Performance
   WORKER_CONCURRENCY=4
   CACHE_TTL_SECONDS=3600
   ```

2. **Deploy production stack:**
   ```bash
   docker-compose --env-file .env.production up -d
   ```

3. **Set up SSL termination:**
   ```bash
   # Using nginx proxy
   docker run -d \
     --name nginx-proxy \
     -p 80:80 -p 443:443 \
     -v /var/run/docker.sock:/tmp/docker.sock:ro \
     -v ./ssl:/etc/nginx/certs \
     nginxproxy/nginx-proxy
   ```

### Production Checklist

- [ ] Secure passwords and secrets
- [ ] SSL/TLS certificates configured
- [ ] Firewall rules configured
- [ ] Backup strategy implemented
- [ ] Monitoring and alerting enabled
- [ ] Log aggregation configured
- [ ] Resource limits set
- [ ] Health checks configured

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3.0+ (optional)
- Ingress controller (nginx, traefik)
- Cert-manager for SSL (optional)

### Basic Kubernetes Deployment

1. **Create namespace:**
   ```bash
   kubectl apply -f deployment/kubernetes/namespace.yaml
   ```

2. **Configure secrets:**
   ```bash
   # Edit secrets with your values
   kubectl apply -f deployment/kubernetes/secrets.yaml
   ```

3. **Deploy databases:**
   ```bash
   kubectl apply -f deployment/kubernetes/databases.yaml
   
   # Wait for databases to be ready
   kubectl wait --for=condition=ready pod -l app=postgres -n code-intelligence --timeout=300s
   kubectl wait --for=condition=ready pod -l app=neo4j -n code-intelligence --timeout=300s
   kubectl wait --for=condition=ready pod -l app=redis -n code-intelligence --timeout=300s
   ```

4. **Deploy application:**
   ```bash
   kubectl apply -f deployment/kubernetes/api-deployment.yaml
   
   # Wait for API to be ready
   kubectl wait --for=condition=available deployment/code-intelligence-api -n code-intelligence --timeout=300s
   ```

5. **Configure ingress:**
   ```bash
   kubectl apply -f deployment/kubernetes/ingress.yaml
   ```

### Kubernetes Configuration

#### Resource Limits
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

#### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: code-intelligence-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: code-intelligence-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Persistent Volumes
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: neo4j-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

### Helm Deployment

1. **Add Helm repository:**
   ```bash
   helm repo add code-intelligence https://charts.code-intelligence.example.com
   helm repo update
   ```

2. **Install with Helm:**
   ```bash
   helm install code-intelligence code-intelligence/code-intelligence \
     --namespace code-intelligence \
     --create-namespace \
     --values values.yaml
   ```

3. **Upgrade deployment:**
   ```bash
   helm upgrade code-intelligence code-intelligence/code-intelligence \
     --values values.yaml
   ```

## Cloud Deployment

### AWS Deployment with Terraform

1. **Configure AWS credentials:**
   ```bash
   aws configure
   ```

2. **Initialize Terraform:**
   ```bash
   cd deployment/terraform
   terraform init
   ```

3. **Plan deployment:**
   ```bash
   terraform plan -var="environment=production" -out=tfplan
   ```

4. **Apply infrastructure:**
   ```bash
   terraform apply tfplan
   ```

5. **Configure kubectl:**
   ```bash
   aws eks update-kubeconfig --region us-west-2 --name code-intelligence-production
   ```

### AWS Architecture

The Terraform configuration creates:

- **EKS Cluster**: Managed Kubernetes cluster
- **RDS PostgreSQL**: Managed database with pgvector
- **ElastiCache Redis**: Managed Redis cluster
- **Application Load Balancer**: For ingress traffic
- **VPC**: Isolated network with public/private subnets
- **IAM Roles**: Least-privilege access
- **CloudWatch**: Logging and monitoring
- **S3 Buckets**: Data storage and backups

### GCP Deployment

1. **Configure GCP:**
   ```bash
   gcloud auth login
   gcloud config set project your-project-id
   ```

2. **Create GKE cluster:**
   ```bash
   gcloud container clusters create code-intelligence \
     --zone us-central1-a \
     --num-nodes 3 \
     --enable-autoscaling \
     --min-nodes 1 \
     --max-nodes 10
   ```

3. **Deploy application:**
   ```bash
   kubectl apply -f deployment/kubernetes/
   ```

### Azure Deployment

1. **Configure Azure CLI:**
   ```bash
   az login
   az account set --subscription your-subscription-id
   ```

2. **Create AKS cluster:**
   ```bash
   az aks create \
     --resource-group code-intelligence-rg \
     --name code-intelligence-aks \
     --node-count 3 \
     --enable-addons monitoring \
     --generate-ssh-keys
   ```

3. **Get credentials:**
   ```bash
   az aks get-credentials --resource-group code-intelligence-rg --name code-intelligence-aks
   ```

## Configuration Management

### Environment Variables

#### Core Configuration
```env
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Database
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure-password
POSTGRES_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://redis:6379

# Security
JWT_SECRET=your-jwt-secret
ENCRYPTION_KEY=your-encryption-key

# Performance
WORKER_CONCURRENCY=4
CACHE_TTL_SECONDS=3600
MAX_QUERY_TIME_SECONDS=300
```

#### Feature Flags
```env
# Features
ENABLE_AUTHENTICATION=true
ENABLE_RATE_LIMITING=true
ENABLE_METRICS=true
ENABLE_WEBSOCKETS=true

# Debugging
ENABLE_DEBUG_ENDPOINTS=false
ENABLE_PROFILING=false
```

### Configuration Files

#### Production docker-compose.yml
```yaml
version: '3.8'
services:
  api:
    image: code-intelligence/api:latest
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health/"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Scaling and Performance

### Horizontal Scaling

#### API Scaling
```bash
# Docker Compose
docker-compose up -d --scale api=5

# Kubernetes
kubectl scale deployment code-intelligence-api --replicas=5
```

#### Database Scaling
- **Neo4j**: Use clustering for read replicas
- **PostgreSQL**: Use read replicas and connection pooling
- **Redis**: Use Redis Cluster for horizontal scaling

### Vertical Scaling

#### Resource Allocation
```yaml
# Kubernetes resource requests/limits
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Performance Optimization

#### Caching Strategy
- **Application Cache**: Redis for query results
- **Database Cache**: Query result caching
- **CDN**: Static asset caching

#### Database Optimization
- **Indexes**: Ensure proper indexing on frequently queried fields
- **Connection Pooling**: Use pgbouncer for PostgreSQL
- **Query Optimization**: Monitor and optimize slow queries

## Monitoring and Observability

### Prometheus Metrics

1. **Configure Prometheus:**
   ```yaml
   # prometheus.yml
   scrape_configs:
     - job_name: 'code-intelligence-api'
       static_configs:
         - targets: ['api:8000']
       metrics_path: '/api/v1/health/prometheus'
   ```

2. **Deploy monitoring stack:**
   ```bash
   kubectl apply -f deployment/monitoring/
   ```

### Grafana Dashboards

Import pre-built dashboards:
- System Overview
- API Performance
- Agent Performance
- Database Metrics
- Business Metrics

### Alerting Rules

Key alerts to configure:
- API downtime
- High error rates
- Database connectivity issues
- High response times
- Resource exhaustion

## Security

### Network Security

#### Firewall Rules
```bash
# Allow only necessary ports
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw deny 8000/tcp   # Block direct API access
```

#### Network Policies (Kubernetes)
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: code-intelligence-network-policy
spec:
  podSelector:
    matchLabels:
      app: code-intelligence-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
```

### Data Security

#### Encryption at Rest
- Database encryption
- File system encryption
- Backup encryption

#### Encryption in Transit
- TLS 1.3 for all connections
- Certificate management with cert-manager
- Internal service mesh encryption

### Access Control

#### RBAC (Kubernetes)
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: code-intelligence-operator
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
```

## Backup and Recovery

### Database Backups

#### Neo4j Backup
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
neo4j-admin backup --backup-dir=/backups --name=backup_$DATE
```

#### PostgreSQL Backup
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h postgres -U postgres code_intelligence > /backups/postgres_$DATE.sql
```

### Disaster Recovery

#### Recovery Procedures
1. **Database Recovery**: Restore from latest backup
2. **Application Recovery**: Redeploy from known good image
3. **Data Validation**: Verify data integrity
4. **Service Validation**: Run health checks

#### RTO/RPO Targets
- **Recovery Time Objective (RTO)**: 4 hours
- **Recovery Point Objective (RPO)**: 1 hour

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker-compose logs api
kubectl logs -l app=code-intelligence-api

# Check configuration
docker-compose config
kubectl describe deployment code-intelligence-api
```

#### Database Connection Issues
```bash
# Test connectivity
docker exec -it postgres psql -U postgres -c "SELECT 1;"
docker exec -it neo4j cypher-shell -u neo4j -p password "RETURN 1;"

# Check network
docker network ls
kubectl get services
```

#### Performance Issues
```bash
# Check resource usage
docker stats
kubectl top pods

# Check metrics
curl http://localhost:8000/api/v1/health/metrics
```

### Debugging Tools

#### Log Analysis
```bash
# Centralized logging with ELK stack
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  elasticsearch:7.17.0

# View logs
kubectl logs -f deployment/code-intelligence-api
```

#### Performance Profiling
```bash
# Enable profiling
export ENABLE_PROFILING=true

# Generate profile
curl http://localhost:8000/debug/pprof/profile
```

## Maintenance

### Regular Maintenance Tasks

#### Daily
- [ ] Check system health
- [ ] Review error logs
- [ ] Monitor resource usage

#### Weekly
- [ ] Update security patches
- [ ] Review performance metrics
- [ ] Clean up old logs

#### Monthly
- [ ] Update dependencies
- [ ] Review backup integrity
- [ ] Capacity planning review

### Update Procedures

#### Rolling Updates (Kubernetes)
```bash
# Update image
kubectl set image deployment/code-intelligence-api api=code-intelligence/api:v1.1.0

# Monitor rollout
kubectl rollout status deployment/code-intelligence-api

# Rollback if needed
kubectl rollout undo deployment/code-intelligence-api
```

#### Blue-Green Deployment
```bash
# Deploy new version
kubectl apply -f deployment-green.yaml

# Switch traffic
kubectl patch service code-intelligence-api -p '{"spec":{"selector":{"version":"green"}}}'

# Clean up old version
kubectl delete -f deployment-blue.yaml
```

## Next Steps

- [Monitoring Guide](../admin-guides/monitoring.md)
- [Security Guide](../admin-guides/security.md)
- [Performance Tuning](../admin-guides/performance.md)
- [Backup Strategy](../admin-guides/backup.md)