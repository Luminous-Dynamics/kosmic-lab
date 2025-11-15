# Deployment Guide

**Version**: 1.1.0
**Last Updated**: 2025-11-15
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Deployment Options](#deployment-options)
4. [Docker Deployment](#docker-deployment)
5. [Bare Metal Deployment](#bare-metal-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [HPC/Cluster Deployment](#hpccluster-deployment)
8. [Configuration](#configuration)
9. [Monitoring & Logging](#monitoring--logging)
10. [Backup & Recovery](#backup--recovery)
11. [Security Hardening](#security-hardening)
12. [Troubleshooting](#troubleshooting)
13. [Production Checklist](#production-checklist)

---

## Overview

Kosmic Lab can be deployed in multiple environments:

- **Local/Development**: Quick testing and development
- **Docker**: Containerized single-node deployment
- **Bare Metal**: Direct installation on servers
- **Cloud**: AWS, GCP, Azure deployments
- **HPC**: High-Performance Computing clusters (Slurm, PBS)
- **Kubernetes**: Orchestrated multi-node deployment (coming soon)

---

## Quick Start

### Option 1: Docker (Recommended for Production)

```bash
# Clone repository
git clone https://github.com/Luminous-Dynamics/kosmic-lab.git
cd kosmic-lab

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Build and run
docker-compose up -d

# Verify
docker-compose ps
docker-compose logs -f
```

### Option 2: Bare Metal

```bash
# Clone repository
git clone https://github.com/Luminous-Dynamics/kosmic-lab.git
cd kosmic-lab

# Run setup
./scripts/setup_dev_env.sh

# Configure
cp .env.example .env
# Edit .env with production settings

# Run experiments
poetry run python fre/track_b_runner.py
```

---

## Deployment Options

### Comparison Table

| Option | Complexity | Isolation | Scalability | Best For |
|--------|------------|-----------|-------------|----------|
| **Local** | Low | None | Single node | Development, testing |
| **Docker** | Medium | Container | Single node | Production single-node |
| **Bare Metal** | Low | None | Single node | Dedicated servers |
| **Cloud** | Medium | VM | Multi-node | Elastic workloads |
| **HPC** | High | Job queue | Cluster | Large-scale simulations |
| **Kubernetes** | High | Container | Cluster | Enterprise orchestration |

---

## Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum (8GB+ recommended)
- 10GB disk space

### Build and Run

**1. Build Docker image**:
```bash
docker build -t kosmic-lab:1.1.0 .
```

**2. Run with Docker Compose**:
```bash
docker-compose up -d
```

**3. Run experiments**:
```bash
# Track B (SAC Controller)
docker-compose exec kosmic-lab poetry run python fre/track_b_runner.py

# Track C (Bioelectric Rescue)
docker-compose exec kosmic-lab poetry run python fre/track_c_runner.py

# Dashboard
docker-compose exec kosmic-lab poetry run python scripts/kosmic_dashboard.py
```

**4. Access dashboard**:
```
http://localhost:8050
```

### Volume Mounts

By default, Docker Compose mounts:
- `./data:/app/data` - Experiment data
- `./logs:/app/logs` - Log files
- `./results:/app/results` - K-Codex records

### Customization

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  kosmic-lab:
    build: .
    container_name: kosmic-lab
    restart: unless-stopped
    environment:
      - LOG_LEVEL=INFO
      - N_CORES=4
      - USE_GPU=false
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./results:/app/results
    ports:
      - "8050:8050"  # Dashboard
    command: poetry run python scripts/kosmic_dashboard.py
```

### Docker Best Practices

1. **Use multi-stage builds** (already in Dockerfile)
2. **Pin versions** in requirements
3. **Don't run as root** (use non-root user)
4. **Limit resources**:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '4'
         memory: 8G
   ```
5. **Health checks**:
   ```yaml
   healthcheck:
     test: ["CMD", "poetry", "run", "python", "-c", "import core.logging_config"]
     interval: 30s
     timeout: 10s
     retries: 3
   ```

---

## Bare Metal Deployment

### Prerequisites

- Ubuntu 20.04+ / CentOS 8+ / macOS 11+
- Python 3.10+
- Poetry 1.5+
- Git
- 8GB+ RAM for production workloads

### Installation

**1. System dependencies**:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip git build-essential

# CentOS/RHEL
sudo yum install -y python310 python310-devel git gcc

# macOS (Homebrew)
brew install python@3.10 poetry git
```

**2. Clone and setup**:

```bash
git clone https://github.com/Luminous-Dynamics/kosmic-lab.git
cd kosmic-lab
./scripts/setup_dev_env.sh
```

**3. Configure**:

```bash
cp .env.example .env

# Edit production settings
nano .env
```

**4. Verify**:

```bash
make test
./scripts/check_code_quality.sh
```

### Systemd Service (Linux)

Create `/etc/systemd/system/kosmic-dashboard.service`:

```ini
[Unit]
Description=Kosmic Lab Dashboard
After=network.target

[Service]
Type=simple
User=kosmic
WorkingDirectory=/opt/kosmic-lab
Environment="PATH=/opt/kosmic-lab/.venv/bin:/usr/bin"
ExecStart=/opt/kosmic-lab/.venv/bin/poetry run python scripts/kosmic_dashboard.py
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

**Enable and start**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable kosmic-dashboard
sudo systemctl start kosmic-dashboard
sudo systemctl status kosmic-dashboard
```

---

## Cloud Deployment

### AWS EC2

**1. Launch instance**:
- **AMI**: Ubuntu 22.04 LTS
- **Instance type**: t3.xlarge (4 vCPU, 16GB RAM)
- **Storage**: 50GB gp3
- **Security group**: Allow SSH (22), HTTP (8050)

**2. SSH and setup**:
```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Clone and setup
git clone https://github.com/Luminous-Dynamics/kosmic-lab.git
cd kosmic-lab
./scripts/setup_dev_env.sh
```

**3. Configure for production**:
```bash
cp .env.example .env
nano .env

# Set production values
LOG_LEVEL=INFO
N_CORES=4
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
```

**4. Run with screen/tmux**:
```bash
screen -S kosmic-dashboard
poetry run python scripts/kosmic_dashboard.py
# Detach with Ctrl+A, D
```

### AWS ECS (Elastic Container Service)

**1. Push Docker image to ECR**:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com

docker build -t kosmic-lab:1.1.0 .
docker tag kosmic-lab:1.1.0 your-account.dkr.ecr.us-east-1.amazonaws.com/kosmic-lab:1.1.0
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/kosmic-lab:1.1.0
```

**2. Create ECS task definition** (`task-definition.json`):
```json
{
  "family": "kosmic-lab",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "kosmic-lab",
      "image": "your-account.dkr.ecr.us-east-1.amazonaws.com/kosmic-lab:1.1.0",
      "portMappings": [
        {
          "containerPort": 8050,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "LOG_LEVEL", "value": "INFO"},
        {"name": "N_CORES", "value": "4"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/kosmic-lab",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**3. Deploy**:
```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cluster your-cluster --service-name kosmic-lab --task-definition kosmic-lab --desired-count 1
```

### Google Cloud Platform (GCP)

**1. Create VM**:
```bash
gcloud compute instances create kosmic-lab \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB
```

**2. SSH and setup**:
```bash
gcloud compute ssh kosmic-lab --zone=us-central1-a

# Same setup as AWS EC2
git clone https://github.com/Luminous-Dynamics/kosmic-lab.git
cd kosmic-lab
./scripts/setup_dev_env.sh
```

### Azure

**1. Create VM**:
```bash
az vm create \
  --resource-group kosmic-rg \
  --name kosmic-lab \
  --image UbuntuLTS \
  --size Standard_D4s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys
```

**2. SSH and setup**:
```bash
ssh azureuser@your-vm-ip

# Same setup process
```

---

## HPC/Cluster Deployment

### Slurm

**1. Create job script** (`slurm_job.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=kosmic-lab
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=compute

# Load modules
module load python/3.10
module load poetry/1.5

# Activate environment
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# Run experiment
poetry run python fre/track_b_runner.py --config configs/production.yaml

# Collect results
cp logs/*.json results/
```

**2. Submit job**:
```bash
sbatch slurm_job.sh
```

**3. Monitor**:
```bash
squeue -u $USER
sacct -j <job-id>
```

### PBS/Torque

**1. Create PBS script** (`pbs_job.sh`):

```bash
#!/bin/bash
#PBS -N kosmic-lab
#PBS -l nodes=1:ppn=16
#PBS -l mem=32gb
#PBS -l walltime=24:00:00
#PBS -o logs/pbs-$PBS_JOBID.out
#PBS -e logs/pbs-$PBS_JOBID.err

cd $PBS_O_WORKDIR
source .venv/bin/activate

poetry run python fre/track_b_runner.py
```

**2. Submit**:
```bash
qsub pbs_job.sh
```

---

## Configuration

### Environment Variables

See `.env.example` for all available options. Key production settings:

```bash
# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/kosmic-lab/app.log
LOG_COLORED=false  # Disable for file logging

# Performance
N_CORES=8  # Match your CPU count
USE_GPU=false  # Enable if GPU available
MEMORY_LIMIT_GB=16

# Dashboard
DASHBOARD_HOST=0.0.0.0  # Allow external connections
DASHBOARD_PORT=8050
DASHBOARD_DEBUG=false  # Disable in production

# Reproducibility
GIT_SHA_TRACKING=true
DEFAULT_SEED=42

# External Services
HOLOCHAIN_CONDUCTOR_URL=http://localhost:8888  # If using Mycelix
```

### Production vs Development

**Development** (`.env.dev`):
```bash
LOG_LEVEL=DEBUG
LOG_COLORED=true
DASHBOARD_DEBUG=true
PYTHONWARNINGS=default
```

**Production** (`.env.prod`):
```bash
LOG_LEVEL=INFO
LOG_COLORED=false
DASHBOARD_DEBUG=false
PYTHONWARNINGS=ignore
PYTHONOPTIMIZE=2
```

---

## Monitoring & Logging

### Logging

**1. Centralized logging** (already configured in `core/logging_config.py`):

```python
from core.logging_config import setup_logging

setup_logging(
    level="INFO",
    log_file="/var/log/kosmic-lab/app.log",
    colored=False  # Production
)
```

**2. Log rotation** (using `logrotate`):

Create `/etc/logrotate.d/kosmic-lab`:
```
/var/log/kosmic-lab/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 kosmic kosmic
    sharedscripts
    postrotate
        systemctl reload kosmic-dashboard
    endscript
}
```

### Metrics

**1. Performance monitoring** (custom):

```bash
# Run benchmarks periodically
make benchmarks-save

# Check results
ls -l benchmarks/results/
```

**2. System monitoring** (Prometheus + Grafana - Future):

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'kosmic-lab'
    static_configs:
      - targets: ['localhost:9090']
```

### Health Checks

**1. Simple health check script** (`scripts/health_check.sh`):

```bash
#!/bin/bash
set -e

# Check Python imports
poetry run python -c "import core.logging_config; import fre.metrics.k_index"

# Check K-Index computation
poetry run python -c "
from fre.metrics.k_index import k_index
import numpy as np
rng = np.random.default_rng(42)
x = rng.random(100)
y = rng.random(100)
k = k_index(x, y)
assert 0 <= k <= 1, f'K-Index out of range: {k}'
print('✓ Health check passed')
"
```

**2. Run periodically** (cron):
```bash
# Check every 5 minutes
*/5 * * * * /opt/kosmic-lab/scripts/health_check.sh >> /var/log/kosmic-lab/health.log 2>&1
```

---

## Backup & Recovery

### What to Backup

1. **K-Codex records** (`logs/*.json`, `results/*.json`)
2. **Experiment data** (`data/`)
3. **Configuration** (`.env`, `configs/`)
4. **Custom code** (if modified)

### Backup Script

**scripts/backup.sh**:
```bash
#!/bin/bash
set -e

BACKUP_DIR="/backups/kosmic-lab"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/kosmic-lab-$DATE.tar.gz"

mkdir -p "$BACKUP_DIR"

# Backup essential data
tar -czf "$BACKUP_FILE" \
    logs/ \
    results/ \
    data/ \
    .env \
    configs/

# Remove backups older than 30 days
find "$BACKUP_DIR" -name "kosmic-lab-*.tar.gz" -mtime +30 -delete

echo "✓ Backup completed: $BACKUP_FILE"
```

### Automated Backups (cron)

```bash
# Daily backup at 2 AM
0 2 * * * /opt/kosmic-lab/scripts/backup.sh
```

### Cloud Backups

**AWS S3**:
```bash
# Install AWS CLI
pip install awscli

# Sync to S3
aws s3 sync logs/ s3://your-bucket/kosmic-lab/logs/
aws s3 sync results/ s3://your-bucket/kosmic-lab/results/
```

**GCP Cloud Storage**:
```bash
# Install gcloud
# ...

# Sync to GCS
gsutil -m rsync -r logs/ gs://your-bucket/kosmic-lab/logs/
```

---

## Security Hardening

### 1. File Permissions

```bash
# Set restrictive permissions
chmod 600 .env
chmod 700 scripts/*.sh
chmod 755 /opt/kosmic-lab

# Create dedicated user
sudo useradd -r -s /bin/bash -d /opt/kosmic-lab kosmic
sudo chown -R kosmic:kosmic /opt/kosmic-lab
```

### 2. Firewall

**UFW (Ubuntu)**:
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8050/tcp  # Dashboard
sudo ufw enable
```

**firewalld (CentOS)**:
```bash
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-port=8050/tcp
sudo firewall-cmd --reload
```

### 3. HTTPS/TLS

Use a reverse proxy (nginx) with Let's Encrypt:

**nginx.conf**:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. Security Scanning

```bash
# Pre-commit hooks (already configured)
poetry run pre-commit run bandit --all-files

# Dependency vulnerabilities
poetry run pip-audit

# Docker image scanning
docker scan kosmic-lab:1.1.0
```

### 5. Access Control

**Authentication** (if exposing dashboard):
- Use VPN (WireGuard, OpenVPN)
- Add authentication middleware
- Restrict by IP whitelist

---

## Troubleshooting

### Issue: Dashboard won't start

**Symptoms**: `poetry run python scripts/kosmic_dashboard.py` fails

**Solutions**:
1. Check port availability: `lsof -i :8050`
2. Check dependencies: `poetry install`
3. Check logs: `tail -f logs/app.log`
4. Try different port: `DASHBOARD_PORT=8051 poetry run python scripts/kosmic_dashboard.py`

### Issue: Out of memory

**Symptoms**: Process killed, `Killed` message

**Solutions**:
1. Reduce `N_CORES`: `export N_CORES=2`
2. Reduce sample size: Use smaller `n_samples` in config
3. Increase system memory
4. Use swap file:
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Issue: Slow performance

**Symptoms**: Experiments taking too long

**Solutions**:
1. Enable parallelization: `export N_CORES=4`
2. Run benchmarks: `make benchmarks`
3. Profile code: `python -m cProfile -o profile.stats your_script.py`
4. Check CPU/memory usage: `htop`

### Issue: Import errors

**Symptoms**: `ModuleNotFoundError`

**Solutions**:
1. Activate environment: `poetry shell`
2. Install dependencies: `poetry install`
3. Check Python version: `python --version` (must be 3.10+)

---

## Production Checklist

Before deploying to production:

### Pre-Deployment
- [ ] Run full test suite: `make test`
- [ ] Run code quality checks: `./scripts/check_code_quality.sh`
- [ ] Run security scan: `poetry run bandit -r core/ fre/`
- [ ] Update dependencies: `poetry update`
- [ ] Review `.env` configuration
- [ ] Set production log level (INFO)
- [ ] Disable debug mode
- [ ] Test backup/restore procedure

### Infrastructure
- [ ] Set up monitoring (logs, metrics)
- [ ] Configure automated backups
- [ ] Set up log rotation
- [ ] Configure firewall rules
- [ ] Enable HTTPS/TLS (if exposing publicly)
- [ ] Create systemd service (Linux)
- [ ] Set up health checks

### Security
- [ ] Review SECURITY.md
- [ ] Scan for vulnerabilities: `poetry run bandit`
- [ ] Update dependencies: Check Dependabot PRs
- [ ] Verify `.gitignore` protects secrets
- [ ] Set restrictive file permissions
- [ ] Enable firewall
- [ ] Review access controls

### Documentation
- [ ] Update CHANGELOG.md
- [ ] Document deployment specifics
- [ ] Create runbook for operations
- [ ] Document incident response procedures

### Post-Deployment
- [ ] Verify all services running
- [ ] Check logs for errors
- [ ] Run health check: `./scripts/health_check.sh`
- [ ] Test backup procedure
- [ ] Monitor performance metrics
- [ ] Document lessons learned

---

## Additional Resources

- [DOCKER.md](DOCKER.md) - Detailed Docker guide
- [SECURITY.md](SECURITY.md) - Security policy
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development guide
- [FAQ.md](FAQ.md) - Frequently asked questions

---

## Support

For deployment support:
- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/Luminous-Dynamics/kosmic-lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Luminous-Dynamics/kosmic-lab/discussions)

---

*Last updated: 2025-11-15 | Version: 1.1.0*
