# Docker Guide for Kosmic Lab

This guide explains how to run Kosmic Lab in Docker containers for easy deployment and reproducibility.

---

## Quick Start

### 1. Build the Image

```bash
docker build -t kosmic-lab:latest .
```

### 2. Run Hello Kosmic Example

```bash
docker run --rm kosmic-lab:latest python examples/01_hello_kosmic.py
```

### 3. Run with Data Persistence

```bash
docker run --rm \
  -v $(pwd)/logs:/logs \
  -v $(pwd)/data:/data \
  kosmic-lab:latest \
  python examples/01_hello_kosmic.py
```

---

## Using Docker Compose

Docker Compose simplifies multi-container setups.

### Basic Usage

```bash
# Run example
docker-compose up kosmic-lab

# Run dashboard (in background)
docker-compose --profile dashboard up -d dashboard

# View logs
docker-compose logs -f kosmic-lab

# Stop all services
docker-compose down
```

### Run Experiments

```bash
# Interactive shell
docker-compose run kosmic-lab bash

# Inside container:
make fre-run
make dashboard
make help
```

### Run Specific Commands

```bash
# Run tests
docker-compose run kosmic-lab make test

# Generate analysis
docker-compose run kosmic-lab make notebook

# View coverage
docker-compose run kosmic-lab make coverage
```

---

## Production Deployment

### Environment Variables

Create `.env` file:

```env
LOG_LEVEL=INFO
DASHBOARD_PORT=8050
DATA_DIR=/data
LOGS_DIR=/logs
```

Use in docker-compose:

```yaml
env_file:
  - .env
```

### Persistent Storage

Mount volumes for data persistence:

```bash
docker run -d \
  --name kosmic-lab \
  -v kosmic-data:/data \
  -v kosmic-logs:/logs \
  -p 8050:8050 \
  kosmic-lab:latest
```

### Dashboard in Production

```bash
# Run dashboard as service
docker-compose --profile dashboard up -d

# Access at http://localhost:8050
```

---

## Kubernetes Deployment

### Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kosmic-lab
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kosmic-lab
  template:
    metadata:
      labels:
        app: kosmic-lab
    spec:
      containers:
      - name: kosmic-lab
        image: kosmic-lab:latest
        ports:
        - containerPort: 8050
        volumeMounts:
        - name: data
          mountPath: /data
        - name: logs
          mountPath: /logs
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: kosmic-data
      - name: logs
        persistentVolumeClaim:
          claimName: kosmic-logs
---
apiVersion: v1
kind: Service
metadata:
  name: kosmic-lab
spec:
  selector:
    app: kosmic-lab
  ports:
  - port: 8050
    targetPort: 8050
  type: LoadBalancer
```

Apply:

```bash
kubectl apply -f k8s/deployment.yaml
```

---

## Mycelix Integration

Run with Holochain for decentralized storage:

```bash
# Start Holochain node
docker-compose --profile mycelix up -d holochain

# Run kosmic-lab with Mycelix
docker-compose run kosmic-lab make holochain-publish
```

---

## Development with Docker

### Development Image

```dockerfile
# Dockerfile.dev
FROM kosmic-lab:latest

# Install dev dependencies
RUN poetry install --with dev

# Enable hot reload
ENV FLASK_ENV=development
```

Build and run:

```bash
docker build -f Dockerfile.dev -t kosmic-lab:dev .
docker run -v $(pwd):/app kosmic-lab:dev
```

### VS Code DevContainer

Create `.devcontainer/devcontainer.json`:

```json
{
  "name": "Kosmic Lab",
  "dockerFile": "../Dockerfile.dev",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance"
      ]
    }
  },
  "postCreateCommand": "poetry install --with dev"
}
```

---

## Troubleshooting

### Issue: Permission denied on logs/

**Solution**:
```bash
# Fix permissions
docker run --rm -v $(pwd)/logs:/logs kosmic-lab:latest \
  bash -c "chown -R $(id -u):$(id -g) /logs"
```

### Issue: Out of memory

**Solution**:
```bash
# Increase Docker memory limit
docker run --memory=4g kosmic-lab:latest
```

### Issue: Cannot connect to dashboard

**Solution**:
```bash
# Ensure port is exposed
docker run -p 8050:8050 kosmic-lab:latest make dashboard

# Check if port is in use
lsof -i :8050
```

---

## Best Practices

1. **Use specific tags**: `kosmic-lab:1.0.0` instead of `latest`
2. **Multi-stage builds**: Separate builder and runtime stages
3. **Non-root user**: Run as `kosmic` user for security
4. **Health checks**: Monitor container health
5. **Volume mounts**: Persist data and logs
6. **Resource limits**: Set CPU and memory limits
7. **Logging**: Use structured logging
8. **Secrets**: Use Docker secrets, not environment variables

---

## Image Registry

### Push to Registry

```bash
# Tag for registry
docker tag kosmic-lab:latest registry.example.com/kosmic-lab:1.0.0

# Push
docker push registry.example.com/kosmic-lab:1.0.0
```

### Pull from Registry

```bash
docker pull registry.example.com/kosmic-lab:1.0.0
```

---

## CI/CD Integration

### GitHub Actions

```yaml
- name: Build Docker image
  run: docker build -t kosmic-lab:${{ github.sha }} .

- name: Run tests in container
  run: docker run kosmic-lab:${{ github.sha }} make test

- name: Push to registry
  run: |
    docker tag kosmic-lab:${{ github.sha }} registry/kosmic-lab:latest
    docker push registry/kosmic-lab:latest
```

---

**For more help**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an issue.
