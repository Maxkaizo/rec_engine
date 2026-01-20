#!/bin/bash

# MovieLens Recommender System - K8s Automated Setup
# This script sets up a local Kind cluster and deploys the application.

set -e

CLUSTER_NAME="rec-engine-cluster"
IMAGE_NAME="rec-engine:latest"

echo "🚀 Starting Kubernetes Automated Setup..."

# 1. Check Prerequisites
command -v kind >/dev/null 2>&1 || { echo >&2 "❌ Kind is not installed. Aborting."; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo >&2 "❌ kubectl is not installed. Aborting."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo >&2 "❌ Docker is not installed. Aborting."; exit 1; }

# 2. Create Kind Cluster
if kind get clusters | grep -q "^$CLUSTER_NAME$"; then
    echo "✅ Cluster '$CLUSTER_NAME' already exists."
else
    echo "Creating Kind cluster '$CLUSTER_NAME'..."
    kind create cluster --name "$CLUSTER_NAME"
fi

# 3. Build Docker Image
echo "📦 Building Docker image '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" .

# 4. Load Image into Kind
echo "🚚 Loading image into cluster nodes..."
kind load docker-image "$IMAGE_NAME" --name "$CLUSTER_NAME"

# 5. Install Metrics Server (For HPA)
echo "📊 Installing Metrics Server..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

echo "Waiting for Metrics Server deployment to exist..."
sleep 2 # Small pause to let K8s register the deployment
kubectl patch -n kube-system deployment metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

# 6. Apply Manifests
echo "🛠 Deploying Application Manifests..."
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl apply -f deployment/hpa.yaml

echo ""
echo "===================================================="
echo "🎉 Setup Complete!"
echo "===================================================="
echo "1. Verify Pods status:"
echo "   kubectl get pods --watch"
echo ""
echo "2. Access the API (Port-Forward):"
echo "   kubectl port-forward service/rec-engine-service 8080:80"
echo ""
echo "3. Run Integration Tests (in another terminal):"
echo "   python src/test_api.py --platform k8s"
echo "===================================================="
