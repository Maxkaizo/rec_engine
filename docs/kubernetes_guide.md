# Kubernetes Deployment Guide

This project includes manifests to deploy the Recommender Engine to a local Kubernetes cluster using **Kind**.

## 1. Prerequisites
Ensure you have the following tools installed:
*   [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
*   [kubectl](https://kubernetes.io/docs/tasks/tools/)

## 1. Automated Setup (Recommended)
If you want to set up everything (cluster, image, metrics, and deployment) with a single command, run:
```bash
chmod +x scripts/setup_k8s.sh
./scripts/setup_k8s.sh
```

---

## 2. Manual Local Practice (Kind)
### Create the Cluster
Run the following command to create a local cluster named `rec-engine-cluster`:
```bash
kind create cluster --name rec-engine-cluster
```

### Build and Load the Image
Since the local cluster cannot access your local Docker daemon directly, you must load the image into the cluster nodes:

1.  **Build the image**:
    ```bash
    docker build -t rec-engine:latest .
    ```
2.  **Load image into Kind**:
    ```bash
    kind load docker-image rec-engine:latest --name rec-engine-cluster
    ```

## 3. Metrics Server Installation (For HPA)
To enable the **Horizontal Pod Autoscaler (HPA)**, the cluster requires a Metrics Server. Deploy and patch it for local development:

```bash
# Install Metrics Server components
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Patch to allow insecure TLS (required for Kind's self-signed certs)
kubectl patch -n kube-system deployment metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

## 4. Deploying the Application

Apply the manifests in the following order:
```bash
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl apply -f deployment/hpa.yaml
```

## 5. Verification and Access

### Check Status
Verify that pods, services, and HPA are running correctly:
```bash
kubectl get pods
kubectl get svc
kubectl get hpa
```

### Accessing the API (Port-Forward)
Since Kind does not provide a physical LoadBalancer IP, use port-forwarding to access the API from your local machine:
```bash
kubectl port-forward service/rec-engine-service 8080:80
```
While the command is running, you can access the API at `http://localhost:8080/docs`.

### Running Integration Tests
Use the provided test script to verify the deployment:
```bash
python src/test_api.py --platform k8s
```

## 6. Advanced Features
*   **Autoscaling**: The HPA is configured to scale from 1 to 5 replicas based on CPU utilization (70% threshold).
*   **Resource Management**: Memory is limited to 1Gi to ensure stability for matrix operations.
