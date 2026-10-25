# Local stack (kind + Terraform)

Kubernetes in Docker on your laptop, provisioned and deployed with Terraform: control-plane
API (NodePort 8000), MLflow (5001) and Triton serving the agent's winning model (HTTP 8080,
metrics 8082). Free to run; `make down` deletes the cluster.

Prereqs: Docker Desktop (8 GB RAM, 4 CPUs), `kind`, `kubectl`, `terraform`.

```bash
export EDA_API_TOKEN=... EDA_WORKER_TOKEN=...
make images      # API + MLflow images, pull Triton (~12 GB once)
make up          # export triton_repo from results/latest.json, terraform apply
make status      # pods and services
make bench       # Triton latency + transcription parity vs direct ONNX Runtime
make down        # terraform destroy
```

The Pi worker points at `http://<mac-lan-ip>:8000` exactly as before; the agent logs to
MLflow at `http://127.0.0.1:5001`.

The Kubernetes resources are provider-agnostic; `../nebius` holds the managed-Kubernetes
target, which is written but has not been applied.
