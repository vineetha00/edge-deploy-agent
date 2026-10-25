# Local stack: kind (Kubernetes in Docker) + API + MLflow + Triton, provisioned by Terraform.
# Prereqs: Docker Desktop, kind, kubectl, terraform. Tokens: export EDA_API_TOKEN/EDA_WORKER_TOKEN.
TF_DIR := infra/terraform/local
TRITON_IMAGE ?= nvcr.io/nvidia/tritonserver:26.08-py3
RUN ?= results/latest.json

.PHONY: images triton-repo up down status logs bench

images:  ## build the API and MLflow images, pull Triton (~12 GB the first time)
	docker build -t eda-api:local .
	docker build -f Dockerfile.mlflow -t eda-mlflow:local .
	docker pull $(TRITON_IMAGE)

triton-repo:  ## export the agent's winning config from $(RUN) as a Triton model repository
	python scripts/export_triton.py --from-run $(RUN) --models-dir models --out triton_repo

up: triton-repo  ## create the cluster and deploy everything
	mkdir -p mlflow-data
	@test -n "$$EDA_API_TOKEN" && test -n "$$EDA_WORKER_TOKEN" || \
	  (echo "export EDA_API_TOKEN and EDA_WORKER_TOKEN first" && exit 1)
	cd $(TF_DIR) && terraform init -upgrade && \
	  TF_VAR_api_token=$$EDA_API_TOKEN TF_VAR_worker_token=$$EDA_WORKER_TOKEN terraform apply

down:  ## delete the cluster (data in mlflow-data/ and triton_repo/ stays on disk)
	cd $(TF_DIR) && TF_VAR_api_token=x TF_VAR_worker_token=x terraform destroy

status:
	kubectl --context kind-eda -n eda get pods,svc

logs:
	kubectl --context kind-eda -n eda logs deploy/triton --tail=50

bench:  ## latency + transcription parity of the Triton-served model
	python scripts/triton_bench.py --url http://127.0.0.1:8080 --iters 20 \
	  --eval-dir data/eval_set --vocab data/vocab.json --n 10
