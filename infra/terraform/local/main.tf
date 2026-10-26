# Local stack on kind (Kubernetes in Docker): control-plane API, MLflow and Triton.
# Costs nothing; `terraform destroy` removes the cluster. The same Kubernetes resources
# would run unchanged on a managed cluster (see ../nebius for the cloud target).

locals {
  repo_root   = abspath("${path.module}/../../..")
  triton_repo = "${local.repo_root}/triton_repo"
  mlflow_data = "${local.repo_root}/mlflow-data"
  ns          = "eda"
}

resource "kind_cluster" "this" {
  name           = var.cluster_name
  wait_for_ready = true

  kind_config {
    kind        = "Cluster"
    api_version = "kind.x-k8s.io/v1alpha4"

    node {
      role = "control-plane"

      # NodePorts below are published on the Mac so the Pi and the agent can reach them.
      extra_port_mappings {
        container_port = 30080
        host_port      = var.api_host_port
      }
      extra_port_mappings {
        container_port = 30500
        host_port      = var.mlflow_host_port
      }
      extra_port_mappings {
        container_port = 30800
        host_port      = var.triton_host_port
      }
      extra_port_mappings {
        container_port = 30802
        host_port      = var.triton_metrics_host_port
      }

      # Model repository exported from the agent's winning config, and MLflow's data.
      extra_mounts {
        host_path      = local.triton_repo
        container_path = "/models"
      }
      extra_mounts {
        host_path      = local.mlflow_data
        container_path = "/mlflow-data"
      }
    }
  }
}

# Our two images are built on the Mac (make images) and copied into the kind node.
# (The kind provider's `kind_load` resource is not in a released version yet, so this uses
# Terraform's built-in terraform_data + the kind CLI; it re-runs when the cluster or image changes.)
# Triton is NOT copied: `kind load` of a multi-arch image from Docker Desktop's containerd
# store fails ("content digest ... not found"), so the node pulls it from nvcr.io itself.
locals {
  images = {
    api    = var.api_image
    mlflow = var.mlflow_image
  }
  # Content hashes of what goes into each image: a rebuilt image keeps its tag, so these
  # are what tell Terraform to reload it into the node and roll the pods.
  image_hash = {
    api = sha1(join("", concat(
      [filesha1("${local.repo_root}/Dockerfile"), filesha1("${local.repo_root}/pyproject.toml")],
      [for f in sort(fileset("${local.repo_root}/src", "**/*.py")) : filesha1("${local.repo_root}/src/${f}")]
    )))
    mlflow = filesha1("${local.repo_root}/Dockerfile.mlflow")
  }
}

resource "terraform_data" "load_image" {
  for_each         = local.images
  triggers_replace = [kind_cluster.this.id, each.value, local.image_hash[each.key]]

  provisioner "local-exec" {
    command = "kind load docker-image ${each.value} --name ${kind_cluster.this.name}"
  }
}

resource "kubernetes_namespace_v1" "eda" {
  metadata {
    name = local.ns
  }
}

resource "kubernetes_secret_v1" "tokens" {
  metadata {
    name      = "eda-tokens"
    namespace = kubernetes_namespace_v1.eda.metadata[0].name
  }
  data = {
    EDA_API_TOKEN    = var.api_token
    EDA_WORKER_TOKEN = var.worker_token
  }
}

# ---------------------------------------------------------------- control-plane API
resource "kubernetes_deployment_v1" "api" {
  depends_on = [terraform_data.load_image["api"]]
  metadata {
    name      = "api"
    namespace = local.ns
    labels    = { app = "api" }
  }
  spec {
    replicas = 1
    selector {
      match_labels = { app = "api" }
    }
    template {
      metadata {
        labels      = { app = "api" }
        annotations = { "eda/image-hash" = local.image_hash["api"] }
      }
      spec {
        container {
          name              = "api"
          image             = var.api_image
          image_pull_policy = "Never"
          port {
            container_port = 8000
          }
          env_from {
            secret_ref {
              name = kubernetes_secret_v1.tokens.metadata[0].name
            }
          }
          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            period_seconds = 5
          }
          resources {
            requests = { cpu = "100m", memory = "128Mi" }
            limits   = { memory = "512Mi" }
          }
        }
      }
    }
  }
}

resource "kubernetes_service_v1" "api" {
  metadata {
    name      = "api"
    namespace = local.ns
  }
  spec {
    type     = "NodePort"
    selector = { app = "api" }
    port {
      port        = 8000
      target_port = 8000
      node_port   = 30080
    }
  }
}

# ---------------------------------------------------------------- MLflow
resource "kubernetes_deployment_v1" "mlflow" {
  depends_on = [terraform_data.load_image["mlflow"]]
  metadata {
    name      = "mlflow"
    namespace = local.ns
    labels    = { app = "mlflow" }
  }
  spec {
    replicas = 1
    selector {
      match_labels = { app = "mlflow" }
    }
    template {
      metadata {
        labels      = { app = "mlflow" }
        annotations = { "eda/image-hash" = local.image_hash["mlflow"] }
      }
      spec {
        container {
          name              = "mlflow"
          image             = var.mlflow_image
          image_pull_policy = "Never"
          port {
            container_port = 5000
          }
          volume_mount {
            name       = "data"
            mount_path = "/mlflow-data"
          }
          readiness_probe {
            http_get {
              path = "/health"
              port = 5000
            }
            period_seconds = 10
          }
          resources {
            requests = { cpu = "100m", memory = "512Mi" }
            limits   = { memory = "1536Mi" }
          }
        }
        volume {
          name = "data"
          host_path {
            path = "/mlflow-data"
            type = "Directory"
          }
        }
      }
    }
  }
}

resource "kubernetes_service_v1" "mlflow" {
  metadata {
    name      = "mlflow"
    namespace = local.ns
  }
  spec {
    type     = "NodePort"
    selector = { app = "mlflow" }
    port {
      port        = 5000
      target_port = 5000
      node_port   = 30500
    }
  }
}

# ---------------------------------------------------------------- Triton (CPU)
resource "kubernetes_deployment_v1" "triton" {
  metadata {
    name      = "triton"
    namespace = local.ns
    labels    = { app = "triton" }
  }
  spec {
    replicas = 1
    selector {
      match_labels = { app = "triton" }
    }
    template {
      metadata {
        labels = { app = "triton" }
      }
      spec {
        container {
          name              = "triton"
          image             = var.triton_image
          image_pull_policy = "IfNotPresent"
          args              = ["tritonserver", "--model-repository=/models", "--strict-readiness=true"]
          port {
            name           = "http"
            container_port = 8000
          }
          port {
            name           = "metrics"
            container_port = 8002
          }
          volume_mount {
            name       = "models"
            mount_path = "/models"
            read_only  = true
          }
          readiness_probe {
            http_get {
              path = "/v2/health/ready"
              port = 8000
            }
            initial_delay_seconds = 10
            period_seconds        = 10
          }
          resources {
            requests = { cpu = "1", memory = "1Gi" }
            limits   = { memory = "4Gi" }
          }
        }
        volume {
          name = "models"
          host_path {
            path = "/models"
            type = "Directory"
          }
        }
      }
    }
  }
  timeouts {
    create = "40m" # first start pulls the ~12 GB image inside the node
    update = "40m"
  }
}

resource "kubernetes_service_v1" "triton" {
  metadata {
    name      = "triton"
    namespace = local.ns
  }
  spec {
    type     = "NodePort"
    selector = { app = "triton" }
    port {
      name        = "http"
      port        = 8000
      target_port = 8000
      node_port   = 30800
    }
    port {
      name        = "metrics"
      port        = 8002
      target_port = 8002
      node_port   = 30802
    }
  }
}
