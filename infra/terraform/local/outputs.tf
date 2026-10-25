output "api_url" {
  value = "http://127.0.0.1:${var.api_host_port} (Pi worker: http://<mac-lan-ip>:${var.api_host_port})"
}

output "mlflow_url" {
  value = "http://127.0.0.1:${var.mlflow_host_port}"
}

output "triton_url" {
  value = "http://127.0.0.1:${var.triton_host_port}"
}

output "triton_metrics_url" {
  value = "http://127.0.0.1:${var.triton_metrics_host_port}/metrics"
}

output "kubeconfig_path" {
  value = kind_cluster.this.kubeconfig_path
}
