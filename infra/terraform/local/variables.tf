variable "cluster_name" {
  type    = string
  default = "eda"
}

variable "api_image" {
  description = "Control-plane API image, built locally (make images)."
  type        = string
  default     = "eda-api:local"
}

variable "mlflow_image" {
  type    = string
  default = "eda-mlflow:local"
}

variable "triton_image" {
  description = "Triton Inference Server; the py3 tag is multi-arch (amd64 + arm64)."
  type        = string
  default     = "nvcr.io/nvidia/tritonserver:26.08-py3"
}

variable "api_token" {
  description = "Bearer token for /jobs (agent side). Set with TF_VAR_api_token."
  type        = string
  sensitive   = true
}

variable "worker_token" {
  description = "Bearer token for /worker/* (Pi side). Set with TF_VAR_worker_token."
  type        = string
  sensitive   = true
}

variable "api_host_port" {
  description = "Host port for the API; the Pi worker connects to <mac-ip>:<this>."
  type        = number
  default     = 8000
}

variable "mlflow_host_port" {
  description = "Host port for MLflow (not 5000: macOS AirPlay Receiver uses it)."
  type        = number
  default     = 5001
}

variable "triton_host_port" {
  type    = number
  default = 8080
}

variable "triton_metrics_host_port" {
  type    = number
  default = 8082
}
