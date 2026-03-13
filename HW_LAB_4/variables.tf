variable "project_id" {
  type        = string
  description = "GCP project id."
  default     = "researchlineage-488918"
}

variable "region" {
  type        = string
  description = "GCP region."
  default     = "us-central1"
}

variable "bucket_name" {
  type        = string
  description = "GCS bucket used for fine-tuning data + artifacts."
  default     = "test-bucket-cs-3"
}

variable "artifact_registry_repo" {
  type        = string
  description = "Artifact Registry repository id (Docker format)."
  default     = "training"
}

variable "create_bucket" {
  type        = bool
  description = "If true, create the GCS bucket. If false, assume it already exists."
  default     = true
}

variable "create_artifact_registry_repo" {
  type        = bool
  description = "If true, create the Artifact Registry repository. If false, assume it already exists."
  default     = true
}

