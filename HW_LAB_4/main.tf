terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  artifact_repo_image_prefix = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo}"
}

# Enable required APIs
resource "google_project_service" "vertex_ai" {
  service            = "aiplatform.googleapis.com"
  disable_on_destroy = true
}

resource "google_project_service" "artifact_registry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = true
}

resource "google_project_service" "storage" {
  service                    = "storage.googleapis.com"
  disable_on_destroy         = true
  disable_dependent_services = true
}


# Artifact Registry (Docker)
resource "google_artifact_registry_repository" "training" {
  count         = var.create_artifact_registry_repo ? 1 : 0
  location      = var.region
  repository_id = var.artifact_registry_repo
  description   = "Training containers (e.g. llama3-lora)"
  format        = "DOCKER"

  depends_on = [google_project_service.artifact_registry]
}


resource "google_storage_bucket" "artifacts" {
  count                       = var.create_bucket ? 1 : 0
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.storage]
}

