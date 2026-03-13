output "artifact_registry_repo" {
  value       = var.artifact_registry_repo
  description = "Artifact Registry repository id."
}

output "artifact_registry_image_prefix" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo}"
  description = "Artifact Registry image prefix for docker push/pull."
}

output "bucket_name" {
  value       = var.bucket_name
  description = "GCS bucket name for data/artifacts."
}

