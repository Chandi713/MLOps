## Terraform (Vertex AI fine-tuning infra)

This folder provisions the **baseline GCP resources** used by the fine-tuning code in `temporary/`:

- Vertex AI API enabled
- Artifact Registry (Docker) repository for training images
- (Optional) GCS bucket for training data + model artifacts

### Auth (uses your current project credentials)

Terraform **does not store credentials in `.tf`**. Use either:

- **ADC (recommended)**:

```bash
gcloud auth application-default login
```

- Or a service account key (not recommended long term):

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/abs/path/to/key.json"
```

### Run

```bash
cd temporary/terraform
cp terraform.tfvars.example terraform.tfvars  # edit if needed
terraform init
terraform plan
terraform apply
```

### Notes

- If your bucket already exists, leave `create_bucket=false`. To manage it in state, import it:

```bash
terraform import 'google_storage_bucket.artifacts[0]' <bucket-name>
```

