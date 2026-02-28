import os
import io
import joblib
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set Google credentials from .env if available
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if credentials_path and os.path.exists(credentials_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("google-cloud-storage not installed. Run: pip install google-cloud-storage")


def get_storage_client():
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not installed. Run: pip install google-cloud-storage")
    return storage.Client()


def get_model_version(bucket_name, version_file_name):
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(version_file_name)
    
    if blob.exists():
        version_str = blob.download_as_text()
        return int(version_str.strip())
    return 0


def update_model_version(bucket_name, version_file_name, version):
    if not isinstance(version, int):
        raise ValueError("Version must be an integer")
    
    try:
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(version_file_name)
        blob.upload_from_string(str(version))
        return True
    except Exception as e:
        print(f"Error updating model version: {e}")
        return False


def ensure_folder_exists(bucket, folder_name):
    blob = bucket.blob(f"{folder_name}/")
    if not blob.exists():
        blob.upload_from_string('')
        print(f"Created folder: {folder_name}")


def save_model_to_gcs(model, bucket_name, degree, version=None):
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    ensure_folder_exists(bucket, "trained_models")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    version_str = f"_v{version}" if version else ""
    blob_name = f"trained_models/polynomial{version_str}_degree{degree}_{timestamp}.pkl"
    
    blob = bucket.blob(blob_name)
    buffer = io.BytesIO()
    model_to_save = model.model if hasattr(model, 'model') else model
    joblib.dump(model_to_save, buffer)
    buffer.seek(0)
    
    blob.upload_from_file(buffer, content_type='application/octet-stream')
    
    return blob_name


def download_model_from_gcs(bucket_name, blob_name, local_path=None):
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    if local_path:
        blob.download_to_filename(local_path)
        return local_path
    else:
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        return joblib.load(buffer)


def list_models_in_gcs(bucket_name, prefix="trained_models/"):
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs if blob.name.endswith('.pkl')]
