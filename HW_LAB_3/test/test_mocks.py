import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


class TestGCSMocks:
    def test_get_model_version_exists(self):
        with patch('src.gcs_utils.storage.Client') as mock_client:
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            
            mock_client.return_value.bucket.return_value = mock_bucket
            mock_bucket.blob.return_value = mock_blob
            mock_blob.exists.return_value = True
            mock_blob.download_as_text.return_value = '5'
            
            from src.gcs_utils import get_model_version
            
            version = get_model_version("test-bucket", "version.txt")
            
            assert version == 5
            mock_client.return_value.bucket.assert_called_once_with("test-bucket")
            mock_bucket.blob.assert_called_once_with("version.txt")

    def test_get_model_version_not_exists(self):
        with patch('src.gcs_utils.storage.Client') as mock_client:
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            
            mock_client.return_value.bucket.return_value = mock_bucket
            mock_bucket.blob.return_value = mock_blob
            mock_blob.exists.return_value = False
            
            from src.gcs_utils import get_model_version
            
            version = get_model_version("test-bucket", "version.txt")
            
            assert version == 0
            mock_blob.download_as_text.assert_not_called()

    def test_update_model_version_success(self):
        with patch('src.gcs_utils.storage.Client') as mock_client:
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            
            mock_client.return_value.bucket.return_value = mock_bucket
            mock_bucket.blob.return_value = mock_blob
            
            from src.gcs_utils import update_model_version
            
            result = update_model_version("test-bucket", "version.txt", 3)
            
            assert result == True
            mock_blob.upload_from_string.assert_called_once_with('3')

    def test_update_model_version_invalid_type(self):
        from src.gcs_utils import update_model_version
        
        with pytest.raises(ValueError):
            update_model_version("test-bucket", "version.txt", "invalid")

    def test_save_model_to_gcs(self):
        with patch('src.gcs_utils.storage.Client') as mock_client:
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            
            mock_client.return_value.bucket.return_value = mock_bucket
            mock_bucket.blob.return_value = mock_blob
            mock_blob.exists.return_value = True
            
            from src.gcs_utils import save_model_to_gcs
            from sklearn.linear_model import LinearRegression
            
            model = LinearRegression()
            result = save_model_to_gcs(model, "test-bucket", degree=2, version=1)
            
            assert "trained_models/" in result
            assert "degree2" in result
            mock_blob.upload_from_file.assert_called_once()

    def test_ensure_folder_exists_creates_folder(self):
        with patch('src.gcs_utils.storage.Client') as mock_client:
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            
            mock_bucket.blob.return_value = mock_blob
            mock_blob.exists.return_value = False
            
            from src.gcs_utils import ensure_folder_exists
            
            ensure_folder_exists(mock_bucket, "test_folder")
            
            mock_bucket.blob.assert_called_with("test_folder/")
            mock_blob.upload_from_string.assert_called_once_with('')

    def test_ensure_folder_exists_already_exists(self):
        with patch('src.gcs_utils.storage.Client') as mock_client:
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            
            mock_bucket.blob.return_value = mock_blob
            mock_blob.exists.return_value = True
            
            from src.gcs_utils import ensure_folder_exists
            
            ensure_folder_exists(mock_bucket, "test_folder")
            
            mock_blob.upload_from_string.assert_not_called()


