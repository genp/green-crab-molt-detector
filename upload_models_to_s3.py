#!/usr/bin/env python3
"""
Upload model files to S3 for Heroku deployment.

Usage:
    python upload_models_to_s3.py <bucket_name>
    
Environment variables needed:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION (optional, defaults to us-east-1)
"""

import sys
import os
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_file_to_s3(bucket_name: str, local_path: str, s3_key: str) -> bool:
    """Upload a file to S3."""
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_path, bucket_name, s3_key)
        logger.info(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
        return True
    except FileNotFoundError:
        logger.error(f"Local file {local_path} not found")
        return False
    except NoCredentialsError:
        logger.error("AWS credentials not found")
        return False
    except ClientError as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error uploading {local_path}: {e}")
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python upload_models_to_s3.py <bucket_name>")
        sys.exit(1)
    
    bucket_name = sys.argv[1]
    models_dir = Path("models")
    
    if not models_dir.exists():
        logger.error(f"Models directory {models_dir} not found")
        sys.exit(1)
    
    # Model files to upload (using actual files, not symlinks)
    model_files = [
        "best_cnn_regressor.joblib",
        "cnn_scaler.joblib", 
        "random_forest_model.joblib",
        # Also upload as the expected names for the app
        ("best_cnn_regressor.joblib", "molt_regressor_cnn_random_forest.joblib"),
        ("cnn_scaler.joblib", "molt_scaler_cnn.joblib")
    ]
    
    # Check AWS credentials
    try:
        boto3.client('s3').list_buckets()
        logger.info("AWS credentials verified")
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error verifying AWS credentials: {e}")
        sys.exit(1)
    
    # Upload each model file
    success_count = 0
    for model_item in model_files:
        if isinstance(model_item, tuple):
            # Tuple: (local_file, s3_name)
            local_file, s3_name = model_item
            local_path = models_dir / local_file
            s3_key = f"models/{s3_name}"
        else:
            # String: same name for local and S3
            local_file = model_item
            local_path = models_dir / local_file
            s3_key = f"models/{local_file}"
        
        if local_path.exists():
            if upload_file_to_s3(bucket_name, str(local_path), s3_key):
                success_count += 1
        else:
            logger.warning(f"Model file {local_path} not found, skipping")
    
    logger.info(f"Successfully uploaded {success_count}/{len(model_files)} model files")
    
    if success_count > 0:
        logger.info("\nTo deploy to Heroku, set these environment variables:")
        logger.info(f"  heroku config:set S3_BUCKET_NAME={bucket_name}")
        logger.info("  heroku config:set AWS_ACCESS_KEY_ID=<your_access_key>")
        logger.info("  heroku config:set AWS_SECRET_ACCESS_KEY=<your_secret_key>")
        logger.info("  heroku config:set AWS_DEFAULT_REGION=us-east-1")


if __name__ == "__main__":
    main()