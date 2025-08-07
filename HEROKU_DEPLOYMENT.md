# Heroku Deployment with S3 Model Storage

This guide shows how to deploy the Green Crab Molt Detection app to Heroku using S3 for model storage.

## Prerequisites

1. **AWS Account** with S3 access
2. **Heroku Account** with CLI installed
3. **AWS CLI** configured with credentials

## Step 1: Create S3 Bucket

```bash
# Create S3 bucket (replace 'your-bucket-name' with your desired bucket name)
aws s3 mb s3://your-green-crab-models-bucket

# Set bucket policy for read access (optional - use IAM roles in production)
# aws s3api put-bucket-policy --bucket your-green-crab-models-bucket --policy file://bucket-policy.json
```

## Step 2: Upload Models to S3

```bash
# Upload model files to S3
python upload_models_to_s3.py your-green-crab-models-bucket
```

## Step 3: Create Heroku App

```bash
# Login to Heroku
heroku login

# Create new Heroku app
heroku create your-app-name

# Set environment variables
heroku config:set S3_BUCKET_NAME=your-green-crab-models-bucket
heroku config:set AWS_ACCESS_KEY_ID=your-aws-access-key
heroku config:set AWS_SECRET_ACCESS_KEY=your-aws-secret-key
heroku config:set AWS_DEFAULT_REGION=us-east-1
```

## Step 4: Deploy to Heroku

```bash
# Add files to git
git add .
git commit -m "Add S3 model downloading for Heroku deployment"

# Deploy to Heroku
git push heroku main

# Check logs
heroku logs --tail
```

## Step 5: Test the Deployment

```bash
# Open the app
heroku open

# Check health endpoint
curl https://your-app-name.herokuapp.com/health
```

## Environment Variables

The app requires these environment variables:

- `S3_BUCKET_NAME`: Name of the S3 bucket containing model files
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key  
- `AWS_DEFAULT_REGION`: AWS region (default: us-east-1)

## File Structure

The app expects these model files in the S3 bucket under the `models/` prefix:

- `models/molt_regressor_cnn_random_forest.joblib`
- `models/cnn_scaler.joblib`
- `models/best_cnn_regressor.joblib`
- `models/molt_scaler_cnn.joblib`
- `models/random_forest_model.joblib`

## How It Works

1. On startup, the app checks for the `S3_BUCKET_NAME` environment variable
2. If found, it downloads model files from S3 to `/tmp/models/`
3. The app then loads models from either `/tmp/models/` or local `models/` directory
4. Models persist in `/tmp/` for the lifetime of the dyno (until restart)

## Troubleshooting

### Model Download Issues
- Check AWS credentials and bucket permissions
- Verify model files exist in S3 under `models/` prefix
- Check Heroku logs for download errors

### Memory Issues  
- Use Heroku performance dynos for better memory limits
- Consider model optimization if needed

### Cold Start Times
- First request may be slower due to model downloads
- Consider using Heroku Scheduler to ping the app periodically

## Production Considerations

1. **Security**: Use IAM roles instead of hardcoded AWS keys
2. **Performance**: Consider Heroku Redis for model caching
3. **Monitoring**: Add application monitoring (e.g., New Relic)
4. **Scaling**: Configure auto-scaling based on traffic