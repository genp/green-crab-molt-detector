# Green Crab Molt Detection - Deployment Guide 🚀

This guide provides comprehensive instructions for deploying the Green Crab Molt Detection system to various cloud platforms.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Platform-Specific Deployment](#platform-specific-deployment)
  - [Heroku](#heroku)
  - [AWS EC2](#aws-ec2)
  - [Google Cloud Run](#google-cloud-run)
  - [DigitalOcean](#digitalocean)
- [Docker Deployment](#docker-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring & Maintenance](#monitoring--maintenance)

## Prerequisites

Before deploying, ensure you have:
1. The application running locally successfully
2. All models trained and saved in the `models/` directory
3. Required cloud platform account and CLI tools installed
4. Docker installed (for containerized deployments)

## Environment Configuration

Create a `.env` file for production settings:

```bash
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key-here

# Model Configuration
MODEL_PATH=/app/models/
DATA_PATH=/app/data/

# Server Configuration
PORT=5000
WORKERS=4
THREADS=2

# Optional: Cloud Storage (for model files)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
S3_BUCKET=your-bucket-name
```

## Platform-Specific Deployment

### Heroku

#### 1. Prerequisites
```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login to Heroku
heroku login
```

#### 2. Create Heroku App
```bash
# Create new app
heroku create green-crab-molt-detector

# Add buildpacks
heroku buildpacks:add heroku/python
```

#### 3. Create Required Files

**Procfile:**
```
web: gunicorn app:app --workers 4 --threads 2 --worker-class sync
```

**runtime.txt:**
```
python-3.10.12
```

#### 4. Deploy
```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial deployment"

# Add Heroku remote
heroku git:remote -a green-crab-molt-detector

# Deploy
git push heroku main

# Open the app
heroku open
```

#### 5. Configure Environment Variables
```bash
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
```

### AWS EC2

#### 1. Launch EC2 Instance
- AMI: Ubuntu Server 22.04 LTS
- Instance Type: t3.medium (minimum)
- Security Group: Allow HTTP (80), HTTPS (443), SSH (22)
- Storage: 20GB minimum

#### 2. Connect and Setup
```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3-pip python3-venv nginx supervisor git -y

# Clone repository
git clone https://github.com/your-username/green-crab-molt-detector.git
cd green-crab-molt-detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
pip install gunicorn
```

#### 3. Configure Nginx
```bash
sudo nano /etc/nginx/sites-available/green-crab
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /home/ubuntu/green-crab-molt-detector/static;
    }

    client_max_body_size 50M;
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/green-crab /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx
```

#### 4. Configure Supervisor
```bash
sudo nano /etc/supervisor/conf.d/green-crab.conf
```

Add:
```ini
[program:green-crab]
command=/home/ubuntu/green-crab-molt-detector/venv/bin/gunicorn app:app --workers 4 --bind 127.0.0.1:5000
directory=/home/ubuntu/green-crab-molt-detector
user=ubuntu
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/green-crab/app.log
environment=PATH="/home/ubuntu/green-crab-molt-detector/venv/bin",FLASK_ENV="production"
```

Start supervisor:
```bash
sudo mkdir -p /var/log/green-crab
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start green-crab
```

### Google Cloud Run

#### 1. Prerequisites
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize and authenticate
gcloud init
gcloud auth login
```

#### 2. Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy application files
COPY . .

# Create directories for models and data
RUN mkdir -p models data/processed

# Set environment variables
ENV FLASK_ENV=production
ENV PORT=8080

# Run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
```

#### 3. Build and Deploy
```bash
# Build container
gcloud builds submit --tag gcr.io/YOUR-PROJECT-ID/green-crab-molt

# Deploy to Cloud Run
gcloud run deploy green-crab-molt \
  --image gcr.io/YOUR-PROJECT-ID/green-crab-molt \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --min-instances 1
```

### DigitalOcean

#### 1. Create Droplet
- Image: Ubuntu 22.04 LTS
- Size: Basic, 2GB RAM minimum
- Add SSH key for authentication

#### 2. Initial Setup
```bash
# Connect to droplet
ssh root@your-droplet-ip

# Create non-root user
adduser deploy
usermod -aG sudo deploy
su - deploy

# Setup firewall
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

#### 3. Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install python3-pip python3-venv nginx git -y

# Clone repository
git clone https://github.com/your-username/green-crab-molt-detector.git
cd green-crab-molt-detector

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
```

#### 4. Create Systemd Service
```bash
sudo nano /etc/systemd/system/green-crab.service
```

Add:
```ini
[Unit]
Description=Green Crab Molt Detection
After=network.target

[Service]
User=deploy
Group=deploy
WorkingDirectory=/home/deploy/green-crab-molt-detector
Environment="PATH=/home/deploy/green-crab-molt-detector/venv/bin"
Environment="FLASK_ENV=production"
ExecStart=/home/deploy/green-crab-molt-detector/venv/bin/gunicorn app:app --workers 4 --bind unix:green-crab.sock

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl start green-crab
sudo systemctl enable green-crab
```

#### 5. Configure Nginx
```bash
sudo nano /etc/nginx/sites-available/green-crab
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/deploy/green-crab-molt-detector/green-crab.sock;
    }

    location /static {
        alias /home/deploy/green-crab-molt-detector/static;
    }

    client_max_body_size 50M;
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/green-crab /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx
```

## Docker Deployment

### Local Docker
```bash
# Build image
docker build -t green-crab-molt:latest .

# Run container
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name green-crab \
  green-crab-molt:latest
```

### Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "80:5000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=${SECRET_KEY}
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped
```

Deploy:
```bash
docker-compose up -d
```

## Production Considerations

### Security
1. **SSL/TLS Certificate**: Use Let's Encrypt for free SSL
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

2. **Environment Variables**: Never commit secrets to git
   - Use `.env` files (add to `.gitignore`)
   - Use platform-specific secret management

3. **Rate Limiting**: Add to nginx config
   ```nginx
   limit_req_zone $binary_remote_addr zone=one:10m rate=10r/s;
   location / {
       limit_req zone=one burst=20;
       # ... rest of config
   }
   ```

### Performance Optimization

1. **Model Loading**: Load models once at startup
2. **Caching**: Implement Redis for frequently accessed data
3. **CDN**: Use CloudFlare or AWS CloudFront for static assets
4. **Image Optimization**: Compress uploaded images before processing

### Scaling

1. **Horizontal Scaling**: 
   - Use load balancer (nginx, HAProxy)
   - Deploy multiple app instances
   - Consider Kubernetes for orchestration

2. **Database** (if needed):
   - PostgreSQL for production
   - Connection pooling
   - Read replicas for scaling

## Monitoring & Maintenance

### Logging
```python
# Add to app.py
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/green-crab.log', 
                                      maxBytes=10240000, 
                                      backupCount=10)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
```

### Health Check Endpoint
```python
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(os.listdir('models/')) > 0,
        'timestamp': datetime.now().isoformat()
    })
```

### Monitoring Tools
- **Application**: New Relic, DataDog, or Sentry
- **Infrastructure**: CloudWatch (AWS), Stackdriver (GCP)
- **Uptime**: UptimeRobot, Pingdom

### Backup Strategy
1. **Models**: Store in S3 or Cloud Storage
2. **Data**: Regular database backups
3. **Code**: Git repository (GitHub/GitLab)

### Updates and Maintenance
```bash
# Update application
git pull origin main
source venv/bin/activate
pip install -r requirements.txt

# Restart service (systemd)
sudo systemctl restart green-crab

# Restart service (supervisor)
sudo supervisorctl restart green-crab

# Restart Docker
docker-compose down
docker-compose up -d --build
```

## Troubleshooting

### Common Issues

1. **Model not loading**: Check file paths and permissions
2. **Out of memory**: Increase instance size or optimize model loading
3. **Slow predictions**: Consider GPU instances for faster inference
4. **502 Bad Gateway**: Check if application is running and nginx config

### Debug Commands
```bash
# Check application logs
tail -f /var/log/green-crab/app.log

# Check nginx logs
sudo tail -f /var/log/nginx/error.log

# Check system resources
htop
df -h

# Test application locally
curl http://localhost:5000/health
```

## Support

For deployment issues or questions:
- Create an issue on GitHub
- Check existing documentation in README.md
- Review application logs for error messages

---

*Last updated: 2025*