FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080
ENV TORCH_HOME=/app/.cache/torch

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download ViT weights to avoid runtime download on first request.
RUN python -c "import torchvision.models as models; from torchvision.models import ViT_B_16_Weights; models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1); print('ViT weights cached')"

COPY . /app

EXPOSE 8080

CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8080"]
