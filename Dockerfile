FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PORT=7860

# libGL/libglib dibutuhkan opencv-python; libgomp untuk scikit-learn dan torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch CPU dipasang lebih dulu supaya ultralytics tidak menarik wheel CUDA
# (beberapa GB dan tidak terpakai di CPU-only Space).
COPY requirements.txt requirements-ml.txt ./
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-ml.txt

COPY app ./app
COPY models ./models

RUN mkdir -p app/static/samples app/static/assets models

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
