# PneumoScan Backend

Flask API for PneumoScan X-ray prediction.

## Local Run

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Local API:

```text
http://127.0.0.1:5001
```

Health check:

```bash
curl http://127.0.0.1:5001/health
```

Prediction route:

```bash
curl -X POST http://127.0.0.1:5001/predict \
  -F "file=@/path/to/xray.jpg"
```

## Docker Run

```bash
cd backend
docker build -t pneumoscan-backend .
docker run --rm -p 5001:5000 --env-file .env.example pneumoscan-backend
```

Docker API:

```text
http://127.0.0.1:5001
```

## Railway

Deploy the `backend/` folder as the Railway service root. Railway will use the `Dockerfile`.

Recommended environment variables:

```text
MODEL_PATH=/app/final_model.keras
UPLOAD_DIR=/tmp
MAX_UPLOAD_MB=16
WEB_CONCURRENCY=1
GUNICORN_THREADS=2
GUNICORN_TIMEOUT=120
```

Railway provides `PORT` automatically, so do not hardcode it there.

After deployment, copy the Railway public URL and update the Android app's Retrofit base URL.
