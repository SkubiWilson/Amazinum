
# Simple CV: Edge Detection API (FastAPI)

A minimal, production-ready example of a **computer vision** service that performs **Canny edge detection** on input images. Deployed as a **REST API** using **FastAPI**.

---

## Overview
- **Task:** Given an input image, detect edges using the classic Canny method (acts as our "model").
- **Why this model?** It's tiny, deterministic, and runs anywhere without a GPU. It’s perfect to demonstrate deployment and evaluation quickly.
- **Outputs:** Processed image (PNG) or JSON with metrics + the processed image in base64.

---

## Deployment info
You can run locally with `uvicorn` or containerize.

### Local run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# start the API (http://127.0.0.1:8000)
python app.py
# or equivalently:
# uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker (optional)
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
Build & run:
```bash
docker build -t simple-cv .
docker run --rm -p 8000:8000 simple-cv
```

---

## Installation instructions
- Python 3.10+ recommended.
- Install system dependencies for `opencv-python` if your OS requires (on many systems wheels are prebuilt and no extra steps are needed).
- Install Python deps:
```bash
pip install -r requirements.txt
```

---

## Modeling info
- **Algorithm:** Canny edge detection.
- **Pipeline:**
  1. Load image (Pillow), convert to RGB.
  2. Convert to grayscale (OpenCV).
  3. Apply Canny with configurable `low`, `high` thresholds.
  4. Return a 3-channel visualization of edges for convenience.
- **Metrics exposed:** `edge_pixel_ratio` – fraction of pixels detected as edges.

---

## Interface description

### `GET /health`
- **Purpose:** Liveness probe.
- **Input:** none
- **Output:** `text/plain` – `"ok"`

### `POST /edges`
- **Purpose:** Returns processed image with edges.
- **Input:** `multipart/form-data` with field `file` (image). Optional query params: `low`, `high`.
- **Output:** `image/png` – the edge map visualization.

**cURL example:**
```bash
curl -X POST "http://127.0.0.1:8000/edges?low=50&high=150"   -H "Content-Type: multipart/form-data"   -F "file=@sample_input.png"   --output edges.png
```

### `POST /edges/json`
- **Purpose:** Returns metrics and the processed image as base64.
- **Input:** `multipart/form-data` with field `file` (image). Optional query params: `low`, `high`.
- **Output:** `application/json` with fields:
  - `width`, `height`
  - `canny_thresholds.low`, `canny_thresholds.high`
  - `edge_pixel_ratio` (float)
  - `image_png_base64` (string, PNG in base64)

**cURL example:**
```bash
curl -X POST "http://127.0.0.1:8000/edges/json?low=80&high=200"   -H "Content-Type: multipart/form-data"   -F "file=@sample_input.png"
```

---

## Example processes (screenshots/logs)

### Uvicorn logs (expected)
When starting the server:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
On a request to `/edges` you should see lines like:
```
INFO:     127.0.0.1:XXXXX - "POST /edges?low=50&high=150 HTTP/1.1" 200 OK
```

### Programmatic test
Use the provided `test_client.py` to send an image to the API and save the output image and JSON.

---

## Extras
- **Interactive docs:** Once running, open `http://127.0.0.1:8000/docs` for Swagger UI.
- **Changing thresholds:** Tune `low`/`high` via query params to get sharper or softer edges.

---

## License
MIT
