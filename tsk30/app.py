from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
import uvicorn
import numpy as np
from PIL import Image
import io
import cv2

app = FastAPI(title="Simple CV: Edge Detection API", version="1.0.0",
              description="Minimal FastAPI service that performs Canny edge detection on images.")

def _read_image(file_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")
    return np.array(img)

def _png_bytes(arr: np.ndarray) -> bytes:
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def _canny(rgb: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=low, threshold2=high)
    # convert to 3-channel so it's a nice PNG (white edges on black background)
    vis = np.stack([edges]*3, axis=-1)
    return vis

@app.get("/health", response_class=PlainTextResponse, summary="Liveness probe")
def health():
    return "ok"

@app.post("/edges", summary="Return edges as image/png")
async def edges(file: UploadFile = File(...), low: int = 100, high: int = 200):
    data = await file.read()
    rgb = _read_image(data)
    vis = _canny(rgb, low, high)
    png = _png_bytes(vis)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")

@app.post("/edges/json", summary="Return edges as base64 + metadata in JSON")
async def edges_json(file: UploadFile = File(...), low: int = 100, high: int = 200):
    data = await file.read()
    rgb = _read_image(data)
    h, w = rgb.shape[:2]
    vis = _canny(rgb, low, high)
    # simple metric: ratio of edge pixels
    edge_count = int((vis[...,0] > 0).sum())
    ratio = edge_count / float(h*w)
    png = _png_bytes(vis)
    b64 = base64.b64encode(png).decode("utf-8")
    return JSONResponse({
        "width": w,
        "height": h,
        "canny_thresholds": {"low": low, "high": high},
        "edge_pixel_ratio": ratio,
        "image_png_base64": b64
    })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
