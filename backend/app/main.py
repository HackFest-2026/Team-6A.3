import base64
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
import logging
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import JSONResponse

from app.bounding import SuspiciousRegionDetector
from app.config import settings
from app.images import decode_upload, read_upload
from app.predict import ClassifierEnsemble
from app.schemas import HistoryItem, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
history: deque[HistoryItem] = deque(maxlen=settings.history_limit)


def _image_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _array_base64(image: np.ndarray) -> str:
    return _image_base64(Image.fromarray(image))


@asynccontextmanager
async def lifespan(application: FastAPI):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Starting pneumonia inference service on %s", device_name)
    try:
        application.state.classifier = ClassifierEnsemble(settings.model_dir, torch.device(device_name))
        application.state.detector = SuspiciousRegionDetector(settings.model_dir, device_name)
    except Exception:
        logger.exception("Model loading failed; check the configured checkpoints in %s", settings.model_dir)
        raise
    yield
    logger.info("Pneumonia inference service stopped")


app = FastAPI(title="Pneumonia Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def log_http_error(request: Request, error: HTTPException) -> JSONResponse:
    logger.warning("Request rejected %s %s: %s", request.method, request.url.path, error.detail)
    return JSONResponse(
        status_code=error.status_code,
        content={"detail": error.detail},
        headers=error.headers,
    )


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "classifier"):
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    started = time.perf_counter()
    logger.info("Prediction requested: %s", file.filename)
    raw = await read_upload(file)
    image, patient_details = decode_upload(raw, file.filename or "")
    try:
        probabilities = request.app.state.classifier.predict(image)
        predicted_class = max(probabilities, key=probabilities.get)
        annotated, yolo_confidence = request.app.state.detector.annotate(np.asarray(image))
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Inference failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="Inference failed. Check server logs.") from error

    elapsed_ms = (time.perf_counter() - started) * 1000
    created_at = datetime.now(timezone.utc)
    item = HistoryItem(
        id=str(uuid4()),
        filename=Path(file.filename or "image").name,
        predicted_class=predicted_class,
        confidence=probabilities[predicted_class],
        created_at=created_at,
        inference_ms=elapsed_ms,
    )
    history.appendleft(item)
    logger.info("Prediction complete: %s in %.1f ms; YOLO max confidence=%s", predicted_class, elapsed_ms, yolo_confidence)
    return PredictionResponse(
        id=item.id,
        filename=item.filename,
        predicted_class=predicted_class,
        probabilities=probabilities,
        yolo_confidence=yolo_confidence,
        original_image=_image_base64(image),
        annotated_image=_array_base64(annotated),
        patient_details=patient_details,
        created_at=created_at,
        inference_ms=elapsed_ms,
    )


@app.get("/history", response_model=list[HistoryItem])
def get_history() -> list[HistoryItem]:
    return list(history)


@app.delete("/history")
def clear_history() -> dict[str, int]:
    deleted = len(history)
    history.clear()
    return {"deleted": deleted}
