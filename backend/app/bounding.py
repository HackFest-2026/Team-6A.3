import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import settings

logger = logging.getLogger(__name__)


class SuspiciousRegionDetector:
    def __init__(self, model_dir: Path, device: str) -> None:
        checkpoint_path = next(
            (
                path
                for path in (model_dir / "yolo.pth", model_dir / "controlled_v1_best.pt")
                if path.is_file()
            ),
            None,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Required YOLO checkpoint not found in {model_dir}; expected yolo.pth or controlled_v1_best.pt."
            )
        self.model = YOLO(str(checkpoint_path))
        self.device = device
        logger.info("Loaded localization checkpoint %s", checkpoint_path)

    def annotate(self, image: np.ndarray) -> tuple[np.ndarray, float | None]:
        result = self.model.predict(
            source=image,
            device=self.device,
            conf=settings.confidence_threshold,
            verbose=False,
        )[0]
        annotated = image.copy()
        confidences: list[float] = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 32, 32), 3)
                confidences.append(float(box.conf[0].item()))
        return annotated, max(confidences, default=None)
