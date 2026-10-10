from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))
    max_upload_bytes: int = 5 * 1024 * 1024
    history_limit: int = 20
    confidence_threshold: float = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".dcm", ".dicom")
    classes: tuple[str, ...] = ("Healthy", "Not Healthy", "Pneumonia")


settings = Settings()
